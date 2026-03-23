import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from safetensors.torch import load_file as load_safetensors
import setproctitle
from torch.utils.data import Dataset, Subset

from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)

from models.architecture import PECEngine
from models.data import PECDataset, PECCollator, EntityMasker, SharedMaskProbability
from models.dataset_mixing import (
    BlendResult,
    load_stage1_blended_dataset,
    load_stage23_blended_dataset,
    save_blend_metadata,
    save_sampled_by_source_as_jsonl,
)
from models.losses import GateL1Trainer


@dataclass(frozen=True)
class StageSpec:
    name: str
    visible_prompt_mode: str
    train_samples: int
    with_replacement: bool
    mask_prob_start: Optional[float] = None
    mask_prob_end: Optional[float] = None


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train PEC with stage-aware masking.")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs" / "PEC")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--init-from-checkpoint", type=Path, default=None)
    parser.add_argument("--profiler-model-path", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--composer-model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["stage1", "stage23", "stage2", "stage3"],
        default=["stage1", "stage23"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.02)

    parser.add_argument("--stage1-train-samples", type=int, default=30_000)
    parser.add_argument("--stage23-train-samples", type=int, default=200_000)
    parser.add_argument("--stage23-mask-prob-start", type=float, default=0.7)
    parser.add_argument("--stage23-mask-prob-end", type=float, default=0.2)
    parser.add_argument("--process-name", type=str, default="pec_training")
    parser.add_argument("--num-query-tokens", type=int, default=64)
    parser.add_argument(
        "--freeze-profiler",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freeze-extruder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument(
        "--gradient-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--extruder-learning-rate",
        type=float,
        default=None,
        help="Optional override for Extruder LR. Defaults to --learning-rate when unset.",
    )
    parser.add_argument(
        "--projector-learning-rate",
        type=float,
        default=None,
        help="Optional override for Projector LR. Defaults to --learning-rate when unset.",
    )
    parser.add_argument("--profiler-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--gate-l1-max-lambda", type=float, default=2e-4)
    parser.add_argument("--gate-l1-warmup-ratio", type=float, default=0.3)
    parser.add_argument("--projector-raw-l2-lambda", type=float, default=1e-5)
    parser.add_argument(
        "--morehop-align-lambda",
        type=float,
        default=0.1,
        help="Lambda for the MoreHopQA z_pool alignment auxiliary loss.",
    )
    parser.add_argument(
        "--morehop-target-span-mask-prob",
        type=float,
        default=0.5,
        help="Mask probability for MoreHopQA support-answer token spans.",
    )
    parser.add_argument(
        "--morehop-base-mask-prob",
        type=float,
        default=0.1,
        help="Background random masking probability for non-target MoreHopQA tokens.",
    )
    parser.add_argument("--max-profiler-len", type=int, default=6144)
    parser.add_argument("--max-composer-len", type=int, default=6144)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            cuda_index = int(local_rank)
            torch.cuda.set_device(cuda_index)
            return torch.device("cuda", cuda_index)
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


class BlendedMessagesToPECSamples(Dataset):
    """Normalizes blended samples into a stable prompt/answer schema."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        normalized = {
            "prompt": str(sample.get("prompt", "")).strip(),
            "answer": str(sample.get("answer", "")).strip(),
            "source": sample.get("source", ""),
        }
        for key in (
            "task_type",
            "mh_target_texts",
        ):
            if key in sample:
                normalized[key] = sample[key]
        return normalized


def split_dataset_indices(dataset_len: int, test_size: float = 0.02, seed: int = 42) -> Dict[str, List[int]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    eval_len = max(1, int(dataset_len * test_size))
    train_len = dataset_len - eval_len
    if train_len <= 0:
        raise ValueError("Dataset is too small after applying test_size.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    return {
        "train": permutation[:train_len],
        "eval": permutation[train_len:],
    }


def normalize_stage_names(stage_names: List[str]) -> List[str]:
    normalized: List[str] = []
    for stage_name in stage_names:
        canonical_name = "stage23" if stage_name in {"stage2", "stage3"} else stage_name
        if canonical_name not in normalized:
            normalized.append(canonical_name)
    return normalized


def get_stage_specs(args: argparse.Namespace) -> Dict[str, StageSpec]:
    return {
        "stage1": StageSpec(
            name="stage1",
            visible_prompt_mode="empty",
            train_samples=args.stage1_train_samples,
            with_replacement=False,
            mask_prob_start=None,
            mask_prob_end=None,
        ),
        "stage23": StageSpec(
            name="stage23",
            visible_prompt_mode="masked",
            train_samples=args.stage23_train_samples,
            with_replacement=True,
            mask_prob_start=args.stage23_mask_prob_start,
            mask_prob_end=args.stage23_mask_prob_end,
        ),
    }


class MaskingCurriculumCallback(TrainerCallback):
    """Linearly anneals train-time masking from heavy to light over steps."""

    def __init__(
            self,
            shared_mask_prob: SharedMaskProbability,
            start_prob: float,
            end_prob: float,
    ) -> None:
        self.shared_mask_prob = shared_mask_prob
        self.start_prob = float(start_prob)
        self.end_prob = float(end_prob)
        self.current_prob = float(start_prob)

    def _resolve_progress(self, state) -> float:
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 1:
            return 1.0
        return min(max(float(state.global_step) / float(max_steps - 1), 0.0), 1.0)

    def _update_prob(self, state) -> float:
        progress = self._resolve_progress(state)
        self.current_prob = self.start_prob + ((self.end_prob - self.start_prob) * progress)
        self.shared_mask_prob.set(self.current_prob)
        return self.current_prob

    def on_train_begin(self, args, state, control, **kwargs):
        del args, kwargs
        self._update_prob(state)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        del args, kwargs
        self._update_prob(state)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, state, kwargs
        if logs is not None:
            logs["mask_prob"] = self.current_prob
        return control


def load_blend_for_stage(
        stage: StageSpec,
        seed: int,
        *,
        max_profiler_tokens: int,
        max_composer_tokens: int,
) -> BlendResult:
    if stage.name == "stage1":
        return load_stage1_blended_dataset(
            split="train",
            seed=seed,
            epoch_size=stage.train_samples,
            with_replacement=stage.with_replacement,
            max_profiler_tokens=max_profiler_tokens,
            max_composer_tokens=max_composer_tokens,
        )

    return load_stage23_blended_dataset(
        split="train",
        seed=seed,
        epoch_size=stage.train_samples,
        with_replacement=stage.with_replacement,
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
    )


def build_stage_datasets(
        *,
        stage: StageSpec,
        base_dataset: Dataset,
        composer_tokenizer,
        eval_ratio: float,
        seed: int,
        morehop_target_span_mask_prob: float,
        morehop_base_mask_prob: float,
):
    split_indices = split_dataset_indices(len(base_dataset), test_size=eval_ratio, seed=seed)
    train_base = Subset(base_dataset, split_indices["train"])
    eval_base = Subset(base_dataset, split_indices["eval"])

    shared_mask_prob = None
    if stage.mask_prob_start is not None:
        shared_mask_prob = SharedMaskProbability(stage.mask_prob_start)
        train_masker = EntityMasker(
            mask_prob=stage.mask_prob_start,
            shared_mask_prob=shared_mask_prob,
            tokenizer=composer_tokenizer,
            target_span_mask_prob=morehop_target_span_mask_prob,
            morehop_base_mask_prob=morehop_base_mask_prob,
        )
        eval_mask_prob = stage.mask_prob_end if stage.mask_prob_end is not None else stage.mask_prob_start
        eval_masker = EntityMasker(
            mask_prob=eval_mask_prob,
            tokenizer=composer_tokenizer,
            target_span_mask_prob=morehop_target_span_mask_prob,
            morehop_base_mask_prob=morehop_base_mask_prob,
        )
    else:
        train_masker = None
        eval_masker = None

    train_dataset = PECDataset(
        data=train_base,
        query_masker=train_masker,
        composer_tokenizer=composer_tokenizer,
        composer_enable_thinking=False,
        visible_prompt_mode=stage.visible_prompt_mode,
    )
    eval_dataset = PECDataset(
        data=eval_base,
        query_masker=eval_masker,
        composer_tokenizer=composer_tokenizer,
        composer_enable_thinking=False,
        visible_prompt_mode=stage.visible_prompt_mode,
    )
    return train_dataset, eval_dataset, shared_mask_prob


def build_training_arguments(args: argparse.Namespace, output_dir: Path, device: torch.device) -> TrainingArguments:
    is_cuda = device.type == "cuda"
    is_mps = device.type == "mps"

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpoint,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        fp16=is_mps,
        bf16=is_cuda,
        tf32=is_cuda,
        logging_steps=args.logging_steps,
        report_to="none",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        dataloader_num_workers=8 if is_cuda else 0,
        dataloader_pin_memory=is_cuda,
        remove_unused_columns=False,
    )


def build_optimizer(model: PECEngine, args: argparse.Namespace) -> torch.optim.Optimizer:
    extruder_learning_rate = (
        args.extruder_learning_rate if args.extruder_learning_rate is not None else args.learning_rate
    )
    projector_learning_rate = (
        args.projector_learning_rate if args.projector_learning_rate is not None else args.learning_rate
    )
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    profiler_params = [parameter for name, parameter in named_parameters if name.startswith("profiler.")]
    extruder_params = [parameter for name, parameter in named_parameters if name.startswith("extruder.")]
    projector_params = [parameter for name, parameter in named_parameters if name.startswith("projector.")]
    covered_prefixes = ("profiler.", "extruder.", "projector.")
    misc_params = [
        parameter
        for name, parameter in named_parameters
        if not name.startswith(covered_prefixes)
    ]

    optimizer_grouped_parameters = [
        {
            "params": profiler_params,
            "lr": args.profiler_learning_rate,
        },
        {
            "params": extruder_params,
            "lr": extruder_learning_rate,
        },
        {
            "params": projector_params,
            "lr": projector_learning_rate,
        },
        {
            "params": misc_params,
            "lr": args.learning_rate,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)


def persist_stage_metadata(
        *,
        base_dir: Path,
        stage: StageSpec,
        blend_result: BlendResult,
) -> None:
    processed_data_dir = base_dir / "data" / "processed_pec_dataset" / stage.name
    save_blend_metadata(blend_result, processed_data_dir / "blend_metadata.json")
    save_sampled_by_source_as_jsonl(
        blend_result.dataset,
        processed_data_dir / "sampled_rows.jsonl",
        source_names=blend_result.source_names,
        samples_per_source=3,
        seed=42,
    )


def load_state_dict_from_checkpoint(checkpoint_dir: Path) -> Dict[str, Any]:
    safetensor_path = checkpoint_dir / "model.safetensors"
    pytorch_path = checkpoint_dir / "pytorch_model.bin"

    if safetensor_path.exists():
        state_dict = load_safetensors(str(safetensor_path))
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")

    if all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def initialize_model_from_checkpoint(model: PECEngine, checkpoint_dir: Path) -> None:
    state_dict = load_state_dict_from_checkpoint(checkpoint_dir)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    critical_missing = [
        key for key in missing_keys
        if not (
            key.startswith("composer.")
            or key.startswith("profiler.")
            or key.startswith("prev_span_head.")
            or key.startswith("expr_head.")
        )
    ]
    if critical_missing:
        raise RuntimeError(f"Critical weights are missing from checkpoint {checkpoint_dir}: {critical_missing}")
    filtered_unexpected_keys = [
        key for key in unexpected_keys
        if not (
            key.startswith("prev_sent_head.")
            or key.startswith("prev_span_head.")
            or key.startswith("expr_head.")
        )
    ]
    if filtered_unexpected_keys:
        raise RuntimeError(f"Unexpected keys in checkpoint {checkpoint_dir}: {filtered_unexpected_keys}")

    print(f"Initialized model weights from checkpoint: {checkpoint_dir}")


def _checkpoint_sort_key(checkpoint_dir: Path) -> int:
    try:
        return int(checkpoint_dir.name.rsplit("-", 1)[-1])
    except (IndexError, ValueError):
        return -1


def _resolve_resumable_checkpoint_source(trainer: GateL1Trainer, stage_output_dir: Path) -> Optional[Path]:
    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if best_checkpoint is not None:
        best_checkpoint_path = Path(best_checkpoint)
        if best_checkpoint_path.exists():
            return best_checkpoint_path

    checkpoint_dirs = sorted(
        (path for path in stage_output_dir.glob("checkpoint-*") if path.is_dir()),
        key=_checkpoint_sort_key,
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1]
    return None


def save_resumable_final_checkpoint(
        trainer: GateL1Trainer,
        *,
        stage_output_dir: Path,
        final_output_dir: Path,
) -> None:
    source_checkpoint = _resolve_resumable_checkpoint_source(trainer, stage_output_dir)

    if trainer.is_world_process_zero():
        if source_checkpoint is None:
            raise RuntimeError(
                "No checkpoint directory was found to export as a resumable final checkpoint. "
                "Keep at least one saved checkpoint or lower --save-steps."
            )

        print(f"Exporting resumable final checkpoint from: {source_checkpoint}")
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir)
        shutil.copytree(source_checkpoint, final_output_dir)

        trainer_state_path = final_output_dir / "trainer_state.json"
        if trainer_state_path.exists():
            trainer_state = json.loads(trainer_state_path.read_text())
            if trainer_state.get("best_model_checkpoint"):
                trainer_state["best_model_checkpoint"] = str(final_output_dir)
            trainer_state_path.write_text(json.dumps(trainer_state, indent=2, sort_keys=True) + "\n")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main() -> None:
    args = parse_args()
    if args.resume_from_checkpoint is not None and args.init_from_checkpoint is not None:
        raise ValueError("--resume-from-checkpoint and --init-from-checkpoint cannot be used together.")
    if args.init_from_checkpoint is not None and not args.init_from_checkpoint.is_dir():
        raise ValueError(f"--init-from-checkpoint must point to a checkpoint directory: {args.init_from_checkpoint}")

    device = get_device()
    base_dir = Path(__file__).resolve().parent
    args.stages = normalize_stage_names(args.stages)
    stage_specs = get_stage_specs(args)

    setproctitle.setproctitle(args.process_name + '-' + args.stages[0])
    print(f"Device: {device}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Stages: {args.stages}")

    print("Loading tokenizers...")
    profiler_tokenizer = AutoTokenizer.from_pretrained(args.profiler_model_path)
    composer_tokenizer = AutoTokenizer.from_pretrained(args.composer_model_name)
    ensure_tokenizer_padding(profiler_tokenizer)
    ensure_tokenizer_padding(composer_tokenizer)

    print("Initializing PEC Engine...")
    model = PECEngine(
        profiler_path=str(args.profiler_model_path),
        composer_path=args.composer_model_name,
        num_query_tokens=args.num_query_tokens,
        morehop_align_lambda=args.morehop_align_lambda,
        freeze_profiler=args.freeze_profiler,
        freeze_extruder=args.freeze_extruder,
    )
    if args.init_from_checkpoint is not None:
        initialize_model_from_checkpoint(model, args.init_from_checkpoint)

    data_collator = PECCollator(
        profiler_tokenizer=profiler_tokenizer,
        composer_tokenizer=composer_tokenizer,
        max_profiler_len=args.max_profiler_len,
        max_composer_len=args.max_composer_len,
    )

    resume_checkpoint = args.resume_from_checkpoint
    for stage_name in args.stages:
        stage = stage_specs[stage_name]
        stage_output_dir = args.output_dir / stage.name
        print(f"\n===== {stage.name.upper()} =====")
        if stage.mask_prob_start is None:
            print("Composer visible prompt: soft prompt only")
        else:
            print(
                "Composer visible prompt mask curriculum: "
                f"{stage.mask_prob_start:.4f} -> {stage.mask_prob_end:.4f}"
            )

        blend_result = load_blend_for_stage(
            stage,
            seed=args.seed,
            max_profiler_tokens=args.max_profiler_len,
            max_composer_tokens=args.max_composer_len,
        )
        print(
            "Blend counts: "
            f"{dict(zip(blend_result.source_names, blend_result.per_dataset_counts))} "
            f"(total={blend_result.total_samples})"
        )
        persist_stage_metadata(base_dir=base_dir, stage=stage, blend_result=blend_result)

        pec_ready_dataset = BlendedMessagesToPECSamples(blend_result.dataset)
        train_dataset, eval_dataset, shared_mask_prob = build_stage_datasets(
            stage=stage,
            base_dataset=pec_ready_dataset,
            composer_tokenizer=composer_tokenizer,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            morehop_target_span_mask_prob=args.morehop_target_span_mask_prob,
            morehop_base_mask_prob=args.morehop_base_mask_prob,
        )

        training_args = build_training_arguments(args, stage_output_dir, device)
        optimizer = build_optimizer(model, args)
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        if shared_mask_prob is not None and stage.mask_prob_end is not None:
            callbacks.append(
                MaskingCurriculumCallback(
                    shared_mask_prob=shared_mask_prob,
                    start_prob=stage.mask_prob_start,
                    end_prob=stage.mask_prob_end,
                )
            )

        trainer = GateL1Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=(optimizer, None),
            gate_l1_max_lambda=args.gate_l1_max_lambda,
            gate_l1_warmup_ratio=args.gate_l1_warmup_ratio,
            projector_raw_l2_lambda=args.projector_raw_l2_lambda,
            callbacks=callbacks,
        )

        print(f"Starting {stage.name} training...")
        if resume_checkpoint is not None:
            print(f"Resuming {stage.name} from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        resume_checkpoint = None
        save_resumable_final_checkpoint(
            trainer,
            stage_output_dir=stage_output_dir,
            final_output_dir=stage_output_dir / "final",
        )


if __name__ == "__main__":
    main()
