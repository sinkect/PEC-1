import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)

from models.architecture import PECEngine
from models.data import PECDataset, PECCollator, EntityMasker, SharedMaskProbability
from models.dataset_mixing import (
    BlendResult,
    load_stage1_kd_blended_dataset,
    load_stage23_blended_dataset,
    save_blend_metadata,
    save_sampled_by_source_as_jsonl,
)
from models.losses import GateL1Trainer

try:
    import setproctitle
except ModuleNotFoundError:
    setproctitle = None


@dataclass(frozen=True)
class StageSpec:
    name: str
    visible_prompt_mode: str
    include_teacher: bool
    train_samples: int
    with_replacement: bool
    mask_prob_start: Optional[float] = None
    mask_prob_end: Optional[float] = None


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train PEC with stage-aware masking and Stage 1 KD.")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "outputs" / "PEC")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
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
    parser.add_argument(
        "--stage2-mask-prob",
        type=float,
        default=0.7,
        help="Backward-compatible alias for the curriculum start mask probability.",
    )
    parser.add_argument(
        "--stage3-mask-prob",
        type=float,
        default=0.2,
        help="Backward-compatible alias for the curriculum end mask probability.",
    )
    parser.add_argument("--stage23-mask-prob-start", type=float, default=None)
    parser.add_argument("--stage23-mask-prob-end", type=float, default=None)
    parser.add_argument("--stage1-kd-lambda", type=float, default=0.3)
    parser.add_argument("--stage1-kd-temperature", type=float, default=2.0)

    parser.add_argument("--num-query-tokens", type=int, default=64)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--profiler-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--gate-l1-max-lambda", type=float, default=1e-3)
    parser.add_argument("--gate-l1-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-profiler-len", type=int, default=6080)
    parser.add_argument("--max-composer-len", type=int, default=6080)
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


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_training_process_title(stage_name: str) -> None:
    if setproctitle is None:
        return
    setproctitle.setproctitle(f"sinkect_training_{stage_name}")


class BlendedMessagesToPECSamples(Dataset):
    """Normalizes blended samples into a stable prompt/answer schema."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        return {
            "prompt": str(sample.get("prompt", "")).strip(),
            "answer": str(sample.get("answer", "")).strip(),
            "source": sample.get("source", ""),
        }


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
    stage23_mask_prob_start = (
        args.stage23_mask_prob_start if args.stage23_mask_prob_start is not None else args.stage2_mask_prob
    )
    stage23_mask_prob_end = (
        args.stage23_mask_prob_end if args.stage23_mask_prob_end is not None else args.stage3_mask_prob
    )
    return {
        "stage1": StageSpec(
            name="stage1",
            visible_prompt_mode="empty",
            include_teacher=True,
            train_samples=args.stage1_train_samples,
            with_replacement=False,
            mask_prob_start=None,
            mask_prob_end=None,
        ),
        "stage23": StageSpec(
            name="stage23",
            visible_prompt_mode="masked",
            include_teacher=False,
            train_samples=args.stage23_train_samples,
            with_replacement=True,
            mask_prob_start=stage23_mask_prob_start,
            mask_prob_end=stage23_mask_prob_end,
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


def load_blend_for_stage(stage: StageSpec, seed: int) -> BlendResult:
    if stage.name == "stage1":
        return load_stage1_kd_blended_dataset(
            split="train",
            seed=seed,
            epoch_size=stage.train_samples,
            with_replacement=stage.with_replacement,
        )

    return load_stage23_blended_dataset(
        split="train",
        seed=seed,
        epoch_size=stage.train_samples,
        with_replacement=stage.with_replacement,
    )


def build_stage_datasets(
    *,
    stage: StageSpec,
    base_dataset: Dataset,
    composer_tokenizer,
    eval_ratio: float,
    seed: int,
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
        )
        eval_mask_prob = stage.mask_prob_end if stage.mask_prob_end is not None else stage.mask_prob_start
        eval_masker = EntityMasker(mask_prob=eval_mask_prob)
    else:
        train_masker = None
        eval_masker = None

    train_dataset = PECDataset(
        data=train_base,
        query_masker=train_masker,
        composer_tokenizer=composer_tokenizer,
        composer_enable_thinking=False,
        visible_prompt_mode=stage.visible_prompt_mode,
        include_teacher=stage.include_teacher,
        teacher_visible_prompt_mode="full",
    )
    eval_dataset = PECDataset(
        data=eval_base,
        query_masker=eval_masker,
        composer_tokenizer=composer_tokenizer,
        composer_enable_thinking=False,
        visible_prompt_mode=stage.visible_prompt_mode,
        include_teacher=stage.include_teacher,
        teacher_visible_prompt_mode="full",
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
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "profiler" in n and p.requires_grad],
            "lr": args.profiler_learning_rate,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if ("extruder" in n or "projector" in n) and p.requires_grad
            ],
            "lr": args.learning_rate,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)


def load_teacher_model(model_name: str, device: torch.device, dtype: torch.dtype):
    kwargs = {"low_cpu_mem_usage": True}
    if device.type != "cpu":
        kwargs["dtype"] = dtype

    teacher_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    return teacher_model


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


def main() -> None:
    args = parse_args()
    device = get_device()
    dtype = get_model_dtype(device)
    base_dir = Path(__file__).resolve().parent
    args.stages = normalize_stage_names(args.stages)
    stage_specs = get_stage_specs(args)

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
    )

    teacher_model = None
    if "stage1" in args.stages:
        print("Loading frozen Stage 1 teacher model...")
        teacher_model = load_teacher_model(args.composer_model_name, device=device, dtype=dtype)

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
        set_training_process_title(stage.name)
        print(f"\n===== {stage.name.upper()} =====")
        if stage.mask_prob_start is None:
            print("Composer visible prompt: soft prompt only")
        else:
            print(
                "Composer visible prompt mask curriculum: "
                f"{stage.mask_prob_start:.4f} -> {stage.mask_prob_end:.4f}"
            )
        if stage.include_teacher:
            print(
                "Teacher supervision: enabled "
                f"(kl_lambda={args.stage1_kd_lambda}, temperature={args.stage1_kd_temperature})"
            )

        blend_result = load_blend_for_stage(stage, seed=args.seed)
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
            teacher_model=teacher_model if stage.include_teacher else None,
            distill_kl_lambda=args.stage1_kd_lambda if stage.include_teacher else 0.0,
            distill_kl_temperature=args.stage1_kd_temperature if stage.include_teacher else 1.0,
            callbacks=callbacks,
        )

        print(f"Starting {stage.name} training...")
        if resume_checkpoint is not None:
            print(f"Resuming {stage.name} from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        resume_checkpoint = None
        trainer.save_model(str(stage_output_dir / "final"))

        if stage.include_teacher and teacher_model is not None:
            teacher_model.to("cpu")
            teacher_model = None
            clear_memory()


if __name__ == "__main__":
    main()
