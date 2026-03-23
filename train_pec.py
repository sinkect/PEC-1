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
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from models.architecture import PECEngine
from models.data import PECDataset, PECCollator, EntityMasker, SharedMaskProbability
from models.dataset_mixing import (
    BlendResult,
    build_morehop_prompt,
    load_stage1_blended_dataset,
    save_blend_metadata,
    save_sampled_by_source_as_jsonl,
)


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
        default=["stage1"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.02)

    parser.add_argument(
        "--stage1-train-samples",
        "--stage2-train-samples",
        "--stage23-train-samples",
        dest="stage1_train_samples",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--stage1-mask-prob-start",
        "--stage2-mask-prob-start",
        "--stage23-mask-prob-start",
        dest="stage1_mask_prob_start",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--stage1-mask-prob-end",
        "--stage2-mask-prob-end",
        "--stage23-mask-prob-end",
        dest="stage1_mask_prob_end",
        type=float,
        default=0.2,
    )
    parser.add_argument("--process-name", type=str, default="pec_training")
    parser.add_argument("--num-query-tokens", type=int, default=16, help="Number of latent memory slots.")
    parser.add_argument("--memory-upper-layers", type=int, default=8)
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
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps per stage. When > 0, overrides --num-train-epochs.",
    )
    parser.add_argument(
        "--gradient-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
        help="Optional override for memory-projector LR. Defaults to --learning-rate when unset.",
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
    parser.add_argument(
        "--morehop-align-lambda",
        type=float,
        default=0.01,
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
        canonical_name = "stage1" if stage_name in {"stage1", "stage2", "stage23", "stage3"} else stage_name
        if canonical_name not in normalized:
            normalized.append(canonical_name)
    return normalized


def get_stage_specs(args: argparse.Namespace) -> Dict[str, StageSpec]:
    return {
        "stage1": StageSpec(
            name="stage1",
            visible_prompt_mode="masked",
            train_samples=args.stage1_train_samples,
            with_replacement=True,
            mask_prob_start=args.stage1_mask_prob_start,
            mask_prob_end=args.stage1_mask_prob_end,
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


FIXED_SAMPLE_PROMPTS = [
    {
        "name": "hello",
        "prompt": "hello",
    },
    {
        "name": "ueda_pond",
        "prompt": (
            "What is the historical significance of the pond lowered by one shaku as specified by the lord "
            "of the Ueda Domain? How did this decision impact the surrounding area and the people living "
            "there?\n\n"
            "Context:\n"
            "Yamada Pond was the largest reservoir in Shiodadaira until Sawayama Pond was made in 1938.\n"
            "The pond has a long history, and while the date of its construction is unknown, in 1650, two "
            "ponds that were side by side were combined to form the present pond. The following story remains "
            "from when that work was done.\n"
            "Before construction, the lord of the Ueda Domain looked over the plans, saying, \"If we make such "
            "a large pond, there's a possibility Ueda Castle and the town could be damaged if the embankments "
            "were to break, so make the embankments one shaku (about 30 cm/1 foot) lower.\" Making the "
            "embankments even one shaku higher would make the reservoir hold much more water. But if the "
            "embankments were to break for some reason, a vast amount of water would assail the castle and town. "
            "As a result, the embankments were lowered by one shaku and Yamada Pond was made.\n"
            "Around 1840, carp were raised at this pond, and the Ueda Domain grew medicinal plants as a "
            "business in a 3,000 tsubo (1 hectare/2.45 acre) herb garden on the shores of the pond.\n"
            "Yamada Pond was constructed straddling Yagisawa and Yamada lands, but the person who oversaw all "
            "construction on the pond came from \"Yamada in Ise (present day Mie Prefecture),\" so it is said "
            "he took that place name and named the pond after it. At Yagisawa Funakubo, from where you can "
            "overlook the entire pond, is enshrined Amaterasu, which is said to have been brought from Ise "
            "Shrine, the highest rank shrine in the country, as a guardian deity for the pond.\n"
            "|address||Yagisawa, Ueda City|"
        ),
    },
    {
        "name": "morehop_reverse_seldom",
        "prompt": build_morehop_prompt(
            {
                "question": (
                    "What is the reverse order of the letters in the first name of the role the actor who "
                    "played Colonel Sherman T. Potter in \"M*A*S*H\" played in \"Kentucky Jones\"?"
                ),
                "context": [
                    [
                        "Kentucky Jones",
                        [
                            "Kentucky Jones is a half-hour comedy/drama starring Dennis Weaver as Kenneth "
                            "Yarborough \"K.Y. or Kentucky\" Jones, D.V.M., a recently widowed former horse "
                            "trainer and active horse farm owner, who becomes the guardian of Dwight Eisenhower "
                            "\"Ike\" Wong, a 10-year-old Chinese orphan, played by Ricky Der.",
                            "Harry Morgan, previously of the CBS sitcoms \"December Bride\" and \"Pete and "
                            "Gladys\", was featured in the series as Seldom Jackson, a former jockey who assists "
                            "Dr. Jones.",
                            "Cherylene Lee appears as Annie Ng, Ike's friend.",
                            "Arthur Wong portrays Mr. Ng, Annie's father.",
                            "Keye Luke (1904-1991) stars as Mr. Wong, a friend of Dr. Jones.",
                            "Nancy Rennick (1932-2006) appears as Miss Throncroft, a social worker.",
                            "\"Kentucky Jones\", which ran on NBC from September 19, 1964, to September 11, 1965, "
                            "was the first of four television series starring Weaver after he left the role of the "
                            "marshall's helper Chester Goode on CBS's western classic \"Gunsmoke\".",
                        ],
                    ],
                    [
                        "Harry Morgan",
                        [
                            "Harry Morgan (born Harry Bratsberg, April 10, 1915 - December 7, 2011) was an "
                            "American actor and director whose television and film career spanned six decades.",
                            "Morgan's major roles included Pete Porter in both \"December Bride\" (1954-1959) "
                            "and \"Pete and Gladys\" (1960-1962); Officer Bill Gannon on \"Dragnet\" "
                            "(1967-1970); Amos Coogan on \"Hec Ramsey\" (1972-1974); and his starring role as "
                            "Colonel Sherman T. Potter in \"M*A*S*H\" (1975-1983) and \"AfterMASH\" "
                            "(1983-1984).",
                            "Morgan appeared in more than 100 films.",
                        ],
                    ],
                ],
            }
        ),
    },
]


class FixedSampleGenerationCallback(TrainerCallback):
    def __init__(
            self,
            *,
            stage_name: str,
            stage_output_dir: Path,
            profiler_tokenizer,
            composer_tokenizer,
            max_profiler_len: int,
            max_composer_len: int,
            every_steps: int = 500,
            max_new_tokens: int = 128,
    ) -> None:
        self.stage_name = stage_name
        self.stage_output_dir = stage_output_dir
        self.profiler_tokenizer = profiler_tokenizer
        self.composer_tokenizer = composer_tokenizer
        self.max_profiler_len = max_profiler_len
        self.max_composer_len = max_composer_len
        self.every_steps = int(every_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.output_path = self.stage_output_dir / "sample_generations.jsonl"

    def _build_visible_prompt(self, prompt: str) -> str:
        return prompt

    def _build_composer_prompt_text(self, visible_prompt: str) -> str:
        return self.composer_tokenizer.apply_chat_template(
            [{"role": "user", "content": visible_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _generate_completion(self, model: PECEngine, prompt: str) -> Dict[str, str]:
        visible_prompt = self._build_visible_prompt(prompt)
        composer_prompt_text = self._build_composer_prompt_text(visible_prompt)

        profiler_inputs = self.profiler_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_profiler_len,
        )
        composer_inputs = self.composer_tokenizer(
            composer_prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_composer_len,
        )

        generated_ids = model.generate_with_memory(
            profiler_input_ids=profiler_inputs["input_ids"],
            profiler_attention_mask=profiler_inputs["attention_mask"],
            composer_input_ids=composer_inputs["input_ids"],
            composer_attention_mask=composer_inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
        )
        generation = self.composer_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        return {
            "prompt": prompt,
            "visible_prompt": visible_prompt,
            "generation": generation,
        }

    def _write_record(self, record: Dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_step_end(self, args, state, control, **kwargs):
        del args
        if self.every_steps <= 0:
            return control
        if int(getattr(state, "global_step", 0) or 0) <= 0:
            return control
        if int(state.global_step) % self.every_steps != 0:
            return control
        if not bool(getattr(state, "is_world_process_zero", True)):
            return control

        model = kwargs.get("model")
        if model is None:
            return control
        if hasattr(model, "module"):
            model = model.module
        if not isinstance(model, PECEngine):
            return control

        previous_training_mode = model.training
        model.eval()
        try:
            samples = []
            for sample_spec in FIXED_SAMPLE_PROMPTS:
                sample_output = self._generate_completion(model, sample_spec["prompt"])
                sample_output["name"] = sample_spec["name"]
                samples.append(sample_output)
        finally:
            if previous_training_mode:
                model.train()

        record = {
            "stage": self.stage_name,
            "global_step": int(state.global_step),
            "samples": samples,
        }
        self._write_record(record)

        print(
            f"[sample_generation stage={self.stage_name} step={int(state.global_step)}] "
            f"saved to {self.output_path}",
            flush=True,
        )
        for sample in samples:
            print(f"[sample_generation:{sample['name']}] {sample['generation']}", flush=True)
        return control


def load_blend_for_stage(
        stage: StageSpec,
        seed: int,
        *,
        max_profiler_tokens: int,
        max_composer_tokens: int,
) -> BlendResult:
    return load_stage1_blended_dataset(
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
        max_steps=args.max_steps,
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
    projector_params = [
        parameter
        for name, parameter in named_parameters
        if name.startswith("k_mem_proj.") or name.startswith("v_mem_proj.")
    ]
    covered_prefixes = ("profiler.", "extruder.", "k_mem_proj.", "v_mem_proj.")
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
            or key.startswith("k_mem_proj.")
            or key.startswith("v_mem_proj.")
        )
    ]
    if critical_missing:
        raise RuntimeError(f"Critical weights are missing from checkpoint {checkpoint_dir}: {critical_missing}")
    filtered_unexpected_keys = [
        key for key in unexpected_keys
        if not (
            key.startswith("projector.")
            or key == "soft_prompt_scale"
            or key == "sep_token"
            or ".gate_proj." in key
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


def _resolve_resumable_checkpoint_source(trainer: Trainer, stage_output_dir: Path) -> Optional[Path]:
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
        trainer: Trainer,
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
        memory_upper_layers=args.memory_upper_layers,
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
        callbacks.append(
            FixedSampleGenerationCallback(
                stage_name=stage.name,
                stage_output_dir=stage_output_dir,
                profiler_tokenizer=profiler_tokenizer,
                composer_tokenizer=composer_tokenizer,
                max_profiler_len=args.max_profiler_len,
                max_composer_len=args.max_composer_len,
                every_steps=500,
                max_new_tokens=128,
            )
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=(optimizer, None),
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
