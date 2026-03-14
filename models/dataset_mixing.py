from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from torch.utils.data import ConcatDataset, Dataset, Subset
from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import snapshot_download



PromptAnswerSample = Dict[str, str]


@dataclass(frozen=True)
class BlendResult:
    """Container for blended dataset artifacts.

    Attributes:
        dataset: Concat dataset that follows a fixed source ratio.
        per_dataset_counts: Number of selected samples per source dataset.
        total_samples: Total number of blended samples.
        source_names: Human-readable source names.
    """

    dataset: ConcatDataset
    per_dataset_counts: List[int]
    total_samples: int
    source_names: List[str]


class HFDatasetAdapter(Dataset):
    """Adapts a HuggingFace dataset into a unified dialogue-sample schema.

    Each item is normalized to:
        {
            "prompt": str,
            "answer": str,
            "source": str,
        }
    """

    def __init__(
        self,
        dataset: Dataset,
        formatter: Callable[[Dict[str, Any]], PromptAnswerSample],
        source_name: str,
    ) -> None:
        self._dataset = dataset
        self._formatter = formatter
        self._source_name = source_name

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._dataset[index]
        prompt_answer = self._formatter(sample)
        return {
            "prompt": prompt_answer.get("prompt", ""),
            "answer": prompt_answer.get("answer", ""),
            "source": self._source_name,
        }


def format_open_platypus(sample: Dict[str, Any]) -> PromptAnswerSample:
    """Formats one Open-Platypus row into prompt/answer fields."""
    instruction = str(sample.get("instruction", "")).strip()
    input_text = str(sample.get("input", "")).strip()
    answer = str(sample.get("output", "")).strip()

    prompt = instruction
    if input_text:
        prompt = f"{instruction}\n\nInput:\n{input_text}" if instruction else input_text
    return {"prompt": prompt, "answer": answer}


def format_long_magpie(sample: Dict[str, Any]) -> PromptAnswerSample:
    """Formats one LongMagpie row into prompt/answer fields."""
    query = str(sample.get("query", "")).strip()
    context = str(sample.get("context", "")).strip()
    answer = str(sample.get("answer", "")).strip()

    prompt = query if not context else f"{query}\n\nContext:\n{context}"
    return {"prompt": prompt, "answer": answer}


def _estimate_word_length(prompt_answer: PromptAnswerSample) -> int:
    """Estimates total prompt/response length by whitespace token count."""
    total_text = f"{prompt_answer.get('prompt', '')} {prompt_answer.get('answer', '')}".strip()
    return len(total_text.split())


def _build_qwen_training_text(composer_tokenizer, prompt: str, answer: str) -> str:
    return composer_tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


@lru_cache(maxsize=4)
def _get_long_context_filter_tokenizers(
    profiler_tokenizer_name: str = "answerdotai/ModernBERT-base",
    composer_tokenizer_name: str = "Qwen/Qwen3-1.7B",
):
    from transformers import AutoTokenizer

    profiler_tokenizer = AutoTokenizer.from_pretrained(profiler_tokenizer_name)
    composer_tokenizer = AutoTokenizer.from_pretrained(composer_tokenizer_name)
    # These tokenizers are used only for exact length measurement during dataset filtering.
    # Disable built-in max-length warnings so overlong samples can be counted and rejected cleanly.
    profiler_tokenizer.model_max_length = int(1e9)
    composer_tokenizer.model_max_length = int(1e9)
    if composer_tokenizer.pad_token_id is None:
        composer_tokenizer.pad_token = composer_tokenizer.eos_token
    return profiler_tokenizer, composer_tokenizer


def is_prompt_answer_within_token_limits(
    prompt_answer: PromptAnswerSample,
    *,
    profiler_tokenizer,
    composer_tokenizer,
    max_profiler_tokens: int = 6080,
    max_composer_tokens: int = 6080,
) -> bool:
    prompt = str(prompt_answer.get("prompt", "")).strip()
    answer = str(prompt_answer.get("answer", "")).strip()
    if not prompt and not answer:
        return False

    profiler_tokens = profiler_tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    if len(profiler_tokens) > max_profiler_tokens:
        return False

    composer_full_text = _build_qwen_training_text(composer_tokenizer, prompt, answer)
    composer_tokens = composer_tokenizer(
        composer_full_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    return len(composer_tokens) <= max_composer_tokens


def _normalize_batch_slice(batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(batch.keys())
    if not keys:
        return []
    batch_size = len(batch[keys[0]])
    return [{key: batch[key][index] for key in keys} for index in range(batch_size)]


def is_within_long_context_limit(
    sample: Dict[str, Any],
    *,
    max_profiler_tokens: int = 6080,
    max_composer_tokens: int = 6080,
    profiler_tokenizer_name: str = "answerdotai/ModernBERT-base",
    composer_tokenizer_name: str = "Qwen/Qwen3-1.7B",
) -> bool:
    """Returns True when a LongMagpie sample fits the actual tokenizer budgets."""
    prompt_answer = format_long_magpie(sample)
    profiler_tokenizer, composer_tokenizer = _get_long_context_filter_tokenizers(
        profiler_tokenizer_name=profiler_tokenizer_name,
        composer_tokenizer_name=composer_tokenizer_name,
    )
    return is_prompt_answer_within_token_limits(
        prompt_answer,
        profiler_tokenizer=profiler_tokenizer,
        composer_tokenizer=composer_tokenizer,
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
    )


def build_long_magpie_subset_with_length_limit(
    dataset: HFDataset,
    *,
    max_profiler_tokens: int = 6080,
    max_composer_tokens: int = 6080,
    profiler_tokenizer_name: str = "answerdotai/ModernBERT-base",
    composer_tokenizer_name: str = "Qwen/Qwen3-1.7B",
    batch_size: int = 256,
) -> Subset:
    """Builds a deterministic subset of LongMagpie samples under tokenizer budgets."""
    profiler_tokenizer, composer_tokenizer = _get_long_context_filter_tokenizers(
        profiler_tokenizer_name=profiler_tokenizer_name,
        composer_tokenizer_name=composer_tokenizer_name,
    )
    kept_indices: List[int] = []
    for start in range(0, len(dataset), batch_size):
        batch = _normalize_batch_slice(dataset[start:start + batch_size])
        prompt_answers = [format_long_magpie(sample) for sample in batch]
        prompts = [str(sample.get("prompt", "")).strip() for sample in prompt_answers]
        answers = [str(sample.get("answer", "")).strip() for sample in prompt_answers]

        profiler_lengths = [0] * len(prompt_answers)
        composer_lengths = [0] * len(prompt_answers)
        non_empty_indices = [index for index, (prompt, answer) in enumerate(zip(prompts, answers)) if prompt or answer]

        if non_empty_indices:
            filtered_prompts = [prompts[index] for index in non_empty_indices]
            filtered_answers = [answers[index] for index in non_empty_indices]
            profiler_batches = profiler_tokenizer(
                filtered_prompts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
            composer_full_texts = [
                _build_qwen_training_text(composer_tokenizer, prompt, answer)
                for prompt, answer in zip(filtered_prompts, filtered_answers)
            ]
            composer_batches = composer_tokenizer(
                composer_full_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]

            for local_index, profiler_ids, composer_ids in zip(non_empty_indices, profiler_batches, composer_batches):
                profiler_lengths[local_index] = len(profiler_ids)
                composer_lengths[local_index] = len(composer_ids)

        for offset, prompt_answer in enumerate(prompt_answers):
            if not prompts[offset] and not answers[offset]:
                continue
            if profiler_lengths[offset] <= max_profiler_tokens and composer_lengths[offset] <= max_composer_tokens:
                kept_indices.append(start + offset)
    return Subset(dataset, kept_indices)


def format_no_robots(sample: Dict[str, Any]) -> PromptAnswerSample:
    """Returns no_robots first prompt/answer pair without role-based processing."""
    messages = sample.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return {"prompt": "", "answer": ""}

    prompt = str(messages[0].get("content", "")).strip() if isinstance(messages[0], dict) else ""
    answer = str(messages[1].get("content", "")).strip() if isinstance(messages[1], dict) else ""
    return {"prompt": prompt, "answer": answer}


def _select_indices(
    dataset_len: int,
    num_samples: int,
    rng: random.Random,
    *,
    with_replacement: bool,
) -> List[int]:
    if num_samples <= 0:
        return []

    if with_replacement:
        return [rng.randrange(dataset_len) for _ in range(num_samples)]

    if num_samples > dataset_len:
        raise ValueError(
            f"Requested {num_samples} samples from dataset length {dataset_len} without replacement."
        )

    return rng.sample(range(dataset_len), num_samples)


def build_ratio_concat_dataset(
    datasets: Sequence[Dataset],
    ratios: Sequence[int],
    *,
    seed: int = 42,
    epoch_size: Optional[int] = None,
    with_replacement: bool = False,
    source_names: Optional[Sequence[str]] = None,
) -> BlendResult:
    """Builds a ratio-preserving blended ConcatDataset.

    Args:
        datasets: Source datasets to blend.
        ratios: Integer ratios aligned with datasets.
        seed: Random seed for index sampling.
        epoch_size: Optional explicit blended sample count.
        with_replacement: Enables oversampling when epoch_size is large.
        source_names: Optional names for reporting.
    """
    if len(datasets) != len(ratios):
        raise ValueError("datasets and ratios must have equal length.")
    if not datasets:
        raise ValueError("At least one dataset is required.")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive integers.")

    names = list(source_names) if source_names is not None else [f"dataset_{i}" for i in range(len(datasets))]

    source_lengths = [len(ds) for ds in datasets]
    empty_sources = [name for name, length in zip(names, source_lengths) if length == 0]
    if empty_sources:
        length_report = {name: length for name, length in zip(names, source_lengths)}
        raise ValueError(
            "One or more source datasets are empty before blending. "
            f"empty_sources={empty_sources}, source_lengths={length_report}"
        )

    ratio_sum = sum(ratios)
    rng = random.Random(seed)

    if epoch_size is not None and epoch_size <= 0:
        raise ValueError("epoch_size must be > 0 when provided.")

    if epoch_size is None:
        if with_replacement:
            raise ValueError("epoch_size is required when with_replacement=True.")
        ratio_units = [len(ds) // r for ds, r in zip(datasets, ratios)]
        max_unit = min(ratio_units)
        if max_unit <= 0:
            raise ValueError("At least one dataset is too small for the given ratio.")
        counts = [r * max_unit for r in ratios]
    else:
        counts = [(epoch_size * r) // ratio_sum for r in ratios]
        remainder = epoch_size - sum(counts)
        for index in range(remainder):
            counts[index % len(counts)] += 1

    subsets: List[Subset] = []
    for ds, count in zip(datasets, counts):
        indices = _select_indices(len(ds), count, rng, with_replacement=with_replacement)
        subsets.append(Subset(ds, indices))

    blend_dataset = ConcatDataset(subsets)
    return BlendResult(
        dataset=blend_dataset,
        per_dataset_counts=counts,
        total_samples=sum(counts),
        source_names=names,
    )


def load_default_4_4_2_blended_dataset(
    *,
    split: str = "train",
    seed: int = 42,
    epoch_size: Optional[int] = None,
    with_replacement: bool = False,
    max_profiler_tokens: int = 6080,
    max_composer_tokens: int = 6080,
) -> BlendResult:
    """Loads and blends Open-Platypus, LongMagpie, and no_robots by 4:4:2."""
    return load_blended_dataset(
        split=split,
        seed=seed,
        epoch_size=epoch_size,
        with_replacement=with_replacement,
        ratios=[4, 4, 2],
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
    )


def load_stage1_kd_blended_dataset(
    *,
    split: str = "train",
    seed: int = 42,
    epoch_size: int = 30_000,
    with_replacement: bool = False,
    max_profiler_tokens: int = 6080,
    max_composer_tokens: int = 6080,
) -> BlendResult:
    """Loads the 30k Stage 1 KD blend with LongMagpie-heavy sampling."""
    return load_blended_dataset(
        split=split,
        seed=seed,
        epoch_size=epoch_size,
        with_replacement=with_replacement,
        ratios=[2, 7, 1],
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
    )


def load_stage23_blended_dataset(
    *,
    split: str = "train",
    seed: int = 42,
    epoch_size: int = 200_000,
    with_replacement: bool = True,
    max_profiler_tokens: int = 6080,
    max_composer_tokens: int = 6080,
) -> BlendResult:
    """Loads the 200k Stage 2/3 blend with 40/40/20 sampling."""
    return load_blended_dataset(
        split=split,
        seed=seed,
        epoch_size=epoch_size,
        with_replacement=with_replacement,
        ratios=[4, 4, 2],
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
    )


def load_blended_dataset(
    *,
    split: str = "train",
    seed: int = 42,
    epoch_size: Optional[int] = None,
    with_replacement: bool = False,
    ratios: Sequence[int] = (4, 4, 2),
    max_profiler_tokens: int = 6080,
    max_composer_tokens: int = 6080,
) -> BlendResult:
    """Loads and blends Open-Platypus, LongMagpie, and no_robots with custom ratios."""
    if load_dataset is None:
        raise ModuleNotFoundError("The 'datasets' package is required to load blended datasets.")

    open_platypus_hf = load_dataset("garage-bAInd/Open-Platypus", split=split)
    long_magpie_hf = _load_long_magpie_dataset(split=split)
    no_robots_hf = load_dataset("HuggingFaceH4/no_robots", split=split)

    long_magpie_hf = build_long_magpie_subset_with_length_limit(
        long_magpie_hf,
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
    )

    open_platypus_ds = HFDatasetAdapter(open_platypus_hf, format_open_platypus, "open_platypus")
    long_magpie_ds = HFDatasetAdapter(long_magpie_hf, format_long_magpie, "long_magpie")
    no_robots_ds = HFDatasetAdapter(no_robots_hf, format_no_robots, "no_robots")

    return build_ratio_concat_dataset(
        datasets=[open_platypus_ds, long_magpie_ds, no_robots_ds],
        ratios=ratios,
        seed=seed,
        epoch_size=epoch_size,
        with_replacement=with_replacement,
        source_names=["open_platypus", "long_magpie", "no_robots"],
    )


def _load_long_magpie_dataset(split: str = "train") -> HFDataset:
    """Loads LongMagpie single-doc long-context dataset via snapshot download.

    Dataset source:
        caskcsg/LongMagpie_multidoc_longcontext_dataset
    """
    if load_dataset is None:
        raise ModuleNotFoundError("The 'datasets' package is required to load LongMagpie.")
    if snapshot_download is None:
        raise ModuleNotFoundError("The 'huggingface_hub' package is required to load LongMagpie.")

    repo_id = "caskcsg/LongMagpie_singledoc_longcontext_dataset"
    snapshot_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return load_dataset(snapshot_dir, split=split)


def save_dataset_as_jsonl(
    dataset: Dataset,
    output_path: Path | str,
    *,
    max_samples: Optional[int] = None,
) -> int:
    """Saves a torch-style dataset into a JSONL file.

    Args:
        dataset: Any dataset implementing __len__ and __getitem__.
        output_path: Destination JSONL path.
        max_samples: Optional cap for the number of dumped samples.

    Returns:
        Number of saved samples.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    limit = total if max_samples is None else min(total, max_samples)

    with destination.open("w", encoding="utf-8") as file:
        for index in range(limit):
            sample = dataset[index]
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return limit


def save_blend_metadata(
    blend_result: BlendResult,
    output_path: Path | str,
) -> None:
    """Saves blend composition metadata into a JSON file."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_names": blend_result.source_names,
        "per_dataset_counts": blend_result.per_dataset_counts,
        "total_samples": blend_result.total_samples,
    }

    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_sampled_by_source_as_jsonl(
    dataset: Dataset,
    output_path: Path | str,
    *,
    source_names: Sequence[str],
    samples_per_source: int = 3,
    seed: int = 42,
) -> Dict[str, int]:
    """Saves a small, balanced sample from each source into one JSONL file.

    Args:
        dataset: Dataset containing a 'source' field in each sample.
        output_path: Destination JSONL file path.
        source_names: Expected source names to sample from.
        samples_per_source: Number of rows to save per source.
        seed: Random seed for deterministic sampling.

    Returns:
        Dictionary of saved sample counts per source.
    """
    if samples_per_source <= 0:
        raise ValueError("samples_per_source must be > 0.")

    target_sources = set(source_names)
    sampled_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in source_names}
    seen_counts: Dict[str, int] = {name: 0 for name in source_names}
    rng = random.Random(seed)

    for index in range(len(dataset)):
        row = dataset[index]
        source = str(row.get("source", ""))
        if source not in target_sources:
            continue

        seen_counts[source] += 1
        bucket = sampled_rows[source]

        if len(bucket) < samples_per_source:
            bucket.append(row)
            continue

        replacement_index = rng.randrange(seen_counts[source])
        if replacement_index < samples_per_source:
            bucket[replacement_index] = row

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    ordered_rows: List[Dict[str, Any]] = []
    for source in source_names:
        ordered_rows.extend(sampled_rows[source])

    with destination.open("w", encoding="utf-8") as file:
        for row in ordered_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {name: len(sampled_rows[name]) for name in source_names}
