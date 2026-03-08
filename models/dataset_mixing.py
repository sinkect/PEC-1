from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from datasets import Dataset as HFDataset
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import ConcatDataset, Dataset, Subset


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
    answer = str(sample.get("output", "")).strip()

    return {"prompt": instruction, "answer": answer}


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


def is_within_long_context_limit(
    sample: Dict[str, Any],
    *,
    max_word_estimate: int = 6000,
) -> bool:
    """Returns True when a LongMagpie sample is safely under the context budget.

    The heuristic `max_word_estimate=6000` is used to stay below an 8k-token
    model limit with margin.
    """
    prompt_answer = format_long_magpie(sample)
    if not prompt_answer.get("prompt") and not prompt_answer.get("answer"):
        return False
    return _estimate_word_length(prompt_answer) <= max_word_estimate


def build_long_magpie_subset_with_length_limit(
    dataset: HFDataset,
    *,
    max_word_estimate: int = 6000,
) -> Subset:
    """Builds a deterministic subset of LongMagpie samples under length limit."""
    kept_indices: List[int] = []
    for index in range(len(dataset)):
        if is_within_long_context_limit(dataset[index], max_word_estimate=max_word_estimate):
            kept_indices.append(index)
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
) -> BlendResult:
    """Loads and blends Open-Platypus, LongMagpie, and no_robots by 4:4:2."""
    open_platypus_hf = load_dataset("garage-bAInd/Open-Platypus", split=split)
    long_magpie_hf = _load_long_magpie_dataset(split=split)
    no_robots_hf = load_dataset("HuggingFaceH4/no_robots", split=split)

    long_magpie_hf = build_long_magpie_subset_with_length_limit(
        long_magpie_hf,
        max_word_estimate=6000,
    )

    open_platypus_ds = HFDatasetAdapter(open_platypus_hf, format_open_platypus, "open_platypus")
    long_magpie_ds = HFDatasetAdapter(long_magpie_hf, format_long_magpie, "long_magpie")
    no_robots_ds = HFDatasetAdapter(no_robots_hf, format_no_robots, "no_robots")

    return build_ratio_concat_dataset(
        datasets=[open_platypus_ds, long_magpie_ds, no_robots_ds],
        ratios=[4, 4, 2],
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