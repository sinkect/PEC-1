from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from torch.utils.data import ConcatDataset, Dataset, Subset

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
except ModuleNotFoundError:  # Optional for lightweight utility imports/tests.
    HFDataset = Any
    load_dataset = None

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:  # Optional for lightweight utility imports/tests.
    snapshot_download = None


PromptAnswerSample = Dict[str, Any]
LONG_MAGPIE_FILTER_CACHE_VERSION = 1
HARP_DATA_URL = "https://github.com/aadityasingh/HARP/raw/main/HARP.jsonl.zip"


def _is_primary_dataset_process() -> bool:
    rank = os.environ.get("RANK")
    return rank is None or rank == "0"


def _dataset_log(message: str) -> None:
    if not _is_primary_dataset_process():
        return

    rank = os.environ.get("RANK")
    prefix = "[dataset]" if rank is None else f"[dataset rank={rank}]"
    print(f"{prefix} {message}", flush=True)


def _dataset_cache_root() -> Path:
    override = os.environ.get("PEC_DATASET_CACHE_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / ".cache" / "pec_datasets"


def _long_magpie_filter_cache_dir() -> Path:
    return _dataset_cache_root() / "long_magpie_filters"


def _distributed_world_size() -> int:
    try:
        return max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    except ValueError:
        return 1


def _build_long_magpie_filter_cache_metadata(
    *,
    dataset: Dataset,
    max_profiler_tokens: int,
    max_composer_tokens: int,
    profiler_tokenizer_name: str,
    composer_tokenizer_name: str,
) -> Dict[str, Any]:
    dataset_fingerprint = getattr(dataset, "_fingerprint", None)
    if dataset_fingerprint is None:
        dataset_fingerprint = f"{dataset.__class__.__name__}:{len(dataset)}"

    return {
        "cache_version": LONG_MAGPIE_FILTER_CACHE_VERSION,
        "dataset_fingerprint": str(dataset_fingerprint),
        "dataset_rows": int(len(dataset)),
        "max_profiler_tokens": int(max_profiler_tokens),
        "max_composer_tokens": int(max_composer_tokens),
        "profiler_tokenizer_name": str(profiler_tokenizer_name),
        "composer_tokenizer_name": str(composer_tokenizer_name),
    }


def _long_magpie_filter_cache_path(cache_metadata: Dict[str, Any]) -> Path:
    cache_key = hashlib.sha256(
        json.dumps(cache_metadata, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return _long_magpie_filter_cache_dir() / f"{cache_key}.json"


def _load_long_magpie_filter_cache(
    cache_path: Path,
    *,
    cache_metadata: Dict[str, Any],
) -> Optional[List[int]]:
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if payload.get("cache_metadata") != cache_metadata:
        return None

    kept_indices = payload.get("kept_indices")
    if not isinstance(kept_indices, list):
        return None
    if not all(isinstance(index, int) for index in kept_indices):
        return None
    return kept_indices


def _save_long_magpie_filter_cache(
    cache_path: Path,
    *,
    cache_metadata: Dict[str, Any],
    kept_indices: List[int],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_metadata": cache_metadata,
        "kept_indices": kept_indices,
    }
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(cache_path)


def _wait_for_long_magpie_filter_cache(
    cache_path: Path,
    *,
    cache_metadata: Dict[str, Any],
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 5.0,
) -> Optional[List[int]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        kept_indices = _load_long_magpie_filter_cache(
            cache_path,
            cache_metadata=cache_metadata,
        )
        if kept_indices is not None:
            return kept_indices
        time.sleep(poll_interval_seconds)
    return None


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
        normalized_sample = dict(prompt_answer)
        normalized_sample["prompt"] = str(prompt_answer.get("prompt", "")).strip()
        normalized_sample["answer"] = str(prompt_answer.get("answer", "")).strip()
        normalized_sample["source"] = self._source_name
        return normalized_sample


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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        normalized = [_normalize_text(item) for item in value]
        return [item for item in normalized if item]
    return []


def _normalize_morehop_context_entries(sample: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    raw_context = sample.get("context")
    if raw_context is None:
        raw_context = sample.get("paragraphs")
    if not isinstance(raw_context, (list, tuple)):
        return []

    normalized_entries: List[Tuple[str, List[str]]] = []
    for entry in raw_context:
        title = ""
        sentences: List[str] = []

        if isinstance(entry, dict):
            title = _normalize_text(
                entry.get("title")
                or entry.get("paragraph_support_title")
                or entry.get("name")
            )
            sentences = _normalize_string_list(
                entry.get("sentences")
                or entry.get("content")
                or entry.get("paragraph")
                or entry.get("paragraphs")
                or entry.get("context")
                or entry.get("text")
            )
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            title = _normalize_text(entry[0])
            sentences = _normalize_string_list(entry[1])

        if title or sentences:
            normalized_entries.append((title, sentences))
    return normalized_entries


def _extract_morehop_support_answers(sample: Dict[str, Any]) -> List[str]:
    decomposition = sample.get("question_decomposition")
    if not isinstance(decomposition, list):
        return []

    support_answers: List[str] = []
    for step in decomposition:
        if not isinstance(step, dict):
            continue
        support_title = _normalize_text(step.get("paragraph_support_title"))
        if not support_title:
            continue
        answer = _normalize_text(step.get("answer"))
        if answer:
            support_answers.append(answer)
    return support_answers


def build_morehop_prompt(sample: Dict[str, Any]) -> str:
    question = _normalize_text(sample.get("question"))
    context_entries = _normalize_morehop_context_entries(sample)

    prompt_parts = ["Question:\n", question, "\n\nContext:\n"]
    prompt = "".join(prompt_parts)

    for paragraph_index, (title, sentences) in enumerate(context_entries):
        prompt += f"[{title}]\n" if title else "[Untitled]\n"
        for sentence in sentences:
            sentence_text = _normalize_text(sentence)
            prompt += sentence_text
            prompt += "\n"

        if paragraph_index != len(context_entries) - 1:
            prompt += "\n"

    return prompt.strip()


def format_morehopqa(sample: Dict[str, Any]) -> PromptAnswerSample:
    prompt = build_morehop_prompt(sample)
    return {
        "prompt": prompt,
        "answer": _normalize_text(sample.get("answer")),
        "task_type": "morehopqa",
        "mh_target_texts": _extract_morehop_support_answers(sample),
    }


def _first_non_empty(values: Sequence[Any]) -> str:
    for value in values:
        normalized = _normalize_text(value)
        if normalized:
            return normalized
    return ""


def _extract_nq_short_answer_text(sample: Dict[str, Any]) -> str:
    answers = sample.get("answers")
    if isinstance(answers, dict):
        span_text = answers.get("span_text")
        if isinstance(span_text, list):
            return _first_non_empty(span_text)
        return _normalize_text(span_text)
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, dict):
                answer_text = _normalize_text(answer.get("span_text") or answer.get("text"))
                if answer_text:
                    return answer_text
            else:
                answer_text = _normalize_text(answer)
                if answer_text:
                    return answer_text
    return _normalize_text(sample.get("short answer") or sample.get("short_answer"))


def format_nq_short(sample: Dict[str, Any]) -> PromptAnswerSample:
    question = _normalize_text(sample.get("question") or sample.get("questions"))
    context = _normalize_text(
        sample.get("context")
        or sample.get("contexts")
        or sample.get("paragraph")
        or sample.get("document_text")
    )
    answer = _extract_nq_short_answer_text(sample)
    prompt = f"Question:\n{question}\n\nContext:\n{context}".strip()
    return {
        "prompt": prompt,
        "answer": answer,
        "task_type": "nq_short",
    }


def format_harp(sample: Dict[str, Any]) -> PromptAnswerSample:
    problem = _normalize_text(sample.get("problem"))
    answer = _normalize_text(sample.get("answer"))
    prompt = f"Problem:\n{problem}".strip()
    return {
        "prompt": prompt,
        "answer": answer,
        "task_type": "harp",
    }


def _keep_harp_sample(sample: Dict[str, Any]) -> bool:
    return sample.get("multiple_choice_only") is False


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
    max_profiler_tokens: int = 6144,
    max_composer_tokens: int = 6144,
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
    max_profiler_tokens: int = 6144,
    max_composer_tokens: int = 6144,
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
    max_profiler_tokens: int = 6144,
    max_composer_tokens: int = 6144,
    profiler_tokenizer_name: str = "answerdotai/ModernBERT-base",
    composer_tokenizer_name: str = "Qwen/Qwen3-1.7B",
    batch_size: int = 256,
) -> Subset:
    """Builds a deterministic subset of LongMagpie samples under tokenizer budgets."""
    cache_metadata = _build_long_magpie_filter_cache_metadata(
        dataset=dataset,
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
        profiler_tokenizer_name=profiler_tokenizer_name,
        composer_tokenizer_name=composer_tokenizer_name,
    )
    cache_path = _long_magpie_filter_cache_path(cache_metadata)
    cached_indices = _load_long_magpie_filter_cache(
        cache_path,
        cache_metadata=cache_metadata,
    )
    if cached_indices is not None:
        _dataset_log(
            f"Reusing cached LongMagpie filter: {cache_path} "
            f"(kept={len(cached_indices)}/{len(dataset)})"
        )
        return Subset(dataset, cached_indices)

    should_wait_for_primary_cache = _distributed_world_size() > 1 and not _is_primary_dataset_process()
    if should_wait_for_primary_cache:
        cached_indices = _wait_for_long_magpie_filter_cache(
            cache_path,
            cache_metadata=cache_metadata,
        )
        if cached_indices is not None:
            return Subset(dataset, cached_indices)

    profiler_tokenizer, composer_tokenizer = _get_long_context_filter_tokenizers(
        profiler_tokenizer_name=profiler_tokenizer_name,
        composer_tokenizer_name=composer_tokenizer_name,
    )

    total_rows = len(dataset)
    total_batches = max(1, (total_rows + batch_size - 1) // batch_size)
    progress_interval = max(1, total_batches // 20)
    started_at = time.monotonic()
    _dataset_log(
        "Filtering LongMagpie by tokenizer length "
        f"(rows={total_rows}, batch_size={batch_size}, "
        f"max_profiler_tokens={max_profiler_tokens}, max_composer_tokens={max_composer_tokens})"
    )

    kept_indices: List[int] = []
    for batch_index, start in enumerate(range(0, total_rows, batch_size), start=1):
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

        if batch_index == 1 or batch_index % progress_interval == 0 or batch_index == total_batches:
            processed_rows = min(start + len(batch), total_rows)
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rows_per_second = processed_rows / elapsed
            _dataset_log(
                "LongMagpie filter progress: "
                f"{processed_rows}/{total_rows} rows ({(processed_rows / total_rows) * 100:.1f}%), "
                f"kept={len(kept_indices)}, "
                f"speed={rows_per_second:.1f} rows/s"
            )

    _dataset_log(
        f"LongMagpie filter complete: kept={len(kept_indices)}/{total_rows} rows "
        f"({(len(kept_indices) / max(total_rows, 1)) * 100:.1f}%)"
    )
    _save_long_magpie_filter_cache(
        cache_path,
        cache_metadata=cache_metadata,
        kept_indices=kept_indices,
    )
    _dataset_log(f"Saved LongMagpie filter cache: {cache_path}")
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
    max_profiler_tokens: int = 6144,
    max_composer_tokens: int = 6144,
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


def load_stage1_blended_dataset(
    *,
    split: str = "train",
    seed: int = 42,
    epoch_size: int = 30_000,
    with_replacement: bool = False,
    max_profiler_tokens: int = 6144,
    max_composer_tokens: int = 6144,
) -> BlendResult:
    """Loads the 30k Stage 1 blend with LongMagpie-heavy sampling."""
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
    max_profiler_tokens: int = 6144,
    max_composer_tokens: int = 6144,
) -> BlendResult:
    """Loads the Stage 2/3 blend with MoreHopQA/HARP/NQ-short 50/35/15 sampling."""
    del max_profiler_tokens, max_composer_tokens
    if load_dataset is None:
        raise ModuleNotFoundError("The 'datasets' package is required to load blended datasets.")

    _dataset_log(
        "Loading Stage 23 datasets "
        f"(split={split}, ratios=[50, 35, 15])"
    )
    morehopqa_hf = load_dataset("alabnii/morehopqa", split="test")
    harp_hf = _load_harp_dataset()
    nq_short_hf = load_dataset("ghmfx/natural-questions-short", split=split)

    if hasattr(harp_hf, "filter"):
        harp_hf = harp_hf.filter(
            _keep_harp_sample,
            desc="Filtering HARP rows with multiple_choice_only == false",
        )

    if hasattr(nq_short_hf, "filter"):
        nq_short_hf = nq_short_hf.filter(
            lambda sample: bool(sample.get("has_correct_context", True)) and bool(_extract_nq_short_answer_text(sample)),
            desc="Filtering Natural Questions short-answer rows with valid contexts",
        )

    _dataset_log(
        "Stage 23 sources loaded: "
        f"morehopqa={len(morehopqa_hf)}, "
        f"harp={len(harp_hf)}, "
        f"nq_short={len(nq_short_hf)}"
    )

    morehopqa_ds = HFDatasetAdapter(morehopqa_hf, format_morehopqa, "morehopqa")
    harp_ds = HFDatasetAdapter(harp_hf, format_harp, "harp")
    nq_short_ds = HFDatasetAdapter(nq_short_hf, format_nq_short, "nq_short")

    return build_ratio_concat_dataset(
        datasets=[morehopqa_ds, harp_ds, nq_short_ds],
        ratios=[50, 35, 15],
        seed=seed,
        epoch_size=epoch_size,
        with_replacement=with_replacement,
        source_names=["morehopqa", "harp", "nq_short"],
    )


def load_blended_dataset(
    *,
    split: str = "train",
    seed: int = 42,
    epoch_size: Optional[int] = None,
    with_replacement: bool = False,
    ratios: Sequence[int] = (4, 4, 2),
    max_profiler_tokens: int = 6144,
    max_composer_tokens: int = 6144,
) -> BlendResult:
    """Loads and blends Open-Platypus, LongMagpie, and no_robots with custom ratios."""
    if load_dataset is None:
        raise ModuleNotFoundError("The 'datasets' package is required to load blended datasets.")

    _dataset_log(
        "Loading blended datasets "
        f"(split={split}, ratios={list(ratios)}, "
        f"max_profiler_tokens={max_profiler_tokens}, max_composer_tokens={max_composer_tokens})"
    )
    open_platypus_hf = load_dataset("garage-bAInd/Open-Platypus", split=split)
    long_magpie_hf = _load_long_magpie_dataset(split=split)
    no_robots_hf = load_dataset("HuggingFaceH4/no_robots", split=split)
    _dataset_log(
        "Source datasets loaded: "
        f"open_platypus={len(open_platypus_hf)}, "
        f"long_magpie={len(long_magpie_hf)}, "
        f"no_robots={len(no_robots_hf)}"
    )

    long_magpie_hf = build_long_magpie_subset_with_length_limit(
        long_magpie_hf,
        max_profiler_tokens=max_profiler_tokens,
        max_composer_tokens=max_composer_tokens,
    )
    _dataset_log(f"LongMagpie filtered size: {len(long_magpie_hf)}")

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
    _dataset_log(f"Resolving LongMagpie snapshot from Hugging Face: repo_id={repo_id}")
    snapshot_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    _dataset_log(f"LongMagpie snapshot ready: {snapshot_dir}")
    return load_dataset(snapshot_dir, split=split)


def _load_harp_dataset() -> HFDataset:
    if load_dataset is None:
        raise ModuleNotFoundError("The 'datasets' package is required to load HARP.")

    return load_dataset(
        "json",
        data_files={"train": HARP_DATA_URL},
        split="train",
    )


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
