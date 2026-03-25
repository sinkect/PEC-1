from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

try:
    from rouge_score import rouge_scorer
except ModuleNotFoundError:  # Optional for lightweight unit tests.
    rouge_scorer = None

try:
    from transformers import PreTrainedTokenizerBase
except ModuleNotFoundError:  # Optional for lightweight unit tests.
    PreTrainedTokenizerBase = Any

DEFAULT_BASE_MODELS: Sequence[str] = (
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
)
THINK_END_MARKER = "</think>"


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
    """Creates deterministic train/eval index lists from one dataset."""
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


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary_csv(rows: Sequence[Dict[str, Any]], path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def slugify_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def render_qwen_user_prompt(
    tokenizer: PreTrainedTokenizerBase,
    user_text: str,
    *,
    enable_thinking: bool = False,
) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def resolve_think_end_token_index(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    max_length: int,
    add_special_tokens: bool = True,
) -> int:
    normalized_text = str(text)
    marker_index = normalized_text.find(THINK_END_MARKER)
    if marker_index < 0:
        return 0

    prefix_text = normalized_text[: marker_index + len(THINK_END_MARKER)]
    tokenized_prefix = tokenizer(
        prefix_text,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return len(tokenized_prefix)


def resolve_think_end_token_indices(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    max_length: int,
    add_special_tokens: bool = True,
) -> List[int]:
    return [
        resolve_think_end_token_index(
            tokenizer,
            text,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        for text in texts
    ]


def thinking_mode_name(enable_thinking: bool) -> str:
    return "thinking_on" if enable_thinking else "thinking_off"


def strip_thinking_trace(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", " ", str(text), flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def apply_mask_deterministically(text: str, masker: Callable[[str], str], seed: int) -> str:
    state = random.getstate()
    random.seed(seed)
    try:
        return masker(text)
    finally:
        random.setstate(state)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match_score(reference: str, prediction: str) -> float:
    return float(normalize_text(reference) == normalize_text(prediction))


def token_f1_score(reference: str, prediction: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()

    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    common = Counter(ref_tokens) & Counter(pred_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


_ROUGE_SCORER = None if rouge_scorer is None else rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_f1_score(reference: str, prediction: str) -> float:
    if not reference.strip() and not prediction.strip():
        return 1.0
    if not reference.strip() or not prediction.strip():
        return 0.0
    if _ROUGE_SCORER is None:
        return exact_match_score(reference, prediction)
    return float(_ROUGE_SCORER.score(reference, prediction)["rougeL"].fmeasure)


def compute_text_metrics(reference: str, prediction: str) -> Dict[str, float]:
    return {
        "exact_match": exact_match_score(reference, prediction),
        "token_f1": token_f1_score(reference, prediction),
        "rougeL_f1": rouge_l_f1_score(reference, prediction),
    }


def aggregate_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {"num_samples": 0, "exact_match": 0.0, "token_f1": 0.0, "rougeL_f1": 0.0}

    metric_names = ["exact_match", "token_f1", "rougeL_f1"]
    summary = {"num_samples": len(records)}
    for metric_name in metric_names:
        summary[metric_name] = sum(float(record[metric_name]) for record in records) / len(records)
    return summary


def resolve_processed_dataset_jsonl(base_dir: Path) -> Path:
    return base_dir / "data" / "processed_pec_dataset" / "blended_messages_pec_ready.jsonl"


def build_heldout_eval_samples(
    *,
    base_dir: Path,
    dataset_jsonl: Optional[Path] = None,
    split: str = "train",
    seed: int = 42,
    epoch_size: int = 200_000,
    with_replacement: bool = True,
    eval_ratio: float = 0.02,
    max_samples: Optional[int] = None,
    persist_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    dataset_jsonl = dataset_jsonl or resolve_processed_dataset_jsonl(base_dir)

    if dataset_jsonl.exists():
        all_samples = read_jsonl(dataset_jsonl)
        split_indices = split_dataset_indices(len(all_samples), test_size=eval_ratio, seed=seed)
        eval_samples = [all_samples[index] for index in split_indices["eval"]]
    else:
        from .dataset_mixing import load_default_4_4_2_blended_dataset

        blend_result = load_default_4_4_2_blended_dataset(
            split=split,
            seed=seed,
            epoch_size=epoch_size,
            with_replacement=with_replacement,
        )
        pec_ready_dataset = BlendedMessagesToPECSamples(blend_result.dataset)
        split_indices = split_dataset_indices(len(pec_ready_dataset), test_size=eval_ratio, seed=seed)
        eval_samples = [pec_ready_dataset[index] for index in split_indices["eval"]]

    if max_samples is not None:
        eval_samples = eval_samples[:max_samples]

    if persist_path is not None:
        write_jsonl(eval_samples, persist_path)

    return eval_samples
