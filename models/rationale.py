from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


SpanBoundary = Tuple[int, int]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NON_EMPTY_LINE_RE = re.compile(r"[^\n]+")


def make_example_id(*, index: int, prefix: str = "blend") -> str:
    return f"{prefix}:{int(index):09d}"


def _trim_span(text: str, start: int, end: int) -> SpanBoundary | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return start, end


def _iter_non_empty_line_spans(text: str) -> List[SpanBoundary]:
    spans: List[SpanBoundary] = []
    for match in _NON_EMPTY_LINE_RE.finditer(text):
        trimmed = _trim_span(text, match.start(), match.end())
        if trimmed is not None:
            spans.append(trimmed)
    return spans


def _split_span_into_sentences(text: str, span: SpanBoundary) -> List[SpanBoundary]:
    start, end = span
    segment = text[start:end]
    sentence_spans: List[SpanBoundary] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(segment):
        local_span = _trim_span(text, start + cursor, start + match.start())
        if local_span is not None:
            sentence_spans.append(local_span)
        cursor = match.end()
    tail_span = _trim_span(text, start + cursor, end)
    if tail_span is not None:
        sentence_spans.append(tail_span)
    return sentence_spans or [span]


def _is_heading_text(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(":") and len(stripped) <= 40


def _is_title_text(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 2 and stripped.startswith("[") and stripped.endswith("]")


def _merge_heading_spans(text: str, spans: Sequence[SpanBoundary]) -> List[SpanBoundary]:
    if not spans:
        return []

    merged: List[SpanBoundary] = []
    index = 0
    while index < len(spans):
        current = spans[index]
        current_text = text[current[0]:current[1]]
        if index + 1 < len(spans) and (_is_heading_text(current_text) or _is_title_text(current_text)):
            merged.append((current[0], spans[index + 1][1]))
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def _merge_short_spans(text: str, spans: Sequence[SpanBoundary], *, min_chars: int) -> List[SpanBoundary]:
    if not spans:
        return []

    merged: List[SpanBoundary] = []
    active_start, active_end = spans[0]
    for start, end in spans[1:]:
        active_len = len(text[active_start:active_end].strip())
        current_len = len(text[start:end].strip())
        if active_len < min_chars or current_len < max(8, min_chars // 2):
            active_end = end
            continue
        merged.append((active_start, active_end))
        active_start, active_end = start, end

    if merged and len(text[active_start:active_end].strip()) < min_chars:
        prev_start, _ = merged[-1]
        merged[-1] = (prev_start, active_end)
    else:
        merged.append((active_start, active_end))
    return merged


def segment_prompt_into_spans(
    prompt: str,
    *,
    task_type: str | None = None,
    min_chars: int = 24,
) -> List[List[int]]:
    text = str(prompt or "")
    if not text.strip():
        return []

    line_spans = _iter_non_empty_line_spans(text)
    candidate_spans: List[SpanBoundary] = []
    for line_span in line_spans:
        line_text = text[line_span[0]:line_span[1]].strip()
        should_split_line = (
            len(line_text) >= 80
            and not _is_heading_text(line_text)
            and not _is_title_text(line_text)
            and any(char in line_text for char in ".!?")
        )
        if task_type == "morehopqa" and _is_title_text(line_text):
            should_split_line = False
        if should_split_line:
            candidate_spans.extend(_split_span_into_sentences(text, line_span))
            continue
        candidate_spans.append(line_span)

    merged_spans = _merge_heading_spans(text, candidate_spans)
    merged_spans = _merge_short_spans(text, merged_spans, min_chars=max(8, int(min_chars)))
    return [[start, end] for start, end in merged_spans]


def normalize_relevance_distribution(values: Sequence[float]) -> List[float]:
    cleaned = [max(float(value), 0.0) for value in values]
    total = sum(cleaned)
    if total <= 0.0:
        if not cleaned:
            return []
        uniform = 1.0 / float(len(cleaned))
        return [uniform for _ in cleaned]
    return [value / total for value in cleaned]


def load_teacher_rationale_records(path: str | Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}

    rationale_path = Path(path)
    if not rationale_path.exists():
        raise FileNotFoundError(f"Teacher rationale file not found: {rationale_path}")

    if rationale_path.suffix == ".pt":
        payload = torch.load(rationale_path, map_location="cpu")
        if isinstance(payload, dict):
            if "records" in payload and isinstance(payload["records"], list):
                records = payload["records"]
            else:
                records = list(payload.values())
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"Unsupported teacher rationale payload type: {type(payload)!r}")
    else:
        records = []
        with rationale_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(json.loads(stripped))

    loaded: Dict[str, Dict[str, Any]] = {}
    for record in records:
        example_id = str(record["example_id"])
        loaded[example_id] = {
            "example_id": example_id,
            "span_boundaries": [
                [int(boundary[0]), int(boundary[1])]
                for boundary in record.get("span_boundaries", [])
            ],
            "teacher_relevance": normalize_relevance_distribution(record.get("teacher_relevance", [])),
        }
    return loaded


def save_teacher_rationale_records(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> None:
    rationale_path = Path(path)
    rationale_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_records = [
        {
            "example_id": str(record["example_id"]),
            "span_boundaries": [
                [int(boundary[0]), int(boundary[1])]
                for boundary in record.get("span_boundaries", [])
            ],
            "teacher_relevance": normalize_relevance_distribution(record.get("teacher_relevance", [])),
        }
        for record in records
    ]

    if rationale_path.suffix == ".pt":
        torch.save({"records": normalized_records}, rationale_path)
        return

    with rationale_path.open("w", encoding="utf-8") as handle:
        for record in normalized_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def char_spans_to_token_spans(
    *,
    offset_mapping: Sequence[Sequence[int]] | torch.Tensor,
    span_boundaries: Sequence[Sequence[int]],
    valid_token_count: int | None = None,
) -> List[Tuple[int, int, int]]:
    if isinstance(offset_mapping, torch.Tensor):
        offsets = offset_mapping.tolist()
    else:
        offsets = [list(offset) for offset in offset_mapping]

    max_token_count = len(offsets) if valid_token_count is None else min(len(offsets), int(valid_token_count))
    normalized_offsets: List[Tuple[int, int]] = [
        (int(offset[0]), int(offset[1]))
        for offset in offsets[:max_token_count]
    ]

    token_spans: List[Tuple[int, int, int]] = []
    for span_index, boundary in enumerate(span_boundaries):
        if len(boundary) != 2:
            continue
        span_start = int(boundary[0])
        span_end = int(boundary[1])
        if span_end <= span_start:
            continue

        token_indices = [
            token_index
            for token_index, (token_start, token_end) in enumerate(normalized_offsets)
            if token_end > token_start and token_start < span_end and token_end > span_start
        ]
        if not token_indices:
            continue
        token_spans.append((span_index, token_indices[0], token_indices[-1] + 1))
    return token_spans
