from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.rationale import (
    char_spans_to_token_spans,
    make_example_id,
    normalize_relevance_distribution,
    save_teacher_rationale_records,
    segment_prompt_into_spans,
)
from models.dataset_mixing import load_stage1_blended_dataset


class BlendedMessagesToPECSamples:
    def __init__(self, base_dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        normalized = {
            "example_id": make_example_id(index=idx),
            "prompt": str(sample.get("prompt", "")).strip(),
            "answer": str(sample.get("answer", "")).strip(),
            "source": sample.get("source", ""),
        }
        for key in ("task_type", "mh_target_texts"):
            if key in sample:
                normalized[key] = sample[key]
        return normalized


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build offline teacher rationale targets for PEC training.")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--stage", choices=["stage1"], default="stage1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-train-samples", type=int, default=200_000)
    parser.add_argument("--teacher-model-name", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--upper-layers", type=int, default=4)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=base_dir / ".cache" / "teacher_rationale")
    return parser.parse_args()


def get_device(requested_device: str | None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_dataset(args: argparse.Namespace) -> BlendedMessagesToPECSamples:
    if args.stage != "stage1":
        raise ValueError(f"Unsupported stage for teacher rationale generation: {args.stage}")
    blend_result = load_stage1_blended_dataset(
        split="train",
        seed=args.seed,
        epoch_size=args.stage1_train_samples,
        with_replacement=True,
        max_profiler_tokens=args.max_seq_len,
        max_composer_tokens=args.max_seq_len,
    )
    return BlendedMessagesToPECSamples(blend_result.dataset)


def render_qwen_text(tokenizer, *, prompt: str, answer: str | None = None, add_generation_prompt: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def find_subsequence(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int | None:
    if not needle:
        return None
    if len(needle) > len(haystack):
        return None
    last_start = len(haystack) - len(needle)
    for index in range(max(int(start), 0), last_start + 1):
        if list(haystack[index:index + len(needle)]) == list(needle):
            return index
    return None


def aggregate_answer_to_prompt_attention(
    *,
    attentions: Sequence[torch.Tensor],
    answer_positions: torch.Tensor,
    prompt_positions: torch.Tensor,
    upper_layers: int,
) -> torch.Tensor:
    selected_layers = list(attentions[-int(upper_layers):]) if upper_layers > 0 else list(attentions)
    if not selected_layers:
        raise ValueError("No attention tensors were returned by the teacher model.")

    layer_tensors = [layer[0].float() for layer in selected_layers]  # [H, T, T]
    stacked = torch.stack(layer_tensors, dim=0)  # [L, H, T, T]
    answer_attention = stacked[:, :, answer_positions, :]
    prompt_attention = answer_attention[..., prompt_positions]
    return prompt_attention.mean(dim=(0, 1, 2))  # [Tprompt]


def build_teacher_record(
    *,
    sample: Dict[str, Any],
    tokenizer,
    model,
    device: torch.device,
    max_seq_len: int,
    upper_layers: int,
) -> Dict[str, Any] | None:
    prompt = str(sample.get("prompt", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    task_type = sample.get("task_type")
    if not prompt or not answer:
        return None

    span_boundaries = segment_prompt_into_spans(prompt, task_type=task_type)
    if not span_boundaries:
        return None

    prompt_text = render_qwen_text(
        tokenizer,
        prompt=prompt,
        answer=None,
        add_generation_prompt=True,
    )
    full_text = render_qwen_text(
        tokenizer,
        prompt=prompt,
        answer=answer,
        add_generation_prompt=False,
    )

    prompt_chat_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )["input_ids"]
    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )
    full_input_ids = full_inputs["input_ids"][0].tolist()
    if len(full_input_ids) <= len(prompt_chat_ids):
        return None

    raw_prompt_inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_len,
    )
    raw_prompt_ids = raw_prompt_inputs["input_ids"]
    raw_prompt_start = find_subsequence(prompt_chat_ids, raw_prompt_ids)
    if raw_prompt_start is None:
        raw_prompt_start = find_subsequence(full_input_ids[:len(prompt_chat_ids)], raw_prompt_ids)
    if raw_prompt_start is None:
        raise ValueError(f"Failed to locate raw prompt token window for example_id={sample['example_id']}")

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )["input_ids"]
    answer_start = len(prompt_chat_ids)
    answer_end = min(answer_start + len(answer_ids), len(full_input_ids))
    if answer_end <= answer_start:
        return None

    full_inputs = {key: value.to(device) for key, value in full_inputs.items()}
    with torch.inference_mode():
        outputs = model(
            **full_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    prompt_positions = torch.arange(
        raw_prompt_start,
        raw_prompt_start + len(raw_prompt_ids),
        device=device,
        dtype=torch.long,
    )
    answer_positions = torch.arange(answer_start, answer_end, device=device, dtype=torch.long)
    token_relevance = aggregate_answer_to_prompt_attention(
        attentions=outputs.attentions,
        answer_positions=answer_positions,
        prompt_positions=prompt_positions,
        upper_layers=upper_layers,
    ).detach().cpu()

    token_spans = char_spans_to_token_spans(
        offset_mapping=raw_prompt_inputs["offset_mapping"],
        span_boundaries=span_boundaries,
        valid_token_count=len(raw_prompt_ids),
    )
    span_scores = [0.0 for _ in span_boundaries]
    for span_index, token_start, token_end in token_spans:
        span_scores[span_index] = float(token_relevance[token_start:token_end].sum().item())

    return {
        "example_id": str(sample["example_id"]),
        "span_boundaries": span_boundaries,
        "teacher_relevance": normalize_relevance_distribution(span_scores),
    }


def save_records_incrementally(
    *,
    output_path: Path,
    records: Iterable[Dict[str, Any]],
    append: bool,
) -> None:
    if output_path.suffix == ".pt":
        save_teacher_rationale_records(records, output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Teacher rationale generation currently supports --batch-size 1 only.")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dataset = build_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model_name,
        cache_dir=str(args.cache_dir),
    )
    ensure_tokenizer_padding(tokenizer)

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_name,
        torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        cache_dir=str(args.cache_dir),
    )
    model.to(device)
    model.eval()

    start_index = max(int(args.start_index), 0)
    end_index = len(dataset) if args.limit is None else min(len(dataset), start_index + int(args.limit))
    print(
        f"Building teacher rationale records: stage={args.stage} samples={end_index - start_index} "
        f"model={args.teacher_model_name} device={device}"
    )

    buffered_records: List[Dict[str, Any]] = []
    wrote_any_records = False
    written_record_count = 0
    for dataset_index in range(start_index, end_index):
        sample = dataset[dataset_index]
        if "example_id" not in sample:
            sample["example_id"] = make_example_id(index=dataset_index)
        record = build_teacher_record(
            sample=sample,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_seq_len=args.max_seq_len,
            upper_layers=args.upper_layers,
        )
        if record is None:
            continue
        buffered_records.append(record)

        if len(buffered_records) >= max(1, int(args.flush_every)) and args.output_path.suffix != ".pt":
            save_records_incrementally(
                output_path=args.output_path,
                records=buffered_records,
                append=wrote_any_records,
            )
            wrote_any_records = True
            written_record_count += len(buffered_records)
            print(f"  flushed {len(buffered_records)} records at dataset_index={dataset_index}", flush=True)
            buffered_records = []

    if buffered_records:
        if args.output_path.suffix == ".pt":
            save_records_incrementally(
                output_path=args.output_path,
                records=buffered_records,
                append=False,
            )
        else:
            save_records_incrementally(
                output_path=args.output_path,
                records=buffered_records,
                append=wrote_any_records,
            )
        written_record_count += len(buffered_records)
        wrote_any_records = True

    print(f"Saved teacher rationale records to {args.output_path} (records_written={written_record_count})")


if __name__ == "__main__":
    main()
