from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # Optional for helper tests/imports.
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of dataset shards. Defaults to WORLD_SIZE when launched with torchrun, else 1.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="0-based shard index for this process. Defaults to RANK when launched with torchrun, else 0.",
    )
    parser.add_argument(
        "--merge-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When multiple shards are used, merge shard outputs into --output-path on rank 0 after completion.",
    )
    parser.add_argument(
        "--keep-shard-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep per-shard output files after merge.",
    )
    parser.add_argument("--cache-dir", type=Path, default=base_dir / ".cache" / "teacher_rationale")
    return parser.parse_args()


def get_device(requested_device: str | None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    local_rank = os.environ.get("LOCAL_RANK")
    if torch.cuda.is_available():
        if local_rank is not None:
            return torch.device("cuda", int(local_rank))
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def resolve_shard_spec(args: argparse.Namespace) -> tuple[int, int]:
    env_world_size = max(_parse_env_int("WORLD_SIZE", 1), 1)
    env_rank = max(_parse_env_int("RANK", 0), 0)

    num_shards = int(args.num_shards) if args.num_shards is not None else env_world_size
    shard_index = int(args.shard_index) if args.shard_index is not None else env_rank

    if num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must satisfy 0 <= shard_index < num_shards (got {shard_index}, {num_shards})")
    return num_shards, shard_index


def resolve_processing_range(
    *,
    dataset_len: int,
    start_index: int,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> tuple[int, int, int, int]:
    global_start = max(int(start_index), 0)
    global_end = dataset_len if limit is None else min(dataset_len, global_start + int(limit))
    global_count = max(global_end - global_start, 0)

    shard_base = global_count // num_shards
    shard_remainder = global_count % num_shards
    shard_offset = (shard_index * shard_base) + min(shard_index, shard_remainder)
    shard_size = shard_base + (1 if shard_index < shard_remainder else 0)
    shard_start = global_start + shard_offset
    shard_end = shard_start + shard_size
    return global_start, global_end, shard_start, shard_end


def resolve_shard_output_path(output_path: Path, *, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return output_path
    return output_path.with_name(
        f"{output_path.stem}.shard-{shard_index:05d}-of-{num_shards:05d}{output_path.suffix}"
    )


def expected_shard_output_paths(output_path: Path, *, num_shards: int) -> List[Path]:
    return [
        resolve_shard_output_path(output_path, num_shards=num_shards, shard_index=shard_index)
        for shard_index in range(num_shards)
    ]


def maybe_init_process_group(device: torch.device, *, num_shards: int) -> bool:
    if num_shards <= 1:
        return False
    if not torch.distributed.is_available():
        return False
    env_world_size = max(_parse_env_int("WORLD_SIZE", 1), 1)
    if env_world_size <= 1:
        return False
    if torch.distributed.is_initialized():
        return True
    backend = "nccl" if device.type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    return True


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


def merge_shard_outputs(
    *,
    output_path: Path,
    num_shards: int,
    keep_shard_files: bool,
) -> int:
    shard_paths = expected_shard_output_paths(output_path, num_shards=num_shards)
    missing_paths = [path for path in shard_paths if not path.exists()]
    if missing_paths:
        missing_report = ", ".join(str(path) for path in missing_paths[:3])
        raise FileNotFoundError(f"Cannot merge shards; missing shard outputs: {missing_report}")

    merged_record_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".pt":
        merged_records: List[Dict[str, Any]] = []
        for shard_path in shard_paths:
            payload = torch.load(shard_path, map_location="cpu")
            shard_records = payload.get("records", []) if isinstance(payload, dict) else payload
            merged_records.extend(list(shard_records))
            merged_record_count += len(shard_records)
        save_teacher_rationale_records(merged_records, output_path)
    else:
        with output_path.open("w", encoding="utf-8") as merged_handle:
            for shard_path in shard_paths:
                with shard_path.open("r", encoding="utf-8") as shard_handle:
                    for line in shard_handle:
                        if not line.strip():
                            continue
                        merged_handle.write(line)
                        merged_record_count += 1

    if not keep_shard_files:
        for shard_path in shard_paths:
            shard_path.unlink(missing_ok=True)
    return merged_record_count


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Teacher rationale generation currently supports --batch-size 1 only.")
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ModuleNotFoundError("transformers is required to build teacher rationale records.")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    num_shards, shard_index = resolve_shard_spec(args)
    distributed_ready = maybe_init_process_group(device, num_shards=num_shards)
    dataset = build_dataset(args)
    global_start, global_end, shard_start, shard_end = resolve_processing_range(
        dataset_len=len(dataset),
        start_index=args.start_index,
        limit=args.limit,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    shard_output_path = resolve_shard_output_path(args.output_path, num_shards=num_shards, shard_index=shard_index)

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

    print(
        f"Building teacher rationale records: stage={args.stage} "
        f"global_range=[{global_start}, {global_end}) shard={shard_index}/{num_shards} "
        f"shard_range=[{shard_start}, {shard_end}) model={args.teacher_model_name} device={device} "
        f"output={shard_output_path}",
        flush=True,
    )

    buffered_records: List[Dict[str, Any]] = []
    wrote_any_records = False
    written_record_count = 0
    for dataset_index in range(shard_start, shard_end):
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
                output_path=shard_output_path,
                records=buffered_records,
                append=wrote_any_records,
            )
            wrote_any_records = True
            written_record_count += len(buffered_records)
            print(
                f"[shard {shard_index}/{num_shards}] flushed {len(buffered_records)} records "
                f"at dataset_index={dataset_index}",
                flush=True,
            )
            buffered_records = []

    if buffered_records:
        if shard_output_path.suffix == ".pt":
            save_records_incrementally(
                output_path=shard_output_path,
                records=buffered_records,
                append=False,
            )
        else:
            save_records_incrementally(
                output_path=shard_output_path,
                records=buffered_records,
                append=wrote_any_records,
            )
        written_record_count += len(buffered_records)
        wrote_any_records = True

    print(
        f"[shard {shard_index}/{num_shards}] saved teacher rationale records to {shard_output_path} "
        f"(records_written={written_record_count})",
        flush=True,
    )

    if distributed_ready:
        torch.distributed.barrier()

    if args.merge_shards and num_shards > 1 and shard_index == 0:
        merged_record_count = merge_shard_outputs(
            output_path=args.output_path,
            num_shards=num_shards,
            keep_shard_files=args.keep_shard_files,
        )
        print(
            f"Merged {num_shards} shard files into {args.output_path} "
            f"(records_written={merged_record_count})",
            flush=True,
        )
    elif num_shards > 1 and shard_index == 0:
        print(
            f"Shard outputs kept separate. Final shard files: "
            f"{expected_shard_output_paths(args.output_path, num_shards=num_shards)}",
            flush=True,
        )

    if distributed_ready and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
