import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.dataset_mixing import load_stage1_kd_blended_dataset


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate the Stage 1 teacher distill dataset.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=base_dir / "data" / "distill" / "stage1_teacher_qwen3_1p7b.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def render_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def iter_batches(items: List[Dict[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    dtype = get_dtype(device)

    print(f"Device: {device}")
    print(f"Loading Stage 1 subset: num_samples={args.num_samples}")
    blend_result = load_stage1_kd_blended_dataset(
        split="train",
        seed=args.seed,
        epoch_size=args.num_samples,
        with_replacement=False,
    )
    dataset = [blend_result.dataset[index] for index in range(len(blend_result.dataset))]
    print(
        "Blend counts: "
        f"{dict(zip(blend_result.source_names, blend_result.per_dataset_counts))} "
        f"(total={blend_result.total_samples})"
    )

    print(f"Loading teacher model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_tokenizer_padding(tokenizer)

    kwargs = {"low_cpu_mem_usage": True}
    if device.type != "cpu":
        kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    model.to(device)
    model.eval()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with args.output_path.open("w", encoding="utf-8") as file:
        for batch_start, batch_rows in iter_batches(dataset, args.batch_size):
            prompts = [str(row.get("prompt", "")).strip() for row in batch_rows]
            rendered_prompts = [render_prompt(tokenizer, prompt) for prompt in prompts]
            encoded = tokenizer(rendered_prompts, return_tensors="pt", padding=True, truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            generated_ids = output_ids[:, encoded["input_ids"].shape[1]:]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for row, teacher_output in zip(batch_rows, generated_texts):
                payload = {
                    "prompt": str(row.get("prompt", "")).strip(),
                    "answer": str(row.get("answer", "")).strip(),
                    "teacher_output": teacher_output.strip(),
                    "source": row.get("source", ""),
                }
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if (batch_start // args.batch_size) % 25 == 0:
                processed = min(batch_start + len(batch_rows), len(dataset))
                print(f"Processed {processed}/{len(dataset)}")

    print(f"Saved distill dataset to {args.output_path}")


if __name__ == "__main__":
    main()
