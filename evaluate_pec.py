from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.architecture import PECEngine
from models.data import EntityMasker
from models.eval_utils import (
    DEFAULT_BASE_MODELS,
    aggregate_metrics,
    apply_mask_deterministically,
    build_heldout_eval_samples,
    compute_text_metrics,
    render_qwen_user_prompt,
    slugify_model_name,
    strip_thinking_trace,
    thinking_mode_name,
    write_jsonl,
    write_summary_csv,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_inference_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def ensure_tokenizer_padding(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def get_sampling_params(enable_thinking: bool) -> Dict[str, float]:
    """Return sampling hyper-parameters keyed to the thinking mode.

    Thinking ON:  Temperature=0.6, TopP=0.95, TopK=20, MinP=0
    Thinking OFF: Temperature=0.7, TopP=0.8,  TopK=20, MinP=0
    """
    if enable_thinking:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


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


def move_tokenized_batch(tokenized: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in tokenized.items()}


def top_p_sample(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    filtered_logits = logits.clone()
    scatter_mask = torch.zeros_like(logits, dtype=torch.bool)
    scatter_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    filtered_logits = filtered_logits.masked_fill(scatter_mask, float("-inf"))
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def select_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    temperature = max(temperature, 1e-5)
    scaled_logits = logits / temperature
    if top_p < 1.0:
        return top_p_sample(scaled_logits, top_p=top_p)

    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate_base_response(
    *,
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    min_p: float = 0.0,
    enable_thinking: bool,
) -> str:
    prompt = render_qwen_user_prompt(tokenizer, prompt_text, enable_thinking=enable_thinking)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = move_tokenized_batch(encoded, device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        generate_kwargs["top_k"] = top_k
        if min_p > 0.0:
            generate_kwargs["min_p"] = min_p

    output_ids = model.generate(**encoded, **generate_kwargs)
    prompt_len = encoded["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_pec_response(
    *,
    model: PECEngine,
    profiler_tokenizer,
    composer_tokenizer,
    clean_prompt_text: str,
    visible_prompt_text: str,
    device: torch.device,
    max_profiler_len: int,
    max_composer_len: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    min_p: float = 0.0,
    enable_thinking: bool,
) -> Tuple[str, Dict[str, float]]:
    profiler_inputs = profiler_tokenizer(
        clean_prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_profiler_len,
    )
    profiler_inputs = move_tokenized_batch(profiler_inputs, device)

    composer_prompt_text = render_qwen_user_prompt(
        composer_tokenizer,
        visible_prompt_text,
        enable_thinking=enable_thinking,
    )
    composer_inputs = composer_tokenizer(
        composer_prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_composer_len,
    )
    composer_inputs = move_tokenized_batch(composer_inputs, device)

    prof_outputs = model.profiler(
        input_ids=profiler_inputs["input_ids"],
        attention_mask=profiler_inputs["attention_mask"],
    )
    prof_hidden = prof_outputs.last_hidden_state
    extruded, gate_scores = model.extruder(
        context=prof_hidden,
        attn_mask=profiler_inputs["attention_mask"],
        return_gate_scores=True,
    )
    soft_prompts = model.projector(model.post_extruder_norm(extruded))

    text_embeds = model.composer.get_input_embeddings()(composer_inputs["input_ids"])
    final_inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)

    soft_prompt_mask = torch.ones(
        (soft_prompts.shape[0], soft_prompts.shape[1]),
        device=device,
        dtype=composer_inputs["attention_mask"].dtype,
    )
    final_attention_mask = torch.cat([soft_prompt_mask, composer_inputs["attention_mask"]], dim=1)

    outputs = model.composer(
        inputs_embeds=final_inputs_embeds,
        attention_mask=final_attention_mask,
        use_cache=True,
        return_dict=True,
    )
    next_token_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    generated_tokens: List[torch.Tensor] = []
    current_attention_mask = final_attention_mask

    for _ in range(max_new_tokens):
        next_token = select_next_token(
            next_token_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        generated_tokens.append(next_token.unsqueeze(1))

        if composer_tokenizer.eos_token_id is not None and torch.all(next_token == composer_tokenizer.eos_token_id):
            break

        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones((current_attention_mask.shape[0], 1), device=device, dtype=current_attention_mask.dtype),
            ],
            dim=1,
        )

        outputs = model.composer(
            input_ids=next_token.unsqueeze(1),
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    if generated_tokens:
        generated_ids = torch.cat(generated_tokens, dim=1)
        generated_text = composer_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    else:
        generated_text = ""

    gate_stats = {
        "gate_abs_mean": float(gate_scores.abs().mean().item()),
        "gate_abs_max": float(gate_scores.abs().max().item()),
    }
    return generated_text, gate_stats


def load_base_model(model_name: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_tokenizer_padding(tokenizer)

    kwargs = {"low_cpu_mem_usage": True}
    if device.type != "cpu":
        kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_pec_model(
    *,
    checkpoint_dir: Path,
    profiler_path: Path,
    composer_model_name: str,
    num_query_tokens: int,
    device: torch.device,
):
    profiler_tokenizer = AutoTokenizer.from_pretrained(profiler_path)
    composer_tokenizer = AutoTokenizer.from_pretrained(composer_model_name)
    ensure_tokenizer_padding(composer_tokenizer)

    model = PECEngine(
        profiler_path=str(profiler_path),
        composer_path=composer_model_name,
        num_query_tokens=num_query_tokens,
    )
    state_dict = load_state_dict_from_checkpoint(checkpoint_dir)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys in PEC checkpoint: {unexpected_keys}")

    critical_missing = [
        key for key in missing_keys
        if key.startswith(("extruder", "projector", "post_extruder_norm"))
    ]
    if critical_missing:
        raise RuntimeError(f"Critical PEC weights are missing from checkpoint: {critical_missing}")

    model.to(device)
    model.eval()
    return model, profiler_tokenizer, composer_tokenizer


def evaluate_base_experiment(
    *,
    model_name: str,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    do_sample: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    current_thinking_mode = thinking_mode_name(enable_thinking)
    sampling = get_sampling_params(enable_thinking)
    print(f"\n[Base] Loading {model_name} ({current_thinking_mode})")
    print(f"  Sampling: temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
          f"top_k={sampling['top_k']}, min_p={sampling['min_p']}")
    model, tokenizer = load_base_model(model_name, device, dtype)
    predictions: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        raw_prediction = generate_base_response(
            model=model,
            tokenizer=tokenizer,
            prompt_text=sample["prompt"],
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            enable_thinking=enable_thinking,
        )
        scored_prediction = strip_thinking_trace(raw_prediction)
        metrics = compute_text_metrics(sample["answer"], scored_prediction)
        record = {
            "sample_index": index - 1,
            "source": sample.get("source", ""),
            "scenario": "base_no_hint",
            "thinking_mode": current_thinking_mode,
            "enable_thinking": enable_thinking,
            "model_name": model_name,
            "prompt_mode": "full",
            "soft_prompt_first": False,
            "soft_prompt_tokens": 0,
            "reference": sample["answer"],
            "prediction": scored_prediction,
            "raw_prediction": raw_prediction,
            **metrics,
        }
        predictions.append(record)
        if index % 10 == 0 or index == len(samples):
            print(f"  - processed {index}/{len(samples)}")

    slug = slugify_model_name(model_name)
    write_jsonl(predictions, output_dir / f"base_no_hint__{current_thinking_mode}__{slug}.jsonl")
    summary = {
        "scenario": "base_no_hint",
        "thinking_mode": current_thinking_mode,
        "enable_thinking": enable_thinking,
        "model_name": model_name,
        **aggregate_metrics(predictions),
    }

    del model
    clear_memory()
    return summary


def evaluate_pec_experiment(
    *,
    scenario: str,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    checkpoint_dir: Path,
    profiler_path: Path,
    composer_model_name: str,
    num_query_tokens: int,
    mask_probability: float,
    mask_seed: int,
    max_profiler_len: int,
    max_composer_len: int,
    max_new_tokens: int,
    do_sample: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    current_thinking_mode = thinking_mode_name(enable_thinking)
    sampling = get_sampling_params(enable_thinking)
    print(f"\n[PEC] Loading checkpoint for scenario={scenario} ({current_thinking_mode})")
    print(f"  Sampling: temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
          f"top_k={sampling['top_k']}, min_p={sampling['min_p']}")
    model, profiler_tokenizer, composer_tokenizer = load_pec_model(
        checkpoint_dir=checkpoint_dir,
        profiler_path=profiler_path,
        composer_model_name=composer_model_name,
        num_query_tokens=num_query_tokens,
        device=device,
    )

    masker = EntityMasker(mask_prob=mask_probability) if scenario == "pec_masked_hint" else None
    predictions: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        visible_prompt = sample["prompt"]
        if masker is not None:
            visible_prompt = apply_mask_deterministically(sample["prompt"], masker, seed=mask_seed + index - 1)

        raw_prediction, gate_stats = generate_pec_response(
            model=model,
            profiler_tokenizer=profiler_tokenizer,
            composer_tokenizer=composer_tokenizer,
            clean_prompt_text=sample["prompt"],
            visible_prompt_text=visible_prompt,
            device=device,
            max_profiler_len=max_profiler_len,
            max_composer_len=max_composer_len,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            enable_thinking=enable_thinking,
        )
        scored_prediction = strip_thinking_trace(raw_prediction)
        metrics = compute_text_metrics(sample["answer"], scored_prediction)
        record = {
            "sample_index": index - 1,
            "source": sample.get("source", ""),
            "scenario": scenario,
            "thinking_mode": current_thinking_mode,
            "enable_thinking": enable_thinking,
            "model_name": composer_model_name,
            "prompt_mode": "masked_30" if masker is not None else "full",
            "soft_prompt_first": True,
            "soft_prompt_tokens": num_query_tokens,
            "profiler_prompt": sample["prompt"],
            "composer_visible_prompt": visible_prompt,
            "reference": sample["answer"],
            "prediction": scored_prediction,
            "raw_prediction": raw_prediction,
            **metrics,
            **gate_stats,
        }
        predictions.append(record)
        if index % 10 == 0 or index == len(samples):
            print(f"  - processed {index}/{len(samples)}")

    write_jsonl(predictions, output_dir / f"{scenario}__{current_thinking_mode}__{slugify_model_name(composer_model_name)}.jsonl")
    summary = {
        "scenario": scenario,
        "thinking_mode": current_thinking_mode,
        "enable_thinking": enable_thinking,
        "model_name": composer_model_name,
        **aggregate_metrics(predictions),
    }

    del model
    clear_memory()
    return summary


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    default_output_dir = base_dir / "outputs" / "pec_eval_ablation"

    parser = argparse.ArgumentParser(description="Run held-out PEC ablation evaluation.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--persist-heldout-jsonl", type=Path, default=None)
    parser.add_argument("--blend-split", type=str, default="train")
    parser.add_argument("--epoch-size", type=int, default=200_000)
    replacement_group = parser.add_mutually_exclusive_group()
    replacement_group.add_argument("--with-replacement", dest="with_replacement", action="store_true")
    replacement_group.add_argument("--without-replacement", dest="with_replacement", action="store_false")
    parser.set_defaults(with_replacement=True)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--base-models", nargs="+", default=list(DEFAULT_BASE_MODELS))
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-pec", action="store_true")

    parser.add_argument("--pec-checkpoint-dir", type=Path, default=base_dir / "models" / "PEC")
    parser.add_argument("--profiler-path", type=Path, default=base_dir / "models" / "profiler")
    parser.add_argument("--pec-composer-model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-query-tokens", type=int, default=64)
    parser.add_argument("--mask-probability", type=float, default=0.3)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--max-profiler-len", type=int, default=8192)
    parser.add_argument("--max-composer-len", type=int, default=4096)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling (default)")
    sampling_group.add_argument("--no-sample", dest="do_sample", action="store_false", help="Use greedy decoding")
    parser.set_defaults(do_sample=True)
    parser.add_argument(
        "--thinking-modes",
        nargs="+",
        choices=["off", "on"],
        default=["off", "on"],
        help="Compare Qwen chat-template thinking modes. Default evaluates both off and on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype = get_inference_dtype(device)
    print(f"Device={device}, dtype={dtype}")
    print("Reconstructing held-out eval split used outside PEC training updates...")

    eval_samples = build_heldout_eval_samples(
        base_dir=Path(__file__).parent,
        dataset_jsonl=args.dataset_jsonl,
        split=args.blend_split,
        seed=args.seed,
        epoch_size=args.epoch_size,
        with_replacement=args.with_replacement,
        eval_ratio=args.eval_ratio,
        max_samples=args.max_samples,
        persist_path=args.persist_heldout_jsonl,
    )
    print(f"Loaded held-out eval samples: {len(eval_samples)}")

    thinking_modes = [mode == "on" for mode in args.thinking_modes]

    config_payload = {
        "device": str(device),
        "dtype": str(dtype),
        "num_eval_samples": len(eval_samples),
        "base_models": args.base_models,
        "pec_composer_model": args.pec_composer_model,
        "soft_prompt_tokens": args.num_query_tokens,
        "mask_probability": args.mask_probability,
        "soft_prompt_order": "[compressed soft prompts] + [visible prompt text]",
        "thinking_modes": [thinking_mode_name(mode) for mode in thinking_modes],
        "sampling_params": {
            thinking_mode_name(mode): get_sampling_params(mode)
            for mode in thinking_modes
        },
        "thinking_scoring": "metrics are computed after removing optional <think>...</think> traces; raw_prediction preserves the original generation",
        "seed": args.seed,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summaries: List[Dict[str, Any]] = []

    if not args.skip_base:
        for model_name in args.base_models:
            for enable_thinking in thinking_modes:
                summaries.append(
                    evaluate_base_experiment(
                        model_name=model_name,
                        samples=eval_samples,
                        output_dir=run_dir,
                        device=device,
                        dtype=dtype,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        enable_thinking=enable_thinking,
                    )
                )

    if not args.skip_pec:
        for scenario in ("pec_with_hint", "pec_masked_hint"):
            for enable_thinking in thinking_modes:
                summaries.append(
                    evaluate_pec_experiment(
                        scenario=scenario,
                        samples=eval_samples,
                        output_dir=run_dir,
                        device=device,
                        checkpoint_dir=args.pec_checkpoint_dir,
                        profiler_path=args.profiler_path,
                        composer_model_name=args.pec_composer_model,
                        num_query_tokens=args.num_query_tokens,
                        mask_probability=args.mask_probability,
                        mask_seed=args.mask_seed,
                        max_profiler_len=args.max_profiler_len,
                        max_composer_len=args.max_composer_len,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        enable_thinking=enable_thinking,
                    )
                )

    write_summary_csv(summaries, run_dir / "summary.csv")
    (run_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDone. Summary written to:", run_dir / "summary.csv")


if __name__ == "__main__":
    main()

