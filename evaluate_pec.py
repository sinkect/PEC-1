from __future__ import annotations
import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    import setproctitle
except ModuleNotFoundError:
    setproctitle = None

from tqdm.auto import tqdm

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
    read_jsonl,
    slugify_model_name,
    strip_thinking_trace,
    thinking_mode_name,
    write_jsonl,
    write_summary_csv,
)
from models.re2_benchmark import (
    RE2_DEFAULT_ACTS,
    RE2_DEFAULT_DATASETS,
    build_arc_total_summary,
    build_re2_prompt,
    dataset_supports_act,
    get_dataset_config,
    get_example_id,
    get_reference_answer_text,
    load_re2_benchmark_examples,
    normalize_dataset_name,
    parse_re2_answer,
    score_re2_prediction,
    summarize_re2_records,
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


def prepare_generation_tokenizer(tokenizer) -> None:
    ensure_tokenizer_padding(tokenizer)
    tokenizer.padding_side = "left"


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


def iter_batches(items: Sequence[Any], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def create_progress_bar(*, total: int, description: str):
    if tqdm is None or not sys.stdout.isatty():
        return None
    return tqdm(total=total, desc=description, unit="sample", dynamic_ncols=True, mininterval=1.0)


def format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds < 0:
        return "--:--"

    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_device_memory_summary(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "mem=n/a"

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    properties = torch.cuda.get_device_properties(device_index)
    allocated_gib = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
    reserved_gib = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
    total_gib = properties.total_memory / (1024 ** 3)
    return f"mem={allocated_gib:.1f}/{reserved_gib:.1f}/{total_gib:.1f}GiB"


def update_progress(
    progress_bar,
    *,
    description: str,
    processed: int,
    total: int,
    step: int,
    start_time: float,
    device: torch.device,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    samples_per_sec = processed / elapsed
    remaining = max(total - processed, 0)
    eta_seconds = remaining / samples_per_sec if samples_per_sec > 0 else float("inf")
    memory_summary = get_device_memory_summary(device)

    if progress_bar is not None:
        progress_bar.update(step)
        progress_bar.set_postfix(
            {
                "sps": f"{samples_per_sec:.2f}",
                "eta": format_duration(eta_seconds),
                "gpu": memory_summary.removeprefix("mem="),
            },
            refresh=False,
        )
        return

    percent = (processed / total * 100.0) if total else 100.0
    print(
        f"  - {description} | processed {processed}/{total} ({percent:.1f}%) | "
        f"{samples_per_sec:.2f} sample/s | ETA {format_duration(eta_seconds)} | {memory_summary}",
        flush=True,
    )


def compact_preview_text(text: str, limit: int = 240) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def print_sample_preview(
    *,
    label: str,
    sample_index: int,
    prompt: str,
    reference: str,
    prediction: str,
    raw_prediction: str,
    visible_prompt: str | None = None,
) -> None:
    print(f"\n[Preview] {label} | sample_index={sample_index}", flush=True)
    print(f"  Prompt: {compact_preview_text(prompt)}", flush=True)
    if visible_prompt is not None and visible_prompt != prompt:
        print(f"  Visible prompt: {compact_preview_text(visible_prompt)}", flush=True)
    print(f"  Reference: {compact_preview_text(reference)}", flush=True)
    print(f"  Prediction: {compact_preview_text(prediction)}", flush=True)
    if raw_prediction != prediction:
        print(f"  Raw prediction: {compact_preview_text(raw_prediction)}", flush=True)


def stream_worker_output(*, process: subprocess.Popen, log_handle, prefix: str) -> None:
    if process.stdout is None:
        return

    for line in process.stdout:
        log_handle.write(line)
        log_handle.flush()
        print(f"[{prefix}] {line}", end="", flush=True)


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
    return generate_base_responses(
        model=model,
        tokenizer=tokenizer,
        prompt_texts=[prompt_text],
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        enable_thinking=enable_thinking,
    )[0]


@torch.inference_mode()
def generate_base_responses(
    *,
    model,
    tokenizer,
    prompt_texts: Sequence[str],
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    min_p: float = 0.0,
    enable_thinking: bool,
) -> List[str]:
    prompts = [render_qwen_user_prompt(tokenizer, prompt_text, enable_thinking=enable_thinking) for prompt_text in prompt_texts]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
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
    generated_ids = output_ids[:, encoded["input_ids"].shape[1]:]
    return [text.strip() for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]


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
) -> Tuple[str, Dict[str, Any]]:
    generated_texts, gate_stats = generate_pec_responses(
        model=model,
        profiler_tokenizer=profiler_tokenizer,
        composer_tokenizer=composer_tokenizer,
        clean_prompt_texts=[clean_prompt_text],
        visible_prompt_texts=[visible_prompt_text],
        device=device,
        max_profiler_len=max_profiler_len,
        max_composer_len=max_composer_len,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        enable_thinking=enable_thinking,
    )
    return generated_texts[0], gate_stats[0]


@torch.inference_mode()
def generate_pec_responses(
    *,
    model: PECEngine,
    profiler_tokenizer,
    composer_tokenizer,
    clean_prompt_texts: Sequence[str],
    visible_prompt_texts: Sequence[str],
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
) -> Tuple[List[str], List[Dict[str, Any]]]:
    def summarize_tensor(tensor: torch.Tensor, *, prefix: str) -> Dict[str, Any]:
        tensor_float = tensor.detach().float()
        return {
            f"{prefix}_norm": float(torch.linalg.vector_norm(tensor_float).item()),
            f"{prefix}_mean": float(tensor_float.mean().item()),
            f"{prefix}_std": float(tensor_float.std(unbiased=False).item()),
        }

    def summarize_soft_prompt_artifacts(
        extruder_latents: torch.Tensor,
        soft_prompts: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        p_slice = soft_prompts[:, :5, :8].detach().float().cpu().tolist()
        return [
            {
                **summarize_tensor(z_sample, prefix="latent_z"),
                **summarize_tensor(p_sample, prefix="soft_prompt_p"),
                "soft_prompt_p_slice": p_head,
            }
            for z_sample, p_sample, p_head in zip(extruder_latents, soft_prompts, p_slice)
        ]

    def summarize_gated_attention(
        gate_scores: torch.Tensor,
        gate_logits: torch.Tensor | None,
    ) -> List[Dict[str, Any]]:
        if gate_scores.ndim != 4:
            raise ValueError(f"Expected gate_scores to have shape [B, L, N_q, D], got {tuple(gate_scores.shape)}")
        if gate_logits is not None and gate_logits.shape != gate_scores.shape:
            raise ValueError(
                "gate_logits must match gate_scores shape; "
                f"got {tuple(gate_logits.shape)} vs {tuple(gate_scores.shape)}"
            )

        if gate_scores.shape[1] == 0:
            zero = gate_scores.new_zeros((gate_scores.shape[0],))
            return [
                {
                    "gate_mean": 0.0,
                    "gate_max": 0.0,
                    "gate_abs_mean": 0.0,
                    "gate_abs_max": 0.0,
                    "gate_logit_mean": 0.0,
                    "gate_logit_abs_max": 0.0,
                    "gate_layer_query_mean": [],
                    "gate_logit_layer_query_mean": [],
                }
                for _ in zero
            ]

        gate_mean = gate_scores.mean(dim=(1, 2, 3))
        gate_max = gate_scores.amax(dim=(1, 2, 3))
        gate_layer_query_mean = gate_scores.mean(dim=-1)

        if gate_logits is None:
            gate_logit_mean = gate_scores.new_zeros((gate_scores.shape[0],))
            gate_logit_abs_max = gate_scores.new_zeros((gate_scores.shape[0],))
            gate_logit_layer_query_mean = gate_scores.new_zeros(gate_scores.shape[:3])
        else:
            gate_logit_mean = gate_logits.mean(dim=(1, 2, 3))
            gate_logit_abs_max = gate_logits.abs().amax(dim=(1, 2, 3))
            gate_logit_layer_query_mean = gate_logits.mean(dim=-1)

        return [
            {
                "gate_mean": float(mean.item()),
                "gate_max": float(maximum.item()),
                "gate_abs_mean": float(mean.item()),
                "gate_abs_max": float(maximum.item()),
                "gate_logit_mean": float(logit_mean.item()),
                "gate_logit_abs_max": float(logit_abs_max.item()),
                "gate_layer_query_mean": layer_query_mean.detach().cpu().tolist(),
                "gate_logit_layer_query_mean": logit_layer_query_mean.detach().cpu().tolist(),
            }
            for mean, maximum, logit_mean, logit_abs_max, layer_query_mean, logit_layer_query_mean in zip(
                gate_mean,
                gate_max,
                gate_logit_mean,
                gate_logit_abs_max,
                gate_layer_query_mean,
                gate_logit_layer_query_mean,
            )
        ]

    profiler_inputs = profiler_tokenizer(
        list(clean_prompt_texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_profiler_len,
    )
    profiler_inputs = move_tokenized_batch(profiler_inputs, device)

    composer_prompt_texts = [
        render_qwen_user_prompt(
            composer_tokenizer,
            visible_prompt_text,
            enable_thinking=enable_thinking,
        )
        for visible_prompt_text in visible_prompt_texts
    ]
    composer_inputs = composer_tokenizer(
        composer_prompt_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_composer_len,
    )
    composer_inputs = move_tokenized_batch(composer_inputs, device)

    artifacts = model.build_soft_prompt_artifacts(
        profiler_input_ids=profiler_inputs["input_ids"],
        profiler_attention_mask=profiler_inputs["attention_mask"],
        return_gate_scores=True,
        return_gate_logits=True,
    )
    soft_prompts = artifacts["soft_prompts"]
    extruder_latents = artifacts["extruder_latents"]
    gate_scores = artifacts["gate_scores"]
    gate_logits = artifacts["gate_logits"]

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
    finished = torch.zeros(composer_inputs["input_ids"].shape[0], dtype=torch.bool, device=device)
    eos_token_id = composer_tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        next_token = select_next_token(
            next_token_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        if eos_token_id is not None:
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
        generated_tokens.append(next_token.unsqueeze(1))

        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if torch.all(finished):
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
        generated_texts = [text.strip() for text in composer_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
    else:
        generated_texts = ["" for _ in clean_prompt_texts]

    gate_stats = summarize_gated_attention(gate_scores, gate_logits)
    soft_prompt_stats = summarize_soft_prompt_artifacts(extruder_latents, soft_prompts)
    gate_stats = [
        {
            **per_gate_stats,
            **per_soft_prompt_stats,
        }
        for per_gate_stats, per_soft_prompt_stats in zip(gate_stats, soft_prompt_stats)
    ]
    return generated_texts, gate_stats


def load_base_model(model_name: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prepare_generation_tokenizer(tokenizer)

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
    profiler_path: str | Path,
    composer_model_name: str,
    num_query_tokens: int,
    device: torch.device,
):
    profiler_tokenizer = AutoTokenizer.from_pretrained(profiler_path)
    ensure_tokenizer_padding(profiler_tokenizer)
    composer_tokenizer = AutoTokenizer.from_pretrained(composer_model_name)
    prepare_generation_tokenizer(composer_tokenizer)

    model = PECEngine(
        profiler_path=str(profiler_path),
        composer_path=composer_model_name,
        num_query_tokens=num_query_tokens,
    )
    state_dict = load_state_dict_from_checkpoint(checkpoint_dir)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys in PEC checkpoint: {unexpected_keys}")

    dynamic_query_prefixes = ("extruder.delta_mlp", "extruder.gate_mlp")
    critical_missing = [
        key for key in missing_keys
        if key.startswith(("extruder", "projector", "post_extruder_norm"))
        and not key.startswith(dynamic_query_prefixes)
    ]
    if critical_missing:
        raise RuntimeError(f"Critical PEC weights are missing from checkpoint: {critical_missing}")

    model.to(dtype=torch.bfloat16,device=device)
    model.eval()
    return model, profiler_tokenizer, composer_tokenizer


def build_re2_job_name(job: Dict[str, Any]) -> str:
    return (
        f"{job['kind']}__{job['dataset_name']}__{job['act']}__read{job['read_times']}__"
        f"{slugify_model_name(job['model_name'])}"
    )


def describe_decoding_mode(*, do_sample: bool, enable_thinking: bool) -> str:
    mode = "sampling" if do_sample else "greedy"
    thinking = "thinking_on" if enable_thinking else "thinking_off"
    return f"{mode}/{thinking}"


def build_re2_prediction_record(
    *,
    dataset_name: str,
    act: str,
    read_times: int,
    model_name: str,
    kind: str,
    sample_index: int,
    example: Dict[str, Any],
    input_prompt: str,
    raw_prediction: str,
    gate_stats: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cleaned_prediction = strip_thinking_trace(raw_prediction)
    strict_parsed_prediction = parse_re2_answer(
        cleaned_prediction,
        dataset_name,
        example,
        act=act,
        mode="strict",
    )
    relaxed_parsed_prediction = parse_re2_answer(
        cleaned_prediction,
        dataset_name,
        example,
        act=act,
        mode="relaxed",
    )
    strict_score = score_re2_prediction(dataset_name, strict_parsed_prediction, example)
    relaxed_score = score_re2_prediction(dataset_name, relaxed_parsed_prediction, example)
    config = get_dataset_config(dataset_name)
    expects_box = act != "pal"
    record = {
        "sample_index": sample_index,
        "example_id": get_example_id(example, sample_index),
        "dataset_name": dataset_name,
        "dataset_label": config.pretty_name,
        "category": config.category,
        "kind": kind,
        "model_name": model_name,
        "act": act,
        "read_times": read_times,
        "use_re2": read_times > 1,
        "input_prompt": input_prompt,
        "reference": get_reference_answer_text(example, dataset_name),
        "prediction_text": cleaned_prediction,
        "strict_parsed_prediction": strict_parsed_prediction,
        "relaxed_parsed_prediction": relaxed_parsed_prediction,
        "parsed_prediction": relaxed_parsed_prediction,
        "raw_prediction": raw_prediction,
        "strict_score": int(strict_score),
        "relaxed_score": int(relaxed_score),
        "score": int(relaxed_score),
        "no_boxed": int(expects_box and bool(raw_prediction) and "boxed" not in raw_prediction.lower()),
    }
    if gate_stats is not None:
        record.update(gate_stats)
    return record


def evaluate_base_re2_benchmark(
    *,
    dataset_name: str,
    model_name: str,
    act: str,
    read_times: int,
    examples: Sequence[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    preview_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    print(
        f"\n[RE2/Base] dataset={dataset_name} act={act} read_times={read_times} model={model_name}",
        flush=True,
    )
    print(f"  Decoding: {describe_decoding_mode(do_sample=do_sample, enable_thinking=enable_thinking)}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    model, tokenizer = load_base_model(model_name, device, dtype)
    predictions: List[Dict[str, Any]] = []
    preview_printed = 0
    progress_description = build_re2_job_name(
        {
            "kind": "base",
            "dataset_name": dataset_name,
            "act": act,
            "read_times": read_times,
            "model_name": model_name,
        }
    )
    progress_bar = create_progress_bar(total=len(examples), description=progress_description)
    progress_start_time = time.perf_counter()

    try:
        for batch_start, batch_examples in iter_batches(examples, batch_size):
            input_prompts = [
                build_re2_prompt(example, dataset_name, act=act, read_times=read_times)
                for example in batch_examples
            ]
            raw_predictions = generate_base_responses(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=input_prompts,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=0.0,
                top_p=1.0,
                top_k=20,
                min_p=0.0,
                enable_thinking=enable_thinking,
            )

            for sample_index, example, input_prompt, raw_prediction in zip(
                range(batch_start, batch_start + len(batch_examples)),
                batch_examples,
                input_prompts,
                raw_predictions,
            ):
                record = build_re2_prediction_record(
                    dataset_name=dataset_name,
                    act=act,
                    read_times=read_times,
                    model_name=model_name,
                    kind="base",
                    sample_index=sample_index,
                    example=example,
                    input_prompt=input_prompt,
                    raw_prediction=raw_prediction,
                )
                predictions.append(record)

                if preview_printed < preview_samples:
                    print_sample_preview(
                        label=progress_description,
                        sample_index=sample_index,
                        prompt=input_prompt,
                        reference=record["reference"],
                        prediction=str(record["parsed_prediction"]),
                        raw_prediction=raw_prediction,
                    )
                    preview_printed += 1

            processed = batch_start + len(batch_examples)
            update_progress(
                progress_bar,
                description=progress_description,
                processed=processed,
                total=len(examples),
                step=len(batch_examples),
                start_time=progress_start_time,
                device=device,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    output_path = output_dir / f"{progress_description}.jsonl"
    write_jsonl(predictions, output_path)
    summary = {
        "dataset_name": dataset_name,
        "dataset_label": get_dataset_config(dataset_name).pretty_name,
        "category": get_dataset_config(dataset_name).category,
        "kind": "base",
        "model_name": model_name,
        "act": act,
        "read_times": read_times,
        "output_file": output_path.name,
        **summarize_re2_records(predictions),
    }

    del model
    clear_memory()
    return summary


def evaluate_pec_re2_benchmark(
    *,
    dataset_name: str,
    act: str,
    read_times: int,
    examples: Sequence[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    checkpoint_dir: Path,
    profiler_path: str | Path,
    composer_model_name: str,
    num_query_tokens: int,
    max_profiler_len: int,
    max_composer_len: int,
    batch_size: int,
    preview_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    print(
        f"\n[RE2/PEC] dataset={dataset_name} act={act} read_times={read_times} model={composer_model_name}",
        flush=True,
    )
    print(f"  Decoding: {describe_decoding_mode(do_sample=do_sample, enable_thinking=enable_thinking)}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    model, profiler_tokenizer, composer_tokenizer = load_pec_model(
        checkpoint_dir=checkpoint_dir,
        profiler_path=profiler_path,
        composer_model_name=composer_model_name,
        num_query_tokens=num_query_tokens,
        device=device,
    )
    predictions: List[Dict[str, Any]] = []
    preview_printed = 0
    progress_description = build_re2_job_name(
        {
            "kind": "pec",
            "dataset_name": dataset_name,
            "act": act,
            "read_times": read_times,
            "model_name": composer_model_name,
        }
    )
    progress_bar = create_progress_bar(total=len(examples), description=progress_description)
    progress_start_time = time.perf_counter()

    try:
        for batch_start, batch_examples in iter_batches(examples, batch_size):
            input_prompts = [
                build_re2_prompt(example, dataset_name, act=act, read_times=read_times)
                for example in batch_examples
            ]
            raw_predictions, gate_stats_batch = generate_pec_responses(
                model=model,
                profiler_tokenizer=profiler_tokenizer,
                composer_tokenizer=composer_tokenizer,
                clean_prompt_texts=input_prompts,
                visible_prompt_texts=input_prompts,
                device=device,
                max_profiler_len=max_profiler_len,
                max_composer_len=max_composer_len,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=0.0,
                top_p=1.0,
                top_k=20,
                min_p=0.0,
                enable_thinking=enable_thinking,
            )

            for sample_index, example, input_prompt, raw_prediction, gate_stats in zip(
                range(batch_start, batch_start + len(batch_examples)),
                batch_examples,
                input_prompts,
                raw_predictions,
                gate_stats_batch,
            ):
                record = build_re2_prediction_record(
                    dataset_name=dataset_name,
                    act=act,
                    read_times=read_times,
                    model_name=composer_model_name,
                    kind="pec",
                    sample_index=sample_index,
                    example=example,
                    input_prompt=input_prompt,
                    raw_prediction=raw_prediction,
                    gate_stats=gate_stats,
                )
                predictions.append(record)

                if preview_printed < preview_samples:
                    print_sample_preview(
                        label=progress_description,
                        sample_index=sample_index,
                        prompt=input_prompt,
                        reference=record["reference"],
                        prediction=str(record["parsed_prediction"]),
                        raw_prediction=raw_prediction,
                    )
                    preview_printed += 1

            processed = batch_start + len(batch_examples)
            update_progress(
                progress_bar,
                description=progress_description,
                processed=processed,
                total=len(examples),
                step=len(batch_examples),
                start_time=progress_start_time,
                device=device,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    output_path = output_dir / f"{progress_description}.jsonl"
    write_jsonl(predictions, output_path)
    summary = {
        "dataset_name": dataset_name,
        "dataset_label": get_dataset_config(dataset_name).pretty_name,
        "category": get_dataset_config(dataset_name).category,
        "kind": "pec",
        "model_name": composer_model_name,
        "act": act,
        "read_times": read_times,
        "output_file": output_path.name,
        **summarize_re2_records(predictions),
    }

    del model
    clear_memory()
    return summary


def load_re2_benchmark_collections(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    datasets = [normalize_dataset_name(name) for name in args.benchmark_datasets]
    collections: Dict[str, List[Dict[str, Any]]] = {}
    for dataset_name in datasets:
        config = get_dataset_config(dataset_name)
        examples = load_re2_benchmark_examples(
            dataset_name,
            args.benchmark_data_dir,
            max_samples=args.max_samples,
            download_if_missing=args.download_benchmark_data,
        )
        print(
            f"Loaded benchmark dataset {config.pretty_name} ({dataset_name}) with {len(examples)} samples",
            flush=True,
        )
        collections[dataset_name] = examples
    return collections


def build_re2_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    dataset_names = [normalize_dataset_name(name) for name in args.benchmark_datasets]
    acts = [act.lower() for act in args.acts]

    for dataset_name in dataset_names:
        for act in acts:
            if not dataset_supports_act(dataset_name, act):
                print(f"Skipping unsupported act={act} for dataset={dataset_name}", flush=True)
                continue
            for read_times in args.read_times:
                if not args.skip_base:
                    for model_name in args.base_models:
                        jobs.append(
                            {
                                "kind": "base",
                                "dataset_name": dataset_name,
                                "model_name": model_name,
                                "act": act,
                                "read_times": read_times,
                            }
                        )
                if not args.skip_pec:
                    jobs.append(
                        {
                            "kind": "pec",
                            "dataset_name": dataset_name,
                            "model_name": args.pec_composer_model,
                            "act": act,
                            "read_times": read_times,
                        }
                    )
    return jobs


def build_re2_worker_command(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    eval_samples_path: Path,
    job: Dict[str, Any],
    summary_path: Path,
) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--protocol",
        "re2_paper",
        "--run-dir",
        str(run_dir),
        "--eval-samples-jsonl",
        str(eval_samples_path),
        "--batch-size",
        str(args.batch_size),
        "--preview-samples",
        str(args.preview_samples),
        "--benchmark-data-dir",
        str(args.benchmark_data_dir),
        "--pec-checkpoint-dir",
        str(args.pec_checkpoint_dir),
        "--profiler-path",
        str(args.profiler_path),
        "--pec-composer-model",
        args.pec_composer_model,
        "--num-query-tokens",
        str(args.num_query_tokens),
        "--max-profiler-len",
        str(args.max_profiler_len),
        "--max-composer-len",
        str(args.max_composer_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--worker-kind",
        job["kind"],
        "--worker-protocol",
        "re2_paper",
        "--worker-dataset-name",
        job["dataset_name"],
        "--worker-act",
        job["act"],
        "--worker-read-times",
        str(job["read_times"]),
        "--worker-summary-path",
        str(summary_path),
    ]

    command.append("--do-sample" if args.do_sample else "--no-sample")
    command.append("--enable-thinking" if args.enable_thinking else "--disable-thinking")

    if job["kind"] == "base":
        command.extend(["--worker-model-name", job["model_name"]])

    return command


def run_parallel_re2_jobs(
    *,
    jobs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    benchmark_collections: Dict[str, List[Dict[str, Any]]],
    gpu_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    dataset_payload_dir = run_dir / "benchmark_payloads"
    logs_dir = run_dir / "logs"
    worker_summary_dir = run_dir / "worker_summaries"
    dataset_payload_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_summary_dir.mkdir(parents=True, exist_ok=True)

    dataset_payload_paths: Dict[str, Path] = {}
    for dataset_name, samples in benchmark_collections.items():
        payload_path = dataset_payload_dir / f"{dataset_name}.jsonl"
        write_jsonl(samples, payload_path)
        dataset_payload_paths[dataset_name] = payload_path

    completed_summaries: List[Tuple[int, Dict[str, Any]]] = []
    pending_jobs: List[Tuple[int, Dict[str, Any]]] = []
    for job_index, job in enumerate(jobs):
        summary_path = worker_summary_dir / f"{build_re2_job_name(job)}.json"
        if summary_path.exists():
            completed_summaries.append((job_index, json.loads(summary_path.read_text(encoding="utf-8"))))
        else:
            pending_jobs.append((job_index, job))

    running_jobs: Dict[int, Dict[str, Any]] = {}

    print(f"Running {len(jobs)} RE2 benchmark jobs across GPUs {list(gpu_ids)}", flush=True)
    print(f"Worker logs: {logs_dir}", flush=True)
    if completed_summaries:
        print(f"Resuming existing run: {len(completed_summaries)} jobs already completed, {len(pending_jobs)} remaining", flush=True)

    try:
        while pending_jobs or running_jobs:
            available_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in running_jobs]
            while available_gpu_ids and pending_jobs:
                gpu_id = available_gpu_ids.pop(0)
                job_index, job = pending_jobs.pop(0)
                job_name = build_re2_job_name(job)
                log_path = logs_dir / f"{job_name}.log"
                summary_path = worker_summary_dir / f"{job_name}.json"
                worker_command = build_re2_worker_command(
                    args=args,
                    run_dir=run_dir,
                    eval_samples_path=dataset_payload_paths[job["dataset_name"]],
                    job=job,
                    summary_path=summary_path,
                )
                log_handle = log_path.open("w", encoding="utf-8", buffering=1)
                worker_env = os.environ.copy()
                worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                worker_env["PYTHONUNBUFFERED"] = "1"

                print(
                    f"  - launching [{job_index + 1}/{len(jobs)}] {job_name} on GPU {gpu_id} -> {log_path}",
                    flush=True,
                )
                process = subprocess.Popen(
                    worker_command,
                    cwd=Path(__file__).parent,
                    env=worker_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                stream_thread = threading.Thread(
                    target=stream_worker_output,
                    kwargs={
                        "process": process,
                        "log_handle": log_handle,
                        "prefix": f"gpu{gpu_id}:{job_name}",
                    },
                    daemon=True,
                )
                stream_thread.start()
                running_jobs[gpu_id] = {
                    "job_index": job_index,
                    "job_name": job_name,
                    "log_path": log_path,
                    "log_handle": log_handle,
                    "process": process,
                    "summary_path": summary_path,
                    "stream_thread": stream_thread,
                }

            finished_gpu_ids: List[int] = []
            for gpu_id, job_state in running_jobs.items():
                process = job_state["process"]
                return_code = process.poll()
                if return_code is None:
                    continue

                if process.stdout is not None:
                    process.stdout.close()
                job_state["stream_thread"].join(timeout=5)
                job_state["log_handle"].close()
                if return_code != 0:
                    raise RuntimeError(
                        f"RE2 eval worker failed for {job_state['job_name']} on GPU {gpu_id}. "
                        f"See {job_state['log_path']}"
                    )

                summary = json.loads(job_state["summary_path"].read_text(encoding="utf-8"))
                completed_summaries.append((job_state["job_index"], summary))
                finished_gpu_ids.append(gpu_id)
                print(f"  - finished {job_state['job_name']} on GPU {gpu_id}", flush=True)

            for gpu_id in finished_gpu_ids:
                del running_jobs[gpu_id]

            if running_jobs:
                print(
                    f"  - progress: {len(completed_summaries)}/{len(jobs)} jobs finished, "
                    f"{len(running_jobs)} running, {len(pending_jobs)} pending",
                    flush=True,
                )
                time.sleep(5)
    except Exception:
        for job_state in running_jobs.values():
            process = job_state["process"]
            if process.poll() is None:
                process.terminate()
            if process.stdout is not None:
                process.stdout.close()
            job_state["stream_thread"].join(timeout=5)
            job_state["log_handle"].close()
        raise

    completed_summaries.sort(key=lambda item: item[0])
    return [summary for _, summary in completed_summaries]


def run_re2_paper_protocol(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.preview_samples < 0:
        raise ValueError("--preview-samples must be at least 0")
    if not args.acts:
        raise ValueError("--acts must not be empty")
    if not args.read_times:
        raise ValueError("--read-times must not be empty")

    run_dir = args.run_dir or (args.output_dir / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype = get_inference_dtype(device)
    print(f"Device={device}, dtype={dtype}", flush=True)
    benchmark_collections = load_re2_benchmark_collections(args)
    jobs = build_re2_jobs(args)
    gpu_ids = resolve_gpu_ids(args)
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpu-ids must not contain duplicates")
    print(f"Prepared {len(jobs)} benchmark jobs", flush=True)
    print(f"Decoding mode: {describe_decoding_mode(do_sample=args.do_sample, enable_thinking=args.enable_thinking)}", flush=True)

    config_payload = {
        "protocol": "re2_paper",
        "paper": "https://aclanthology.org/2024.emnlp-main.871",
        "benchmark_datasets": list(benchmark_collections.keys()),
        "dataset_sizes": {name: len(samples) for name, samples in benchmark_collections.items()},
        "acts": [act.lower() for act in args.acts],
        "read_times": list(args.read_times),
        "paper_default_read_times": 2,
        "paper_default_acts": ["vanilla", "cot"],
        "benchmark_data_dir": str(args.benchmark_data_dir),
        "download_benchmark_data": bool(args.download_benchmark_data),
        "base_models": list(args.base_models),
        "skip_base": bool(args.skip_base),
        "skip_pec": bool(args.skip_pec),
        "gpu_ids": gpu_ids,
        "num_eval_jobs": len(jobs),
        "pec_composer_model": args.pec_composer_model,
        "num_query_tokens": args.num_query_tokens,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "enable_thinking": bool(args.enable_thinking),
        "decoding_note": "Paper Table 1/2 uses zero-shot greedy decoding with temperature=0. This protocol defaults to greedy decoding and boxed-answer extraction.",
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    worker_summary_dir = run_dir / "worker_summaries"
    worker_summary_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda" and len(gpu_ids) > 1 and len(jobs) > 1:
        summaries = run_parallel_re2_jobs(
            jobs=jobs,
            args=args,
            run_dir=run_dir,
            benchmark_collections=benchmark_collections,
            gpu_ids=gpu_ids,
        )
    else:
        summaries = []
        for job_index, job in enumerate(jobs, start=1):
            summary_path = worker_summary_dir / f"{build_re2_job_name(job)}.json"
            if summary_path.exists():
                print(f"[Job {job_index}/{len(jobs)}] skipping completed {build_re2_job_name(job)}", flush=True)
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            print(
                f"[Job {job_index}/{len(jobs)}] {build_re2_job_name(job)}",
                flush=True,
            )
            examples = benchmark_collections[job["dataset_name"]]
            if job["kind"] == "base":
                summary = evaluate_base_re2_benchmark(
                    dataset_name=job["dataset_name"],
                    model_name=job["model_name"],
                    act=job["act"],
                    read_times=job["read_times"],
                    examples=examples,
                    output_dir=run_dir,
                    device=device,
                    dtype=dtype,
                    batch_size=args.batch_size,
                    preview_samples=args.preview_samples,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    enable_thinking=args.enable_thinking,
                )
            else:
                summary = evaluate_pec_re2_benchmark(
                    dataset_name=job["dataset_name"],
                    act=job["act"],
                    read_times=job["read_times"],
                    examples=examples,
                    output_dir=run_dir,
                    device=device,
                    checkpoint_dir=args.pec_checkpoint_dir,
                    profiler_path=args.profiler_path,
                    composer_model_name=args.pec_composer_model,
                    num_query_tokens=args.num_query_tokens,
                    max_profiler_len=args.max_profiler_len,
                    max_composer_len=args.max_composer_len,
                    batch_size=args.batch_size,
                    preview_samples=args.preview_samples,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    enable_thinking=args.enable_thinking,
                )
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            summaries.append(summary)

    summaries.extend(build_arc_total_summary(summaries))
    write_summary_csv(summaries, run_dir / "summary.csv")
    (run_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDone. Summary written to:", run_dir / "summary.csv", flush=True)


def evaluate_base_experiment(
    *,
    scenario: str,
    model_name: str,
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    prompt_repeat_count: int,
    preview_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    current_thinking_mode = thinking_mode_name(enable_thinking)
    sampling = get_sampling_params(enable_thinking)
    print(f"\n[Base] Loading {model_name} ({current_thinking_mode})", flush=True)
    print(f"  Sampling: temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
          f"top_k={sampling['top_k']}, min_p={sampling['min_p']}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    print(f"  Prompt repeat count: {prompt_repeat_count}", flush=True)
    print(f"  Preview samples: {preview_samples}", flush=True)
    model, tokenizer = load_base_model(model_name, device, dtype)
    predictions: List[Dict[str, Any]] = []
    preview_printed = 0
    progress_description = f"{scenario}:{slugify_model_name(model_name)}:{current_thinking_mode}"
    progress_bar = create_progress_bar(
        total=len(samples),
        description=progress_description,
    )
    progress_start_time = time.perf_counter()

    try:
        for batch_start, batch_samples in iter_batches(samples, batch_size):
            input_prompts = ["\n\n".join([sample["prompt"]] * prompt_repeat_count) for sample in batch_samples]
            raw_predictions = generate_base_responses(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=input_prompts,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                min_p=sampling["min_p"],
                enable_thinking=enable_thinking,
            )

            for sample_index, sample, input_prompt, raw_prediction in zip(
                range(batch_start, batch_start + len(batch_samples)),
                batch_samples,
                input_prompts,
                raw_predictions,
            ):
                scored_prediction = strip_thinking_trace(raw_prediction)
                metrics = compute_text_metrics(sample["answer"], scored_prediction)
                record = {
                    "sample_index": sample_index,
                    "source": sample.get("source", ""),
                    "scenario": scenario,
                    "thinking_mode": current_thinking_mode,
                    "enable_thinking": enable_thinking,
                    "model_name": model_name,
                    "prompt_mode": "full" if prompt_repeat_count == 1 else f"full_x{prompt_repeat_count}",
                    "prompt_repeat_count": prompt_repeat_count,
                    "soft_prompt_first": False,
                    "soft_prompt_tokens": 0,
                    "input_prompt": input_prompt,
                    "reference": sample["answer"],
                    "prediction": scored_prediction,
                    "raw_prediction": raw_prediction,
                    **metrics,
                }
                predictions.append(record)

                if preview_printed < preview_samples:
                    print_sample_preview(
                        label=progress_description,
                        sample_index=sample_index,
                        prompt=sample["prompt"],
                        visible_prompt=input_prompt,
                        reference=sample["answer"],
                        prediction=scored_prediction,
                        raw_prediction=raw_prediction,
                    )
                    preview_printed += 1

            processed = batch_start + len(batch_samples)
            update_progress(
                progress_bar,
                description=progress_description,
                processed=processed,
                total=len(samples),
                step=len(batch_samples),
                start_time=progress_start_time,
                device=device,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    slug = slugify_model_name(model_name)
    write_jsonl(predictions, output_dir / f"{scenario}__{current_thinking_mode}__{slug}.jsonl")
    summary = {
        "scenario": scenario,
        "thinking_mode": current_thinking_mode,
        "enable_thinking": enable_thinking,
        "model_name": model_name,
        "prompt_repeat_count": prompt_repeat_count,
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
    profiler_path: str | Path,
    composer_model_name: str,
    num_query_tokens: int,
    mask_probability: float,
    mask_seed: int,
    max_profiler_len: int,
    max_composer_len: int,
    batch_size: int,
    preview_samples: int,
    max_new_tokens: int,
    do_sample: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    current_thinking_mode = thinking_mode_name(enable_thinking)
    sampling = get_sampling_params(enable_thinking)
    print(f"\n[PEC] Loading checkpoint for scenario={scenario} ({current_thinking_mode})", flush=True)
    print(f"  Sampling: temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
          f"top_k={sampling['top_k']}, min_p={sampling['min_p']}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    print(f"  Preview samples: {preview_samples}", flush=True)
    model, profiler_tokenizer, composer_tokenizer = load_pec_model(
        checkpoint_dir=checkpoint_dir,
        profiler_path=profiler_path,
        composer_model_name=composer_model_name,
        num_query_tokens=num_query_tokens,
        device=device,
    )

    masker = EntityMasker(mask_prob=mask_probability) if scenario == "pec_masked_hint" else None
    predictions: List[Dict[str, Any]] = []
    preview_printed = 0
    progress_description = f"pec:{scenario}:{current_thinking_mode}"
    progress_bar = create_progress_bar(
        total=len(samples),
        description=progress_description,
    )
    progress_start_time = time.perf_counter()

    try:
        for batch_start, batch_samples in iter_batches(samples, batch_size):
            visible_prompts: List[str] = []
            for sample_index, sample in enumerate(batch_samples, start=batch_start + 1):
                visible_prompt = sample["prompt"]
                if masker is not None:
                    visible_prompt = apply_mask_deterministically(sample["prompt"], masker, seed=mask_seed + sample_index - 1)
                visible_prompts.append(visible_prompt)

            raw_predictions, gate_stats_batch = generate_pec_responses(
                model=model,
                profiler_tokenizer=profiler_tokenizer,
                composer_tokenizer=composer_tokenizer,
                clean_prompt_texts=[sample["prompt"] for sample in batch_samples],
                visible_prompt_texts=visible_prompts,
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

            for sample_index, sample, visible_prompt, raw_prediction, gate_stats in zip(
                range(batch_start, batch_start + len(batch_samples)),
                batch_samples,
                visible_prompts,
                raw_predictions,
                gate_stats_batch,
            ):
                scored_prediction = strip_thinking_trace(raw_prediction)
                metrics = compute_text_metrics(sample["answer"], scored_prediction)
                record = {
                    "sample_index": sample_index,
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

                if preview_printed < preview_samples:
                    print_sample_preview(
                        label=progress_description,
                        sample_index=sample_index,
                        prompt=sample["prompt"],
                        visible_prompt=visible_prompt,
                        reference=sample["answer"],
                        prediction=scored_prediction,
                        raw_prediction=raw_prediction,
                    )
                    preview_printed += 1

            processed = batch_start + len(batch_samples)
            update_progress(
                progress_bar,
                description=progress_description,
                processed=processed,
                total=len(samples),
                step=len(batch_samples),
                start_time=progress_start_time,
                device=device,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

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


def build_eval_jobs(
    *,
    base_models: Sequence[str],
    base_scenarios: Sequence[str],
    composer_model_name: str,
    thinking_modes: Sequence[bool],
    skip_base: bool,
    skip_pec: bool,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    if not skip_base:
        prompt_repeat_counts = {
            "base_no_hint": 1,
            "base_prompt_twice": 2,
        }
        for model_name in base_models:
            for enable_thinking in thinking_modes:
                for scenario in base_scenarios:
                    jobs.append(
                        {
                            "kind": "base",
                            "scenario": scenario,
                            "model_name": model_name,
                            "enable_thinking": enable_thinking,
                            "prompt_repeat_count": prompt_repeat_counts[scenario],
                        }
                    )

    if not skip_pec:
        for scenario in ("pec_with_hint", "pec_masked_hint"):
            for enable_thinking in thinking_modes:
                jobs.append(
                    {
                        "kind": "pec",
                        "scenario": scenario,
                        "model_name": composer_model_name,
                        "enable_thinking": enable_thinking,
                    }
                )

    return jobs


def eval_job_name(job: Dict[str, Any]) -> str:
    scenario = job.get("scenario", "base_no_hint")
    return f"{scenario}__{thinking_mode_name(job['enable_thinking'])}__{slugify_model_name(job['model_name'])}"


def run_eval_job(
    *,
    job: Dict[str, Any],
    args: argparse.Namespace,
    eval_samples: Sequence[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    if job["kind"] == "base":
        return evaluate_base_experiment(
            scenario=job["scenario"],
            model_name=job["model_name"],
            samples=eval_samples,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            prompt_repeat_count=job["prompt_repeat_count"],
            preview_samples=args.preview_samples,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            enable_thinking=job["enable_thinking"],
        )

    return evaluate_pec_experiment(
        scenario=job["scenario"],
        samples=eval_samples,
        output_dir=output_dir,
        device=device,
        checkpoint_dir=args.pec_checkpoint_dir,
        profiler_path=args.profiler_path,
        composer_model_name=job["model_name"],
        num_query_tokens=args.num_query_tokens,
        mask_probability=args.mask_probability,
        mask_seed=args.mask_seed,
        max_profiler_len=args.max_profiler_len,
        max_composer_len=args.max_composer_len,
        batch_size=args.batch_size,
        preview_samples=args.preview_samples,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        enable_thinking=job["enable_thinking"],
    )


def resolve_gpu_ids(args: argparse.Namespace) -> List[int]:
    if args.gpu_ids is not None:
        return list(args.gpu_ids)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        return [int(device_id.strip()) for device_id in visible_devices.split(",") if device_id.strip()]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def load_eval_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.eval_samples_jsonl is not None:
        print(f"Loading precomputed eval samples from {args.eval_samples_jsonl}", flush=True)
        return read_jsonl(args.eval_samples_jsonl)

    print("Reconstructing held-out eval split used outside PEC training updates...", flush=True)
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
    return eval_samples


def build_worker_command(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    eval_samples_path: Path,
    job: Dict[str, Any],
    summary_path: Path,
) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--run-dir",
        str(run_dir),
        "--eval-samples-jsonl",
        str(eval_samples_path),
        "--batch-size",
        str(args.batch_size),
        "--preview-samples",
        str(args.preview_samples),
        "--pec-checkpoint-dir",
        str(args.pec_checkpoint_dir),
        "--profiler-path",
        str(args.profiler_path),
        "--pec-composer-model",
        args.pec_composer_model,
        "--num-query-tokens",
        str(args.num_query_tokens),
        "--mask-probability",
        str(args.mask_probability),
        "--mask-seed",
        str(args.mask_seed),
        "--max-profiler-len",
        str(args.max_profiler_len),
        "--max-composer-len",
        str(args.max_composer_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--worker-kind",
        job["kind"],
        "--worker-thinking-mode",
        "on" if job["enable_thinking"] else "off",
        "--worker-summary-path",
        str(summary_path),
        "--worker-prompt-repeat-count",
        str(job.get("prompt_repeat_count", 1)),
    ]

    command.append("--do-sample" if args.do_sample else "--no-sample")

    if job["kind"] == "base":
        command.extend(["--worker-model-name", job["model_name"], "--worker-scenario", job["scenario"]])
    else:
        command.extend(["--worker-scenario", job["scenario"]])

    return command


def run_parallel_eval_jobs(
    *,
    jobs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    eval_samples: Sequence[Dict[str, Any]],
    gpu_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    eval_samples_path = run_dir / "heldout_eval_samples.jsonl"
    write_jsonl(eval_samples, eval_samples_path)

    logs_dir = run_dir / "logs"
    worker_summary_dir = run_dir / "worker_summaries"
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_summary_dir.mkdir(parents=True, exist_ok=True)

    pending_jobs = list(enumerate(jobs))
    running_jobs: Dict[int, Dict[str, Any]] = {}
    completed_summaries: List[Tuple[int, Dict[str, Any]]] = []

    print(f"Running {len(jobs)} eval jobs across GPUs {list(gpu_ids)}", flush=True)
    print(f"Worker logs: {logs_dir}", flush=True)

    try:
        while pending_jobs or running_jobs:
            available_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in running_jobs]
            while available_gpu_ids and pending_jobs:
                gpu_id = available_gpu_ids.pop(0)
                job_index, job = pending_jobs.pop(0)
                job_name = eval_job_name(job)
                log_path = logs_dir / f"{job_name}.log"
                summary_path = worker_summary_dir / f"{job_name}.json"
                worker_command = build_worker_command(
                    args=args,
                    run_dir=run_dir,
                    eval_samples_path=eval_samples_path,
                    job=job,
                    summary_path=summary_path,
                )
                log_handle = log_path.open("w", encoding="utf-8", buffering=1)
                worker_env = os.environ.copy()
                worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                worker_env["PYTHONUNBUFFERED"] = "1"

                print(f"  - launching {job_name} on GPU {gpu_id} -> {log_path}", flush=True)
                process = subprocess.Popen(
                    worker_command,
                    cwd=Path(__file__).parent,
                    env=worker_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                stream_thread = threading.Thread(
                    target=stream_worker_output,
                    kwargs={
                        "process": process,
                        "log_handle": log_handle,
                        "prefix": f"gpu{gpu_id}:{job_name}",
                    },
                    daemon=True,
                )
                stream_thread.start()
                running_jobs[gpu_id] = {
                    "job_index": job_index,
                    "job": job,
                    "job_name": job_name,
                    "log_path": log_path,
                    "log_handle": log_handle,
                    "process": process,
                    "summary_path": summary_path,
                    "stream_thread": stream_thread,
                }

            finished_gpu_ids: List[int] = []
            for gpu_id, job_state in running_jobs.items():
                process = job_state["process"]
                return_code = process.poll()
                if return_code is None:
                    continue

                if process.stdout is not None:
                    process.stdout.close()
                job_state["stream_thread"].join(timeout=5)
                job_state["log_handle"].close()
                if return_code != 0:
                    raise RuntimeError(
                        f"Eval worker failed for {job_state['job_name']} on GPU {gpu_id}. "
                        f"See {job_state['log_path']}"
                    )

                summary = json.loads(job_state["summary_path"].read_text(encoding="utf-8"))
                completed_summaries.append((job_state["job_index"], summary))
                finished_gpu_ids.append(gpu_id)
                print(f"  - finished {job_state['job_name']} on GPU {gpu_id}", flush=True)

            for gpu_id in finished_gpu_ids:
                del running_jobs[gpu_id]

            if running_jobs:
                time.sleep(1)
    except Exception:
        for job_state in running_jobs.values():
            process = job_state["process"]
            if process.poll() is None:
                process.terminate()
            if process.stdout is not None:
                process.stdout.close()
            job_state["stream_thread"].join(timeout=5)
            job_state["log_handle"].close()
        raise

    completed_summaries.sort(key=lambda item: item[0])
    return [summary for _, summary in completed_summaries]


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    default_output_dir = base_dir / "outputs" / "pec_eval_re2"

    parser = argparse.ArgumentParser(description="Run held-out PEC ablation evaluation.")
    parser.add_argument(
        "--protocol",
        choices=["re2_paper", "heldout_pec"],
        default="re2_paper",
        help="Evaluation protocol. `re2_paper` matches the RE2 paper benchmark style; `heldout_pec` keeps the old blended held-out eval.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--benchmark-data-dir",
        type=Path,
        default=base_dir / "data" / "re2_benchmarks",
        help="Directory containing the RE2 paper benchmark JSONL files.",
    )
    parser.add_argument(
        "--benchmark-datasets",
        nargs="+",
        default=list(RE2_DEFAULT_DATASETS),
        help="Benchmark datasets to evaluate in RE2 paper mode.",
    )
    parser.add_argument(
        "--acts",
        nargs="+",
        default=list(RE2_DEFAULT_ACTS),
        choices=["vanilla", "cot", "ps", "pal"],
        help="Prompting methods to evaluate in RE2 paper mode.",
    )
    parser.add_argument(
        "--read-times",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Question repetition counts. The paper's RE2 setting corresponds to `2`.",
    )
    benchmark_download_group = parser.add_mutually_exclusive_group()
    benchmark_download_group.add_argument(
        "--download-benchmark-data",
        dest="download_benchmark_data",
        action="store_true",
        help="Download missing RE2 benchmark JSONL files from the official repository.",
    )
    benchmark_download_group.add_argument(
        "--no-download-benchmark-data",
        dest="download_benchmark_data",
        action="store_false",
        help="Fail instead of downloading missing benchmark files.",
    )
    parser.set_defaults(download_benchmark_data=True)
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

    parser.add_argument("--base-models", nargs="+", default=list(DEFAULT_BASE_MODELS[:1]))
    parser.add_argument(
        "--base-scenarios",
        nargs="+",
        choices=["base_no_hint", "base_prompt_twice"],
        default=["base_no_hint", "base_prompt_twice"],
        help="Select which base baselines to evaluate.",
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-pec", action="store_true")
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="CUDA GPU IDs to use for parallel eval jobs. Defaults to all visible GPUs.",
    )

    parser.add_argument("--pec-checkpoint-dir", type=Path, default=base_dir / "models" / "PEC")
    parser.add_argument("--profiler-path", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--pec-composer-model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-query-tokens", type=int, default=64)
    parser.add_argument("--mask-probability", type=float, default=0.3)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--max-profiler-len", type=int, default=6144)
    parser.add_argument("--max-composer-len", type=int, default=6080)

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of samples to evaluate per generation batch. Increase to use more GPU memory.",
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=1,
        help="Number of sample previews to print per eval job.",
    )
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling")
    sampling_group.add_argument("--no-sample", dest="do_sample", action="store_false", help="Use greedy decoding")
    parser.set_defaults(do_sample=False)
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Enable Qwen thinking mode. Off by default because it is not part of the RE2 paper protocol.",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen thinking mode.",
    )
    parser.set_defaults(enable_thinking=False)
    parser.add_argument(
        "--thinking-modes",
        nargs="+",
        choices=["off", "on"],
        default=["off", "on"],
        help="Compare Qwen chat-template thinking modes. Default evaluates both off and on.",
    )

    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--eval-samples-jsonl", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-kind", choices=["base", "pec"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-protocol", choices=["re2_paper", "heldout_pec"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-prompt-repeat-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-act", choices=["vanilla", "cot", "ps", "pal"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-read-times", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-scenario",
        choices=["base_no_hint", "base_prompt_twice", "pec_with_hint", "pec_masked_hint"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-thinking-mode", choices=["off", "on"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-summary-path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_kind is not None:
        if args.run_dir is None:
            raise ValueError("--run-dir is required in worker mode")
        if args.worker_summary_path is None:
            raise ValueError("--worker-summary-path is required in worker mode")
        if args.eval_samples_jsonl is None:
            raise ValueError("--eval-samples-jsonl is required in worker mode")

        worker_protocol = args.worker_protocol or args.protocol
        device = get_device()
        dtype = get_inference_dtype(device)
        print(f"Device={device}, dtype={dtype}", flush=True)

        if worker_protocol == "re2_paper":
            if args.worker_dataset_name is None:
                raise ValueError("--worker-dataset-name is required for RE2 worker jobs")
            if args.worker_act is None:
                raise ValueError("--worker-act is required for RE2 worker jobs")
            if args.worker_read_times < 1:
                raise ValueError("--worker-read-times must be at least 1")

            eval_samples = read_jsonl(args.eval_samples_jsonl)
            print(f"Loaded RE2 worker samples: {len(eval_samples)}", flush=True)
            if args.worker_kind == "base":
                if args.worker_model_name is None:
                    raise ValueError("--worker-model-name is required for RE2 base worker jobs")
                summary = evaluate_base_re2_benchmark(
                    dataset_name=args.worker_dataset_name,
                    model_name=args.worker_model_name,
                    act=args.worker_act,
                    read_times=args.worker_read_times,
                    examples=eval_samples,
                    output_dir=args.run_dir,
                    device=device,
                    dtype=dtype,
                    batch_size=args.batch_size,
                    preview_samples=args.preview_samples,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    enable_thinking=args.enable_thinking,
                )
            else:
                summary = evaluate_pec_re2_benchmark(
                    dataset_name=args.worker_dataset_name,
                    act=args.worker_act,
                    read_times=args.worker_read_times,
                    examples=eval_samples,
                    output_dir=args.run_dir,
                    device=device,
                    checkpoint_dir=args.pec_checkpoint_dir,
                    profiler_path=args.profiler_path,
                    composer_model_name=args.pec_composer_model,
                    num_query_tokens=args.num_query_tokens,
                    max_profiler_len=args.max_profiler_len,
                    max_composer_len=args.max_composer_len,
                    batch_size=args.batch_size,
                    preview_samples=args.preview_samples,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    enable_thinking=args.enable_thinking,
                )
            args.worker_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Worker finished: {build_re2_job_name(summary)}", flush=True)
            return

        if args.worker_prompt_repeat_count < 1:
            raise ValueError("--worker-prompt-repeat-count must be at least 1")
        if args.worker_thinking_mode is None:
            raise ValueError("--worker-thinking-mode is required in worker mode")

        run_dir = args.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_samples = load_eval_samples(args)
        print(f"Loaded held-out eval samples: {len(eval_samples)}", flush=True)
        worker_job = {
            "kind": args.worker_kind,
            "model_name": args.worker_model_name or args.pec_composer_model,
            "enable_thinking": args.worker_thinking_mode == "on",
            "prompt_repeat_count": args.worker_prompt_repeat_count,
        }
        if args.worker_kind == "pec":
            if args.worker_scenario is None:
                raise ValueError("--worker-scenario is required for PEC worker jobs")
            worker_job["scenario"] = args.worker_scenario
        else:
            if args.worker_model_name is None:
                raise ValueError("--worker-model-name is required for base worker jobs")
            worker_job["scenario"] = args.worker_scenario or "base_no_hint"

        summary = run_eval_job(
            job=worker_job,
            args=args,
            eval_samples=eval_samples,
            output_dir=run_dir,
            device=device,
            dtype=dtype,
        )
        args.worker_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Worker finished: {eval_job_name(worker_job)}", flush=True)
        return

    if args.protocol == "re2_paper":
        run_re2_paper_protocol(args)
        return

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.preview_samples < 0:
        raise ValueError("--preview-samples must be at least 0")
    if args.worker_prompt_repeat_count < 1:
        raise ValueError("--worker-prompt-repeat-count must be at least 1")

    run_dir = args.run_dir or (args.output_dir / time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype = get_inference_dtype(device)
    print(f"Device={device}, dtype={dtype}", flush=True)
    eval_samples = load_eval_samples(args)
    print(f"Loaded held-out eval samples: {len(eval_samples)}", flush=True)

    thinking_modes = [mode == "on" for mode in args.thinking_modes]
    jobs = build_eval_jobs(
        base_models=args.base_models,
        base_scenarios=args.base_scenarios,
        composer_model_name=args.pec_composer_model,
        thinking_modes=thinking_modes,
        skip_base=args.skip_base,
        skip_pec=args.skip_pec,
    )
    gpu_ids = resolve_gpu_ids(args)

    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpu-ids must not contain duplicates")

    config_payload = {
        "device": str(device),
        "dtype": str(dtype),
        "num_eval_samples": len(eval_samples),
        "base_models": args.base_models,
        "base_prompt_scenarios": args.base_scenarios,
        "pec_composer_model": args.pec_composer_model,
        "soft_prompt_tokens": args.num_query_tokens,
        "batch_size": args.batch_size,
        "preview_samples": args.preview_samples,
        "gpu_ids": gpu_ids,
        "num_eval_jobs": len(jobs),
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

    if device.type == "cuda" and len(gpu_ids) > 1 and len(jobs) > 1:
        summaries = run_parallel_eval_jobs(
            jobs=jobs,
            args=args,
            run_dir=run_dir,
            eval_samples=eval_samples,
            gpu_ids=gpu_ids,
        )
    else:
        summaries = [
            run_eval_job(
                job=job,
                args=args,
                eval_samples=eval_samples,
                output_dir=run_dir,
                device=device,
                dtype=dtype,
            )
            for job in jobs
        ]

    write_summary_csv(summaries, run_dir / "summary.csv")
    (run_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDone. Summary written to:", run_dir / "summary.csv", flush=True)


if __name__ == "__main__":
    if setproctitle is not None:
        setproctitle.setproctitle("sinkect")
    main()
