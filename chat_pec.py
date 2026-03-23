from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Any, Callable, TextIO

import torch

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
MULTILINE_START_COMMANDS = {"/multiline", "/paste"}
MULTILINE_SEND_COMMANDS = {"/send", "/submit", "/end"}
MULTILINE_CANCEL_COMMANDS = {"/cancel", "/abort"}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Interactive chat/repl for base and PEC models with memory-KV inspection."
    )
    parser.add_argument("--mode", choices=["pec", "base", "compare"], default="pec")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--pec-checkpoint-dir", type=Path, default=base_dir / "models" / "PEC")
    parser.add_argument("--profiler-path", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--pec-composer-model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num-query-tokens", type=int, default=64)
    parser.add_argument("--mask-probability", type=float, default=0.3)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--max-profiler-len", type=int, default=6144)
    parser.add_argument("--max-composer-len", type=int, default=6080)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA GPU id to use. Defaults to current auto-detected device.")
    parser.add_argument("--show-raw", action="store_true", help="Print raw output even when it matches cleaned output.")
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt once without interactive chat.")
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--apply-mask", dest="apply_mask", action="store_true", help="Mask the composer-visible prompt like PEC masked eval.")
    mask_group.add_argument("--no-apply-mask", dest="apply_mask", action="store_false", help="Do not mask the composer-visible prompt.")
    parser.set_defaults(apply_mask=False)

    composer_group = parser.add_mutually_exclusive_group()
    composer_group.add_argument(
        "--soft-prompt-only",
        dest="soft_prompt_only",
        action="store_true",
        help="Hide the original text from the PEC composer and answer using only latent memory.",
    )
    composer_group.add_argument(
        "--with-visible-prompt",
        dest="soft_prompt_only",
        action="store_false",
        help="Show the original text to the PEC composer.",
    )
    parser.set_defaults(soft_prompt_only=False)

    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--do-sample", dest="do_sample", action="store_true", help="Enable sampling.")
    sampling_group.add_argument("--no-sample", dest="do_sample", action="store_false", help="Use greedy decoding.")
    parser.set_defaults(do_sample=False)

    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=False)
    return parser.parse_args()


def resolve_device(gpu_id: int | None) -> torch.device:
    from evaluate_pec import get_device

    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-id was provided but CUDA is not available.")
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return get_device()


def print_header(
    args: argparse.Namespace,
    device: torch.device,
    sampling: dict[str, float],
) -> None:
    from models.eval_utils import thinking_mode_name

    print("Interactive chat ready.", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Thinking: {thinking_mode_name(args.enable_thinking)}", flush=True)
    print(f"  Decoding: {'sampling' if args.do_sample else 'greedy'}", flush=True)
    if args.do_sample:
        print(
            f"  Sampling params: temperature={sampling['temperature']}, top_p={sampling['top_p']}, "
            f"top_k={sampling['top_k']}, min_p={sampling['min_p']}",
            flush=True,
        )
    print(f"  Composer prompt: {'memory_only' if args.soft_prompt_only else 'visible_prompt'}", flush=True)
    if args.soft_prompt_only:
        print("  Apply mask: False (ignored by memory_only)", flush=True)
    else:
        print(f"  Apply mask: {args.apply_mask}", flush=True)
    if args.mode in {"base", "compare"}:
        print(f"  Base model: {args.base_model}", flush=True)
    if args.mode in {"pec", "compare"}:
        print(f"  PEC checkpoint: {args.pec_checkpoint_dir}", flush=True)
        print(f"  PEC composer model: {args.pec_composer_model}", flush=True)
    print("  Commands: /help, /multiline, /reset, /exit", flush=True)


def print_response_block(
    *,
    label: str,
    clean_prompt: str,
    visible_prompt: str | None,
    raw_prediction: str,
    cleaned_prediction: str,
    elapsed_seconds: float,
    memory_stats: dict[str, Any] | None = None,
    show_raw: bool = False,
) -> None:
    def print_tensor_slice(title: str, slice_values: list[list[float]]) -> None:
        if not slice_values:
            print(f"  {title}: <none>", flush=True)
            return

        print(f"  {title}:", flush=True)
        for row_index, row_values in enumerate(slice_values):
            row = ", ".join(f"{value:.4f}" for value in row_values)
            print(f"    q{row_index:02d} [{row}]", flush=True)

    visible_prompt_display = "<empty>" if visible_prompt == "" else visible_prompt

    print(f"\n[{label}]", flush=True)
    print(f"  Elapsed: {elapsed_seconds:.2f}s", flush=True)
    print(f"  Prompt: {clean_prompt}", flush=True)
    if visible_prompt is not None and visible_prompt != clean_prompt:
        print(f"  Visible prompt: {visible_prompt_display}", flush=True)
    print(f"  Prediction: {cleaned_prediction}", flush=True)
    if show_raw or raw_prediction != cleaned_prediction:
        print(f"  Raw prediction: {raw_prediction}", flush=True)
    if memory_stats is not None:
        print(
            f"  Latent Z: norm={memory_stats['latent_z_norm']:.6f}, "
            f"mean={memory_stats['latent_z_mean']:.6f}, std={memory_stats['latent_z_std']:.6f}",
            flush=True,
        )
        print(
            f"  Memory K: norm={memory_stats['memory_k_norm']:.6f}, "
            f"mean={memory_stats['memory_k_mean']:.6f}, std={memory_stats['memory_k_std']:.6f}",
            flush=True,
        )
        print(
            f"  Memory V: norm={memory_stats['memory_v_norm']:.6f}, "
            f"mean={memory_stats['memory_v_mean']:.6f}, std={memory_stats['memory_v_std']:.6f}",
            flush=True,
        )
        print(f"  Memory slots: {memory_stats['memory_slots']}", flush=True)
        print_tensor_slice("K_mem[0, 0, :5, :8]", memory_stats["memory_key_slice"])
        print_tensor_slice("V_mem[0, 0, :5, :8]", memory_stats["memory_value_slice"])


def print_help() -> None:
    print("  /help  Show commands", flush=True)
    print("  /multiline  Enter multiline input mode; finish with /send or discard with /cancel", flush=True)
    print("  /reset Reset the turn counter used for deterministic masking", flush=True)
    print("  /exit  Quit", flush=True)


def normalize_user_input(text: str) -> str:
    if not text.strip():
        return ""
    return text.rstrip("\n")


def read_prompt_from_stream(stream: TextIO) -> str:
    return normalize_user_input(stream.read())


def stdin_has_pending_line(stream: TextIO, timeout_seconds: float = 0.05) -> bool:
    try:
        ready, _, _ = select.select([stream], [], [], timeout_seconds)
    except (OSError, TypeError, ValueError):
        return False
    return bool(ready)


def drain_ready_lines(
    *,
    input_fn: Callable[[str], str] = input,
    has_pending_input: Callable[[], bool],
    prompt: str = "",
) -> list[str]:
    lines: list[str] = []
    while has_pending_input():
        try:
            lines.append(input_fn(prompt))
        except EOFError:
            break
    return lines


def read_multiline_input(
    *,
    input_fn: Callable[[str], str] = input,
    prompt: str = "... ",
) -> str | None:
    print("Entering multiline mode. Finish with /send. Discard with /cancel.", flush=True)
    lines: list[str] = []
    while True:
        try:
            line = input_fn(prompt)
        except EOFError:
            return normalize_user_input("\n".join(lines))

        command = line.strip()
        if command in MULTILINE_SEND_COMMANDS:
            return normalize_user_input("\n".join(lines))
        if command in MULTILINE_CANCEL_COMMANDS:
            return None
        lines.append(line)


def run_one_turn(
    *,
    user_input: str,
    turn_index: int,
    args: argparse.Namespace,
    base_model,
    base_tokenizer,
    pec_model,
    profiler_tokenizer,
    composer_tokenizer,
    masker,
    device: torch.device,
    sampling: dict[str, float],
) -> None:
    from evaluate_pec import generate_base_response, generate_pec_response
    from models.eval_utils import apply_mask_deterministically, strip_thinking_trace

    visible_prompt = user_input
    if args.soft_prompt_only:
        visible_prompt = ""
    elif masker is not None:
        visible_prompt = apply_mask_deterministically(
            user_input,
            masker,
            seed=args.mask_seed + turn_index,
        )

    if args.mode in {"base", "compare"}:
        started_at = time.perf_counter()
        raw_base_prediction = generate_base_response(
            model=base_model,
            tokenizer=base_tokenizer,
            prompt_text=user_input,
            device=device,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            enable_thinking=args.enable_thinking,
        )
        base_elapsed = time.perf_counter() - started_at
        cleaned_base_prediction = strip_thinking_trace(raw_base_prediction)
        print_response_block(
            label="BASE",
            clean_prompt=user_input,
            visible_prompt=None,
            raw_prediction=raw_base_prediction,
            cleaned_prediction=cleaned_base_prediction,
            elapsed_seconds=base_elapsed,
            show_raw=args.show_raw,
        )

    if args.mode in {"pec", "compare"}:
        started_at = time.perf_counter()
        raw_pec_prediction, memory_stats = generate_pec_response(
            model=pec_model,
            profiler_tokenizer=profiler_tokenizer,
            composer_tokenizer=composer_tokenizer,
            clean_prompt_text=user_input,
            visible_prompt_text=visible_prompt,
            device=device,
            max_profiler_len=args.max_profiler_len,
            max_composer_len=args.max_composer_len,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            enable_thinking=args.enable_thinking,
        )
        pec_elapsed = time.perf_counter() - started_at
        cleaned_pec_prediction = strip_thinking_trace(raw_pec_prediction)
        print_response_block(
            label="PEC",
            clean_prompt=user_input,
            visible_prompt=visible_prompt,
            raw_prediction=raw_pec_prediction,
            cleaned_prediction=cleaned_pec_prediction,
            elapsed_seconds=pec_elapsed,
            memory_stats=memory_stats,
            show_raw=args.show_raw,
        )


def main() -> None:
    args = parse_args()
    from evaluate_pec import (
        clear_memory,
        get_inference_dtype,
        get_sampling_params,
        load_base_model,
        load_pec_model,
    )
    from models.data import EntityMasker

    device = resolve_device(args.gpu_id)
    dtype = get_inference_dtype(device)
    sampling = get_sampling_params(args.enable_thinking)

    base_model = None
    base_tokenizer = None
    pec_model = None
    profiler_tokenizer = None
    composer_tokenizer = None
    masker = None

    try:
        if args.mode in {"base", "compare"}:
            base_model, base_tokenizer = load_base_model(args.base_model, device, dtype)

        if args.mode in {"pec", "compare"}:
            pec_model, profiler_tokenizer, composer_tokenizer = load_pec_model(
                checkpoint_dir=args.pec_checkpoint_dir,
                profiler_path=args.profiler_path,
                composer_model_name=args.pec_composer_model,
                num_query_tokens=args.num_query_tokens,
                device=device,
            )
            if args.apply_mask:
                masker = EntityMasker(mask_prob=args.mask_probability)

        print_header(
            args,
            device,
            sampling,
        )

        if args.prompt is not None:
            run_one_turn(
                user_input=args.prompt,
                turn_index=0,
                args=args,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                pec_model=pec_model,
                profiler_tokenizer=profiler_tokenizer,
                composer_tokenizer=composer_tokenizer,
                masker=masker,
                device=device,
                sampling=sampling,
            )
            return

        if not sys.stdin.isatty():
            stdin_prompt = read_prompt_from_stream(sys.stdin)
            if not stdin_prompt:
                raise RuntimeError(
                    "Interactive stdin is not attached and no prompt was provided. "
                    "Pass `--prompt \"...\"` or pipe prompt text through stdin."
                )
            run_one_turn(
                user_input=stdin_prompt,
                turn_index=0,
                args=args,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                pec_model=pec_model,
                profiler_tokenizer=profiler_tokenizer,
                composer_tokenizer=composer_tokenizer,
                masker=masker,
                device=device,
                sampling=sampling,
            )
            return

        turn_index = 0
        while True:
            try:
                first_line = input("\nuser> ")
            except EOFError:
                print("", flush=True)
                break

            command = first_line.strip()
            if not command:
                continue
            if command in EXIT_COMMANDS:
                break
            if command == "/help":
                print_help()
                continue
            if command == "/reset":
                turn_index = 0
                print("Turn counter reset.", flush=True)
                continue
            if command in MULTILINE_START_COMMANDS:
                user_input = read_multiline_input()
                if user_input is None:
                    print("Multiline input discarded.", flush=True)
                    continue
            else:
                pasted_lines = drain_ready_lines(
                    input_fn=input,
                    has_pending_input=lambda: stdin_has_pending_line(sys.stdin),
                )
                if pasted_lines:
                    user_input = normalize_user_input("\n".join([first_line, *pasted_lines]))
                else:
                    user_input = normalize_user_input(first_line)

            if not user_input:
                continue

            run_one_turn(
                user_input=user_input,
                turn_index=turn_index,
                args=args,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                pec_model=pec_model,
                profiler_tokenizer=profiler_tokenizer,
                composer_tokenizer=composer_tokenizer,
                masker=masker,
                device=device,
                sampling=sampling,
            )
            turn_index += 1
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        clear_memory()


if __name__ == "__main__":
    main()
