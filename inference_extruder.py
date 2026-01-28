import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from pathlib import Path
from models.architecture import PECEngine


def main():
    # ------------------------------------------------------------------
    # 1. 설정 및 모델 로드
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    base_dir = Path(__file__).parent
    checkpoint_path = base_dir / "models/extruder/pytorch_model.bin"  # 경로 확인 필요

    print("Loading Model Architecture...")
    model = PECEngine(profiler_path="profiler", composer_path="Qwen/Qwen3-1.7B")

    if checkpoint_path.exists():
        print(f"Loading Weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model.half().to(device)
    model.eval()

    print("Loading Tokenizers...")
    profiler_tokenizer = AutoTokenizer.from_pretrained(base_dir / "models/profiler", use_fast=True)
    composer_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", use_fast=True)

    # ------------------------------------------------------------------
    # 2. 추론 루프
    # ------------------------------------------------------------------
    print("Loading Dataset...")
    dataset = load_dataset("ChicagoHAI/CaseSumm", split="train")
    dataset = dataset.select(range(len(dataset) - 5, len(dataset)))  # 5개 테스트

    print("\nStarting Inference (Trigger-Based Mode)...\n")

    for i, example in enumerate(dataset):
        doc = example.get("opinion") or example.get("text")
        target_text = example.get("syllabus") or example.get("summary")
        if not doc: continue

        # [수정 1] Profiler 입력: 학습 때처럼 "본문 + 프롬프트" 결합
        # 프롬프트도 압축 대상에 포함시켜야 모델이 뭘 해야 할지 압니다.
        prompt = "Summarize the facts, procedural history, and holding of the following case.",
        full_input = f"--- INSTRUCTION ---\n{prompt}\n\n--- SOURCE DOCUMENT ---\n{doc}"

        p_inputs = profiler_tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(device)

        with torch.no_grad():
            # --- Step 1: Extruder로 압축 (Latents 생성) ---
            prof_outputs = model.profiler(
                input_ids=p_inputs.input_ids,
                attention_mask=p_inputs.attention_mask
            )
            prof_hidden = prof_outputs.last_hidden_state.to(dtype=model.composer.dtype)

            extruded = model.extruder(context=prof_hidden, attn_mask=p_inputs.attention_mask.to(device))
            projected = model.projector(extruded)

            # Separator 추가
            batch_size = projected.shape[0]
            sep = model.sep_token.expand(batch_size, -1, -1).to(dtype=projected.dtype, device=device)
            latents_embeds = torch.cat([projected, sep], dim=1)

            # ------------------------------------------------------------------
            # [수정 2] Trigger 추가 (학습 환경 재현)
            # ------------------------------------------------------------------
            # 학습 때 Composer는 "<|im_start|>assistant\n"만 보고 생성을 시작했습니다.
            # 추론 때도 이 '시작 신호'를 줘야 Latents를 읽고 답을 합니다.

            a_start_text = "<|im_start|>assistant\n"
            a_start_ids = composer_tokenizer(a_start_text, return_tensors="pt").input_ids.to(device)
            a_start_embeds = model.composer.get_input_embeddings()(a_start_ids)

            u_start_text = "<|im_start|>user\n"
            u_start_ids = composer_tokenizer(u_start_text, return_tensors="pt").input_ids.to(device)
            u_start_embeds = model.composer.get_input_embeddings()(u_start_ids)
            u_end_text = "<|im_end|>\n"
            u_end_ids = composer_tokenizer(u_end_text, return_tensors="pt").input_ids.to(device)
            u_end_embeds = model.composer.get_input_embeddings()(u_end_ids)

            # [Trigger] + [Latents] 순서로 결합
            # Qwen은 "Assistant Start" 신호를 보고 -> "Latents"를 참고하여 -> "답변"을 생성함
            combined_embeds = torch.cat([u_start_embeds, latents_embeds, u_end_embeds, a_start_embeds], dim=1)

            attention_mask = torch.ones(
                combined_embeds.shape[:2],
                dtype=torch.long,
                device=device
            )

            # --- Step 2: 생성 (Generate) ---
            generated_ids = model.composer.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=composer_tokenizer.eos_token_id
            )

        output_text = composer_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"--- Example {i + 1} ---")
        print(f"[Input Length]: {len(doc.split())} words")
        print(f"[Generated Summary]:\n{output_text}\n")
        print(f"[Ground Truth]:\n{target_text[:200]}... (생략)\n")
        print("=" * 50)


if __name__ == "__main__":
    main()
