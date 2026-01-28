from pathlib import Path
import random
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback

from models.architecture import PECEngine

from models.data import PECDataCollator


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def format_for_pec(example):
    doc = (example.get("opinion") or "").strip()
    answer = (example.get("syllabus") or "").strip()
    prompt_variations = [
        "Summarize the facts, procedural history, and holding of the following case.",
        "Read the compressed legal document and provide a formal syllabus.",
        "Generate a summary for the provided legal opinion.",
        "Extract the key details and summarize the case below.",
        "What is the summary of this legal text?",
    ]
    prompt = random.choice(prompt_variations)
    return {
        "doc": doc,
        "prompt": prompt,
        "answer": answer,
    }


def tokenize_function(examples, profiler_tokenizer, composer_tokenizer):
    profiler_inputs = []
    qwen_inputs = []
    meta_anchor = (
        "<|im_start|>system\n"
        "Decode the compressed instructions and content from the vectors and execute the task.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Process the compressed information:\n"
    )

    for doc, prompt in zip(examples["doc"], examples["prompt"]):
        combined_input = f"--- INSTRUCTION ---\n{prompt}\n\n--- SOURCE DOCUMENT ---\n{doc}"
        profiler_inputs.append(combined_input)

        if random.random() < 0.5:

            qwen_inputs.append(meta_anchor)
        else:

            hybrid_prompt = (
                f"<|im_start|>user\n"
                f"{prompt}\n"
                f"Here is the compressed context regarding the above instruction:\n"
            )
            qwen_inputs.append(hybrid_prompt)
    profiler_enc = profiler_tokenizer(
        profiler_inputs, padding=False, truncation=True, max_length=8192
    )

    qwen_enc = composer_tokenizer(
        qwen_inputs, padding=False, truncation=True, max_length=512, add_special_tokens=False
    )
    answer_enc = composer_tokenizer(
        examples["answer"], padding=False, truncation=True, max_length=512
    )

    return {
        "profiler_input_ids": profiler_enc["input_ids"],
        "profiler_attention_mask": profiler_enc["attention_mask"],
        "qwen_prompt_ids": qwen_enc["input_ids"],
        "labels": answer_enc["input_ids"],
    }


def main():
    is_load_checkpoint = False
    base_dir = Path(__file__).parent
    model = PECEngine(profiler_path="profiler", composer_path="Qwen/Qwen3-1.7B")
    checkpoint_path = base_dir / "models/extruder/pytorch_model.bin"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        is_load_checkpoint = True
        print(f"Loaded weights from {checkpoint_path}")
    profiler_tokenizer = AutoTokenizer.from_pretrained(base_dir / "models/profiler")
    composer_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    dataset = load_dataset("ChicagoHAI/CaseSumm", split="train")
    dataset = dataset.map(format_for_pec)
    dataset = dataset.filter(lambda x: bool(x["doc"]) and bool(x["answer"]))
    if not is_load_checkpoint:
        dataset = dataset.filter(lambda x: len(x["doc"]) < 3000)
    dataset = dataset.map(
        lambda x: tokenize_function(x, profiler_tokenizer, composer_tokenizer)
        , batched=True
        , remove_columns=dataset.column_names
        , num_proc=16
    )

    dataset = dataset.train_test_split(test_size=0.05)

    data_collector = PECDataCollator(profiler_tokenizer=profiler_tokenizer, composer_tokenizer=composer_tokenizer)


    training_args = TrainingArguments(
        output_dir=str(base_dir / "outputs" / "extruder"),
        num_train_epochs=5,

        per_device_train_batch_size=12,
        gradient_accumulation_steps=3,

        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        fp16=False,
        bf16=True,
        tf32=True,

        remove_unused_columns=False,
        logging_steps=10,
        report_to="none",

        dataloader_num_workers=16,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        save_safetensors=False,

        eval_strategy="steps",
        eval_steps=50 if not is_load_checkpoint else 400,
        save_strategy="steps",
        save_steps=50 if not is_load_checkpoint else 400,

        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,


    )

    optimizer_grouped_parameters = [
        {

            "params": [p for n, p in model.named_parameters() if "profiler" in n and p.requires_grad],
            "lr": 1e-5,
        },
        {

            "params": [p for n, p in model.named_parameters() if
                       ("extruder" in n or "projector" in n) and p.requires_grad],
            "lr": 5e-4 if not is_load_checkpoint else 1e-4,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=(optimizer,None),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collector,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(base_dir / "outputs" / "extruder")
    composer_tokenizer.save_pretrained(base_dir / "outputs" / "extruder")


if __name__ == "__main__":
    print(get_device())
    main()
