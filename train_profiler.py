import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_qa_pairs(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def tokenize_function(examples, tokenizer):
    texts = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)


def main():
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "qa_pairs.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing QA file: {data_path}")

    profiler_path = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(profiler_path, use_fast=True)

    items = load_qa_pairs(str(data_path))
    dataset = Dataset.from_list(items)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    dataset = dataset.remove_columns(["question", "answer"])
    dataset = dataset.train_test_split(test_size=0.05)

    model = AutoModelForSequenceClassification.from_pretrained(profiler_path, num_labels=2)
    model.to(get_device(), dtype=torch.float16)

    training_args = TrainingArguments(
        output_dir=str(base_dir / "outputs" / "profiler"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        half_precision_backend="mps",
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
