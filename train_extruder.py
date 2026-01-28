import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

from models.architecture import PECEngine


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_qa_pairs(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def tokenize_function(examples, profiler_tokenizer, composer_tokenizer):
    profiler_enc = profiler_tokenizer(
        examples["question"], padding="max_length", truncation=True, max_length=128
    )
    composer_enc = composer_tokenizer(
        examples["answer"], padding="max_length", truncation=True, max_length=128
    )

    return {
        "profiler_input_ids": profiler_enc["input_ids"],
        "profiler_attention_mask": profiler_enc["attention_mask"],
        "composer_input_ids": composer_enc["input_ids"],
        "composer_attention_mask": composer_enc["attention_mask"],
    }


def main():
    base_dir = Path(__file__).parent
    data_path = base_dir / "data" / "qa_pairs.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing QA file: {data_path}")

    model = PECEngine()
    profiler_tokenizer = model.profiler_tokenizer
    composer_tokenizer = model.composer_tokenizer

    items = load_qa_pairs(str(data_path))
    dataset = Dataset.from_list(items)
    dataset = dataset.map(
        lambda x: tokenize_function(x, profiler_tokenizer, composer_tokenizer), batched=True
    )
    dataset = dataset.remove_columns(["question", "answer"])
    dataset = dataset.train_test_split(test_size=0.05)

    training_args = TrainingArguments(
        output_dir=str(base_dir / "outputs" / "extruder"),
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
    )

    trainer.train()


if __name__ == "__main__":
    main()
