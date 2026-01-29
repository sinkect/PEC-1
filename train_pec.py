import os
import random
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from models.architecture import PECEngine
from models.data import PECDataset, PECCollator, EntityMasker


def get_device() -> torch.device:
    """Returns the available computation device.

    Returns:
        torch.device: CUDA, MPS, or CPU device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_hotpotqa(example: Dict[str, Any]) -> Dict[str, str]:
    """Formats the HotpotQA dataset for the PEC model.

    HotpotQA provides a list of contexts (titles and sentences).
    This function flattens them into a single long context string to simulate
    a long document retrieval/reasoning scenario.

    Args:
        example: Raw example from HotpotQA (distractor configuration).

    Returns:
        Dictionary with 'question', 'context', 'answer'.
    """
    # 1. Extract raw fields
    question = example["question"]
    answer = example["answer"]

    # 2. Flatten Context
    # HotpotQA context structure:
    # {
    #   'title': ['Wiki Title 1', 'Wiki Title 2', ...],
    #   'sentences': [['Sent 1', 'Sent 2'], ['Sent A', 'Sent B'], ...]
    # }
    context_titles = example["context"]["title"]
    context_sentences = example["context"]["sentences"]

    formatted_context_parts = []

    for title, sentences in zip(context_titles, context_sentences):
        # Join sentences into a single paragraph
        paragraph = "".join(sentences)
        formatted_context_parts.append(f"Title: {title}\nContent: {paragraph}")

    # Join all paragraphs with double newlines
    full_context = "\n\n".join(formatted_context_parts)

    # 3. Dynamic Instructions (Reasoning focused)
    instruction_pool = [
        "Read the following documents and answer the question using multi-hop reasoning.",
        "Extract relevant information from the context below to answer the user's query.",
        "Identify the bridge entities in the documents and deduce the answer.",
        "Based on the provided wiki passages, answer the question step-by-step.",
        "Locate the evidence in the text and provide the precise answer.",
    ]

    selected_instruction = random.choice(instruction_pool)

    return {
        "question": selected_instruction + f"\n\nUser Query: {question}",
        "context": full_context,
        "answer": answer,
    }

def main():
    """Main entry point for training the PEC model."""

    # 1. Configuration and Paths
    device = get_device()
    base_dir = Path(__file__).parent
    output_dir = base_dir / "outputs" / "PEC"

    profiler_model_path = base_dir / "models/profiler"
    composer_model_name = "Qwen/Qwen3-1.7B"

    print(f"Device: {device}")
    print(f"Output Directory: {output_dir}")

    # 2. Load Tokenizers
    print("Loading tokenizers...")
    profiler_tokenizer = AutoTokenizer.from_pretrained(profiler_model_path)
    composer_tokenizer = AutoTokenizer.from_pretrained(composer_model_name)

    # 3. Load and Preprocess Dataset
    print("Loading HotpotQA (distractor) dataset...")
    raw_dataset = load_dataset("hotpot_qa", "distractor", split="train")

    # Standardize format (rename columns, inject instructions)
    formatted_dataset = raw_dataset.map(
        format_case_summary,
        num_proc=16,
        remove_columns=raw_dataset.column_names
    )

    # Filter valid examples
    formatted_dataset = formatted_dataset.filter(lambda x: len(x["context"]) > 100)

    # Split into Train and Validation sets
    split_dataset = formatted_dataset.train_test_split(test_size=0.02)

    # 4. Initialize Dynamic Datasets (The Core of Information Gap Training)
    # We wrap the HF dataset with our custom PECDataset to enable on-the-fly masking.

    # Train set: Apply aggressive masking (40%) to force latent usage.
    train_context_masker = EntityMasker(mask_prob=0.4)
    train_query_masker = EntityMasker(mask_prob=0.15)
    train_dataset = PECDataset(
        data=split_dataset["train"],
        context_masker=train_context_masker,
        query_masker=train_query_masker
    )

    # Eval set: No masking (or low masking) to evaluate true generation capability.
    eval_masker = EntityMasker(mask_prob=0.0)

    eval_dataset = PECDataset(
        data=split_dataset["test"],
        context_masker=eval_masker,
        query_masker=eval_masker
    )

    # 5. Initialize PEC Engine
    print("Initializing PEC Engine (Profiler-Extruder-Composer)...")
    model = PECEngine(
        profiler_path=str(profiler_model_path),
        composer_path=composer_model_name
    )
    model.to(device)

    # 6. Initialize Data Collator
    # Handles dual tokenization and label masking for the Composer
    data_collator = PECCollator(
        profiler_tokenizer=profiler_tokenizer,
        composer_tokenizer=composer_tokenizer,
        max_profiler_len=8192,
        max_composer_len=4096
    )

    # 7. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,

        # Batch size configuration
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,

        # Optimization
        learning_rate=5e-4,  # Base learning rate (will be overridden by param groups)
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        # Mixed Precision
        fp16=False,
        bf16=True,
        tf32=True,

        # Logging and Saving
        logging_steps=10,
        report_to="none",
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500  ,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",

        # Dataloader
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        remove_unused_columns=False  # Important for custom datasets
    )

    # 8. Define Optimizer Groups (Differential Learning Rates)
    optimizer_grouped_parameters = [
        {
            # Profiler (Encoder): Low LR to preserve pretrained knowledge
            "params": [p for n, p in model.named_parameters() if "profiler" in n and p.requires_grad],
            "lr": 1e-5,
        },
        {
            # Extruder (Bridge): Higher LR to learn compression rapidly
            "params": [p for n, p in model.named_parameters() if
                       ("extruder" in n or "projector" in n) and p.requires_grad],
            "lr": 5e-4,
        },

    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    # 9. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 10. Start Training
    print("Starting training...")
    trainer.train()

    # 11. Save Final Model
    print(f"Saving artifacts to {output_dir}")
    trainer.save_model(str(output_dir))
    composer_tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
