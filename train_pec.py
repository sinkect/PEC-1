from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback
)

from models.architecture import PECEngine
from models.data import PECDataset, PECCollator, EntityMasker
from models.dataset_mixing import (
    load_default_4_4_2_blended_dataset,
    save_blend_metadata,
    save_dataset_as_jsonl,
    save_sampled_by_source_as_jsonl,
)
from models.losses import GateL1Trainer


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


class BlendedMessagesToPECSamples(Dataset):
    """Normalizes blended samples into a stable prompt/answer schema."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_dataset[idx]
        return {
            "prompt": str(sample.get("prompt", "")).strip(),
            "answer": str(sample.get("answer", "")).strip(),
            "source": sample.get("source", ""),
        }


def split_dataset_indices(dataset_len: int, test_size: float = 0.02, seed: int = 42) -> Dict[str, Subset]:
    """Creates deterministic train/eval subsets from one dataset."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    eval_len = max(1, int(dataset_len * test_size))
    train_len = dataset_len - eval_len

    if train_len <= 0:
        raise ValueError("Dataset is too small after applying test_size.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    train_indices = permutation[:train_len]
    eval_indices = permutation[train_len:]

    return {
        "train": Subset(range(dataset_len), train_indices),
        "eval": Subset(range(dataset_len), eval_indices),
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

    # 3. Load and Preprocess Blended Dataset (4:4:2)
    print("Loading blended datasets (Open-Platypus, LongMagpie, no_robots)...")
    blend_result = load_default_4_4_2_blended_dataset(
        split="train",
        seed=42,
        epoch_size=200_000,
        with_replacement=True,
    )

    print(
        "Blend counts: "
        f"{dict(zip(blend_result.source_names, blend_result.per_dataset_counts))} "
        f"(total={blend_result.total_samples})"
    )

    pec_ready_dataset = BlendedMessagesToPECSamples(blend_result.dataset)

    split_indices = split_dataset_indices(
        dataset_len=len(pec_ready_dataset),
        test_size=0.02,
        seed=42,
    )

    train_base = Subset(pec_ready_dataset, split_indices["train"].indices)
    eval_base = Subset(pec_ready_dataset, split_indices["eval"].indices)

    # Optional: Persist processed datasets for reproducibility and reuse.
    processed_data_dir = base_dir / "data" / "processed_pec_dataset"
    save_blend_metadata(blend_result, processed_data_dir / "blend_metadata.json")
    saved_per_source = save_sampled_by_source_as_jsonl(
        blend_result.dataset,
        processed_data_dir / "blended_messages_sampled.jsonl",
        source_names=blend_result.source_names,
        samples_per_source=3,
        seed=42,
    )

    saved_pec_ready = save_dataset_as_jsonl(
        pec_ready_dataset,
        processed_data_dir / "blended_messages_pec_ready.jsonl",
    )

    print(
        "Saved sampled conversion-check dataset: "
        f"counts={saved_per_source} "
        f"to {processed_data_dir / 'blended_messages_sampled.jsonl'}"
    )
    print(
        "Saved BlendedMessagesToPECSamples result: "
        f"count={saved_pec_ready} "
        f"to {processed_data_dir / 'blended_messages_pec_ready.jsonl'}"
    )


    # 4. Initialize Dynamic Datasets (The Core of Information Gap Training)
    # We wrap the HF dataset with our custom PECDataset to enable on-the-fly masking.

    # Train set: Apply masking (30%) to force latent usage.
    train_query_masker = EntityMasker(mask_prob=0.3)
    train_dataset = PECDataset(
        data=train_base,
        query_masker=train_query_masker,
        composer_tokenizer=composer_tokenizer,
        composer_enable_thinking=False,
    )

    # Eval set: No masking (or low masking) to evaluate true generation capability.
    eval_masker = EntityMasker(mask_prob=0.15)

    eval_dataset = PECDataset(
        data=eval_base,
        query_masker=eval_masker,
        composer_tokenizer=composer_tokenizer,
        composer_enable_thinking=False,
    )

    # 5. Initialize PEC Engine
    print("Initializing PEC Engine (Profiler-Extruder-Composer)...")
    model = PECEngine(
        profiler_path=str(profiler_model_path),
        composer_path=composer_model_name
    )

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
        num_train_epochs=2,

        # Batch size configuration
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        # Optimization
        learning_rate=5e-4,  # Base learning rate (will be overridden by param groups)
        max_grad_norm=1.0,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        # Mixed Precision
        fp16=False if get_device() == torch.device("cuda") else True,
        bf16=True if get_device() == torch.device("cuda") else False,
        tf32=True if get_device() == torch.device("cuda") else False,

        # Logging and Saving
        logging_steps=10,
        report_to="none",
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",

        # Dataloader
        dataloader_num_workers=8 if get_device() == torch.device("cuda") else 0,
        dataloader_pin_memory=True if get_device() == torch.device("cuda") else False,
        remove_unused_columns=False,  # Important for custom datasets

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
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    # 9. Initialize Trainer
    trainer = GateL1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        gate_l1_max_lambda=1e-3,
        gate_l1_warmup_ratio=0.1,

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
