import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

# === Configuration for NVIDIA H200 ===
MODEL_ID = "answerdotai/ModernBERT-base"
DATA_PATH = "data/processed_profiler_dataset"
OUTPUT_DIR = "output/profiler"

# === H200 Optimization ===
# H200 (141GB VRAM) allows massive batches.
# 8k context is huge, but H200 can handle it.
BATCH_SIZE = 16  # Try 16 or 32. If OOM, reduce to 8.
GRAD_ACCUMULATION = 1  # No need to accumulate much with big batch
LEARNING_RATE = 5e-5  # Slightly higher LR for larger batch
NUM_EPOCHS = 3

# SimCSE Settings
USE_SIMCSE = True
SIMCSE_WEIGHT = 0.05
TEMP = 0.05


class ProfilerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        mlm_loss = outputs.loss
        total_loss = mlm_loss

        if USE_SIMCSE:
            # Pass 1
            out1 = model.base_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            # Pass 2
            out2 = model.base_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # Mean Pooling
            z1 = self._mean_pooling(out1.last_hidden_state, inputs['attention_mask'])
            z2 = self._mean_pooling(out2.last_hidden_state, inputs['attention_mask'])

            # Cosine Sim
            cos_sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2) / TEMP
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            simcse_loss = F.cross_entropy(cos_sim, labels)

            total_loss += (SIMCSE_WEIGHT * simcse_loss)

        return (total_loss, outputs) if return_outputs else total_loss

    def _mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    print(f"🚀 Running on NVIDIA H200 Setup")

    # 1. Load Data
    dataset = load_from_disk(DATA_PATH)

    # 2. Load Model (Enable Flash Attention 2)
    print(f"🔄 Loading Model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID
    )
    model.config.max_position_embeddings = 8192

    # Dropouts for SimCSE
    if USE_SIMCSE:
        model.config.attention_dropout = 0.1
        model.config.mlp_dropout = 0.1

    # 3. Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.30
    )

    # 4. Training Arguments (H200 Optimized)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,

        # --- GPU Optimization ---
        fp16=True,
        tf32=True,
        bf16=False,  # Enable TF32 for matmul

        gradient_checkpointing=True,  # Optional on H200 depending on batch size
        logging_steps=10,
        save_strategy="epoch",
        dataloader_num_workers=16,  # Linux supports multi-workers!
        report_to="none"
    )

    trainer = ProfilerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Training Complete on H200!")


if __name__ == "__main__":
    main()