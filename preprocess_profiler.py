import os
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer

# === Configuration ===
MODEL_ID = "answerdotai/ModernBERT-base"
# ModernBERT supports 8192, but for Phase 1 on limited hardware,
# 4096 or 8192 is fine. Let's stick to 8192 as planned.
MAX_LENGTH = 8192
SAVE_PATH = "data/processed_profiler_dataset"


def main():
    print(f"🔄 Loading Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.model_max_length = MAX_LENGTH

    print("📚 Loading Dataset: HuggingFaceFW/fineweb-edu (Parquet)...")
    # Use 'sample-10BT' subset for quicker download/processing
    # Streaming=True allows processing without downloading everything first
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

    # Take top 10,000 documents (Approx 100MB~ data) for Phase 1
    # You can increase this if you want longer training
    dataset = dataset.take(10000)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True
        )

    print("⚙️ Tokenizing & Chunking...")
    # Since we are streaming, we cannot use 'num_proc' in map directly easily with saving.
    # We will iterate and process.
    # BUT: 'dataset.map' with streaming works differently.

    # Efficient pipeline for streaming dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True,
                                    remove_columns=["text", "id", "dump", "url", "file_path", "language",
                                                    "language_score", "token_count"])

    print(f"💾 Saving processed dataset to {SAVE_PATH}...")
    # Saving streaming dataset to disk requires iterating
    # This might take a moment, but it's safe.

    # To save properly as an Arrow dataset for the Trainer:
    # We generator-save it.
    import datasets
    def gen():
        for ex in tokenized_dataset:
            yield ex

    # Re-create a static dataset from generator to save to disk safely
    final_dataset = datasets.Dataset.from_generator(gen)
    final_dataset.save_to_disk(SAVE_PATH)

    print("✅ Preprocessing Complete!")


if __name__ == "__main__":
    main()