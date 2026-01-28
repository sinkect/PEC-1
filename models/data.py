import torch
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class PECDataCollator:
    profiler_tokenizer: Any
    composer_tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        # 1. Pad Profiler Inputs
        prof_inputs = [
            {"input_ids": f["profiler_input_ids"], "attention_mask": f["profiler_attention_mask"]}
            for f in features
        ]
        prof_batch = self.profiler_tokenizer.pad(
            prof_inputs,
            padding=True,
            return_tensors="pt"
        )
        prompt_inputs = [{"input_ids": f["qwen_prompt_ids"]} for f in features]
        prompt_batch = self.composer_tokenizer.pad(
            prompt_inputs,
            padding=True,
            return_tensors="pt"
        )
        # 2. Pad Labels
        label_inputs = [{"input_ids": f["labels"]} for f in features]
        label_batch = self.composer_tokenizer.pad(
            label_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Mask padding tokens with -100 for loss calculation
        labels = label_batch["input_ids"]

        # 4. Construct Batch
        batch["profiler_input_ids"] = prof_batch["input_ids"]
        batch["profiler_attention_mask"] = prof_batch["attention_mask"]
        batch["qwen_prompt_ids"] = prompt_batch["input_ids"]
        batch["labels"] = labels

        return batch