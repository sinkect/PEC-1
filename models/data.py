import multiprocessing as mp
import random
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer




PromptMasker = Callable[[str], str]


class SharedMaskProbability:
    """Cross-process mutable mask probability for curriculum masking."""

    def __init__(self, initial_value: float):
        self._value = mp.Value("d", float(initial_value))

    def get(self) -> float:
        with self._value.get_lock():
            return float(self._value.value)

    def set(self, value: float) -> None:
        with self._value.get_lock():
            self._value.value = float(value)


class EntityMasker:
    """Applies selective masking to text, targeting content-bearing tokens."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        mask_token: str = "[MASK]",
        mask_prob: float = 0.4,
        shared_mask_prob: Optional[SharedMaskProbability] = None,
    ):
        try:
            import spacy
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("spaCy is required to use EntityMasker.") from exc

        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise OSError(f"Spacy model '{model_name}' not found. Please download it.") from exc

        self.mask_token = mask_token
        self.mask_prob = mask_prob
        self.shared_mask_prob = shared_mask_prob

    def get_mask_prob(self) -> float:
        if self.shared_mask_prob is not None:
            return self.shared_mask_prob.get()
        return float(self.mask_prob)

    def set_mask_prob(self, value: float) -> None:
        self.mask_prob = float(value)
        if self.shared_mask_prob is not None:
            self.shared_mask_prob.set(value)

    def __call__(self, text: str) -> str:
        doc = self.nlp(text)
        mask_prob = self.get_mask_prob()
        tokens = []
        for token in doc:
            is_candidate = token.pos_ in ["PROPN", "NUM", "NOUN"]
            if is_candidate and random.random() < mask_prob:
                tokens.append(self.mask_token)
            else:
                tokens.append(token.text)
        return " ".join(tokens)


class PECDataset(Dataset):
    """Dataset wrapper for stage-aware PEC training."""

    def __init__(
            self,
            data: List[Dict[str, Any]],
            query_masker: Optional[PromptMasker],
            composer_tokenizer: Optional[PreTrainedTokenizer] = None,
            composer_enable_thinking: bool = False,
            visible_prompt_mode: str = "masked",
            include_teacher: bool = False,
            teacher_visible_prompt_mode: str = "full",
    ):
        self.data = data
        self.query_masker = query_masker
        self.composer_tokenizer = composer_tokenizer
        self.composer_enable_thinking = composer_enable_thinking
        self.visible_prompt_mode = visible_prompt_mode
        self.include_teacher = include_teacher
        self.teacher_visible_prompt_mode = teacher_visible_prompt_mode
        self.eos_token = "<|im_end|>"

    def _render_qwen_chat(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        if self.composer_tokenizer is None:
            raise ValueError("composer_tokenizer is required to render Qwen chat template.")

        return self.composer_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.composer_enable_thinking,
        )

    def _build_visible_prompt(
        self,
        prompt: str,
        *,
        mode: str,
        masker: Optional[PromptMasker],
    ) -> str:
        if mode == "full":
            return prompt
        if mode == "empty":
            return ""
        if mode == "masked":
            if masker is None:
                raise ValueError("query_masker is required when visible_prompt_mode='masked'.")
            return masker(prompt)
        raise ValueError(f"Unsupported visible_prompt_mode: {mode}")

    def _build_composer_texts(self, visible_prompt: str, answer: str) -> Dict[str, str]:
        if self.composer_tokenizer is not None:
            prompt_messages = [{"role": "user", "content": visible_prompt}]
            full_messages = [
                {"role": "user", "content": visible_prompt},
                {"role": "assistant", "content": answer},
            ]
            composer_prompt_text = self._render_qwen_chat(prompt_messages, add_generation_prompt=True)
            composer_full_text = self._render_qwen_chat(full_messages, add_generation_prompt=False)
        else:
            composer_prompt_text = f"{visible_prompt}\n"
            composer_full_text = composer_prompt_text + answer + self.eos_token

        return {
            "prompt_text": composer_prompt_text,
            "full_text": composer_full_text,
            "visible_prompt_text": visible_prompt,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]
        prompt = str(item.get("prompt", item.get("input", ""))).strip()
        answer = str(item.get("answer", "")).strip()

        visible_prompt = self._build_visible_prompt(
            prompt,
            mode=self.visible_prompt_mode,
            masker=self.query_masker,
        )
        composer_texts = self._build_composer_texts(visible_prompt, answer)

        result = {
            "profiler_input_text": prompt,
            "composer_prompt_text": composer_texts["prompt_text"],
            "composer_full_text": composer_texts["full_text"],
            "composer_visible_prompt_text": composer_texts["visible_prompt_text"],
        }

        if self.include_teacher:
            teacher_visible_prompt = self._build_visible_prompt(
                prompt,
                mode=self.teacher_visible_prompt_mode,
                masker=None,
            )
            teacher_texts = self._build_composer_texts(teacher_visible_prompt, answer)
            result.update(
                {
                    "teacher_prompt_text": teacher_texts["prompt_text"],
                    "teacher_full_text": teacher_texts["full_text"],
                }
            )

        return result


class PECCollator:
    """Tokenizes stage-aware PEC batches for student and optional teacher."""

    def __init__(
            self,
            profiler_tokenizer: PreTrainedTokenizer,
            composer_tokenizer: PreTrainedTokenizer,
            max_profiler_len: int = 6080,
            max_composer_len: int = 6080
    ):
        self.profiler_tokenizer = profiler_tokenizer
        self.composer_tokenizer = composer_tokenizer
        self.max_profiler_len = max_profiler_len
        self.max_composer_len = max_composer_len

    def _tokenize_composer_texts(
        self,
        full_texts: List[str],
        prompt_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        composer_inputs = self.composer_tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_composer_len,
        )

        input_ids = composer_inputs["input_ids"]
        attention_mask = composer_inputs["attention_mask"]
        labels = input_ids.clone()

        for index, prompt in enumerate(prompt_texts):
            prompt_tokens = self.composer_tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_tokens)
            if prompt_len < labels.shape[1]:
                labels[index, :prompt_len] = -100

        labels[input_ids == self.composer_tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        profiler_texts = [item["profiler_input_text"] for item in batch]
        profiler_inputs = self.profiler_tokenizer(
            profiler_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_profiler_len,
        )

        composer_batch = self._tokenize_composer_texts(
            full_texts=[item["composer_full_text"] for item in batch],
            prompt_texts=[item["composer_prompt_text"] for item in batch],
        )

        collated = {
            "profiler_input_ids": profiler_inputs["input_ids"],
            "profiler_attention_mask": profiler_inputs["attention_mask"],
            "composer_input_ids": composer_batch["input_ids"],
            "composer_attention_mask": composer_batch["attention_mask"],
            "labels": composer_batch["labels"],
        }

        if "teacher_full_text" in batch[0]:
            teacher_batch = self._tokenize_composer_texts(
                full_texts=[item["teacher_full_text"] for item in batch],
                prompt_texts=[item["teacher_prompt_text"] for item in batch],
            )
            collated.update(
                {
                    "teacher_input_ids": teacher_batch["input_ids"],
                    "teacher_attention_mask": teacher_batch["attention_mask"],
                    "teacher_labels": teacher_batch["labels"],
                }
            )

        return collated
