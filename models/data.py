import multiprocessing as mp
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

try:
    from transformers import PreTrainedTokenizer
except ModuleNotFoundError:  # Optional for lightweight tests/utilities.
    PreTrainedTokenizer = Any

from .rationale import char_spans_to_token_spans, normalize_relevance_distribution




PromptMasker = Callable[..., str]


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
    """Applies sample-aware masking, with span-priority masking for MoreHopQA."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        mask_token: str = "[MASK]",
        mask_prob: float = 0.4,
        shared_mask_prob: Optional[SharedMaskProbability] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        target_span_mask_prob: Optional[float] = None,
        morehop_base_mask_prob: float = 0.1,
        morehop_earlier_target_span_mask_prob: Optional[float] = None,
        morehop_last_target_span_mask_prob: Optional[float] = None,
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
        self.tokenizer = tokenizer
        self.morehop_base_mask_prob = float(morehop_base_mask_prob)
        uniform_target_span_mask_prob = 0.5 if target_span_mask_prob is None else float(target_span_mask_prob)
        self.target_span_mask_prob = uniform_target_span_mask_prob
        self.morehop_earlier_target_span_mask_prob = (
            uniform_target_span_mask_prob
            if morehop_earlier_target_span_mask_prob is None
            else float(morehop_earlier_target_span_mask_prob)
        )
        self.morehop_last_target_span_mask_prob = (
            uniform_target_span_mask_prob
            if morehop_last_target_span_mask_prob is None
            else float(morehop_last_target_span_mask_prob)
        )

    def get_mask_prob(self) -> float:
        if self.shared_mask_prob is not None:
            return self.shared_mask_prob.get()
        return float(self.mask_prob)

    def set_mask_prob(self, value: float) -> None:
        self.mask_prob = float(value)
        if self.shared_mask_prob is not None:
            self.shared_mask_prob.set(value)

    @staticmethod
    def _find_all_substring_spans(text: str, substring: str) -> List[Tuple[int, int]]:
        if not text or not substring:
            return []

        spans: List[Tuple[int, int]] = []
        search_start = 0
        while True:
            match_index = text.find(substring, search_start)
            if match_index < 0:
                break
            spans.append((match_index, match_index + len(substring)))
            search_start = match_index + len(substring)
        return spans

    @staticmethod
    def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        merged: List[Tuple[int, int]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
                continue
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    @staticmethod
    def _morehop_target_span_mask_probability(
        target_index: int,
        num_targets: int,
        *,
        earlier_prob: float,
        last_prob: float,
    ) -> float:
        if num_targets <= 1:
            return float(last_prob)
        progress = min(max(float(target_index) / float(num_targets - 1), 0.0), 1.0)
        return float(earlier_prob + ((last_prob - earlier_prob) * progress))

    def _mask_with_entity_policy(self, text: str) -> str:
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

    def _resolve_morehop_char_spans(
        self,
        text: str,
        target_texts: Sequence[str] | None,
    ) -> List[Tuple[int, int, int]]:
        if not target_texts:
            return []

        spans: List[Tuple[int, int, int]] = []
        for target_index, target_text in enumerate(target_texts):
            if not target_text:
                continue
            spans.extend(
                (span_start, span_end, target_index)
                for span_start, span_end in self._find_all_substring_spans(text, str(target_text))
            )
        return sorted(spans, key=lambda item: (item[0], item[1], item[2]))

    def _tokenize_with_offsets(self, text: str) -> Optional[List[Tuple[int, int]]]:
        if self.tokenizer is None:
            return None

        try:
            tokenized = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (TypeError, ValueError):
            return None

        offset_mapping = tokenized.get("offset_mapping")
        if offset_mapping is None:
            return None
        return [tuple(offset) for offset in offset_mapping]

    def _mask_morehop_text(
        self,
        text: str,
        *,
        target_texts: Sequence[str] | None,
    ) -> str:
        # Convert the prompt into token offsets so we can map answer-string matches
        # from character spans to token spans before masking.
        offset_mapping = self._tokenize_with_offsets(text)
        if not offset_mapping:
            return self._mask_with_entity_policy(text)

        normalized_target_texts = [
            str(target_text).strip()
            for target_text in (target_texts or [])
            if target_text is not None and str(target_text).strip()
        ]
        num_targets = len(normalized_target_texts)
        if num_targets == 0:
            return self._mask_with_entity_policy(text)

        # Find every occurrence of the grounded support answers in the raw prompt.
        char_spans = self._resolve_morehop_char_spans(text, normalized_target_texts)
        token_target_indices: List[Optional[int]] = []
        for token_start, token_end in offset_mapping:
            if token_end <= token_start:
                token_target_indices.append(None)
                continue
            overlapping_targets = [
                target_index
                for span_start, span_end, target_index in char_spans
                if token_start < span_end and token_end > span_start
            ]
            token_target_indices.append(max(overlapping_targets) if overlapping_targets else None)

        # Collapse adjacent target tokens into contiguous ranges so each support answer
        # span is masked as a unit instead of token-by-token.
        target_ranges: Dict[int, Tuple[int, int]] = {}
        token_index = 0
        while token_index < len(token_target_indices):
            target_index = token_target_indices[token_index]
            if target_index is None:
                token_index += 1
                continue
            range_start = token_index
            token_index += 1
            while (
                token_index < len(token_target_indices)
                and token_target_indices[token_index] == target_index
            ):
                token_index += 1
            target_ranges[range_start] = (token_index, target_index)

        pieces: List[str] = []
        previous_end = 0
        # MoreHopQA uses a dedicated low-probability background masking rate for
        # non-target tokens so masking stays concentrated on grounded sub-answer spans.
        base_mask_prob = self.morehop_base_mask_prob
        token_index = 0
        while token_index < len(offset_mapping):
            token_start, token_end = offset_mapping[token_index]
            if token_end <= token_start:
                token_index += 1
                continue

            if token_start > previous_end:
                pieces.append(text[previous_end:token_start])

            span_info = target_ranges.get(token_index)
            if span_info is not None:
                # Target spans get the higher masking probability to bias the visible
                # prompt toward hiding later grounded answers more aggressively.
                span_end_index, target_index = span_info
                span_end = offset_mapping[span_end_index - 1][1]
                span_mask_prob = self._morehop_target_span_mask_probability(
                    target_index,
                    num_targets,
                    earlier_prob=self.morehop_earlier_target_span_mask_prob,
                    last_prob=self.morehop_last_target_span_mask_prob,
                )
                if span_mask_prob >= 1.0:
                    pieces.append(self.mask_token)
                elif span_mask_prob <= 0.0:
                    pieces.append(text[token_start:span_end])
                elif random.random() < span_mask_prob:
                    pieces.append(self.mask_token)
                else:
                    pieces.append(text[token_start:span_end])
                previous_end = span_end
                token_index = span_end_index
                continue

            # Non-target tokens still receive low-probability random masking so the
            # model does not overfit to only one masking pattern.
            if base_mask_prob >= 1.0:
                pieces.append(self.mask_token)
            elif base_mask_prob <= 0.0:
                pieces.append(text[token_start:token_end])
            elif random.random() < base_mask_prob:
                pieces.append(self.mask_token)
            else:
                pieces.append(text[token_start:token_end])
            previous_end = token_end
            token_index += 1

        if previous_end < len(text):
            pieces.append(text[previous_end:])
        return "".join(pieces)

    def __call__(self, text: str, *, item: Optional[Dict[str, Any]] = None) -> str:
        if item is not None and item.get("task_type") == "morehopqa":
            return self._mask_morehop_text(
                text,
                target_texts=item.get("mh_target_texts"),
            )
        return self._mask_with_entity_policy(text)


class PECDataset(Dataset):
    """Dataset wrapper for stage-aware PEC training."""

    def __init__(
            self,
            data: List[Dict[str, Any]],
            query_masker: Optional[PromptMasker],
            composer_tokenizer: Optional[PreTrainedTokenizer] = None,
            composer_enable_thinking: bool = False,
            visible_prompt_mode: str = "masked",
            teacher_rationale_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.data = data
        self.query_masker = query_masker
        self.composer_tokenizer = composer_tokenizer
        self.composer_enable_thinking = composer_enable_thinking
        self.visible_prompt_mode = visible_prompt_mode
        self.teacher_rationale_lookup = teacher_rationale_lookup or {}
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
        item: Optional[Dict[str, Any]] = None,
    ) -> str:
        if mode == "full":
            return prompt
        if mode == "empty":
            return ""
        if mode == "masked":
            if masker is None:
                raise ValueError("query_masker is required when visible_prompt_mode='masked'.")
            try:
                return masker(prompt, item=item)
            except TypeError:
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        prompt = str(item.get("prompt", item.get("input", ""))).strip()
        answer = str(item.get("answer", "")).strip()

        visible_prompt = self._build_visible_prompt(
            prompt,
            mode=self.visible_prompt_mode,
            masker=self.query_masker,
            item=item,
        )
        composer_texts = self._build_composer_texts(visible_prompt, answer)

        result = {
            "profiler_input_text": prompt,
            "composer_prompt_text": composer_texts["prompt_text"],
            "composer_full_text": composer_texts["full_text"],
            "composer_visible_prompt_text": composer_texts["visible_prompt_text"],
        }
        for key in (
            "example_id",
            "task_type",
            "source",
            "mh_target_texts",
        ):
            if key in item:
                result[key] = item[key]
        example_id = result.get("example_id")
        if example_id is not None:
            rationale_record = self.teacher_rationale_lookup.get(str(example_id))
            if rationale_record is not None:
                result["span_boundaries"] = rationale_record.get("span_boundaries", [])
                result["teacher_relevance"] = rationale_record.get("teacher_relevance", [])
        return result


class PECCollator:
    """Tokenizes stage-aware PEC batches."""

    def __init__(
            self,
            profiler_tokenizer: PreTrainedTokenizer,
            composer_tokenizer: PreTrainedTokenizer,
            max_profiler_len: int = 6144,
            max_composer_len: int = 6144
    ):
        self.profiler_tokenizer = profiler_tokenizer
        self.composer_tokenizer = composer_tokenizer
        self.max_profiler_len = max_profiler_len
        self.max_composer_len = max_composer_len

    def _tokenize_texts_with_optional_offsets(
        self,
        tokenizer: PreTrainedTokenizer,
        texts: Sequence[str],
        *,
        max_length: int,
        include_offsets: bool,
    ) -> Dict[str, Any]:
        common_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": max_length,
        }
        if include_offsets:
            try:
                return tokenizer(
                    list(texts),
                    return_offsets_mapping=True,
                    **common_kwargs,
                )
            except (TypeError, ValueError):
                pass
        return tokenizer(list(texts), **common_kwargs)

    def _build_morehop_target_batches(
        self,
        *,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, List[torch.Tensor]]:
        target_text_lists: List[List[str]] = []
        max_targets = 0
        for item in batch:
            raw_targets = item.get("mh_target_texts")
            if isinstance(raw_targets, (list, tuple)):
                normalized_targets = []
                for target in raw_targets:
                    if target is None:
                        continue
                    target_text = str(target).strip()
                    if target_text:
                        normalized_targets.append(target_text)
            else:
                normalized_targets = []
            target_text_lists.append(normalized_targets)
            max_targets = max(max_targets, len(normalized_targets))

        if max_targets == 0:
            return {}

        pad_token_id = self.profiler_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        mh_target_input_ids_list: List[torch.Tensor] = []
        mh_target_attention_mask_list: List[torch.Tensor] = []
        for target_index in range(max_targets):
            target_texts: List[str] = []
            active_rows = torch.zeros(len(batch), dtype=torch.bool)

            for sample_index, targets in enumerate(target_text_lists):
                if target_index < len(targets):
                    target_texts.append(targets[target_index])
                    active_rows[sample_index] = True
                else:
                    target_texts.append("")

            target_inputs = self._tokenize_texts_with_optional_offsets(
                self.profiler_tokenizer,
                target_texts,
                max_length=self.max_profiler_len,
                include_offsets=False,
            )
            input_ids = target_inputs["input_ids"].clone()
            attention_mask = target_inputs["attention_mask"].clone()
            if torch.any(~active_rows):
                input_ids[~active_rows] = pad_token_id
                attention_mask[~active_rows] = 0

            mh_target_input_ids_list.append(input_ids)
            mh_target_attention_mask_list.append(attention_mask)

        return {
            "mh_target_input_ids_list": mh_target_input_ids_list,
            "mh_target_attention_mask_list": mh_target_attention_mask_list,
        }

    def _build_rationale_batch(
        self,
        *,
        batch: List[Dict[str, Any]],
        profiler_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not any("teacher_relevance" in item and "span_boundaries" in item for item in batch):
            return {}

        offset_mapping = profiler_inputs.get("offset_mapping")
        if offset_mapping is None:
            return {}

        attention_mask = profiler_inputs["attention_mask"]
        raw_span_boundaries: List[List[List[int]]] = []
        per_sample_token_spans: List[List[Tuple[int, int, float]]] = []
        max_spans = 0
        for sample_index, item in enumerate(batch):
            sample_span_boundaries = [
                [int(boundary[0]), int(boundary[1])]
                for boundary in item.get("span_boundaries", [])
                if isinstance(boundary, (list, tuple)) and len(boundary) == 2
            ]
            raw_span_boundaries.append(sample_span_boundaries)

            sample_teacher_relevance = item.get("teacher_relevance", [])
            if not sample_span_boundaries or not sample_teacher_relevance:
                per_sample_token_spans.append([])
                continue

            valid_token_count = int(attention_mask[sample_index].sum().item())
            mapped_token_spans = char_spans_to_token_spans(
                offset_mapping=offset_mapping[sample_index],
                span_boundaries=sample_span_boundaries,
                valid_token_count=valid_token_count,
            )
            if not mapped_token_spans:
                per_sample_token_spans.append([])
                continue

            selected_scores = [
                float(sample_teacher_relevance[span_index])
                for span_index, _, _ in mapped_token_spans
                if span_index < len(sample_teacher_relevance)
            ]
            if not selected_scores:
                per_sample_token_spans.append([])
                continue

            normalized_scores = normalize_relevance_distribution(selected_scores)
            token_spans_with_scores = [
                (token_start, token_end, normalized_scores[score_index])
                for score_index, (_, token_start, token_end) in enumerate(mapped_token_spans)
            ]
            per_sample_token_spans.append(token_spans_with_scores)
            max_spans = max(max_spans, len(token_spans_with_scores))

        if max_spans <= 0:
            return {}

        span_token_boundaries = torch.full(
            (len(batch), max_spans, 2),
            fill_value=-1,
            dtype=torch.long,
        )
        span_mask = torch.zeros((len(batch), max_spans), dtype=torch.bool)
        teacher_relevance = torch.zeros((len(batch), max_spans), dtype=torch.float32)

        for sample_index, token_spans_with_scores in enumerate(per_sample_token_spans):
            for span_index, (token_start, token_end, relevance_score) in enumerate(token_spans_with_scores):
                span_token_boundaries[sample_index, span_index, 0] = int(token_start)
                span_token_boundaries[sample_index, span_index, 1] = int(token_end)
                span_mask[sample_index, span_index] = True
                teacher_relevance[sample_index, span_index] = float(relevance_score)

        return {
            "example_ids": [str(item.get("example_id", "")) for item in batch],
            "span_boundaries": raw_span_boundaries,
            "span_token_boundaries": span_token_boundaries,
            "span_mask": span_mask,
            "teacher_relevance": teacher_relevance,
        }

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
        prompt_inputs = self.composer_tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_composer_len,
        )
        prompt_attention_mask = prompt_inputs.get("attention_mask")

        for index, prompt in enumerate(prompt_texts):
            del prompt
            if prompt_attention_mask is not None:
                prompt_len = int(prompt_attention_mask[index].sum().item())
            else:
                prompt_tokens = self.composer_tokenizer(
                    prompt_texts[index],
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_composer_len,
                )["input_ids"]
                prompt_len = len(prompt_tokens)
            prompt_len = min(prompt_len, labels.shape[1])
            labels[index, :prompt_len] = -100

        labels[input_ids == self.composer_tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        profiler_texts = [item["profiler_input_text"] for item in batch]
        profiler_inputs = self._tokenize_texts_with_optional_offsets(
            self.profiler_tokenizer,
            profiler_texts,
            max_length=self.max_profiler_len,
            include_offsets=any("span_boundaries" in item for item in batch),
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
        collated.update(self._build_morehop_target_batches(batch=batch))
        collated.update(self._build_rationale_batch(batch=batch, profiler_inputs=profiler_inputs))
        return collated
