import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .bridge import Extruder


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PECEngine(nn.Module):
    """
    PEC Engine that composes ModernBERT (Profiler) with Qwen 3 (Composer).
    Shapes: [B, T, D]
    """

    def __init__(
        self,
        profiler_path="answerdotai/ModernBERT-base",
        composer_path="Qwen/Qwen3-Instruct",
        num_query_tokens=64,
        freeze_profiler=True,
        freeze_composer=True,
    ):
        super().__init__()
        self.device = get_device()

        self.profiler = AutoModel.from_pretrained(profiler_path)
        self.composer = AutoModel.from_pretrained(composer_path)

        if freeze_profiler:
            for param in self.profiler.parameters():
                param.requires_grad = False
        if freeze_composer:
            for param in self.composer.parameters():
                param.requires_grad = False

        self.profiler_tokenizer = AutoTokenizer.from_pretrained(profiler_path, use_fast=True)
        self.composer_tokenizer = AutoTokenizer.from_pretrained(composer_path, use_fast=True)

        self.extruder = Extruder(
            hidden_size=self.profiler.config.hidden_size,
            num_query_tokens=num_query_tokens,
        )

        self.sep_token = nn.Parameter(
            torch.randn(1, 1, self.profiler.config.hidden_size, dtype=torch.float16)
        )

        self.to(self.device, dtype=torch.float16)

    def forward(self, profiler_input_ids, profiler_attention_mask, composer_input_ids, composer_attention_mask):
        """
        profiler_input_ids: [B, T]
        profiler_attention_mask: [B, T]
        composer_input_ids: [B, T]
        composer_attention_mask: [B, T]
        """
        profiler_input_ids = profiler_input_ids.to(self.device)
        profiler_attention_mask = profiler_attention_mask.to(self.device)
        composer_input_ids = composer_input_ids.to(self.device)
        composer_attention_mask = composer_attention_mask.to(self.device)

        profiler_outputs = self.profiler(
            input_ids=profiler_input_ids,
            attention_mask=profiler_attention_mask,
        )
        profiler_hidden = profiler_outputs.last_hidden_state

        extruded = self.extruder(profiler_hidden)

        sep_token = self.sep_token.to(device=self.device, dtype=extruded.dtype)
        sep_token = sep_token.repeat(extruded.size(0), 1, 1)

        # Concatenate [B, Q, D] + [B, 1, D]
        composer_prefix = torch.cat([extruded, sep_token], dim=1)

        composer_outputs = self.composer(
            input_ids=composer_input_ids,
            attention_mask=composer_attention_mask,
        )
        composer_hidden = composer_outputs.last_hidden_state

        return composer_prefix, composer_hidden
