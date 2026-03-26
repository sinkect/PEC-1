from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .bridge import Extruder, MemoryCompressor


def _resolve_query_positions(
        *,
        cache_position: torch.LongTensor | None,
        query_length: int,
        token_kv_length: int,
        device: torch.device,
) -> torch.Tensor:
    if cache_position is None:
        start = max(token_kv_length - query_length, 0)
        return torch.arange(start, start + query_length, device=device)

    if cache_position.ndim > 1:
        cache_position = cache_position[0]
    return cache_position.to(device=device)


def _build_causal_keep_mask(
        *,
        batch_size: int,
        query_length: int,
        token_kv_length: int,
        cache_position: torch.LongTensor | None,
        device: torch.device,
) -> torch.Tensor:
    query_positions = _resolve_query_positions(
        cache_position=cache_position,
        query_length=query_length,
        token_kv_length=token_kv_length,
        device=device,
    )  # [Tq]
    key_positions = torch.arange(token_kv_length, device=device)  # [Tk]
    keep_mask = key_positions.view(1, 1, 1, token_kv_length) <= query_positions.view(1, 1, query_length, 1)
    return keep_mask.expand(batch_size, 1, query_length, token_kv_length)  # [B, 1, Tq, Tk]


def _build_memory_attention_mask(
        *,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        query_length: int,
        token_kv_length: int,
        memory_kv_length: int,
        cache_position: torch.LongTensor | None,
        device: torch.device,
) -> torch.Tensor:
    query_positions = _resolve_query_positions(
        cache_position=cache_position,
        query_length=query_length,
        token_kv_length=token_kv_length,
        device=device,
    )  # [Tq]

    if attention_mask is None:
        key_positions = torch.arange(token_kv_length, device=device)  # [Tk]
        token_mask = key_positions.view(1, 1, 1, token_kv_length) <= query_positions.view(1, 1, query_length, 1)
        token_mask = token_mask.expand(batch_size, 1, query_length, token_kv_length)
        memory_mask = torch.ones((batch_size, 1, query_length, memory_kv_length), dtype=torch.bool, device=device)
        return torch.cat([memory_mask, token_mask], dim=-1)  # [B, 1, Tq, M + Tk]

    if attention_mask.ndim == 2:
        token_padding_mask = attention_mask[:, None, None, :token_kv_length].to(device=device, dtype=torch.bool)
        key_positions = torch.arange(token_kv_length, device=device)  # [Tk]
        token_mask = key_positions.view(1, 1, 1, token_kv_length) <= query_positions.view(1, 1, query_length, 1)
        token_mask = token_mask.expand(batch_size, 1, query_length, token_kv_length)
        memory_mask = torch.ones((batch_size, 1, query_length, memory_kv_length), dtype=torch.bool, device=device)
        return torch.cat([memory_mask, token_mask & token_padding_mask], dim=-1)  # [B, 1, Tq, M + Tk]

    if attention_mask.ndim != 4:
        raise ValueError(f"Unsupported attention_mask shape for memory attention: {tuple(attention_mask.shape)}")

    token_attention_mask = attention_mask[..., :token_kv_length].to(device=device)  # [B, 1, Tq, Tk]
    if token_attention_mask.dtype == torch.bool:
        memory_mask = torch.ones(
            (*token_attention_mask.shape[:-1], memory_kv_length),
            dtype=torch.bool,
            device=device,
        )
    else:
        memory_mask = torch.zeros(
            (*token_attention_mask.shape[:-1], memory_kv_length),
            dtype=token_attention_mask.dtype,
            device=device,
        )
    return torch.cat([memory_mask, token_attention_mask], dim=-1)  # [B, 1, Tq, M + Tk]


def _expand_memory_batch(memory_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if memory_tensor.shape[0] == batch_size:
        return memory_tensor  # [B, Hkv, M, Dh]
    if memory_tensor.shape[0] == 1:
        return memory_tensor.expand(batch_size, -1, -1, -1)  # [B, Hkv, M, Dh]
    raise ValueError(
        "Memory KV batch size does not match decoder batch size: "
        f"{memory_tensor.shape[0]} vs {batch_size}"
    )


def _record_memory_attention_mass(
        self,
        *,
        full_attention_mask: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        memory_kv_length: int,
) -> None:
    memory_holder = getattr(self, "_pec_memory_state_holder", None)
    if memory_holder is None or not bool(getattr(memory_holder, "capture_attention_mass", False)):
        return

    with torch.no_grad():
        attention_scores = torch.matmul(
            query_states.float(),
            key_states.float().transpose(-1, -2),
        ) * float(self.scaling)  # [B, H, Tq, Tk]

        attention_bias = full_attention_mask
        if attention_bias.shape[1] == 1 and attention_scores.shape[1] != 1:
            attention_bias = attention_bias.expand(-1, attention_scores.shape[1], -1, -1)
        if attention_bias.dtype == torch.bool:
            attention_scores = attention_scores.masked_fill(~attention_bias, float("-inf"))
        else:
            attention_scores = attention_scores + attention_bias.to(
                device=attention_scores.device,
                dtype=attention_scores.dtype,
            )

        attention_probs = torch.softmax(attention_scores, dim=-1)  # [B, H, Tq, Tk]
        memory_mass = attention_probs[..., :memory_kv_length].sum(dim=-1).mean(dim=1)  # [B, Tq]

    attention_mass_records = getattr(memory_holder, "attention_mass_records", None)
    if attention_mass_records is None:
        attention_mass_records = []
        memory_holder.attention_mass_records = attention_mass_records
    attention_mass_records.append(
        {
            "layer_idx": int(self.layer_idx),
            "mass": memory_mass.detach().cpu(),
        }
    )


def _qwen_memory_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
):
    memory_holder = getattr(self, "_pec_memory_state_holder", None)
    memory_bundle = getattr(memory_holder, "memory_kv", None)
    if memory_bundle is None:
        return self._pec_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Build token Q/K/V from decoder hidden states.  # [B, Tq, D] -> [B, H, Tq, Dh] / [B, Hkv, Tq, Dh]
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B, H, Tq, Dh]
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B, Hkv, Tq, Dh]
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, Hkv, Tq, Dh]

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # Cached token KV now covers the full decoded token prefix.  # [B, Hkv, Tk, Dh]

    batch_size = hidden_states.shape[0]
    memory_keys = _expand_memory_batch(memory_bundle["memory_keys"], batch_size).to(
        device=key_states.device,
        dtype=key_states.dtype,
    )  # [B, Hkv, M, Dh]
    memory_values = _expand_memory_batch(memory_bundle["memory_values"], batch_size).to(
        device=value_states.device,
        dtype=value_states.dtype,
    )  # [B, Hkv, M, Dh]

    # Concatenate latent memory KV before token KV.  # [B, Hkv, M + Tk, Dh]
    key_states = torch.cat([memory_keys, key_states], dim=2)
    value_states = torch.cat([memory_values, value_states], dim=2)

    full_attention_mask = _build_memory_attention_mask(
        attention_mask=attention_mask,
        batch_size=batch_size,
        query_length=query_states.shape[2],
        token_kv_length=key_states.shape[2] - memory_keys.shape[2],
        memory_kv_length=memory_keys.shape[2],
        cache_position=cache_position,
        device=query_states.device,
    )

    key_states = repeat_kv(key_states, self.num_key_value_groups)  # [B, H, M + Tk, Dh]
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # [B, H, M + Tk, Dh]
    _record_memory_attention_mass(
        self,
        full_attention_mask=full_attention_mask,
        query_states=query_states,
        key_states=key_states,
        memory_kv_length=memory_keys.shape[2],
    )
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=full_attention_mask,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        scale=self.scaling,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()  # [B, Tq, D]
    attn_output = self.o_proj(attn_output)  # [B, Tq, D]
    return attn_output, None


class MemoryRationaleHead(nn.Module):
    def __init__(
        self,
        *,
        memory_dim: int,
        span_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        projection_dim = int(hidden_dim) if hidden_dim is not None else max(int(memory_dim), int(span_dim))
        self.memory_proj = nn.Linear(memory_dim, projection_dim, bias=False)
        self.span_proj = nn.Linear(span_dim, projection_dim, bias=False)

    def forward(
        self,
        *,
        memory_slots: torch.Tensor,
        span_representations: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, num_spans = memory_slots.shape[0], memory_slots.shape[1], span_representations.shape[1]
        projected_memory = self.memory_proj(memory_slots).float()  # [B, S, Dr]
        projected_spans = self.span_proj(span_representations).float()  # [B, K, Dr]
        raw_scores = torch.einsum("bsd,bkd->bsk", projected_memory, projected_spans)  # [B, S, K]

        span_mask = span_mask.to(device=raw_scores.device, dtype=torch.bool)
        slot_distributions = raw_scores.new_zeros((batch_size, memory_slots.shape[1], num_spans))
        active_rows = torch.any(span_mask, dim=-1)
        if torch.any(active_rows):
            active_scores = raw_scores[active_rows].masked_fill(
                ~span_mask[active_rows].unsqueeze(1),
                torch.finfo(raw_scores.dtype).min,
            )
            active_distributions = torch.softmax(active_scores, dim=-1)
            active_distributions = active_distributions * span_mask[active_rows].unsqueeze(1).to(
                dtype=active_distributions.dtype
            )
            slot_distributions[active_rows] = active_distributions

        student_relevance = slot_distributions.mean(dim=1)  # [B, K]
        student_relevance = student_relevance * span_mask.to(dtype=student_relevance.dtype)
        normalization = student_relevance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        student_relevance = student_relevance / normalization
        return student_relevance, slot_distributions


class PECEngine(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(
            self,
            profiler_path="answerdotai/ModernBERT-base",
            composer_path="Qwen/Qwen3-1.7B",
            num_query_tokens=64,
            num_memory_slots: int | None = 8,
            attn_mix_alpha: float = 0.0,
            morehop_align_lambda: float = 0.1,
            rationale_lambda: float = 0.01,
            morehop_align_mode: str = "weighted",
            freeze_profiler=False,
            freeze_composer=True,
            freeze_extruder=False,
            memory_upper_layers: int = 8,
    ):
        super().__init__()

        self.profiler = (
            AutoModel.from_pretrained(
                profiler_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            if torch.cuda.is_available()
            else AutoModel.from_pretrained(profiler_path)
        )
        self.composer = (
            AutoModelForCausalLM.from_pretrained(
                composer_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(composer_path)
        self.prof_dim = self.profiler.config.hidden_size
        self.comp_dim = self.composer.config.hidden_size
        self.num_query_tokens = int(num_query_tokens)
        self.num_memory_slots = int(num_memory_slots) if num_memory_slots is not None else self.num_query_tokens
        self.attn_mix_alpha = float(attn_mix_alpha)
        self.morehop_align_lambda = float(morehop_align_lambda)
        self.rationale_lambda = float(rationale_lambda)
        if morehop_align_mode not in {"weighted", "last"}:
            raise ValueError(f"Unsupported MoreHopQA align mode: {morehop_align_mode}")
        self.morehop_align_mode = str(morehop_align_mode)
        self.memory_upper_layers = int(memory_upper_layers)

        if freeze_profiler:
            self.profiler.requires_grad_(False)
        if freeze_composer:
            self.composer.requires_grad_(False)

        self.extruder = Extruder(
            hidden_size=self.prof_dim,
            num_query_tokens=self.num_query_tokens,
        )
        if freeze_extruder:
            self.extruder.requires_grad_(False)

        self.post_extruder_norm = nn.RMSNorm(self.prof_dim, eps=1e-6)
        self.memory_num_key_value_heads, self.memory_head_dim = self._resolve_memory_projection_shape()
        memory_proj_dtype = self._composer_parameter_dtype()
        memory_proj_dim = self.memory_num_key_value_heads * self.memory_head_dim
        self.mem_proj = nn.Sequential(
            nn.Linear(self.prof_dim, self.comp_dim, bias=False),
            nn.RMSNorm(self.comp_dim),
        ).to(dtype=memory_proj_dtype)
        self.slot_proj = nn.Linear(self.num_query_tokens, self.num_memory_slots, bias=False).to(dtype=memory_proj_dtype)
        self.memory_compressor = MemoryCompressor(
            dim=self.comp_dim,
            num_memory_slots=self.num_memory_slots,
            num_heads=self._resolve_memory_compressor_num_heads(),
        ).to(dtype=memory_proj_dtype)
        self.k_mem_out_proj = nn.Sequential(
            nn.Linear(self.comp_dim, memory_proj_dim, bias=False),
            nn.RMSNorm(memory_proj_dim),
        ).to(dtype=memory_proj_dtype)
        self.v_mem_out_proj = nn.Sequential(
            nn.Linear(self.comp_dim, memory_proj_dim, bias=False),
            nn.RMSNorm(memory_proj_dim),
        ).to(dtype=memory_proj_dtype)
        self.rationale_head = MemoryRationaleHead(
            memory_dim=self.comp_dim,
            span_dim=self.prof_dim,
            hidden_dim=self.comp_dim,
        )

        self.pad_token_id = self.composer.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.composer.config.eos_token_id

        self._gradient_checkpointing_enabled = False
        self._composer_use_cache_default = getattr(self.composer.config, "use_cache", None)
        self._composer_memory_holder = SimpleNamespace(memory_kv=None)
        self.memory_start_layer = self._install_memory_attention_adapters()

        self._init_weights()

    def _composer_parameter_dtype(self) -> torch.dtype:
        try:
            return next(self.composer.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _resolve_memory_projection_shape(self) -> tuple[int, int]:
        composer_model = getattr(self.composer, "model", None)
        layers = getattr(composer_model, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            self_attn = getattr(layers[0], "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "head_dim") and hasattr(self_attn, "k_proj"):
                head_dim = int(self_attn.head_dim)
                num_key_value_heads = max(1, int(self_attn.k_proj.out_features // head_dim))
                return num_key_value_heads, head_dim
        return 1, self.comp_dim

    def _resolve_memory_compressor_num_heads(self) -> int:
        max_heads = min(8, self.comp_dim)
        for num_heads in range(max_heads, 0, -1):
            if self.comp_dim % num_heads == 0:
                return num_heads
        return 1

    def _install_memory_attention_adapters(self) -> int:
        composer_model = getattr(self.composer, "model", None)
        layers = getattr(composer_model, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
            return 0

        memory_layer_count = min(max(self.memory_upper_layers, 0), len(layers))
        start_layer = len(layers) - memory_layer_count
        for layer_idx, layer in enumerate(layers):
            if layer_idx < start_layer:
                continue

            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None or getattr(self_attn, "_pec_memory_patched", False):
                continue
            required_attrs = (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_norm",
                "k_norm",
                "head_dim",
                "num_key_value_groups",
                "attention_dropout",
                "scaling",
                "layer_idx",
            )
            if not all(hasattr(self_attn, name) for name in required_attrs):
                continue

            self_attn._pec_original_forward = self_attn.forward
            self_attn._pec_memory_state_holder = self._composer_memory_holder
            self_attn._pec_memory_patched = True
            self_attn.forward = MethodType(_qwen_memory_attention_forward, self_attn)
        return max(start_layer, 0)

    def _init_weights(self):
        for p in self.extruder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.xavier_uniform_(self.slot_proj.weight)
        nn.init.xavier_uniform_(self.mem_proj[0].weight)
        nn.init.xavier_uniform_(self.k_mem_out_proj[0].weight)
        nn.init.xavier_uniform_(self.v_mem_out_proj[0].weight)
        nn.init.xavier_uniform_(self.rationale_head.memory_proj.weight)
        nn.init.xavier_uniform_(self.rationale_head.span_proj.weight)
        nn.init.ones_(self.post_extruder_norm.weight)

    @property
    def is_gradient_checkpointing(self) -> bool:
        if self._gradient_checkpointing_enabled:
            return True
        return any(
            bool(getattr(module, "is_gradient_checkpointing", False))
            for module in (self.profiler, self.composer, self.extruder)
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self._gradient_checkpointing_enabled = True

        for module in (self.profiler, self.composer, self.extruder):
            method = getattr(module, "gradient_checkpointing_enable", None)
            if method is None:
                continue
            if gradient_checkpointing_kwargs is None:
                method()
                continue
            try:
                method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                method()

        if hasattr(self.composer.config, "use_cache"):
            self.composer.config.use_cache = False

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing_enabled = False

        for module in (self.profiler, self.composer, self.extruder):
            method = getattr(module, "gradient_checkpointing_disable", None)
            if method is not None:
                method()

        if self._composer_use_cache_default is not None and hasattr(self.composer.config, "use_cache"):
            self.composer.config.use_cache = self._composer_use_cache_default

    def build_memory_artifacts(
            self,
            profiler_input_ids: torch.Tensor,
            profiler_attention_mask: torch.Tensor,
    ):
        prof_outputs = self.profiler(
            input_ids=profiler_input_ids,
            attention_mask=profiler_attention_mask,
        )
        prof_hidden = prof_outputs.last_hidden_state  # [B, Sprof, Dprof]
        extruder_latents, gate_scores = self.extruder(
            context=prof_hidden,
            attn_mask=profiler_attention_mask,
            return_gate_scores=True,
        )  # [B, Nq, Dprof]

        projected_input = self.post_extruder_norm(extruder_latents)  # [B, Nq, Dprof]
        memory_dtype = self.mem_proj[0].weight.dtype
        shared_comp_input = self.mem_proj(projected_input.to(dtype=memory_dtype))  # [B, Nq, Dcomp]
        linear_memory = self.slot_proj(
            shared_comp_input.transpose(1, 2)
        ).transpose(1, 2).contiguous()  # [B, M, Dcomp]
        attention_memory = self.memory_compressor(shared_comp_input)  # [B, M, Dcomp]
        compressed_memory = linear_memory + (float(self.attn_mix_alpha) * attention_memory)  # [B, M, Dcomp]
        batch_size, num_slots, _ = compressed_memory.shape
        memory_keys = self.k_mem_out_proj(compressed_memory)  # [B, M, Hkv * Dh]
        memory_values = self.v_mem_out_proj(compressed_memory)  # [B, M, Hkv * Dh]

        memory_keys = memory_keys.view(batch_size, num_slots, self.memory_num_key_value_heads, self.memory_head_dim)
        memory_values = memory_values.view(batch_size, num_slots, self.memory_num_key_value_heads, self.memory_head_dim)
        memory_keys = memory_keys.transpose(1, 2).contiguous()  # [B, Hkv, M, Dh]
        memory_values = memory_values.transpose(1, 2).contiguous()  # [B, Hkv, M, Dh]

        return {
            "extruder_latents": extruder_latents,
            "gate_scores": gate_scores,
            "projected_input": projected_input,
            "shared_comp_input": shared_comp_input,
            "linear_memory": linear_memory,
            "attention_memory": attention_memory,
            "compressed_memory": compressed_memory,
            "memory_keys": memory_keys,
            "memory_values": memory_values,
            "profiler_hidden": prof_hidden,
        }

    @contextmanager
    def composer_memory_context(
            self,
            *,
            memory_keys: torch.Tensor,
            memory_values: torch.Tensor,
            capture_attention_mass: bool = False,
    ):
        if self._composer_memory_holder is None:
            yield
            return

        previous_memory = getattr(self._composer_memory_holder, "memory_kv", None)
        previous_capture_attention_mass = bool(getattr(self._composer_memory_holder, "capture_attention_mass", False))
        previous_attention_mass_records = getattr(self._composer_memory_holder, "attention_mass_records", None)
        self._composer_memory_holder.memory_kv = {
            "memory_keys": memory_keys,
            "memory_values": memory_values,
        }
        self._composer_memory_holder.capture_attention_mass = bool(capture_attention_mass)
        self._composer_memory_holder.attention_mass_records = [] if capture_attention_mass else None
        try:
            yield
        finally:
            self._composer_memory_holder.memory_kv = previous_memory
            self._composer_memory_holder.capture_attention_mass = previous_capture_attention_mass
            self._composer_memory_holder.attention_mass_records = previous_attention_mass_records

    @torch.no_grad()
    def generate_with_memory(
            self,
            *,
            profiler_input_ids: torch.Tensor,
            profiler_attention_mask: torch.Tensor,
            composer_input_ids: torch.Tensor,
            composer_attention_mask: torch.Tensor,
            max_new_tokens: int = 128,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        profiler_input_ids = profiler_input_ids.to(device)  # [B, Sprof]
        profiler_attention_mask = profiler_attention_mask.to(device)  # [B, Sprof]
        composer_input_ids = composer_input_ids.to(device)  # [B, Stok]
        composer_attention_mask = composer_attention_mask.to(device)  # [B, Stok]

        artifacts = self.build_memory_artifacts(
            profiler_input_ids=profiler_input_ids,
            profiler_attention_mask=profiler_attention_mask,
        )

        eos_token_id = getattr(self.composer.config, "eos_token_id", None)
        previous_use_cache = getattr(self.composer.config, "use_cache", None)
        batch_size = composer_input_ids.shape[0]
        current_attention_mask = composer_attention_mask  # [B, Stok]

        try:
            if previous_use_cache is not None:
                self.composer.config.use_cache = True

            with self.composer_memory_context(
                    memory_keys=artifacts["memory_keys"],
                    memory_values=artifacts["memory_values"],
            ):
                outputs = self.composer(
                    input_ids=composer_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]

                generated_tokens = []
                for _ in range(max_new_tokens):
                    generated_tokens.append(next_token)
                    if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
                        break

                    step_attention_mask = torch.ones(
                        (batch_size, 1),
                        device=device,
                        dtype=current_attention_mask.dtype,
                    )  # [B, 1]
                    current_attention_mask = torch.cat([current_attention_mask, step_attention_mask],
                                                       dim=1)  # [B, <= Stok + max_new_tokens]
                    outputs = self.composer(
                        input_ids=next_token,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        finally:
            if previous_use_cache is not None:
                self.composer.config.use_cache = previous_use_cache

        if not generated_tokens:
            return composer_input_ids.new_empty((batch_size, 0))
        return torch.cat(generated_tokens, dim=1)  # [B, Tgen]

    def forward(
            self,
            profiler_input_ids,
            profiler_attention_mask,
            composer_input_ids,
            composer_attention_mask,
            labels=None,
            mh_target_input_ids_list=None,
            mh_target_attention_mask_list=None,
            example_ids=None,
            span_boundaries=None,
            span_token_boundaries=None,
            span_mask=None,
            teacher_relevance=None,
            return_projector_raw: bool = False,
            return_logits: bool = False,
    ):
        del example_ids, span_boundaries
        artifacts = self.build_memory_artifacts(
            profiler_input_ids=profiler_input_ids,
            profiler_attention_mask=profiler_attention_mask,
        )
        z_pool = artifacts["extruder_latents"].mean(dim=1)  # [B, Dprof]
        previous_memory = None
        should_restore_memory = True
        if self._composer_memory_holder is not None:
            previous_memory = getattr(self._composer_memory_holder, "memory_kv", None)
            self._composer_memory_holder.memory_kv = {
                "memory_keys": artifacts["memory_keys"],
                "memory_values": artifacts["memory_values"],
            }
            # Transformer layer checkpointing recomputes attention blocks during backward.
            # Keep the memory KV mounted until the next forward so recomputation sees the
            # same patched attention path instead of falling back to the original one.
            should_restore_memory = not (self.training and self.is_gradient_checkpointing)

        try:
            outputs = self.composer(
                input_ids=composer_input_ids,
                attention_mask=composer_attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        except Exception:
            if self._composer_memory_holder is not None:
                self._composer_memory_holder.memory_kv = previous_memory
            raise
        else:
            if self._composer_memory_holder is not None and should_restore_memory:
                self._composer_memory_holder.memory_kv = previous_memory

        answer_loss = outputs.loss
        mh_align_loss = self._compute_morehop_align_loss(
            z_pool=z_pool,
            mh_target_input_ids_list=mh_target_input_ids_list,
            mh_target_attention_mask_list=mh_target_attention_mask_list,
        )
        rationale_outputs = self._compute_rationale_outputs(
            profiler_hidden=artifacts["profiler_hidden"],
            memory_slots=artifacts["compressed_memory"],
            span_token_boundaries=span_token_boundaries,
            span_mask=span_mask,
            teacher_relevance=teacher_relevance,
        )

        total_loss = answer_loss
        if mh_align_loss is not None:
            total_loss = total_loss + (self.morehop_align_lambda * mh_align_loss)
        if rationale_outputs["rationale_loss"] is not None:
            total_loss = total_loss + (self.rationale_lambda * rationale_outputs["rationale_loss"])

        outputs_dict = {
            "loss": total_loss,
            "answer_loss": answer_loss,
            "mh_align_loss": mh_align_loss,
            "rationale_loss": rationale_outputs["rationale_loss"],
            "teacher_relevance_entropy": rationale_outputs["teacher_relevance_entropy"],
            "student_relevance_entropy": rationale_outputs["student_relevance_entropy"],
            "rationale_top1_hit": rationale_outputs["rationale_top1_hit"],
            "rationale_top3_overlap": rationale_outputs["rationale_top3_overlap"],
            "student_relevance": rationale_outputs["student_relevance"],
            "gate_scores": artifacts["gate_scores"],
            "logits": outputs.logits if return_logits else None,
        }
        if return_projector_raw:
            outputs_dict["projector_raw"] = artifacts["shared_comp_input"]
        return outputs_dict

    @staticmethod
    def _build_span_representations(
        *,
        profiler_hidden: torch.Tensor,
        span_token_boundaries: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = profiler_hidden.float()
        batch_size, _, hidden_dim = hidden.shape
        prefix_sum = torch.cat([hidden.new_zeros((batch_size, 1, hidden_dim)), hidden.cumsum(dim=1)], dim=1)

        span_starts = span_token_boundaries[..., 0].clamp(min=0, max=hidden.shape[1])
        span_ends = span_token_boundaries[..., 1].clamp(min=0, max=hidden.shape[1])
        gather_shape = (-1, -1, hidden_dim)
        gathered_starts = prefix_sum.gather(1, span_starts.unsqueeze(-1).expand(*gather_shape))
        gathered_ends = prefix_sum.gather(1, span_ends.unsqueeze(-1).expand(*gather_shape))
        span_sums = gathered_ends - gathered_starts

        span_lengths = (span_ends - span_starts).clamp_min(1).unsqueeze(-1).to(dtype=span_sums.dtype)
        span_representations = span_sums / span_lengths
        return span_representations * span_mask.unsqueeze(-1).to(dtype=span_representations.dtype)

    @staticmethod
    def _mean_entropy(distribution: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor | None:
        valid_rows = torch.any(span_mask, dim=-1)
        if not torch.any(valid_rows):
            return None
        distribution = distribution.float() * span_mask.to(dtype=distribution.dtype)
        distribution = distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy[valid_rows].mean()

    @staticmethod
    def _compute_topk_alignment_metrics(
        *,
        teacher_distribution: torch.Tensor,
        student_distribution: torch.Tensor,
        span_mask: torch.Tensor,
        k: int = 3,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        top1_hits: list[float] = []
        topk_overlaps: list[float] = []
        for teacher_row, student_row, mask_row in zip(
            teacher_distribution.float(),
            student_distribution.float(),
            span_mask.bool(),
        ):
            if not torch.any(mask_row):
                continue
            teacher_valid = teacher_row[mask_row]
            student_valid = student_row[mask_row]
            if teacher_valid.numel() == 0:
                continue
            top1_hits.append(float(int(torch.argmax(teacher_valid).item()) == int(torch.argmax(student_valid).item())))
            k_eff = min(int(k), int(teacher_valid.numel()))
            teacher_topk = set(torch.topk(teacher_valid, k=k_eff).indices.tolist())
            student_topk = set(torch.topk(student_valid, k=k_eff).indices.tolist())
            topk_overlaps.append(len(teacher_topk & student_topk) / float(k_eff))

        if not top1_hits:
            return None, None

        return (
            teacher_distribution.new_tensor(sum(top1_hits) / len(top1_hits), dtype=torch.float32),
            teacher_distribution.new_tensor(sum(topk_overlaps) / len(topk_overlaps), dtype=torch.float32),
        )

    def _compute_rationale_outputs(
        self,
        *,
        profiler_hidden: torch.Tensor,
        memory_slots: torch.Tensor,
        span_token_boundaries: torch.Tensor | None,
        span_mask: torch.Tensor | None,
        teacher_relevance: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        if span_token_boundaries is None or teacher_relevance is None:
            return {
                "rationale_loss": None,
                "teacher_relevance_entropy": None,
                "student_relevance_entropy": None,
                "rationale_top1_hit": None,
                "rationale_top3_overlap": None,
                "student_relevance": None,
            }

        span_token_boundaries = span_token_boundaries.to(device=profiler_hidden.device)
        if span_mask is None:
            span_mask = span_token_boundaries[..., 0] >= 0
        span_mask = span_mask.to(device=profiler_hidden.device, dtype=torch.bool)
        teacher_distribution = teacher_relevance.to(device=profiler_hidden.device, dtype=torch.float32)
        teacher_distribution = teacher_distribution * span_mask.to(dtype=teacher_distribution.dtype)
        teacher_norm = teacher_distribution.sum(dim=-1, keepdim=True)
        valid_rows = torch.any(span_mask, dim=-1) & (teacher_norm.squeeze(-1) > 0.0)
        if not torch.any(valid_rows):
            return {
                "rationale_loss": None,
                "teacher_relevance_entropy": None,
                "student_relevance_entropy": None,
                "rationale_top1_hit": None,
                "rationale_top3_overlap": None,
                "student_relevance": None,
            }

        teacher_distribution = torch.where(
            teacher_norm > 0.0,
            teacher_distribution / teacher_norm.clamp_min(1e-8),
            teacher_distribution,
        )

        span_representations = self._build_span_representations(
            profiler_hidden=profiler_hidden,
            span_token_boundaries=span_token_boundaries,
            span_mask=span_mask,
        )
        student_distribution, _ = self.rationale_head(
            memory_slots=memory_slots,
            span_representations=span_representations,
            span_mask=span_mask,
        )
        student_distribution = student_distribution.float()
        kl_per_row = F.kl_div(
            student_distribution.clamp_min(1e-8).log(),
            teacher_distribution,
            reduction="none",
        ).sum(dim=-1)
        rationale_loss = kl_per_row[valid_rows].mean()

        teacher_relevance_entropy = self._mean_entropy(teacher_distribution[valid_rows], span_mask[valid_rows])
        student_relevance_entropy = self._mean_entropy(student_distribution[valid_rows], span_mask[valid_rows])
        rationale_top1_hit, rationale_top3_overlap = self._compute_topk_alignment_metrics(
            teacher_distribution=teacher_distribution[valid_rows],
            student_distribution=student_distribution[valid_rows],
            span_mask=span_mask[valid_rows],
            k=3,
        )
        return {
            "rationale_loss": rationale_loss,
            "teacher_relevance_entropy": teacher_relevance_entropy,
            "student_relevance_entropy": student_relevance_entropy,
            "rationale_top1_hit": rationale_top1_hit,
            "rationale_top3_overlap": rationale_top3_overlap,
            "student_relevance": student_distribution,
        }

    @staticmethod
    def _morehop_alignment_weight_schedule(num_targets: int) -> list[float]:
        if num_targets <= 0:
            return []
        if num_targets == 1:
            return [1.0]
        if num_targets == 2:
            return [0.3, 1.0]
        if num_targets == 3:
            return [0.2, 0.5, 1.0]
        return torch.linspace(0.2, 1.0, steps=num_targets, dtype=torch.float32).tolist()

    def _compute_morehop_align_loss(
            self,
            *,
            z_pool: torch.Tensor,
            mh_target_input_ids_list: list[torch.Tensor] | None,
            mh_target_attention_mask_list: list[torch.Tensor] | None,
    ):
        if not mh_target_input_ids_list or not mh_target_attention_mask_list:
            return None

        total_loss = z_pool.new_zeros(())
        total_weight = z_pool.new_zeros(())
        active_rows_per_target = [
            torch.any(target_attention_mask.bool(), dim=1).to(device=z_pool.device)
            for target_attention_mask in mh_target_attention_mask_list
        ]
        targets_per_row = torch.stack(active_rows_per_target, dim=0).sum(dim=0)

        for target_index, (target_input_ids, target_attention_mask, active_rows) in enumerate(zip(
                mh_target_input_ids_list,
                mh_target_attention_mask_list,
                active_rows_per_target,
        )):
            if self.morehop_align_mode == "last":
                active_rows = active_rows & (targets_per_row == (target_index + 1))
            if not torch.any(active_rows):
                continue

            target_outputs = self.profiler(
                input_ids=target_input_ids,
                attention_mask=target_attention_mask,
            )
            target_hidden = target_outputs.last_hidden_state  # [B, Saux, Dprof]

            mask = target_attention_mask.unsqueeze(-1).to(dtype=target_hidden.dtype)  # [B, Saux, 1]
            denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
            r_k = (target_hidden * mask).sum(dim=1) / denom  # [B, Dprof]

            similarities = F.cosine_similarity(
                z_pool[active_rows].float(),
                r_k[active_rows].float(),
                dim=-1,
            )
            if self.morehop_align_mode == "last":
                target_weights = torch.ones_like(similarities)
            else:
                target_weights = similarities.new_tensor(
                    [
                        self._morehop_alignment_weight_schedule(int(target_count))[target_index]
                        for target_count in targets_per_row[active_rows].tolist()
                    ]
                )
            total_loss = total_loss + (((1.0 - similarities) * target_weights).sum()).to(dtype=total_loss.dtype)
            total_weight = total_weight + target_weights.sum().to(dtype=total_weight.dtype)

        if not bool(total_weight > 0):
            return None

        return total_loss / total_weight
