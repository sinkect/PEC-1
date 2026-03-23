from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .bridge import Extruder


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
    if attention_mask is None:
        token_mask = _build_causal_keep_mask(
            batch_size=batch_size,
            query_length=query_length,
            token_kv_length=token_kv_length,
            cache_position=cache_position,
            device=device,
        )
        memory_mask = torch.ones((batch_size, 1, query_length, memory_kv_length), dtype=torch.bool, device=device)
        return torch.cat([memory_mask, token_mask], dim=-1)  # [B, 1, Tq, M + Tk]

    if attention_mask.ndim == 2:
        token_padding_mask = attention_mask[:, None, None, :token_kv_length].to(device=device, dtype=torch.bool)
        token_mask = _build_causal_keep_mask(
            batch_size=batch_size,
            query_length=query_length,
            token_kv_length=token_kv_length,
            cache_position=cache_position,
            device=device,
        )
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


class PECEngine(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        profiler_path="answerdotai/ModernBERT-base",
        composer_path="Qwen/Qwen3-1.7B",
        num_query_tokens=16,
        morehop_align_lambda: float = 0.1,
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
        self.morehop_align_lambda = float(morehop_align_lambda)
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

        self.k_mem_proj = nn.Sequential(
            nn.Linear(self.prof_dim, memory_proj_dim, bias=False),
            nn.RMSNorm(memory_proj_dim),
        ).to(dtype=memory_proj_dtype)
        self.v_mem_proj = nn.Sequential(
            nn.Linear(self.prof_dim, memory_proj_dim, bias=False),
            nn.RMSNorm(memory_proj_dim),
        ).to(dtype=memory_proj_dtype)

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

        nn.init.xavier_uniform_(self.k_mem_proj[0].weight)
        nn.init.xavier_uniform_(self.v_mem_proj[0].weight)
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
        extruder_latents = self.extruder(
            context=prof_hidden,
            attn_mask=profiler_attention_mask,
        )  # [B, M, Dprof]

        projected_input = self.post_extruder_norm(extruder_latents)  # [B, M, Dprof]
        memory_dtype = self.k_mem_proj[0].weight.dtype
        batch_size, num_slots, _ = projected_input.shape
        memory_keys = self.k_mem_proj(projected_input.to(dtype=memory_dtype))  # [B, M, Hkv * Dh]
        memory_values = self.v_mem_proj(projected_input.to(dtype=memory_dtype))  # [B, M, Hkv * Dh]

        memory_keys = memory_keys.view(batch_size, num_slots, self.memory_num_key_value_heads, self.memory_head_dim)
        memory_values = memory_values.view(batch_size, num_slots, self.memory_num_key_value_heads, self.memory_head_dim)
        memory_keys = memory_keys.transpose(1, 2).contiguous()  # [B, Hkv, M, Dh]
        memory_values = memory_values.transpose(1, 2).contiguous()  # [B, Hkv, M, Dh]

        return {
            "extruder_latents": extruder_latents,
            "projected_input": projected_input,
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
    ):
        if self._composer_memory_holder is None:
            yield
            return

        previous_memory = getattr(self._composer_memory_holder, "memory_kv", None)
        self._composer_memory_holder.memory_kv = {
            "memory_keys": memory_keys,
            "memory_values": memory_values,
        }
        try:
            yield
        finally:
            self._composer_memory_holder.memory_kv = previous_memory

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
                    current_attention_mask = torch.cat([current_attention_mask, step_attention_mask], dim=1)  # [B, <= Stok + max_new_tokens]
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
        return_logits: bool = False,
    ):
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

        total_loss = answer_loss
        if mh_align_loss is not None:
            total_loss = total_loss + (self.morehop_align_lambda * mh_align_loss)

        return {
            "loss": total_loss,
            "answer_loss": answer_loss,
            "mh_align_loss": mh_align_loss,
            "logits": outputs.logits if return_logits else None,
        }

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
        total_pairs = 0

        for target_input_ids, target_attention_mask in zip(
            mh_target_input_ids_list,
            mh_target_attention_mask_list,
        ):
            active_rows = torch.any(target_attention_mask.bool(), dim=1)
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
            total_loss = total_loss + (1.0 - similarities).sum()
            total_pairs += int(active_rows.sum().item())

        if total_pairs == 0:
            return None

        return total_loss / total_pairs
