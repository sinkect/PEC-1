from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


class GroupQueryAttention(nn.Module):
    """
    Cross-Attention with GQA.
    - Query: Learnable Latents (No Positional Embedding in Query)
    - Key/Value: Input Context
    """

    def __init__(self, hidden_dim: int, head_dim: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads

        self.q_dim = head_dim * num_heads
        self.kv_dim = head_dim * num_key_value_heads
        self.output_dim = hidden_dim

        # Separate projections for Latents and Context
        self.q_proj = nn.Linear(hidden_dim, self.q_dim, bias=False)
        self.kv_proj = nn.Linear(hidden_dim, self.kv_dim * 2, bias=False)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Project latent slots into grouped query heads.  # [B, Nq, D] -> [B, Nq, Hq, Dh]
        query = self.q_proj(latents)
        query = rearrange(query, "B Nq (H Dh) -> B Nq H Dh", H=self.num_heads)

        # Project context into grouped keys and values.  # [B, S, D] -> [B, S, 2, Hkv, Dh]
        kv = self.kv_proj(context)
        kv = rearrange(kv, "B S (Two H Dh) -> B S Two H Dh", Two=2, H=self.num_key_value_heads)
        key, value = kv.unbind(dim=2)  # each: [B, S, Hkv, Dh]

        # SDPA expects heads before sequence.  # [B, Nq, Hq, Dh] -> [B, Hq, Nq, Dh]
        query, key, value = map(lambda tensor: rearrange(tensor, "B S H Dh -> B H S Dh"), (query, key, value))

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=True,
        )

        # Merge grouped heads back into the latent hidden dimension.  # [B, Hq, Nq, Dh] -> [B, Nq, D]
        return rearrange(attn_output, "B H Nq Dh -> B Nq (H Dh)")


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.norm_q = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(hidden_dim, eps=1e-6)

        self.attn = GroupQueryAttention(
            hidden_dim,
            hidden_dim // num_heads,
            num_heads,
            num_key_value_heads,
        )

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Normalize latent slots and context independently before cross-attention.  # [B, Nq, D], [B, S, D]
        latents_norm = self.norm_q(latents)  # [B, Nq, D]
        context_norm = self.norm_k(context)  # [B, S, D]

        attn_out = self.attn(latents_norm, context_norm, attn_mask=attn_mask)  # [B, Nq, D]
        out = self.out_proj(attn_out)  # [B, Nq, D]
        return latents + out  # [B, Nq, D]


class MemoryCompressor(nn.Module):
    def __init__(self, dim: int, num_memory_slots: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_memory_slots = num_memory_slots
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.memory_queries = nn.Parameter(torch.randn(1, num_memory_slots, dim) * 0.02)
        self.norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_latents, hidden_dim = z.shape
        num_slots = self.num_memory_slots
        num_heads = self.num_heads
        head_dim = self.head_dim

        queries = self.memory_queries.expand(batch_size, -1, -1)
        queries = queries.view(batch_size, num_slots, num_heads, head_dim).transpose(1, 2)
        keys = z.view(batch_size, num_latents, num_heads, head_dim).transpose(1, 2)
        values = z.view(batch_size, num_latents, num_heads, head_dim).transpose(1, 2)

        compressed = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            is_causal=False,
        )
        compressed = compressed.transpose(1, 2).contiguous().view(batch_size, num_slots, hidden_dim)
        return self.norm(compressed)


class Extruder(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(
        self,
        hidden_size: int,
        num_query_tokens: int = 64,
        nums_layers: int = 3,
        num_heads: int = 8,
        num_key_value_heads: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        self._gradient_checkpointing_enabled = False
        self._gradient_checkpointing_kwargs = {}

        # Keep the original parameter name for checkpoint compatibility.
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        conditioning_dim = hidden_size * 2
        self.delta_mlp = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_query_tokens * hidden_size),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_query_tokens),
        )
        self.query_self_attn = AttentionBlock(hidden_size, num_heads, num_key_value_heads)
        self.layers = nn.ModuleList([
            AttentionBlock(hidden_size, num_heads, num_key_value_heads)
            for _ in range(nums_layers)
        ])

        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self._init_dynamic_query()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing_enabled

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self._gradient_checkpointing_enabled = True
        self._gradient_checkpointing_kwargs = dict(gradient_checkpointing_kwargs or {})

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing_enabled = False
        self._gradient_checkpointing_kwargs = {}

    def _init_dynamic_query(self) -> None:
        for module in (self.delta_mlp, self.gate_mlp):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        delta_out = self.delta_mlp[-1]
        gate_out = self.gate_mlp[-1]
        nn.init.zeros_(delta_out.weight)
        nn.init.zeros_(delta_out.bias)
        nn.init.zeros_(gate_out.weight)
        nn.init.zeros_(gate_out.bias)

    def _build_conditioning_vector(
        self,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cls_pool = context[:, 0, :]  # [B, D]
        if attn_mask is None:
            mean_pool = context.mean(dim=1)  # [B, D]
        else:
            mask = attn_mask.unsqueeze(-1).to(dtype=context.dtype)  # [B, S, 1]
            mean_pool = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, D]
        return torch.cat([cls_pool, mean_pool], dim=-1)  # [B, 2D]

    def build_query_tokens(
        self,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_gate_scores: bool = False,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        conditioning = self._build_conditioning_vector(context, attn_mask=attn_mask)  # [B, 2D]
        delta = self.delta_mlp(conditioning).view(batch_size, self.num_query_tokens, self.hidden_size)  # [B, Nq, D]
        gate = torch.sigmoid(self.gate_mlp(conditioning)).view(batch_size, self.num_query_tokens, 1)  # [B, Nq, 1]
        base_queries = self.query_tokens.expand(batch_size, -1, -1)  # [B, Nq, D]
        query_tokens = base_queries + (gate * delta)  # [B, Nq, D]
        if return_gate_scores:
            return query_tokens, gate
        return query_tokens

    def _should_checkpoint(self, *tensors: Optional[torch.Tensor]) -> bool:
        if not self._gradient_checkpointing_enabled or not self.training:
            return False
        return any(tensor is not None and tensor.requires_grad for tensor in tensors)

    def _checkpoint_layer(self, layer, *args):
        checkpoint_kwargs = dict(self._gradient_checkpointing_kwargs)
        use_reentrant = checkpoint_kwargs.pop("use_reentrant", False)
        return checkpoint(layer, *args, use_reentrant=use_reentrant, **checkpoint_kwargs)

    def forward(
        self,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_gate_scores: bool = False,
    ) -> torch.Tensor:
        """
        Input: context [B, S, D]
        Output: latents [B, Nq, D]
        """
        context = context.to(dtype=self.query_tokens.dtype)
        if return_gate_scores:
            latents, gate_scores = self.build_query_tokens(
                context,
                attn_mask=attn_mask,
                return_gate_scores=True,
            )
        else:
            latents = self.build_query_tokens(context, attn_mask=attn_mask)  # [B, Nq, D]
            gate_scores = None
        should_checkpoint = self._should_checkpoint(latents)
        if should_checkpoint:
            def self_attn_forward(current_latents: torch.Tensor):
                return self.query_self_attn(current_latents, current_latents)

            latents = self._checkpoint_layer(self_attn_forward, latents)
        else:
            latents = self.query_self_attn(latents, latents)

        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn_mask = rearrange(attn_mask, "B S -> B 1 1 S")  # [B, 1, 1, S]

        # Refine latent slots with cross-attention over the profiler context.  # [B, Nq, D]
        for layer in self.layers:
            should_checkpoint = self._should_checkpoint(latents, context)
            if should_checkpoint:
                if attn_mask is None:
                    def layer_forward(current_latents: torch.Tensor, current_context: torch.Tensor):
                        return layer(current_latents, current_context)

                    latents = self._checkpoint_layer(layer_forward, latents, context)
                else:
                    def layer_forward(
                        current_latents: torch.Tensor,
                        current_context: torch.Tensor,
                        current_attn_mask: torch.Tensor,
                    ):
                        return layer(current_latents, current_context, attn_mask=current_attn_mask)

                    latents = self._checkpoint_layer(layer_forward, latents, context, attn_mask)
            else:
                latents = layer(latents, context, attn_mask=attn_mask)

        latents = self.final_norm(latents)  # [B, Nq, D]
        if return_gate_scores:
            return latents, gate_scores
        return latents
