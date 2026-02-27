from typing import Tuple, Optional
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    """
    Cross-Attention with GQA.
    - Query: Learnable Latents (No Positional Embedding in Query)
    - Key/Value: Input Context
    """

    def __init__(self, hidden_dim, head_dim, num_heads, num_key_value_heads):
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

    def forward(self, latents, context, attn_mask=None):

        # 1. Projections
        query = self.q_proj(latents) # [B, S, H*D]
        query = rearrange(query, 'B S (H D) -> B S H D', H=self.num_heads)

        kv = self.kv_proj(context)
        kv = rearrange(kv, 'B S (TWO H D) -> B S TWO H D', TWO=2, H=self.num_key_value_heads)
        key, value = kv.unbind(dim=2) # [B, S, H, D]

        # 2. Transpose for SDPA [B, H, S, D]
        query, key, value = map(lambda t: rearrange(t, 'B S H D-> B H S D'), (query, key, value))

        # 4. Attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=True
        )

        # 5. Output Projection
        # [B, H, S, D] -> [B, S, (H D)]
        attn_output = rearrange(attn_output, 'B H S D -> B S (H D)')

        return attn_output


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_key_value_heads):
        super().__init__()
        self.norm_q = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(hidden_dim, eps=1e-6)

        self.attn = GroupQueryAttention(
            hidden_dim,
            hidden_dim // num_heads,
            num_heads,
            num_key_value_heads
        )

        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, latents, context,  attn_mask=None):
        # Cross-Attention Block

        latents_norm = self.norm_q(latents)
        context_norm = self.norm_k(context)

        attn_out = self.attn(latents_norm, context_norm, attn_mask=attn_mask)
        gate = F.silu(self.gate_proj(latents_norm))
        gated_out = attn_out * gate
        out = self.out_proj(gated_out)
        return latents + out


class Extruder(nn.Module):
    def __init__(self, hidden_size,num_query_tokens=64,  nums_layers=3, num_heads=8,
                 num_key_value_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens

        # Initialize Learnable Query
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.layers = nn.ModuleList([
            AttentionBlock(hidden_size, num_heads, num_key_value_heads)
            for _ in range(nums_layers)
        ])

        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, context, attn_mask=None):
        """
        Input: context [Batch, Doc_Len, Dim]
        Output: latents [Batch, Num_Queries, Dim]
        """
        batch_size = context.shape[0]

        # Expand latents for batch
        latents = self.query_tokens.expand(batch_size, -1, -1)


        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn_mask = rearrange(attn_mask, 'B S -> B 1 1 S')

        # Iterative refinement
        for layer in self.layers:
            latents = layer(latents, context, attn_mask=attn_mask)

        return self.final_norm(latents)