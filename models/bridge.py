from typing import Optional, Tuple, Union
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

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_gate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Cross-Attention Block

        latents_norm = self.norm_q(latents)  # [B, N_q, D]
        context_norm = self.norm_k(context)  # [B, S_ctx, D]

        attn_out = self.attn(latents_norm, context_norm, attn_mask=attn_mask)  # [B, N_q, D]
        gate_scores = torch.sigmoid(self.gate_proj(latents_norm))  # [B, N_q, D]
        gated_out = attn_out * gate_scores  # [B, N_q, D]
        out = self.out_proj(gated_out)  # [B, N_q, D]
        updated_latents = latents + out  # [B, N_q, D]

        if return_gate:
            return updated_latents, gate_scores
        return updated_latents


class Extruder(nn.Module):
    def __init__(self, hidden_size,num_query_tokens=64,  nums_layers=3, num_heads=8,
                 num_key_value_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens

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

        self.layers = nn.ModuleList([
            AttentionBlock(hidden_size, num_heads, num_key_value_heads)
            for _ in range(nums_layers)
        ])

        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self._init_dynamic_query()

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
        cls_pool = context[:, 0, :]
        if attn_mask is None:
            mean_pool = context.mean(dim=1)
        else:
            mask = attn_mask.unsqueeze(-1).to(dtype=context.dtype)
            mean_pool = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return torch.cat([cls_pool, mean_pool], dim=-1)

    def build_query_tokens(
        self,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        conditioning = self._build_conditioning_vector(context, attn_mask=attn_mask)
        delta = self.delta_mlp(conditioning).view(batch_size, self.num_query_tokens, self.hidden_size)
        gate = torch.sigmoid(self.gate_mlp(conditioning)).view(batch_size, self.num_query_tokens, 1)
        return self.query_tokens.expand(batch_size, -1, -1) + (gate * delta)

    def forward(
        self,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_gate_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Input: context [Batch, Doc_Len, Dim]
        Output: latents [Batch, Num_Queries, Dim]
        """
        latents = self.build_query_tokens(context, attn_mask=attn_mask)


        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn_mask = rearrange(attn_mask, 'B S -> B 1 1 S')  # [B, 1, 1, S_ctx]

        # Iterative refinement
        collected_gate_scores = []
        for layer in self.layers:
            if return_gate_scores:
                latents, gate_scores = layer(
                    latents,
                    context,
                    attn_mask=attn_mask,
                    return_gate=True,
                )
                collected_gate_scores.append(gate_scores)  # each: [B, N_q, D]
            else:
                latents = layer(latents, context, attn_mask=attn_mask)

        latents = self.final_norm(latents)  # [B, N_q, D]

        if return_gate_scores:
            stacked_gate_scores = torch.stack(collected_gate_scores, dim=1)  # [B, L, N_q, D]
            return latents, stacked_gate_scores

        return latents
