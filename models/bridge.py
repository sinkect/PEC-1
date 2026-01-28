from typing import Tuple, Optional
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F

CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freq = torch.outer(t, inv_freq)
        emb = torch.cat((freq, freq), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len=None):
        if seq_len is not None:
            return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]
        return self.cos_cached, self.sin_cached


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, expansion: float = 4.0):
        super().__init__()
        inter = _find_multiple(round(hidden_dim * expansion * 2 / 3), 256)
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * inter)
        self.down_proj = nn.Linear(inter, hidden_dim)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class GroupQueryAttention(nn.Module):
    """
    Cross-Attention with GQA.
    - Query: Learnable Latents
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
        self.o_proj = nn.Linear(self.q_dim, hidden_dim, bias=False)

    def forward(self, latents, context, cos_sin: Optional[CosSin] = None, attn_mask=None):

        # 1. Projections
        query = self.q_proj(latents) # [B, S, H*D]
        query = rearrange(query, 'B S (H D) -> B S H D', H=self.num_heads)

        kv = self.kv_proj(context)
        kv = rearrange(kv, 'B S (TWO H D) -> B S TWO H D', TWO=2, H=self.num_key_value_heads)
        key, value = kv.unbind(dim=2) # [B, S, H, D]

        # 2. Transpose for SDPA [B, H, S, D]
        query, key, value = map(lambda t: rearrange(t, 'B S H D-> B H S D'), (query, key, value))

        # 3. RoPE (Applied only to Key/Context)
        if cos_sin is not None:
            cos, sin = cos_sin

            # Broadcast shapes manually since we only rotate Key
            k_cos = rearrange(cos, 'S D -> 1 1 S D')
            k_sin = rearrange(sin, 'S D -> 1 1 S D')
            key = (key * k_cos) + (rotate_half(key) * k_sin)



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

        return self.o_proj(attn_output)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_key_value_heads):
        super().__init__()
        self.norm_q = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm_ffn = nn.RMSNorm(hidden_dim, eps=1e-6)

        self.attn = GroupQueryAttention(
            hidden_dim,
            hidden_dim // num_heads,
            num_heads,
            num_key_value_heads
        )
        self.swiglu = SwiGLU(hidden_dim, expansion=4.0)

    def forward(self, latents, context, cos_sin: Optional[CosSin] = None, attn_mask=None):
        # Cross-Attention Block
        residual = latents
        latents_norm = self.norm_q(latents)
        context_norm = self.norm_k(context)

        attn_out = self.attn(latents_norm, context_norm, cos_sin=cos_sin, attn_mask=attn_mask)
        latents = residual + attn_out

        # FFN Block
        residual = latents
        latents = self.norm_ffn(latents)
        out = self.swiglu(latents)
        return residual + out


class Extruder(nn.Module):
    def __init__(self, hidden_size,num_query_tokens=64, max_position_embeddings=8192,  nums_layers=4, num_heads=16,
                 num_key_value_heads=4):
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

        self.rotary_emb = RotaryEmbedding(hidden_size // num_heads, max_position_embeddings)
        self.final_norm = nn.RMSNorm(hidden_size, eps=1e-6)

    def forward(self, context, attn_mask=None):
        """
        Input: context [Batch, Doc_Len, Dim]
        Output: latents [Batch, Num_Queries, Dim]
        """
        batch_size = context.shape[0]

        # Expand latents for batch
        latents = self.query_tokens.expand(batch_size, -1, -1)

        # Prepare RoPE for Context
        seq_len = context.shape[1]
        cos, sin = self.rotary_emb(seq_len)
        cos, sin = cos.to(context.device), sin.to(context.device)

        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn_mask = rearrange(attn_mask, 'B S -> B 1 1 S')

        # Iterative refinement
        for layer in self.layers:
            latents = layer(latents, context, cos_sin=(cos, sin), attn_mask=attn_mask)

        return self.final_norm(latents)