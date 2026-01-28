import torch
import torch.nn as nn


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Extruder(nn.Module):
    """
    Lightweight Transformer Decoder that maps profiler embeddings to composer query tokens.
    Shapes: [B, T, D]
    """

    def __init__(self, hidden_size, num_query_tokens=64, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_size, dtype=torch.float16)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, profiler_hidden_states):
        """
        profiler_hidden_states: [B, T, D]
        Returns: [B, Q, D]
        """
        device = profiler_hidden_states.device
        dtype = profiler_hidden_states.dtype
        query_tokens = self.query_tokens.to(device=device, dtype=dtype)
        query_tokens = query_tokens.repeat(profiler_hidden_states.size(0), 1, 1)

        # Decoder expects tgt [B, Q, D] and memory [B, T, D]
        extruded = self.decoder(tgt=query_tokens, memory=profiler_hidden_states)
        return extruded
