"""Temporal world model with rotary temporal-spatial attention."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.latent_wam.config import LatentWAMConfig
from navsim.agents.latent_wam.modules.rope import (
    PositionGetter,
    PositionGetter4D,
    RotaryPositionEmbedding3D,
    RotaryPositionEmbedding4D,
)


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with 3D/4D RoPE support and Flash Attention.

    This module implements multi-head attention that applies 3D or 4D Rotary Position
    Embeddings to query and key tensors, then uses PyTorch's scaled_dot_product_attention
    for optimized computation (Flash Attention on supported hardware).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, rope_module=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.rope_module = rope_module  # Can be 3D or 4D RoPE

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, tgt_len, embed_dim]
            key: [batch_size, src_len, embed_dim]
            value: [batch_size, src_len, embed_dim]
            query_pos: [batch_size, tgt_len, D] - D=3 for 3D or D=4 for 4D positions
            key_pos: [batch_size, src_len, D] - D=3 for 3D or D=4 for 4D positions
            attn_mask: [tgt_len, src_len] or [batch_size * num_heads, tgt_len, src_len]
            key_padding_mask: [batch_size, src_len]
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Linear projections and reshape to multi-head format
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE if available and positions are provided
        if self.rope_module is not None:
            if query_pos is not None:
                q = self.rope_module(q, query_pos)
            if key_pos is not None:
                k = self.rope_module(k, key_pos)

        # Prepare attention mask for SDPA
        final_mask = None
        if attn_mask is not None:
            # attn_mask: [tgt_len, src_len] or [batch*heads, tgt_len, src_len]
            # Convert to [batch, num_heads, tgt_len, src_len] or broadcastable format
            if attn_mask.dim() == 2:
                final_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, src_len]
            elif attn_mask.dim() == 3:
                # Assume [batch*heads, tgt_len, src_len], reshape to [batch, heads, tgt_len, src_len]
                final_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)

        # Merge key_padding_mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch_size, src_len], True means IGNORE
            # Convert to additive mask: -inf for ignored positions
            padding_mask = torch.zeros(batch_size, 1, 1, src_len, dtype=q.dtype, device=q.device)
            padding_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

            if final_mask is None:
                final_mask = padding_mask
            else:
                final_mask = final_mask + padding_mask

        # Use Flash Attention via SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=final_mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        output = self.out_proj(attn_output)

        return output


class RoPETransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with 3D/4D RoPE support."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, rope_module=None):
        super().__init__()

        # Self-attention with RoPE
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout, rope_module)

        # Cross-attention with RoPE
        self.cross_attn = RoPEMultiheadAttention(d_model, nhead, dropout, rope_module)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, tgt_len, d_model]
            memory: [batch_size, src_len, d_model]
            tgt_pos: [batch_size, tgt_len, D] - D=3 for 3D or D=4 for 4D positions
            memory_pos: [batch_size, src_len, D] - D=3 for 3D or D=4 for 4D positions
        """
        # Self-attention block
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_pos, tgt_pos, tgt_mask, tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention block
        tgt2 = self.cross_attn(tgt, memory, memory, tgt_pos, memory_pos, memory_mask, memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class RoPETransformerDecoder(nn.Module):
    """Transformer decoder with 3D/4D RoPE support."""

    def __init__(self, decoder_layer: RoPETransformerDecoderLayer, num_layers: int):
        super().__init__()
        # The released checkpoint intentionally contains one layer reused four
        # times. Keep parameter sharing here for numerical compatibility.
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, tgt_len, d_model]
            memory: [batch_size, src_len, d_model]
            tgt_pos: [batch_size, tgt_len, D] - D=3 for 3D or D=4 for 4D positions
            memory_pos: [batch_size, src_len, D] - D=3 for 3D or D=4 for 4D positions
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_pos,
                memory_pos,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )

        return output


class TemporalWorldModel(nn.Module):
    """Temporal world model for Latent WAM with 3D/4D RoPE support.

    Args:
        config: LatentWAMConfig
        use_4d_rope: If True, use 4D RoPE (time, view, height, width).
                     If False, use 3D RoPE (time, height, width). Default: False
    """

    def __init__(self, config: LatentWAMConfig, use_4d_rope: bool = False):
        super().__init__()
        self._config = config
        self.use_4d_rope = use_4d_rope

        if use_4d_rope:
            # 4D RoPE (time, view, height, width)
            self.rope_module = RotaryPositionEmbedding4D(
                temporal_frequency=10.0,  # 12 frames
                view_frequency=10.0,  # 3 views
                spatial_frequency=100.0,  # 16x16 grid
            )
            self.position_getter = PositionGetter4D()
        else:
            # 3D RoPE (time, height, width)
            self.rope_module = RotaryPositionEmbedding3D(temporal_frequency=10.0, spatial_frequency=100.0)
            self.position_getter = PositionGetter()

        # Create decoder layer with RoPE support
        decoder_layer = RoPETransformerDecoderLayer(
            d_model=config.transformer.model_dim,
            nhead=config.transformer.num_heads,
            dim_feedforward=config.transformer.feedforward_dim,
            dropout=config.transformer.dropout,
            rope_module=self.rope_module,
        )

        # Create decoder with multiple layers
        self.future_decoder = RoPETransformerDecoder(
            decoder_layer=decoder_layer, num_layers=config.transformer.num_layers
        )

    def get_attention_mask(self, seq_len, block_size):
        mask = torch.zeros((seq_len, seq_len))
        num_blocks = seq_len // block_size
        assert seq_len % block_size == 0, "seq_len must be divisible by block_size"

        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            mask[start:end, start:end] = 1

        return mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

    def forward(self, query, keyval, time_frames, height, width, views=None, tgt_mask=None, memory_mask=None):
        """Forward pass of the temporal world model.

        Args:
            query: [batch_size, tgt_len, d_model]
            keyval: [batch_size, src_len, d_model]
            time_frames: Number of temporal frames
            height: Height of the spatial grid
            width: Width of the spatial grid
            views: Number of camera views (only required for 4D RoPE). Default: None
            tgt_mask: Attention mask for target sequence
            memory_mask: Attention mask for memory sequence

        Note:
            For 3D RoPE: tgt_len = time_frames * height * width
            For 4D RoPE: tgt_len = time_frames * views * height * width
        """
        batch_size = query.size(0)
        device = query.device

        # Automatically generate positions based on RoPE type
        if self.use_4d_rope:
            assert views is not None, "views parameter is required when using 4D RoPE"
            # Generate 4D positions (t, v, h, w)
            query_pos = self.position_getter(batch_size, time_frames, views, height, width, device)
            keyval_pos = self.position_getter(batch_size, time_frames, views, height, width, device)
        else:
            # Generate 3D positions (t, v, s)
            num_views = height
            num_scene_tokens = width
            if time_frames == 1:
                query_time_idx = [12]
                keyval_time_idx = [4]
            elif time_frames == 3:
                query_time_idx = [4, 8, 12]
                keyval_time_idx = [1, 4, 8]
            elif time_frames == 7:
                query_time_idx = [2, 3, 4, 6, 8, 10, 12]
                keyval_time_idx = [1, 2, 3, 4, 6, 8, 10]
            else:
                raise ValueError(f"unsupported temporal world-model frame count: {time_frames}")
            query_pos = self.position_getter(
                batch_size, time_frames, num_views, num_scene_tokens, device, time_list=query_time_idx
            ).to(torch.long)
            keyval_pos = self.position_getter(
                batch_size, time_frames, num_views, num_scene_tokens, device, time_list=keyval_time_idx
            ).to(torch.long)

        output = self.future_decoder(
            tgt=query,
            memory=keyval,
            tgt_pos=query_pos,
            memory_pos=keyval_pos,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return output
