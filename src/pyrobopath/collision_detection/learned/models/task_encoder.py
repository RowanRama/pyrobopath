"""Stage 1: Task Encoder (TCN).

Robot-agnostic, offset-agnostic, world-frame encoder.
Input:  (B, L, F) world-frame feature tensor.
Output: (B, L', D) sequence embedding.

IMPORTANT: This module must NEVER receive base positions, dt, arm parameters,
or any robot-identity information. This invariant is tested in
tests/test_models_shapes.py::test_encoder_robot_agnostic.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyrobopath.collision_detection.learned.utils.config import EncoderConfig


class _DilatedResBlock(nn.Module):
    """A single dilated 1-D convolutional residual block.

    Architecture: Conv1d -> Norm -> Act -> Dropout -> Conv1d -> Norm -> (+residual)

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
        dropout: Dropout probability.
        use_groupnorm: Use GroupNorm instead of BatchNorm.
        groupnorm_groups: Number of groups for GroupNorm.
        activation: Activation function name ("gelu" | "relu").
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_groupnorm: bool,
        groupnorm_groups: int,
        activation: str,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # same-length padding

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=padding
        )

        def _make_norm(ch: int) -> nn.Module:
            if use_groupnorm:
                g = min(groupnorm_groups, ch)
                while ch % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, ch)
            return nn.BatchNorm1d(ch)

        self.norm1 = _make_norm(out_ch)
        self.norm2 = _make_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

        # Residual projection if channels differ
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor of shape (B, C_in, L).

        Returns:
            Output tensor of shape (B, C_out, L).
        """
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class TaskEncoder(nn.Module):
    """Stage 1 Temporal Convolutional Network (TCN) task encoder.

    Encodes a world-frame trajectory tensor into a fixed-length sequence
    embedding that is reusable across robot configurations.

    The encoder is strictly robot-agnostic: it accepts only kinematic features
    derived from the world-frame polyline and produces an embedding that is
    independent of arm identity, base position, or time offset.

    Args:
        cfg: :class:`EncoderConfig` specifying all hyperparameters.
        num_input_features: Number of input features per token (F).
    """

    def __init__(self, cfg: EncoderConfig, num_input_features: int = 11) -> None:
        super().__init__()
        self._cfg = cfg
        self._in_features = num_input_features

        # Input projection
        channels = [num_input_features] + list(cfg.num_channels)
        blocks: list[nn.Module] = []
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            dil = cfg.dilation_base**i
            blocks.append(
                _DilatedResBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=cfg.kernel_size,
                    dilation=dil,
                    dropout=cfg.dropout,
                    use_groupnorm=cfg.use_groupnorm,
                    groupnorm_groups=cfg.groupnorm_groups,
                    activation=cfg.activation,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        # Project to embed_dim
        final_ch = channels[-1]
        self.proj = nn.Conv1d(final_ch, cfg.embed_dim, kernel_size=1)

        # Adaptive temporal downsampling: L -> L'
        self.temporal_pool = nn.AdaptiveAvgPool1d(cfg.out_seq_len)

        # Positional encoding for output sequence
        self.register_buffer(
            "pos_embed",
            self._build_sinusoidal_pe(cfg.out_seq_len, cfg.embed_dim),
            persistent=True,
        )

    @staticmethod
    def _build_sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional encoding table.

        Args:
            seq_len: Sequence length.
            d_model: Embedding dimension.

        Returns:
            Tensor of shape (1, seq_len, d_model).
        """
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe.unsqueeze(0)  # (1, L', D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a world-frame trajectory to a sequence embedding.

        Args:
            x: Feature tensor of shape (B, L, F). All values in world frame.
                No robot-specific information should be included.

        Returns:
            Sequence embedding of shape (B, L', D) where L'=cfg.out_seq_len
            and D=cfg.embed_dim.
        """
        # (B, L, F) -> (B, F, L) for Conv1d
        h = x.permute(0, 2, 1)
        h = self.blocks(h)
        h = self.proj(h)                    # (B, D, L)
        h = self.temporal_pool(h)           # (B, D, L')
        h = h.permute(0, 2, 1)             # (B, L', D)
        h = h + self.pos_embed             # add positional encoding
        return h
