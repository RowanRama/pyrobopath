"""Stage 2: Pairwise Clearance Head.

Takes two cached sequence embeddings (from Stage 1), world-frame base positions,
time offset dt, and arm-box parameters, and predicts:
- Signed minimum clearance (regression, primary output)
- Binary collision probability (classification, auxiliary output)

Architecture:
1. Enrich each embedding with base-relative geometry features.
2. Differentiably shift/align embedding B by dt on the token-time axis.
3. Bidirectional cross-attention (num_layers, num_heads).
4. Global pooling -> MLP -> two output heads.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyrobopath.collision_detection.learned.utils.config import HeadConfig


class _BaseRelEnricher(nn.Module):
    """Appends base-relative geometry features to each token in an embedding.

    For each token position, the module uses the (x_world, y_world) at that
    token (recovered from feat_seq) to compute:
    - length: Euclidean distance from base to EE position.
    - heading_from_base: atan2(dy_base, dx_base).
    - dx_from_base, dy_from_base: signed offsets from base.

    Args:
        embed_dim: Embedding dimension D (input channels).
        base_rel_features: Number of appended features (4).
        out_dim: Output dimension after projection.
    """

    def __init__(self, embed_dim: int, base_rel_features: int, out_dim: int) -> None:
        super().__init__()
        self._base_rel_dim = base_rel_features
        self.proj = nn.Linear(embed_dim + base_rel_features, out_dim)

    def forward(
        self,
        emb: torch.Tensor,
        raw_feat: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        """Enrich embedding with base-relative geometry.

        Args:
            emb: Sequence embedding, shape (B, L', D).
            raw_feat: Raw feature sequence, shape (B, L, F); used to extract
                world-frame positions (columns 0,1).  We use pooled positions
                that align with L' tokens via adaptive average pooling.
            base: World-frame base positions, shape (B, 2).

        Returns:
            Enriched tensor of shape (B, L', out_dim).
        """
        B, Lp, D = emb.shape
        # Extract x,y from raw_feat and downsample to L'
        xy = raw_feat[:, :, :2]  # (B, L, 2)
        xy = xy.permute(0, 2, 1)  # (B, 2, L)
        xy_ds = F.adaptive_avg_pool1d(xy, Lp)  # (B, 2, L')
        xy_ds = xy_ds.permute(0, 2, 1)  # (B, L', 2)

        base_exp = base.unsqueeze(1).expand_as(xy_ds)  # (B, L', 2)
        dx = xy_ds[:, :, 0:1] - base_exp[:, :, 0:1]   # (B, L', 1)
        dy = xy_ds[:, :, 1:2] - base_exp[:, :, 1:2]
        length = torch.sqrt(dx**2 + dy**2 + 1e-8)
        heading = torch.atan2(dy, dx)

        extra = torch.cat([length, heading, dx, dy], dim=-1)  # (B, L', 4)
        combined = torch.cat([emb, extra], dim=-1)            # (B, L', D+4)
        return self.proj(combined)


class _GatedDtAligner(nn.Module):
    """Differentiably shift embedding B tokens by dt along the time axis.

    Implements linear interpolation between adjacent token time positions
    combined with a gating mechanism to modulate alignment confidence.

    Args:
        embed_dim: Embedding dimension.
        max_seq_len: Maximum sequence length L'.
    """

    def __init__(self, embed_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self._D = embed_dim
        self._L = max_seq_len
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, emb_b: torch.Tensor, dt: torch.Tensor, token_dt: float) -> torch.Tensor:
        """Shift embedding B by dt.

        Args:
            emb_b: Embedding for task B, shape (B, L', D).
            dt: Time offsets, shape (B,).
            token_dt: Seconds per token in the embedding grid (duration / L').

        Returns:
            Aligned embedding, shape (B, L', D).
        """
        B, Lp, D = emb_b.shape
        # Fractional token shift
        shift_tokens = dt / (token_dt + 1e-8)  # (B,)
        shift_int = shift_tokens.long().clamp(-Lp + 1, Lp - 1)
        shift_frac = (shift_tokens - shift_int.float()).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)

        # Roll along time dimension
        emb_shifted = torch.roll(emb_b, int(shift_int.float().mean().item()), dims=1)
        emb_next = torch.roll(emb_b, int(shift_int.float().mean().item()) + 1, dims=1)

        # Linear interpolation
        aligned = emb_shifted * (1 - shift_frac) + emb_next * shift_frac

        # Gated modulation
        dt_feat = dt.unsqueeze(-1)  # (B, 1)
        gate = self.gate_mlp(dt_feat).unsqueeze(1)  # (B, 1, D)
        return aligned * gate


class _CrossAttentionLayer(nn.Module):
    """Bidirectional cross-attention layer between two sequences.

    A -> B cross-attention + B -> A cross-attention, both using the other's
    output as key/value.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_ab = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_ba = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_a1 = nn.LayerNorm(embed_dim)
        self.norm_b1 = nn.LayerNorm(embed_dim)
        self.norm_a2 = nn.LayerNorm(embed_dim)
        self.norm_b2 = nn.LayerNorm(embed_dim)
        self.ff_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ff_b = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply bidirectional cross-attention.

        Args:
            emb_a: Sequence A embedding, shape (B, L', D).
            emb_b: Sequence B embedding, shape (B, L', D).

        Returns:
            Updated (emb_a, emb_b) with same shapes.
        """
        # A attends to B
        a_ca, _ = self.attn_ab(emb_a, emb_b, emb_b)
        emb_a = self.norm_a1(emb_a + a_ca)
        emb_a = self.norm_a2(emb_a + self.ff_a(emb_a))

        # B attends to A
        b_ca, _ = self.attn_ba(emb_b, emb_a, emb_a)
        emb_b = self.norm_b1(emb_b + b_ca)
        emb_b = self.norm_b2(emb_b + self.ff_b(emb_b))

        return emb_a, emb_b


class PairwiseHead(nn.Module):
    """Stage 2 pairwise clearance and collision prediction head.

    Inputs:
    - Two sequence embeddings from Stage 1: (B, L', D) each.
    - Two raw feature sequences for base-relative enrichment: (B, L, F) each.
    - World-frame base positions: (B, 2) each.
    - Time offset dt: (B,).
    - Arm box parameters: (B, 4) each.

    Outputs:
    - clearance: Signed minimum clearance in metres, shape (B,).
    - p_collision: Collision probability in [0, 1], shape (B,).

    Args:
        cfg: :class:`HeadConfig`.
    """

    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self._cfg = cfg
        D = cfg.embed_dim

        # Base-relative enrichers (project D+4 -> D)
        self.enricher_a = _BaseRelEnricher(D, cfg.base_rel_features, D)
        self.enricher_b = _BaseRelEnricher(D, cfg.base_rel_features, D)

        # dt aligner for embedding B
        if cfg.use_dt_gated_alignment:
            self.dt_aligner: Optional[_GatedDtAligner] = _GatedDtAligner(D, 64)
        else:
            self.dt_aligner = None

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList(
            [
                _CrossAttentionLayer(D, cfg.cross_attn_heads, cfg.cross_attn_dropout)
                for _ in range(cfg.cross_attn_layers)
            ]
        )

        # Global context encoder: arm params + dt -> context vector
        context_in = cfg.arm_param_features * 2 + cfg.dt_features
        self.context_encoder = nn.Sequential(
            nn.Linear(context_in, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, D),
        )

        # MLP head input: global pool of A + global pool of B + context
        mlp_in = D * 2 + D
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.mlp_dropout),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden // 2),
            nn.GELU(),
        )

        self.clearance_head = nn.Linear(cfg.mlp_hidden // 2, 1)
        self.collision_head = nn.Linear(cfg.mlp_hidden // 2, 1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        base_a: torch.Tensor,
        base_b: torch.Tensor,
        dt: torch.Tensor,
        arm_params_a: torch.Tensor,
        arm_params_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            emb_a: Stage 1 embedding for task A, shape (B, L', D).
            emb_b: Stage 1 embedding for task B, shape (B, L', D).
            feat_a: Raw features for task A, shape (B, L, F).
            feat_b: Raw features for task B, shape (B, L, F).
            base_a: Arm A world base, shape (B, 2).
            base_b: Arm B world base, shape (B, 2).
            dt: Time offset B relative to A, shape (B,).
            arm_params_a: Arm A box params, shape (B, 4).
            arm_params_b: Arm B box params, shape (B, 4).

        Returns:
            Tuple (clearance, p_collision):
            - clearance: shape (B,), metres (signed).
            - p_collision: shape (B,), probability in [0, 1].
        """
        # 1. Enrich with base-relative geometry
        ea = self.enricher_a(emb_a, feat_a, base_a)
        eb = self.enricher_b(emb_b, feat_b, base_b)

        # 2. Align embedding B by dt
        if self.dt_aligner is not None:
            # Estimate seconds-per-token from feat_a durations
            # feat_a[:, -1, 2] = last t_abs; feat_a[:, 0, 2] = first t_abs
            dur = feat_a[:, -1, 2] - feat_a[:, 0, 2]  # (B,)
            token_dt = float((dur.mean() / emb_a.shape[1]).item()) + 1e-6
            eb = self.dt_aligner(eb, dt, token_dt)

        # 3. Bidirectional cross-attention
        for layer in self.cross_attn_layers:
            ea, eb = layer(ea, eb)

        # 4. Global average pooling
        ga = ea.mean(dim=1)  # (B, D)
        gb = eb.mean(dim=1)  # (B, D)

        # 5. Context from arm params + dt
        ctx_in = torch.cat(
            [arm_params_a, arm_params_b, dt.unsqueeze(-1)], dim=-1
        )  # (B, 9)
        ctx = self.context_encoder(ctx_in)  # (B, D)

        # 6. MLP
        fused = torch.cat([ga, gb, ctx], dim=-1)  # (B, 3D)
        h = self.mlp(fused)

        clearance = self.clearance_head(h).squeeze(-1)           # (B,)
        p_collision = torch.sigmoid(self.collision_head(h)).squeeze(-1)  # (B,)

        return clearance, p_collision
