"""Full model: end-to-end wrapper combining Stage 1 + Stage 2.

Supports two modes:
1. End-to-end training: encode both tasks and run the pairwise head.
2. Cache-based inference: accept pre-computed embeddings directly.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from pyrobopath.collision_detection.learned.models.pairwise_head import PairwiseHead
from pyrobopath.collision_detection.learned.models.task_encoder import TaskEncoder
from pyrobopath.collision_detection.learned.utils.config import Config, EncoderConfig, HeadConfig


class FullModel(nn.Module):
    """End-to-end collision surrogate model.

    Args:
        encoder_cfg: Stage 1 encoder configuration.
        head_cfg: Stage 2 head configuration.
        num_input_features: Number of input features per token.
    """

    def __init__(
        self,
        encoder_cfg: EncoderConfig,
        head_cfg: HeadConfig,
        num_input_features: int = 11,
    ) -> None:
        super().__init__()
        self.encoder = TaskEncoder(encoder_cfg, num_input_features=num_input_features)
        self.head = PairwiseHead(head_cfg)

    @classmethod
    def from_config(cls, cfg: Config) -> "FullModel":
        """Construct a FullModel from a full Config object.

        Args:
            cfg: Full configuration.

        Returns:
            :class:`FullModel` instance.
        """
        return cls(
            encoder_cfg=cfg.model.encoder,
            head_cfg=cfg.model.head,
            num_input_features=cfg.data.num_features,
        )

    def encode_task(self, feat: torch.Tensor) -> torch.Tensor:
        """Encode a single task's feature tensor.

        This is the cacheable Stage 1 operation. The result depends only on
        world-frame kinematic features, never on robot configuration.

        Args:
            feat: Feature tensor of shape (B, L, F).

        Returns:
            Sequence embedding of shape (B, L', D).
        """
        return self.encoder(feat)

    def predict_from_embeddings(
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
        """Run Stage 2 given pre-computed embeddings (cache-based inference).

        Args:
            emb_a: Cached Stage 1 embedding for task A, shape (B, L', D).
            emb_b: Cached Stage 1 embedding for task B, shape (B, L', D).
            feat_a: Raw features for task A (needed for base-rel geometry), (B, L, F).
            feat_b: Raw features for task B, (B, L, F).
            base_a: Arm A world base, shape (B, 2).
            base_b: Arm B world base, shape (B, 2).
            dt: Time offset B relative to A, shape (B,).
            arm_params_a: Arm A box params, shape (B, 4).
            arm_params_b: Arm B box params, shape (B, 4).

        Returns:
            (clearance, p_collision) each shape (B,).
        """
        return self.head(
            emb_a, emb_b, feat_a, feat_b, base_a, base_b, dt, arm_params_a, arm_params_b
        )

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        base_a: torch.Tensor,
        base_b: torch.Tensor,
        dt: torch.Tensor,
        arm_params_a: torch.Tensor,
        arm_params_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass: encode both tasks then predict.

        Args:
            feat_a: Task A feature tensor, shape (B, L, F).
            feat_b: Task B feature tensor, shape (B, L, F).
            base_a: Arm A world base, shape (B, 2).
            base_b: Arm B world base, shape (B, 2).
            dt: Time offset B relative to A, shape (B,).
            arm_params_a: Arm A box params, shape (B, 4).
            arm_params_b: Arm B box params, shape (B, 4).

        Returns:
            (clearance, p_collision) each shape (B,).
        """
        emb_a = self.encoder(feat_a)
        emb_b = self.encoder(feat_b)
        return self.head(
            emb_a, emb_b, feat_a, feat_b, base_a, base_b, dt, arm_params_a, arm_params_b
        )

    def num_parameters(self) -> dict[str, int]:
        """Count trainable parameters by module.

        Returns:
            Dict with 'encoder', 'head', and 'total' parameter counts.
        """
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "encoder": enc_params,
            "head": head_params,
            "total": enc_params + head_params,
        }
