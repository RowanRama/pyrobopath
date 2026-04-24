"""Pydantic-backed configuration with YAML loading and merging."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    gcode_dir: str = "data/gcodes"
    task_store: str = "data/task_store"
    pair_cache: str = "data/pair_cache"
    embedding_cache: str = "data/embedding_cache"
    checkpoint_dir: str = "runs/checkpoints"
    log_dir: str = "runs"


class DataConfig(BaseModel):
    resample_length: int = 128
    num_features: int = 11
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True


class PairSamplingConfig(BaseModel):
    pairs_per_epoch: int = 100_000
    collision_fraction: float = 0.5
    near_boundary_fraction: float = 0.2
    near_boundary_clearance_thresh: float = 0.05
    base_sample_mode: Literal["uniform", "clustered", "uniform_clustered"] = "uniform_clustered"
    workspace_x_range: list[float] = Field(default_factory=lambda: [-1.0, 1.0])
    workspace_y_range: list[float] = Field(default_factory=lambda: [-1.0, 1.0])
    dt_modes: list[str] = Field(default_factory=lambda: ["zero", "small", "full"])
    dt_small_range: list[float] = Field(default_factory=lambda: [-0.1, 0.1])
    arm_box_width_range: list[float] = Field(default_factory=lambda: [0.05, 0.15])
    arm_box_safety_margin_range: list[float] = Field(default_factory=lambda: [0.01, 0.05])
    arm_box_min_length: float = 0.2
    arm_box_max_length: float = 0.8
    hard_negative_buffer_size: int = 10_000
    hard_negative_prob: float = 0.1


class EncoderConfig(BaseModel):
    seq_len: int = 128
    out_seq_len: int = 32
    embed_dim: int = 128
    num_channels: list[int] = Field(default_factory=lambda: [64, 96, 128])
    kernel_size: int = 5
    dilation_base: int = 2
    dropout: float = 0.1
    use_groupnorm: bool = True
    groupnorm_groups: int = 8
    activation: str = "gelu"


class HeadConfig(BaseModel):
    embed_dim: int = 128
    base_rel_features: int = 4
    arm_param_features: int = 4
    dt_features: int = 1
    cross_attn_layers: int = 2
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1
    mlp_hidden: int = 256
    mlp_dropout: float = 0.1
    use_dt_gated_alignment: bool = True


class ModelConfig(BaseModel):
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    head: HeadConfig = Field(default_factory=HeadConfig)

    @model_validator(mode="after")
    def check_embed_dims_match(self) -> "ModelConfig":
        if self.encoder.embed_dim != self.head.embed_dim:
            raise ValueError(
                f"encoder.embed_dim ({self.encoder.embed_dim}) must equal "
                f"head.embed_dim ({self.head.embed_dim})"
            )
        return self


class TrainingConfig(BaseModel):
    batch_size: int = 256
    grad_accum_steps: int = 1
    max_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    lr_scheduler: Literal["cosine", "step"] = "cosine"
    use_amp: bool = True
    use_compile: bool = False
    use_ddp: bool = False
    checkpoint_every_n_steps: int = 1000
    eval_every_n_epochs: int = 1
    best_metric: str = "recall_at_fpr"
    best_metric_fpr: float = 0.05
    resume_from: Optional[str] = None


class LossConfig(BaseModel):
    lambda_clearance: float = 1.0
    lambda_binary: float = 2.0
    lambda_consistency: float = 0.5
    huber_delta: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    false_negative_weight: float = 5.0
    disable_clearance_loss: bool = False


class EvalConfig(BaseModel):
    fpr_target: float = 0.05
    near_boundary_thresh: float = 0.05
    latency_n_warmup: int = 20
    latency_n_repeats: int = 200
    pruning_threshold_tau: float = 0.03
    pruning_safety_alpha: float = 0.1


class LoggingConfig(BaseModel):
    log_every_n_steps: int = 50
    flush_every_n_steps: int = 200
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "collision_surrogate"
    wandb_entity: Optional[str] = None
    console_refresh_rate: int = 10


class InferenceConfig(BaseModel):
    batch_size: int = 512
    device: str = "auto"
    export_torchscript: bool = True
    export_onnx: bool = False
    onnx_opset: int = 17


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class Config(BaseModel):
    """Top-level configuration object for the collision surrogate system."""

    run_id: str = "default"
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    pair_sampling: PairSamplingConfig = Field(default_factory=PairSamplingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict.

    Args:
        base: Base dictionary.
        override: Override dictionary (takes precedence).

    Returns:
        Merged dictionary.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file and construct a validated Config object.

    Args:
        path: Path to the YAML config file.
        overrides: Optional dict of key-value overrides to apply after loading.

    Returns:
        Validated :class:`Config` instance.
    """
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)


def load_default_config() -> Config:
    """Load the built-in default configuration.

    Returns:
        Validated :class:`Config` instance with default values.
    """
    default_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    if default_path.exists():
        return load_config(default_path)
    return Config()
