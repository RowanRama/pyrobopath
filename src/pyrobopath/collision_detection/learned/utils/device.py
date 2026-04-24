"""Device selection and multi-GPU utilities."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.

    Args:
        device_str: "auto", "cuda", "cpu", or a specific "cuda:N".

    Returns:
        Resolved :class:`torch.device`.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def get_local_rank() -> int:
    """Return LOCAL_RANK for DDP training; 0 if not in a distributed context.

    Returns:
        Integer local rank.
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Return the DDP world size; 1 if not distributed.

    Returns:
        Integer world size.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True if this is the rank-0 process (or non-distributed).

    Returns:
        Boolean.
    """
    return get_local_rank() == 0


def setup_ddp(local_rank: int) -> None:
    """Initialize the default process group for DDP.

    Args:
        local_rank: The local GPU rank for this process.
    """
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def teardown_ddp() -> None:
    """Destroy the process group after DDP training completes."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def gpu_memory_mb(device: torch.device | None = None) -> float:
    """Return current GPU memory allocated in MB.

    Args:
        device: Target device; defaults to current CUDA device.

    Returns:
        Memory in megabytes, or 0.0 on CPU.
    """
    if not torch.cuda.is_available():
        return 0.0
    dev = device or torch.device("cuda")
    return torch.cuda.memory_allocated(dev) / 1024**2
