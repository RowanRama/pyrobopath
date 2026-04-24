"""Pre-compute and persist Stage 1 embeddings for all tasks.

The cache stores (L', D) embedding tensors keyed by (gcode_id, task_id).
Crucially, this cache remains valid when arms are added, removed, or reassigned
because Stage 1 is strictly robot-agnostic.

Cache format: PyTorch .pt file containing a dict mapping
"gcode_id__task_id" -> tensor of shape (L', D).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch

from pyrobopath.collision_detection.learned.data.gcode_ingest import RawTask
from pyrobopath.collision_detection.learned.data.resample import task_to_tensor
from pyrobopath.collision_detection.learned.models.full_model import FullModel
from pyrobopath.collision_detection.learned.utils.config import Config

logger = logging.getLogger(__name__)

_KEY_SEP = "__"


def _make_key(gcode_id: str, task_id: str) -> str:
    return f"{gcode_id}{_KEY_SEP}{task_id}"


class EmbeddingCache:
    """Persistent cache for Stage 1 task embeddings.

    The cache is valid across arm reassignment and robot fleet changes
    because Stage 1 depends only on world-frame kinematic features.

    Args:
        cache_path: Path to the .pt cache file.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)
        self._store: dict[str, torch.Tensor] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        data = torch.load(str(self._path), map_location="cpu")
        self._store = data
        logger.info("Loaded %d cached embeddings from %s", len(self._store), self._path)

    def save(self) -> None:
        """Persist the cache to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._store, str(self._path))
        logger.info("Saved %d embeddings to %s", len(self._store), self._path)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return _make_key(*key) in self._store

    def get(self, gcode_id: str, task_id: str) -> Optional[torch.Tensor]:
        """Retrieve a cached embedding.

        Args:
            gcode_id: G-code file identifier.
            task_id: Task identifier.

        Returns:
            Tensor of shape (L', D) or None if not cached.
        """
        return self._store.get(_make_key(gcode_id, task_id))

    def put(self, gcode_id: str, task_id: str, embedding: torch.Tensor) -> None:
        """Store an embedding.

        Args:
            gcode_id: G-code file identifier.
            task_id: Task identifier.
            embedding: Tensor of shape (L', D).
        """
        self._store[_make_key(gcode_id, task_id)] = embedding.cpu()

    def __len__(self) -> int:
        return len(self._store)

    def keys(self) -> list[tuple[str, str]]:
        """Return all (gcode_id, task_id) pairs in the cache.

        Returns:
            List of 2-tuples.
        """
        result = []
        for k in self._store:
            parts = k.split(_KEY_SEP, 1)
            result.append((parts[0], parts[1]))
        return result


@torch.no_grad()
def build_embedding_cache(
    model: FullModel,
    tasks: list[RawTask],
    cfg: Config,
    device: torch.device,
    cache_path: str | Path,
    batch_size: int = 256,
    overwrite: bool = False,
) -> EmbeddingCache:
    """Pre-compute Stage 1 embeddings for all tasks and save to disk.

    Args:
        model: Trained :class:`FullModel` (only Stage 1 encoder is used).
        tasks: List of :class:`RawTask` objects to encode.
        cfg: Full configuration.
        device: Target device.
        cache_path: Output .pt file path.
        batch_size: Number of tasks to encode per batch.
        overwrite: If False, skip tasks already in an existing cache.

    Returns:
        :class:`EmbeddingCache` with all embeddings populated.
    """
    cache = EmbeddingCache(cache_path)
    model.eval()
    L = cfg.model.encoder.seq_len
    F = cfg.data.num_features

    pending = [
        t for t in tasks if overwrite or (t.gcode_id, t.task_id) not in cache
    ]
    logger.info("Encoding %d tasks (skipping %d cached)", len(pending), len(tasks) - len(pending))

    for i in range(0, len(pending), batch_size):
        batch_tasks = pending[i : i + batch_size]
        feat_list = [task_to_tensor(t, L) for t in batch_tasks]
        feat_batch = torch.stack(feat_list).to(device)  # (B, L, F)

        use_amp = cfg.training.use_amp and torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=use_amp):
            embs = model.encode_task(feat_batch)  # (B, L', D)

        for j, task in enumerate(batch_tasks):
            cache.put(task.gcode_id, task.task_id, embs[j])

        if (i // batch_size) % 10 == 0:
            logger.info("Encoded %d / %d tasks", i + len(batch_tasks), len(pending))

    cache.save()
    logger.info("Embedding cache complete: %d entries", len(cache))
    return cache
