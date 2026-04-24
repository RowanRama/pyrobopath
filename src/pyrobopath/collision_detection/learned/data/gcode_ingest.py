"""G-code ingestion: wraps the user-provided extract_tasks_from_gcode function.

This module is an ADAPTER only. It never reimplements G-code parsing.
The user must supply a callable with the following signature::

    def extract_tasks_from_gcode(path: str | Path) -> list[Task]

Where :class:`Task` is a dataclass/object with at least the attributes:

    - ``polyline``: sequence of (x, y, t) tuples in world-frame metres/seconds
    - ``task_id``: str or int identifier unique within the G-code file
    - Any additional metadata is preserved and stored alongside the resampled tensor.

The adapter wraps that function, adds error handling, and fans out across files
using a multiprocessing pool to saturate I/O and CPUs.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class RawTask:
    """Canonical task representation produced by the adapter.

    Attributes:
        gcode_id: Identifier of the source G-code file (stem or hash).
        task_id: Task identifier within the G-code file.
        polyline: List of (x_world, y_world, t_from_start) tuples, world-frame metres/seconds.
        duration: Total task duration in seconds (max t_from_start).
        metadata: Any extra fields from the user's Task object.
    """

    gcode_id: str
    task_id: str
    polyline: list[tuple[float, float, float]]  # (x, y, t)
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def world_bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box (x_min, y_min, x_max, y_max).

        Returns:
            Four-tuple of floats.
        """
        xs = [p[0] for p in self.polyline]
        ys = [p[1] for p in self.polyline]
        return min(xs), min(ys), max(xs), max(ys)


def _adapt_user_task(gcode_id: str, user_task: Any) -> RawTask:
    """Convert a user-supplied Task object into a :class:`RawTask`.

    Args:
        gcode_id: Identifier for the parent G-code file.
        user_task: Any object exposing ``polyline`` and ``task_id``.

    Returns:
        :class:`RawTask` instance.
    """
    # Normalise polyline to list of 3-tuples
    raw_poly = user_task.polyline
    if raw_poly and hasattr(raw_poly[0], "__iter__"):
        polyline = [(float(p[0]), float(p[1]), float(p[2])) for p in raw_poly]
    else:
        polyline = [(float(p.x), float(p.y), float(p.t)) for p in raw_poly]

    duration = polyline[-1][2] if polyline else 0.0

    # Collect any extra metadata attributes
    reserved = {"polyline", "task_id"}
    metadata = {}
    for attr in dir(user_task):
        if attr.startswith("_") or attr in reserved:
            continue
        try:
            val = getattr(user_task, attr)
            if not callable(val):
                metadata[attr] = val
        except Exception:
            pass

    return RawTask(
        gcode_id=gcode_id,
        task_id=str(getattr(user_task, "task_id", id(user_task))),
        polyline=polyline,
        duration=duration,
        metadata=metadata,
    )


def _ingest_one(args: tuple[Path, Callable]) -> list[RawTask]:
    """Worker function: ingest a single G-code file.

    Args:
        args: Tuple of (path, extract_fn).

    Returns:
        List of :class:`RawTask` objects (may be empty on failure).
    """
    path, extract_fn = args
    gcode_id = path.stem
    try:
        user_tasks = extract_fn(path)
        return [_adapt_user_task(gcode_id, t) for t in user_tasks]
    except Exception as exc:
        logger.warning("Failed to ingest %s: %s", path, exc)
        return []


class GcodeIngestor:
    """Batched G-code ingestion using multiprocessing.

    Args:
        extract_fn: User-supplied ``extract_tasks_from_gcode`` callable.
        num_workers: Number of parallel worker processes.
    """

    def __init__(
        self,
        extract_fn: Callable[[Path], list[Any]],
        num_workers: int = 4,
    ) -> None:
        self._extract_fn = extract_fn
        self._num_workers = num_workers

    def iter_files(self, gcode_dir: str | Path, glob: str = "**/*.gcode") -> list[Path]:
        """List all G-code files under a directory.

        Args:
            gcode_dir: Root directory to search.
            glob: Glob pattern relative to gcode_dir.

        Returns:
            Sorted list of :class:`Path` objects.
        """
        root = Path(gcode_dir)
        paths = sorted(root.glob(glob))
        logger.info("Found %d G-code files in %s", len(paths), root)
        return paths

    def ingest(
        self,
        paths: list[Path],
        *,
        chunksize: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[RawTask]:
        """Ingest a list of G-code files and yield :class:`RawTask` objects.

        Args:
            paths: List of G-code file paths to process.
            chunksize: Files per multiprocessing chunk.
            progress_callback: Optional ``fn(done, total)`` for progress reporting.

        Yields:
            :class:`RawTask` objects in order of completion (not necessarily file order).
        """
        args = [(p, self._extract_fn) for p in paths]
        total = len(args)
        done = 0

        if self._num_workers <= 1:
            for arg in args:
                for task in _ingest_one(arg):
                    yield task
                done += 1
                if progress_callback:
                    progress_callback(done, total)
            return

        ctx = mp.get_context("spawn")
        with ctx.Pool(self._num_workers) as pool:
            for tasks in pool.imap_unordered(_ingest_one, args, chunksize=chunksize):
                for task in tasks:
                    yield task
                done += 1
                if progress_callback:
                    progress_callback(done, total)

    def ingest_dir(
        self,
        gcode_dir: str | Path,
        glob: str = "**/*.gcode",
        **kwargs: Any,
    ) -> Iterator[RawTask]:
        """Convenience method: list + ingest a directory.

        Args:
            gcode_dir: Root directory containing G-code files.
            glob: File glob pattern.
            **kwargs: Forwarded to :meth:`ingest`.

        Yields:
            :class:`RawTask` objects.
        """
        paths = self.iter_files(gcode_dir, glob)
        yield from self.ingest(paths, **kwargs)
