"""Replan a multi-agent toolpath schedule from a given point in time.

This module adds the ability to resume planning after some tasks have already
executed.  Three concerns are handled separately:

1. **Completed tasks** — any ``ContourEvent`` whose end time falls at or before
   ``t_replan`` is considered done.  It is marked complete in a *copy* of the
   dependency graph so that its successors unlock, and it is excluded from any
   new assignment.

2. **In-progress tasks** — a ``ContourEvent`` whose time window straddles
   ``t_replan`` is preserved in the new schedule.  Its remaining portion
   (along with any trailing depart/home events in the same chain) is sliced
   out, offset to start at ``t = 0``, and pre-populated onto that agent's new
   schedule so execution continues from where it was.

3. **Disabled agents** — agents whose ids appear in ``disabled_agents`` are
   removed from the agent roster handed to the planner.  They receive no new
   tasks.  An in-progress task on a disabled agent cannot continue (the agent
   isn't there) — that task is dropped and its successors will remain blocked.

The resulting schedule uses *fresh absolute time* starting at ``t = 0``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from pyrobopath.process import AgentModel, DependencyGraph
from pyrobopath.toolpath import Toolpath

from .metrics import PlanningMetrics
from .schedule import (
    ContourEvent,
    MoveEvent,
    MultiAgentToolpathSchedule,
    ToolpathSchedule,
)
from .toolpath_scheduler import MultiAgentToolpathPlanner, PlanningOptions


# ─────────────────────────────── data types ───────────────────────────────────


@dataclass
class InProgressTask:
    """State of a single task that is mid-execution at the replan point.

    Attributes
    ----------
    contour_id : int
        ID of the in-progress contour.  Used by the planner to know this
        contour will complete at ``remaining_duration`` in the new schedule
        (so its successors unlock correctly).
    remaining_duration : float
        Wall-clock seconds remaining for the complete event chain (contour +
        depart + home) relative to the new ``t = 0``.
    end_position : np.ndarray
        Where the agent will be when those remaining events finish.
    pre_populated_events : list[MoveEvent]
        Sliced events, already offset so they start at ``t = 0``, to be added
        to the new schedule for this agent before planning.
    """

    contour_id: int
    remaining_duration: float
    end_position: np.ndarray
    pre_populated_events: List[MoveEvent]


@dataclass
class ReplanContext:
    """Snapshot of progress at the replan point.

    Use :func:`extract_replan_context` to build one from an existing schedule.
    """

    t_replan: float
    completed_ids: Set[int]
    in_progress: Dict[str, InProgressTask]
    disabled_agents: Set[str] = field(default_factory=set)


# ──────────────────────────── slicing / extraction ────────────────────────────


def _slice_event_after(event: MoveEvent, t_cut: float) -> Optional[MoveEvent]:
    """Return a new MoveEvent covering the portion of ``event`` from
    ``max(event.start, t_cut)`` to ``event.end``, starting at t = 0.

    Returns ``None`` when the event has already finished (``event.end <= t_cut``)
    or the remaining segment is degenerate.
    """
    if event.end <= t_cut:
        return None

    if t_cut <= event.start:
        path = [np.asarray(p).copy() for p in event.data]
    else:
        sliced = event.traj.slice(t_cut, event.end)
        path = [np.asarray(p.data).copy() for p in sliced.points]

    if len(path) < 2:
        return None
    return MoveEvent(0.0, path, event.velocity)


def _collect_agent_tail(
    sched: ToolpathSchedule, t_cut: float
) -> List[MoveEvent]:
    """Return events in ``sched`` whose end time is strictly after ``t_cut``,
    offset so the earliest surviving event starts at ``t = 0``.

    The relative timing between subsequent events is preserved.
    """
    tail: List[MoveEvent] = []
    for ev in sched._events:
        if ev.end <= t_cut:
            continue

        if ev.start < t_cut:
            # Straddles the cut — slice to start at 0.
            sliced = _slice_event_after(ev, t_cut)
            if sliced is None:
                continue
            tail.append(sliced)
        else:
            # Entirely after cut — shift so absolute times become (old - t_cut).
            path = [np.asarray(p).copy() for p in ev.data]
            shifted = MoveEvent(ev.start - t_cut, path, ev.velocity)
            tail.append(shifted)
    return tail


def _find_in_progress_contour(
    sched: ToolpathSchedule, t_replan: float
) -> Optional[ContourEvent]:
    """Return the ContourEvent whose window contains ``t_replan``, or None."""
    for ev in sched._events:
        if isinstance(ev, ContourEvent) and ev.start <= t_replan < ev.end:
            return ev
    return None


def _collect_completed_ids(
    sched: ToolpathSchedule, t_replan: float
) -> Set[int]:
    return {
        ev.contour.id
        for ev in sched._events
        if isinstance(ev, ContourEvent) and ev.end <= t_replan
    }


def extract_replan_context(
    schedule: MultiAgentToolpathSchedule,
    t_replan: float,
    disabled_agents: Optional[Set[str]] = None,
) -> ReplanContext:
    """Build a :class:`ReplanContext` from an existing schedule at ``t_replan``.

    Parameters
    ----------
    schedule
        The schedule being replanned from.  Not modified.
    t_replan
        Time (in the old schedule's frame) at which replanning should begin.
    disabled_agents
        Optional set of agent ids to exclude from the new plan.
    """
    completed: Set[int] = set()
    in_progress: Dict[str, InProgressTask] = {}

    for agent, sched in schedule.schedules.items():
        completed.update(_collect_completed_ids(sched, t_replan))

        ip_ev = _find_in_progress_contour(sched, t_replan)
        if ip_ev is None:
            continue

        tail = _collect_agent_tail(sched, t_replan)
        if not tail:
            continue

        remaining_duration = max(ev.end for ev in tail)
        end_position = np.asarray(tail[-1].data[-1]).copy()

        in_progress[agent] = InProgressTask(
            contour_id=ip_ev.contour.id,
            remaining_duration=float(remaining_duration),
            end_position=end_position,
            pre_populated_events=tail,
        )

    return ReplanContext(
        t_replan=float(t_replan),
        completed_ids=completed,
        in_progress=in_progress,
        disabled_agents=set(disabled_agents or ()),
    )


# ─────────────────────────────── orchestration ────────────────────────────────


def _contour_id_to_index(toolpath: Toolpath) -> Dict[int, int]:
    """Return a map from ``Contour.id`` to its index in ``toolpath.contours``.

    The dependency graph produced by :func:`create_dependency_graph_by_z` (and
    the scheduler's task manager) key nodes by list index rather than
    ``Contour.id``, so we convert whenever crossing that boundary.
    """
    return {c.id: i for i, c in enumerate(toolpath.contours)}


def _build_pruned_dg(
    dg: DependencyGraph,
    completed_ids: Set[int],
    toolpath: Toolpath,
) -> DependencyGraph:
    """Return a shallow copy of ``dg`` with ``completed_ids`` marked done.

    Starts from a *fresh* completion set so a previously-planned dg (which
    the scheduler mutates) doesn't carry stale completions into the replan.
    The graph structure is shared with the original (never mutated here).
    """
    id_to_idx = _contour_id_to_index(toolpath)
    new_dg = copy.copy(dg)
    new_dg._completed_tasks = set()
    for cid in completed_ids:
        idx = id_to_idx.get(cid)
        if idx is not None:
            new_dg.mark_complete(idx)
    return new_dg


def _build_initial_state(
    active_agents: Dict[str, AgentModel],
    context: ReplanContext,
    toolpath: Toolpath,
):
    """Convert a ReplanContext into the initial-state kwargs expected by
    :meth:`MultiAgentToolpathPlanner.plan`.

    Only agents still present in ``active_agents`` (i.e. not disabled) get
    pre-populated events and start-time offsets.  Agents whose in-progress
    tasks cannot continue (disabled agent) are silently dropped.
    """
    id_to_idx = _contour_id_to_index(toolpath)

    initial_schedule = MultiAgentToolpathSchedule()
    initial_schedule.add_agents(active_agents.keys())
    initial_start_times: Dict[str, float] = {}
    initial_positions: Dict[str, np.ndarray] = {}
    initial_in_progress: Dict[int, float] = {}

    for agent, ip in context.in_progress.items():
        if agent not in active_agents:
            continue
        for ev in ip.pre_populated_events:
            initial_schedule.add_event(ev, agent)
        initial_start_times[agent] = ip.remaining_duration
        initial_positions[agent] = np.asarray(ip.end_position)
        idx = id_to_idx.get(ip.contour_id)
        if idx is not None:
            initial_in_progress[idx] = ip.remaining_duration

    return (
        initial_schedule,
        initial_start_times,
        initial_positions,
        initial_in_progress,
    )


def replan(
    agent_models: Dict[str, AgentModel],
    toolpath: Toolpath,
    dg: DependencyGraph,
    options: PlanningOptions,
    context: ReplanContext,
    metrics: Optional[PlanningMetrics] = None,
) -> MultiAgentToolpathSchedule:
    """Produce a new schedule that continues execution from a ReplanContext.

    Parameters
    ----------
    agent_models
        Full agent roster.  Any agent id in ``context.disabled_agents`` is
        filtered out before planning.
    toolpath, dg, options
        Same semantics as :meth:`MultiAgentToolpathPlanner.plan`.  The
        dependency graph is copied so the caller's is never mutated.
    context
        The replan state (see :func:`extract_replan_context`).
    metrics
        Optional metrics collector for the new planning run.
    """
    active_agents = {
        a: m for a, m in agent_models.items() if a not in context.disabled_agents
    }
    if not active_agents:
        raise ValueError("replan(): no active agents remain after filtering")

    pruned_dg = _build_pruned_dg(dg, context.completed_ids, toolpath)

    (
        initial_schedule,
        initial_start_times,
        initial_positions,
        initial_in_progress,
    ) = _build_initial_state(active_agents, context, toolpath)

    planner = MultiAgentToolpathPlanner(active_agents)
    return planner.plan(
        toolpath=toolpath,
        dg=pruned_dg,
        options=options,
        metrics=metrics,
        initial_schedule=initial_schedule,
        initial_start_times=initial_start_times,
        initial_positions=initial_positions,
        initial_in_progress=initial_in_progress,
    )
