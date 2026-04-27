from __future__ import annotations
from typing import Dict, Optional
from dataclasses import dataclass
import time
import numpy as np

from pyrobopath.toolpath import Toolpath, Contour
from pyrobopath.process import AgentModel, DependencyGraph

from .schedule import ContourEvent, MoveEvent, MultiAgentToolpathSchedule
from .toolpath_collision import events_cause_collision
from .metrics import PlanningMetrics


@dataclass
class PlanningOptions:
    retract_height: float = 50.0
    collision_offset: float = 5.0
    collision_gap_threshold: float = 1.0
    # "out_degree"             — original behaviour (most successors first)
    # "farthest_centroid"      — max distance from other agents' active
    #                            ellipsoid centroids (targets MVEE separation)
    # "nearest_start"          — min travel distance from current position
    #                            to the task's first waypoint
    # "nearest_base"           — prefer tasks whose centroid is closest
    #                            to this agent's base frame (territorial)
    # "min_ellipsoid"          — prefer small ellipsoids first (small footprint
    #                            ⇒ rarely overlaps anything concurrent)
    # "farthest_other_bases"   — max-min distance from this task's centroid
    #                            to OTHER agents' base frames (targets reach
    #                            check — other arms can't reach this region)
    # "prune_aware"            — for each candidate, count how many currently
    #                            active other-agent contours the prefilter
    #                            would certify as safe; pick the task that
    #                            prunes the most.  Direct optimisation of the
    #                            metric; slightly more expensive.
    # "hybrid_base_separation" — weighted sum of nearest_base and
    #                            farthest_centroid (see priority_weight)
    # "hybrid_travel_prune"    — weighted sum of nearest_start and
    #                            farthest_centroid (see priority_weight)
    task_priority: str = "out_degree"

    # Weight for hybrid strategies.  0.0 = pure spatial-separation term,
    # 1.0 = pure base-proximity / travel term. Default 0.5 (equal blend).
    priority_weight: float = 0.5


def _other_agent_centroids(agent, schedule, agent_models, t):
    """Return a list of (2,) centroids for tasks currently active on other agents.

    Uses the prefilter cache (``mu_e``) when available, otherwise falls back
    to the last known position of the agent end-effector.
    """
    centroids = []
    for a, sched in schedule.schedules.items():
        if a == agent:
            continue
        pf = agent_models[a].collision_prefilter
        cache = getattr(pf, "_cache", {}) if pf is not None else {}
        for ev in sched._events:
            if isinstance(ev, ContourEvent) and ev.start <= t <= ev.end:
                entry = cache.get(ev.contour.id)
                if entry is not None and hasattr(entry, "mu_e"):
                    centroids.append(np.asarray(entry.mu_e[:2], dtype=float))
                else:
                    centroids.append(np.asarray(ev.contour.path[0][:2], dtype=float))
                break
    return centroids


def _farthest_centroid_key(node, contours, agent, schedule, agent_models, t):
    """Sort key: minimum distance from this task's centroid to any other active
    agent's centroid.  Larger = further away = better for prefilter pruning.

    Falls back to 0.0 when no other agent is active (all values equal, so
    ordering is arbitrary but stable).
    """
    contour = contours[node]
    agent_model = agent_models[agent]
    pf = agent_model.collision_prefilter
    cache = getattr(pf, "_cache", {}) if pf is not None else {}
    entry = cache.get(contour.id)
    if entry is not None and hasattr(entry, "mu_e"):
        c = np.asarray(entry.mu_e[:2], dtype=float)
    else:
        c = np.asarray(contour.path[0][:2], dtype=float)

    others = _other_agent_centroids(agent, schedule, agent_models, t)
    if not others:
        return 0.0
    return float(min(np.linalg.norm(c - o) for o in others))


def _nearest_base_key(node, contours, agent_model):
    """Sort key: negative distance from task centroid to agent base frame.
    Uses mu_e from the prefilter cache when available, otherwise path[0].
    Negated so descending sort gives nearest-to-base first.
    """
    contour = contours[node]
    pf = agent_model.collision_prefilter
    cache = getattr(pf, "_cache", {}) if pf is not None else {}
    entry = cache.get(contour.id)
    if entry is not None and hasattr(entry, "mu_e"):
        c = np.asarray(entry.mu_e[:2], dtype=float)
    else:
        c = np.asarray(contour.path[0][:2], dtype=float)
    base = np.asarray(agent_model.base_frame_position[:2], dtype=float)
    return -float(np.linalg.norm(c - base))


def _nearest_start_key(node, contours, p_current):
    """Sort key: negative travel distance from current position to the task's
    first waypoint.  Negated so that sorting descending gives nearest-first.
    """
    start = np.asarray(contours[node].path[0], dtype=float)
    return -float(np.linalg.norm(np.asarray(p_current, dtype=float) - start))


def _contour_centroid(contour, agent_model):
    """Return the task's 2-D centroid, preferring the cached MVEE centre."""
    pf = agent_model.collision_prefilter
    cache = getattr(pf, "_cache", {}) if pf is not None else {}
    entry = cache.get(contour.id)
    if entry is not None and hasattr(entry, "mu_e"):
        return np.asarray(entry.mu_e[:2], dtype=float)
    return np.asarray(contour.path[0][:2], dtype=float)


def _contour_ellipsoid_area(contour, agent_model):
    """Return a size metric for the task's MVEE — product of semi-axes (∝ area).
    Larger = bigger ellipsoid = more likely to overlap other ellipsoids.
    Falls back to the contour bounding-box diagonal when no cache entry exists.
    """
    pf = agent_model.collision_prefilter
    cache = getattr(pf, "_cache", {}) if pf is not None else {}
    entry = cache.get(contour.id)
    if entry is not None and hasattr(entry, "evals"):
        # semi-axis a_i = 1 / sqrt(λ_i); area ∝ a_0 * a_1
        evals = np.asarray(entry.evals, dtype=float)
        axes = 1.0 / np.sqrt(np.maximum(evals, 1e-12))
        return float(axes[0] * axes[1])
    # fallback: bounding-box diagonal squared (proportional to area)
    pts = np.asarray([p[:2] for p in contour.path], dtype=float)
    if len(pts) < 2:
        return 0.0
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return float(diag * diag * 0.25)  # rough area proxy


def _min_ellipsoid_key(node, contours, agent_model):
    """Sort key: negative ellipsoid area — smaller ellipsoids first.
    Tiny tasks rarely overlap larger concurrent ones, boosting prune rate.
    """
    return -_contour_ellipsoid_area(contours[node], agent_model)


def _farthest_other_bases_key(node, contours, agent, agent_models):
    """Sort key: min distance from this task's centroid to all OTHER agents'
    base frames.  Larger = further from other arms' reach, reducing reach-
    check failures inside the prefilter.
    """
    agent_model = agent_models[agent]
    c = _contour_centroid(contours[node], agent_model)
    other_bases = [
        np.asarray(m.base_frame_position[:2], dtype=float)
        for a, m in agent_models.items() if a != agent
    ]
    if not other_bases:
        return 0.0
    return float(min(np.linalg.norm(c - b) for b in other_bases))


def _prune_aware_key(node, contours, agent, schedule, agent_models, t):
    """Sort key: count of currently-active other-agent contours that the
    prefilter would certify as safe against this candidate.  Higher = more
    pairs will be pruned if this task is assigned now.

    When no prefilter is attached, no other agent is active, or the cache
    lookup fails, falls through to 0.0.
    """
    agent_model = agent_models[agent]
    prefilter = agent_model.collision_prefilter
    if prefilter is None:
        return 0.0

    contour = contours[node]
    base_a = np.asarray(agent_model.base_frame_position, dtype=float)

    count = 0
    for a, sched in schedule.schedules.items():
        if a == agent:
            continue
        base_b = np.asarray(agent_models[a].base_frame_position, dtype=float)
        for ev in sched._events:
            if isinstance(ev, ContourEvent) and ev.start <= t <= ev.end:
                try:
                    if prefilter.is_safe_pair(
                        contour.id, ev.contour.id, base_a, base_b
                    ):
                        count += 1
                except Exception:
                    pass
                break
    return float(count)


def _hybrid_base_separation_key(
    node, contours, agent, schedule, agent_models, t, weight,
):
    """Sort key: weighted blend of nearest_base and farthest_centroid.

    Returns:  w * (−dist_to_own_base) + (1−w) * min_dist_to_other_centroids
    Larger = better.  w is clamped to [0, 1].
    """
    w = max(0.0, min(1.0, float(weight)))
    agent_model = agent_models[agent]

    c = _contour_centroid(contours[node], agent_model)
    base = np.asarray(agent_model.base_frame_position[:2], dtype=float)
    base_term = -float(np.linalg.norm(c - base))

    others = _other_agent_centroids(agent, schedule, agent_models, t)
    sep_term = 0.0 if not others else \
        float(min(np.linalg.norm(c - o) for o in others))

    return w * base_term + (1.0 - w) * sep_term


def _hybrid_travel_prune_key(
    node, contours, agent, schedule, agent_models, t, p_current, weight,
):
    """Sort key: weighted blend of nearest_start and farthest_centroid.

    Returns:  w * (−travel_dist) + (1−w) * min_dist_to_other_centroids
    Larger = better.  w is clamped to [0, 1].
    """
    w = max(0.0, min(1.0, float(weight)))
    agent_model = agent_models[agent]
    contour = contours[node]

    start = np.asarray(contour.path[0], dtype=float)
    travel_term = -float(np.linalg.norm(np.asarray(p_current, dtype=float) - start))

    c = _contour_centroid(contour, agent_model)
    others = _other_agent_centroids(agent, schedule, agent_models, t)
    sep_term = 0.0 if not others else \
        float(min(np.linalg.norm(c - o) for o in others))

    return w * travel_term + (1.0 - w) * sep_term


class SchedulingContext:
    def __init__(self, agent_models: Dict[str, AgentModel], options: PlanningOptions):
        self.agent_models = agent_models
        self.options = options
        self.reset()

    def reset(self):
        self.start_times = dict.fromkeys(self.agent_models.keys(), 0.0)
        self.positions = dict()
        for agent in self.agent_models:
            self.positions[agent] = self.agent_models[agent].home_position

    def get_agents_with_start_time(self, time):
        min_time_agents = [a for (a, t) in self.start_times.items() if t == time]
        return min_time_agents

    def get_unique_start_times(self):
        return sorted(set(self.start_times.values()))

    def set_agent_start_time(self, agent, time):
        self.start_times[agent] = time

    def get_current_position(self, agent):
        return self.positions[agent]


class TaskManager:
    def __init__(self, toolpath: Toolpath, dg: DependencyGraph):
        self.contours = toolpath.contours
        self.dg = dg

        # task sets
        self.frontier = set()
        self.in_progress: Dict[str, float] = dict()

    def add_inprogress(self, id, t_end):
        self.in_progress[id] = t_end

    def mark_inprogress_complete(self, time):
        complete = [k for (k, v) in self.in_progress.items() if time >= v]
        for c in complete:
            self.dg.mark_complete(c)
            self.in_progress.pop(c)

    def has_frontier(self):
        return bool(self.frontier)

    def get_available_tasks(self, *args):
        available = [n for n in self.frontier if self.dg.can_start(n)]
        return available


def build_event_chain(
    t_start, p_start, contour: Contour, agent, context: SchedulingContext
):
    # travel + approach event
    p_approach = contour.path[0].copy()
    p_approach[2] += context.options.retract_height
    path_travel = [p_start, p_approach, contour.path[0]]
    if (p_start == p_approach).all():
        path_travel.pop(0)
    e_travel = MoveEvent(
        t_start, path_travel, context.agent_models[agent].travel_velocity
    )

    # contour event
    e_contour = ContourEvent(
        e_travel.end, contour, context.agent_models[agent].velocity
    )

    # depart + home events
    p_depart = contour.path[-1].copy()
    p_depart[2] += context.options.retract_height
    e_depart = MoveEvent(
        e_contour.end,
        [contour.path[-1], p_depart],
        context.agent_models[agent].travel_velocity,
    )
    e_home = MoveEvent(
        e_depart.end,
        [p_depart, context.agent_models[agent].home_position],
        context.agent_models[agent].travel_velocity,
    )

    return [e_travel, e_contour, e_depart, e_home]


class MultiAgentToolpathPlanner:
    def __init__(self, agent_models: Dict[str, AgentModel]):
        self._agent_models = agent_models

    def plan(
        self,
        toolpath: Toolpath,
        dg: DependencyGraph,
        options: PlanningOptions,
        metrics: Optional[PlanningMetrics] = None,
        initial_schedule: Optional[MultiAgentToolpathSchedule] = None,
        initial_start_times: Optional[Dict[str, float]] = None,
        initial_positions: Optional[Dict[str, "np.ndarray"]] = None,
        initial_in_progress: Optional[Dict[int, float]] = None,
    ) -> MultiAgentToolpathSchedule:
        """Plan a multi-agent toolpath schedule.

        If any AgentModel has a non-None ``collision_prefilter``, all agents
        with a non-None prefilter must reference the SAME instance; otherwise
        a ``ValueError`` is raised. ``CachedCollisionModel`` uses a class-level
        cache shared across instances of the same subclass, so the single
        instance is used to populate the cache once per call.

        Parameters
        ----------
        toolpath : Toolpath
            The toolpath to schedule.
        dg : DependencyGraph
            Task dependency graph.
        options : PlanningOptions
            Planner configuration.
        metrics : PlanningMetrics, optional
            If provided, timing and count data are accumulated into this object
            without altering any planning decisions.
        initial_schedule, initial_start_times, initial_positions, initial_in_progress
            Optional hooks used by :mod:`pyrobopath.toolpath_scheduling.replan`
            to resume planning from a partially-executed state.  See that
            module's :func:`replan` helper.  All default to ``None``, matching
            the original fresh-plan behaviour.
        """
        self._validate_toolpath(toolpath)

        if initial_schedule is not None:
            schedule = initial_schedule
            # Ensure every agent has a slot even if it had no pre-populated events
            for a in self._agent_models.keys():
                if a not in schedule.schedules:
                    schedule.add_agent(a)
        else:
            schedule = MultiAgentToolpathSchedule()
            schedule.add_agents(self._agent_models.keys())

        context = SchedulingContext(self._agent_models, options)
        if initial_start_times:
            for a, t_val in initial_start_times.items():
                if a in context.start_times:
                    context.set_agent_start_time(a, float(t_val))
        if initial_positions:
            for a, pos in initial_positions.items():
                if a in context.positions:
                    context.positions[a] = pos

        tm = TaskManager(toolpath, dg)
        in_progress_ids = set((initial_in_progress or {}).keys())
        # Initial frontier: nodes not completed, not currently in-progress,
        # with all predecessors already complete.  For a fresh plan this is
        # equivalent to dg.roots().
        tm.frontier.update(
            n for n in dg._graph.nodes
            if n not in dg._completed_tasks
            and n not in in_progress_ids
            and dg.can_start(n)
        )
        if initial_in_progress:
            for cid, t_end in initial_in_progress.items():
                tm.in_progress[cid] = float(t_end)

        t = min(context.start_times.values(), default=0.0)

        if metrics is not None:
            metrics.n_tasks = len(toolpath.contours)

        # Enforce single shared prefilter instance across agents.
        # All agents must either all have None, or share the SAME
        # CachedCollisionModel instance. This is required because the
        # shared class-level cache relies on a single build_cache call.
        prefilters = [
            m.collision_prefilter
            for m in self._agent_models.values()
            if m.collision_prefilter is not None
        ]
        if len(prefilters) > 0:
            if len({id(p) for p in prefilters}) > 1:
                raise ValueError(
                    "All agents must share the same CachedCollisionModel instance. "
                    "Create one instance and assign it to all "
                    "AgentModel.collision_prefilter fields."
                )
            # Build cache once with per-task timing when metrics requested.
            prefilter = prefilters[0]
            if metrics is not None:
                t0_cache = time.perf_counter()
                # Instrument build_cache to record per-entry times.
                original_build_entry = prefilter._build_entry

                def _timed_build_entry(contour):
                    te0 = time.perf_counter()
                    result = original_build_entry(contour)
                    metrics.cache_build_times_per_task_s.append(
                        time.perf_counter() - te0
                    )
                    return result

                prefilter._build_entry = _timed_build_entry
                prefilter.build_cache(list(toolpath.contours))
                prefilter._build_entry = original_build_entry  # restore
                metrics.record_cache_build(time.perf_counter() - t0_cache)
            else:
                prefilter.build_cache(list(toolpath.contours))

        t0_planning = time.perf_counter()
        while tm.has_frontier():
            tm.mark_inprogress_complete(t)
            idle_agents = set()

            for agent in context.get_agents_with_start_time(t):
                agent_model = self._agent_models[agent]
                feasible = [
                    n
                    for n in tm.get_available_tasks()
                    if tm.contours[n].tool in agent_model.capabilities
                ]

                if not feasible:
                    idle_agents.add(agent)
                    continue

                p_start = schedule[agent].get_state(t, agent_model.home_position)

                if options.task_priority == "farthest_centroid":
                    key = lambda n: _farthest_centroid_key(
                        n, tm.contours, agent, schedule, self._agent_models, t
                    )
                elif options.task_priority == "nearest_start":
                    key = lambda n: _nearest_start_key(n, tm.contours, p_start)
                elif options.task_priority == "nearest_base":
                    key = lambda n: _nearest_base_key(n, tm.contours, agent_model)
                elif options.task_priority == "min_ellipsoid":
                    key = lambda n: _min_ellipsoid_key(n, tm.contours, agent_model)
                elif options.task_priority == "farthest_other_bases":
                    key = lambda n: _farthest_other_bases_key(
                        n, tm.contours, agent, self._agent_models
                    )
                elif options.task_priority == "prune_aware":
                    key = lambda n: _prune_aware_key(
                        n, tm.contours, agent, schedule, self._agent_models, t
                    )
                elif options.task_priority == "hybrid_base_separation":
                    key = lambda n: _hybrid_base_separation_key(
                        n, tm.contours, agent, schedule, self._agent_models,
                        t, options.priority_weight,
                    )
                elif options.task_priority == "hybrid_travel_prune":
                    key = lambda n: _hybrid_travel_prune_key(
                        n, tm.contours, agent, schedule, self._agent_models,
                        t, p_start, options.priority_weight,
                    )
                else:
                    key = lambda n: dg._graph.out_degree(n)
                nodes = sorted(feasible, key=key, reverse=True)  # type: ignore

                for node in nodes:
                    contour = tm.contours[node]
                    events = build_event_chain(t, p_start, contour, agent, context)

                    if events_cause_collision(
                        events,
                        agent,
                        schedule,
                        self._agent_models,
                        options.collision_gap_threshold,
                        metrics=metrics,
                    ):
                        continue

                    # Task allocated — record timing.
                    t_alloc0 = time.perf_counter()

                    # slice home event if overlap
                    if schedule[agent].end_time() > events[0].start:
                        prev_home_event = schedule[agent]._events.pop()
                        if prev_home_event.start != events[0].start:
                            sliced_home = self._slice_home_event(
                                prev_home_event, events[0].start
                            )
                            schedule.add_event(sliced_home, agent)

                    schedule.add_events(events, agent)
                    tm.add_inprogress(node, events[1].end)
                    tm.frontier.remove(node)
                    tm.frontier.update(dg._graph.successors(node))
                    context.positions[agent] = events[2].data[-1]
                    context.set_agent_start_time(agent, events[2].end)

                    if metrics is not None:
                        metrics.record_allocation(time.perf_counter() - t_alloc0)
                    break
                else:
                    # if all cause collisions
                    context.set_agent_start_time(agent, t + options.collision_offset)

            # advance global time
            t = min(tv for tv in context.start_times.values() if tv != t)

            t_feasible = min(tm.in_progress.values(), default=t)
            for agent in idle_agents:
                context.set_agent_start_time(agent, t_feasible)

        if metrics is not None:
            metrics.record_total_planning_time(time.perf_counter() - t0_planning)

        return schedule

    def _validate_toolpath(self, toolpath):
        required_tools = set(toolpath.tools())
        provided_tools = set(
            [cap for a in self._agent_models.values() for cap in a.capabilities]
        )
        if not required_tools.issubset(provided_tools):
            raise ValueError("Agents cannot provide all required capabilities")

    def _slice_home_event(self, home_event: MoveEvent, end_time: float):
        new_traj = home_event.traj.slice(home_event.start, end_time)
        path = [p.data for p in new_traj.points]
        return MoveEvent(home_event.start, path, home_event.velocity)
