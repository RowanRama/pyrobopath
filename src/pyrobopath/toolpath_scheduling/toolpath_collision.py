from typing import List, Dict, Hashable, Optional
import numpy as np

from pyrobopath.process import AgentModel
from pyrobopath.collision_detection import (
    Trajectory,
    TrajectoryPoint,
    trajectory_collision_query,
)
from pyrobopath.scheduling import Interval

from .schedule import ContourEvent, MoveEvent, ToolpathSchedule, MultiAgentToolpathSchedule


def _move_segments_xy(event: MoveEvent):
    """Yield consecutive (p, q) 2-D segment pairs from a MoveEvent's path."""
    path = event.data
    for i in range(len(path) - 1):
        p = np.asarray(path[i][:2],   dtype=float)
        q = np.asarray(path[i+1][:2], dtype=float)
        yield p, q


def _travel_intersects_ellipsoids(move_events, contour_events, cache, safety_m):
    """Return True if any segment in move_events passes through the inflated
    ellipsoid of any contour in contour_events.

    Used to check whether travel/depart/home moves cross the other agent's
    active task regions — the gap not covered by the contour-pair prefilter.
    """
    try:
        from pyrobopath.collision_detection.ellipsoid_filter import (
            _segment_intersects_ellipsoid,
        )
    except ImportError:
        return False   # can't check — be conservative and allow FCL

    for ev in move_events:
        if not isinstance(ev, MoveEvent) or isinstance(ev, ContourEvent):
            continue
        for p, q in _move_segments_xy(ev):
            for ce in contour_events:
                record = cache.get(ce.contour.id)
                if record is None or not hasattr(record, "mu_e"):
                    return True   # no cached geometry — be conservative
                M_inf = record.inflated_M(safety_m)
                if _segment_intersects_ellipsoid(p, q, record.mu_e, M_inf):
                    return True
    return False


def _all_contour_pairs_prefiltered_safe(
    prefilter,
    events_a,
    events_b,
    base_a,
    base_b,
    t_start: float = None,
    t_end: float = None,
) -> bool:
    """Return True only if the prefilter can certify the full event chain in
    events_a as safe against all temporally overlapping events in events_b.

    Two checks are performed:

    1. Contour-pair check — every ContourEvent in events_a is certified safe
       against every temporally overlapping ContourEvent in events_b using the
       cached ellipsoid filter.

    2. Travel-segment check — every non-contour MoveEvent (travel, depart,
       home) in events_a is checked for intersection with the inflated
       ellipsoids of concurrent ContourEvents in events_b, and vice versa.
       This catches cases where a robot's transition moves pass through a
       region the other robot is actively printing.

    Returns False (fall back to FCL) if either check fails or if the necessary
    cache data is unavailable.
    """
    if t_start is not None and t_end is not None:
        concurrent_b = [
            e for e in events_b
            if isinstance(e, ContourEvent) and e.start < t_end and e.end > t_start
        ]
    else:
        concurrent_b = [e for e in events_b if isinstance(e, ContourEvent)]

    contours_a = [e.contour.id for e in events_a if isinstance(e, ContourEvent)]
    contours_b = [e.contour.id for e in concurrent_b]

    if not contours_a or not contours_b:
        return False

    # 1. Contour-pair spatial check
    for id_a in contours_a:
        for id_b in contours_b:
            if not prefilter.is_safe_pair(id_a, id_b, base_a, base_b):
                return False

    # 2. Travel-segment intersection check (A's moves vs B's concurrent contours)
    cache = getattr(prefilter, "_cache", {})
    safety_m = float(getattr(prefilter, "safety_m", 0.0))

    if _travel_intersects_ellipsoids(events_a, concurrent_b, cache, safety_m):
        return False

    # Also check B's move events vs A's concurrent contours
    if t_start is not None and t_end is not None:
        b_moves = [
            e for e in events_b
            if isinstance(e, MoveEvent)
            and not isinstance(e, ContourEvent)
            and e.start < t_end and e.end > t_start
        ]
    else:
        b_moves = [
            e for e in events_b
            if isinstance(e, MoveEvent) and not isinstance(e, ContourEvent)
        ]

    contour_events_a = [e for e in events_a if isinstance(e, ContourEvent)]
    if _travel_intersects_ellipsoids(b_moves, contour_events_a, cache, safety_m):
        return False

    return True


def schedule_to_trajectory(
    schedule: ToolpathSchedule, t_start: float, t_end: float, default_state
) -> Trajectory:
    """Slice a toolpath schedule to a continuous trajectory.

    The trajectory is guaranteed to have points at times t_start and t_end.
    The trajectories in any ContourEvent that fall in the time window are
    concatenated in the resulting trajectory

    :param schedule: The schedule to be sliced
    :type schedule: ToolpathSchedule
    :param t_start: Start time of slice
    :type t_start: float
    :param t_end: End time of slice
    :type t_end: float
    :param default_state: The default trajectory state for times with no known
                          state in the schedule
    :type default_state: np.ndarray

    :return: The trajectory inferred from the schedule
    :rtype: Trajectory
    """

    traj = Trajectory()
    traj.add_traj_point(TrajectoryPoint(np.empty(3), float("nan")))
    for e in schedule._events:
        if e.end < t_start:
            continue
        if e.start >= t_end:
            break
        event_traj = e.traj.slice(max(e.start, t_start), min(e.end, t_end))

        # remove duplicates between contiguous events
        if event_traj[0].time == traj.points[-1].time:
            event_traj.points.pop(0)

        traj += event_traj
    traj.points.pop(0)

    # add endpoints if traj_start != t_start or traj_end != t_end
    start_state, end_state = None, None
    if traj.n_points():
        if traj.start_time() > t_start:
            start_state = schedule.get_state(t_start, default_state)
        if traj.end_time() < t_end:
            end_state = schedule.get_state(t_end, default_state)
    # add last known states as start and end of traj
    else:
        start_state = schedule.get_state(t_start, default_state)
        end_state = schedule.get_state(t_end, default_state)

    if start_state is not None:
        traj.insert_traj_point(0, TrajectoryPoint(start_state, t_start))
    if end_state is not None:
        traj.add_traj_point((TrajectoryPoint(end_state, t_end)))
    return traj


def schedule_to_trajectories(
    schedule: ToolpathSchedule, t_start: float, t_end: float
) -> List[Trajectory]:
    """Slice a toolpath schedule to a list of continuous trajectories.

    The trajectories of MoveEvents are sliced to fit in the window
    [t_start, t_end].

    :param schedule: The schedule to be sliced
    :type schedule: ToolpathSchedule
    :param t_start: start time of slice
    :type t_start: float
    :param t_end: end time of slice
    :type t_end: float

    :return: The list trajectories inferred from the schedule
    :rtype: List[Trajectory]
    """

    trajs = []
    interval = Interval(t_start, t_end)
    for e in schedule._events:
        # event is not in interval
        if e.precedes(interval):
            continue
        if e.preceded_by(interval):
            break

        # use Allen's interval algebra to determine which parts of the
        # event's trajectory should be added to the list
        traj = Trajectory()
        if e.meets(interval):
            traj.add_traj_point(e.traj.points[-1])
        elif e.overlaps(interval):
            traj = e.traj.slice(interval.start, e.end)
        elif (
            e.starts(interval)
            or e.during(interval)
            or e.finishes(interval)
            or e.equals(interval)
        ):
            traj = e.traj
        elif e.finished_by(interval) or e.contains(interval) or e.started_by(interval):
            traj = e.traj.slice(interval.start, interval.end)
        elif e.overlapped_by(interval):
            traj = e.traj.slice(e.start, interval.end)
        elif e.met_by(interval):
            traj.add_traj_point(e.traj.points[0])
        trajs.append(traj)

    return trajs


def event_causes_collision(
    event: ContourEvent,
    agent: Hashable,
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[Hashable, AgentModel],
    threshold: float,
    metrics=None,
):
    """Determines if adding 'event' to 'schedule' will cause a collision in the
    resulting trajectory.

    The events in schedule from event.start_time() to schedule.end_time() are
    checked for collision. This is necessary because adding an event at a
    previous time can cause collisions in the future.

    :param event: The event to be added (with Event.data = Contour)
    :type event: ContourEvent
    :param agent: The agent for the event
    :type agent: Hashable
    :param schedule: A schedule that is assumed collision-free
    :type schedule: MultiAgentToolpathSchedule
    :param agent_models: Context info about the system
    :type agent_models: Dict[str, AgentModel]
    :param threshold: The maximum collision checking step distance (in units
                      equivalent to points in `event`)
    :type threshold: float

    :return: True if event causes collision, False else
    :rtype: bool
    """

    et = max(schedule.end_time(), event.end)

    new_sched = ToolpathSchedule()
    new_sched.add_event(event)
    event_traj = schedule_to_trajectory(
        new_sched, event.start, et, agent_models[agent].home_position
    )

    prefilter = agent_models[agent].collision_prefilter
    base_a = np.array(agent_models[agent].base_frame_position)

    for a, s in schedule.schedules.items():
        if a == agent:
            continue

        if metrics is not None:
            metrics.record_pair_checked()

        if prefilter is not None:
            base_b = np.array(agent_models[a].base_frame_position)
            if _all_contour_pairs_prefiltered_safe(
                prefilter, [event], s._events, base_a, base_b,
                t_start=event.start, t_end=et,
            ):
                if metrics is not None:
                    metrics.record_prefilter_pruned()
                continue

        if metrics is not None:
            metrics.record_fcl_checked()
        traj = schedule_to_trajectory(s, event.start, et, agent_models[a].home_position)
        collide = trajectory_collision_query(
            agent_models[agent].collision_model,
            event_traj,
            agent_models[a].collision_model,
            traj,
            threshold,
        )
        if collide:
            if metrics is not None:
                metrics.record_collision_detected()
            return True
    return False


def concurrent_trajectory_pairs(tr1: List[Trajectory], tr2: List[Trajectory]):
    """Finds the intersections of intervals in two lists of trajectories.
    Each trajectory is sliced according to the interval intersections.

    Runs in O(n + m)

    :param tr1: The first list of trajectories
    :type tr1: List[Trajectory]
    :param tr2: The second list of trajectories
    :type tr2: List[Trajectory]

    :return: A list of concurrent trajectory pairs
    :rtype: List[Tuple(Trajectory, Trajectory)]
    """

    i = j = 0
    traj_pairs = []
    while i < len(tr1) and j < len(tr2):
        a_start, a_end = tr1[i].start_time(), tr1[i].end_time()
        b_start, b_end = tr2[j].start_time(), tr2[j].end_time()
        if a_start <= b_end and b_start <= a_end:
            st = max(a_start, b_start)
            et = min(a_end, b_end)
            traj_pairs.append((tr1[i].slice(st, et), tr2[j].slice(st, et)))

        if a_end < b_end:
            i += 1
        else:
            j += 1
    return traj_pairs


def events_cause_collision(
    events: List[ContourEvent],
    agent: Hashable,
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[Hashable, AgentModel],
    threshold: float,
    metrics=None,
):
    """
    Determines if adding the 'events' to 'schedule' will cause a collision in the
    resulting trajectory.

    For timesteps with no event, agents are assumed to be in collision-free positions

    Args:
        events (List[ContourEvent]): The events to be added
        agent (Hashable): The agent for the event
        schedule (MultiAgentSchedule): A schedule that is assumed collision-free
        agent_models (Dict[str, AgentModel]): Context info about the system
        threshold: The maximum collision checking step distance
    """
    st = min([e.start for e in events])
    et = max([e.end for e in events])

    new_sched = ToolpathSchedule()
    for event in events:
        new_sched.add_event(event)

    event_trajs = schedule_to_trajectories(new_sched, st, et)

    prefilter = agent_models[agent].collision_prefilter
    base_a = np.array(agent_models[agent].base_frame_position)

    for a, s in schedule.schedules.items():
        if a == agent:
            continue

        if metrics is not None:
            metrics.record_pair_checked()

        if prefilter is not None:
            base_b = np.array(agent_models[a].base_frame_position)
            if _all_contour_pairs_prefiltered_safe(
                prefilter, events, s._events, base_a, base_b,
                t_start=st, t_end=et,
            ):
                if metrics is not None:
                    metrics.record_prefilter_pruned()
                continue

        if metrics is not None:
            metrics.record_fcl_checked()
        trajs = schedule_to_trajectories(s, st, et)
        concurrent_trajs = concurrent_trajectory_pairs(event_trajs, trajs)
        for pair in concurrent_trajs:
            collide = trajectory_collision_query(
                agent_models[agent].collision_model,
                pair[0],
                agent_models[a].collision_model,
                pair[1],
                threshold,
            )

            if collide:
                if metrics is not None:
                    metrics.record_collision_detected()
                return True
    return False
