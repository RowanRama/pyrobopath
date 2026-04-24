from typing import Dict, Optional

import numpy as np

from .schedule import Event, Schedule, MultiAgentSchedule


def draw_schedule(s: Schedule, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(s.start_time(), s.end_time())
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("time")

    colors = plt.get_cmap("Pastel2")(np.linspace(0.15, 0.85, s.n_events()))

    for event, color in zip(s._events, colors):
        p = ax.barh(
            "agent",
            left=event.start,
            width=event.duration,
            height=0.5,
            edgecolor="black",
            color=color,
        )
        ax.bar_label(p, label_type="center")

    if show:
        plt.show()
    return fig, ax


def draw_multi_agent_schedule(s: MultiAgentSchedule, show=True):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    category_colors = plt.get_cmap("Pastel1")(np.linspace(0.15, 0.85, s.n_agents()))

    for (agent, schedule), color in zip(s.schedules.items(), category_colors):
        for event in schedule._events:
            p = ax.barh(
                agent,
                left=event.start,
                width=event.duration,
                height=0.5,
                edgecolor="black",
                color=color,
            )
            ax.bar_label(p, label_type="center")

    if show:
        plt.show()
    return fig, ax


def _ellipse_patch_from_record(record, safety_m, color, alpha, zorder):
    """Build a ``matplotlib.patches.Ellipse`` for an ``EllipsoidRecord``.

    The ellipse represents the MVEE of the task's polyline, optionally
    inflated by ``safety_m`` metres in world-space on each semi-axis.

    Parameters
    ----------
    record : EllipsoidRecord
        Precomputed ellipsoid geometry for the contour.
    safety_m : float
        Safety inflation in metres (0 = tight MVEE, no inflation).
    color : colour spec
        Fill and edge colour for the patch.
    alpha : float
        Fill transparency.
    zorder : int
        Drawing order.

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    from matplotlib.patches import Ellipse
    import numpy as np

    # Semi-axes (metres) after safety inflation: a_i' = 1/sqrt(λ_i) + safety_m
    _EPS = 1e-9
    axes = 1.0 / np.sqrt(np.maximum(record.evals, _EPS)) + safety_m  # (2,)

    # evecs columns are eigenvectors; the eigenvector for evals[0] (smaller λ =
    # larger axis) gives the direction of the major axis.
    # angle of major-axis eigenvector (column 0) from +x, in degrees.
    major_vec = record.evecs[:, 0]  # direction of largest semi-axis
    angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))

    patch = Ellipse(
        xy=(float(record.mu_e[0]), float(record.mu_e[1])),
        width=2.0 * float(axes[0]),   # full diameter along major axis
        height=2.0 * float(axes[1]),  # full diameter along minor axis
        angle=angle_deg,
        linewidth=1.5,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        zorder=zorder,
    )
    return patch


def animate_multi_agent_schedule(
    schedule,
    agent_models: Dict,
    toolpath=None,
    show: bool = True,
    playback_speed: float = 1.0,
):
    """Interactive matplotlib playback of a MultiAgentToolpathSchedule.

    Renders a 2D XY workspace view (top) with each agent as a coloured circle
    at its interpolated position, a Gantt-style schedule strip with a time
    cursor (middle), and a Play/Pause button + draggable time slider
    (bottom). Dragging the slider pauses playback; releasing resumes if it
    was playing before the drag.

    A **Show Ellipsoids** check-button toggles per-agent MVEE ellipsoid overlays
    for the task each arm is currently executing.  Ellipsoids are drawn using
    the prefilter's cache (``AgentModel.collision_prefilter._cache``) when
    available; if a prefilter is attached they also respect its ``safety_m``
    inflation so the displayed ellipsoid matches the one used for pruning.

    Args:
        schedule: A ``MultiAgentToolpathSchedule``.
        agent_models: Mapping agent-id -> ``AgentModel``.
        toolpath: Optional ``Toolpath``; unstarted contours are drawn faintly.
        show: If True, call ``plt.show()`` before returning.
        playback_speed: Multiplier on wall-clock playback speed (1.0 = real).

    Returns:
        ``(fig, controls)`` where ``controls`` is a dict of the interactive
        widgets (button, slider, timer, check-button, axes) — keep a reference
        to these or they will be garbage-collected and the interaction will die.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, CheckButtons

    agents = list(schedule.schedules.keys())
    n_agents = max(len(agents), 1)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(n_agents, 2)))
    agent_color = {a: colors[i % len(colors)] for i, a in enumerate(agents)}

    t_start = min(
        (s.start_time() for s in schedule.schedules.values() if s._events),
        default=0.0,
    )
    t_end = max(
        (s.end_time() for s in schedule.schedules.values() if s._events),
        default=0.0,
    )
    if t_end <= t_start:
        t_end = t_start + 1.0

    fig = plt.figure(figsize=(10, 9))
    # layout: workspace (62%), timeline (12%), controls row (play btn + check + slider)
    ax_ws  = fig.add_axes([0.08, 0.30, 0.88, 0.64])
    ax_tl  = fig.add_axes([0.08, 0.17, 0.88, 0.10])
    ax_btn = fig.add_axes([0.08, 0.03, 0.10, 0.07])   # play/pause
    ax_chk = fig.add_axes([0.20, 0.01, 0.18, 0.11])   # ellipsoid check-button
    ax_sld = fig.add_axes([0.42, 0.05, 0.55, 0.04])   # time slider

    ax_ws.set_aspect("equal", adjustable="datalim")
    ax_ws.set_xlabel("x")
    ax_ws.set_ylabel("y")
    ax_ws.set_title("Multi-agent schedule playback")

    # gather workspace bounds
    xs, ys = [], []
    if toolpath is not None:
        for contour in toolpath.contours:
            for p in contour.path:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    for a, s in schedule.schedules.items():
        for ev in s._events:
            data = getattr(ev, "data", None)
            if data is None:
                continue
            for p in data:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    if xs and ys:
        pad = max((max(xs) - min(xs)), (max(ys) - min(ys)), 1.0) * 0.1 + 0.5
        ax_ws.set_xlim(min(xs) - pad, max(xs) + pad)
        ax_ws.set_ylim(min(ys) - pad, max(ys) + pad)
    else:
        ax_ws.set_xlim(-1, 1)
        ax_ws.set_ylim(-1, 1)

    # static: faint unstarted contours from toolpath
    if toolpath is not None:
        for contour in toolpath.contours:
            pts = np.array([(float(p[0]), float(p[1])) for p in contour.path])
            if len(pts) >= 2:
                ax_ws.plot(pts[:, 0], pts[:, 1], "-", color="lightgray",
                           linewidth=1.0, alpha=0.6, zorder=1)

    # timeline: Gantt bars + cursor
    ax_tl.set_yticks(range(len(agents)))
    ax_tl.set_yticklabels([str(a) for a in agents])
    ax_tl.set_xlim(t_start, t_end)
    ax_tl.set_xlabel("time")
    for i, a in enumerate(agents):
        for ev in schedule.schedules[a]._events:
            ax_tl.barh(
                i,
                left=ev.start,
                width=ev.duration,
                height=0.6,
                color=agent_color[a],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )
    cursor_line = ax_tl.axvline(t_start, color="red", linewidth=1.5, zorder=10)

    # dynamic workspace artists
    agent_dots = {}
    contour_lines = {}
    for a in agents:
        (dot,) = ax_ws.plot(
            [], [], "o", color=agent_color[a], markersize=14,
            markeredgecolor="black", zorder=5, label=str(a),
        )
        agent_dots[a] = dot
        (line,) = ax_ws.plot(
            [], [], "-", color=agent_color[a], linewidth=2.5, alpha=0.9, zorder=3,
        )
        contour_lines[a] = line
    if agents:
        ax_ws.legend(loc="upper right", fontsize=8)

    state = {"playing": False, "t": t_start, "drag_was_playing": False,
             "suppress_slider_cb": False, "show_ellipsoids": False}

    # ── ellipsoid cache resolution ────────────────────────────────────────────
    # Build a mapping: contour.id -> EllipsoidRecord, sourced from whichever
    # agent's prefilter has a populated cache.  Also read safety_m from the
    # prefilter so the displayed ellipsoid matches the one used for pruning.
    _ellipsoid_cache = {}   # contour.id -> EllipsoidRecord
    _safety_m = 0.0
    for _model in agent_models.values():
        _pf = getattr(_model, "collision_prefilter", None)
        if _pf is None:
            continue
        _cache = getattr(_pf, "_cache", {})
        _safety_m = float(getattr(_pf, "safety_m", 0.0))
        for _cid, _entry in _cache.items():
            # heuristic models store EllipsoidRecord directly;
            # LearnedCachedModel stores an int sentinel — skip those.
            from pyrobopath.collision_detection.ellipsoid_filter import EllipsoidRecord
            if isinstance(_entry, EllipsoidRecord):
                _ellipsoid_cache[_cid] = _entry
        if _ellipsoid_cache:
            break   # one agent's cache is enough — all are identical

    # Active ellipsoid patches: agent -> current Ellipse patch (or None)
    _ellipse_patches = {a: None for a in agents}

    def _clear_ellipse_patches():
        for a in agents:
            if _ellipse_patches[a] is not None:
                try:
                    _ellipse_patches[a].remove()
                except ValueError:
                    pass
                _ellipse_patches[a] = None

    def _agent_position(agent, t):
        s = schedule.schedules[agent]
        model = agent_models.get(agent)
        default = None
        if model is not None and hasattr(model, "home_position"):
            default = np.asarray(model.home_position)
        if not s._events:
            return default if default is not None else np.array([0.0, 0.0, 0.0])
        if t >= s.end_time():
            pos = s.get_state(s.end_time() - 1e-9, default)
        else:
            pos = s.get_state(t, default)
        return pos if pos is not None else (default if default is not None
                                            else np.array([0.0, 0.0, 0.0]))

    def _active_contour_event(agent, t):
        """Return the active ContourEvent for agent at time t, or None."""
        from pyrobopath.toolpath_scheduling.schedule import ContourEvent
        for ev in schedule.schedules[agent]._events:
            if isinstance(ev, ContourEvent) and ev.start <= t <= ev.end:
                return ev
        return None

    def _update_display(t):
        _clear_ellipse_patches()
        for a in agents:
            pos = _agent_position(a, t)
            try:
                x = float(pos[0])
                y = float(pos[1])
            except (TypeError, IndexError):
                x, y = 0.0, 0.0
            agent_dots[a].set_data([x], [y])

            ev = _active_contour_event(a, t)
            if ev is not None:
                pts = np.array([(float(p[0]), float(p[1])) for p in ev.contour.path])
                contour_lines[a].set_data(pts[:, 0], pts[:, 1])

                # Ellipsoid overlay (only when enabled and cache entry exists)
                if state["show_ellipsoids"] and ev.contour.id in _ellipsoid_cache:
                    record = _ellipsoid_cache[ev.contour.id]
                    patch = _ellipse_patch_from_record(
                        record,
                        safety_m=_safety_m,
                        color=agent_color[a],
                        alpha=0.18,
                        zorder=2,
                    )
                    ax_ws.add_patch(patch)
                    _ellipse_patches[a] = patch
            else:
                contour_lines[a].set_data([], [])

        cursor_line.set_xdata([t, t])
        fig.canvas.draw_idle()

    # Controls
    btn_play = Button(ax_btn, "▶ Play")
    check_ellipsoid = CheckButtons(ax_chk, ["Show ellipsoids"], [False])
    # Style the check-button axis to blend with the figure background
    ax_chk.set_facecolor(fig.get_facecolor())
    for spine in ax_chk.spines.values():
        spine.set_visible(False)

    slider = Slider(ax_sld, "t", t_start, t_end, valinit=t_start,
                    valstep=(t_end - t_start) / 500.0)

    timer = fig.canvas.new_timer(interval=33)

    def _tick():
        if not state["playing"]:
            return
        dt = 0.033 * playback_speed
        new_t = min(state["t"] + dt, t_end)
        state["t"] = new_t
        state["suppress_slider_cb"] = True
        slider.set_val(new_t)
        state["suppress_slider_cb"] = False
        _update_display(new_t)
        if new_t >= t_end:
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            timer.stop()

    def _on_play_pause(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text("⏸ Pause" if state["playing"] else "▶ Play")
        if state["playing"]:
            if state["t"] >= t_end:
                state["t"] = t_start
                slider.set_val(t_start)
            timer.start()
        else:
            timer.stop()

    def _on_slider_changed(val):
        state["t"] = float(val)
        _update_display(float(val))

    def _on_press(event):
        if event.inaxes is ax_sld:
            state["drag_was_playing"] = state["playing"]
            if state["playing"]:
                state["playing"] = False
                timer.stop()
                btn_play.label.set_text("▶ Play")

    def _on_release(event):
        if state["drag_was_playing"]:
            state["playing"] = True
            btn_play.label.set_text("⏸ Pause")
            timer.start()
        state["drag_was_playing"] = False

    def _on_check_ellipsoid(label):
        """Toggle ellipsoid visibility and immediately redraw."""
        state["show_ellipsoids"] = not state["show_ellipsoids"]
        if not state["show_ellipsoids"]:
            _clear_ellipse_patches()
            fig.canvas.draw_idle()
        else:
            _update_display(state["t"])

    timer.add_callback(_tick)
    slider.on_changed(_on_slider_changed)
    btn_play.on_clicked(_on_play_pause)
    check_ellipsoid.on_clicked(_on_check_ellipsoid)
    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("button_release_event", _on_release)

    _update_display(t_start)

    controls = {
        "button": btn_play,
        "check_ellipsoid": check_ellipsoid,
        "slider": slider,
        "timer": timer,
        "ax_workspace": ax_ws,
        "ax_timeline": ax_tl,
        "ax_check": ax_chk,
        "state": state,
        "ellipsoid_cache": _ellipsoid_cache,
    }

    if show:
        plt.show()
    return fig, controls


if __name__ == "__main__":
    schedule = Schedule()
    schedule.add_event(Event("eventA", 0.0, 5.0))
    schedule.add_event(Event("eventB", 5.0, 2.0))
    schedule.add_event(Event("eventC", 7.0, 5.0))
    schedule.add_event(Event("eventD", 12.0, 10.0))
    schedule.add_event(Event("eventE", 22.0, 45.0))
    schedule.add_event(Event("eventF", 67.0, 15.0))

    draw_schedule(schedule)

    schedule = MultiAgentSchedule()
    schedule.add_event(Event("eventA1", -1.0, 5.0), "agent1")
    schedule.add_event(Event("eventB1", 5.0, 2.0), "agent1")
    schedule.add_event(Event("eventC1", 7.0, 5.0), "agent1")
    schedule.add_event(Event("eventD1", 12.0, 10.0), "agent1")
    schedule.add_event(Event("eventE1", 22.0, 45.0), "agent1")
    schedule.add_event(Event("eventF1", 67.0, 15.0), "agent1")

    schedule.add_event(Event("eventA2", 0.0, 5.0), "agent2")
    schedule.add_event(Event("eventB2", 5.0, 4.0), "agent2")
    schedule.add_event(Event("eventC2", 9.0, 10.0), "agent2")
    schedule.add_event(Event("eventD2", 19.0, 10.0), "agent2")
    schedule.add_event(Event("eventE2", 67.0, 16.0), "agent2")

    other = Schedule()
    other.add_event(Event("eventA3", -2.0, 5.0))
    other.add_event(Event("eventB3", 70.0, 20.0))
    schedule.add_schedule(other, "agent3")

    draw_multi_agent_schedule(schedule)
