from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.gridspec import GridSpec

from pyrobopath.process import AgentModel
from pyrobopath.toolpath import Toolpath
from pyrobopath.collision_detection import (
    FCLRobotBBCollisionModel,
)

from .schedule import ContourEvent, MultiAgentToolpathSchedule, ToolpathSchedule

_ELLIPSE_EPS = 1e-9


def _ellipse_patch_from_record(record, safety_m, color, alpha, zorder,
                               filled=True, linewidth=1.5):
    """Build a matplotlib Ellipse for an EllipsoidRecord.

    ``safety_m`` is added uniformly to each semi-axis before drawing, matching
    the same inflation the prefilter applies at query time.
    """
    from matplotlib.patches import Ellipse

    # evals from eigh are ascending: evals[0] = smallest λ = largest semi-axis
    axes = 1.0 / np.sqrt(np.maximum(record.evals, _ELLIPSE_EPS)) + safety_m
    major_vec = record.evecs[:, 0]   # eigenvector of largest semi-axis
    angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    return Ellipse(
        xy=(float(record.mu_e[0]), float(record.mu_e[1])),
        width=2.0 * float(axes[0]),
        height=2.0 * float(axes[1]),
        angle=angle_deg,
        linewidth=linewidth,
        edgecolor=color,
        facecolor=color if filled else "none",
        alpha=alpha,
        zorder=zorder,
    )


# temporary fix to override user-specific rcParams that distort scheduling
# animations
import matplotlib as mpl

mpl.rcParams = mpl.rcParamsDefault


def draw_multi_agent_schedule(s: MultiAgentToolpathSchedule, show=True):
    fig, ax = plt.subplots(figsize=(9, 4))

    agent_colors = _agent_color_map(s)
    _plot_multi_agent_schedule(s, ax, agent_colors=agent_colors)

    ax.set_xlabel("Time")
    ax.set_title("Multi-agent Schedule")
    if show:
        plt.show()
    return fig, ax


def _agent_color_map(schedule: MultiAgentToolpathSchedule):
    """Return a dict mapping agent id -> RGBA colour (tab10 palette)."""
    agents = list(schedule.schedules.keys())
    n = max(len(agents), 2)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, n))
    return {a: colors[i % n] for i, a in enumerate(agents)}


def _plot_multi_agent_schedule(s: MultiAgentToolpathSchedule, ax,
                               agent_colors: dict = None):
    if agent_colors is None:
        agent_colors = _agent_color_map(s)

    for agent, schedule in s.schedules.items():
        color = agent_colors[agent]
        for event in schedule._events:
            bar_color = color if isinstance(event, ContourEvent) else "lightgrey"
            ax.barh(
                agent,
                left=event.start,
                width=event.duration,
                height=0.5,
                edgecolor="black",
                linewidth=0.6,
                color=bar_color,
            )


def animate_multi_agent_toolpath_schedule(
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[str, AgentModel],
    step,
    plot_toolpath=True,
    show=True,
):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_xlim((-6, 6))
    ax.set_ylim((-3, 3))
    ax.autoscale_view(False)

    # create models to animate
    anim_models = []
    for a in schedule.agents():
        model = None
        if isinstance(agent_models[a].collision_model, FCLRobotBBCollisionModel):
            model = RobotBBAnimationModel(agent_models[a], schedule[a], ax)
        else:
            model = AnimationModel(agent_models[a], schedule[a], ax)
        anim_models.append(model)

    # update all models on slider change
    def update(val):
        for model in anim_models:
            model.update(val)
        fig.canvas.draw_idle()

    # add slider control
    axtime = plt.axes((0.25, 0.1, 0.65, 0.03))
    anim_slider = Slider(
        ax=axtime,
        label="time",
        valmin=schedule.start_time(),
        valmax=schedule.end_time(),
        valstep=step,
        valinit=schedule.start_time(),
    )
    anim_slider.on_changed(update)
    update(schedule.start_time())
    ax.set_aspect("equal")

    if show:
        plt.show()
    return fig, ax


def animate_multi_agent_toolpath_full(
    toolpath: Toolpath,
    schedule: MultiAgentToolpathSchedule,
    agent_models: Dict[str, AgentModel],
    step=0.01,
    limits=((-500, 500), (-500, 500)),
    show=True,
    playback_speed: float = 10.0,
):
    t_start = schedule.start_time()
    t_end = schedule.end_time()

    # ── figure layout ──────────────────────────────────────────────────────────
    # rows: [schedule Gantt, workspace, play+check row, time slider]
    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(
        4, 2,
        height_ratios=[1, 4, 0.45, 0.25],
        width_ratios=[1, 50],
        hspace=0.35, wspace=0.05,
    )
    sched_ax  = fig.add_subplot(gs[0, :])
    anim_ax   = fig.add_subplot(gs[1, 1])
    layer_ax  = fig.add_subplot(gs[1, 0])
    ctrl_ax   = fig.add_subplot(gs[2, :])   # play btn + check btn row
    slider_ax = fig.add_subplot(gs[3, :])

    ctrl_ax.set_visible(False)   # invisible container — children placed via fig.add_axes

    # ── agent colour map (shared by Gantt, workspace, ellipsoids, layer view) ──
    agents = list(schedule.schedules.keys())
    agent_color = _agent_color_map(schedule)

    # Build contour_id -> agent lookup from the schedule
    contour_agent: dict = {}
    for a, sched in schedule.schedules.items():
        for ev in sched._events:
            if isinstance(ev, ContourEvent):
                contour_agent[ev.contour.id] = a

    # ── Gantt schedule ─────────────────────────────────────────────────────────
    _plot_multi_agent_schedule(schedule, sched_ax, agent_colors=agent_color)
    (sched_cursor,) = sched_ax.plot(
        [], [], lw=2, color=(0, 1, 0.31),
        path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()],
    )
    sched_ax.set_xlabel("Time")
    sched_ax.set_title("Multi-agent Schedule")

    # ── toolpath layer display (coloured by assigned robot) ───────────────────
    contour_z = []
    for contour in toolpath.contours:
        z_values = np.sort(np.array(contour.path)[:, 2])
        contour_z.append(z_values[0])
    unique_z = sorted(set(contour_z))

    contour_lines = []

    def update_layer(val):
        for line in contour_lines:
            line.pop(0).remove()
        contour_lines.clear()
        z_height = unique_z[int(val) - 1]
        for idx, contour in enumerate(toolpath.contours):
            if contour_z[idx] != z_height:
                continue
            path = np.array(contour.path)
            assigned = contour_agent.get(contour.id)
            color = agent_color[assigned] if assigned is not None else "lightgrey"
            contour_lines.append(
                anim_ax.plot(
                    path[:, 0], path[:, 1],
                    path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
                    color=color,
                    zorder=0,
                )
            )

    layer_slider = Slider(
        ax=layer_ax, label="Layer",
        valmin=1, valmax=max(len(unique_z), 1), valstep=1,
        orientation="vertical",
    )
    layer_slider.on_changed(update_layer)
    update_layer(1)

    # ── workspace ─────────────────────────────────────────────────────────────
    anim_ax.set_xlim(limits[0])
    anim_ax.set_ylim(limits[1])
    anim_ax.autoscale_view(False)
    anim_ax.set_aspect("equal")

    anim_models = []
    for a in agents:
        color = agent_color[a]
        if isinstance(agent_models[a].collision_model, FCLRobotBBCollisionModel):
            m = RobotBBAnimationModel(agent_models[a], schedule[a], anim_ax, color=color)
        else:
            m = AnimationModel(agent_models[a], schedule[a], anim_ax, color=color)
        anim_models.append(m)

    # ── ellipsoid cache ────────────────────────────────────────────────────────
    try:
        from pyrobopath.collision_detection.ellipsoid_filter import EllipsoidRecord as _EllipsoidRecord
    except ImportError:
        _EllipsoidRecord = None

    _ellipsoid_cache: dict = {}
    _safety_m = 0.0
    for _model in agent_models.values():
        _pf = getattr(_model, "collision_prefilter", None)
        if _pf is None:
            continue
        _safety_m = float(getattr(_pf, "safety_m", 0.0))
        for _cid, _entry in getattr(_pf, "_cache", {}).items():
            if _EllipsoidRecord is not None and isinstance(_entry, _EllipsoidRecord):
                _ellipsoid_cache[_cid] = _entry
        if _ellipsoid_cache:
            break

    if not _ellipsoid_cache:
        print(
            "[animate] No ellipsoid cache found — "
            "attach a CachedCollisionModel prefilter to agent_models to enable ellipsoid overlay."
        )

    # Each agent slot holds a list of active Ellipse patches (tight + inflated).
    _ellipse_patches: dict = {a: [] for a in agents}

    def _clear_ellipses():
        for a in agents:
            for p in _ellipse_patches[a]:
                try:
                    p.remove()
                except ValueError:
                    pass
            _ellipse_patches[a] = []

    def _active_contour_event(agent, t):
        for ev in schedule.schedules[agent]._events:
            if isinstance(ev, ContourEvent) and ev.start <= t <= ev.end:
                return ev
        return None

    # ── interactive controls ───────────────────────────────────────────────────
    # Manually placed so they sit in the ctrl_ax row without fighting tight_layout
    ctrl_bb = ctrl_ax.get_position()
    btn_ax  = fig.add_axes([ctrl_bb.x0,            ctrl_bb.y0, 0.09, ctrl_bb.height])
    chk_ax  = fig.add_axes([ctrl_bb.x0 + 0.10,     ctrl_bb.y0, 0.16, ctrl_bb.height])

    btn_play       = Button(btn_ax, "▶ Play")
    check_ellipsoid = CheckButtons(chk_ax, ["Show ellipsoids"], [False])
    chk_ax.set_facecolor(fig.get_facecolor())
    for spine in chk_ax.spines.values():
        spine.set_visible(False)

    anim_slider = Slider(
        ax=slider_ax, label="time",
        valmin=t_start, valmax=t_end,
        valstep=step, valinit=t_start,
    )

    state = {
        "playing": False, "t": t_start,
        "drag_was_playing": False,
        "suppress_slider_cb": False,
        "show_ellipsoids": False,
    }

    timer = fig.canvas.new_timer(interval=33)

    def _update_display(t):
        _clear_ellipses()
        for a, anim_m in zip(agents, anim_models):
            anim_m.update(t)
            if state["show_ellipsoids"]:
                ev = _active_contour_event(a, t)
                if ev is not None and ev.contour.id in _ellipsoid_cache:
                    record = _ellipsoid_cache[ev.contour.id]
                    color = agent_color[a]

                    # tight MVEE — filled, shows the bare contour footprint
                    tight = _ellipse_patch_from_record(
                        record, safety_m=0.0,
                        color=color, alpha=0.25, zorder=2,
                        filled=True, linewidth=0,
                    )
                    anim_ax.add_patch(tight)
                    _ellipse_patches[a].append(tight)

                    # inflated MVEE — outline only, shows the safety margin
                    if _safety_m > 0.0:
                        inflated = _ellipse_patch_from_record(
                            record, safety_m=_safety_m,
                            color=color, alpha=0.8, zorder=2,
                            filled=False, linewidth=1.5,
                        )
                        inflated.set_linestyle("--")
                        anim_ax.add_patch(inflated)
                        _ellipse_patches[a].append(inflated)

        sched_cursor.set_data([t, t], [-3, 3])
        fig.canvas.draw_idle()

    def _tick():
        if not state["playing"]:
            return
        new_t = min(state["t"] + 0.033 * playback_speed, t_end)
        state["t"] = new_t
        state["suppress_slider_cb"] = True
        anim_slider.set_val(new_t)
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
                anim_slider.set_val(t_start)
            timer.start()
        else:
            timer.stop()

    def _on_slider_changed(val):
        if state["suppress_slider_cb"]:
            return
        state["t"] = float(val)
        _update_display(float(val))

    def _on_press(event):
        if event.inaxes is slider_ax:
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
        state["show_ellipsoids"] = not state["show_ellipsoids"]
        if not state["show_ellipsoids"]:
            _clear_ellipses()
            fig.canvas.draw_idle()
        else:
            _update_display(state["t"])

    timer.add_callback(_tick)
    anim_slider.on_changed(_on_slider_changed)
    btn_play.on_clicked(_on_play_pause)
    check_ellipsoid.on_clicked(_on_check_ellipsoid)
    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("button_release_event", _on_release)

    _update_display(t_start)

    # keep references so widgets aren't GC'd
    _controls = {
        "button": btn_play,
        "check_ellipsoid": check_ellipsoid,
        "layer_slider": layer_slider,
        "anim_slider": anim_slider,
        "timer": timer,
        "state": state,
        "ellipsoid_cache": _ellipsoid_cache,
    }

    if show:
        plt.show()
    return fig, _controls


class AnimationModel(object):
    def __init__(self, agent_model: AgentModel, schedule: ToolpathSchedule, ax,
                 color="steelblue"):
        self.model = agent_model
        self.sched = schedule
        self.ax = ax

        (self.line,) = ax.plot([], [], lw=2, color=color)

    def update(self, t):
        pos = self.sched.get_state(t, default=self.model.home_position)
        self.line.set_data(
            [self.model.base_frame_position[0], pos[0]],
            [self.model.base_frame_position[1], pos[1]],
        )


class RobotBBAnimationModel(AnimationModel):
    def __init__(self, agent_model: AgentModel, schedule: ToolpathSchedule, ax,
                 color="steelblue"):
        super(RobotBBAnimationModel, self).__init__(agent_model, schedule, ax, color=color)

        # create bounding box
        self.dim = agent_model.collision_model.dims
        self.rect = patches.Rectangle(
            (0, 0), self.dim[0], self.dim[1], fill=None, linewidth=2
        )
        ax.add_patch(self.rect)

        # modify attach line
        (self.line,) = ax.plot([], [], lw=2, linestyle="--", color="black")

        # create base marker
        r = 0.3 * self.dim[1]
        bf = self.model.base_frame_position[:2]
        hatch = r * np.cos(np.pi / 4)
        base = patches.Circle(
            self.model.base_frame_position[:2], r,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(base)
        ax.plot(
            (bf[0] - hatch, bf[0] + hatch),
            (bf[1] - hatch, bf[1] + hatch),
            linewidth=1, color=color,
        )
        ax.plot(
            (bf[0] - hatch, bf[0] + hatch),
            (bf[1] + hatch, bf[1] - hatch),
            linewidth=1, color=color,
        )

    def update(self, t):
        pos = self.sched.get_state(t, default=self.model.home_position)
        self.model.collision_model.translation = pos

        # end-effector in world frame
        T_w_e = np.identity(3)
        T_w_e[:2, :2] = self.model.collision_model.rotation[:2, :2]
        T_w_e[:2, 2] = self.model.collision_model.translation[:2]

        # bottom-left corner in end-effector frame
        T_e_bl = np.identity(3)
        T_e_bl[:2, 2] = self.model.collision_model.offset[:2] + np.array(
            [-self.dim[0], -self.dim[1] / 2]
        )

        T_w_bl = T_w_e @ T_e_bl
        tf = transforms.Affine2D(T_w_bl)
        self.rect.set_transform(tf + self.ax.transData)

        self.line.set_data(
            [self.model.base_frame_position[0], pos[0]],
            [self.model.base_frame_position[1], pos[1]],
        )
