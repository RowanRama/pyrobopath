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


def _build_contour_agent_map(schedule: MultiAgentToolpathSchedule) -> dict:
    """Return a ``{contour_id: agent}`` dict for every ContourEvent in ``schedule``."""
    out: dict = {}
    for agent, sched in schedule.schedules.items():
        for ev in sched._events:
            if isinstance(ev, ContourEvent):
                out[ev.contour.id] = agent
    return out


def _collect_ellipsoid_cache(agent_models: Dict[str, AgentModel]):
    """Return ``(cache, safety_m)`` pulled from any attached prefilter."""
    try:
        from pyrobopath.collision_detection.ellipsoid_filter import (
            EllipsoidRecord as _EllipsoidRecord,
        )
    except ImportError:
        _EllipsoidRecord = None

    cache: dict = {}
    safety_m = 0.0
    for model in agent_models.values():
        pf = getattr(model, "collision_prefilter", None)
        if pf is None:
            continue
        safety_m = float(getattr(pf, "safety_m", 0.0))
        for cid, entry in getattr(pf, "_cache", {}).items():
            if _EllipsoidRecord is not None and isinstance(entry, _EllipsoidRecord):
                cache[cid] = entry
        if cache:
            break
    return cache, safety_m


def _contour_z_heights(toolpath: Toolpath):
    """Return ``(per_contour_z, sorted_unique_z)`` for a toolpath."""
    contour_z = [
        float(np.sort(np.array(c.path)[:, 2])[0]) for c in toolpath.contours
    ]
    return contour_z, sorted(set(contour_z))


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
    replan_fn=None,
):
    """Interactive playback of a multi-agent toolpath schedule.

    When ``replan_fn`` is provided, the UI gains a per-agent enable/disable
    panel plus two buttons:

    - **Replan at t=cursor** — calls ``replan_fn(t_replan, disabled_agents)``
      and swaps in the returned schedule.  Tasks completed before ``t_replan``
      become grey in the layer view.  Disabled agents stay at their home
      position.
    - **Reset** — calls ``replan_fn(0.0, disabled_agents)`` to produce a fresh
      schedule with the current enable/disable selection.  The completed
      override is cleared.

    Parameters
    ----------
    replan_fn : Callable[[float, Set[str]], MultiAgentToolpathSchedule], optional
        When provided, enables the replan UI.  Receives the replan start time
        and the set of currently disabled agent ids.
    """
    axes = _create_animate_layout(fig_size=(13, 10), has_replan=replan_fn is not None)
    fig = axes["fig"]

    # Immutable collaborators
    agents = list(schedule.schedules.keys())
    agent_color = _agent_color_map(schedule)
    contour_z, unique_z = _contour_z_heights(toolpath)

    # Mutable view — everything re-rendered on replan lives here
    view = _ScheduleView(
        schedule=schedule,
        contour_agent=_build_contour_agent_map(schedule),
        anim_models=_build_anim_models(agents, schedule, agent_models, axes["anim"], agent_color),
        ellipsoid_cache={}, ellipsoid_safety=0.0,
    )
    cache, safety = _collect_ellipsoid_cache(agent_models)
    view.ellipsoid_cache = cache
    view.ellipsoid_safety = safety
    if not cache:
        print(
            "[animate] No ellipsoid cache found — "
            "attach a CachedCollisionModel prefilter to agent_models to enable ellipsoid overlay."
        )

    # Workspace setup (static)
    axes["anim"].set_xlim(limits[0])
    axes["anim"].set_ylim(limits[1])
    axes["anim"].autoscale_view(False)
    axes["anim"].set_aspect("equal")

    # Initial Gantt and title.  The cursor is held in a single-element list so
    # _rerender_gantt can swap it for a fresh Line2D after axes.clear().
    _render_gantt(view.schedule, axes["sched"], agent_color)
    sched_cursor_box = [_make_sched_cursor(axes["sched"])]

    # Playback / UI state
    state = {
        "playing": False, "t": view.schedule.start_time(),
        "drag_was_playing": False,
        "suppress_slider_cb": False,
        "show_ellipsoids": False,
        "completed_override": set(),  # contour ids finished in a prior plan
        "disabled_agents": set(),
    }
    ellipse_patches: dict = {a: [] for a in agents}
    contour_lines: list = []

    # ── render helpers bound to this view / state ─────────────────────────────

    def _clear_ellipses():
        for a in agents:
            for p in ellipse_patches[a]:
                try:
                    p.remove()
                except ValueError:
                    pass
            ellipse_patches[a] = []

    def _contour_color(contour_id):
        if contour_id in state["completed_override"]:
            return "lightgrey"
        assigned = view.contour_agent.get(contour_id)
        return agent_color[assigned] if assigned is not None else "lightgrey"

    def _update_layer(val):
        for line in contour_lines:
            line.pop(0).remove()
        contour_lines.clear()
        z_height = unique_z[int(val) - 1]
        for idx, contour in enumerate(toolpath.contours):
            if contour_z[idx] != z_height:
                continue
            path = np.array(contour.path)
            contour_lines.append(
                axes["anim"].plot(
                    path[:, 0], path[:, 1],
                    path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()],
                    color=_contour_color(contour.id),
                    zorder=0,
                )
            )

    def _update_display(t):
        _clear_ellipses()
        for a, anim_m in zip(agents, view.anim_models):
            anim_m.update(t)
            if state["show_ellipsoids"]:
                _draw_ellipses_for_agent(
                    a, t, view, agent_color, axes["anim"], ellipse_patches,
                )
        sched_cursor_box[0].set_data([t, t], [-3, 3])
        fig.canvas.draw_idle()

    # Layer slider
    layer_slider = Slider(
        ax=axes["layer"], label="Layer",
        valmin=1, valmax=max(len(unique_z), 1), valstep=1,
        orientation="vertical",
    )
    layer_slider.on_changed(_update_layer)
    _update_layer(1)

    # Time slider
    t0, t1 = view.schedule.start_time(), view.schedule.end_time()
    if t1 <= t0:
        t1 = t0 + 1.0
    anim_slider = Slider(
        ax=axes["slider"], label="time",
        valmin=t0, valmax=t1, valstep=step, valinit=t0,
    )

    # Play/pause/ellipsoid core controls
    btn_play = Button(axes["play_btn"], "▶ Play")
    check_ellipsoid = CheckButtons(axes["chk"], ["Show ellipsoids"], [False])
    axes["chk"].set_facecolor(fig.get_facecolor())
    for spine in axes["chk"].spines.values():
        spine.set_visible(False)

    timer = fig.canvas.new_timer(interval=33)

    # ── event handlers ────────────────────────────────────────────────────────
    def _tick():
        if not state["playing"]:
            return
        new_t = min(state["t"] + 0.033 * playback_speed, anim_slider.valmax)
        state["t"] = new_t
        state["suppress_slider_cb"] = True
        anim_slider.set_val(new_t)
        state["suppress_slider_cb"] = False
        _update_display(new_t)
        if new_t >= anim_slider.valmax:
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            timer.stop()

    def _on_play_pause(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text("⏸ Pause" if state["playing"] else "▶ Play")
        if state["playing"]:
            if state["t"] >= anim_slider.valmax:
                state["t"] = anim_slider.valmin
                anim_slider.set_val(state["t"])
            timer.start()
        else:
            timer.stop()

    def _on_slider_changed(val):
        if state["suppress_slider_cb"]:
            return
        state["t"] = float(val)
        _update_display(float(val))

    def _on_press(event):
        if event.inaxes is axes["slider"]:
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

    # ── replan UI (optional) ──────────────────────────────────────────────────
    replan_widgets = {}
    if replan_fn is not None:
        def _apply_replan(t_replan: float):
            """Call replan_fn, swap view in-place, rebuild affected displays."""
            state["playing"] = False
            timer.stop()
            btn_play.label.set_text("▶ Play")

            # Contours assigned pre-replan but not re-planned = completed.
            previous_assigned = set(view.contour_agent.keys())

            new_sched = replan_fn(t_replan, set(state["disabled_agents"]))
            new_assigned = set()
            for s in new_sched.schedules.values():
                for ev in s._events:
                    if isinstance(ev, ContourEvent):
                        new_assigned.add(ev.contour.id)
            newly_completed = previous_assigned - new_assigned

            view.schedule = new_sched
            view.contour_agent = _build_contour_agent_map(new_sched)
            for a, anim_m in zip(agents, view.anim_models):
                anim_m.set_schedule(new_sched.schedules.get(a, ToolpathSchedule()))

            state["completed_override"] |= newly_completed
            # In-progress continuations start at t=0 of the new schedule so
            # they're NOT completed yet; remove them from the override.
            state["completed_override"] -= new_assigned

            sched_cursor_box[0] = _rerender_gantt(
                view.schedule, axes["sched"], agent_color,
            )
            _update_layer(int(layer_slider.val))

            new_t_end = max(view.schedule.end_time(), view.schedule.start_time() + 1e-3)
            _rescale_time_slider(anim_slider, 0.0, new_t_end, step)
            state["t"] = 0.0
            state["suppress_slider_cb"] = True
            anim_slider.set_val(0.0)
            state["suppress_slider_cb"] = False
            _update_display(0.0)

        def _on_replan(event):
            _apply_replan(float(state["t"]))

        def _on_reset(event):
            state["completed_override"] = set()
            _apply_replan(0.0)

        def _on_agent_toggle(label):
            if label in state["disabled_agents"]:
                state["disabled_agents"].remove(label)
            else:
                state["disabled_agents"].add(label)

        replan_widgets = _build_replan_ui(
            fig, axes, agents, _on_agent_toggle, _on_replan, _on_reset,
        )
        replan_widgets["apply_replan"] = _apply_replan

    _update_display(view.schedule.start_time())

    controls = {
        "button": btn_play,
        "check_ellipsoid": check_ellipsoid,
        "layer_slider": layer_slider,
        "anim_slider": anim_slider,
        "timer": timer,
        "state": state,
        "view": view,
        "ellipsoid_cache": view.ellipsoid_cache,
        **replan_widgets,
    }

    if show:
        plt.show()
    return fig, controls


# ─────────────────────── helpers for animate_multi_agent_toolpath_full ────────


class _ScheduleView:
    """Mutable derived state that is recomputed when the schedule changes.

    Holds the current schedule, the contour→agent mapping, the workspace
    animation models, and the ellipsoid cache used for overlay rendering.
    """

    __slots__ = ("schedule", "contour_agent", "anim_models",
                 "ellipsoid_cache", "ellipsoid_safety")

    def __init__(self, schedule, contour_agent, anim_models,
                 ellipsoid_cache, ellipsoid_safety):
        self.schedule = schedule
        self.contour_agent = contour_agent
        self.anim_models = anim_models
        self.ellipsoid_cache = ellipsoid_cache
        self.ellipsoid_safety = ellipsoid_safety


def _create_animate_layout(fig_size, has_replan: bool):
    """Create the figure + axes layout.  Returns a dict of axes by role."""
    if has_replan:
        fig = plt.figure(figsize=(fig_size[0] + 2, fig_size[1] + 1))
        gs = GridSpec(
            4, 2,
            height_ratios=[1, 4, 0.55, 0.25],
            width_ratios=[1, 50],
            hspace=0.35, wspace=0.05,
        )
    else:
        fig = plt.figure(figsize=fig_size)
        gs = GridSpec(
            4, 2,
            height_ratios=[1, 4, 0.45, 0.25],
            width_ratios=[1, 50],
            hspace=0.35, wspace=0.05,
        )

    sched_ax  = fig.add_subplot(gs[0, :])
    anim_ax   = fig.add_subplot(gs[1, 1])
    layer_ax  = fig.add_subplot(gs[1, 0])
    ctrl_ax   = fig.add_subplot(gs[2, :])
    slider_ax = fig.add_subplot(gs[3, :])
    ctrl_ax.set_visible(False)

    ctrl_bb = ctrl_ax.get_position()
    play_btn_ax = fig.add_axes([ctrl_bb.x0,        ctrl_bb.y0, 0.09, ctrl_bb.height])
    chk_ax      = fig.add_axes([ctrl_bb.x0 + 0.10, ctrl_bb.y0, 0.16, ctrl_bb.height])

    return {
        "fig": fig, "sched": sched_ax, "anim": anim_ax, "layer": layer_ax,
        "ctrl": ctrl_ax, "slider": slider_ax,
        "play_btn": play_btn_ax, "chk": chk_ax,
    }


def _build_anim_models(agents, schedule, agent_models, anim_ax, agent_color):
    """Create AnimationModel instances for each agent pinned to anim_ax."""
    anim_models = []
    for a in agents:
        color = agent_color[a]
        sched = schedule.schedules.get(a, ToolpathSchedule())
        if isinstance(agent_models[a].collision_model, FCLRobotBBCollisionModel):
            m = RobotBBAnimationModel(agent_models[a], sched, anim_ax, color=color)
        else:
            m = AnimationModel(agent_models[a], sched, anim_ax, color=color)
        anim_models.append(m)
    return anim_models


def _render_gantt(schedule, sched_ax, agent_colors):
    _plot_multi_agent_schedule(schedule, sched_ax, agent_colors=agent_colors)
    sched_ax.set_xlabel("Time")
    sched_ax.set_title("Multi-agent Schedule")


def _make_sched_cursor(sched_ax):
    """Create the green time-cursor line for the Gantt axis."""
    (cursor,) = sched_ax.plot(
        [], [], lw=2, color=(0, 1, 0.31),
        path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()],
    )
    return cursor


def _rerender_gantt(schedule, sched_ax, agent_colors):
    """Clear and redraw the Gantt axis.  Returns a freshly created cursor
    line that the caller should swap into its cursor reference holder."""
    sched_ax.clear()
    _render_gantt(schedule, sched_ax, agent_colors)
    return _make_sched_cursor(sched_ax)


def _rescale_time_slider(slider, vmin, vmax, step):
    slider.valmin = vmin
    slider.valmax = vmax
    slider.valstep = step
    slider.ax.set_xlim(vmin, vmax)


def _draw_ellipses_for_agent(agent, t, view, agent_color, anim_ax, patches_by_agent):
    """Draw tight + inflated ellipses on ``anim_ax`` for agent's active contour."""
    sched = view.schedule.schedules.get(agent)
    if sched is None:
        return
    active = None
    for ev in sched._events:
        if isinstance(ev, ContourEvent) and ev.start <= t <= ev.end:
            active = ev
            break
    if active is None or active.contour.id not in view.ellipsoid_cache:
        return

    record = view.ellipsoid_cache[active.contour.id]
    color = agent_color[agent]

    tight = _ellipse_patch_from_record(
        record, safety_m=0.0, color=color, alpha=0.25, zorder=2,
        filled=True, linewidth=0,
    )
    anim_ax.add_patch(tight)
    patches_by_agent[agent].append(tight)

    if view.ellipsoid_safety > 0.0:
        inflated = _ellipse_patch_from_record(
            record, safety_m=view.ellipsoid_safety,
            color=color, alpha=0.8, zorder=2,
            filled=False, linewidth=1.5,
        )
        inflated.set_linestyle("--")
        anim_ax.add_patch(inflated)
        patches_by_agent[agent].append(inflated)


def _build_replan_ui(fig, axes, agents, on_toggle, on_replan, on_reset):
    """Add per-agent checkboxes + Replan + Reset buttons to the figure.

    Returns a dict of the widgets so the caller can keep references.
    """
    ctrl_bb = axes["ctrl"].get_position()

    # Checkbox strip placed to the right of the play+ellipsoid controls
    chk_x = ctrl_bb.x0 + 0.30
    chk_width = min(0.32, ctrl_bb.x1 - chk_x - 0.22)
    agent_chk_ax = fig.add_axes([chk_x, ctrl_bb.y0, chk_width, ctrl_bb.height])
    agent_chk_ax.set_facecolor(fig.get_facecolor())
    for spine in agent_chk_ax.spines.values():
        spine.set_visible(False)
    agent_chk = CheckButtons(
        agent_chk_ax, [str(a) for a in agents], [True] * len(agents),
    )

    def _handle(label):
        # Toggle propagates the *disable* semantics (checked = enabled)
        on_toggle(label)
    agent_chk.on_clicked(_handle)

    # Replan + Reset buttons on the far right
    rp_x = ctrl_bb.x1 - 0.20
    replan_btn_ax = fig.add_axes([rp_x,       ctrl_bb.y0, 0.09, ctrl_bb.height])
    reset_btn_ax  = fig.add_axes([rp_x + 0.10, ctrl_bb.y0, 0.09, ctrl_bb.height])
    btn_replan = Button(replan_btn_ax, "Replan")
    btn_reset  = Button(reset_btn_ax, "Reset")
    btn_replan.on_clicked(on_replan)
    btn_reset.on_clicked(on_reset)

    return {
        "agent_checks": agent_chk,
        "btn_replan": btn_replan,
        "btn_reset": btn_reset,
    }


class AnimationModel(object):
    def __init__(self, agent_model: AgentModel, schedule: ToolpathSchedule, ax,
                 color="steelblue"):
        self.model = agent_model
        self.sched = schedule
        self.ax = ax

        (self.line,) = ax.plot([], [], lw=2, color=color)

    def set_schedule(self, schedule: ToolpathSchedule):
        """Swap the backing schedule (used by replan flows)."""
        self.sched = schedule

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
