"""Matplotlib helpers: trajectory plot, GIF animation, occupancy-grid plot."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works on servers and in CI
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from config import VehicleParams  # noqa: E402
from simulation.environment import Environment  # noqa: E402
from simulation.obstacles import ObstacleFrame  # noqa: E402
from simulation.occupancy_grid import OccupancyGrid  # noqa: E402
from simulation.simulator import SimulationResult  # noqa: E402


def _prepare_axes(env: Environment, title: str):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b = env.bounds
    ax.set_xlim(b.x_min, b.x_max)
    ax.set_ylim(b.y_min, b.y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.add_patch(
        Circle(env.goal, env.goal_tolerance, color="tab:green", alpha=0.35, zorder=1)
    )
    ax.annotate("GOAL", env.goal, ha="center", va="bottom", color="darkgreen")
    return fig, ax


def _make_patch(frame: ObstacleFrame, **style):
    if frame.kind == "circle":
        return Circle((frame.x, frame.y), frame.width / 2.0, **style)
    return Rectangle(
        (frame.x - frame.width / 2.0, frame.y - frame.height / 2.0),
        frame.width,
        frame.height,
        **style,
    )


def _move_patch(patch, frame: ObstacleFrame) -> None:
    if frame.kind == "circle":
        patch.set_center((frame.x, frame.y))
    else:
        patch.set_xy((frame.x - frame.width / 2.0, frame.y - frame.height / 2.0))


def save_trajectory_plot(
    result: SimulationResult, env: Environment, path: Path
) -> None:
    """Driven path coloured by speed; obstacles at start (dashed) and end (solid)."""
    fig, ax = _prepare_axes(env, f"{result.scenario}: {result.reason.value}")
    xs = [r.x for r in result.records]
    ys = [r.y for r in result.records]
    sc = ax.scatter(xs, ys, c=[r.speed for r in result.records], s=6, zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="speed (m/s)")
    first, last = result.obstacle_history[0], result.obstacle_history[-1]
    for i, (f0, f1) in enumerate(zip(first, last, strict=True)):
        ax.add_patch(_make_patch(f0, fill=False, ls="--", ec="tab:red"))
        ax.add_patch(_make_patch(f1, color="tab:red", alpha=0.6))
        if (f0.x, f0.y) != (f1.x, f1.y):
            trail = [frames[i] for frames in result.obstacle_history]
            ax.plot([t.x for t in trail], [t.y for t in trail], ":", color="tab:red")
    ax.plot(xs[0], ys[0], "ks", label="start")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_animation(
    result: SimulationResult,
    env: Environment,
    path: Path,
    stride: int = 4,
    fps: int = 20,
) -> None:
    """Save a GIF of the whole run (every ``stride``-th step)."""
    fig, ax = _prepare_axes(env, f"{result.scenario}: {result.reason.value}")
    frames = list(range(0, len(result.records), stride))
    if frames[-1] != len(result.records) - 1:
        frames.append(len(result.records) - 1)
    radius = VehicleParams().radius
    vehicle_patch = Circle((0, 0), radius, color="tab:blue", zorder=4)
    ax.add_patch(vehicle_patch)
    (heading_line,) = ax.plot([], [], color="white", lw=2, zorder=5)
    (trail,) = ax.plot([], [], color="tab:blue", lw=1, alpha=0.6, zorder=2)
    patches = [
        ax.add_patch(_make_patch(f, color="tab:red", alpha=0.7))
        for f in result.obstacle_history[0]
    ]

    def update(i: int):
        rec = result.records[i]
        vehicle_patch.set_center((rec.x, rec.y))
        heading_line.set_data(
            [rec.x, rec.x + radius * math.cos(rec.heading)],
            [rec.y, rec.y + radius * math.sin(rec.heading)],
        )
        trail.set_data(
            [r.x for r in result.records[: i + 1]],
            [r.y for r in result.records[: i + 1]],
        )
        for patch, frame in zip(patches, result.obstacle_history[i], strict=True):
            _move_patch(patch, frame)
        return [vehicle_patch, heading_line, trail, *patches]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_occupancy_grid_plot(grid: OccupancyGrid, env: Environment, path: Path) -> None:
    """Show the (inflated) static occupancy grid used for route planning."""
    fig, ax = _prepare_axes(
        env, f"occupancy grid ({grid.occupied_fraction:.1%} blocked)"
    )
    b = env.bounds
    ax.imshow(
        grid.data,
        origin="lower",
        extent=(b.x_min, b.x_max, b.y_min, b.y_max),
        cmap="Greys",
        alpha=0.8,
        zorder=2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
