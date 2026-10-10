"""Matplotlib helpers: static trajectory plot and GIF animation (headless-safe)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; works on servers and in CI

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from simulation.environment import Environment  # noqa: E402
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


def save_trajectory_plot(
    result: SimulationResult, env: Environment, path: Path
) -> None:
    """Save the driven path, colored by speed, with obstacle start/end positions."""
    fig, ax = _prepare_axes(env, f"{result.scenario}: {result.reason.value}")
    xs = [r.x for r in result.records]
    ys = [r.y for r in result.records]
    speeds = [r.speed for r in result.records]
    sc = ax.scatter(xs, ys, c=speeds, s=6, cmap="viridis", zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="speed (m/s)")

    first, last = result.obstacle_history[0], result.obstacle_history[-1]
    for (x0, y0, r), (x1, y1, _) in zip(first, last, strict=True):
        ax.add_patch(Circle((x0, y0), r, fill=False, ls="--", ec="tab:red"))
        ax.add_patch(Circle((x1, y1), r, color="tab:red", alpha=0.6))
        if (x0, y0) != (x1, y1):
            ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "->"})
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

    vehicle_radius = 1.2
    vehicle_patch = Circle((0, 0), vehicle_radius, color="tab:blue", zorder=4)
    ax.add_patch(vehicle_patch)
    (heading_line,) = ax.plot([], [], color="white", lw=2, zorder=5)
    (trail,) = ax.plot([], [], color="tab:blue", lw=1, alpha=0.6, zorder=2)
    obstacle_patches = [
        ax.add_patch(Circle((x, y), r, color="tab:red", alpha=0.7))
        for (x, y, r) in result.obstacle_history[0]
    ]

    def update(i: int):
        rec = result.records[i]
        vehicle_patch.set_center((rec.x, rec.y))
        heading_line.set_data(
            [rec.x, rec.x + vehicle_radius * math.cos(rec.heading)],
            [rec.y, rec.y + vehicle_radius * math.sin(rec.heading)],
        )
        trail.set_data(
            [r.x for r in result.records[: i + 1]],
            [r.y for r in result.records[: i + 1]],
        )
        for patch, (x, y, _) in zip(
            obstacle_patches, result.obstacle_history[i], strict=True
        ):
            patch.set_center((x, y))
        return [vehicle_patch, heading_line, trail, *obstacle_patches]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
