"""One-page picture of what the sensors saw during a run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from sensors.metrics import detections_dataframe, frames_dataframe  # noqa: E402
from simulation.environment import Environment  # noqa: E402
from simulation.simulator import SimulationResult  # noqa: E402

COLORS = {"camera": "tab:blue", "lidar": "tab:green", "radar": "tab:orange"}


def save_sensor_overview(
    result: SimulationResult, env: Environment, path: Path, title: str = ""
) -> None:
    """Left: where detections landed. Right: error over time, detections per frame."""
    det = detections_dataframe(result.sensor_frames)
    frames = frames_dataframe(result.sensor_frames)

    fig = plt.figure(figsize=(14, 6.5))
    grid = GridSpec(2, 2, width_ratios=[1.35, 1], figure=fig)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_err = fig.add_subplot(grid[0, 1])
    ax_cnt = fig.add_subplot(grid[1, 1])

    b = env.bounds
    ax_map.set_xlim(b.x_min, b.x_max)
    ax_map.set_ylim(b.y_min, b.y_max)
    ax_map.set_aspect("equal")
    ax_map.set_title(title or f"{result.scenario}: sensor detections")
    for frame in result.obstacle_history[0]:
        if frame.kind == "circle":
            ax_map.add_patch(Circle((frame.x, frame.y), frame.width / 2, color="0.8"))
        else:
            ax_map.add_patch(
                Rectangle(
                    (frame.x - frame.width / 2, frame.y - frame.height / 2),
                    frame.width,
                    frame.height,
                    color="0.8",
                )
            )
    ax_map.plot(
        [r.x for r in result.records], [r.y for r in result.records], "k-", lw=1
    )

    for name, color in COLORS.items():
        sub = det[det["sensor"] == name] if not det.empty else det
        if sub.empty:
            continue
        good = sub[sub["truth_fault"] == ""]
        bad = sub[sub["truth_fault"] != ""]
        ax_map.scatter(good["x"], good["y"], s=4, color=color, alpha=0.5, label=name)
        ax_map.scatter(bad["x"], bad["y"], s=14, color=color, marker="x", alpha=0.9)
        matched = sub.dropna(subset=["error_m"])
        ax_err.scatter(
            matched["timestamp"],
            matched["error_m"].clip(lower=1e-3),
            s=3,
            color=color,
            alpha=0.4,
            label=name,
        )

    ax_map.legend(loc="upper left", markerscale=3, title="x = outlier / ghost")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("position error (m)")
    ax_err.set_title("Measurement error vs time")
    ax_err.grid(alpha=0.3)

    for name, color in COLORS.items():
        sub = frames[frames["sensor"] == name]
        if sub.empty:
            continue
        smooth = sub["n_detections"].rolling(10, min_periods=1).mean()
        ax_cnt.plot(sub["timestamp"], smooth, color=color, label=name)

    ax_cnt.set_xlabel("time (s)")
    ax_cnt.set_ylabel("detections / frame")
    ax_cnt.set_title("Detections per frame (10-frame average)")
    ax_cnt.grid(alpha=0.3)
    ax_cnt.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
