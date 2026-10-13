"""Reliability timelines: what the engine believed versus what was really wrong."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STATUS_COLORS = {"healthy": "tab:green", "degraded": "tab:orange", "failed": "tab:red"}


def save_health_timeline(
    df: pd.DataFrame,
    path: Path,
    title: str = "",
    degraded_below: float = 0.65,
    failed_below: float = 0.30,
) -> None:
    """One panel per sensor. Dots = reliability coloured by status.

    Shaded background = what was truly wrong: red = clearly faulty (severity >= 0.5),
    yellow = faint fault. The engine never sees this shading.
    """
    sensors = sorted(df["sensor"].unique())
    fig, axes = plt.subplots(
        len(sensors), 1, figsize=(11, 2.6 * len(sensors)), sharex=True
    )
    axes = [axes] if len(sensors) == 1 else list(axes)
    for ax, sensor in zip(axes, sensors, strict=True):
        g = df[df["sensor"] == sensor].sort_values("time")
        t = g["time"].to_numpy()
        sev = g["truth_severity"].to_numpy()
        for mask, color in (
            (sev >= 0.5, "tab:red"),
            ((sev > 0) & (sev < 0.5), "gold"),
        ):
            ax.fill_between(t, 0, 1.05, where=mask, color=color, alpha=0.18, step="mid")
        ax.axhline(degraded_below, color="tab:orange", ls="--", lw=0.8)
        ax.axhline(failed_below, color="tab:red", ls="--", lw=0.8)
        ax.plot(t, g["reliability"], color="0.35", lw=1.0)
        ax.scatter(
            t,
            g["reliability"],
            s=9,
            c=[STATUS_COLORS[s] for s in g["status"]],
            zorder=3,
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{sensor}\nreliability")
        ax.grid(alpha=0.25)
    axes[0].set_title(title or "Sensor reliability over time")
    axes[-1].set_xlabel("time (s)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=name)
        for name, c in STATUS_COLORS.items()
    ]
    axes[0].legend(
        handles=handles, loc="lower left", ncol=3, fontsize=8, title="engine status"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
