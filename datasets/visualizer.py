"""Multimodal dataset visualizer for AURA.

Plots camera frames/detections, LiDAR Bird's Eye View point clouds,
radar velocity vectors, ground truth bounding boxes, and ego state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from datasets.kitti_loader import KittiLoader
from datasets.nuscenes_loader import NuScenesMiniLoader
from datasets.schema import UnifiedFrame


def plot_unified_frame(
    frame: UnifiedFrame,
    save_path: str | Path | None = None,
    show: bool = False,
    title_suffix: str = "",
) -> plt.Figure:
    """Render a 4-panel diagnostic multimodal view of a UnifiedFrame."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#0b1320")

    text_color = "#e2e8f0"
    accent_blue = "#38bdf8"
    accent_green = "#22c55e"
    accent_amber = "#f59e0b"
    accent_red = "#ef4444"

    for ax in axes.flat:
        ax.set_facecolor("#111a2e")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")

    # Panel 1: Camera detections (Ego Front Projection)
    ax_cam = axes[0, 0]
    ax_cam.set_title(
        f"Camera Detections (Conf: {frame.sensor_confidence.get('camera', 1.0):.0%})",
        color=text_color,
        fontsize=12,
        pad=10,
    )
    ax_cam.set_xlim(-15, 15)
    ax_cam.set_ylim(0, 40)
    ax_cam.set_xlabel("Lateral Offset (m)", color=text_color)
    ax_cam.set_ylabel("Longitudinal Distance (m)", color=text_color)

    # Plot vehicle footprint
    ax_cam.plot([0], [0], "^", color=accent_blue, markersize=12, label="Ego Vehicle")
    for det in frame.camera_detections:
        if det.center_ego is not None:
            dx, dy = det.center_ego
            ax_cam.scatter(dy, dx, color=accent_blue, s=120, edgecolors="white", zorder=5)
            ax_cam.text(
                dy + 0.5,
                dx,
                f"{det.category}\n{det.confidence:.0%}",
                color=accent_blue,
                fontsize=9,
            )
    ax_cam.legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor=text_color)

    # Panel 2: LiDAR BEV Point Cloud & Clusters
    ax_lidar = axes[0, 1]
    ax_lidar.set_title(
        f"LiDAR BEV Point Cloud (Conf: {frame.sensor_confidence.get('lidar', 1.0):.0%})",
        color=text_color,
        fontsize=12,
        pad=10,
    )
    ax_lidar.set_xlim(-20, 20)
    ax_lidar.set_ylim(-5, 45)
    ax_lidar.set_xlabel("Lateral Y (m)", color=text_color)
    ax_lidar.set_ylabel("Forward X (m)", color=text_color)

    if len(frame.lidar_points) > 0:
        pts = frame.lidar_points
        # In ego frame: pts[:, 0] forward, pts[:, 1] lateral
        ax_lidar.scatter(pts[:, 1], pts[:, 0], s=3, c="#64748b", alpha=0.6, label="Point Cloud")

    for obs in frame.lidar_observations:
        ox, oy, _ = obs.center
        l, w, _ = obs.size
        rect = Rectangle(
            (oy - w / 2, ox - l / 2),
            w,
            l,
            linewidth=2,
            edgecolor=accent_green,
            facecolor="none",
            label="LiDAR Box" if "LiDAR Box" not in [l.get_label() for l in ax_lidar.lines] else "",
        )
        ax_lidar.add_patch(rect)
        ax_lidar.text(oy + 0.5, ox, f"{obs.track_id or 'Cluster'}", color=accent_green, fontsize=9)

    # Panel 3: Radar Detections & Doppler Velocity
    ax_rad = axes[1, 0]
    ax_rad.set_title(
        f"Radar Doppler Vectors (Conf: {frame.sensor_confidence.get('radar', 1.0):.0%})",
        color=text_color,
        fontsize=12,
        pad=10,
    )
    ax_rad.set_xlim(-20, 20)
    ax_rad.set_ylim(-5, 45)
    ax_rad.set_xlabel("Lateral Y (m)", color=text_color)
    ax_rad.set_ylabel("Forward X (m)", color=text_color)

    for rad in frame.radar_detections:
        ax_rad.scatter(rad.y_ego, rad.x_ego, color=accent_amber, s=90, marker="D", label="Radar Target")
        # Draw radial velocity vector
        v_arrow = rad.radial_velocity_mps * 1.5
        ax_rad.arrow(
            rad.y_ego,
            rad.x_ego,
            0.0,
            v_arrow,
            color=accent_amber,
            head_width=0.8,
            head_length=0.8,
            length_includes_head=True,
        )
        ax_rad.text(
            rad.y_ego + 0.6,
            rad.x_ego,
            f"{rad.radial_velocity_mps:+.1f} m/s",
            color=accent_amber,
            fontsize=9,
        )

    # Panel 4: Unified Multimodal Sensor Fusion & Ground Truth
    ax_fuse = axes[1, 1]
    ax_fuse.set_title(
        "Unified Multimodal Perception vs Ground Truth",
        color=text_color,
        fontsize=12,
        pad=10,
    )
    ax_fuse.set_xlim(-20, 20)
    ax_fuse.set_ylim(-5, 45)
    ax_fuse.set_xlabel("Lateral Y (m)", color=text_color)
    ax_fuse.set_ylabel("Forward X (m)", color=text_color)

    # Ego vehicle footprint in center
    ego_box = Rectangle((-0.7, -1.25), 1.4, 2.5, color=accent_blue, alpha=0.8, label="Ego AMR")
    ax_fuse.add_patch(ego_box)

    # Ground truth boxes
    for gt in frame.ground_truth_objects:
        dx = gt.x - frame.ego_pose.x
        dy = gt.y - frame.ego_pose.y
        gt_rect = Rectangle(
            (dy - gt.width / 2, dx - gt.length / 2),
            gt.width,
            gt.length,
            linewidth=2,
            linestyle="--",
            edgecolor=accent_red,
            facecolor="none",
            label="Ground Truth" if "Ground Truth" not in [l.get_label() for l in ax_fuse.lines] else "",
        )
        ax_fuse.add_patch(gt_rect)
        ax_fuse.text(dy - 1.0, dx + gt.length / 2 + 0.4, f"GT: {gt.category}", color=accent_red, fontsize=9)

    fig.suptitle(
        f"AURA Unified Perception Frame: {frame.frame_id} (t={frame.timestamp:.2f}s) {title_suffix}",
        color="white",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)

    if show:
        plt.show()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA Multimodal Dataset Inspector")
    parser.add_argument(
        "--dataset",
        choices=["nuscenes", "kitti"],
        default="nuscenes",
        help="Dataset to inspect",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_preview.png"),
        help="Path to save rendered image",
    )
    args = parser.parse_args()

    if args.dataset == "nuscenes":
        loader = NuScenesMiniLoader()
        frame = loader.get_frame(min(args.frame, len(loader) - 1))
    else:
        loader = KittiLoader()
        frame = loader.get_frame(min(args.frame, len(loader) - 1))

    plot_unified_frame(frame, save_path=args.output)
    print(f"Visualized {args.dataset} frame {frame.frame_id} -> {args.output}")


if __name__ == "__main__":
    main()
