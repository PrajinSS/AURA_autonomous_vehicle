"""Virtual camera: good angle and class information, weak depth, hurt by
darkness, dust and glare (model these with noise / dropout / range faults)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sensors.base import BaseSensor, SensorConfig
from sensors.faults import FaultEffects
from simulation.world_state import ObstacleTruth, WorldState

CAMERA_CONFIG = SensorConfig(
    name="camera",
    rate_hz=10.0,
    max_range=40.0,
    fov_deg=90.0,
    range_sigma0=0.3,
    range_sigma_per_m=0.02,  # monocular depth gets worse with distance
    bearing_sigma_deg=0.5,
    base_dropout=0.03,
    far_dropout=0.3,
    occludable=True,
)

# How the world's obstacle categories appear to the classifier.
LABEL_MAP = {
    "static": "static",
    "worker": "person",
    "vehicle": "vehicle",
    "dynamic": "moving_object",
}
ALL_LABELS = ("static", "person", "vehicle", "moving_object")


class CameraSensor(BaseSensor):
    DEFAULT_CONFIG = CAMERA_CONFIG
    BASE_MISCLASS = 0.03

    def _extra_fields(
        self,
        target: ObstacleTruth,
        true_range: float,
        fx: FaultEffects,
        world: WorldState,
    ) -> dict[str, Any]:
        true_label = LABEL_MAP.get(target.category, "moving_object")
        p_wrong = min(0.95, self.BASE_MISCLASS + 0.4 * fx.severity)
        label = true_label
        if self.rng.random() < p_wrong:
            label = str(
                self.rng.choice([name for name in ALL_LABELS if name != true_label])
            )
        confidence = (
            0.97 - 0.6 * fx.severity - 0.002 * true_range + self.rng.normal(0, 0.02)
        )
        return {
            "label": label,
            "label_confidence": float(np.clip(confidence, 0.05, 0.99)),
        }

    def _ghost_fields(self) -> dict[str, Any]:
        return {
            "label": str(self.rng.choice(ALL_LABELS)),
            "label_confidence": float(self.rng.uniform(0.3, 0.7)),
        }
