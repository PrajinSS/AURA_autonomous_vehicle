"""The vehicle's sensor suite: runs all sensors and delivers frames on time."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sensors.base import BaseSensor
from sensors.camera import CameraSensor
from sensors.detection import SensorFrame
from sensors.faults import FaultSchedule
from sensors.lidar import LidarSensor
from sensors.radar import RadarSensor
from simulation.world_state import WorldState


class SensorSuite:
    """Sensors measure at their own rates; delayed frames arrive later."""

    def __init__(self, sensors: Sequence[BaseSensor]) -> None:
        names = [s.name for s in sensors]
        if len(names) != len(set(names)):
            raise ValueError("Sensor names must be unique.")
        self.sensors = list(sensors)
        self._pending: list[SensorFrame] = []

    @classmethod
    def default(
        cls, faults: FaultSchedule | None = None, seed: int = 42
    ) -> SensorSuite:
        """Camera + LiDAR + radar. Each sensor gets its own random stream, so
        adding a fault to one sensor never changes the noise of the others."""
        classes = (CameraSensor, LidarSensor, RadarSensor)
        streams = np.random.SeedSequence(seed).spawn(len(classes))
        return cls(
            [
                c(faults, np.random.default_rng(s))
                for c, s in zip(classes, streams, strict=True)
            ]
        )

    def step(self, world: WorldState) -> list[SensorFrame]:
        """Measure, then return every frame that has arrived by ``world.time``."""
        for sensor in self.sensors:
            frame = sensor.sample(world)
            if frame is not None:
                self._pending.append(frame)
        ready = [f for f in self._pending if f.arrival_time <= world.time + 1e-9]
        self._pending = [f for f in self._pending if f.arrival_time > world.time + 1e-9]
        return sorted(ready, key=lambda f: f.arrival_time)
