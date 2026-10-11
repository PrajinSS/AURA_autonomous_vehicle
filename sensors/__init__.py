"""Virtual camera, LiDAR and radar with configurable fault injection."""

from sensors.camera import CameraSensor
from sensors.detection import Detection, SensorFrame
from sensors.faults import FaultEffects, FaultSchedule, FaultSpec, FaultType
from sensors.lidar import LidarSensor
from sensors.radar import RadarSensor
from sensors.suite import SensorSuite

__all__ = [
    "CameraSensor",
    "Detection",
    "FaultEffects",
    "FaultSchedule",
    "FaultSpec",
    "FaultType",
    "LidarSensor",
    "RadarSensor",
    "SensorFrame",
    "SensorSuite",
]
