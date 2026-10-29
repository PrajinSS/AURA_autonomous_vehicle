"""Unit tests for Phase 6: Factory environment, AMR vehicle physics, and simulation bridge."""

import numpy as np

from simulation.amr_vehicle import AMRParams, AMRState, AMRVehicle
from simulation.carla_bridge.client import CarlaSimulationBridge
from simulation.carla_bridge.fallback_twin import FallbackDigitalTwin
from simulation.factory_world import FactoryWorld


def test_factory_world() -> None:
    factory = FactoryWorld()
    env = factory.build_environment(t=0.0)
    assert len(env.obstacles) >= 5
    assert "Dock A" in factory.DOCKS
    assert "Dock B" in factory.DOCKS

    # Test dynamic worker motion
    w = factory.workers[0]
    p0 = w.position_at(0.0)
    p1 = w.position_at(5.0)
    assert p0 != p1


def test_amr_vehicle_kinematics_and_battery() -> None:
    vehicle = AMRVehicle(AMRState(x=0.0, y=0.0, heading=0.0))
    initial_soc = vehicle.state.battery_soc_pct

    # Accelerate forward
    for _ in range(20):
        vehicle.step(accel_cmd=1.0, steer_cmd=0.0, dt=0.05)

    assert vehicle.state.speed > 0.5
    assert vehicle.state.x > 0.0
    assert vehicle.state.battery_soc_pct <= initial_soc
    assert vehicle.state.runtime_remaining_hours > 0.0
    assert vehicle.state.status == "OPERATIONAL"


def test_carla_simulation_bridge() -> None:
    bridge = CarlaSimulationBridge()
    # Runs smoothly whether CARLA server is online or offline
    state, frame = bridge.step(accel_cmd=0.5, steer_cmd=0.1)

    assert state.speed >= 0.0
    assert frame.ego_pose is not None
    assert "camera" in frame.sensor_confidence
    assert "lidar" in frame.sensor_confidence
    assert "radar" in frame.sensor_confidence

    # Test sensor degradation injection
    bridge.set_sensor_fault("camera", True)
    _, frame_degraded = bridge.step()
    assert frame_degraded.sensor_confidence["camera"] < 0.5
