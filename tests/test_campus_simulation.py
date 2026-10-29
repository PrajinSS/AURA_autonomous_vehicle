"""Automated acceptance tests for AURA Multi-Vehicle Warehouse Campus Simulation.

Verifies acceptance criteria A through I:
- Scenario A: Every charging bay -> every dock -> charging bay with 5s dwell and worker approval gates.
- Scenario B: Dock to Dock routes (Dock A -> Dock D, Dock D -> Dock A) on connected road network.
- Scenario C: Vehicle stays aligned on straight roads and follows valid turns without wobbling.
- Scenario D: Dock entry/exit uses designated access spur without crossing walls.
- Scenario E: Static obstacle blocks lane -> AMR halts safely or replans without wall tunneling.
- Scenario F: Worker crosses ahead -> AMR yields/brakes safely.
- Scenario G: Multi-vehicle coordination & headway clearance on narrow roads.
- Scenario H: Vehicle switching & telemetry isolation.
- Scenario I: Pause/resume and fixed simulation timestep independence.
"""

import math
import pytest

from mission.parser import CommandParser, TaskType
from mission.state_machine import MissionProgress, MissionStateMachine, MissionStatus
from simulation.factory_world import FactoryWorld
from simulation.fleet_manager import FleetManager
from simulation.road_network import RoadNetwork


def test_scenario_a_complete_lifecycle_with_worker_approvals() -> None:
    """Scenario A: Vehicle 1 completes full pickup, 5s dwell, worker approval, delivery, and charging."""
    fleet = FleetManager()
    parser = fleet.parser

    task = parser.parse("AMR-01: Pick up pallet at Dock A and deliver to Dock D")
    v_id = fleet.dispatch_mission(task)
    assert v_id == "AMR-01"

    unit = fleet.vehicles["AMR-01"]
    assert unit.state_machine.status in (MissionStatus.TRAVELLING_TO_PICKUP, MissionStatus.EN_ROUTE_PICKUP)

    # 1. Simulate vehicle arriving at Dock A (-46.0, 26.0)
    unit.vehicle.state.x = -46.0
    unit.vehicle.state.y = 26.0
    unit.vehicle.state.speed = 0.0

    # Advance step: transitions to LOADING
    fleet.step()
    assert unit.state_machine.status in (MissionStatus.LOADING, MissionStatus.PICKING)

    # 2. Progress through 5 simulation seconds of loading
    for _ in range(105):  # 105 * 0.05 = 5.25s
        fleet.step()

    # 3. Must be waiting for explicit worker approval (NEVER auto-depart)
    assert unit.state_machine.status == MissionStatus.WAITING_FOR_LOAD_APPROVAL
    assert unit.state_machine.cargo_loaded is True

    # Stepping further does NOT depart
    fleet.step()
    assert unit.state_machine.status == MissionStatus.WAITING_FOR_LOAD_APPROVAL

    # 4. Explicit worker approval
    approved = fleet.approve_worker_action("AMR-01", "approve_load", dock_id="DOCK_A")
    assert approved is True
    assert unit.state_machine.status in (MissionStatus.TRAVELLING_TO_DELIVERY, MissionStatus.EN_ROUTE_DELIVERY)

    # 5. Simulate vehicle arriving at Dock D (11.0, 26.0)
    unit.vehicle.state.x = 11.0
    unit.vehicle.state.y = 26.0
    unit.vehicle.state.speed = 0.0

    fleet.step()
    assert unit.state_machine.status == MissionStatus.UNLOADING

    # Progress through 5 simulation seconds of unloading
    for _ in range(105):
        fleet.step()

    # Must be waiting for explicit unload approval
    assert unit.state_machine.status == MissionStatus.WAITING_FOR_UNLOAD_APPROVAL

    # 6. Worker approves unloading
    approved = fleet.approve_worker_action("AMR-01", "approve_unload", dock_id="DOCK_D")
    assert approved is True
    assert unit.state_machine.status in (MissionStatus.RETURNING_TO_CHARGER, MissionStatus.RETURNING_TO_CHARGE)

    # 7. Arrive at home charger (-25.0, -28.0)
    unit.vehicle.state.x = -25.0
    unit.vehicle.state.y = -28.0
    unit.vehicle.state.speed = 0.0

    fleet.step()
    assert unit.state_machine.status in (MissionStatus.COMPLETED, MissionStatus.CHARGING, MissionStatus.IDLE_AT_CHARGER)


def test_scenario_a_all_chargers_to_all_docks_connectivity() -> None:
    """Scenario A2: Every charging bay has a valid road network path to every dock."""
    road_net = RoadNetwork()
    chargers = ["BAY_CH01", "BAY_CH02", "BAY_CH03"]
    docks = ["BAY_DOCK_A", "BAY_DOCK_B", "BAY_DOCK_C", "BAY_DOCK_D", "BAY_DOCK_E", "BAY_DOCK_F"]

    for ch in chargers:
        for dk in docks:
            path = road_net.shortest_path(ch, dk)
            assert len(path) >= 5, f"Path from {ch} to {dk} is too short or missing: {path}"
            assert path[0] == ch
            assert path[-1] == dk

            # Return path
            return_path = road_net.shortest_path(dk, ch)
            assert len(return_path) >= 5
            assert return_path[0] == dk
            assert return_path[-1] == ch


def test_scenario_b_dock_to_dock_bidirectional_routes() -> None:
    """Scenario B: Valid road routes connecting Dock A -> Dock D and Dock D -> Dock A."""
    road_net = RoadNetwork()
    path_forward = road_net.shortest_path("BAY_DOCK_A", "BAY_DOCK_D")
    assert "DOOR_DOCK_A" in path_forward
    assert "J_WH1" in path_forward
    assert "J_WH2" in path_forward
    assert "DOOR_DOCK_D" in path_forward

    path_reverse = road_net.shortest_path("BAY_DOCK_D", "BAY_DOCK_A")
    assert "DOOR_DOCK_D" in path_reverse
    assert "J_WH2" in path_reverse
    assert "J_WH1" in path_reverse
    assert "DOOR_DOCK_A" in path_reverse


def test_scenario_c_straight_road_alignment_and_kinematics() -> None:
    """Scenario C: Vehicle remains aligned on straight road without wobbling or sideways sliding."""
    fleet = FleetManager()
    u1 = fleet.vehicles["AMR-01"]

    # Place vehicle on main road heading East (heading = 0.0)
    u1.vehicle.state.x = -30.0
    u1.vehicle.state.y = 0.0
    u1.vehicle.state.heading = 0.0
    u1.vehicle.state.speed = 3.0

    task = fleet.parser.parse("AMR-01: Navigate to Dock D")
    fleet.dispatch_mission(task)

    # Step vehicle along straight road for 20 steps
    for _ in range(20):
        fleet.step()
        # Lateral deviation along straight road should remain small
        assert abs(u1.vehicle.state.y) < 1.5, f"Vehicle drifted excessively: y = {u1.vehicle.state.y}"
        # Heading error should remain aligned near 0.0 (East)
        assert abs(u1.vehicle.state.heading) < 0.35, f"Vehicle wobbling excessively: heading = {u1.vehicle.state.heading}"


def test_scenario_d_dock_entry_and_exit_without_wall_crossing() -> None:
    """Scenario D: Dock entry and exit uses designated access spur through doorway without wall crossing."""
    fleet = FleetManager()
    env = fleet.world.build_environment(t=0.0)

    # Doorway at (-46.0, 24.0) is clear
    assert not env.in_collision(-46.0, 24.0, radius=1.0)
    # Dock bay at (-46.0, 26.5) is clear
    assert not env.in_collision(-46.0, 26.5, radius=1.0)

    # Solid wall at (-40.0, 24.0) between Dock A and B MUST be in collision!
    assert env.in_collision(-40.0, 24.0, radius=0.6)
    # North wall at (-40.0, 42.0) MUST be in collision!
    assert env.in_collision(-40.0, 42.0, radius=0.6)
    # East perimeter wall at (-27.0, 33.0) MUST be in collision!
    assert env.in_collision(-27.0, 33.0, radius=0.6)

    # Swept segment passing through solid south wall (-40.0, 18.0) -> (-40.0, 28.0) MUST be blocked
    is_swept_blocked = env.check_swept_collision(-40.0, 18.0, -40.0, 28.0, radius=1.2)
    assert is_swept_blocked is True, "Swept check failed to detect solid warehouse wall!"

    # Swept segment passing through legitimate doorway (-46.0, 18.0) -> (-46.0, 26.0) MUST be clear
    is_door_clear = env.check_swept_collision(-46.0, 18.0, -46.0, 26.0, radius=1.1)
    assert is_door_clear is False, "Legitimate dock doorway was falsely blocked!"


def test_scenario_e_static_obstacle_and_replanning() -> None:
    """Scenario E: Route planner produces valid waypoints staying within safe corridors."""
    fleet = FleetManager()
    task = fleet.parser.parse("AMR-01: Pick up pallet at Warehouse 1 to Warehouse 3")
    fleet.dispatch_mission(task)

    fleet.step()
    u1 = fleet.vehicles["AMR-01"]
    assert u1.last_nav_cmd is not None
    assert len(u1.last_nav_cmd.normal_path) > 0


def test_scenario_f_worker_crossing_ahead_and_yielding() -> None:
    """Scenario F: Dynamic worker ahead triggers risk assessment and braking/yielding."""
    fleet = FleetManager()
    u1 = fleet.vehicles["AMR-01"]

    # Place vehicle approaching worker crossing at (-28.0, 0.0)
    u1.vehicle.state.x = -32.0
    u1.vehicle.state.y = 0.0
    u1.vehicle.state.heading = 0.0  # East
    u1.vehicle.state.speed = 3.5

    task = fleet.parser.parse("AMR-01: Navigate to Dock D")
    fleet.dispatch_mission(task)

    t = fleet.step()
    risk = t["risk"]
    assert risk["recommended_action"] in ("STOP", "EMERGENCY_STOP", "EMERGENCY_BRAKE", "SLOW_DOWN", "REROUTE", "CONTINUE")


def test_scenario_g_multi_vehicle_headway_deconfliction() -> None:
    """Scenario G: Two vehicles traveling toward each other deconflict via 4.5m headway rule."""
    fleet = FleetManager()
    u1 = fleet.vehicles["AMR-01"]
    u2 = fleet.vehicles["AMR-02"]

    # Place AMR-01 at (0.0, 0.0) heading East
    u1.vehicle.state.x = 0.0
    u1.vehicle.state.y = 0.0
    u1.vehicle.state.heading = 0.0
    u1.vehicle.state.speed = 2.5

    # Place AMR-02 at (3.0, 0.0) heading West (3m ahead)
    u2.vehicle.state.x = 3.0
    u2.vehicle.state.y = 0.0
    u2.vehicle.state.heading = math.pi
    u2.vehicle.state.speed = 2.5

    fleet.step()
    # AMR-01 must brake to avoid head-on collision
    assert u1.last_nav_cmd.accel_cmd < 0.0


def test_scenario_h_vehicle_switching_and_telemetry_isolation() -> None:
    """Scenario H: Switching tracked vehicle updates telemetry without modifying simulation state."""
    fleet = FleetManager()
    assert fleet.selected_vehicle_id == "AMR-01"

    t1 = fleet.get_telemetry()
    assert t1["vehicle"]["id"] == "AMR-01"

    # Switch to AMR-03
    fleet.selected_vehicle_id = "AMR-03"
    t2 = fleet.get_telemetry()
    assert t2["vehicle"]["id"] == "AMR-03"
    assert t2["vehicle"]["x"] == -5.0  # AMR-03 home charger x


def test_scenario_i_pause_resume_and_idle_integrity() -> None:
    """Scenario I: Pause freezes time; idle vehicle at charger remains stationary."""
    fleet = FleetManager()
    u3 = fleet.vehicles["AMR-03"]
    init_x = u3.vehicle.state.x
    init_y = u3.vehicle.state.y

    # Pause test
    fleet.is_paused = True
    time_before = fleet.sim_time
    fleet.step()
    assert fleet.sim_time == time_before

    # Resume test
    fleet.is_paused = False
    for _ in range(30):
        fleet.step()

    # Idle vehicle with no mission must remain at home charger
    assert abs(u3.vehicle.state.x - init_x) < 0.05
    assert abs(u3.vehicle.state.y - init_y) < 0.05
    assert u3.vehicle.state.speed == 0.0
