"""Unit tests for Phase 9: Global A* planner, local spline tracker, and adaptive controller."""

import numpy as np

from perception.tracker import TrackedObstacle
from planning.adaptive_controller import AdaptiveAction, AdaptivePlannerController
from planning.global_planner import AStarGlobalPlanner
from planning.local_planner import LocalSplinePlanner
from risk.risk_assessor import AssessmentReport, RiskLevel
from simulation.amr_vehicle import AMRState
from simulation.factory_world import FactoryWorld


def test_global_planner() -> None:
    factory = FactoryWorld()
    env = factory.build_environment(t=0.0)
    planner = AStarGlobalPlanner()

    start = (-15.0, 0.0)
    goal = (-7.0, 18.0)  # Dock B

    # Nominal route
    path = planner.plan_path(start, goal, env)
    assert len(path) >= 2
    assert path[0].name.startswith("WP-")
    assert math_dist(path[-1].x, path[-1].y, goal[0], goal[1]) < 3.0

    # Alternative route avoiding obstacle along nominal path
    avoid = [(-11.0, 9.0, 2.5)]
    alt_path = planner.plan_path(start, goal, env, avoid_regions=avoid)
    assert len(alt_path) >= 2
    assert alt_path[0].name.startswith("ALT-")


def test_local_planner_control() -> None:
    planner = LocalSplinePlanner()
    global_p = AStarGlobalPlanner()
    factory = FactoryWorld()
    env = factory.build_environment(t=0.0)

    start = (0.0, 0.0)
    goal = (10.0, 0.0)
    wps = global_p.plan_path(start, goal, env)

    accel, steer = planner.compute_control(
        current_pose=(0.0, 0.0, 0.0),
        current_speed=1.0,
        waypoints=wps,
        tracked_obstacles=[],
    )

    assert -3.5 <= accel <= 2.0
    assert -0.8 <= steer <= 0.8


def test_adaptive_controller_slow_down_and_reroute() -> None:
    controller = AdaptivePlannerController()
    factory = FactoryWorld()
    env = factory.build_environment(t=0.0)
    state = AMRState(x=-15.0, y=0.0, heading=0.0, speed=1.5, battery_soc_pct=72.0)

    # Fake a high-risk report
    report = AssessmentReport(
        composite_risk_score=80.0,
        overall_risk_level=RiskLevel.HIGH,
        obstacle_risks=[],
        min_ttc_sec=2.1,
        sensor_confidence={"camera": 0.9, "lidar": 0.95, "radar": 0.85},
        vehicle_health_status="OPERATIONAL",
        battery_soc_pct=72.0,
        recommended_action="SLOW DOWN + REROUTE",
        decision_reason="Obstacle detected",
        explanation="Test explanation",
    )

    output = controller.plan_step(
        vehicle_state=state,
        goal=(-7.0, 18.0),
        env=env,
        tracks=[],
        risk_report=report,
    )

    assert output.action == AdaptiveAction.SLOW_DOWN_AND_REROUTE
    assert output.is_rerouting
    assert output.normal_path is not None
    assert output.alternative_path is not None


def math_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))
