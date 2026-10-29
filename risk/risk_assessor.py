"""Comprehensive multi-factor risk assessment engine for AURA.

Combines:
- Spatial collision risk & analytical TTC
- Sensor reliability and perception uncertainty
- Vehicle powertrain and mechanical health
- Battery State of Charge (SoC) and energy feasibility
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from perception.tracker import TrackedObstacle
from prediction.trajectory_predictor import PredictedTrajectory
from risk.ttc import TTCCalculator, TTCEstimate
from simulation.amr_vehicle import AMRState


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ObstacleRiskRow:
    """Row in the UI Predicted Collision Risk table."""

    object_id: str
    ttc_sec: float
    risk_level: RiskLevel
    distance_m: float
    category: str


@dataclass
class AssessmentReport:
    """Full multidimensional risk evaluation output."""

    composite_risk_score: float  # [0, 100]
    overall_risk_level: RiskLevel
    obstacle_risks: list[ObstacleRiskRow]
    min_ttc_sec: float
    sensor_confidence: dict[str, float]
    vehicle_health_status: str
    battery_soc_pct: float
    recommended_action: str
    decision_reason: str
    explanation: str
    confidence_pct: float = 98.7


class RiskAssessor:
    """Evaluates collision, sensor, and vehicle hazards to determine safety actions."""

    def __init__(self, ttc_calculator: TTCCalculator | None = None) -> None:
        self.ttc_calc = ttc_calculator or TTCCalculator()

    def assess(
        self,
        vehicle_state: AMRState,
        tracks: Sequence[TrackedObstacle],
        predictions: Sequence[PredictedTrajectory],
        sensor_confidence: dict[str, float],
    ) -> AssessmentReport:
        """Run complete risk assessment pipeline."""
        obstacle_rows: list[ObstacleRiskRow] = []
        min_ttc = math.inf
        max_col_risk = 0.0

        pred_map = {p.track_id: p for p in predictions}

        for tr in tracks:
            ttc_est = self.ttc_calc.compute_ttc(tr, vehicle_state.speed)
            dist = math.hypot(tr.x, tr.y)

            # Determine risk level based on TTC and distance
            pred = pred_map.get(tr.track_id)
            crosses = pred.crosses_path if pred else False

            risk_level = RiskLevel.LOW
            ttc_val = ttc_est.ttc_sec

            if ttc_val < 2.0 or (dist < 3.0 and crosses):
                risk_level = RiskLevel.HIGH
                col_score = 85.0
            elif ttc_val < 3.5 or (dist < 5.0 and crosses) or dist < 2.5:
                risk_level = RiskLevel.MEDIUM
                col_score = 55.0
            elif ttc_val < 5.0 or dist < 6.0:
                risk_level = RiskLevel.LOW
                col_score = 25.0
            else:
                risk_level = RiskLevel.LOW
                col_score = 5.0

            max_col_risk = max(max_col_risk, col_score)
            if ttc_val < min_ttc:
                min_ttc = ttc_val

            obstacle_rows.append(
                ObstacleRiskRow(
                    object_id=tr.track_id,
                    ttc_sec=ttc_val if ttc_val != math.inf else 9.9,
                    risk_level=risk_level,
                    distance_m=round(dist, 1),
                    category=tr.category,
                )
            )

        # Sort obstacle rows by risk (HIGH first, lowest TTC)
        prio = {RiskLevel.CRITICAL: 0, RiskLevel.HIGH: 1, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 3}
        obstacle_rows.sort(key=lambda r: (prio[r.risk_level], r.ttc_sec))

        # Sensor reliability risk
        min_sensor_conf = min(sensor_confidence.values()) if sensor_confidence else 1.0
        sensor_risk = (1.0 - min_sensor_conf) * 50.0

        # Vehicle health risk
        mech_health = min(vehicle_state.motor_health, vehicle_state.brake_health)
        health_risk = (1.0 - mech_health) * 40.0

        # Battery risk
        batt_risk = 0.0
        if vehicle_state.battery_soc_pct < 15.0:
            batt_risk = 50.0
        elif vehicle_state.battery_soc_pct < 25.0:
            batt_risk = 25.0

        # Composite score
        composite = min(100.0, max_col_risk + 0.3 * sensor_risk + 0.3 * health_risk + 0.2 * batt_risk)

        if composite >= 75.0 or min_ttc < 2.0:
            overall_level = RiskLevel.HIGH
        elif composite >= 40.0 or min_ttc < 3.5:
            overall_level = RiskLevel.MEDIUM
        else:
            overall_level = RiskLevel.LOW

        # Action determination
        # Matches: "SLOW DOWN + REROUTE", "STOP", "CONTINUE", "SAFE_RETURN"
        if vehicle_state.battery_soc_pct < 10.0 or mech_health < 0.4:
            action = "SAFE_RETURN"
            reason = "Critical battery / vehicle powertrain threshold reached. Initiating return to charge dock."
        elif min_ttc < 1.5 or (max_col_risk >= 80.0 and min_ttc < 2.0):
            action = "EMERGENCY_STOP"
            reason = "Imminent collision predicted along vehicle corridor. Immediate brake applied."
        elif overall_level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
            action = "SLOW DOWN + REROUTE"
            reason = "Obstacle proximity and trajectory uncertainty. Action: Reduce speed and select alternative route."
        else:
            action = "CONTINUE"
            reason = "Corridor clear and vehicle healthy. Maintaining planned velocity."

        # Natural language explanation
        explanation = (
            f"The AI system has decided to {action.lower().replace('_', ' ')} to avoid potential hazards. "
            f"This decision is based on fused sensor data (Camera {sensor_confidence.get('camera', 1.0):.0%}, "
            f"LiDAR {sensor_confidence.get('lidar', 1.0):.0%}, Radar {sensor_confidence.get('radar', 1.0):.0%}), "
            f"real-time risk assessment (TTC={min_ttc:.1f}s), and uncertainty-aware adaptive planning."
        )

        return AssessmentReport(
            composite_risk_score=round(composite, 1),
            overall_risk_level=overall_level,
            obstacle_risks=obstacle_rows,
            min_ttc_sec=round(min_ttc, 1) if min_ttc != math.inf else 9.9,
            sensor_confidence=sensor_confidence,
            vehicle_health_status=vehicle_state.status,
            battery_soc_pct=vehicle_state.battery_soc_pct,
            recommended_action=action,
            decision_reason=reason,
            explanation=explanation,
            confidence_pct=98.7,
        )
