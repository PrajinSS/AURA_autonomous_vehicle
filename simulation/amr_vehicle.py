"""Industrial Autonomous Mobile Robot (AMR) & Electric Forklift model for AURA.

Features:
- Non-linear vehicle kinematics with acceleration and steering limits
- Battery State of Charge (SoC) and energy consumption physics
- Powertrain health (motor health, brake health, inverter temperature)
- Cargo load awareness (pallet mass dynamics)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import VehicleParams
from simulation.geometry import wrap_angle


@dataclass
class AMRParams:
    """Industrial electric forklift / AMR specifications."""

    wheelbase: float = 1.8  # m
    track_width: float = 1.2  # m
    radius: float = 1.3  # m, collision footprint
    empty_mass_kg: float = 1200.0
    pallet_mass_kg: float = 450.0
    max_speed: float = 3.5  # m/s (~12.6 km/h)
    max_accel: float = 1.5  # m/s^2
    max_brake: float = 3.5  # m/s^2
    max_steer_rad: float = 0.75  # ~43 deg tight turning radius
    max_steer_rate_rad_s: float = 1.2  # rad/s
    battery_capacity_kwh: float = 24.0
    nominal_voltage_v: float = 48.0


@dataclass
class AMRState:
    """State vector of the AMR at a discrete time step."""

    x: float
    y: float
    heading: float  # radians
    speed: float = 0.0  # m/s
    steer: float = 0.0  # radians
    battery_soc_pct: float = 72.0  # State of Charge %
    battery_voltage_v: float = 48.2
    motor_health: float = 0.98  # [0, 1]
    brake_health: float = 0.96  # [0, 1]
    temperature_c: float = 38.5  # Inverter/motor temp
    runtime_remaining_hours: float = 4.8  # e.g. "4h 48m"
    cargo_loaded: bool = False
    status: str = "OPERATIONAL"

    @property
    def is_operational(self) -> bool:
        return self.status == "OPERATIONAL" and self.battery_soc_pct > 5.0 and self.motor_health > 0.2


class AMRVehicle:
    """Simulates physical dynamics and electro-mechanical sub-systems of the AMR."""

    def __init__(self, initial_state: AMRState | None = None, params: AMRParams | None = None) -> None:
        self.params = params or AMRParams()
        self.state = initial_state or AMRState(x=-15.0, y=0.0, heading=0.0)

    @property
    def total_mass(self) -> float:
        mass = self.params.empty_mass_kg
        if self.state.cargo_loaded:
            mass += self.params.pallet_mass_kg
        return mass

    def step(self, accel_cmd: float, steer_cmd: float, dt: float) -> AMRState:
        """Advance AMR state by dt seconds."""
        p = self.params
        s = self.state

        # 1. Clamp steering command and enforce rate limit
        target_steer = max(-p.max_steer_rad, min(p.max_steer_rad, steer_cmd))
        d_steer = target_steer - s.steer
        max_d_steer = p.max_steer_rate_rad_s * dt
        actual_steer = s.steer + max(-max_d_steer, min(max_d_steer, d_steer))

        # 2. Acceleration limits based on motor & brake health
        effective_max_accel = p.max_accel * s.motor_health
        effective_max_brake = p.max_brake * s.brake_health
        actual_accel = max(-effective_max_brake, min(effective_max_accel, accel_cmd))

        # 3. Kinematic update (Bicycle model)
        new_speed = max(0.0, min(p.max_speed, s.speed + actual_accel * dt))
        v_mid = 0.5 * (s.speed + new_speed)

        # Yaw rate = v / L * tan(delta)
        yaw_rate = (v_mid / p.wheelbase) * math.tan(actual_steer)
        new_heading = wrap_angle(s.heading + yaw_rate * dt)

        # Displacements
        dx = v_mid * math.cos(new_heading) * dt
        dy = v_mid * math.sin(new_heading) * dt
        new_x = s.x + dx
        new_y = s.y + dy

        # 4. Energy & Battery consumption model
        # Idle power + kinetic drive power (P = F * v = m * a * v)
        p_idle_kw = 0.35  # electronics, sensors, computers
        p_drive_kw = 0.0
        if actual_accel > 0:
            p_drive_kw = (self.total_mass * actual_accel * v_mid) / 1000.0 / 0.85  # 85% inverter efficiency
        p_total_kw = p_idle_kw + p_drive_kw
        energy_kwh_used = (p_total_kw * dt) / 3600.0
        soc_drop_pct = (energy_kwh_used / p.battery_capacity_kwh) * 100.0
        new_soc = max(0.0, s.battery_soc_pct - soc_drop_pct)

        # Estimated runtime remaining
        avg_power_kw = 2.5  # typical average logistics power
        remaining_kwh = (new_soc / 100.0) * p.battery_capacity_kwh
        runtime_hrs = remaining_kwh / avg_power_kw

        # 5. Mechanical & Thermal degradation model
        # Motor degrades slightly under high power; brake pads degrade under hard braking
        brake_wear = 0.0
        if actual_accel < -1.5:
            brake_wear = 0.00002 * abs(actual_accel) * dt
        motor_wear = 0.000005 * (p_total_kw / 5.0) * dt

        new_motor_health = max(0.1, s.motor_health - motor_wear)
        new_brake_health = max(0.1, s.brake_health - brake_wear)

        # Thermal heating
        temp_target = 35.0 + 4.0 * (p_total_kw / 3.0)
        temp_rate = 0.1  # thermal lag
        new_temp = s.temperature_c + (temp_target - s.temperature_c) * temp_rate * dt

        # Operational status check
        status = "OPERATIONAL"
        if new_soc < 10.0:
            status = "LOW_BATTERY"
        elif new_motor_health < 0.5 or new_brake_health < 0.5 or new_temp > 65.0:
            status = "MAINTENANCE_REQUIRED"

        self.state = AMRState(
            x=new_x,
            y=new_y,
            heading=new_heading,
            speed=new_speed,
            steer=actual_steer,
            battery_soc_pct=round(new_soc, 2),
            battery_voltage_v=round(44.0 + (new_soc / 100.0) * 8.0, 1),
            motor_health=round(new_motor_health, 3),
            brake_health=round(new_brake_health, 3),
            temperature_c=round(new_temp, 1),
            runtime_remaining_hours=round(runtime_hrs, 1),
            cargo_loaded=s.cargo_loaded,
            status=status,
        )
        return self.state
