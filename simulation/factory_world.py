"""Industrial factory and warehouse logistics environment model for AURA.

Campus Layout (Expanded, realistic industrial scale):
- Site bounds: X in [-65.0, 75.0], Y in [-40.0, 48.0] (140m x 88m)
- 3 Distinct Warehouses:
    * Warehouse 1 (West: Docks A & B, X in [-53, -27], Y in [24, 42])
    * Warehouse 2 (Central: Docks C & D, X in [-8, 18], Y in [24, 42])
    * Warehouse 3 (East: Docks E & F, X in [37, 63], Y in [24, 42])
- Dedicated Charging Yard (South, Y = -28.0):
    * Charger 1 (Bay CH-01, X = -25.0, Y = -28.0)
    * Charger 2 (Bay CH-02, X = -15.0, Y = -28.0)
    * Charger 3 (Bay CH-03, X = -5.0, Y = -28.0)
- Main Arterial Two-Lane Highway (Y = 0.0, width 8.0m)
- Solid warehouse perimeter walls with explicit dock doorways (Y = 24.0)
- Interior storage racks, exterior staging pallets, and crossing worker patrols
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.motion import LinearMotion, StaticMotion
from simulation.obstacles import Obstacle
from simulation.road_network import RoadNetwork
from simulation.shapes import Circle, Rectangle


@dataclass(frozen=True)
class DockLocation:
    name: str
    x: float
    y: float
    heading: float  # approach heading in radians (pi/2 = North into dock)
    dock_id: str
    warehouse_name: str = "Warehouse 1"
    is_pickup: bool = True


@dataclass(frozen=True)
class ChargingBay:
    name: str
    bay_id: str
    x: float
    y: float
    heading: float = 0.0


@dataclass
class WarehouseBuilding:
    name: str
    building_id: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    color_hex: int = 0x1e293b
    docks: list[str] = field(default_factory=list)


@dataclass
class WorkerAgent:
    id: str
    name: str
    waypoints: list[tuple[float, float]]
    speed: float = 0.9
    radius: float = 0.45
    t_offset: float = 0.0

    def position_at(self, t: float) -> tuple[float, float]:
        """Compute cyclic waypoint trajectory position at time t."""
        if len(self.waypoints) < 2:
            return self.waypoints[0]

        dists = []
        total_dist = 0.0
        n_pts = len(self.waypoints)
        for i in range(n_pts - 1):
            d = float(np.hypot(
                self.waypoints[i + 1][0] - self.waypoints[i][0],
                self.waypoints[i + 1][1] - self.waypoints[i][1],
            ))
            dists.append(d)
            total_dist += d

        d_loop = float(np.hypot(
            self.waypoints[0][0] - self.waypoints[-1][0],
            self.waypoints[0][1] - self.waypoints[-1][1],
        ))
        dists.append(d_loop)
        total_dist += d_loop

        cycle_time = total_dist / max(self.speed, 0.1)
        phase = ((t + self.t_offset) % cycle_time) * self.speed

        cum = 0.0
        for i in range(n_pts):
            seg_len = dists[i]
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i + 1) % n_pts]
            if cum + seg_len >= phase:
                ratio = (phase - cum) / max(seg_len, 1e-4)
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                return x, y
            cum += seg_len

        return self.waypoints[0]


class FactoryWorld:
    """Expanded multi-warehouse logistics campus digital twin."""

    # 3 Warehouses
    BUILDINGS: dict[str, WarehouseBuilding] = {
        "Warehouse 1": WarehouseBuilding(
            name="Warehouse 1",
            building_id="WH_01",
            x_min=-53.0,
            x_max=-27.0,
            y_min=24.0,
            y_max=42.0,
            color_hex=0x1e3a5f,
            docks=["Dock A", "Dock B"],
        ),
        "Warehouse 2": WarehouseBuilding(
            name="Warehouse 2",
            building_id="WH_02",
            x_min=-8.0,
            x_max=18.0,
            y_min=24.0,
            y_max=42.0,
            color_hex=0x1f2937,
            docks=["Dock C", "Dock D"],
        ),
        "Warehouse 3": WarehouseBuilding(
            name="Warehouse 3",
            building_id="WH_03",
            x_min=37.0,
            x_max=63.0,
            y_min=24.0,
            y_max=42.0,
            color_hex=0x312e81,
            docks=["Dock E", "Dock F"],
        ),
    }

    # Charging Bays
    CHARGING_BAYS: dict[str, ChargingBay] = {
        "Charger 1": ChargingBay("Charger 1", "CH_01", -25.0, -28.0, 0.0),
        "Charger 2": ChargingBay("Charger 2", "CH_02", -15.0, -28.0, 0.0),
        "Charger 3": ChargingBay("Charger 3", "CH_03", -5.0, -28.0, 0.0),
    }

    # Primary Docks and Aliases
    DOCKS: dict[str, DockLocation] = {
        # Warehouse 1
        "Warehouse 1 Dock A (Pickup)": DockLocation("Warehouse 1 Dock A (Pickup)", -46.0, 26.0, math.pi / 2, "DOCK_A", "Warehouse 1", is_pickup=True),
        "Warehouse 1 Dock B (Delivery)": DockLocation("Warehouse 1 Dock B (Delivery)", -34.0, 26.0, math.pi / 2, "DOCK_B", "Warehouse 1", is_pickup=False),
        # Warehouse 2
        "Warehouse 2 Dock C (Pickup)": DockLocation("Warehouse 2 Dock C (Pickup)", -1.0, 26.0, math.pi / 2, "DOCK_C", "Warehouse 2", is_pickup=True),
        "Warehouse 2 Dock D (Delivery)": DockLocation("Warehouse 2 Dock D (Delivery)", 11.0, 26.0, math.pi / 2, "DOCK_D", "Warehouse 2", is_pickup=False),
        # Warehouse 3
        "Warehouse 3 Dock E (Pickup)": DockLocation("Warehouse 3 Dock E (Pickup)", 44.0, 26.0, math.pi / 2, "DOCK_E", "Warehouse 3", is_pickup=True),
        "Warehouse 3 Dock F (Delivery)": DockLocation("Warehouse 3 Dock F (Delivery)", 56.0, 26.0, math.pi / 2, "DOCK_F", "Warehouse 3", is_pickup=False),

        # Aliases for concise command parsing
        "Dock A": DockLocation("Dock A", -46.0, 26.0, math.pi / 2, "DOCK_A", "Warehouse 1", is_pickup=True),
        "Dock B": DockLocation("Dock B", -34.0, 26.0, math.pi / 2, "DOCK_B", "Warehouse 1", is_pickup=False),
        "Dock C": DockLocation("Dock C", -1.0, 26.0, math.pi / 2, "DOCK_C", "Warehouse 2", is_pickup=True),
        "Dock D": DockLocation("Dock D", 11.0, 26.0, math.pi / 2, "DOCK_D", "Warehouse 2", is_pickup=False),
        "Dock E": DockLocation("Dock E", 44.0, 26.0, math.pi / 2, "DOCK_E", "Warehouse 3", is_pickup=True),
        "Dock F": DockLocation("Dock F", 56.0, 26.0, math.pi / 2, "DOCK_F", "Warehouse 3", is_pickup=False),

        # Chargers as dock destinations
        "Charger 1": DockLocation("Charger 1", -25.0, -28.0, 0.0, "CH_01", "Charging Yard", is_pickup=False),
        "Charger 2": DockLocation("Charger 2", -15.0, -28.0, 0.0, "CH_02", "Charging Yard", is_pickup=False),
        "Charger 3": DockLocation("Charger 3", -5.0, -28.0, 0.0, "CH_03", "Charging Yard", is_pickup=False),
        "Charging Station": DockLocation("Charging Station", -15.0, -28.0, 0.0, "CH_02", "Charging Yard", is_pickup=False),
    }

    def __init__(self, bounds: tuple[float, float, float, float] = (-65.0, 75.0, -40.0, 48.0)) -> None:
        self.bounds = Bounds(x_min=bounds[0], x_max=bounds[1], y_min=bounds[2], y_max=bounds[3])
        self.road_network = RoadNetwork()
        self.workers: list[WorkerAgent] = [
            # Worker 1 crossing arterial crosswalk near WH1 Avenue (-40, 4.0 to -40, -4.0)
            WorkerAgent(
                id="worker_01",
                name="Worker (Vest) OBSTACLE-01",
                waypoints=[(-28.0, 5.0), (-28.0, -5.0), (-28.0, 5.0)],
                speed=1.0,
                radius=0.45,
            ),
            # Worker 2 patrolling storage yard in South-East
            WorkerAgent(
                id="worker_02",
                name="Technician OBSTACLE-02",
                waypoints=[(20.0, -18.0), (35.0, -18.0), (20.0, -18.0)],
                speed=1.1,
                radius=0.45,
                t_offset=3.0,
            ),
            # Worker 3 crossing arterial crosswalk near WH2 Avenue (18, 5.0 to 18, -5.0)
            WorkerAgent(
                id="worker_03",
                name="Inspector OBSTACLE-03",
                waypoints=[(18.0, -5.0), (18.0, 5.0), (18.0, -5.0)],
                speed=0.9,
                radius=0.45,
                t_offset=6.0,
            ),
        ]

    def build_environment(self, t: float = 0.0, goal: tuple[float, float] = (-46.0, 26.0)) -> Environment:
        """Construct standard Environment object with authoritative solid wall colliders."""
        obstacles: list[Obstacle] = []

        # =========================================================================
        # SOLID WAREHOUSE PERIMETER WALLS (Registered as physical colliders!)
        # Prevents any vehicle from passing through walls at planning or movement time.
        # Doorways are explicit 4.0m gaps at Y = 24.0.
        # =========================================================================

        # ── Warehouse 1 Walls (X in [-53, -27], Y in [24, 42]) ──────────────────
        # North wall: center (-40.0, 42.0), width 26.0, thickness 1.0
        obstacles.append(Obstacle("wh1_wall_north", Rectangle(26.0, 1.0), StaticMotion(-40.0, 42.0), "wall"))
        # West wall: center (-53.0, 33.0), thickness 1.0, depth 18.0
        obstacles.append(Obstacle("wh1_wall_west", Rectangle(1.0, 18.0), StaticMotion(-53.0, 33.0), "wall"))
        # East wall: center (-27.0, 33.0), thickness 1.0, depth 18.0
        obstacles.append(Obstacle("wh1_wall_east", Rectangle(1.0, 18.0), StaticMotion(-27.0, 33.0), "wall"))
        # South front wall (Y = 24.0) with door openings at Dock A (-46) and Dock B (-34):
        # 1. West segment: X in [-53, -48] (width 5.0, center -50.5)
        obstacles.append(Obstacle("wh1_wall_south_w", Rectangle(5.0, 1.0), StaticMotion(-50.5, 24.0), "wall"))
        # 2. Mid segment between Dock A and B: X in [-44, -36] (width 8.0, center -40.0)
        obstacles.append(Obstacle("wh1_wall_south_mid", Rectangle(8.0, 1.0), StaticMotion(-40.0, 24.0), "wall"))
        # 3. East segment: X in [-32, -27] (width 5.0, center -29.5)
        obstacles.append(Obstacle("wh1_wall_south_e", Rectangle(5.0, 1.0), StaticMotion(-29.5, 24.0), "wall"))

        # ── Warehouse 2 Walls (X in [-8, 18], Y in [24, 42]) ────────────────────
        # North wall: center (5.0, 42.0), width 26.0, thickness 1.0
        obstacles.append(Obstacle("wh2_wall_north", Rectangle(26.0, 1.0), StaticMotion(5.0, 42.0), "wall"))
        # West wall: center (-8.0, 33.0), thickness 1.0, depth 18.0
        obstacles.append(Obstacle("wh2_wall_west", Rectangle(1.0, 18.0), StaticMotion(-8.0, 33.0), "wall"))
        # East wall: center (18.0, 33.0), thickness 1.0, depth 18.0
        obstacles.append(Obstacle("wh2_wall_east", Rectangle(1.0, 18.0), StaticMotion(18.0, 33.0), "wall"))
        # South front wall (Y = 24.0) with door openings at Dock C (-1) and Dock D (11):
        # 1. West segment: X in [-8, -3] (width 5.0, center -5.5)
        obstacles.append(Obstacle("wh2_wall_south_w", Rectangle(5.0, 1.0), StaticMotion(-5.5, 24.0), "wall"))
        # 2. Mid segment between Dock C and D: X in [1, 9] (width 8.0, center 5.0)
        obstacles.append(Obstacle("wh2_wall_south_mid", Rectangle(8.0, 1.0), StaticMotion(5.0, 24.0), "wall"))
        # 3. East segment: X in [13, 18] (width 5.0, center 15.5)
        obstacles.append(Obstacle("wh2_wall_south_e", Rectangle(5.0, 1.0), StaticMotion(15.5, 24.0), "wall"))

        # ── Warehouse 3 Walls (X in [37, 63], Y in [24, 42]) ────────────────────
        # North wall: center (50.0, 42.0), width 26.0, thickness 1.0
        obstacles.append(Obstacle("wh3_wall_north", Rectangle(26.0, 1.0), StaticMotion(50.0, 42.0), "wall"))
        # West wall: center (37.0, 33.0), thickness 1.0, depth 18.0
        obstacles.append(Obstacle("wh3_wall_west", Rectangle(1.0, 18.0), StaticMotion(37.0, 33.0), "wall"))
        # East wall: center (63.0, 33.0), thickness 1.0, depth 18.0
        obstacles.append(Obstacle("wh3_wall_east", Rectangle(1.0, 18.0), StaticMotion(63.0, 33.0), "wall"))
        # South front wall (Y = 24.0) with door openings at Dock E (44) and Dock F (56):
        # 1. West segment: X in [37, 42] (width 5.0, center 39.5)
        obstacles.append(Obstacle("wh3_wall_south_w", Rectangle(5.0, 1.0), StaticMotion(39.5, 24.0), "wall"))
        # 2. Mid segment between Dock E and F: X in [46, 54] (width 8.0, center 50.0)
        obstacles.append(Obstacle("wh3_wall_south_mid", Rectangle(8.0, 1.0), StaticMotion(50.0, 24.0), "wall"))
        # 3. East segment: X in [58, 63] (width 5.0, center 60.5)
        obstacles.append(Obstacle("wh3_wall_south_e", Rectangle(5.0, 1.0), StaticMotion(60.5, 24.0), "wall"))

        # =========================================================================
        # INTERIOR WAREHOUSE STORAGE RACKS & OBSTACLES
        # =========================================================================
        obstacles.append(Obstacle("wh1_rack_1", Rectangle(2.5, 9.0), StaticMotion(-40.0, 34.0), "static"))
        obstacles.append(Obstacle("wh2_rack_1", Rectangle(2.5, 9.0), StaticMotion(5.0, 34.0), "static"))
        obstacles.append(Obstacle("wh3_rack_1", Rectangle(2.5, 9.0), StaticMotion(50.0, 34.0), "static"))

        # External storage yard racks (South-East, X in [25, 45], Y in [-10, -22])
        obstacles.append(Obstacle("yard_rack_01", Rectangle(3.6, 8.0), StaticMotion(26.0, -15.0), "static"))
        obstacles.append(Obstacle("yard_rack_02", Rectangle(3.6, 8.0), StaticMotion(36.0, -15.0), "static"))

        # Staging Pallets
        obstacles.append(Obstacle("pallet_stack_01", Rectangle(1.6, 1.4), StaticMotion(-3.0, 9.0), "static"))
        obstacles.append(Obstacle("pallet_stack_02", Rectangle(1.6, 1.4), StaticMotion(22.0, -5.0), "static"))

        # Traffic cones
        obstacles.append(Obstacle("cone_01", Circle(0.35), StaticMotion(-25.0, 6.0), "static"))
        obstacles.append(Obstacle("cone_02", Circle(0.35), StaticMotion(35.0, 6.0), "static"))

        # Dynamic Worker Obstacles
        for w in self.workers:
            wx, wy = w.position_at(t)
            obstacles.append(Obstacle(w.id, Circle(w.radius), StaticMotion(wx, wy), "worker"))

        env = Environment(
            bounds=self.bounds,
            goal=goal,
            obstacles=obstacles,
        )
        env.update(t)
        return env
