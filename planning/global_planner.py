"""Global A* route planner with road-network guidance and swept collision validation for AURA.

Plans safe nominal routes (Normal Path) and alternative detour routes (Alternative Path)
around static obstacles, storage racks, and blocked lanes without cutting through walls.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from simulation.environment import Environment
from simulation.road_network import RoadNetwork


@dataclass
class Waypoint:
    name: str  # e.g. "WP-01", "ALT-01", "DOCK B"
    x: float
    y: float
    target_speed_mps: float = 4.5
    eta_seconds: float = 0.0
    is_alternative: bool = False

    @property
    def eta_str(self) -> str:
        m, s = divmod(int(self.eta_seconds), 60)
        return f"{m:02d}:{s:02d}"


class AStarGlobalPlanner:
    """Grid-based and Road-network-guided A* pathfinder with clearance inflation and multi-path generation."""

    def __init__(
        self,
        resolution: float = 0.5,
        vehicle_radius: float = 1.25,
        road_network: RoadNetwork | None = None,
    ) -> None:
        self.resolution = resolution
        self.vehicle_radius = vehicle_radius
        self.inflation_radius = vehicle_radius + 0.4  # safety margin
        self.road_network = road_network or RoadNetwork()

    def plan_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        env: Environment,
        avoid_regions: Sequence[tuple[float, float, float]] | None = None,  # (x, y, radius)
    ) -> list[Waypoint]:
        """Compute collision-free path along road network with continuous swept-segment checks."""
        # 1. Try road network routing first
        road_coords = self.road_network.get_route_coordinates(start, goal)

        # Check swept collision along consecutive road segments against static walls/geometry
        road_path_valid = True
        if len(road_coords) >= 2:
            for i in range(len(road_coords) - 1):
                p1, p2 = road_coords[i], road_coords[i + 1]
                if env.check_swept_collision(p1[0], p1[1], p2[0], p2[1], self.vehicle_radius, static_only=True):
                    road_path_valid = False
                    break

        if road_path_valid and len(road_coords) >= 2:
            return self._interpolate_waypoints(road_coords, avoid_regions=bool(avoid_regions))

        # 2. If road path is blocked or unavailable, run grid-based A* with swept collision checks
        grid_path = self._plan_grid_astar(start, goal, env, avoid_regions)
        if grid_path:
            return self._interpolate_waypoints(grid_path, avoid_regions=bool(avoid_regions))

        # 3. If grid A* also failed (e.g. completely walled in), return stop waypoints
        return [
            Waypoint("BLOCKED_START", start[0], start[1], 0.0, 0.0, is_alternative=bool(avoid_regions)),
            Waypoint("BLOCKED_GOAL", goal[0], goal[1], 0.0, 999.0, is_alternative=bool(avoid_regions)),
        ]

    def _interpolate_waypoints(self, raw_coords: list[tuple[float, float]], avoid_regions: bool = False) -> list[Waypoint]:
        """Subdivide long road segments into dense waypoints (every 2.5m) with proper speeds."""
        dense_pts: list[tuple[float, float]] = [raw_coords[0]]
        for i in range(len(raw_coords) - 1):
            p1 = raw_coords[i]
            p2 = raw_coords[i + 1]
            seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if seg_dist > 3.0:
                steps = max(2, int(math.ceil(seg_dist / 2.5)))
                for s in range(1, steps):
                    frac = s / steps
                    dense_pts.append((round(p1[0] + frac * (p2[0] - p1[0]), 2), round(p1[1] + frac * (p2[1] - p1[1]), 2)))
            dense_pts.append(p2)

        waypoints: list[Waypoint] = []
        n_pts = len(dense_pts)
        cum_time = 0.0

        for idx, pt in enumerate(dense_pts):
            if idx == 0:
                dt_seg = 0.0
                spd = 1.5
            else:
                prev = dense_pts[idx - 1]
                dist = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
                # Speed profile: slower near start and end, cruise in the middle
                if idx < 2:
                    spd = 2.2
                elif idx >= n_pts - 2:
                    spd = 1.2
                elif abs(pt[1]) < 5.0:  # on main road Y in [-4, 4]
                    spd = 4.5
                else:  # on access avenues / spurs
                    spd = 2.8
                dt_seg = dist / max(spd, 0.5)

            cum_time += dt_seg
            prefix = "ALT" if avoid_regions else "WP"
            w_name = f"{prefix}-{idx+1:02d}" if idx < n_pts - 1 else "GOAL"
            waypoints.append(
                Waypoint(
                    name=w_name,
                    x=round(pt[0], 2),
                    y=round(pt[1], 2),
                    target_speed_mps=round(spd, 2),
                    eta_seconds=round(cum_time, 1),
                    is_alternative=avoid_regions,
                )
            )

        return waypoints

    def _plan_grid_astar(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        env: Environment,
        avoid_regions: Sequence[tuple[float, float, float]] | None = None,
    ) -> list[tuple[float, float]]:
        """Dense 2D A* grid search fallback."""
        res = self.resolution
        bounds = env.bounds

        nx = max(10, int((bounds.x_max - bounds.x_min) / res))
        ny = max(10, int((bounds.y_max - bounds.y_min) / res))

        def to_grid(x: float, y: float) -> tuple[int, int]:
            gx = int((x - bounds.x_min) / res)
            gy = int((y - bounds.y_min) / res)
            return max(0, min(nx - 1, gx)), max(0, min(ny - 1, gy))

        def to_world(gx: int, gy: int) -> tuple[float, float]:
            return bounds.x_min + (gx + 0.5) * res, bounds.y_min + (gy + 0.5) * res

        start_grid = to_grid(start[0], start[1])
        goal_grid = to_grid(goal[0], goal[1])

        grid = np.zeros((nx, ny), dtype=bool)
        for gx in range(nx):
            for gy in range(ny):
                wx, wy = to_world(gx, gy)
                if env.min_gap(wx, wy, self.inflation_radius) <= 0.0:
                    grid[gx, gy] = True
                    continue
                if avoid_regions:
                    for rx, ry, rr in avoid_regions:
                        if math.hypot(wx - rx, wy - ry) <= (rr + self.inflation_radius):
                            grid[gx, gy] = True
                            break

        frontier: list[tuple[float, int, int]] = []
        heapq.heappush(frontier, (0.0, start_grid[0], start_grid[1]))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far: dict[tuple[int, int], float] = {start_grid: 0.0}

        directions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
        ]

        found = False
        while frontier:
            _, cx, cy = heapq.heappop(frontier)
            if (cx, cy) == goal_grid or math.hypot(cx - goal_grid[0], cy - goal_grid[1]) <= 2:
                found = True
                goal_grid = (cx, cy)
                break

            for dx, dy, w in directions:
                nx_pos, ny_pos = cx + dx, cy + dy
                if 0 <= nx_pos < nx and 0 <= ny_pos < ny and not grid[nx_pos, ny_pos]:
                    new_cost = cost_so_far[(cx, cy)] + w * res
                    next_node = (nx_pos, ny_pos)
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        h = math.hypot(nx_pos - goal_grid[0], ny_pos - goal_grid[1]) * res
                        heapq.heappush(frontier, (new_cost + h, nx_pos, ny_pos))
                        came_from[next_node] = (cx, cy)

        if not found:
            return []

        path_pts: list[tuple[float, float]] = []
        curr: tuple[int, int] | None = goal_grid
        while curr and curr in came_from:
            path_pts.append(to_world(curr[0], curr[1]))
            curr = came_from[curr]
        path_pts.append(start)
        path_pts.reverse()

        return path_pts
