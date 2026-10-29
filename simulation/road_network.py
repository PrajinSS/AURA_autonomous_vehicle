"""Authoritative road network and navigation graph for AURA Logistics Campus.

Defines the connected traversable road geometry:
- Main arterial East-West two-lane highway (Y = 0.0)
- South charging yard access avenue and charging lane (Y = -24.0 to -28.0)
- North warehouse access avenues connecting to Warehouse 1, 2, and 3
- Dedicated dock access spurs with explicit entrance/exit doorways
- Graph search (A*) ensuring vehicles navigate only on legitimate roads and access spurs
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class RoadNode:
    """A junction, waypoint, or terminal on the road network."""

    id: str
    x: float
    y: float
    kind: str = "main"  # "main", "avenue", "spur", "dock", "charger"


@dataclass(frozen=True)
class RoadSegment:
    """A traversable lane connecting two road nodes."""

    u: str
    v: str
    width: float = 6.0
    speed_limit: float = 4.5
    bidirectional: bool = True

    @property
    def key(self) -> tuple[str, str]:
        return (self.u, self.v)


class RoadNetwork:
    """Manages the topological and spatial road graph of the campus."""

    def __init__(self) -> None:
        self.nodes: dict[str, RoadNode] = {}
        self.adjacency: dict[str, list[tuple[str, float]]] = {}  # u -> list of (v, cost)
        self.segments: list[RoadSegment] = []
        self._build_network()

    def _add_node(self, node_id: str, x: float, y: float, kind: str = "main") -> None:
        self.nodes[node_id] = RoadNode(id=node_id, x=round(x, 2), y=round(y, 2), kind=kind)
        if node_id not in self.adjacency:
            self.adjacency[node_id] = []

    def _add_edge(self, u: str, v: str, width: float = 6.0, speed_limit: float = 4.5, bidirectional: bool = True) -> None:
        n1 = self.nodes[u]
        n2 = self.nodes[v]
        cost = math.hypot(n1.x - n2.x, n1.y - n2.y)
        self.adjacency[u].append((v, cost))
        self.segments.append(RoadSegment(u=u, v=v, width=width, speed_limit=speed_limit, bidirectional=bidirectional))
        if bidirectional:
            self.adjacency[v].append((u, cost))

    def _build_network(self) -> None:
        """Construct the complete connected campus road network."""
        # 1. Main Arterial Highway (Y = 0.0)
        self._add_node("MAIN_WEST_END", -55.0, 0.0, "main")
        self._add_node("J_WH1", -40.0, 0.0, "main")
        self._add_node("J_CHARGER", -15.0, 0.0, "main")
        self._add_node("J_WH2", 5.0, 0.0, "main")
        self._add_node("J_MID_EAST", 28.0, 0.0, "main")
        self._add_node("J_WH3", 50.0, 0.0, "main")
        self._add_node("MAIN_EAST_END", 65.0, 0.0, "main")

        # Connect main road
        self._add_edge("MAIN_WEST_END", "J_WH1", width=8.0, speed_limit=4.5)
        self._add_edge("J_WH1", "J_CHARGER", width=8.0, speed_limit=4.5)
        self._add_edge("J_CHARGER", "J_WH2", width=8.0, speed_limit=4.5)
        self._add_edge("J_WH2", "J_MID_EAST", width=8.0, speed_limit=4.5)
        self._add_edge("J_MID_EAST", "J_WH3", width=8.0, speed_limit=4.5)
        self._add_edge("J_WH3", "MAIN_EAST_END", width=8.0, speed_limit=4.5)

        # 2. Charging Yard Network (South, Y <= 0)
        # Charging access avenue connects J_CHARGER (-15, 0) to CH_LANE_MID (-15, -24)
        self._add_node("CH_AVE_MID", -15.0, -12.0, "avenue")
        self._add_node("CH_LANE_MID", -15.0, -24.0, "avenue")
        self._add_edge("J_CHARGER", "CH_AVE_MID", width=6.0, speed_limit=3.5)
        self._add_edge("CH_AVE_MID", "CH_LANE_MID", width=6.0, speed_limit=3.5)

        # Charging staging lane along Y = -24.0
        self._add_node("CH_LANE_W", -25.0, -24.0, "spur")
        self._add_node("CH_LANE_E", -5.0, -24.0, "spur")
        self._add_edge("CH_LANE_W", "CH_LANE_MID", width=6.0, speed_limit=2.5)
        self._add_edge("CH_LANE_MID", "CH_LANE_E", width=6.0, speed_limit=2.5)

        # Charging Bays (Y = -28.0)
        self._add_node("BAY_CH01", -25.0, -28.0, "charger")
        self._add_node("BAY_CH02", -15.0, -28.0, "charger")
        self._add_node("BAY_CH03", -5.0, -28.0, "charger")
        self._add_edge("CH_LANE_W", "BAY_CH01", width=4.0, speed_limit=1.5)
        self._add_edge("CH_LANE_MID", "BAY_CH02", width=4.0, speed_limit=1.5)
        self._add_edge("CH_LANE_E", "BAY_CH03", width=4.0, speed_limit=1.5)

        # 3. Warehouse 1 (West: Docks A & B)
        # Access Avenue North from J_WH1 (-40, 0) to WH1_APRON (-40, 18)
        self._add_node("WH1_AVE_MID", -40.0, 9.0, "avenue")
        self._add_node("WH1_APRON", -40.0, 18.0, "avenue")
        self._add_edge("J_WH1", "WH1_AVE_MID", width=6.0, speed_limit=3.5)
        self._add_edge("WH1_AVE_MID", "WH1_APRON", width=6.0, speed_limit=3.5)

        # Dock A Spur (West): Apron -> Approach -> Door -> Bay
        self._add_node("WH1_SPUR_A", -46.0, 18.0, "spur")
        self._add_node("DOOR_DOCK_A", -46.0, 24.0, "spur")
        self._add_node("BAY_DOCK_A", -46.0, 26.5, "dock")
        self._add_edge("WH1_APRON", "WH1_SPUR_A", width=5.0, speed_limit=2.5)
        self._add_edge("WH1_SPUR_A", "DOOR_DOCK_A", width=4.5, speed_limit=1.8)
        self._add_edge("DOOR_DOCK_A", "BAY_DOCK_A", width=4.0, speed_limit=1.2)

        # Dock B Spur (East): Apron -> Approach -> Door -> Bay
        self._add_node("WH1_SPUR_B", -34.0, 18.0, "spur")
        self._add_node("DOOR_DOCK_B", -34.0, 24.0, "spur")
        self._add_node("BAY_DOCK_B", -34.0, 26.5, "dock")
        self._add_edge("WH1_APRON", "WH1_SPUR_B", width=5.0, speed_limit=2.5)
        self._add_edge("WH1_SPUR_B", "DOOR_DOCK_B", width=4.5, speed_limit=1.8)
        self._add_edge("DOOR_DOCK_B", "BAY_DOCK_B", width=4.0, speed_limit=1.2)

        # 4. Warehouse 2 (Central: Docks C & D)
        # Access Avenue North from J_WH2 (5, 0) to WH2_APRON (5, 18)
        self._add_node("WH2_AVE_MID", 5.0, 9.0, "avenue")
        self._add_node("WH2_APRON", 5.0, 18.0, "avenue")
        self._add_edge("J_WH2", "WH2_AVE_MID", width=6.0, speed_limit=3.5)
        self._add_edge("WH2_AVE_MID", "WH2_APRON", width=6.0, speed_limit=3.5)

        # Dock C Spur (West): Apron -> Approach -> Door -> Bay
        self._add_node("WH2_SPUR_C", -1.0, 18.0, "spur")
        self._add_node("DOOR_DOCK_C", -1.0, 24.0, "spur")
        self._add_node("BAY_DOCK_C", -1.0, 26.5, "dock")
        self._add_edge("WH2_APRON", "WH2_SPUR_C", width=5.0, speed_limit=2.5)
        self._add_edge("WH2_SPUR_C", "DOOR_DOCK_C", width=4.5, speed_limit=1.8)
        self._add_edge("DOOR_DOCK_C", "BAY_DOCK_C", width=4.0, speed_limit=1.2)

        # Dock D Spur (East): Apron -> Approach -> Door -> Bay
        self._add_node("WH2_SPUR_D", 11.0, 18.0, "spur")
        self._add_node("DOOR_DOCK_D", 11.0, 24.0, "spur")
        self._add_node("BAY_DOCK_D", 11.0, 26.5, "dock")
        self._add_edge("WH2_APRON", "WH2_SPUR_D", width=5.0, speed_limit=2.5)
        self._add_edge("WH2_SPUR_D", "DOOR_DOCK_D", width=4.5, speed_limit=1.8)
        self._add_edge("DOOR_DOCK_D", "BAY_DOCK_D", width=4.0, speed_limit=1.2)

        # 5. Warehouse 3 (East: Docks E & F)
        # Access Avenue North from J_WH3 (50, 0) to WH3_APRON (50, 18)
        self._add_node("WH3_AVE_MID", 50.0, 9.0, "avenue")
        self._add_node("WH3_APRON", 50.0, 18.0, "avenue")
        self._add_edge("J_WH3", "WH3_AVE_MID", width=6.0, speed_limit=3.5)
        self._add_edge("WH3_AVE_MID", "WH3_APRON", width=6.0, speed_limit=3.5)

        # Dock E Spur (West): Apron -> Approach -> Door -> Bay
        self._add_node("WH3_SPUR_E", 44.0, 18.0, "spur")
        self._add_node("DOOR_DOCK_E", 44.0, 24.0, "spur")
        self._add_node("BAY_DOCK_E", 44.0, 26.5, "dock")
        self._add_edge("WH3_APRON", "WH3_SPUR_E", width=5.0, speed_limit=2.5)
        self._add_edge("WH3_SPUR_E", "DOOR_DOCK_E", width=4.5, speed_limit=1.8)
        self._add_edge("DOOR_DOCK_E", "BAY_DOCK_E", width=4.0, speed_limit=1.2)

        # Dock F Spur (East): Apron -> Approach -> Door -> Bay
        self._add_node("WH3_SPUR_F", 56.0, 18.0, "spur")
        self._add_node("DOOR_DOCK_F", 56.0, 24.0, "spur")
        self._add_node("BAY_DOCK_F", 56.0, 26.5, "dock")
        self._add_edge("WH3_APRON", "WH3_SPUR_F", width=5.0, speed_limit=2.5)
        self._add_edge("WH3_SPUR_F", "DOOR_DOCK_F", width=4.5, speed_limit=1.8)
        self._add_edge("DOOR_DOCK_F", "BAY_DOCK_F", width=4.0, speed_limit=1.2)

    def find_nearest_node(self, x: float, y: float, filter_kind: str | None = None) -> str:
        """Locate closest road graph node to arbitrary world position."""
        best_id = "J_CHARGER"
        best_dist = math.inf
        for nid, node in self.nodes.items():
            if filter_kind and node.kind != filter_kind:
                continue
            d = math.hypot(node.x - x, node.y - y)
            if d < best_dist:
                best_dist = d
                best_id = nid
        return best_id

    def shortest_path(self, start_id: str, goal_id: str) -> list[str]:
        """Dijkstra / A* search along the road graph."""
        if start_id not in self.nodes or goal_id not in self.nodes:
            return []
        if start_id == goal_id:
            return [start_id]

        g_score: dict[str, float] = {start_id: 0.0}
        came_from: dict[str, str] = {}
        target_node = self.nodes[goal_id]

        open_set: list[tuple[float, str]] = []
        heapq.heappush(open_set, (0.0, start_id))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor, cost in self.adjacency.get(current, []):
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = math.hypot(
                        self.nodes[neighbor].x - target_node.x,
                        self.nodes[neighbor].y - target_node.y,
                    )
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

        return []

    def path_length(self, node_path: list[str]) -> float:
        """Calculate total Euclidean length of a sequence of road nodes."""
        if len(node_path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(node_path) - 1):
            n1 = self.nodes[node_path[i]]
            n2 = self.nodes[node_path[i + 1]]
            total += math.hypot(n1.x - n2.x, n1.y - n2.y)
        return total

    def get_route_coordinates(self, start_pos: tuple[float, float], goal_pos: tuple[float, float]) -> list[tuple[float, float]]:
        """Compute ordered (x, y) waypoints connecting start to goal via road network."""
        goal_node = self.find_nearest_node(goal_pos[0], goal_pos[1])
        nearest_start = self.find_nearest_node(start_pos[0], start_pos[1])
        d_near_start = math.hypot(start_pos[0] - self.nodes[nearest_start].x, start_pos[1] - self.nodes[nearest_start].y)

        # 1. If right at a node (e.g. starting docked at a bay or exact junction), start directly from it
        if d_near_start <= 1.5:
            best_path = self.shortest_path(nearest_start, goal_node)
        else:
            # 2. Find the closest road segment (u, v) and evaluate road paths via u or v
            best_seg: tuple[str, str] | None = None
            best_seg_dist = math.inf

            for seg in self.segments:
                u_node = self.nodes[seg.u]
                v_node = self.nodes[seg.v]
                dx = v_node.x - u_node.x
                dy = v_node.y - u_node.y
                l2 = dx * dx + dy * dy
                if l2 < 1e-4:
                    continue
                t = max(0.0, min(1.0, ((start_pos[0] - u_node.x) * dx + (start_pos[1] - u_node.y) * dy) / l2))
                proj_x = u_node.x + t * dx
                proj_y = u_node.y + t * dy
                dist = math.hypot(start_pos[0] - proj_x, start_pos[1] - proj_y)
                if dist < best_seg_dist:
                    best_seg_dist = dist
                    best_seg = (seg.u, seg.v)

            best_path = []
            if best_seg:
                u, v = best_seg
                p_u = self.shortest_path(u, goal_node)
                p_v = self.shortest_path(v, goal_node)

                d_u = math.hypot(start_pos[0] - self.nodes[u].x, start_pos[1] - self.nodes[u].y) + self.path_length(p_u) if p_u else math.inf
                d_v = math.hypot(start_pos[0] - self.nodes[v].x, start_pos[1] - self.nodes[v].y) + self.path_length(p_v) if p_v else math.inf

                best_path = p_u if d_u < d_v else p_v
            else:
                best_path = self.shortest_path(nearest_start, goal_node)

        if not best_path:
            return [start_pos, goal_pos]

        coords: list[tuple[float, float]] = []
        # Connect start to first road node if not right on top of it
        first_node = self.nodes[best_path[0]]
        if math.hypot(start_pos[0] - first_node.x, start_pos[1] - first_node.y) > 0.5:
            coords.append((round(start_pos[0], 2), round(start_pos[1], 2)))

        for nid in best_path:
            n = self.nodes[nid]
            coords.append((n.x, n.y))

        # Connect last road node to goal if not right on top of it
        if math.hypot(goal_pos[0] - coords[-1][0], goal_pos[1] - coords[-1][1]) > 0.5:
            coords.append((round(goal_pos[0], 2), round(goal_pos[1], 2)))

        return coords

