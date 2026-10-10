"""Parse and validate YAML scenario files into ready-to-run worlds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config import VehicleParams
from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.motion import LinearMotion, Motion, StaticMotion, WaypointMotion
from simulation.obstacles import Obstacle
from simulation.shapes import Circle, Rectangle, Shape
from simulation.vehicle import VehicleState


class ScenarioError(ValueError):
    """Raised when a scenario file is malformed. The message names the field."""


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    environment: Environment
    initial_state: VehicleState


def _get(data: dict[str, Any], key: str, ctx: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise ScenarioError(f"{ctx}: missing required field '{key}'")
    return data[key]


def _number(data: dict[str, Any], key: str, ctx: str, default: float | None = None):
    value = data.get(key, default) if default is not None else _get(data, key, ctx)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScenarioError(f"{ctx}.{key}: expected a number, got {value!r}")
    return float(value)


def parse_shape(data: dict[str, Any], ctx: str) -> Shape:
    kind = _get(data, "type", ctx)
    try:
        if kind == "circle":
            return Circle(_number(data, "radius", ctx))
        if kind == "rectangle":
            return Rectangle(_number(data, "width", ctx), _number(data, "height", ctx))
    except ValueError as exc:
        if isinstance(exc, ScenarioError):
            raise
        raise ScenarioError(f"{ctx}: {exc}") from exc
    raise ScenarioError(f"{ctx}.type: unknown shape '{kind}' (circle, rectangle)")


def parse_motion(data: dict[str, Any], ctx: str) -> Motion:
    kind = _get(data, "type", ctx)
    if kind == "static":
        return StaticMotion(_number(data, "x", ctx), _number(data, "y", ctx))
    if kind == "linear":
        return LinearMotion(
            _number(data, "x", ctx),
            _number(data, "y", ctx),
            _number(data, "vx", ctx),
            _number(data, "vy", ctx),
        )
    if kind == "waypoints":
        raw = _get(data, "points", ctx)
        try:
            points = [(float(px), float(py)) for px, py in raw]
            return WaypointMotion(
                points, _number(data, "speed", ctx), data.get("mode", "pingpong")
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ScenarioError):
                raise
            raise ScenarioError(f"{ctx}: {exc}") from exc
    raise ScenarioError(
        f"{ctx}.type: unknown motion '{kind}' (static, linear, waypoints)"
    )


def parse_scenario(
    data: dict[str, Any],
    seed: int = 42,
    source: str = "<dict>",
    vehicle_radius: float | None = None,
) -> ScenarioSpec:
    """Build a world from a dictionary. The seed controls position jitter."""
    radius = VehicleParams().radius if vehicle_radius is None else vehicle_radius
    name = str(_get(data, "name", source))
    map_cfg = _get(data, "map", source)
    width = _number(map_cfg, "width", f"{source}.map")
    height = _number(map_cfg, "height", f"{source}.map")
    if width <= 0 or height <= 0:
        raise ScenarioError(f"{source}.map: width and height must be > 0")
    bounds = Bounds(0.0, width, 0.0, height)
    start = _get(data, "start", source)
    goal = _get(data, "goal", source)
    sx, sy = _number(start, "x", f"{source}.start"), _number(start, "y", source)
    gx, gy = _number(goal, "x", f"{source}.goal"), _number(goal, "y", source)
    tolerance = _number(goal, "tolerance", f"{source}.goal", default=1.5)
    if not bounds.contains_circle(sx, sy, radius):
        raise ScenarioError(f"{source}.start: vehicle would start outside the map")
    if not bounds.contains_circle(gx, gy, 0.0):
        raise ScenarioError(f"{source}.goal: goal is outside the map")
    default_jitter = _number(data, "position_jitter", source, default=0.0)
    rng = np.random.default_rng(seed)
    obstacles: list[Obstacle] = []
    for i, item in enumerate(data.get("obstacles", [])):
        ctx = f"{source}.obstacles[{i}]"
        oid = str(_get(item, "id", ctx))
        shape = parse_shape(_get(item, "shape", ctx), f"{ctx}.shape")
        motion = parse_motion(_get(item, "motion", ctx), f"{ctx}.motion")
        jitter = _number(item, "jitter", ctx, default=default_jitter)
        offset = rng.uniform(-1.0, 1.0, size=2) * jitter  # always draw: stable order
        if jitter > 0:
            motion = motion.shifted(float(offset[0]), float(offset[1]))
        default_cat = "static" if isinstance(motion, StaticMotion) else "dynamic"
        obstacles.append(
            Obstacle(oid, shape, motion, str(item.get("category", default_cat)))
        )
    try:
        env = Environment(bounds, (gx, gy), obstacles, tolerance)
    except ValueError as exc:
        raise ScenarioError(f"{source}: {exc}") from exc
    if env.in_collision(sx, sy, radius):
        raise ScenarioError(f"{source}: vehicle starts inside an obstacle")
    if any(o.distance(gx, gy) <= 0.0 for o in obstacles if o.is_static):
        raise ScenarioError(f"{source}: goal lies inside a static obstacle")
    heading = _number(start, "heading", f"{source}.start", default=0.0)
    return ScenarioSpec(
        name=name,
        description=str(data.get("description", "")),
        environment=env,
        initial_state=VehicleState(sx, sy, heading=heading),
    )


def load_scenario_file(path: Path, seed: int = 42) -> ScenarioSpec:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path}: invalid YAML ({exc})") from exc
    return parse_scenario(data, seed=seed, source=Path(path).name)
