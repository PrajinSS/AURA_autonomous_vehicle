import copy

import pytest

from experiments.scenario_loader import ScenarioError, parse_scenario
from experiments.scenarios import available_scenarios, build_scenario

BASE = {
    "name": "t",
    "map": {"width": 50, "height": 30},
    "start": {"x": 5, "y": 15},
    "goal": {"x": 45, "y": 15},
    "position_jitter": 1.0,
    "obstacles": [
        {
            "id": "o1",
            "shape": {"type": "circle", "radius": 2},
            "motion": {"type": "static", "x": 25, "y": 15},
        },
        {
            "id": "f1",
            "shape": {"type": "rectangle", "width": 3, "height": 2},
            "motion": {"type": "waypoints", "points": [[30, 5], [30, 25]], "speed": 2},
        },
    ],
}


def _bad(mutator):
    data = copy.deepcopy(BASE)
    mutator(data)
    return data


def test_valid_scenario_builds():
    spec = parse_scenario(BASE, seed=1)
    assert len(spec.environment.obstacles) == 2
    assert spec.environment.obstacles[1].category == "dynamic"


def test_same_seed_same_world_different_seed_different_world():
    a = parse_scenario(BASE, seed=5).environment.obstacles[0]
    b = parse_scenario(BASE, seed=5).environment.obstacles[0]
    c = parse_scenario(BASE, seed=6).environment.obstacles[0]
    assert (a.x, a.y) == (b.x, b.y)
    assert (a.x, a.y) != (c.x, c.y)


@pytest.mark.parametrize(
    "mutator,fragment",
    [
        (lambda d: d.pop("map"), "missing required field 'map'"),
        (lambda d: d["map"].update(width=-5), "must be > 0"),
        (lambda d: d["start"].update(x=0.1), "outside the map"),
        (lambda d: d["goal"].update(x=500), "outside the map"),
        (lambda d: d["obstacles"][0]["shape"].update(type="hexagon"), "unknown shape"),
        (
            lambda d: d["obstacles"][0]["motion"].update(type="teleport"),
            "unknown motion",
        ),
        (
            lambda d: d["obstacles"][0]["shape"].update(radius="big"),
            "expected a number",
        ),
        (lambda d: d["obstacles"][0]["motion"].update(x=5, y=15), "starts inside"),
        (lambda d: d["obstacles"][1].update(id="o1"), "unique"),
        (lambda d: d["obstacles"][1]["motion"].update(mode="zigzag"), "mode"),
    ],
)
def test_invalid_scenarios_give_clear_errors(mutator, fragment):
    with pytest.raises(ScenarioError, match=fragment):
        parse_scenario(_bad(mutator), seed=1)


def test_goal_inside_static_obstacle_rejected():
    def move_goal(d):
        d["goal"].update(x=25, y=15)

    with pytest.raises(ScenarioError, match="goal lies inside"):
        parse_scenario(_bad(move_goal), seed=1)


def test_registry_lists_shipped_scenarios():
    assert {"static", "crossing", "cluttered", "warehouse", "intersection"} <= set(
        available_scenarios()
    )


def test_unknown_scenario_message():
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_scenario("nope")
