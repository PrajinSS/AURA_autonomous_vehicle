"""Scenario registry: every ``scenarios/*.yaml`` file is one named scenario."""

from __future__ import annotations

from pathlib import Path

from experiments.scenario_loader import ScenarioSpec, load_scenario_file

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def available_scenarios(directory: Path | None = None) -> list[str]:
    folder = directory or SCENARIO_DIR
    return sorted(p.stem for p in folder.glob("*.yaml"))


def build_scenario(
    name: str, seed: int = 42, directory: Path | None = None
) -> ScenarioSpec:
    """Build a fresh (never shared) world for ``name``. Same seed, same world."""
    folder = directory or SCENARIO_DIR
    if name not in available_scenarios(folder):
        valid = ", ".join(available_scenarios(folder)) or "none found"
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {valid}")
    return load_scenario_file(folder / f"{name}.yaml", seed)
