"""Load fault profiles from ``fault_profiles/*.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sensors.faults import FaultSchedule, FaultSpec, FaultType

FAULT_DIR = Path(__file__).resolve().parent.parent / "fault_profiles"


class FaultProfileError(ValueError):
    """Raised when a fault profile file is malformed."""


def parse_fault_profile(data: Any, source: str = "<dict>") -> FaultSchedule:
    if not isinstance(data, dict) or "name" not in data:
        raise FaultProfileError(f"{source}: missing required field 'name'")
    specs = []
    for i, item in enumerate(data.get("faults") or []):
        ctx = f"{source}.faults[{i}]"
        for key in ("sensor", "type", "level"):
            if key not in item:
                raise FaultProfileError(f"{ctx}: missing required field '{key}'")
        try:
            fault = FaultType(item["type"])
        except ValueError:
            valid = ", ".join(f.value for f in FaultType)
            raise FaultProfileError(
                f"{ctx}.type: unknown fault '{item['type']}' ({valid})"
            ) from None
        try:
            specs.append(
                FaultSpec(
                    sensor=str(item["sensor"]),
                    fault=fault,
                    level=float(item["level"]),
                    start=float(item.get("start", 0.0)),
                    end=None if item.get("end") is None else float(item["end"]),
                    ramp=float(item.get("ramp", 0.0)),
                )
            )
        except (TypeError, ValueError) as exc:
            raise FaultProfileError(f"{ctx}: {exc}") from exc
    return FaultSchedule(specs, str(data["name"]), str(data.get("description", "")))


def available_fault_profiles(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or FAULT_DIR).glob("*.yaml"))


def build_fault_schedule(name: str, directory: Path | None = None) -> FaultSchedule:
    folder = directory or FAULT_DIR
    if name not in available_fault_profiles(folder):
        valid = ", ".join(available_fault_profiles(folder)) or "none found"
        raise ValueError(f"Unknown fault profile '{name}'. Choose from: {valid}")
    path = folder / f"{name}.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FaultProfileError(f"{path.name}: invalid YAML ({exc})") from exc
    return parse_fault_profile(data, path.name)
