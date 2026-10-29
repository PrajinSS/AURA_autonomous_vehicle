"""Natural language and structured mission command parser for AURA.

Parses directives such as:
- "Vehicle 2: take the load from Warehouse 1 to Warehouse 2"
- "Pick up the pallet at Dock A and deliver it to Dock B"
- "Take 2 pallets from Warehouse 1 to Warehouse 3"
- "AMR-01: Return to charging station"
- "Abort mission and stop safely"

Enforces strict validation:
- Validates known warehouses, docks, and charging bays
- Rejects unknown locations (does NOT silently invent a destination)
- Validates load quantities (AMR capacity 1 to 4 pallets)
- Rejects contradictory commands (same source and destination)
- Identifies addressed vehicle (AMR-01, AMR-02, AMR-03) or defaults to auto-assignment
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from simulation.factory_world import DockLocation, FactoryWorld


class TaskType(str, Enum):
    PICKUP_AND_DELIVER = "PICKUP_AND_DELIVER"
    NAVIGATE_TO = "NAVIGATE_TO"
    RETURN_TO_CHARGE = "RETURN_TO_CHARGE"
    ABORT = "ABORT"


@dataclass
class MissionTask:
    task_id: str
    task_type: TaskType
    raw_command: str
    target_vehicle_id: str | None = None  # e.g. "AMR-01", "AMR-02", "AMR-03", or None for auto
    pickup_location: DockLocation | None = None
    delivery_location: DockLocation | None = None
    quantity: int = 1
    priority: int = 1  # 1 (normal) to 5 (highest/emergency)

    @property
    def display_text(self) -> str:
        prefix = f"[{self.target_vehicle_id}] " if self.target_vehicle_id else ""
        if self.task_type == TaskType.PICKUP_AND_DELIVER:
            p_name = self.pickup_location.name if self.pickup_location else "Source"
            d_name = self.delivery_location.name if self.delivery_location else "Destination"
            qty_str = f" ({self.quantity} pallet{'s' if self.quantity > 1 else ''})" if self.quantity > 1 else ""
            return f"{prefix}Pick up{qty_str} at {p_name} and deliver to {d_name}"
        elif self.task_type == TaskType.RETURN_TO_CHARGE:
            bay_name = self.delivery_location.name if self.delivery_location else "Charging Yard"
            return f"{prefix}Return to {bay_name}"
        elif self.task_type == TaskType.ABORT:
            return f"{prefix}Emergency abort and safe stop"
        elif self.task_type == TaskType.NAVIGATE_TO:
            d_name = self.delivery_location.name if self.delivery_location else "Destination"
            return f"{prefix}Navigate to {d_name}"
        return f"{prefix}{self.raw_command}"


class CommandParser:
    """Robust natural language and structured command parser for logistics fleet."""

    def __init__(self, locations: dict[str, DockLocation] | None = None) -> None:
        self.locations = locations or FactoryWorld.DOCKS
        self._counter = 0

    def _extract_vehicle_id(self, text: str) -> tuple[str | None, str]:
        """Extract addressed vehicle (e.g. 'Vehicle 2:', 'AMR-01') and return (vehicle_id, remainder)."""
        m = re.match(r"^\s*(?:vehicle|amr)[-\s]*([1-3]|0[1-3])\s*[:,-]?\s*(.*)$", text, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            return f"AMR-0{num}", m.group(2).strip()

        # Check inline "for vehicle 2" or "using amr-01"
        m_inline = re.search(r"\b(?:for|using|with)\s+(?:vehicle|amr)[-\s]*([1-3]|0[1-3])\b", text, re.IGNORECASE)
        if m_inline:
            num = int(m_inline.group(1))
            cleaned = text[:m_inline.start()] + text[m_inline.end():]
            return f"AMR-0{num}", cleaned.strip()

        return None, text

    def _extract_quantity(self, text: str) -> tuple[int, str]:
        """Extract pallet quantity if specified (e.g. '2 pallets', 'load of 3')."""
        m = re.search(r"\b([1-9][0-9]*)\s*(?:pallet|pallets|load|loads|cargo|units)\b", text, re.IGNORECASE)
        if m:
            qty = int(m.group(1))
            cleaned = text[:m.start()] + text[m.end():]
            return qty, cleaned.strip()
        return 1, text

    def _resolve_location(self, text: str, is_pickup: bool = True) -> DockLocation | None:
        """Find matching dock or warehouse location from user text."""
        cleaned = text.strip().lower()

        # Check exact names in locations dictionary
        for name, loc in self.locations.items():
            if name.lower() in cleaned:
                return loc

        # Warehouse matching: "warehouse 1", "warehouse 2", "warehouse 3", "wh1", "wh2", "wh3"
        wh_match = re.search(r"\b(?:warehouse|wh)[-\s]*([1-3])\b", cleaned)
        if wh_match:
            wh_num = int(wh_match.group(1))
            if wh_num == 1:
                return self.locations["Warehouse 1 Dock A (Pickup)"] if is_pickup else self.locations["Warehouse 1 Dock B (Delivery)"]
            elif wh_num == 2:
                return self.locations["Warehouse 2 Dock C (Pickup)"] if is_pickup else self.locations["Warehouse 2 Dock D (Delivery)"]
            elif wh_num == 3:
                return self.locations["Warehouse 3 Dock E (Pickup)"] if is_pickup else self.locations["Warehouse 3 Dock F (Delivery)"]

        # Dock letter extraction (e.g. "dock a", "dock b", "bay c")
        dock_match = re.search(r"\b(?:dock|bay|station)\s+([a-f])\b", cleaned)
        if dock_match:
            letter = dock_match.group(1).upper()
            target_name = f"Dock {letter}"
            if target_name in self.locations:
                return self.locations[target_name]

        # Specific charger matching
        ch_match = re.search(r"\b(?:charger|bay|ch)[-\s]*([1-3])\b", cleaned)
        if ch_match:
            ch_num = int(ch_match.group(1))
            return self.locations.get(f"Charger {ch_num}")

        # General charging
        if "charge" in cleaned or "charging" in cleaned:
            return self.locations.get("Charging Station") or self.locations.get("Charger 1")

        return None

    def parse(self, command_text: str, priority: int = 1) -> MissionTask:
        """Parse natural language command string into typed MissionTask with strict validation."""
        self._counter += 1
        task_id = f"task_{self._counter:03d}"
        text = command_text.strip()

        # 1. Extract addressed vehicle if any
        target_veh, remainder = self._extract_vehicle_id(text)

        # 2. Extract load quantity if any
        quantity, remainder = self._extract_quantity(remainder)
        if quantity < 1 or quantity > 4:
            raise ValueError(f"Invalid pallet quantity: {quantity}. Valid AMR payload is 1 to 4 pallets.")

        lower = remainder.lower()

        # 3. Emergency Abort / Stop command
        if any(w in lower for w in ("abort", "stop", "halt", "cancel", "emergency", "e-stop", "estop")):
            return MissionTask(
                task_id=task_id,
                task_type=TaskType.ABORT,
                raw_command=text,
                target_vehicle_id=target_veh,
                priority=5,
            )

        # 4. Return to charging station
        if "charging" in lower or ("return" in lower and "charge" in lower) or lower.startswith("charge"):
            charge_loc = self._resolve_location(remainder, is_pickup=False) or self.locations.get("Charger 1")
            return MissionTask(
                task_id=task_id,
                task_type=TaskType.RETURN_TO_CHARGE,
                raw_command=text,
                target_vehicle_id=target_veh,
                delivery_location=charge_loc,
                priority=priority,
            )

        # 5. Pickup and Deliver ("Take ... from X to Y" / "Pick up ... at X and deliver to Y")
        if any(w in lower for w in ("pick up", "deliver", "take", "transport", "move load", "carry", "transfer")):
            parts = re.split(r"\b(?:and deliver to|and deliver|deliver to|to)\b", remainder, flags=re.IGNORECASE)
            if len(parts) >= 2:
                source_part = parts[0]
                dest_part = parts[1]

                pickup_loc = self._resolve_location(source_part, is_pickup=True)
                delivery_loc = self._resolve_location(dest_part, is_pickup=False)

                if not pickup_loc:
                    raise ValueError(
                        f"Unknown pickup location in command: '{source_part.strip()}'. "
                        f"Supported locations: Warehouse 1, Warehouse 2, Warehouse 3, Docks A through F."
                    )
                if not delivery_loc:
                    raise ValueError(
                        f"Unknown delivery location in command: '{dest_part.strip()}'. "
                        f"Supported locations: Warehouse 1, Warehouse 2, Warehouse 3, Docks A through F."
                    )

                if pickup_loc.dock_id == delivery_loc.dock_id:
                    raise ValueError("Pickup and delivery locations cannot be the same dock.")

                return MissionTask(
                    task_id=task_id,
                    task_type=TaskType.PICKUP_AND_DELIVER,
                    raw_command=text,
                    target_vehicle_id=target_veh,
                    pickup_location=pickup_loc,
                    delivery_location=delivery_loc,
                    quantity=quantity,
                    priority=priority,
                )

        # 6. Single location navigation ("Go to Dock A", "Navigate to Warehouse 2")
        loc = self._resolve_location(remainder, is_pickup=False)
        if loc:
            return MissionTask(
                task_id=task_id,
                task_type=TaskType.NAVIGATE_TO,
                raw_command=text,
                target_vehicle_id=target_veh,
                delivery_location=loc,
                quantity=quantity,
                priority=priority,
            )

        # If none matched, do NOT invent a destination: reject with helpful error!
        raise ValueError(
            f"Could not parse mission directive: '{command_text}'. "
            f"Please specify a valid command (e.g. 'Vehicle 2: take the load from Warehouse 1 to Warehouse 2', "
            f"'Pick up pallet at Dock A and deliver to Dock B', or 'Return to charging station')."
        )
