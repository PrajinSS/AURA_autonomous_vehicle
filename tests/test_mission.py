"""Unit tests for Phase 7: Mission command parser, state machine, and task queue."""

import pytest

from mission.parser import CommandParser, MissionTask, TaskType
from mission.state_machine import MissionProgress, MissionStateMachine, MissionStatus
from mission.task_queue import TaskQueue


def test_command_parser_pickup_and_deliver() -> None:
    parser = CommandParser()
    cmd = "Pick up pallet at Dock A and deliver to Dock B"
    task = parser.parse(cmd)
    assert task.task_type == TaskType.PICKUP_AND_DELIVER
    assert task.pickup_location is not None and task.pickup_location.name == "Dock A"
    assert task.delivery_location is not None and task.delivery_location.name == "Dock B"


def test_command_parser_charge_and_abort() -> None:
    parser = CommandParser()
    t_charge = parser.parse("Return to charging station")
    assert t_charge.task_type == TaskType.RETURN_TO_CHARGE

    t_abort = parser.parse("Abort mission immediately!")
    assert t_abort.task_type == TaskType.ABORT
    assert t_abort.priority == 5


def test_mission_state_machine_lifecycle() -> None:
    parser = CommandParser()
    sm = MissionStateMachine(waypoint_tolerance=1.5)
    task = parser.parse("Pick up pallet at Dock A and deliver to Dock B")

    start_pose = (-25.0, 0.0)
    sm.set_task(task, start_pose)
    assert sm.status == MissionStatus.EN_ROUTE_PICKUP

    # Move near Dock A (-46.0, 26.0)
    dock_a_x, dock_a_y = task.pickup_location.x, task.pickup_location.y
    progress = sm.update((dock_a_x, dock_a_y - 0.5), speed=1.5, dt=0.05)
    assert progress.status == MissionStatus.PICKING
    assert not progress.cargo_loaded

    # Dwell at Dock A for pickup completion
    for _ in range(45):
        progress = sm.update((dock_a_x, dock_a_y), speed=0.0, dt=0.05)

    assert progress.status == MissionStatus.EN_ROUTE_DELIVERY
    assert progress.cargo_loaded

    # Move to Dock B (-34.0, 26.0)
    dock_b_x, dock_b_y = task.delivery_location.x, task.delivery_location.y
    for _ in range(50):
        progress = sm.update((dock_b_x, dock_b_y), speed=1.5, dt=0.05)

    assert progress.status in (MissionStatus.UNLOADING, MissionStatus.COMPLETED)



def test_task_queue() -> None:
    queue = TaskQueue()
    parser = CommandParser()

    t1 = parser.parse("Pick up pallet at Dock A and deliver to Dock B", priority=1)
    t2 = parser.parse("Emergency abort", priority=5)

    queue.push(t1)
    queue.push(t2)

    # Priority 5 popped before priority 1
    popped = queue.pop()
    assert popped is not None
    assert popped.task_type == TaskType.ABORT
