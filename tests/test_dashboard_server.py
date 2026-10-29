"""Unit tests for Phase 10: Dashboard server, REST API, and static files."""

from fastapi.testclient import TestClient

from dashboard.server import app

client = TestClient(app)


def test_index_page() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "AURA" in response.text
    assert "AUTONOMOUS INDUSTRIAL VEHICLE SYSTEM" in response.text


def test_status_api() -> None:
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "mode" in data


def test_command_api() -> None:
    response = client.post("/api/command", json={"command": "Return to charging station"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"


def test_faults_api() -> None:
    response = client.post("/api/faults", json={"camera": True, "lidar": False, "radar": False})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert data["faults"]["camera"] is True


def test_select_vehicle_api() -> None:
    response = client.post("/api/select_vehicle", json={"vehicle_id": "AMR-02"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "selected"
    assert data["vehicle_id"] == "AMR-02"


def test_sim_control_api() -> None:
    # Test pause
    r_pause = client.post("/api/sim/control", json={"action": "pause"})
    assert r_pause.status_code == 200
    assert r_pause.json()["paused"] is True

    # Test speed change
    r_speed = client.post("/api/sim/control", json={"action": "speed", "speed": 2.0})
    assert r_speed.status_code == 200
    assert r_speed.json()["speed"] == 2.0

    # Test resume
    r_resume = client.post("/api/sim/control", json={"action": "resume"})
    assert r_resume.status_code == 200
    assert r_resume.json()["paused"] is False

    # Test reset
    r_reset = client.post("/api/sim/control", json={"action": "reset"})
    assert r_reset.status_code == 200
    assert r_reset.json()["status"] == "ok"


def test_mission_dispatch_and_error_handling() -> None:
    # Valid structured mission dispatch
    payload = {
        "vehicle_id": "AMR-01",
        "quantity": 2,
        "pickup_warehouse": "Warehouse 1",
        "delivery_warehouse": "Warehouse 2",
    }
    r = client.post("/api/mission/dispatch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "accepted"
    assert data["vehicle_id"] == "AMR-01"

    # Same pickup and delivery should fail with 400
    bad_payload = {
        "vehicle_id": "AMR-01",
        "quantity": 1,
        "pickup_warehouse": "Warehouse 1",
        "delivery_warehouse": "Warehouse 1",
    }
    r_bad = client.post("/api/mission/dispatch", json=bad_payload)
    assert r_bad.status_code == 400
    assert "error" in r_bad.json()


def test_cancel_estop_and_approve_api() -> None:
    # E-stop
    r_estop = client.post("/api/mission/estop", json={"vehicle_id": "AMR-01"})
    assert r_estop.status_code == 200
    assert r_estop.json()["status"] == "estop_activated"

    # Cancel
    r_cancel = client.post("/api/mission/cancel", json={"vehicle_id": "AMR-01"})
    assert r_cancel.status_code == 200
    assert r_cancel.json()["status"] == "cancelled"

    # Worker approve when vehicle is not waiting for approval -> correctly rejected with 400
    r_reject = client.post("/api/approve", json={"vehicle_id": "AMR-01", "action": "approve_load", "dock_id": "DOCK_A"})
    assert r_reject.status_code == 400
    assert r_reject.json()["status"] == "rejected"

    # Put vehicle in WAITING_FOR_LOAD_APPROVAL state and verify approval succeeds
    from dashboard.server import fleet
    from mission.state_machine import MissionStatus
    fleet.vehicles["AMR-01"].state_machine.status = MissionStatus.WAITING_FOR_LOAD_APPROVAL
    fleet.vehicles["AMR-01"].state_machine.cargo_loaded = True

    r_approve = client.post("/api/approve", json={"vehicle_id": "AMR-01", "action": "approve_load", "dock_id": "DOCK_A"})
    assert r_approve.status_code == 200
    assert r_approve.json()["status"] == "approved"



