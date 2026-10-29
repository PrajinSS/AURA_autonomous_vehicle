"""FastAPI & WebSocket real-time telemetry streaming server for AURA.

Connects the multi-vehicle autonomous logistics fleet (AMR-01, AMR-02, AMR-03,
perception pipelines, risk assessor, traffic coordination, and warehouse campus)
to the high-performance Web HUD and 3D Viewport.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mission.parser import CommandParser, MissionTask, TaskType
from simulation.fleet_manager import FleetManager

logger = logging.getLogger("aura.server")

app = FastAPI(title="AURA Industrial Vehicle Campus Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Web directory
WEB_DIR = Path(__file__).parent / "web"
app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")


# Request Models
class CommandRequest(BaseModel):
    command: str


class StructuredMissionRequest(BaseModel):
    vehicle_id: str = "auto"
    pickup_warehouse: str = "Warehouse 1"
    delivery_warehouse: str = "Warehouse 2"
    quantity: int = 1


class FaultsRequest(BaseModel):
    vehicle_id: str = "AMR-01"
    camera: bool = False
    lidar: bool = False
    radar: bool = False


class ApprovalRequest(BaseModel):
    vehicle_id: str = "AMR-01"
    action: str  # "approve_load" or "approve_unload"
    dock_id: str | None = None


class SelectVehicleRequest(BaseModel):
    vehicle_id: str


class SimControlRequest(BaseModel):
    action: str  # "play", "pause", "resume", "reset", "speed"
    speed: float = 1.0


# Authoritative Global Fleet Instance
fleet = FleetManager()


@app.get("/")
async def get_index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "mode": "AURA Multi-Vehicle Campus Digital Twin",
        "vehicles_count": len(fleet.vehicles),
        "selected_vehicle": fleet.selected_vehicle_id,
        "sim_time": round(fleet.sim_time, 2),
        "paused": fleet.is_paused,
    }


@app.post("/api/command")
async def post_command(req: CommandRequest):
    """Natural language or directive command endpoint with validation."""
    try:
        task = fleet.parser.parse(req.command)
        assigned_veh = fleet.dispatch_mission(task)
        return {
            "status": "accepted",
            "task_id": task.task_id,
            "vehicle_id": assigned_veh,
            "display_text": task.display_text,
        }
    except Exception as e:
        logger.warning(f"Invalid command '{req.command}': {e}")
        return JSONResponse(status_code=400, content={"status": "rejected", "error": str(e)})


@app.post("/api/mission/dispatch")
async def post_structured_mission(req: StructuredMissionRequest):
    """Structured mission dispatcher from the warehouse UI form."""
    if req.pickup_warehouse.strip().lower() == req.delivery_warehouse.strip().lower():
        return JSONResponse(
            status_code=400,
            content={"status": "rejected", "error": "Pickup warehouse and delivery warehouse cannot be the same."},
        )
    try:
        # Build command string
        veh_prefix = f"{req.vehicle_id}: " if req.vehicle_id and req.vehicle_id != "auto" else ""
        cmd_str = f"{veh_prefix}Take {req.quantity} pallet{'s' if req.quantity > 1 else ''} from {req.pickup_warehouse} to {req.delivery_warehouse}"

        task = fleet.parser.parse(cmd_str)
        assigned_veh = fleet.dispatch_mission(task)
        return {
            "status": "accepted",
            "task_id": task.task_id,
            "vehicle_id": assigned_veh,
            "display_text": task.display_text,
        }
    except Exception as e:
        logger.warning(f"Mission dispatch error: {e}")
        return JSONResponse(status_code=400, content={"status": "rejected", "error": str(e)})


@app.post("/api/mission/cancel")
async def post_cancel_mission(req: SelectVehicleRequest):
    """Cancel mission for specified vehicle."""
    fleet.cancel_mission(req.vehicle_id)
    return {"status": "cancelled", "vehicle_id": req.vehicle_id}


@app.post("/api/mission/estop")
async def post_estop(req: SelectVehicleRequest):
    """Emergency stop for specified vehicle or all."""
    v_id = None if req.vehicle_id == "all" else req.vehicle_id
    fleet.emergency_stop(v_id)
    return {"status": "estop_activated", "vehicle_id": req.vehicle_id}


@app.post("/api/approve")
async def post_approval(req: ApprovalRequest):
    """Worker approval endpoint for loading/unloading departure gates."""
    success = fleet.approve_worker_action(req.vehicle_id, req.action, req.dock_id)
    if success:
        return {"status": "approved", "vehicle_id": req.vehicle_id, "action": req.action}
    return JSONResponse(
        status_code=400,
        content={
            "status": "rejected",
            "vehicle_id": req.vehicle_id,
            "action": req.action,
            "message": f"Vehicle {req.vehicle_id} is not currently waiting for {req.action} approval.",
        },
    )


@app.post("/api/faults")
async def post_faults(req: FaultsRequest):
    """Inject sensor faults for specified vehicle."""
    fleet.set_sensor_fault(req.vehicle_id, req.camera, req.lidar, req.radar)
    return {"status": "updated", "vehicle_id": req.vehicle_id, "faults": req.model_dump()}


@app.post("/api/select_vehicle")
async def post_select_vehicle(req: SelectVehicleRequest):
    """Switch primary tracked vehicle for detailed UI telemetry."""
    if req.vehicle_id in fleet.vehicles:
        fleet.selected_vehicle_id = req.vehicle_id
        fleet.log_event("select", f"Camera & HUD switched to {req.vehicle_id}", vehicle_id=req.vehicle_id)
        return {"status": "selected", "vehicle_id": req.vehicle_id}
    return JSONResponse(status_code=404, content={"status": "not_found", "vehicle_id": req.vehicle_id})


@app.post("/api/sim/control")
async def post_sim_control(req: SimControlRequest):
    """Control simulation playback (pause, resume, reset, speed)."""
    global fleet
    if req.action == "pause":
        fleet.is_paused = True
        fleet.log_event("sim", "Simulation paused")
    elif req.action == "resume" or req.action == "play":
        fleet.is_paused = False
        fleet.log_event("sim", "Simulation resumed")
    elif req.action == "reset":
        fleet = FleetManager()
        fleet.log_event("sim", "Simulation scenario reset to nominal initial conditions")
    elif req.action == "speed":
        fleet.sim_speed = max(0.1, min(10.0, req.speed))
        fleet.log_event("sim", f"Simulation speed set to {fleet.sim_speed}x")

    return {
        "status": "ok",
        "action": req.action,
        "paused": fleet.is_paused,
        "speed": fleet.sim_speed,
    }


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time 20 Hz WebSocket telemetry streaming for all vehicles and campus."""
    await websocket.accept()
    logger.info("WebSocket client connected to AURA fleet telemetry stream.")
    try:
        while True:
            telemetry = fleet.step()
            await websocket.send_text(json.dumps(telemetry))
            await asyncio.sleep(0.05)  # 20 Hz tick
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


def main() -> None:
    import uvicorn

    print("=================================================================")
    print("  AURA | AUTONOMOUS INDUSTRIAL VEHICLE SYSTEM DASHBOARD")
    print("  Multi-Vehicle Logistics Campus Digital Twin")
    print("  Command Center running at: http://localhost:8000")
    print("=================================================================")
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
