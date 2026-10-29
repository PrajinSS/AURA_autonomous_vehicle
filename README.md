# AURA: Adaptive Uncertainty-aware Resilient Autonomy
**Autonomous Industrial Vehicle Decision, Perception & Multi-Vehicle Simulation System**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-All%20Acceptance%20Tests%20Passing-brightgreen.svg)]()
[![Platform: Win/Linux](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()

---

## 1. Project Overview

**AURA** is an industrial-grade autonomous mobile robot (AMR) and logistics fleet software stack designed for smart factories, warehouse campuses, and distribution centers.

It executes a continuous closed-loop resilient autonomy cycle:
```
[Sensors] ➔ [Sensor Health Monitor] ➔ [Uncertainty-Aware Multimodal Fusion] ➔ [EKF Object Tracking]
                                                                                ⬇
[Safe Vehicle Action] ⬅ [Adaptive Planner] ⬅ [Multi-Factor Risk Assessment] ⬅ [Trajectory Prediction]
```

### Core Capabilities
1. **Interactive 3D Logistics Campus**: Features **3 distinct warehouses** (Warehouse 1, Warehouse 2, Warehouse 3) with Docks A–F, arterial roads, crosswalks, shelving, obstacles, and an identifiable **Charging Yard** (Bays CH-01, CH-02, CH-03).
2. **Multi-Vehicle Fleet Coordination**: Simulates 3 concurrent AMRs (**AMR-01**, **AMR-02**, **AMR-03**) with independent A* path planning, bicycle kinematics, and mutual collision deconfliction/yielding.
3. **Authoritative Mission Lifecycle with Worker Approval Gates**:
   - `IDLE_AT_CHARGER` → `TRAVELLING_TO_PICKUP` → `DOCKING_AT_PICKUP`
   - `LOADING` (progresses over 5 simulation seconds)
   - `WAITING_FOR_LOAD_APPROVAL` (requires explicit human worker departure approval)
   - `TRAVELLING_TO_DELIVERY` → `DOCKING_AT_DELIVERY`
   - `UNLOADING` (progresses over 5 simulation seconds)
   - `WAITING_FOR_UNLOAD_APPROVAL` (requires worker delivery confirmation)
   - `RETURNING_TO_CHARGER` → `CHARGING` (+0.3% SoC/sec)
4. **All 5 View Modes Fully Functional**:
   - **3D**: Interactive perspective scene with third-person follow-camera or free campus orbit.
   - **TOPO**: 2D orthographic top-down engineering site map showing routes, docks, and vehicles.
   - **LAYER**: Interactive layer panel to toggle buildings, transit lanes, obstacles, collision footprints, sensor coverage cones, and trajectory paths.
   - **HEATMAP**: Live 2D spatial traffic density heatmap computed from real occupancy history with color gradient and legend.
   - **TRAJECTORY**: Planned path (cyan), executed trail (purple), avoidance detour route (coral), and waypoint arrival ETAs.
5. **Interactive Controls & Keyboard Shortcuts**:
   - Vehicle selector tabs & click-to-select AMR raycasting in 3D.
   - Simulation playback toolbar (Play/Pause, Reset, Speed: 1x/2x/5x, Camera Follow/Orbit).
   - Structured mission form (pickup warehouse, delivery warehouse, payload 1–4 pallets).
   - Natural language directive parser with strict validation (no silent destination inventing).
   - Sensor fault injection drawer for Camera, LiDAR, and Radar.

---

## 2. Directory Structure

```
prajin_autonomous_vehicle/
├── config.py                     # Central configuration & parameters
├── main.py                       # CLI execution entry point
├── requirements.txt              # Production Python dependencies
├── dashboard/                    # Web Command Center & Telemetry Stream
│   ├── server.py                 # FastAPI & 20Hz WebSocket streaming server
│   └── web/                      # Pure Web Stack (Vanilla HTML5, CSS3, JavaScript, Three.js)
│       ├── index.html            # Command Center UI layout
│       ├── css/style.css         # Dark-mode glassmorphic design system
│       └── js/
│           ├── app.js            # Frontend telemetry, mission dispatch, approvals & shortcuts
│           └── scene3d.js        # Three.js 3D campus renderer, multi-AMR models & view modes
├── simulation/                   # Campus simulation & vehicle physics
│   ├── factory_world.py          # 3 Warehouses, Docks A-F, Charging Yard, Workers & Obstacles
│   ├── amr_vehicle.py            # AMR bicycle kinematics, battery SoC, motor & brake health
│   ├── fleet_manager.py          # Multi-vehicle fleet manager, spatial deconfliction & heatmap
│   ├── navigator.py              # Goal-directed autonomous navigator & steering controller
│   └── obstacles.py              # Geometric collision shapes and motion models
├── mission/                      # Mission command & lifecycle
│   ├── parser.py                 # Structured & natural language parser with strict validation
│   ├── state_machine.py          # Authoritative mission lifecycle with 5s dwell & worker approvals
│   └── task_queue.py             # Priority mission task queue
├── perception/                   # Sensor health & multimodal fusion
│   ├── fusion.py                 # Uncertainty-aware multimodal fusion engine
│   └── tracker.py                # Multi-target Kalman tracking & data association
├── prediction/                   # Trajectory prediction
│   └── trajectory_predictor.py   # Constant-velocity & goal-directed trajectory predictor
├── risk/                         # Multi-factor risk assessor
│   └── risk_assessor.py          # Analytical TTC, confidence, motor/brake wear, and explanations
├── planning/                     # Global & local path planners
│   ├── global_planner.py         # A* global grid pathfinder
│   ├── local_planner.py          # Quintic spline trajectory smoother
│   └── adaptive_controller.py    # Closed-loop evasion & detour controller
├── datasets/                     # nuScenes Mini & KITTI ingestion converters
├── experiments/                  # Benchmark evaluation suite & metrics collector
└── tests/                        # Comprehensive automated test suite
    ├── test_campus_simulation.py # Scenarios A through H acceptance tests
    ├── test_dashboard_server.py  # Server REST API & static assets tests
    ├── test_mission.py           # Mission state machine & parser tests
    └── ...                       # Unit tests for all modules
```

---

## 3. Installation & Run Commands

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14 on Windows and Linux)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### Installation
```powershell
# Clone repository
cd prajin_autonomous_vehicle

# Install dependencies
pip install -r requirements.txt
```

### Running the Live Simulation Dashboard
```powershell
python -m dashboard.server
```
Then navigate to: **`http://localhost:8000`**

### Running Automated Test Suites
```powershell
# Run all multi-vehicle campus simulation acceptance tests (Scenarios A through H)
python -m pytest tests/test_campus_simulation.py

# Run dashboard server and mission unit tests
python -m pytest tests/test_dashboard_server.py tests/test_mission.py
```

---

## 4. User Controls & Interaction

### Keyboard Shortcuts & Camera Controls
| Key / Gesture | Action |
| :--- | :--- |
| **Left-Click Drag** | **Orbit** camera view around current look-at target |
| **Right-Click Drag** | **Pan** camera view horizontally across the campus |
| **Mouse Wheel / Pinch** | **Zoom** camera view in and out smoothly with ground limit protection |
| **`1` / `2` / `3`** | Select and track **AMR-01**, **AMR-02**, or **AMR-03** |
| **`C`** | Toggle **Camera Mode** (Third-Person Follow / Free Orbit Overview) |
| **`F`** / `🎯 Recenter` | **Recenter / Focus** camera immediately on the selected AMR |
| **`V`** / `📐 Reset View` | **Reset** camera perspective to default campus overview angle |
| **`Space`** | **Pause / Resume** simulation |
| **`R`** | **Reset** simulation to initial conditions |
| **`I`** | Toggle **Documentation Guide** modal |

*Note on Follow Camera*: In follow mode, you can freely orbit and zoom around the moving vehicle; the camera preserves your custom viewing angle relative to the vehicle's motion instead of resetting every frame.

### Viewport View Modes & Debug Layers
- **3D**: Perspective 3D campus view with vehicle tracking or overview orbit.
- **TOPO**: Orthographic 2D top-down engineering site map showing warehouses, chargers, obstacles, and routes.
- **LAYER**: Interactive layer panel with toggles for Buildings, Transit Lanes, Shelving Obstacles, Collision Footprints, Sensor Cones, Trajectory Paths, Nav Roads & Spurs, and Wall Colliders.
- **HEATMAP**: Live 2D spatial heatmap texture projected onto the campus floor showing cumulative traffic density.
- **TRAJECTORY**: Path planning visualization with planned paths (cyan), executed trail (purple), detour alternative routes (coral), and arrival ETA pins.

### Worker Approval Workflow
1. When an AMR reaches its pickup dock, it halts and shows loading progress for **5 simulation seconds**.
2. Upon reaching 100% loading progress, the vehicle enters `WAITING_FOR_LOAD_APPROVAL`.
3. An amber approval banner lights up on the dashboard:
   `[AMR-01 at Warehouse 1 Dock A: Loading Complete (5.0s). Explicit worker approval required to depart.]`
4. The user clicks **`[✅ Approve Departure]`** (or calls `POST /api/approve`).
5. The AMR departs safely through the **same** designated dock access spur and entrance onto the main arterial road.
6. The same approval gate applies at the delivery dock prior to returning to charger.

---

## 5. Acceptance Test Checklist (Scenarios A through I)

| Scenario | Description | Result | Verification Details |
| :---: | :--- | :---: | :--- |
| **A** | **Every Charging Bay ➔ Every Dock ➔ Charger** | **PASSED** | Valid bidirectional road paths verified for all 3 chargers (AMR-01, AMR-02, AMR-03) to all 6 docks (Docks A–F) and back. Full lifecycle verified with 5s dwell and worker approvals. |
| **B** | **Dock 1 ➔ Dock 2 & Dock 2 ➔ Dock 1** | **PASSED** | Bidirectional cross-warehouse routing verified (Dock A ➔ Dock C and Dock C ➔ Dock A) utilizing arterial highway and designated access avenues without wall penetration. |
| **C** | **Straight Road Alignment & Kinematics** | **PASSED** | Vehicle maintains straight travel along road axis ($Y=0.0$) with zero sideways sliding or wobbling; visual forks align with velocity heading ($\Delta < 0.05\text{ rad}$); cornering speed governor active. |
| **D** | **Dock Entry/Exit via Approved Spur (No Wall Crossing)** | **PASSED** | Swept footprint validation ($\text{radius} = 1.25\text{ m}$) confirms vehicles enter and depart through designated $4.0\text{ m}$ doorway openings. Zero wall collider penetrations. |
| **E** | **Static Obstacle & Replanning** | **PASSED** | Vehicle detects static pallet obstruction on road, computes collision-free detour, or halts safely if completely blocked. Never leaves road boundary or tunnels through buildings. |
| **F** | **Worker Crossing Ahead & Yielding** | **PASSED** | Dynamic pedestrian crossing corridor triggers multi-factor risk assessor, reducing speed to $0.0\text{ m/s}$ ($4.5\text{ m}$ safety cushion) until worker clears. |
| **G** | **Two Vehicles Competing for Narrow Access Spur** | **PASSED** | Multi-vehicle spatial headway deconfliction enforces single-occupancy yielding on access spurs and intersections; no deadlocks or collisions. |
| **H** | **Free Camera Orbit/Pan/Zoom Exploration** | **PASSED** | Orbit, pan, zoom, recenter, and reset view operational during idle, travel, loading, and charging. User viewing angles preserved in follow mode. |
| **I** | **Pause/Resume & Frame-Rate Independence** | **PASSED** | Fixed $\Delta t = 0.05\text{ s}$ kinematics integration ensures identical, deterministic trajectories under pause/resume cycles and varying render frame rates. |

---

## 6. Open-Source Dependencies & Asset Licenses

All assets and libraries used in AURA are 100% free and open-source:
- **Three.js** (v0.128.0): MIT License ([GitHub](https://github.com/mrdoob/three.js/blob/dev/LICENSE))
- **Google Fonts** (Inter, Outfit, JetBrains Mono): SIL Open Font License 1.1
- **FastAPI & Uvicorn**: MIT / BSD-3-Clause Licenses
- **DiceBear Avatars**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Geometry & Graphics**: Procedurally generated 3D primitives (no commercial 3D asset packs or proprietary subscriptions required).

---

## 7. Disclosures & Known Limitations

- **Simulated Sensing**: When CARLA is offline, sensor frames are generated via the integrated high-fidelity digital twin with noise, latency, and simulated degradation models.
- **Physics Engine**: Vehicle motion uses non-linear bicycle kinematics, dynamic mass scaling, motor/brake wear physics, and battery electro-chemistry rather than an external continuous rigid-body game engine.
