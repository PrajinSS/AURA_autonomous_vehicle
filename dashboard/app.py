"""Streamlit monitoring dashboard for AURA (Phase 10 alternative/companion).

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st

from mission.parser import CommandParser
from mission.state_machine import MissionStateMachine
from perception.fusion import UncertaintyAwareFusion
from perception.tracker import KalmanTracker
from prediction.trajectory_predictor import TrajectoryPredictor
from risk.risk_assessor import RiskAssessor
from simulation.carla_bridge.client import CarlaSimulationBridge

st.set_page_config(
    page_title="AURA | Autonomous Industrial Vehicle System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .reportview-container { background: #060c18; }
    .stMetric { background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 10px; }
    .action-badge { background: rgba(244, 63, 94, 0.15); border: 1px solid #f43f5e; color: #f43f5e; padding: 12px; border-radius: 6px; font-weight: bold; font-size: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚚 AURA | Autonomous Industrial Vehicle System")
st.caption("Adaptive Uncertainty-aware Resilient Autonomy — Factory Mission Command Center")

# Sidebar Controls
st.sidebar.header("Mission Command")
cmd_text = st.sidebar.text_input(
    "Mission Command", "Pick up pallet at Dock A and deliver to Dock B"
)
abort_btn = st.sidebar.button("🚨 Emergency Abort")

st.sidebar.header("Sensor Degradation Controls")
cam_fault = st.sidebar.checkbox("Camera Lens Flare / Dropout", False)
lidar_fault = st.sidebar.checkbox("LiDAR Fog / Occlusion", False)
radar_fault = st.sidebar.checkbox("Radar Multipath Noise", False)

if "bridge" not in st.session_state:
    st.session_state.bridge = CarlaSimulationBridge()
    st.session_state.parser = CommandParser()
    st.session_state.mission_sm = MissionStateMachine()
    st.session_state.fusion = UncertaintyAwareFusion()
    st.session_state.tracker = KalmanTracker()
    st.session_state.predictor = TrajectoryPredictor()
    st.session_state.risk = RiskAssessor()

    init_task = st.session_state.parser.parse(cmd_text)
    st.session_state.mission_sm.set_task(init_task, (-15.0, 0.0))

bridge = st.session_state.bridge
bridge.set_sensor_fault("camera", cam_fault)
bridge.set_sensor_fault("lidar", lidar_fault)
bridge.set_sensor_fault("radar", radar_fault)

if abort_btn:
    st.session_state.mission_sm.abort()

# Step simulation
state, frame = bridge.step(accel_cmd=0.8, steer_cmd=0.04)
progress = st.session_state.mission_sm.update((state.x, state.y), speed=state.speed)
fused = st.session_state.fusion.fuse(frame)
tracks = st.session_state.tracker.update(fused)
preds = st.session_state.predictor.predict(tracks)
report = st.session_state.risk.assess(state, tracks, preds, frame.sensor_confidence)

# Layout Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Battery SoC", f"{state.battery_soc_pct}%", f"{state.runtime_remaining_hours}h left")
with col2:
    st.metric("Camera Reliability", f"{frame.sensor_confidence.get('camera', 1.0):.0%}")
with col3:
    st.metric("LiDAR Reliability", f"{frame.sensor_confidence.get('lidar', 1.0):.0%}")
with col4:
    st.metric("Radar Reliability", f"{frame.sensor_confidence.get('radar', 1.0):.0%}")

# Main Panels
main_left, main_right = st.columns([3, 2])

with main_left:
    st.subheader("Mission Progress")
    st.progress(int(progress.percent_complete))
    st.caption(f"Status: {progress.subtext} | Remaining: {progress.estimated_remaining_time_str}")

    st.subheader("Factory Map View")
    # Quick 2D scatter of vehicle and obstacles
    df_pts = pd.DataFrame(
        [{"x": state.x, "y": state.y, "entity": "Ego AMR"}]
        + [{"x": t.x + state.x, "y": t.y + state.y, "entity": t.track_id} for t in tracks]
    )
    st.scatter_chart(df_pts, x="x", y="y", color="entity")

with main_right:
    st.subheader("AI Decision & Risk")
    st.markdown(
        f"""<div class="action-badge">
        ACTION: {report.recommended_action} ➔
    </div>""",
        unsafe_allow_html=True,
    )
    st.write(f"**Reason**: {report.decision_reason}")
    st.write(f"**Explanation**: {report.explanation}")

    st.subheader("Predicted Collision Risk Table")
    risk_data = [
        {"Object": r.object_id, "TTC (sec)": r.ttc_sec, "Risk Level": r.risk_level.value}
        for r in report.obstacle_risks
    ]
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True)

st.info("System Online • Live Digital Twin Active")
