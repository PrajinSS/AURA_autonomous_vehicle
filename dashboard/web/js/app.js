/**
 * AURA Dashboard Client Application
 *
 * Coordinates:
 * - Multi-vehicle fleet tracking & selection (AMR-01, AMR-02, AMR-03)
 * - 3D Logistics campus digital twin viewport (Scene3D)
 * - Structured mission dispatch & NLP directives with error reporting
 * - Real worker approval gates (5s dwell loading & unloading)
 * - Simulation playback toolbar (Play/Pause, Reset, Speed, Camera)
 * - 5 functional view modes (3D, TOPO, LAYER, HEATMAP, TRAJECTORY)
 * - Layer control toggles
 * - Live sensor reliability, vehicle health, risk, and event audit log
 * - Info documentation modal dialog & keyboard shortcuts
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── Initialize 3D Viewport ──────────────────────────────────────────
  const scene3D = new FactoryScene3D(
    'factory-3d-canvas',
    'hud-overlay-container',
    'mini-map-canvas'
  );

  let currentSelectedVehicle = 'AMR-01';
  let isSimPaused = false;
  let alertCount = 0;

  // ── DOM References ──────────────────────────────────────────────────
  // Header & Fleet
  const fleetTabAMR01 = document.getElementById('tab-amr-01');
  const fleetTabAMR02 = document.getElementById('tab-amr-02');
  const fleetTabAMR03 = document.getElementById('tab-amr-03');
  const fleetSocAMR01 = document.getElementById('fleet-soc-amr-01');
  const fleetSocAMR02 = document.getElementById('fleet-soc-amr-02');
  const fleetSocAMR03 = document.getElementById('fleet-soc-amr-03');
  const trackedVehicleBadge = document.getElementById('tracked-vehicle-badge');
  const activeVehicleLabel = document.getElementById('active-vehicle-label');

  // Simulation Controls
  const simPlayPauseBtn = document.getElementById('sim-play-pause-btn');
  const playPauseIcon = document.getElementById('play-pause-icon');
  const simResetBtn = document.getElementById('sim-reset-btn');
  const cameraToggleBtn = document.getElementById('camera-toggle-btn');
  const recenterCamBtn = document.getElementById('recenter-cam-btn');
  const resetCamBtn = document.getElementById('reset-cam-btn');
  const speedButtons = document.querySelectorAll('.speed-btn');

  // Mission & Form Elements
  const missionTextDisplay = document.getElementById('mission-text-display');
  const missionSubtextDisplay = document.getElementById('mission-subtext-display');
  const missionProgressBar = document.getElementById('mission-progress-bar');
  const missionProgressPct = document.getElementById('mission-progress-pct');
  const missionEtaDisplay = document.getElementById('mission-eta-display');
  const dwellProgressContainer = document.getElementById('dwell-progress-container');
  const dwellLabelText = document.getElementById('dwell-label-text');
  const dwellPctText = document.getElementById('dwell-pct-text');
  const dwellProgressBar = document.getElementById('dwell-progress-bar');

  // Structured Mission Form
  const formVehicleSelect = document.getElementById('form-vehicle-select');
  const formQuantityInput = document.getElementById('form-quantity-input');
  const formPickupSelect = document.getElementById('form-pickup-select');
  const formDeliverySelect = document.getElementById('form-delivery-select');
  const formDispatchBtn = document.getElementById('form-dispatch-btn');
  const formEstopBtn = document.getElementById('form-estop-btn');
  const abortMissionBtn = document.getElementById('abort-mission-btn');
  const customCommandInput = document.getElementById('custom-command-input');
  const sendCommandBtn = document.getElementById('send-command-btn');
  const commandErrorMsg = document.getElementById('command-error-msg');

  // Worker Approval Prompt Banner
  const workerApprovalCard = document.getElementById('worker-approval-card');
  const approvalTitleText = document.getElementById('approval-title-text');
  const approvalMessageText = document.getElementById('approval-message-text');
  const approveActionBtn = document.getElementById('approve-action-btn');

  // View Mode & Layer Overlays
  const trajectoryLegend = document.getElementById('trajectory-legend');
  const heatmapLegend = document.getElementById('heatmap-legend');
  const layerControlPanel = document.getElementById('layer-control-panel');

  // Metrics & Sensors
  const camVal = document.getElementById('cam-val');
  const camBar = document.getElementById('cam-bar');
  const lidarVal = document.getElementById('lidar-val');
  const lidarBar = document.getElementById('lidar-bar');
  const radarVal = document.getElementById('radar-val');
  const radarBar = document.getElementById('radar-bar');

  const vhStatusBadge = document.getElementById('vh-status-badge');
  const vhBatteryVal = document.getElementById('vh-battery-val');
  const batteryBar = document.getElementById('battery-bar');
  const vhRuntimeVal = document.getElementById('vh-runtime-val');

  const navStatusText = document.getElementById('nav-status-text');
  const navAccelText = document.getElementById('nav-accel-text');
  const navSteerText = document.getElementById('nav-steer-text');
  const navCargoText = document.getElementById('nav-cargo-text');

  const riskTableBody = document.getElementById('risk-table-body');
  const actionBanner = document.getElementById('action-banner');
  const actionTitleText = document.getElementById('action-title-text');
  const actionReasonText = document.getElementById('action-reason-text');
  const explanationBodyText = document.getElementById('explanation-body-text');
  const aiConfidenceVal = document.getElementById('ai-confidence-val');
  const eventLogContainer = document.getElementById('event-log-container');
  const alertBadge = document.getElementById('alert-badge');

  // Drawer & Modal
  const scenarioDrawer = document.getElementById('scenario-drawer');
  const scenarioPanelToggle = document.getElementById('scenario-panel-toggle');
  const closeDrawerBtn = document.getElementById('close-drawer-btn');
  const toggleCamFault = document.getElementById('toggle-cam-fault');
  const toggleLidarFault = document.getElementById('toggle-lidar-fault');
  const toggleRadarFault = document.getElementById('toggle-radar-fault');
  const faultTargetVehicleSelect = document.getElementById('fault-target-vehicle-select');

  const infoModalBackdrop = document.getElementById('info-modal-backdrop');
  const mainInfoBtn = document.getElementById('main-info-btn');
  const closeInfoModalBtn = document.getElementById('close-info-modal-btn');

  // ── Fleet Vehicle Selection ─────────────────────────────────────────
  function selectVehicle(vId) {
    currentSelectedVehicle = vId;
    scene3D.setSelectedVehicle(vId);

    // Update fleet tabs
    document.querySelectorAll('.fleet-tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-vehicle') === vId);
    });

    if (trackedVehicleBadge) trackedVehicleBadge.innerText = `${vId} TRACKED`;
    if (activeVehicleLabel) activeVehicleLabel.innerText = vId;
    if (faultTargetVehicleSelect) faultTargetVehicleSelect.value = vId;

    // Send selection to server
    fetch('/api/select_vehicle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vehicle_id: vId }),
    }).catch(err => console.error('Select vehicle error:', err));
  }

  document.querySelectorAll('.fleet-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const vId = btn.getAttribute('data-vehicle');
      selectVehicle(vId);
    });
  });

  // Connect 3D click raycasting to selection
  scene3D.onVehicleSelectCallback = (vId) => {
    selectVehicle(vId);
  };

  // ── Simulation Playback Controls ────────────────────────────────────
  simPlayPauseBtn.addEventListener('click', () => {
    isSimPaused = !isSimPaused;
    const action = isSimPaused ? 'pause' : 'resume';
    fetch('/api/sim/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    })
      .then(r => r.json())
      .then(res => {
        playPauseIcon.innerText = res.paused ? '▶ Resume' : '⏸ Pause';
        simPlayPauseBtn.classList.toggle('paused', res.paused);
      });
  });

  simResetBtn.addEventListener('click', () => {
    fetch('/api/sim/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'reset' }),
    }).then(() => {
      playPauseIcon.innerText = '⏸ Pause';
      simPlayPauseBtn.classList.remove('paused');
      isSimPaused = false;
      hideError();
    });
  });

  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const speed = parseFloat(btn.getAttribute('data-speed'));
      fetch('/api/sim/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'speed', speed }),
      });
    });
  });

  cameraToggleBtn.addEventListener('click', () => {
    const isFollow = scene3D.toggleFollowCamera();
    cameraToggleBtn.innerText = isFollow ? '🎥 Follow: ON' : '🌐 Follow: OFF (Orbit)';
    cameraToggleBtn.classList.toggle('orbit', !isFollow);
  });

  if (recenterCamBtn) {
    recenterCamBtn.addEventListener('click', () => {
      scene3D.recenterCamera();
    });
  }

  if (resetCamBtn) {
    resetCamBtn.addEventListener('click', () => {
      scene3D.resetView();
    });
  }

  // ── View Mode Switching ─────────────────────────────────────────────
  document.querySelectorAll('.tab-btn[data-mode]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn[data-mode]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.getAttribute('data-mode');
      scene3D.setViewMode(mode);

      // Overlays visibility
      if (trajectoryLegend) trajectoryLegend.style.display = (mode === 'trajectory' || mode === '3d') ? 'block' : 'none';
      if (heatmapLegend) heatmapLegend.style.display = (mode === 'heatmap') ? 'block' : 'none';
      if (layerControlPanel) layerControlPanel.style.display = (mode === 'layer') ? 'block' : 'none';
    });
  });

  // Layer Toggles
  const layerBindings = [
    { id: 'layer-buildings', name: 'buildings' },
    { id: 'layer-lanes', name: 'lanes' },
    { id: 'layer-obstacles', name: 'obstacles' },
    { id: 'layer-collision', name: 'collision' },
    { id: 'layer-sensors', name: 'sensors' },
    { id: 'layer-trajectories', name: 'trajectories' },
    { id: 'layer-navmesh', name: 'navmesh' },
    { id: 'layer-colliders', name: 'colliders' },
  ];
  layerBindings.forEach(binding => {
    const el = document.getElementById(binding.id);
    if (el) {
      el.addEventListener('change', () => {
        scene3D.setLayerVisibility(binding.name, el.checked);
      });
    }
  });

  // ── Structured Mission Dispatch ─────────────────────────────────────
  function showError(msg) {
    if (commandErrorMsg) {
      commandErrorMsg.innerText = `⚠️ ${msg}`;
      commandErrorMsg.style.display = 'block';
      setTimeout(() => hideError(), 6000);
    }
  }

  function hideError() {
    if (commandErrorMsg) commandErrorMsg.style.display = 'none';
  }

  formDispatchBtn.addEventListener('click', () => {
    hideError();
    const payload = {
      vehicle_id: formVehicleSelect.value,
      quantity: parseInt(formQuantityInput.value) || 1,
      pickup_warehouse: formPickupSelect.value,
      delivery_warehouse: formDeliverySelect.value,
    };

    fetch('/api/mission/dispatch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(async r => {
        const res = await r.json();
        if (!r.ok) {
          showError(res.error || res.message || 'Mission rejected');
          return;
        }
        missionTextDisplay.innerText = `"${res.display_text}"`;
        if (res.vehicle_id) selectVehicle(res.vehicle_id);
        incrementAlerts();
      })
      .catch(err => showError(err.message));
  });

  // Natural Language Command Dispatch
  function sendCommand(text) {
    if (!text.trim()) return;
    hideError();
    fetch('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: text }),
    })
      .then(async r => {
        const res = await r.json();
        if (!r.ok) {
          showError(res.error || 'Invalid directive');
          return;
        }
        missionTextDisplay.innerText = `"${res.display_text || text}"`;
        customCommandInput.value = '';
        if (res.vehicle_id) selectVehicle(res.vehicle_id);
        incrementAlerts();
      })
      .catch(err => showError(err.message));
  }

  sendCommandBtn.addEventListener('click', () => sendCommand(customCommandInput.value));
  customCommandInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendCommand(customCommandInput.value);
  });

  // Abort & E-Stop
  abortMissionBtn.addEventListener('click', () => {
    fetch('/api/mission/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vehicle_id: currentSelectedVehicle }),
    });
  });

  formEstopBtn.addEventListener('click', () => {
    fetch('/api/mission/estop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vehicle_id: currentSelectedVehicle }),
    });
  });

  // ── Worker Approval Action ──────────────────────────────────────────
  approveActionBtn.addEventListener('click', () => {
    const action = approveActionBtn.dataset.action || 'approve_load';
    const dockId = approveActionBtn.dataset.dockId || null;

    fetch('/api/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vehicle_id: currentSelectedVehicle,
        action: action,
        dock_id: dockId,
      }),
    })
      .then(r => r.json())
      .then(res => {
        if (res.status === 'approved') {
          workerApprovalCard.style.display = 'none';
          incrementAlerts();
        }
      });
  });

  // ── Sensor Fault Drawer ─────────────────────────────────────────────
  scenarioPanelToggle.addEventListener('click', () => scenarioDrawer.classList.toggle('open'));
  closeDrawerBtn.addEventListener('click', () => scenarioDrawer.classList.remove('open'));

  function updateFaults() {
    const targetVeh = faultTargetVehicleSelect ? faultTargetVehicleSelect.value : currentSelectedVehicle;
    fetch('/api/faults', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vehicle_id: targetVeh,
        camera: toggleCamFault.checked,
        lidar: toggleLidarFault.checked,
        radar: toggleRadarFault.checked,
      }),
    });
  }

  toggleCamFault.addEventListener('change', updateFaults);
  toggleLidarFault.addEventListener('change', updateFaults);
  toggleRadarFault.addEventListener('change', updateFaults);

  // Preset Buttons
  document.getElementById('preset-w1-to-w2').addEventListener('click', () => {
    sendCommand('Take 1 pallet from Warehouse 1 to Warehouse 2');
  });
  document.getElementById('preset-w2-to-w3').addEventListener('click', () => {
    sendCommand('Take 2 pallets from Warehouse 2 to Warehouse 3');
  });
  document.getElementById('preset-charge-all').addEventListener('click', () => {
    sendCommand('Return to charging station');
  });
  document.getElementById('preset-clear-faults').addEventListener('click', () => {
    toggleCamFault.checked = false;
    toggleLidarFault.checked = false;
    toggleRadarFault.checked = false;
    updateFaults();
  });

  // ── Documentation & Guide Modal ─────────────────────────────────────
  mainInfoBtn.addEventListener('click', () => {
    infoModalBackdrop.style.display = 'flex';
  });
  closeInfoModalBtn.addEventListener('click', () => {
    infoModalBackdrop.style.display = 'none';
  });
  infoModalBackdrop.addEventListener('click', (e) => {
    if (e.target === infoModalBackdrop) infoModalBackdrop.style.display = 'none';
  });

  // ── Keyboard Shortcuts ──────────────────────────────────────────────
  window.addEventListener('keydown', (e) => {
    if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;

    if (e.key === '1') selectVehicle('AMR-01');
    if (e.key === '2') selectVehicle('AMR-02');
    if (e.key === '3') selectVehicle('AMR-03');
    if (e.code === 'Space') {
      e.preventDefault();
      simPlayPauseBtn.click();
    }
    if (e.key.toLowerCase() === 'c') cameraToggleBtn.click();
    if (e.key.toLowerCase() === 'f' && recenterCamBtn) recenterCamBtn.click();
    if (e.key.toLowerCase() === 'v' && resetCamBtn) resetCamBtn.click();
    if (e.key.toLowerCase() === 'r') simResetBtn.click();
    if (e.key.toLowerCase() === 'i') {
      infoModalBackdrop.style.display = (infoModalBackdrop.style.display === 'none') ? 'flex' : 'none';
    }
  });

  function incrementAlerts() {
    alertCount++;
    if (alertBadge) alertBadge.innerText = alertCount;
  }

  // ── WebSocket Telemetry Synchronization ─────────────────────────────
  let ws = null;

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      document.getElementById('system-status-text').innerText = 'System Online';
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateUI(data);
        scene3D.updateTelemetry(data);
      } catch (err) {
        console.error('Failed to parse telemetry packet:', err);
      }
    };

    ws.onclose = () => {
      document.getElementById('system-status-text').innerText = 'Reconnecting...';
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      ws.close();
    };
  }

  // ── UI Synchronization from Authoritative Telemetry ─────────────────
  function updateUI(data) {
    if (!data) return;

    // 1. Update Fleet Tabs (SoC % and status dot)
    if (data.fleet) {
      data.fleet.forEach(v => {
        if (v.vehicle_id === 'AMR-01' && fleetSocAMR01) fleetSocAMR01.innerText = `🔋 ${v.battery_soc_pct}%`;
        if (v.vehicle_id === 'AMR-02' && fleetSocAMR02) fleetSocAMR02.innerText = `🔋 ${v.battery_soc_pct}%`;
        if (v.vehicle_id === 'AMR-03' && fleetSocAMR03) fleetSocAMR03.innerText = `🔋 ${v.battery_soc_pct}%`;
      });
    }

    // 2. Mission Telemetry
    if (data.mission) {
      const m = data.mission;
      if (m.active_task && m.active_task.display_text) {
        missionTextDisplay.innerText = `"${m.active_task.display_text}"`;
      }
      missionSubtextDisplay.innerText = m.subtext || 'IN PROGRESS';

      const pct = Math.round(m.percent_complete || 0);
      missionProgressBar.style.width = `${pct}%`;
      missionProgressPct.innerText = `${pct}% COMPLETE`;
      missionEtaDisplay.innerText = `EST. REMAINING: ${m.estimated_remaining_time_str || '--:--'}`;

      // Loading / Unloading Dwell Bar (5 seconds)
      const isDwelling = (m.status === 'LOADING' || m.status === 'PICKING' || m.status === 'UNLOADING');
      if (isDwelling && dwellProgressContainer) {
        dwellProgressContainer.style.display = 'block';
        const dwellLabel = (m.status === 'UNLOADING') ? 'UNLOADING (5s):' : 'LOADING (5s):';
        dwellLabelText.innerText = dwellLabel;
        const dPct = Math.round(m.dwell_progress_pct || 0);
        dwellPctText.innerText = `${dPct}%`;
        dwellProgressBar.style.width = `${dPct}%`;
      } else if (dwellProgressContainer) {
        dwellProgressContainer.style.display = 'none';
      }

      // Worker Approval Gate Card
      if (m.awaiting_approval && workerApprovalCard) {
        workerApprovalCard.style.display = 'block';
        const actionType = m.approval_type === 'load' ? 'approve_load' : 'approve_unload';
        approveActionBtn.dataset.action = actionType;
        approveActionBtn.dataset.dockId = m.dock_name;

        if (m.approval_type === 'load') {
          approvalTitleText.innerText = 'WORKER LOADING DEPARTURE APPROVAL REQUIRED';
          approvalMessageText.innerText = `${data.selected_vehicle_id} at ${m.dock_name} has completed loading (5.0s). Explicit worker approval required to depart.`;
          approveActionBtn.innerText = '✅ Approve Loading Departure';
        } else {
          approvalTitleText.innerText = 'WORKER UNLOADING COMPLETION APPROVAL REQUIRED';
          approvalMessageText.innerText = `${data.selected_vehicle_id} at ${m.dock_name} has completed delivery. Explicit worker approval required to return to charger.`;
          approveActionBtn.innerText = '✅ Approve Delivery & Release to Charger';
        }
      } else if (workerApprovalCard) {
        workerApprovalCard.style.display = 'none';
      }
    }

    // 3. Sensor Reliability (Fusion confidences)
    if (data.sensors) {
      const c = Math.round(data.sensors.camera * 100);
      const l = Math.round(data.sensors.lidar * 100);
      const r = Math.round(data.sensors.radar * 100);

      camVal.innerText = `${c}%`;
      camBar.style.width = `${c}%`;
      camBar.className = `mini-fill ${c > 70 ? 'cyan' : c > 35 ? 'amber' : 'red'}`;

      lidarVal.innerText = `${l}%`;
      lidarBar.style.width = `${l}%`;
      lidarBar.className = `mini-fill ${l > 70 ? 'green' : l > 35 ? 'amber' : 'red'}`;

      radarVal.innerText = `${r}%`;
      radarBar.style.width = `${r}%`;
      radarBar.className = `mini-fill ${r > 70 ? 'amber' : r > 35 ? 'amber' : 'red'}`;
    }

    // 4. Vehicle Health & Battery
    if (data.vehicle) {
      const v = data.vehicle;
      vhStatusBadge.innerText = v.status;
      vhStatusBadge.className = `stat-badge ${v.status === 'OPERATIONAL' ? 'green' : v.status === 'CHARGING' ? 'cyan' : 'amber'}`;

      const soc = Math.round(v.battery_soc_pct);
      vhBatteryVal.innerText = `🔋 ${soc}%`;
      batteryBar.style.width = `${soc}%`;
      batteryBar.className = `mini-fill ${soc > 30 ? 'green' : soc > 15 ? 'amber' : 'red'}`;

      const rtH = Math.floor(v.runtime_remaining_hours);
      const rtM = Math.round((v.runtime_remaining_hours - rtH) * 60);
      vhRuntimeVal.innerText = `${rtH}h ${rtM}m`;

      navCargoText.innerText = v.cargo_loaded ? '📦 PALLET LOADED' : 'EMPTY';
      navCargoText.style.color = v.cargo_loaded ? 'var(--accent-amber)' : 'var(--text-muted)';
    }

    // 5. Navigation Telemetry
    if (data.navigation) {
      const n = data.navigation;
      navStatusText.innerText = n.is_waiting ? 'WAITING' : n.is_rerouting ? 'REROUTING' : 'NAVIGATING';
      navStatusText.className = `nav-val ${n.is_rerouting ? 'rerouting' : ''}`;
      navAccelText.innerText = `${n.accel_cmd > 0 ? '+' : ''}${n.accel_cmd.toFixed(2)} m/s²`;
      const deg = (n.steer_cmd * 180 / Math.PI).toFixed(1);
      navSteerText.innerText = `${deg}°`;
    }

    // 6. Collision Risk Obstacle Table
    if (data.risk && data.risk.obstacle_risks) {
      let rows = '';
      data.risk.obstacle_risks.slice(0, 4).forEach(o => {
        const pClass = o.risk_level.toLowerCase();
        rows += `
          <tr>
            <td>${o.object_id}</td>
            <td>${o.ttc_sec.toFixed(1)}s</td>
            <td><span class="risk-pill ${pClass}">${o.risk_level}</span></td>
            <td>${o.distance_m.toFixed(1)}m</td>
          </tr>
        `;
      });
      if (rows) riskTableBody.innerHTML = rows;

      const riskNote = document.querySelector('.risk-warning-note');
      if (riskNote) {
        const highs = data.risk.obstacle_risks.filter(r => r.risk_level === 'HIGH').length;
        if (highs > 0) {
          riskNote.innerHTML = `<span class="red-bullet">•</span> ${highs} obstacle(s) high risk. TTC below critical threshold.`;
        } else if (data.risk.obstacle_risks.length > 0) {
          riskNote.innerHTML = `<span class="red-bullet" style="color: var(--accent-amber)">•</span> ${data.risk.obstacle_risks.length} dynamic obstacle(s) tracked.`;
        } else {
          riskNote.innerHTML = `<span class="red-bullet" style="color: var(--accent-green)">•</span> Transit corridor clear.`;
        }
      }
    }

    // 7. AI Decision & Explanation
    if (data.risk) {
      const r = data.risk;
      actionTitleText.innerText = r.recommended_action;
      actionReasonText.innerText = `Reason: ${r.decision_reason}`;
      explanationBodyText.innerText = r.explanation;
      aiConfidenceVal.innerText = `${r.confidence_pct}%`;

      // Color action banner
      if (actionBanner) {
        const act = r.recommended_action;
        if (act.includes('STOP') || act.includes('EMERGENCY')) {
          actionBanner.style.borderColor = 'rgba(239, 68, 68, 0.7)';
          actionBanner.style.background = 'rgba(239, 68, 68, 0.18)';
          actionTitleText.style.color = 'var(--accent-red)';
        } else if (act.includes('SLOW') || act.includes('REROUTE')) {
          actionBanner.style.borderColor = 'rgba(244, 63, 94, 0.5)';
          actionBanner.style.background = 'rgba(244, 63, 94, 0.15)';
          actionTitleText.style.color = 'var(--accent-coral)';
        } else {
          actionBanner.style.borderColor = 'rgba(16, 185, 129, 0.5)';
          actionBanner.style.background = 'rgba(16, 185, 129, 0.12)';
          actionTitleText.style.color = 'var(--accent-green)';
        }
      }
    }

    // 8. Event Audit Log
    if (data.events && eventLogContainer) {
      let eventsHtml = '';
      data.events.forEach(e => {
        const icon = e.type === 'dispatch' ? '🚀' :
                     e.type === 'approval' ? '✅' :
                     e.type === 'fault' ? '⚠️' :
                     e.type === 'estop' ? '🛑' : 'ℹ️';
        eventsHtml += `
          <div class="event-item">
            <span class="event-time">[${e.time.toFixed(1)}s]</span>
            <span class="event-icon">${icon}</span>
            <span class="event-msg">${e.message}</span>
          </div>
        `;
      });
      eventLogContainer.innerHTML = eventsHtml;
    }
  }

  connectWebSocket();
});
