/**
 * AURA 3D Logistics Campus Digital Twin Viewport Engine (Three.js r128)
 *
 * Implements:
 * - 3 Warehouses (Warehouse 1, 2, 3) with solid perimeter walls & explicit dock doorways (Docks A–F)
 * - Dedicated Charging Yard (Bays CH-01, CH-02, CH-03)
 * - Complete connected road network matching navigation geometry
 * - Correct vehicle visual model alignment (front forks aligned with velocity vector)
 * - Unrestricted camera exploration (orbit, right-drag pan, wheel zoom, follow mode preserving angles)
 * - Navigation debug overlays (wall colliders, road network, vehicle footprints, heading vectors)
 * - 5 functional view modes: 3D, TOPO, LAYER, HEATMAP, TRAJECTORY
 */

class FactoryScene3D {
  constructor(canvasId, hudContainerId, miniMapCanvasId) {
    this.canvas = document.getElementById(canvasId);
    this.hudContainer = document.getElementById(hudContainerId);
    this.miniMapCanvas = document.getElementById(miniMapCanvasId);
    this.miniMapCtx = this.miniMapCanvas ? this.miniMapCanvas.getContext('2d') : null;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a1220);
    this.scene.fog = new THREE.FogExp2(0x0a1220, 0.005);

    const width = this.canvas.clientWidth || 800;
    const height = this.canvas.clientHeight || 500;

    // Perspective Camera (3D mode)
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);

    // Orthographic Camera (TOPO mode)
    const aspect = width / height;
    const frustumSize = 130;
    this.orthoCamera = new THREE.OrthographicCamera(
      -frustumSize * aspect / 2, frustumSize * aspect / 2,
      frustumSize / 2, -frustumSize / 2, 0.1, 500
    );
    this.orthoCamera.position.set(5, 120, 4);
    this.orthoCamera.lookAt(5, 0, 4);

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;

    // View state
    this.viewMode = 'trajectory'; // '3d', 'topo', 'layer', 'heatmap', 'trajectory'
    this.isFollowCamera = true;
    this.selectedVehicleId = 'AMR-01';

    // Camera controller state (Spherical coords around target)
    this.camTarget = new THREE.Vector3(-25, 1.2, -28); // LookAt target
    this.camRadius = 26.0;   // Distance to target
    this.camTheta = -0.4;    // Azimuth angle (horizontal orbit)
    this.camPhi = 1.05;      // Polar angle (vertical elevation)
    this.minPhi = 0.12;
    this.maxPhi = Math.PI / 2 - 0.04; // Never clip below floor
    this.minRadius = 5.0;
    this.maxRadius = 180.0;

    // Pointer drag tracking
    this.isDragging = false;
    this.dragButton = 0; // 0 = left (orbit), 2 = right (pan)
    this.prevMousePos = { x: 0, y: 0 };

    // Scene groups for layer toggling
    this.buildingsGroup = new THREE.Group();
    this.lanesGroup = new THREE.Group();
    this.obstaclesGroup = new THREE.Group();
    this.collisionGroup = new THREE.Group();
    this.sensorConesGroup = new THREE.Group();
    this.trajectoryGroup = new THREE.Group();
    this.fleetGroup = new THREE.Group();
    this.workersGroup = new THREE.Group();
    this.debugNavGroup = new THREE.Group();
    this.wallCollidersGroup = new THREE.Group();

    this.scene.add(this.buildingsGroup);
    this.scene.add(this.lanesGroup);
    this.scene.add(this.obstaclesGroup);
    this.scene.add(this.collisionGroup);
    this.scene.add(this.sensorConesGroup);
    this.scene.add(this.trajectoryGroup);
    this.scene.add(this.fleetGroup);
    this.scene.add(this.workersGroup);
    this.scene.add(this.debugNavGroup);
    this.scene.add(this.wallCollidersGroup);

    this.amrMeshes = {};
    this.workerMeshes = {};
    this.trailPoints = { 'AMR-01': [], 'AMR-02': [], 'AMR-03': [] };

    // Heatmap Plane
    this.heatmapMesh = null;
    this.heatmapCanvas = document.createElement('canvas');
    this.heatmapCanvas.width = 140;
    this.heatmapCanvas.height = 90;
    this.heatmapCtx = this.heatmapCanvas.getContext('2d');
    this.heatmapTexture = new THREE.CanvasTexture(this.heatmapCanvas);

    // Raycasting for click-to-select AMR
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.onVehicleSelectCallback = null;

    this._initLighting();
    this._initFloorAndHeatmap();
    this._initCampusWarehouses();
    this._initChargingYard();
    this._initRoadMarkings();
    this._initFleetMeshes();
    this._initStaticObstacles();
    this._initSensorCones();
    this._setupInteraction();

    window.addEventListener('resize', () => this.onResize());
    this.animate();
  }

  // ── Lighting ────────────────────────────────────────────────────────
  _initLighting() {
    const ambientLight = new THREE.AmbientLight(0xcfd8dc, 0.75);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(-25, 55, -35);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    this.scene.add(dirLight);

    const neonBlue = new THREE.PointLight(0x0ea5e9, 0.9, 85);
    neonBlue.position.set(5, 12, 0);
    this.scene.add(neonBlue);

    const warmAmber = new THREE.PointLight(0xf59e0b, 0.6, 60);
    warmAmber.position.set(-15, 8, -25);
    this.scene.add(warmAmber);
  }

  // ── Floor & Dynamic Heatmap Plane ──────────────────────────────────
  _initFloorAndHeatmap() {
    // 1. Concrete floor (160m x 110m)
    const floorGeo = new THREE.PlaneGeometry(160, 110);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x141b2d,
      roughness: 0.6,
      metalness: 0.2,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(5, 0, 4);
    floor.receiveShadow = true;
    this.scene.add(floor);

    // Floor grid
    const grid = new THREE.GridHelper(160, 80, 0x334155, 0x1e293b);
    grid.position.set(5, 0.01, 4);
    this.scene.add(grid);

    // 2. Heatmap Overlay Plane (140m x 88m)
    const heatGeo = new THREE.PlaneGeometry(140, 88);
    const heatMat = new THREE.MeshBasicMaterial({
      map: this.heatmapTexture,
      transparent: true,
      opacity: 0.75,
      depthWrite: false,
    });
    this.heatmapMesh = new THREE.Mesh(heatGeo, heatMat);
    this.heatmapMesh.rotation.x = -Math.PI / 2;
    this.heatmapMesh.position.set(5, 0.03, 4);
    this.heatmapMesh.visible = false;
    this.scene.add(this.heatmapMesh);
  }

  // ── 3 Warehouses Buildings & Docks ──────────────────────────────────
  _initCampusWarehouses() {
    // Warehouse 1 (West: Docks A & B, X in [-53, -27], Z in [24, 42])
    this._createWarehouseBuilding('Warehouse 1', -40.0, 33.0, 26.0, 18.0, 0x1e3a5f, [
      { name: 'DOCK A', x: -46.0, zDoor: 24.0, zBay: 26.5, type: 'PICKUP' },
      { name: 'DOCK B', x: -34.0, zDoor: 24.0, zBay: 26.5, type: 'DELIVERY' },
    ]);

    // Warehouse 2 (Central: Docks C & D, X in [-8, 18], Z in [24, 42])
    this._createWarehouseBuilding('Warehouse 2', 5.0, 33.0, 26.0, 18.0, 0x1e293b, [
      { name: 'DOCK C', x: -1.0, zDoor: 24.0, zBay: 26.5, type: 'PICKUP' },
      { name: 'DOCK D', x: 11.0, zDoor: 24.0, zBay: 26.5, type: 'DELIVERY' },
    ]);

    // Warehouse 3 (East: Docks E & F, X in [37, 63], Z in [24, 42])
    this._createWarehouseBuilding('Warehouse 3', 50.0, 33.0, 26.0, 18.0, 0x312e81, [
      { name: 'DOCK E', x: 44.0, zDoor: 24.0, zBay: 26.5, type: 'PICKUP' },
      { name: 'DOCK F', x: 56.0, zDoor: 24.0, zBay: 26.5, type: 'DELIVERY' },
    ]);
  }

  _createWarehouseBuilding(name, centerX, centerZ, width, depth, colorHex, docks) {
    const group = new THREE.Group();
    const height = 9.0;
    const wallThick = 1.0;

    const wallMat = new THREE.MeshStandardMaterial({
      color: colorHex,
      metalness: 0.35,
      roughness: 0.45,
    });

    const roofMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.8,
    });

    // 1. North Wall (Back)
    const nWall = new THREE.Mesh(new THREE.BoxGeometry(width, height, wallThick), wallMat);
    nWall.position.set(centerX, height / 2, centerZ + depth / 2);
    nWall.castShadow = true;
    group.add(nWall);

    // 2. West Wall
    const wWall = new THREE.Mesh(new THREE.BoxGeometry(wallThick, height, depth), wallMat);
    wWall.position.set(centerX - width / 2, height / 2, centerZ);
    wWall.castShadow = true;
    group.add(wWall);

    // 3. East Wall
    const eWall = new THREE.Mesh(new THREE.BoxGeometry(wallThick, height, depth), wallMat);
    eWall.position.set(centerX + width / 2, height / 2, centerZ);
    eWall.castShadow = true;
    group.add(eWall);

    // 4. South Wall with Doorways (Front, Z = centerZ - depth/2)
    // Front wall is split into 3 solid segments leaving 4.0m door gaps for the two docks
    const frontZ = centerZ - depth / 2;
    const d1 = docks[0];
    const d2 = docks[1];

    // West front section
    const wFrontWidth = (d1.x - 2.0) - (centerX - width / 2);
    const wFrontX = (centerX - width / 2) + wFrontWidth / 2;
    const frontW = new THREE.Mesh(new THREE.BoxGeometry(wFrontWidth, height, wallThick), wallMat);
    frontW.position.set(wFrontX, height / 2, frontZ);
    group.add(frontW);

    // Mid front section between doors
    const midFrontWidth = (d2.x - 2.0) - (d1.x + 2.0);
    const midFrontX = (d1.x + 2.0) + midFrontWidth / 2;
    const frontMid = new THREE.Mesh(new THREE.BoxGeometry(midFrontWidth, height, wallThick), wallMat);
    frontMid.position.set(midFrontX, height / 2, frontZ);
    group.add(frontMid);

    // East front section
    const eFrontWidth = (centerX + width / 2) - (d2.x + 2.0);
    const eFrontX = (d2.x + 2.0) + eFrontWidth / 2;
    const frontE = new THREE.Mesh(new THREE.BoxGeometry(eFrontWidth, height, wallThick), wallMat);
    frontE.position.set(eFrontX, height / 2, frontZ);
    group.add(frontE);

    // Doorway lintels (roof bridge over door openings)
    [d1.x, d2.x].forEach(dx => {
      const lintel = new THREE.Mesh(new THREE.BoxGeometry(4.0, height - 5.0, wallThick), wallMat);
      lintel.position.set(dx, height - (height - 5.0) / 2, frontZ);
      group.add(lintel);
    });

    // 5. Roof with skylights
    const roof = new THREE.Mesh(new THREE.BoxGeometry(width + 0.8, 0.4, depth + 0.8), roofMat);
    roof.position.set(centerX, height + 0.2, centerZ);
    group.add(roof);

    // 6. Large Warehouse Sign Billboard
    this._createSignSprite(name.toUpperCase(), centerX, height + 2.2, centerZ, colorHex, 8.0, 2.0, group);

    // 7. Loading Docks
    docks.forEach(dock => {
      // Dock Bay Floor Pad inside warehouse
      const padColor = (dock.type === 'PICKUP') ? 0x0284c7 : 0x10b981;
      const padMat = new THREE.MeshStandardMaterial({ color: padColor, roughness: 0.4 });
      const pad = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.08, 4.0), padMat);
      pad.position.set(dock.x, 0.04, dock.zBay);
      pad.receiveShadow = true;
      group.add(pad);

      // Yellow striped warning boundary around dock door
      const borderGeo = new THREE.PlaneGeometry(3.8, 0.25);
      const borderMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
      const bMesh = new THREE.Mesh(borderGeo, borderMat);
      bMesh.rotation.x = -Math.PI / 2;
      bMesh.position.set(dock.x, 0.06, dock.zDoor);
      group.add(bMesh);

      // Dock overhead sign
      const signColor = (dock.type === 'PICKUP') ? 0x38bdf8 : 0x34d399;
      this._createSignSprite(`${dock.name} (${dock.type})`, dock.x, 4.8, dock.zDoor, signColor, 3.2, 0.9, group);
    });

    this.buildingsGroup.add(group);
  }

  // ── Charging Yard & Bays ───────────────────────────────────────────
  _initChargingYard() {
    const bays = [
      { name: 'CH-01', x: -25.0, z: -28.0 },
      { name: 'CH-02', x: -15.0, z: -28.0 },
      { name: 'CH-03', x: -5.0, z: -28.0 },
    ];

    bays.forEach(bay => {
      const group = new THREE.Group();

      // Pad
      const padMat = new THREE.MeshStandardMaterial({ color: 0x064e3b, roughness: 0.5 });
      const pad = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.06, 3.8), padMat);
      pad.position.set(bay.x, 0.03, bay.z);
      pad.receiveShadow = true;
      group.add(pad);

      // Charging Pylon Station
      const pylonMat = new THREE.MeshStandardMaterial({ color: 0x10b981, metalness: 0.6 });
      const pylon = new THREE.Mesh(new THREE.BoxGeometry(0.7, 2.5, 0.7), pylonMat);
      pylon.position.set(bay.x, 1.25, bay.z - 2.2);
      group.add(pylon);

      // LED screen
      const screen = new THREE.Mesh(
        new THREE.PlaneGeometry(0.5, 0.8),
        new THREE.MeshBasicMaterial({ color: 0x34d399 })
      );
      screen.position.set(bay.x, 1.8, bay.z - 1.84);
      group.add(screen);

      // Sign
      this._createSignSprite(bay.name, bay.x, 3.2, bay.z - 2.2, 0x10b981, 2.0, 0.8, group);

      this.buildingsGroup.add(group);
    });
  }

  // ── Connected Road Network Markings ────────────────────────────────
  _initRoadMarkings() {
    const laneMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
    const whiteMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });
    const asphaltMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.7 });

    // 1. Main East-West Highway Surface (Z = 0.0, length 120m, width 8.0m)
    const mainAsphalt = new THREE.Mesh(new THREE.PlaneGeometry(120, 8.0), asphaltMat);
    mainAsphalt.rotation.x = -Math.PI / 2;
    mainAsphalt.position.set(5, 0.015, 0);
    mainAsphalt.receiveShadow = true;
    this.lanesGroup.add(mainAsphalt);

    // Highway yellow dashed center line
    for (let x = -50; x <= 60; x += 4.0) {
      const dash = new THREE.Mesh(new THREE.PlaneGeometry(2.0, 0.2), laneMat);
      dash.rotation.x = -Math.PI / 2;
      dash.position.set(x, 0.022, 0);
      this.lanesGroup.add(dash);
    }

    // Highway solid white edges
    [-3.8, 3.8].forEach(offsetZ => {
      const line = new THREE.Mesh(new THREE.PlaneGeometry(120, 0.2), whiteMat);
      line.rotation.x = -Math.PI / 2;
      line.position.set(5, 0.022, offsetZ);
      this.lanesGroup.add(line);
    });

    // 2. Charging Yard Access Avenue (North-South, X = -15, Z in [-24, 0], width 6.0m)
    const chAve = new THREE.Mesh(new THREE.PlaneGeometry(6.0, 24.0), asphaltMat);
    chAve.rotation.x = -Math.PI / 2;
    chAve.position.set(-15, 0.015, -12);
    this.lanesGroup.add(chAve);

    // Charging staging lane (East-West, Z = -24, X in [-28, -2], width 6.0m)
    const chStaging = new THREE.Mesh(new THREE.PlaneGeometry(26.0, 6.0), asphaltMat);
    chStaging.rotation.x = -Math.PI / 2;
    chStaging.position.set(-15, 0.015, -24);
    this.lanesGroup.add(chStaging);

    // 3. Three North-South Warehouse Access Avenues (Z in [0, 18], width 6.0m)
    [-40.0, 5.0, 50.0].forEach(ax => {
      const ave = new THREE.Mesh(new THREE.PlaneGeometry(6.0, 18.0), asphaltMat);
      ave.rotation.x = -Math.PI / 2;
      ave.position.set(ax, 0.015, 9);
      this.lanesGroup.add(ave);

      // Centerline
      for (let z = 2.0; z <= 16.0; z += 3.5) {
        const dash = new THREE.Mesh(new THREE.PlaneGeometry(0.2, 1.8), laneMat);
        dash.rotation.x = -Math.PI / 2;
        dash.position.set(ax, 0.022, z);
        this.lanesGroup.add(dash);
      }
    });

    // 4. Warehouse Aprons & Dock Spurs (Connecting to Doors at Z = 24.0)
    [
      { apronX: -40.0, docks: [-46.0, -34.0] },
      { apronX: 5.0, docks: [-1.0, 11.0] },
      { apronX: 50.0, docks: [44.0, 56.0] },
    ].forEach(w => {
      // Apron transverse road at Z = 18.0
      const apronWidth = Math.abs(w.docks[1] - w.docks[0]) + 10.0;
      const apronMesh = new THREE.Mesh(new THREE.PlaneGeometry(apronWidth, 5.0), asphaltMat);
      apronMesh.rotation.x = -Math.PI / 2;
      apronMesh.position.set(w.apronX, 0.015, 18.0);
      this.lanesGroup.add(apronMesh);

      // Spurs into each dock door
      w.docks.forEach(dx => {
        const spur = new THREE.Mesh(new THREE.PlaneGeometry(4.5, 6.0), asphaltMat);
        spur.rotation.x = -Math.PI / 2;
        spur.position.set(dx, 0.016, 21.0);
        this.lanesGroup.add(spur);
      });
    });

    // Zebra Crosswalks
    this._createCrosswalk(-28.0, 0.0);
    this._createCrosswalk(18.0, 0.0);
  }

  _createCrosswalk(cx, cz) {
    const whiteMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
    for (let i = -3.0; i <= 3.0; i += 1.0) {
      const stripe = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 3.6), whiteMat);
      stripe.rotation.x = -Math.PI / 2;
      stripe.position.set(cx + i, 0.025, cz);
      this.lanesGroup.add(stripe);
    }
  }

  // ── Fleet Vehicles (AMR-01, AMR-02, AMR-03) ──────────────────────────
  _initFleetMeshes() {
    const configs = [
      { id: 'AMR-01', color: 0x0284c7, accent: 0x38bdf8, x: -25.0, z: -28.0 },
      { id: 'AMR-02', color: 0x0d9488, accent: 0x2dd4bf, x: -15.0, z: -28.0 },
      { id: 'AMR-03', color: 0x7c3aed, accent: 0xa78bfa, x: -5.0, z: -28.0 },
    ];

    configs.forEach(cfg => {
      const amrGroup = new THREE.Group();
      amrGroup.name = cfg.id;
      amrGroup.userData = { vehicleId: cfg.id };

      // Root inner visual container (Forklift natural front is +Z)
      const visualRoot = new THREE.Group();
      visualRoot.name = 'visualRoot';

      // 1. Chassis Body
      const bodyMat = new THREE.MeshStandardMaterial({
        color: cfg.color,
        metalness: 0.4,
        roughness: 0.35,
      });
      const chassis = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.65, 2.4), bodyMat);
      chassis.position.y = 0.55;
      chassis.castShadow = true;
      visualRoot.add(chassis);

      // Top Cabin Hood
      const hood = new THREE.Mesh(
        new THREE.BoxGeometry(1.2, 0.55, 1.3),
        new THREE.MeshStandardMaterial({ color: 0x1e293b })
      );
      hood.position.set(0, 1.15, -0.35);
      hood.castShadow = true;
      visualRoot.add(hood);

      // 2. Forklift Mast & Forks (Front = +Z)
      const mastMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.6 });
      const mastL = new THREE.Mesh(new THREE.BoxGeometry(0.12, 2.0, 0.12), mastMat);
      mastL.position.set(-0.45, 1.3, 1.25);
      visualRoot.add(mastL);

      const mastR = new THREE.Mesh(new THREE.BoxGeometry(0.12, 2.0, 0.12), mastMat);
      mastR.position.set(0.45, 1.3, 1.25);
      visualRoot.add(mastR);

      // Fork tines
      const forkMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.8 });
      const forkL = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.08, 1.2), forkMat);
      forkL.position.set(-0.32, 0.18, 1.8);
      visualRoot.add(forkL);

      const forkR = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.08, 1.2), forkMat);
      forkR.position.set(0.32, 0.18, 1.8);
      visualRoot.add(forkR);

      // 3. Cargo Pallet (visible only when loaded)
      const palletGroup = new THREE.Group();
      palletGroup.name = 'cargoPallet';
      const palletBase = new THREE.Mesh(
        new THREE.BoxGeometry(1.3, 0.18, 1.3),
        new THREE.MeshStandardMaterial({ color: 0xb45309, roughness: 0.8 })
      );
      palletBase.position.set(0, 0.32, 1.8);
      palletGroup.add(palletBase);

      const boxCargo = new THREE.Mesh(
        new THREE.BoxGeometry(1.1, 0.9, 1.1),
        new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.6 })
      );
      boxCargo.position.set(0, 0.85, 1.8);
      palletGroup.add(boxCargo);

      palletGroup.visible = false;
      visualRoot.add(palletGroup);

      // 4. Wheels
      const wheelMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.8 });
      [
        [-0.8, 0.3, 0.7],
        [0.8, 0.3, 0.7],
        [-0.8, 0.3, -0.7],
        [0.8, 0.3, -0.7],
      ].forEach(pos => {
        const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.2, 16), wheelMat);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(...pos);
        wheel.castShadow = true;
        visualRoot.add(wheel);
      });

      // 5. Headlights
      const lightL = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      lightL.position.set(-0.5, 0.8, 1.22);
      visualRoot.add(lightL);

      const lightR = new THREE.Mesh(new THREE.SphereGeometry(0.1, 8, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      lightR.position.set(0.5, 0.8, 1.22);
      visualRoot.add(lightR);

      // 6. Amber Warning Beacon
      const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.2, 8), new THREE.MeshBasicMaterial({ color: 0xf59e0b }));
      beacon.position.set(0, 1.5, -0.35);
      visualRoot.add(beacon);

      // 7. Floating ID Tag Sprite above vehicle
      const badge = this._createVehicleBadgeSprite(cfg.id, cfg.accent);
      badge.position.set(0, 2.6, 0);
      visualRoot.add(badge);

      // 8. Collision wireframe circle (Footprint)
      const circleGeo = new THREE.RingGeometry(1.22, 1.28, 32);
      const circleMat = new THREE.MeshBasicMaterial({ color: cfg.accent, side: THREE.DoubleSide });
      const collisionCircle = new THREE.Mesh(circleGeo, circleMat);
      collisionCircle.rotation.x = -Math.PI / 2;
      collisionCircle.position.y = 0.05;
      collisionCircle.name = 'collisionCircle';
      visualRoot.add(collisionCircle);

      // 9. Safety Margin Circle (Outer dashed ring, radius = 1.75m)
      const marginGeo = new THREE.RingGeometry(1.72, 1.78, 32);
      const marginMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
      const marginCircle = new THREE.Mesh(marginGeo, marginMat);
      marginCircle.rotation.x = -Math.PI / 2;
      marginCircle.position.y = 0.04;
      marginCircle.name = 'safetyMarginCircle';
      visualRoot.add(marginCircle);

      // 10. Forward Direction Arrow
      const arrowGeo = new THREE.ConeGeometry(0.2, 0.6, 8);
      const arrowMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
      const arrow = new THREE.Mesh(arrowGeo, arrowMat);
      arrow.rotation.x = Math.PI / 2;
      arrow.position.set(0, 0.12, 2.6);
      arrow.name = 'forwardArrow';
      visualRoot.add(arrow);

      amrGroup.add(visualRoot);
      amrGroup.position.set(cfg.x, 0, cfg.z);
      this.fleetGroup.add(amrGroup);
      this.amrMeshes[cfg.id] = amrGroup;
    });
  }

  // ── Static Obstacles & Shelving ─────────────────────────────────────
  _initStaticObstacles() {
    // External storage yard racks (South-East)
    this._createStorageRack(26.0, -15.0);
    this._createStorageRack(36.0, -15.0);

    // Pallets
    this._createPalletObstacle(-3.0, 9.0);
    this._createPalletObstacle(22.0, -5.0);

    // Cones
    this._createCone(-25.0, 6.0);
    this._createCone(35.0, 6.0);
  }

  _createStorageRack(x, z) {
    const rack = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.6, roughness: 0.3 });
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(3.6, 4.5, 8.0), mat);
    shelf.position.set(x, 2.25, z);
    shelf.castShadow = true;
    rack.add(shelf);
    this.obstaclesGroup.add(rack);
  }

  _createPalletObstacle(x, z) {
    const pallet = new THREE.Group();
    const box = new THREE.Mesh(
      new THREE.BoxGeometry(1.6, 1.2, 1.4),
      new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.7 })
    );
    box.position.set(x, 0.6, z);
    box.castShadow = true;
    pallet.add(box);
    this.obstaclesGroup.add(pallet);
  }

  _createCone(x, z) {
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.35, 0.8, 12),
      new THREE.MeshStandardMaterial({ color: 0xf97316 })
    );
    cone.position.set(x, 0.4, z);
    this.obstaclesGroup.add(cone);
  }

  // ── Sensor Coverage Cones ───────────────────────────────────────────
  _initSensorCones() {
    // 1. Camera FOV Cone
    const camGeo = new THREE.ConeGeometry(12, 16, 16, 1, true, -Math.PI / 4, Math.PI / 2);
    const camMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const camCone = new THREE.Mesh(camGeo, camMat);
    camCone.rotation.x = Math.PI / 2;
    camCone.position.set(0, 0.5, 8);
    this.sensorConesGroup.add(camCone);

    // 2. LiDAR 360 circle
    const lidarGeo = new THREE.RingGeometry(0.5, 20, 32);
    const lidarMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.08,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const lidarRing = new THREE.Mesh(lidarGeo, lidarMat);
    lidarRing.rotation.x = -Math.PI / 2;
    lidarRing.position.set(0, 0.1, 0);
    this.sensorConesGroup.add(lidarRing);
  }

  _createSignSprite(text, x, y, z, colorHex, width = 2.4, height = 0.9, parentGroup = null) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#0f172a';
    ctx.roundRect(4, 4, 248, 88, 12);
    ctx.fill();

    const hex = '#' + colorHex.toString(16).padStart(6, '0');
    ctx.strokeStyle = hex;
    ctx.lineWidth = 4;
    ctx.roundRect(4, 4, 248, 88, 12);
    ctx.stroke();

    ctx.font = 'bold 26px Inter, sans-serif';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 48);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), mat);
    mesh.position.set(x, y, z);
    if (parentGroup) {
      parentGroup.add(mesh);
    } else {
      this.scene.add(mesh);
    }
    return mesh;
  }

  _createVehicleBadgeSprite(name, colorHex) {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#0f172a';
    ctx.roundRect(4, 4, 152, 56, 12);
    ctx.fill();

    const hex = '#' + colorHex.toString(16).padStart(6, '0');
    ctx.strokeStyle = hex;
    ctx.lineWidth = 4;
    ctx.roundRect(4, 4, 152, 56, 12);
    ctx.stroke();

    ctx.font = 'bold 28px Inter, sans-serif';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(name, 80, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.8, 0.72, 1.0);
    return sprite;
  }

  // ── Raycasting Interaction (Click-to-Select Vehicle) & Free Camera ───
  _setupInteraction() {
    // 1. Raycast click selection
    this.canvas.addEventListener('click', (event) => {
      // Ignore click if was a drag
      if (this._hasDragged) return;

      const rect = this.canvas.getBoundingClientRect();
      this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      const activeCamera = (this.viewMode === 'topo') ? this.orthoCamera : this.camera;
      this.raycaster.setFromCamera(this.mouse, activeCamera);

      const intersects = this.raycaster.intersectObjects(this.fleetGroup.children, true);
      if (intersects.length > 0) {
        let obj = intersects[0].object;
        while (obj && !obj.userData.vehicleId && obj.parent) {
          obj = obj.parent;
        }
        if (obj && obj.userData.vehicleId) {
          const clickedId = obj.userData.vehicleId;
          this.setSelectedVehicle(clickedId);
          if (this.onVehicleSelectCallback) {
            this.onVehicleSelectCallback(clickedId);
          }
        }
      }
    });

    // 2. Unrestricted Orbit + Pan + Zoom Controller
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this._hasDragged = false;
      this.dragButton = e.button;
      this.prevMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;

      const dx = e.clientX - this.prevMousePos.x;
      const dy = e.clientY - this.prevMousePos.y;
      if (Math.hypot(dx, dy) > 2) this._hasDragged = true;

      this.prevMousePos = { x: e.clientX, y: e.clientY };

      if (this.dragButton === 0) {
        // Left drag = Orbit
        this.camTheta -= dx * 0.006;
        this.camPhi -= dy * 0.006;
        this.camPhi = Math.max(this.minPhi, Math.min(this.maxPhi, this.camPhi));
      } else if (this.dragButton === 2 || e.shiftKey) {
        // Right drag = Pan lookAt target
        const panSpeed = 0.04 * (this.camRadius / 25.0);
        const forward = new THREE.Vector3(-Math.sin(this.camTheta), 0, -Math.cos(this.camTheta));
        const right = new THREE.Vector3(Math.cos(this.camTheta), 0, -Math.sin(this.camTheta));

        this.camTarget.addScaledVector(right, -dx * panSpeed);
        this.camTarget.addScaledVector(forward, dy * panSpeed);
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // Mouse wheel zoom
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = 1.0 + Math.sign(e.deltaY) * 0.08;
      this.camRadius = Math.max(this.minRadius, Math.min(this.maxRadius, this.camRadius * zoomFactor));
    }, { passive: false });

    // Touch support (1-finger orbit, 2-finger pan/zoom)
    let touchStartDist = 0;
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this._hasDragged = false;
        this.dragButton = 0;
        this.prevMousePos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        touchStartDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
      }
    });

    this.canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && this.isDragging) {
        const dx = e.touches[0].clientX - this.prevMousePos.x;
        const dy = e.touches[0].clientY - this.prevMousePos.y;
        this._hasDragged = true;
        this.prevMousePos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
        this.camTheta -= dx * 0.007;
        this.camPhi = Math.max(this.minPhi, Math.min(this.maxPhi, this.camPhi - dy * 0.007));
      } else if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY
        );
        const factor = touchStartDist / Math.max(dist, 1);
        this.camRadius = Math.max(this.minRadius, Math.min(this.maxRadius, this.camRadius * factor));
        touchStartDist = dist;
      }
    });

    this.canvas.addEventListener('touchend', () => {
      this.isDragging = false;
    });
  }

  setSelectedVehicle(vehicleId) {
    if (this.amrMeshes[vehicleId]) {
      this.selectedVehicleId = vehicleId;
    }
  }

  recenterCamera() {
    const selMesh = this.amrMeshes[this.selectedVehicleId];
    if (selMesh) {
      this.camTarget.set(selMesh.position.x, 1.2, selMesh.position.z);
    } else {
      this.camTarget.set(5, 1.2, 0);
    }
  }

  resetView() {
    this.camTheta = -0.4;
    this.camPhi = 1.05;
    this.camRadius = 26.0;
    this.recenterCamera();
  }

  // ── View Mode Switching ─────────────────────────────────────────────
  setViewMode(mode) {
    this.viewMode = mode;

    if (this.heatmapMesh) {
      this.heatmapMesh.visible = (mode === 'heatmap');
    }

    this.trajectoryGroup.visible = (mode === 'trajectory' || mode === '3d');

    if (mode === 'topo') {
      this.isFollowCamera = false;
      this.orthoCamera.position.set(5, 120, 4);
      this.orthoCamera.lookAt(5, 0, 4);
    } else if (mode === '3d') {
      this.isFollowCamera = true;
    }
  }

  setLayerVisibility(layerName, visible) {
    if (layerName === 'buildings') this.buildingsGroup.visible = visible;
    if (layerName === 'lanes') this.lanesGroup.visible = visible;
    if (layerName === 'obstacles') this.obstaclesGroup.visible = visible;
    if (layerName === 'collision') this.collisionGroup.visible = visible;
    if (layerName === 'sensors') this.sensorConesGroup.visible = visible;
    if (layerName === 'trajectories') this.trajectoryGroup.visible = visible;
    if (layerName === 'navmesh') this.debugNavGroup.visible = visible;
    if (layerName === 'colliders') this.wallCollidersGroup.visible = visible;
  }

  toggleFollowCamera() {
    this.isFollowCamera = !this.isFollowCamera;
    return this.isFollowCamera;
  }

  // ── Real-time Telemetry Updates ─────────────────────────────────────
  updateTelemetry(data) {
    if (!data) return;

    // 1. Update all fleet vehicles
    if (data.fleet) {
      data.fleet.forEach(v => {
        const mesh = this.amrMeshes[v.vehicle_id];
        if (mesh) {
          // Position
          mesh.position.set(v.x, 0, v.y);

          // EXACT MODEL FORWARD ALIGNMENT:
          // In Three.js, local +Z is model front. Heading angle theta moves in (cos theta, sin theta).
          // Math.atan2(cos, sin) aligns local +Z with velocity vector.
          mesh.rotation.y = Math.atan2(Math.cos(v.heading), Math.sin(v.heading));

          // Toggle cargo pallet
          const pallet = mesh.getObjectByName('cargoPallet');
          if (pallet) pallet.visible = !!v.cargo_loaded;

          // Highlight selected vehicle
          const isSelected = (v.vehicle_id === this.selectedVehicleId);
          const circle = mesh.getObjectByName('collisionCircle');
          if (circle) {
            circle.material.color.setHex(isSelected ? 0x38bdf8 : 0x475569);
            circle.scale.setScalar(isSelected ? 1.15 : 1.0);
          }

          // Trail points
          if (!this.trailPoints[v.vehicle_id]) this.trailPoints[v.vehicle_id] = [];
          const pts = this.trailPoints[v.vehicle_id];
          if (pts.length === 0 || Math.hypot(v.x - pts[pts.length - 1].x, v.y - pts[pts.length - 1].z) > 0.4) {
            pts.push(new THREE.Vector3(v.x, 0.08, v.y));
            if (pts.length > 250) pts.shift();
          }
        }
      });
    }

    // 2. Camera Update (Smoothly follow selected vehicle while preserving user orbit angles)
    const selMesh = this.amrMeshes[this.selectedVehicleId];
    if (selMesh) {
      this.sensorConesGroup.position.copy(selMesh.position);
      this.sensorConesGroup.rotation.y = selMesh.rotation.y;

      if (this.isFollowCamera && this.viewMode !== 'topo') {
        this.camTarget.lerp(new THREE.Vector3(selMesh.position.x, 1.2, selMesh.position.z), 0.1);
      }
    }

    // 3. Update Workers
    if (data.workers) {
      data.workers.forEach(w => {
        let wMesh = this.workerMeshes[w.id];
        if (!wMesh) {
          wMesh = this._createWorkerMesh(w.name);
          this.workersGroup.add(wMesh);
          this.workerMeshes[w.id] = wMesh;
        }
        wMesh.position.set(w.x, 0, w.y);
      });
    }

    // 4. Update Path Trajectories
    if (data.navigation) {
      this._updatePathLines(data.navigation.normal_path, data.navigation.alt_path);
    }

    // 5. Update Wall Colliders and Road Network Debug Overlays
    if (data.campus && this.wallCollidersGroup.children.length === 0) {
      this._buildWallCollidersOverlay(data.campus.wall_colliders || []);
      this._buildRoadNetworkOverlay(data.campus.roads || []);
    }

    // 6. Heatmap Texture
    if (data.heatmap && data.heatmap.data && this.viewMode === 'heatmap') {
      this._renderHeatmapTexture(data.heatmap);
    }

    // 7. Mini-map
    if (this.miniMapCtx && data.vehicle) {
      this._renderMiniMap(data);
    }
  }

  _buildWallCollidersOverlay(wallColliders) {
    const wallLineMat = new THREE.LineBasicMaterial({ color: 0xef4444 });
    wallColliders.forEach(w => {
      const boxGeo = new THREE.BoxGeometry(w.w, 3.5, w.h);
      const edges = new THREE.EdgesGeometry(boxGeo);
      const wire = new THREE.LineSegments(edges, wallLineMat);
      wire.position.set(w.x, 1.75, w.y);
      this.wallCollidersGroup.add(wire);
    });
  }

  _buildRoadNetworkOverlay(roads) {
    const lineMat = new THREE.LineBasicMaterial({ color: 0x0ea5e9, transparent: true, opacity: 0.6 });
    roads.forEach(r => {
      const pts = [
        new THREE.Vector3(r.x1, 0.05, r.y1),
        new THREE.Vector3(r.x2, 0.05, r.y2),
      ];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const line = new THREE.Line(geo, lineMat);
      this.debugNavGroup.add(line);
    });
  }

  _createWorkerMesh(name) {
    const group = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.28, 0.28, 1.0, 8),
      new THREE.MeshStandardMaterial({ color: 0xf97316 })
    );
    body.position.y = 0.9;
    group.add(body);

    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 8, 8),
      new THREE.MeshStandardMaterial({ color: 0xfacc15 })
    );
    head.position.y = 1.6;
    group.add(head);
    return group;
  }

  // ── Path Spline Trajectory Visualization ───────────────────────────
  _updatePathLines(normalPath, altPath) {
    while (this.trajectoryGroup.children.length > 0) {
      const obj = this.trajectoryGroup.children[0];
      this.trajectoryGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    if (normalPath && normalPath.length > 1) {
      const pts = normalPath.map(p => new THREE.Vector3(p.x, 0.08, p.y));
      const curve = new THREE.CatmullRomCurve3(pts);
      const curvePts = curve.getPoints(Math.max(20, pts.length * 5));
      const lineGeo = new THREE.BufferGeometry().setFromPoints(curvePts);
      const lineMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 3 });
      const line = new THREE.Line(lineGeo, lineMat);
      this.trajectoryGroup.add(line);

      // Pins at end points
      normalPath.forEach((wp, idx) => {
        if (idx === 0 || idx === normalPath.length - 1) {
          const pinColor = idx === 0 ? 0x38bdf8 : 0x10b981;
          const pin = new THREE.Mesh(
            new THREE.CylinderGeometry(0.25, 0.05, 0.6, 8),
            new THREE.MeshBasicMaterial({ color: pinColor })
          );
          pin.position.set(wp.x, 0.35, wp.y);
          this.trajectoryGroup.add(pin);
        }
      });
    }

    if (altPath && altPath.length > 1) {
      const altPts = altPath.map(p => new THREE.Vector3(p.x, 0.08, p.y));
      const altCurve = new THREE.CatmullRomCurve3(altPts);
      const altLineGeo = new THREE.BufferGeometry().setFromPoints(altCurve.getPoints(Math.max(20, altPts.length * 4)));
      const altLineMat = new THREE.LineDashedMaterial({ color: 0xf43f5e, dashSize: 0.8, gapSize: 0.4 });
      const altLine = new THREE.Line(altLineGeo, altLineMat);
      altLine.computeLineDistances();
      this.trajectoryGroup.add(altLine);
    }
  }

  // ── Dynamic 2D Traffic Density Heatmap Rendering ───────────────────
  _renderHeatmapTexture(heatmapData) {
    const { rows, cols, data } = heatmapData;
    const w = this.heatmapCanvas.width;
    const h = this.heatmapCanvas.height;
    const ctx = this.heatmapCtx;

    ctx.clearRect(0, 0, w, h);
    const cellW = w / cols;
    const cellH = h / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = data[r * cols + c];
        if (val > 0.02) {
          const red = Math.round(Math.min(255, val * 350));
          const green = Math.round(Math.max(0, 200 - val * 200));
          const alpha = Math.min(0.85, val * 1.2);
          ctx.fillStyle = `rgba(${red}, ${green}, 40, ${alpha})`;
          ctx.fillRect(c * cellW, (rows - 1 - r) * cellH, cellW + 1, cellH + 1);
        }
      }
    }
    this.heatmapTexture.needsUpdate = true;
  }

  // ── Mini-Map HUD ───────────────────────────────────────────────────
  _renderMiniMap(data) {
    const ctx = this.miniMapCtx;
    const w = this.miniMapCanvas.width;
    const h = this.miniMapCanvas.height;

    ctx.fillStyle = '#0a1220';
    ctx.fillRect(0, 0, w, h);

    // Range rings
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    [25, 45, 60].forEach(r => {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2);
      ctx.stroke();
    });

    const ego = data.vehicle;
    if (!ego) return;

    // Obstacles relative to ego
    if (data.risk && data.risk.obstacle_risks) {
      data.risk.obstacle_risks.forEach(o => {
        const scale = 2.2;
        const ox = w / 2 + (o.distance_m * scale * 0.7);
        const oy = h / 2;
        ctx.fillStyle = o.risk_level === 'HIGH' ? '#ef4444' : '#f59e0b';
        ctx.beginPath();
        ctx.arc(ox, oy, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Ego marker
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  onResize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    const aspect = width / height;
    const frustumSize = 130;
    this.orthoCamera.left = -frustumSize * aspect / 2;
    this.orthoCamera.right = frustumSize * aspect / 2;
    this.orthoCamera.top = frustumSize / 2;
    this.orthoCamera.bottom = -frustumSize / 2;
    this.orthoCamera.updateProjectionMatrix();

    this.renderer.setSize(width, height);
  }

  // ── Main Render Loop ────────────────────────────────────────────────
  animate() {
    requestAnimationFrame(() => this.animate());

    // Update perspective camera spherical position relative to camTarget
    const x = this.camTarget.x + this.camRadius * Math.sin(this.camPhi) * Math.sin(this.camTheta);
    const y = Math.max(1.2, this.camTarget.y + this.camRadius * Math.cos(this.camPhi));
    const z = this.camTarget.z + this.camRadius * Math.sin(this.camPhi) * Math.cos(this.camTheta);

    this.camera.position.set(x, y, z);
    this.camera.lookAt(this.camTarget);

    const activeCamera = (this.viewMode === 'topo') ? this.orthoCamera : this.camera;
    this.renderer.render(this.scene, activeCamera);
  }
}
