# AURA Dataset & Asset Directory

This directory contains local raw datasets, processed feature caches, and scenario recordings.

> **Note:** Raw dataset binaries, archives, and heavy sensor logs are excluded from Git via `.gitignore` to keep the repository lightweight and maintain version control performance. Only directory structures (`.gitkeep`) and documentation are committed.

## Directory Structure

```
data/
├── raw/                      # Downloaded / extracted benchmark datasets
│   ├── carla/                # CARLA simulator binary distribution & asset cache
│   ├── kitti/                # KITTI 3D object detection / tracking benchmark
│   └── nuscenes/             # nuScenes dataset splits (e.g., v1.0-mini)
├── processed/                # Normalized frames in AURA's unified schema
└── scenarios/                # Recorded telemetry traces & simulation replay files
```

## Adding Datasets

### 1. nuScenes (v1.0-mini)
Download the nuScenes mini split from [nuscenes.org](https://www.nuscenes.org/download) and extract into `data/raw/nuscenes/`:
```
data/raw/nuscenes/
├── samples/
├── sweeps/
├── v1.0-mini/
└── maps/
```

### 2. KITTI Object Detection
Download the left color images, velodyne point clouds, and labels from the [KITTI benchmark suite](http://www.cvlibs.net/datasets/kitti/):
```
data/raw/kitti/
├── data_object_image_2/
├── data_object_velodyne/
└── data_object_label_2/
```

### 3. CARLA Simulator (Optional Hardware-in-the-Loop Twin)
Install CARLA 0.9.15+ into `data/raw/carla/` or configure the path in `config.py` / `simulation/carla_bridge/client.py`.
