# C.R.A.T.E.R — ATEC'26 lunar scout

ROS 2 Jazzy + Gazebo workspace for the ATEC'26 hackathon: a lunar rover that patrols around its
landing station, tracks incoming meteors and evades predicted impacts, and detects/localizes rocks
with an onboard YOLO model on RGB-D data (Module 2 perception).

## Packages (`src/`)

| Package | Contents |
|---|---|
| `crater_sim` | Gazebo world + rover model, lunar rock meshes, Nav2/EKF config, main launch file, dataset/training tools |
| `crater_perception` | `meteor_tracker`, `mission_controller`, `rgbd_sync_node`, `rock_detector_node` (YOLO ONNX), `depth_fusion_node`, `rock_localizer` |
| `crater_dashboard` | Mission Control web dashboard (`dashboard_server`, `telemetry_tx`), served on http://localhost:8080 |
| `crater_interfaces` | `RockDetection(Array)`, `MeteorTrack(Array)` messages |

`dataset/v1_meteor/` holds the auto-labelled dataset (worlds, raw frames + labels, collection logs).

## Build & run

```bash
source /opt/ros/jazzy/setup.bash
cd ~/crater_ws && colcon build && source install/setup.bash

ros2 launch crater_sim crater_sim.launch.py
ros2 launch crater_sim crater_sim.launch.py meteor_period:=15
ros2 launch crater_sim crater_sim.launch.py headless:=true rviz:=false
ros2 launch crater_sim crater_sim.launch.py link_loss:=0.3 link_delay_ms:=250   # harsher radio
```

## Dataset & detector training

```bash
src/crater_sim/tools/make_dataset.sh [frames_per_world] [out_dir]   # auto-labelled YOLO dataset
src/crater_sim/tools/train_yolo.sh [epochs] [dataset_dir]           # trains YOLOv8n, exports
                                                                    # crater_perception/models/rock_yolo.onnx
```

`train_yolo.sh` creates the two Python venvs (`.venv-train`, `.venv-ai`); they are not committed.
