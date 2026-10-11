#!/bin/bash
# Train the onboard AI detector (YOLOv8n, 416 px, CPU) on the auto-labelled dataset and
# install it as crater_perception/models/rock_yolo.onnx (classes: rock, meteor, crater, station).
#
#   tools/train_yolo.sh [epochs] [dataset_dir]
#
# Two isolated Python environments (ROS itself needs the system numpy 1.26):
#   ~/crater_ws/.venv-train  CPU torch + ultralytics (training/export only)
#   ~/crater_ws/.venv-ai     system site-packages + onnxruntime, numpy pinned to 1.26.4
#                            (runtime interpreter of rock_detector_node)
set -eu
EPOCHS=${1:-40}
DATA=${2:-$HOME/crater_ws/dataset}
WS=$HOME/crater_ws
GETPIP=https://bootstrap.pypa.io/get-pip.py
mkvenv() {   # mkvenv <dir> [--system-site-packages]   (python3-venv/ensurepip is not installed)
  [ -x "$1/bin/pip" ] && return
  python3 -m venv --without-pip ${2:-} "$1"
  curl -sSL "$GETPIP" | "$1/bin/python" - -q
}
mkvenv "$WS/.venv-train"
env -i HOME="$HOME" PATH=/usr/bin:/bin "$WS/.venv-train/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
env -i HOME="$HOME" PATH=/usr/bin:/bin "$WS/.venv-train/bin/pip" install -q ultralytics onnx onnxslim
mkvenv "$WS/.venv-ai" --system-site-packages
env -i HOME="$HOME" PATH=/usr/bin:/bin "$WS/.venv-ai/bin/pip" install -q onnxruntime "numpy==1.26.4"

cat > "$DATA/data.yaml" <<YAML
path: $DATA/raw
train: train/images
val: val/images
test: test/images
names:
  0: rock
  1: meteor
  2: crater
  3: station
YAML
cd "$DATA"
Y() { env -i HOME="$HOME" PATH="$WS/.venv-train/bin:/usr/bin:/bin" "$WS/.venv-train/bin/yolo" "$@"; }
Y detect train model=yolov8n.pt data="$DATA/data.yaml" imgsz=416 epochs="$EPOCHS" batch=32 device=cpu \
  workers=8 patience=10 project="$DATA/runs" name=rocks exist_ok=True plots=True \
  fliplr=0.5 hsv_v=0.4 mosaic=1.0 close_mosaic=5
Y detect val model="$DATA/runs/rocks/weights/best.pt" data="$DATA/data.yaml" split=test imgsz=416 device=cpu \
  project="$DATA/runs" name=rocks_test exist_ok=True
Y export model="$DATA/runs/rocks/weights/best.pt" format=onnx imgsz=416 opset=12 simplify=True dynamic=False
cp "$DATA/runs/rocks/weights/best.onnx" "$WS/src/crater_perception/models/rock_yolo.onnx"
echo "installed $WS/src/crater_perception/models/rock_yolo.onnx  (then: colcon build --packages-select crater_perception)"
