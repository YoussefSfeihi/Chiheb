#!/bin/bash
# Build the auto-labelled YOLO dataset (training only): one randomized world per seed
# (sun, ambient light, rock field: count, mesh, size, tilt, burial; craters; rover yaw),
# rover patrolling/evading under a dense meteor shower and teleported to new viewpoints,
# frames + labels (rock, meteor, crater, station) from Gazebo's bounding-box camera.
# Each world runs isolated (own Gazebo partition, ROS domain and process group), so it
# never disturbs another running simulation. Two lanes run in parallel.
#
#   tools/make_dataset.sh [frames_per_world] [out_dir]
#
# Train seeds 101-120, val 151-154, test 201-202 (held out for evaluation).
PER=${1:-180}
OUT=${2:-$HOME/crater_ws/dataset}
TOOLS=$(cd "$(dirname "$0")" && pwd)
source "$HOME/crater_ws/install/setup.bash"
set -u
mkdir -p "$OUT/worlds" "$OUT/logs"

collect() {   # collect <seed> <split>   (GZ_PARTITION / ROS_DOMAIN_ID set by the lane)
  local seed=$1 split=$2 world="$OUT/worlds/s$1.sdf"
  python3 "$TOOLS/gen_world.py" --seed "$seed" --randomize --out "$world" > /dev/null
  setsid ros2 launch crater_sim crater_sim.launch.py headless:=true rviz:=false dashboard:=false \
      perception:=false sensor_viz:=false meteor_period:=3 meteor_first_delay:=4 world:="$world" \
      > "$OUT/logs/s$seed.log" 2>&1 < /dev/null &
  sleep 2
  local lp pgid
  lp=$(pgrep -n -f "^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch crater_sim crater_sim.launch.py .*world:=$world")
  pgid=$(ps -o pgid= -p "$lp" | tr -d ' ')
  sleep 20
  timeout 1200 python3 "$TOOLS/collect_dataset.py" --out "$OUT/raw/$split" --prefix "s$seed" \
      --count "$PER" --seed "$seed" --gt "$OUT/worlds/s${seed}_ground_truth.json" 2>&1 | grep -E "done|Error"
  kill -INT -- -"$pgid" 2>/dev/null; sleep 5; kill -KILL -- -"$pgid" 2>/dev/null; sleep 1
}

lane() {      # lane <id> <seed:split>...
  export GZ_PARTITION=crater_dataset_$1 ROS_DOMAIN_ID=$((76 + $1))
  shift
  for job in "$@"; do collect "${job%%:*}" "${job##*:}"; done
}

A=() B=()
for s in $(seq 101 110); do A+=("$s:train"); done
for s in $(seq 111 120); do B+=("$s:train"); done
A+=(151:val 152:val 201:test)
B+=(153:val 154:val 202:test)
lane 1 "${A[@]}" &
lane 2 "${B[@]}" &
wait
echo "dataset complete: $(find "$OUT/raw" -name '*.jpg' | wc -l) images"
