#!/usr/bin/env python3
"""Measure Module 2 (Perception RGB-D) against simulator ground truth (evaluation only).

Runs next to a live sim (perception:=true) for --duration sim-seconds and reports:
  * detection: TP / FP / FN, precision, recall per class at IoU >= 0.5, comparing
    /perception/detections_2d with Gazebo's bounding-box camera (/camera/bbox, same
    pose, intrinsics and sim time as the RGB camera). GT boxes < --min-box px ignored.
  * depth: for every /perception/rocks entry with depth_valid, the GT rock (world json)
    whose projected centre falls in the box is found with the true camera pose
    (/ground_truth/odom + the camera mount); GT depth = optical-axis distance to the
    rock centre minus half its mean footprint (the visible near face).
    Reports MAE, median, p90 absolute error, and the depth_valid rate.
  * runtime: mean of /perception/metrics per node (FPS, processing ms, sim-time latency,
    CPU %, RSS).
Writes a markdown report (--out).

    eval_perception.py --gt worlds/lunar_crater_ground_truth.json --duration 120 --out eval.md
"""
import argparse
import json
import math
import time
from collections import defaultdict, deque

import numpy as np
import rclpy
from crater_interfaces.msg import RockDetectionArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

GZ_LABEL = {'4': 'rock', '1': 'meteor', '2': 'crater', '3': 'station'}
CLASSES = ['rock', 'meteor', 'crater', 'station']
CAM_OFFSET = np.array([0.35, 0.0, 0.50 + 0.23])     # camera above the GROUND (base_link is 0.23 m up)
CAM_PITCH = -0.2


def key(st):
    return st.sec * 1_000_000_000 + st.nanosec


def boxes(msg, mapping=None):
    out = []
    for d in msg.detections:
        if not d.results:
            continue
        c = d.results[0].hypothesis.class_id
        c = mapping.get(c) if mapping else c
        if c is None:
            continue
        b = d.bbox
        out.append((c, b.center.position.x - b.size_x / 2, b.center.position.y - b.size_y / 2,
                    b.center.position.x + b.size_x / 2, b.center.position.y + b.size_y / 2))
    return out


def iou(a, b):
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0.0, min(a[4], b[4]) - max(a[2], b[2]))
    inter = ix * iy
    ua = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


class Eval(Node):
    def __init__(self, args):
        super().__init__('eval_perception', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.args = args
        with open(args.gt) as f:
            gt = json.load(f)
        self.rocks = gt.get('rocks', [])
        self.rover_yaw0 = gt.get('rover_yaw', 0.0)
        self.gt_boxes = {}
        self.pending_dets = deque(maxlen=50)
        self.odom = deque(maxlen=200)
        self.K = None
        self.stats = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASSES}
        self.frames = 0
        self.depth_err, self.n_rocks, self.n_valid = [], 0, 0
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.t_start = None
        self.create_subscription(Detection2DArray, '/camera/bbox', self._gt_cb, qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, '/perception/detections_2d', lambda m: self.pending_dets.append(m), 10)
        self.create_subscription(RockDetectionArray, '/perception/rocks', self._rocks_cb, 10)
        self.create_subscription(String, '/perception/metrics', self._metrics_cb, 50)
        self.create_subscription(Odometry, '/ground_truth/odom', lambda m: self.odom.append(m), 50)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._info_cb, qos_profile_sensor_data)
        self.create_timer(0.1, self._match)

    def _info_cb(self, m):
        if self.K is None:
            self.K = np.array(m.k, float).reshape(3, 3)

    def _gt_cb(self, m):
        self.gt_boxes[key(m.header.stamp)] = boxes(m, GZ_LABEL)
        if len(self.gt_boxes) > 100:
            del self.gt_boxes[min(self.gt_boxes)]

    def _metrics_cb(self, m):
        d = json.loads(m.data)
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.metrics[d['node']][k].append(v)

    # ------------------------------------------------------------------ detection
    def _match(self):
        while self.pending_dets:
            det = self.pending_dets[0]
            k = key(det.header.stamp)
            gk = min(self.gt_boxes, key=lambda g: abs(g - k), default=None)
            if gk is None or abs(gk - k) > 5_000_000:
                if self.gt_boxes and max(self.gt_boxes) - k > 1_000_000_000:
                    self.pending_dets.popleft()         # its GT is gone
                    continue
                return
            self.pending_dets.popleft()
            if self.t_start is None:
                self.t_start, self.wall_start = k, time.monotonic()
            self.t_last, self.wall_last = k, time.monotonic()
            self.frames += 1
            m = self.args.min_box
            gts = [g for g in self.gt_boxes[gk] if g[3] - g[1] >= m and g[4] - g[2] >= m]
            preds = boxes(det)
            for c in CLASSES:
                P = [p for p in preds if p[0] == c]
                G = [g for g in gts if g[0] == c]
                used = set()
                for p in P:
                    best, bi = 0.0, -1
                    for i, g in enumerate(G):
                        if i not in used and (v := iou(p, g)) > best:
                            best, bi = v, i
                    if best >= 0.5:
                        used.add(bi)
                        self.stats[c]['tp'] += 1
                    else:
                        self.stats[c]['fp'] += 1
                self.stats[c]['fn'] += len(G) - len(used)

    # ------------------------------------------------------------------ depth
    def _cam_pose(self, stamp_ns):
        if not self.odom:
            return None
        o = min(self.odom, key=lambda m: abs(key(m.header.stamp) - stamp_ns))
        if abs(key(o.header.stamp) - stamp_ns) > 100_000_000:
            return None
        p, q = o.pose.pose.position, o.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return np.array([p.x, p.y]), yaw

    def _rocks_cb(self, msg):
        if self.K is None:
            return
        pose = self._cam_pose(key(msg.header.stamp))
        for d in msg.detections:
            self.n_rocks += 1
            self.n_valid += d.depth_valid
        if pose is None:
            return
        (rx, ry), yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        cam = np.array([rx + c * CAM_OFFSET[0], ry + s * CAM_OFFSET[0], CAM_OFFSET[2]])
        # world -> camera body (x fwd, y left, z up), then pitch, then optical (x right, y down, z fwd)
        cp, sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        proj = []
        for r in self.rocks:
            w = np.array([r['x'], r['y'], max(0.02, r['top_z'] / 2)]) - cam
            b = np.array([c * w[0] + s * w[1], -s * w[0] + c * w[1], w[2]])
            b = np.array([cp * b[0] - sp * b[2], b[1], sp * b[0] + cp * b[2]])    # R_y(pitch)^T b
            zc = b[0]
            if zc < 0.3:
                continue
            u, v = cx - fx * b[1] / zc, cy - fy * b[2] / zc
            proj.append((u, v, zc - (r['extent'][0] + r['extent'][1]) / 4))   # near face: centre - mean radius
        for d in msg.detections:
            if not d.depth_valid:
                continue
            x0, x1 = d.pixel_u - d.bbox_w / 2, d.pixel_u + d.bbox_w / 2
            y0, y1 = d.pixel_v - d.bbox_h / 2, d.pixel_v + d.bbox_h / 2
            inside = [p for p in proj if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
            if inside:
                g = min(inside, key=lambda p: abs(p[2] - d.depth_m))
                self.depth_err.append((d.depth_m - g[2], g[2]))

    # ------------------------------------------------------------------ report
    def _rtf(self):
        if self.t_start is None or self.wall_last - self.wall_start < 1:
            return 'n/a'
        return f'{(self.t_last - self.t_start) / 1e9 / (self.wall_last - self.wall_start):.2f}'

    def report(self):
        L = [f'# Module 2 perception: measured evaluation\n',
             f'World: `{self.args.gt}`, {self.frames} frames evaluated '
             f'(detections_2d paired with the Gazebo bounding-box ground truth).\n',
             f'Simulation real-time factor during the run: {self._rtf()} (FPS below are wall-clock).\n',
             '## Detection (IoU >= 0.5)\n', '| class | TP | FP | FN | precision | recall |', '|---|---|---|---|---|---|']
        for c in CLASSES:
            st = self.stats[c]
            p = st['tp'] / (st['tp'] + st['fp']) if st['tp'] + st['fp'] else float('nan')
            r = st['tp'] / (st['tp'] + st['fn']) if st['tp'] + st['fn'] else float('nan')
            L.append(f"| {c} | {st['tp']} | {st['fp']} | {st['fn']} | {p:.3f} | {r:.3f} |")
        L += ['', '## Depth fusion (rocks)\n']
        if self.depth_err:
            e = np.array([x for x, _ in self.depth_err])
            z = np.array([g for _, g in self.depth_err])
            a = np.abs(e)
            L += ['| metric | value |', '|---|---|',
                  f'| rock detections | {self.n_rocks} |',
                  f'| depth_valid rate | {self.n_valid / max(1, self.n_rocks):.3f} |',
                  f'| matched to a GT rock | {len(e)} |',
                  f'| MAE | {a.mean():.3f} m |', f'| median abs error | {np.median(a):.3f} m |',
                  f'| p90 abs error | {np.percentile(a, 90):.3f} m |',
                  f'| bias (mean signed) | {e.mean():+.3f} m |',
                  f'| relative MAE | {np.mean(a / z) * 100:.1f} % |']
        else:
            L.append(f'No rock with valid depth matched a ground-truth rock ({self.n_rocks} rock detections).')
        L += ['', '## Runtime (measured, /perception/metrics means)\n',
              '| node | FPS | proc ms | p95 ms | infer ms | sim latency ms | CPU % | RSS MB |', '|---|---|---|---|---|---|---|---|']
        for node, m in sorted(self.metrics.items()):
            f = lambda k: f'{np.mean(m[k]):.1f}' if m.get(k) else '-'  # noqa: E731
            L.append(f"| {node} | {f('fps')} | {f('proc_ms')} | {f('proc_p95_ms')} | {f('infer_ms')} | "
                     f"{f('latency_ms')} | {f('cpu_pct')} | {f('rss_mb')} |")
        sync = self.metrics.get('rgbd_sync_node', {})
        if sync.get('match_ratio'):
            L.append(f"\nRGB-D sync: {sync['match_ratio'][-1] * 100:.1f} % of RGB frames matched with depth; "
                     f"{int(sync['rgb_without_depth'][-1])} RGB frames without depth dropped.")
        L.append('\nGPU: none (CPU-only machine).')
        return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--duration', type=float, default=120.0, help='sim seconds')
    ap.add_argument('--min-box', type=float, default=6.0)
    ap.add_argument('--out', default='eval_perception.md')
    ap.add_argument('--wall-timeout', type=float, default=1800.0, help='stop and report after this many real seconds')
    args = ap.parse_args()
    rclpy.init()
    node = Eval(args)
    t_wall = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - t_wall < args.wall_timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.t_start and (node.get_clock().now().nanoseconds - node.t_start) / 1e9 > args.duration:
                break
    except KeyboardInterrupt:
        pass
    rep = node.report()
    with open(args.out, 'w') as f:
        f.write(rep)
    print(rep)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
