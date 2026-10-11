#!/usr/bin/env python3
"""Auto-labelled YOLO dataset capture (training only).

Gazebo's bounding-box camera has the same pose and intrinsics as the rover's RGB camera,
and every rock / meteor / crater / station carries a semantic label, so each frame comes
with exact boxes for free. This node pairs every RGB frame with the bbox message rendered at
the same sim time (paired when the bbox message arrives, since it comes after the image)
and writes
    <out>/images/<prefix>_<n>.jpg
    <out>/labels/<prefix>_<n>.txt     YOLO format: class cx cy w h (normalised)
classes: 0 rock, 1 meteor, 2 crater, 3 station.

The rover moves on its own (patrol + evasion); run the sim with a short meteor_period.
With --gt (the world's ground-truth json) the rover is also teleported to a random clear
spot and heading every --teleport-every saved frames, for many more viewpoints per world.
Frames with neither a rock nor a meteor are kept with probability --keep-empty.

    collect_dataset.py --out ~/crater_ws/dataset/raw/train --prefix s101 --count 180 \
        --gt ~/crater_ws/dataset/worlds/s101_ground_truth.json
"""
import argparse
import json
import math
import os
import random
from collections import deque

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

GZ_LABEL_TO_YOLO = {4: 0, 1: 1, 2: 2, 3: 3}      # gz label -> rock, meteor, crater, station
ROCK, METEOR = 0, 1
MIN_BOX_PX = 4


def t_of(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Collector(Node):
    def __init__(self, args):
        super().__init__('dataset_collector', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.args = args
        self.rng = random.Random(args.seed)
        self.bridge = CvBridge()
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(args.out, d), exist_ok=True)
        self.img_buf = deque(maxlen=10)
        self.saved, self.with_meteor, self.with_rock, self.last_t = 0, 0, 0, -1e9
        self.hold_until = -1e9               # after a teleport: wait for the rover to settle
        self.gt = None
        if args.gt:
            with open(args.gt) as f:
                self.gt = json.load(f)
            from gz.transport13 import Node as GzNode
            self.gz = GzNode()
        self.create_subscription(Image, '/camera/rgb/image_raw', lambda m: self.img_buf.append(m), qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, '/camera/bbox', self._bbox_cb, qos_profile_sensor_data)

    def _bbox_cb(self, bbox):
        tb = t_of(bbox.header.stamp)
        msg = next((m for m in self.img_buf if abs(t_of(m.header.stamp) - tb) < 0.005), None)
        if msg is None or tb - self.last_t < self.args.period or tb < self.hold_until:
            return
        W, H = msg.width, msg.height
        lines, has_meteor, has_rock = [], False, False
        for d in bbox.detections:
            if not d.results:
                continue
            cls = GZ_LABEL_TO_YOLO.get(int(d.results[0].hypothesis.class_id))
            b = d.bbox
            x0 = max(0.0, b.center.position.x - b.size_x / 2)
            y0 = max(0.0, b.center.position.y - b.size_y / 2)
            x1 = min(float(W), b.center.position.x + b.size_x / 2)
            y1 = min(float(H), b.center.position.y + b.size_y / 2)
            if cls is None or x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
                continue
            has_meteor |= cls == METEOR
            has_rock |= cls == ROCK
            lines.append(f'{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}')
        if not (has_meteor or has_rock) and self.rng.random() > self.args.keep_empty:
            return
        self.last_t = tb
        name = f'{self.args.prefix}_{self.saved:05d}'
        cv2.imwrite(os.path.join(self.args.out, 'images', name + '.jpg'),
                    self.bridge.imgmsg_to_cv2(msg, 'bgr8'), [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(self.args.out, 'labels', name + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        self.saved += 1
        self.with_meteor += has_meteor
        self.with_rock += has_rock
        if self.saved % 25 == 0:
            self.get_logger().info(f'{self.saved}/{self.args.count} frames '
                                   f'({self.with_rock} with rocks, {self.with_meteor} with a meteor)')
        if self.gt and self.saved % self.args.teleport_every == 0:
            self._teleport(tb)

    def _teleport(self, t_now):
        """Move the rover to a random spot clear of rocks, the station and crater rims,
        usually facing a nearby rock."""
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_pb2 import Pose
        rocks, craters = self.gt.get('rocks', []), self.gt.get('craters', [])
        sx, sy = self.gt['station']
        for _ in range(200):
            a, d = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(0, 18)
            x, y = d * math.cos(a), d * math.sin(a)
            if (math.hypot(x - sx, y - sy) > 2.0
                    and all(math.hypot(x - r['x'], y - r['y']) > 0.9 + max(r['extent'][:2]) / 2 for r in rocks)
                    and all(abs(math.hypot(x - c['x'], y - c['y']) - c['radius']) > 0.8 for c in craters)):
                break
        else:
            return
        yaw = self.rng.uniform(-math.pi, math.pi)
        # 70 %: face a rock 2-9 m away (close-range rocks are rare in a uniform field)
        near = [r for r in rocks if 2.0 < math.hypot(r['x'] - x, r['y'] - y) < 9.0]
        if near and self.rng.random() < 0.7:
            r = self.rng.choice(near)
            yaw = math.atan2(r['y'] - y, r['x'] - x) + math.radians(self.rng.uniform(-35, 35))
        req = Pose()
        req.name = 'crater_rover'
        req.position.x, req.position.y, req.position.z = x, y, 0.26
        req.orientation.z, req.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        try:
            ok, rep = self.gz.request(f'/world/{self.args.world}/set_pose', req, Pose, Boolean, 2000)
            ok = ok and rep.data
        except Exception as e:  # noqa: BLE001 - keep collecting from the current pose
            ok = False
            self.get_logger().warn(f'teleport error: {e}')
        self.get_logger().info(f'teleport to ({x:.1f}, {y:.1f}, yaw {math.degrees(yaw):.0f} deg): '
                               f'{"ok" if ok else "FAILED"}')
        self.hold_until = t_now + 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', default='s0')
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--period', type=float, default=0.3, help='min sim-seconds between saved frames')
    ap.add_argument('--keep-empty', type=float, default=0.2, help='keep prob. of frames with no rock/meteor')
    ap.add_argument('--gt', default='', help="world ground-truth json: enables teleporting the rover")
    ap.add_argument('--teleport-every', type=int, default=8, help='saved frames between teleports')
    ap.add_argument('--world', default='lunar_crater')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rclpy.init()
    node = Collector(args)
    try:
        while rclpy.ok() and node.saved < args.count:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.get_logger().info(f'done: {node.saved} frames ({node.with_rock} with rocks, '
                           f'{node.with_meteor} with a meteor) in {args.out}')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
