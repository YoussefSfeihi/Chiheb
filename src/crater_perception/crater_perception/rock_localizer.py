#!/usr/bin/env python3
"""rock_localizer (Module 3 hook, NOT Module 2): image-space rocks -> map obstacles for Nav2.

Module 2 stops at (class, pixel, depth). This node does the 2D -> 3D step:
    p_cam = depth * K^-1 [u, v, 1]          (depth is the z of the optical frame)
    p_map = T_map<-camera_optical(stamp) p_cam      (tf2 at the frame's stamp)
and estimates the rock's footprint from its box (w = bbox_w * z / fx). Pebbles lower
than `min_rock_height` are ignored: the wheels roll over them.

Observations are merged into a persistent rock map (the lidar is ~0.6 m above the ground
and never sees most rocks, and the camera loses them once they're under the nose).
A rock confirmed by `min_obs` observations is published as
  /perception/rock_points   PointCloud2 (map) disc of points -> Nav2 hazard_layer
  /perception/rock_markers  MarkerArray for RViz
"""
import math
from collections import deque

import numpy as np
import rclpy
from crater_interfaces.msg import RockDetectionArray
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class RockLocalizer(Node):
    def __init__(self):
        super().__init__('rock_localizer')
        for n, v in [('max_range', 12.0), ('min_rock_height', 0.15), ('min_obs', 3), ('min_confidence', 0.45)]:
            self.declare_parameter(n, v)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.max_range, self.min_h = p('max_range'), p('min_rock_height')
        self.min_obs, self.min_conf = p('min_obs'), p('min_confidence')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.K = None
        self.pending = deque(maxlen=10)
        self.rocks = []                        # dicts: x, y, r, n
        self.create_subscription(CameraInfo, '/perception/sync/camera_info', self._info_cb, qos_profile_sensor_data)
        self.create_subscription(RockDetectionArray, '/perception/rocks', lambda m: self.pending.append(m), 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/perception/rock_points', 2)
        self.marker_pub = self.create_publisher(MarkerArray, '/perception/rock_markers', 2)
        self.create_timer(0.05, self._drain)
        self.create_timer(0.5, self._publish)
        self.create_timer(15.0, lambda: self.get_logger().info(
            f'rock map: {sum(r["n"] >= self.min_obs for r in self.rocks)} confirmed rocks '
            f'({len(self.rocks)} candidates)'))

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def _drain(self):
        now = self.get_clock().now()
        while self.pending and self.K is not None:
            msg = self.pending[0]
            stamp = Time.from_msg(msg.header.stamp)
            if self.tf_buffer.can_transform('map', msg.header.frame_id, stamp):
                self.pending.popleft()
                self._process(msg, stamp)
            elif (now - stamp).nanoseconds > 1e9:
                self.pending.popleft()
            else:
                break

    def _process(self, msg, stamp):
        try:
            tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id, stamp)
            ground_z = self.tf_buffer.lookup_transform('map', 'base_footprint', stamp).transform.translation.z
        except TransformException:
            return
        R = quat_to_rot(tf.transform.rotation)
        t = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        for d in msg.detections:
            if not d.depth_valid or d.depth_m > self.max_range or d.confidence < self.min_conf:
                continue
            z = d.depth_m
            width, height = d.bbox_w * z / fx, d.bbox_h * z / fy
            if height < self.min_h:
                continue
            p = R @ (z * np.array([(d.pixel_u - cx) / fx, (d.pixel_v - cy) / fy, 1.0])) + t
            if p[2] > ground_z + 1.5:
                continue                                  # not on the ground: not a rock
            ray = p[:2] - t[:2]
            n = np.linalg.norm(ray)
            r = min(1.0, max(0.1, width / 2))
            c = p[:2] + ray / n * r * 0.5 if n > 1e-6 else p[:2]   # surface -> approximate centre
            self._merge(c[0], c[1], r)

    def _merge(self, x, y, r):
        best = min(self.rocks, key=lambda k: math.hypot(k['x'] - x, k['y'] - y), default=None)
        if best is not None and math.hypot(best['x'] - x, best['y'] - y) < 0.35 + 0.5 * (best['r'] + r):
            w = min(best['n'], 20)
            best['x'] = (best['x'] * w + x) / (w + 1)
            best['y'] = (best['y'] * w + y) / (w + 1)
            best['r'] = (best['r'] * w + r) / (w + 1)
            best['n'] += 1
        else:
            self.rocks.append({'x': x, 'y': y, 'r': r, 'n': 1})

    def _publish(self):
        confirmed = [k for k in self.rocks if k['n'] >= self.min_obs]
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        pts = []
        for k in confirmed:
            for rr in np.arange(0.0, k['r'] + 1e-6, 0.1):
                m = max(1, int(2 * math.pi * rr / 0.1))
                pts += [(k['x'] + rr * math.cos(2 * math.pi * i / m), k['y'] + rr * math.sin(2 * math.pi * i / m), 0.1)
                        for i in range(m)]
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, pts))
        ma = MarkerArray()
        for i, k in enumerate(confirmed):
            mk = Marker(header=header, ns='rocks', id=i, type=Marker.CYLINDER, action=Marker.ADD)
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = k['x'], k['y'], 0.15
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = 2 * k['r']
            mk.scale.z = 0.3
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.35, 0.86, 0.47, 0.8
            mk.lifetime.sec = 2
            ma.markers.append(mk)
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = RockLocalizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
