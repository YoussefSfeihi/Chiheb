#!/usr/bin/env python3
"""rock_detector_node (Module 2, stage 2): YOLOv8n (ONNX, CPU) on the synced RGB frame.

Classes: rock, meteor, crater, station (rock is the Module 2 MVP class). The model was
trained on images auto-labelled by Gazebo's bounding-box camera (tools/make_dataset.sh +
tools/train_yolo.sh); at runtime only the camera image is used.

Newest frame only: frames that arrive while the network is busy are dropped, never queued.
Post-processing (OpenCV): low-confidence boxes dropped, class-aware NMS, boxes clipped to
the image, boxes smaller than `min_box_px` dropped.

  /perception/detections_2d  vision_msgs/Detection2DArray, header = the RGB frame's header
                             (an empty array = "nothing detected in this frame")
  /perception/debug_image    camera POV with a box + "<name> <conf> | <depth> m" per object.
                             Rock depths come from depth_fusion_node (/perception/rocks, same
                             stamp); meteors are fused with the RGB-D tracker (range, time to
                             impact). Shown in the Gazebo GUI's top-left panel.
  /perception/metrics        JSON (fps, inference ms, latency, CPU, memory)
"""
import math
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from crater_interfaces.msg import MeteorTrackArray, RockDetectionArray
from cv_bridge import CvBridge
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from crater_perception.metrics import Metrics
from crater_perception.yolo_onnx import Yolo

COLORS = {'rock': (90, 220, 120), 'meteor': (30, 140, 255), 'crater': (200, 200, 200),
          'station': (60, 200, 250)}   # BGR
DEBUG_WAIT_S = 0.3   # max wall time to wait for the depth of a frame before drawing without it


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def stamp_key(st):
    return st.sec * 1_000_000_000 + st.nanosec


class RockDetector(Node):
    def __init__(self):
        super().__init__('rock_detector_node')
        share = get_package_share_directory('crater_perception')
        for n, v in [('model_path', os.path.join(share, 'models', 'rock_yolo.onnx')), ('imgsz', 416),
                     ('conf_threshold', 0.35), ('iou_threshold', 0.5), ('min_box_px', 6.0), ('threads', 4),
                     ('rgb_topic', '/perception/sync/rgb')]:
            self.declare_parameter(n, v)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.conf, self.iou, self.min_box = p('conf_threshold'), p('iou_threshold'), p('min_box_px')
        self.model = None
        if os.path.exists(p('model_path')):
            self.model = Yolo(p('model_path'), p('imgsz'), p('threads'))
            self.get_logger().info(f'YOLOv8n loaded: {p("model_path")} classes={self.model.names} '
                                   f'(onnxruntime, CPU, {p("threads")} threads, conf>{self.conf})')
        else:
            self.get_logger().error(f'no model at {p("model_path")}: run tools/train_yolo.sh. '
                                    'Publishing empty detections and the raw view with a warning banner.')
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.K = None
        self.tracks = []
        self.latest, self.processed_stamp = None, None
        self.pending = {}                 # stamp -> (wall time, image, dets, track projections, ms)

        self.create_subscription(Image, p('rgb_topic'), lambda m: setattr(self, 'latest', m),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(CameraInfo, '/perception/sync/camera_info', self._info_cb, qos_profile_sensor_data)
        self.create_subscription(MeteorTrackArray, '/meteors/tracks', lambda m: setattr(self, 'tracks', m.tracks), 10)
        self.create_subscription(RockDetectionArray, '/perception/rocks', self._rocks_cb, 10)
        self.det_pub = self.create_publisher(Detection2DArray, '/perception/detections_2d', 10)
        # reliable: the ROS->Gazebo bridge subscribes reliably (also serves best-effort RViz)
        self.dbg_pub = self.create_publisher(Image, '/perception/debug_image',
                                             QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE))
        self.metrics = Metrics(self, 'rock_detector_node')
        self.create_timer(0.01, self._tick)
        self.create_timer(0.05, self._flush_debug)
        self.create_timer(10.0, self._log)

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    # ------------------------------------------------------------------ meteor fusion (RGB-D tracker)
    def _project_tracks(self, frame_id):
        if self.K is None or not self.tracks:
            return []
        try:
            tf = self.tf_buffer.lookup_transform(frame_id, 'map', Time())
        except TransformException:
            return []
        R = quat_to_rot(tf.transform.rotation)
        t = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        out = []
        for tr in self.tracks:
            pc = R @ np.array([tr.position.x, tr.position.y, tr.position.z]) + t
            if pc[2] <= 0.3:
                continue
            u = self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2]
            v = self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2]
            out.append((u, v, float(np.linalg.norm(pc)), tr.time_to_impact, tr.is_threat))
        return out

    # ------------------------------------------------------------------ detection
    def _tick(self):
        msg = self.latest
        if msg is None or msg.header.stamp == self.processed_stamp:
            return
        self.processed_stamp = msg.header.stamp
        t0 = time.perf_counter()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        dets, ms = [], None
        if self.model is not None:
            ti = time.perf_counter()
            dets = self.model(img, self.conf, self.iou, self.min_box)
            ms = (time.perf_counter() - ti) * 1000

        out = Detection2DArray(header=msg.header)
        for name, score, x0, y0, x1, y1 in dets:
            d = Detection2D(header=msg.header)
            d.id = name
            d.bbox.center.position.x, d.bbox.center.position.y = (x0 + x1) / 2, (y0 + y1) / 2
            d.bbox.size_x, d.bbox.size_y = x1 - x0, y1 - y0
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = name, score
            d.results.append(hyp)
            out.detections.append(d)
        self.det_pub.publish(out)
        self.metrics.tick((time.perf_counter() - t0) * 1000, msg.header.stamp)
        if ms is not None:
            self.metrics.extra = {'infer_ms': round(ms, 1), 'detections': len(dets)}
        # the debug view waits for this frame's rock depths (or DEBUG_WAIT_S)
        self.pending[stamp_key(msg.header.stamp)] = (time.monotonic(), msg.header, img, dets,
                                                     self._project_tracks(msg.header.frame_id), ms)
        if self.dbg_pub.get_subscription_count() == 0 and len(self.pending) > 3:
            self.pending.clear()

    def _rocks_cb(self, msg):
        item = self.pending.pop(stamp_key(msg.header.stamp), None)
        if item is not None:
            self._publish_debug(item, msg.detections)

    def _flush_debug(self):
        now = time.monotonic()
        for k in [k for k, v in self.pending.items() if now - v[0] > DEBUG_WAIT_S]:
            self._publish_debug(self.pending.pop(k), [])

    # ------------------------------------------------------------------ debug view
    def _publish_debug(self, item, rocks):
        _t, header, img, dets, proj, ms = item
        if self.dbg_pub.get_subscription_count() == 0:
            return
        self._draw(img, dets, proj, ms, rocks)
        ov = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        ov.header = header
        self.dbg_pub.publish(ov)

    def _draw(self, img, dets, proj, ms, rocks):
        H, W = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for name, score, x0, y0, x1, y1 in dets:
            col = COLORS.get(name, (255, 255, 255))
            label = f'{name} {score:.2f}'
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if name == 'rock':
                r = next((r for r in rocks if abs(r.pixel_u - cx) < 0.5 and abs(r.pixel_v - cy) < 0.5), None)
                if r is not None:
                    label += f' | {r.depth_m:.1f} m' if r.depth_valid else ' | depth ?'
            elif name == 'meteor':
                pad = 12
                inside = [q for q in proj if x0 - pad <= q[0] <= x1 + pad and y0 - pad <= q[1] <= y1 + pad]
                if inside:
                    u, v, rng, tti, threat = min(inside, key=lambda q: math.hypot(q[0] - cx, q[1] - cy))
                    label += f' | {rng:.1f} m' + (f' | impact {tti:.1f}s' if tti > 0 else '')
                    if threat:
                        col = (40, 40, 255)
                        label += ' | THREAT'
            # small objects: draw at least a 14 px box so they stay visible
            hw, hh = max(7.0, (x1 - x0) / 2), max(7.0, (y1 - y0) / 2)
            p0, p1 = (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh))
            cv2.rectangle(img, p0, p1, col, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
            ty = p0[1] - 4 if p0[1] - th - 8 > 26 else p1[1] + th + 6
            tx = min(max(0, p0[0]), W - tw - 6)
            cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 6, ty + 3), col, -1)
            cv2.putText(img, label, (tx + 3, ty), font, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
        # HUD
        cv2.rectangle(img, (0, 0), (W, 24), (15, 15, 15), -1)
        if self.model is None:
            cv2.putText(img, 'AI VIEW | NO MODEL - run tools/train_yolo.sh', (8, 17), font, 0.5,
                        (60, 60, 255), 1, cv2.LINE_AA)
        else:
            counts = {}
            for d in dets:
                counts[d[0]] = counts.get(d[0], 0) + 1
            seen = ', '.join(f'{v} {k}' for k, v in counts.items()) or 'clear'
            fps = getattr(self.metrics, 'last', {}).get('fps', 0.0)
            hud = f'AI VIEW | YOLOv8n ONNX CPU | {fps:4.1f} FPS | {ms:5.1f} ms | {seen}'
            cv2.putText(img, hud, (8, 17), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    def _log(self):
        m = getattr(self.metrics, 'last', None)
        if m:
            self.get_logger().info(f'perf {m}')


def main(args=None):
    rclpy.init(args=args)
    node = RockDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
