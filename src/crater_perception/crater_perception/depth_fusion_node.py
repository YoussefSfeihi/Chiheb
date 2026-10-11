#!/usr/bin/env python3
"""depth_fusion_node (Module 2, stage 3): 2D detections + depth -> RockDetectionArray.

    /perception/detections_2d ┐  ApproximateTimeSynchronizer   ┌> /perception/rocks
    /perception/sync/depth    ┘  (same RGB stamp, slop 5 ms)   ┘   crater_interfaces/RockDetectionArray

For each detection of a kept class (default: rock), the depth is the MEDIAN over the
central part of its box, after rejecting NaN, +-Inf, <= min_range and >= max_range
pixels (see depth_utils.roi_depth). Too few valid pixels -> depth_valid = false,
depth_m = NaN (the detection is still reported).

Output is image-space only (class, confidence, pixel centre, box size, depth): Module 2
never computes global coordinates. An empty array means "no rock in this frame".
"""
import time

import message_filters
import rclpy
from crater_interfaces.msg import RockDetection, RockDetectionArray
from cv_bridge import CvBridge
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from crater_perception.depth_utils import depth_to_metres, roi_depth
from crater_perception.metrics import Metrics


class DepthFusion(Node):
    def __init__(self):
        super().__init__('depth_fusion_node')
        for n, v in [('classes', ['rock']), ('roi_fraction', 0.6), ('min_range', 0.1), ('max_range', 60.0),
                     ('min_valid_ratio', 0.15), ('min_valid_px', 20), ('slop', 0.005), ('queue_size', 30)]:
            self.declare_parameter(n, v)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.classes = set(p('classes'))
        self.kw = dict(roi_frac=p('roi_fraction'), min_range=p('min_range'), max_range=p('max_range'),
                       min_valid_ratio=p('min_valid_ratio'), min_valid_px=p('min_valid_px'))
        self.bridge = CvBridge()
        det = message_filters.Subscriber(self, Detection2DArray, '/perception/detections_2d')
        dep = message_filters.Subscriber(self, Image, '/perception/sync/depth',
                                         qos_profile=QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        # depth arrives first and waits for YOLO: the queue covers ~2 s of frames
        self.sync = message_filters.ApproximateTimeSynchronizer([det, dep], p('queue_size'), p('slop'))
        self.sync.registerCallback(self._fuse)
        self.pub = self.create_publisher(RockDetectionArray, '/perception/rocks', 10)
        self.metrics = Metrics(self, 'depth_fusion_node')
        self.n_det = self.n_valid = 0
        self.create_timer(10.0, self._log)

    def _fuse(self, dets, depth_msg):
        t0 = time.perf_counter()
        out = RockDetectionArray(header=dets.header)
        kept = [d for d in dets.detections if d.results and d.results[0].hypothesis.class_id in self.classes]
        if kept:
            depth = depth_to_metres(self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough'), depth_msg.encoding)
            for d in kept:
                b = d.bbox
                cx, cy = b.center.position.x, b.center.position.y
                box = (cx - b.size_x / 2, cy - b.size_y / 2, cx + b.size_x / 2, cy + b.size_y / 2)
                z, valid, _n, _ratio = roi_depth(depth, box, **self.kw)
                out.detections.append(RockDetection(
                    class_id=d.results[0].hypothesis.class_id, confidence=float(d.results[0].hypothesis.score),
                    pixel_u=float(cx), pixel_v=float(cy), depth_m=float(z), depth_valid=bool(valid),
                    bbox_w=float(b.size_x), bbox_h=float(b.size_y)))
                self.n_det += 1
                self.n_valid += valid
        self.pub.publish(out)
        self.metrics.tick((time.perf_counter() - t0) * 1000, dets.header.stamp)
        self.metrics.extra = {'rocks_total': self.n_det, 'depth_valid_ratio':
                              round(self.n_valid / self.n_det, 3) if self.n_det else None}

    def _log(self):
        m = getattr(self.metrics, 'last', None)
        if m and m['fps'] > 0:
            self.get_logger().info(f'perf {m}')
        else:
            self.get_logger().warn('no synchronised (detections_2d, depth) pairs in the last second')


def main(args=None):
    rclpy.init(args=args)
    node = DepthFusion()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
