#!/usr/bin/env python3
"""rgbd_sync_node (Module 2, stage 1): time-aligned RGB + Depth + CameraInfo.

    /camera/rgb/image_raw  ┐
    /camera/depth/image_raw├─ ApproximateTimeSynchronizer ─> /perception/sync/rgb
    /camera/camera_info    ┘   (queue 10, slop 30 ms,         /perception/sync/depth
                                use_sim_time)                 /perception/sync/camera_info

The three outputs of one match carry the RGB frame's header, so downstream nodes can
pair them by exact stamp. An RGB frame without its depth (or vice versa) is never
forwarded: it is dropped and counted (metrics: rgb_without_depth, depth_without_rgb).
"""
import message_filters
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from crater_perception.metrics import Metrics


class RgbdSync(Node):
    def __init__(self):
        super().__init__('rgbd_sync_node')
        for n, v in [('rgb_topic', '/camera/rgb/image_raw'), ('depth_topic', '/camera/depth/image_raw'),
                     ('info_topic', '/camera/camera_info'), ('slop', 0.03), ('queue_size', 10)]:
            self.declare_parameter(n, v)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        # reliable: best-effort drops most 1.2 MB depth frames (UDP fragments) under CPU load
        q = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.rx = {'rgb': 0, 'depth': 0}
        subs = []
        for key, typ, topic in (('rgb', Image, p('rgb_topic')), ('depth', Image, p('depth_topic')),
                                ('info', CameraInfo, p('info_topic'))):
            s = message_filters.Subscriber(self, typ, topic, qos_profile=q)
            if key in self.rx:
                s.registerCallback(lambda _m, k=key: self.rx.__setitem__(k, self.rx[k] + 1))
            subs.append(s)
        self.sync = message_filters.ApproximateTimeSynchronizer(subs, p('queue_size'), p('slop'))
        self.sync.registerCallback(self._matched)
        out_q = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)   # also serves best-effort subs
        self.pub_rgb = self.create_publisher(Image, '/perception/sync/rgb', out_q)
        self.pub_depth = self.create_publisher(Image, '/perception/sync/depth', out_q)
        self.pub_info = self.create_publisher(CameraInfo, '/perception/sync/camera_info', out_q)
        self.matched = 0
        self.metrics = Metrics(self, 'rgbd_sync_node')
        self.create_timer(1.0, self._stats)
        self.create_timer(10.0, self._log)
        self.get_logger().info(f'syncing {p("rgb_topic")} + {p("depth_topic")} + {p("info_topic")} '
                               f'(slop {p("slop")} s, use_sim_time={self.get_parameter("use_sim_time").value})')

    def _matched(self, rgb, depth, info):
        self.matched += 1
        depth.header = rgb.header
        info.header = rgb.header
        self.pub_rgb.publish(rgb)
        self.pub_depth.publish(depth)
        self.pub_info.publish(info)
        self.metrics.tick(stamp=rgb.header.stamp)

    def _stats(self):
        self.metrics.extra = {
            'rgb_rx': self.rx['rgb'], 'depth_rx': self.rx['depth'], 'matched': self.matched,
            'rgb_without_depth': max(0, self.rx['rgb'] - self.matched),
            'depth_without_rgb': max(0, self.rx['depth'] - self.matched),
            'match_ratio': round(self.matched / self.rx['rgb'], 3) if self.rx['rgb'] else None}

    def _log(self):
        e = self.metrics.extra
        if e.get('rgb_rx'):
            self.get_logger().info(f"sync: {e['matched']}/{e['rgb_rx']} RGB frames matched "
                                   f"({100 * e['match_ratio']:.1f} %), {e['rgb_without_depth']} RGB without depth")
        else:
            self.get_logger().warn('no RGB frames yet on the camera topic')


def main(args=None):
    rclpy.init(args=args)
    node = RgbdSync()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
