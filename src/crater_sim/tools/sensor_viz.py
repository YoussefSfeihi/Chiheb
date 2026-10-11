#!/usr/bin/env python3
"""sensor_viz: draw the rover's RGB-D camera in the Gazebo GUI (visualization only).

  * field-of-view frustum: the camera's real FOV (100 deg x 83.6 deg, tilted 11 deg up),
    from the lens out to `viz_range`. Translucent cyan faces, brighter edges.
  * scan pulse: two staggered rectangular wavefronts sweep from the lens to `viz_range`
    every `pulse_period` seconds, fading as they travel, like a sensor ping.
  * lock lines: a line from the lens to every meteor the rover is tracking
    (/meteors/tracks) with a "M3 18.4 m" tag; red = threat, amber = no threat.

The rover pose comes from Gazebo's own pose stream so the drawing stays glued to the
3D model. Markers go through the GUI's /marker_array service at `rate` Hz; without a
GUI (headless) the service is absent and this node just idles.
"""
import math
import threading
import time

import numpy as np
import rclpy
from crater_interfaces.msg import MeteorTrackArray
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.marker_pb2 import Marker
from gz.msgs10.marker_v_pb2 import Marker_V
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

# must match models/rover/model.sdf (rgbd_camera) and the launch file's SENSOR_TF
CAM_OFFSET = np.array([0.35, 0.0, 0.50])
CAM_PITCH = -0.2                       # rad, negative = nose up
HFOV = 1.745
VFOV = 2 * math.atan(math.tan(HFOV / 2) * 480 / 640)
CYAN = (0.2, 0.85, 1.0)


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class SensorViz(Node):
    def __init__(self):
        super().__init__('sensor_viz')
        for n, v in [('world', 'lunar_crater'), ('model', 'crater_rover'), ('viz_range', 25.0),
                     ('pulse_period', 1.6), ('rate', 20.0)]:
            self.declare_parameter(n, v)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.world, self.model = p('world'), p('model')
        self.range, self.period = float(p('viz_range')), float(p('pulse_period'))

        self.gz = GzNode()
        self.lock = threading.Lock()
        self.pose = None                 # (R 3x3, t 3) of the rover in world
        self.map_to_world = None         # (R, t): map frame = rover's start pose
        self.tracks = []
        self.gz.subscribe(Pose_V, f'/world/{self.world}/dynamic_pose/info', self._poses)
        self.create_subscription(MeteorTrackArray, '/meteors/tracks', lambda m: setattr(self, 'tracks', m.tracks), 10)
        self.t0 = time.time()
        self.gui_ok, self.next_try = False, 0.0
        self.create_timer(1.0 / float(p('rate')), self._draw)

    def _poses(self, msg):
        for ps in msg.pose:
            if ps.name == self.model:
                q, t = ps.orientation, ps.position
                R, tv = quat_to_rot(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
                with self.lock:
                    self.pose = (R, tv)
                    if self.map_to_world is None:       # odometry starts at the spawn pose
                        yaw = math.atan2(R[1, 0], R[0, 0])
                        self.map_to_world = (np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                                       [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]),
                                             np.array([t.x, t.y, 0.0]))
                return

    # ------------------------------------------------------------------ marker helpers
    def _marker(self, ns, mid, mtype, rgba, lifetime=0.0):
        m = Marker()
        m.ns, m.id, m.action, m.type = ns, mid, Marker.ADD_MODIFY, mtype
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        for c in (m.material.ambient, m.material.diffuse, m.material.emissive):
            c.r, c.g, c.b, c.a = rgba
        if lifetime:
            m.lifetime.sec, m.lifetime.nsec = int(lifetime), int((lifetime % 1) * 1e9)
        return m

    @staticmethod
    def _pts(m, pts):
        for x, y, z in pts:
            pt = m.point.add()
            pt.x, pt.y, pt.z = float(x), float(y), float(z)

    # ------------------------------------------------------------------ drawing
    def _draw(self):
        now = time.time()
        if not self.gui_ok and now < self.next_try:
            return
        with self.lock:
            if self.pose is None:
                return
            R, t = self.pose
            m2w = self.map_to_world
        Rc = R @ rot_y(CAM_PITCH)                       # camera frame: x forward, y left, z up
        lens = R @ CAM_OFFSET + t
        hx, hy = math.tan(HFOV / 2), math.tan(VFOV / 2)
        dirs = [Rc @ np.array([1.0, sy * hx, sz * hy]) for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        far = [lens + d * self.range for d in dirs]

        mv = Marker_V()
        # frustum faces (translucent) and edges
        faces = self._marker('rgbd_fov', 1, Marker.TRIANGLE_LIST, (*CYAN, 0.07))
        for i in range(4):
            self._pts(faces, [lens, far[i], far[(i + 1) % 4]])
        edges = self._marker('rgbd_fov', 2, Marker.LINE_LIST, (*CYAN, 0.55))
        for i in range(4):
            self._pts(edges, [lens, far[i], far[i], far[(i + 1) % 4]])
        mv.marker.extend([faces, edges])
        # scan pulses: two wavefronts half a period apart
        for k in range(2):
            frac = ((now - self.t0) / self.period + k / 2) % 1.0
            d = max(0.3, frac * self.range)
            ring = [lens + v * d for v in dirs]
            fade = (1 - frac) ** 1.5
            line = self._marker('rgbd_pulse', 10 + k, Marker.LINE_STRIP, (0.5, 1.0, 1.0, 0.95 * fade))
            self._pts(line, ring + [ring[0]])
            plane = self._marker('rgbd_pulse', 20 + k, Marker.TRIANGLE_LIST, (*CYAN, 0.18 * fade))
            self._pts(plane, [ring[0], ring[1], ring[2], ring[0], ring[2], ring[3]])
            mv.marker.extend([line, plane])
        # lock lines to tracked meteors (map frame -> world)
        for tr in list(self.tracks):
            pw = m2w[0] @ np.array([tr.position.x, tr.position.y, tr.position.z]) + m2w[1] if m2w else None
            if pw is None:
                continue
            col = (1.0, 0.2, 0.15, 0.95) if tr.is_threat else (1.0, 0.7, 0.1, 0.9)
            ln = self._marker('locks', tr.id, Marker.LINE_LIST, col, lifetime=0.3)
            self._pts(ln, [lens, pw])
            txt = self._marker('lock_text', tr.id, Marker.TEXT, col, lifetime=0.3)
            txt.text = f'M{tr.id}  {np.linalg.norm(pw - lens):.1f} m' + ('  THREAT' if tr.is_threat else '')
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(pw[0]), float(pw[1]), float(pw[2]) + 1.0
            txt.scale.x = txt.scale.y = txt.scale.z = 0.8
            mv.marker.extend([ln, txt])

        try:
            ok, _rep = self.gz.request('/marker_array', mv, Marker_V, Boolean, 200)
        except Exception:
            ok = False
        if ok and not self.gui_ok:
            self.get_logger().info('drawing the RGB-D camera frustum + scan pulse in the Gazebo GUI')
        if not ok:
            self.next_try = now + 2.0                    # no GUI (headless) or not up yet: retry later
        self.gui_ok = ok


def main(args=None):
    rclpy.init(args=args)
    node = SensorViz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
