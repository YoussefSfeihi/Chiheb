#!/usr/bin/env python3
"""meteor_tracker: onboard detection of incoming objects with the RGB-D camera.

Only the rover's own sensors are used (depth image + camera intrinsics + TF from the
rover's dead-reckoned pose). No ground truth.

Per depth frame (15 Hz):
 1. Downsample x2, back-project every valid pixel with the pinhole model
        p_cam = Z * K^-1 [u v 1]^T,   p_map = R p_cam + t     (R,t at the frame's stamp)
 2. Keep points more than `min_height` above the ground. The ground, crater rims and
    rocks lying on the surface fall below it; empty sky has no depth return.
 3. Connected components on that image mask -> one 3D blob per object; its centroid
    is the median point pushed back along the viewing ray by the blob's apparent radius
    (the camera sees the near surface).
 4. Nearest-neighbour association to tracks (gate grows with track speed).
 5. Ballistic least-squares fit over the last `fit_window` observations, with gravity known:
        x = x0 + vx t,   y = y0 + vy t,   z + g t^2 / 2 = z0 + vz t
    Moving test: the blob really travelled > `min_displacement` over the window AND
    |v| > `min_speed`. (A gravity-constrained fit invents ~1 m/s for a static blob such as
    the station, so speed alone is not enough.) Confirmed meteors keep their identity for
    up to `coast_timeout` s without observations, following their predicted arc.
 6. Impact prediction: solve z(t_i) = ground + r, giving the impact point (x(t_i), y(t_i))
    and time to impact. Its 1-sigma uncertainty is propagated from the fit covariance
    (position terms + timing error times horizontal speed).
 7. Threat test: the impact is within `danger_radius` (+ 2 sigma) of where the rover is
    now or where it will be at t_i (current velocity).

Publishes /meteors/tracks (crater_interfaces/MeteorTrackArray), /meteors/markers.
"""
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from builtin_interfaces.msg import Duration
from crater_interfaces.msg import MeteorTrack, MeteorTrackArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

G = 1.62  # lunar gravity [m/s^2]


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Track:
    _next = 1

    def __init__(self, t, p, r):
        self.id = Track._next
        Track._next += 1
        self.obs = deque(maxlen=60)       # (t, x, y, z)
        self.obs.append((t, *p))
        self.r = r
        self.first_seen = self.last_seen = t
        self.fit = None                   # dict once enough observations
        self.announced = False

    def add(self, t, p, r):
        self.obs.append((t, *p))
        self.r += (r - self.r) * 0.2
        self.last_seen = t

    def predict(self, t):
        if self.fit is None:
            return np.array(self.obs[-1][1:])
        f = self.fit
        dt = t - f['t0']
        return np.array([f['x'][0] + f['x'][1] * dt, f['y'][0] + f['y'][1] * dt,
                         f['z'][0] + f['z'][1] * dt - 0.5 * G * dt * dt])

    def refit(self, window):
        obs = np.array(list(self.obs)[-window:])
        if len(obs) < 4 or obs[-1, 0] - obs[0, 0] < 0.25:
            self.fit = None
            return
        t0 = obs[0, 0]
        t = obs[:, 0] - t0
        A = np.stack([np.ones_like(t), t], axis=1)
        fit = {'t0': t0, 'n': len(obs)}
        ata_inv = np.linalg.inv(A.T @ A)
        rms = []
        for k, col in (('x', 1), ('y', 2), ('z', 3)):
            b = obs[:, col] + (0.5 * G * t * t if k == 'z' else 0.0)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            res = b - A @ coef
            fit[k] = coef
            rms.append(math.sqrt(float(res @ res) / max(1, len(t) - 2)))
        fit['sigma'] = max(0.05, float(np.mean(rms)))     # per-axis position noise [m]
        fit['cov'] = ata_inv * fit['sigma'] ** 2         # 2x2 covariance of (p0, v)
        self.fit = fit

    def displacement(self, window):
        """How far the object really moved over the fit window [m]. A gravity-constrained
        fit always invents ~1 m/s of motion for a static blob, so this is the motion test."""
        obs = np.array(list(self.obs)[-window:])
        return float(np.linalg.norm(obs[-1, 1:] - obs[0, 1:]))

    def velocity(self, t):
        f = self.fit
        return np.array([f['x'][1], f['y'][1], f['z'][1] - G * (t - f['t0'])])

    def impact(self, t_now, ground_z):
        """-> (t_impact_abs, point_xy, sigma_xy) or None if not descending to the ground."""
        f = self.fit
        z0, vz = f['z']
        target = ground_z + self.r
        # z0 + vz s - G s^2 / 2 = target   (s measured from t0)
        disc = vz * vz - 2 * G * (target - z0)
        if disc < 0:
            return None
        s = (vz + math.sqrt(disc)) / G
        if s + f['t0'] < t_now - 0.2:
            return None
        x = f['x'][0] + f['x'][1] * s
        y = f['y'][0] + f['y'][1] * s
        c = f['cov']
        var_pos = c[0, 0] + 2 * s * c[0, 1] + s * s * c[1, 1]   # same for x and y
        vz_imp = abs(vz - G * s)
        sig_t = math.sqrt(var_pos) / max(vz_imp, 0.5)
        v_h = math.hypot(f['x'][1], f['y'][1])
        sigma = math.sqrt(var_pos + (v_h * sig_t) ** 2)
        return f['t0'] + s, np.array([x, y]), sigma


class MeteorTracker(Node):
    def __init__(self):
        super().__init__('meteor_tracker')
        defaults = [('stride', 2), ('min_height', 0.6), ('max_range', 55.0), ('min_speed', 1.0),
                    ('gate', 2.5), ('fit_window', 15), ('danger_radius', 3.0), ('lost_timeout', 1.0),
                    ('min_displacement', 1.0), ('coast_timeout', 4.0),
                    ('min_peak_altitude', 2.5), ('min_obs_announce', 6)]
        for n, v in defaults:
            self.declare_parameter(n, v)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.stride, self.min_h, self.max_range = p('stride'), p('min_height'), p('max_range')
        self.min_speed, self.gate, self.window = p('min_speed'), p('gate'), p('fit_window')
        self.danger, self.lost_timeout = p('danger_radius'), p('lost_timeout')
        self.min_disp, self.coast_timeout = p('min_displacement'), p('coast_timeout')
        self.min_peak, self.min_obs_announce = p('min_peak_altitude'), p('min_obs_announce')
        self.ground_z = 0.0

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.K = None
        self.rays = None
        self.pending = deque(maxlen=6)
        self.tracks = []
        self.rover_v = np.zeros(2)
        self.stats = {'frames': 0, 'ms': deque(maxlen=100)}

        self.create_subscription(CameraInfo, '/camera/camera_info', self._info_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_raw', self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.tracks_pub = self.create_publisher(MeteorTrackArray, '/meteors/tracks', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/meteors/markers', 10)
        self.create_timer(0.05, self._drain)
        self.create_timer(10.0, self._log_perf)

    # ------------------------------------------------------------------ inputs
    def _info_cb(self, msg):
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        s = self.stride
        v, u = np.mgrid[0:msg.height:s, 0:msg.width:s].astype(np.float32)
        self.rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)   # (h, w, 3)
        self.get_logger().info(f'RGB-D intrinsics fx={fx:.1f}; processing {self.rays.shape[1]}x{self.rays.shape[0]} px')

    def _odom_cb(self, m):
        # body-frame forward speed -> map-frame velocity (map = odom here)
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        v = m.twist.twist.linear.x
        self.rover_v = np.array([v * math.cos(yaw), v * math.sin(yaw)])

    def _depth_cb(self, msg):
        if self.K is not None:
            self.pending.append(msg)

    def _drain(self):
        """Process queued frames once their pose TF exists (never block a callback)."""
        now = self.get_clock().now()
        while self.pending:
            msg = self.pending[0]
            stamp = Time.from_msg(msg.header.stamp)
            if self.tf_buffer.can_transform('map', msg.header.frame_id, stamp):
                self.pending.popleft()
                self._process(msg)
            elif (now - stamp).nanoseconds > 1e9:
                self.pending.popleft()
            else:
                break

    # ------------------------------------------------------------------ core
    def _process(self, msg):
        t_wall = time.perf_counter()
        stamp = Time.from_msg(msg.header.stamp)
        t = stamp.nanoseconds * 1e-9
        try:
            tf = self.tf_buffer.lookup_transform('map', msg.header.frame_id, stamp)
            base = self.tf_buffer.lookup_transform('map', 'base_footprint', stamp).transform.translation
        except TransformException:
            return
        R = quat_to_rot(tf.transform.rotation)
        tvec = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        ground_z = self.ground_z = base.z
        rover_xy = np.array([base.x, base.y])

        depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')[::self.stride, ::self.stride]
        valid = np.isfinite(depth) & (depth > 0.3) & (depth < self.max_range)
        pts = (self.rays * np.where(valid, depth, 0.0)[..., None]) @ R.T + tvec        # (h, w, 3) in map
        mask = valid & (pts[..., 2] > ground_z + self.min_h)

        dets = []
        if mask.any():
            n, labels, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            fx = self.K[0, 0] / self.stride
            for i in range(1, n):
                sel = labels == i
                blob = pts[sel]
                c = np.median(blob, axis=0)
                Z = float(np.median(depth[sel]))
                r = 0.5 * max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) * Z / fx
                ray = c - tvec
                c = c + r * ray / (np.linalg.norm(ray) + 1e-9)      # near surface -> centre
                dets.append((c, r))

        self._associate(t, dets)
        self._publish(msg.header.stamp, t, ground_z, rover_xy)
        self.stats['frames'] += 1
        self.stats['ms'].append((time.perf_counter() - t_wall) * 1e3)

    def _associate(self, t, dets):
        free = list(range(len(dets)))
        for tr in sorted(self.tracks, key=lambda k: -len(k.obs)):
            if not free:
                break
            pred = tr.predict(t)
            speed = np.linalg.norm(tr.velocity(t)) if tr.fit else 5.0
            gate = self.gate + 0.15 * speed
            j = min(free, key=lambda k: np.linalg.norm(dets[k][0] - pred))
            if np.linalg.norm(dets[j][0] - pred) < gate:
                tr.add(t, dets[j][0], dets[j][1])
                tr.refit(self.window)
                free.remove(j)
        for j in free:
            self.tracks.append(Track(t, dets[j][0], dets[j][1]))
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen < self._keep_alive(tr, t)]

    def _keep_alive(self, tr, t):
        """Static/unfitted blobs expire fast. A confirmed meteor coasts on its ballistic
        prediction (e.g. while the rover turns away to evade) until it should have landed."""
        if tr.fit is None or not self._is_moving(tr, t):
            return self.lost_timeout
        return self.coast_timeout

    def _is_moving(self, tr, t):
        """Incoming object = really moving AND seen high in the sky at some point (taller than
        anything standing on the surface, e.g. the 1.9 m station)."""
        peak = max(o[3] for o in tr.obs) - self.ground_z
        return (peak > self.min_peak and tr.displacement(self.window) > self.min_disp
                and np.linalg.norm(tr.velocity(t)) > self.min_speed)

    # ------------------------------------------------------------------ outputs
    def _publish(self, stamp, t, ground_z, rover_xy):
        header = Header(stamp=stamp, frame_id='map')
        out = MeteorTrackArray(header=header)
        markers = MarkerArray(markers=[Marker(header=header, action=Marker.DELETEALL)])
        for tr in self.tracks:
            if tr.fit is None:
                continue
            if not self._is_moving(tr, t):
                continue                                   # static object (e.g. the station)
            v = tr.velocity(t)
            imp = tr.impact(t, ground_z)
            pos = tr.predict(t)
            if pos[2] < ground_z + tr.r - 0.05:
                continue                                   # already landed: the prediction is underground
            m = MeteorTrack(id=tr.id, n_obs=len(tr.obs))
            m.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            m.velocity.x, m.velocity.y, m.velocity.z = (float(c) for c in v)
            m.first_seen = Time(seconds=tr.first_seen).to_msg()
            m.last_seen = Time(seconds=tr.last_seen).to_msg()
            if imp is None:
                m.time_to_impact, m.miss_distance = -1.0, float('inf')
            else:
                t_imp, p_imp, sigma = imp
                tti = t_imp - t
                rover_then = rover_xy + self.rover_v * max(0.0, tti)
                miss = min(np.linalg.norm(p_imp - rover_xy), np.linalg.norm(p_imp - rover_then))
                m.impact_point = Point(x=float(p_imp[0]), y=float(p_imp[1]), z=float(ground_z))
                m.time_to_impact, m.impact_sigma, m.miss_distance = float(tti), float(sigma), float(miss)
                m.is_threat = bool(tti > 0 and miss - 2 * sigma < self.danger)
                if not tr.announced and len(tr.obs) >= self.min_obs_announce:
                    tr.announced = True
                    rng = float(np.linalg.norm(pos[:2] - rover_xy))
                    self.get_logger().info(
                        f'[RGB-D] object #{tr.id} detected: range {rng:.1f} m, altitude {pos[2] - ground_z:.1f} m, '
                        f'speed {np.linalg.norm(v):.1f} m/s, impact in {tti:.1f} s at ({p_imp[0]:.1f},{p_imp[1]:.1f}) '
                        f'±{sigma:.2f} m, miss {miss:.1f} m -> {"THREAT" if m.is_threat else "no threat"}')
            out.tracks.append(m)
            markers.markers.extend(self._markers(header, tr, m, t))
        self.tracks_pub.publish(out)
        self.marker_pub.publish(markers)

    def _markers(self, header, tr, m, t):
        life = Duration(nanosec=500_000_000)
        threat = m.is_threat
        col = (1.0, 0.15, 0.1) if threat else (1.0, 0.8, 0.1)
        out = []
        body = Marker(header=header, ns='meteor', id=tr.id, type=Marker.SPHERE, lifetime=life)
        body.pose.position = m.position
        body.pose.orientation.w = 1.0
        body.scale.x = body.scale.y = body.scale.z = max(0.4, 2 * tr.r)
        body.color.r, body.color.g, body.color.b, body.color.a = 1.0, 0.5, 0.1, 1.0
        out.append(body)
        if m.time_to_impact > 0:
            arc = Marker(header=header, ns='arc', id=tr.id, type=Marker.LINE_STRIP, lifetime=life)
            arc.scale.x = 0.06
            arc.color.r, arc.color.g, arc.color.b, arc.color.a = *col, 0.9
            for k in range(21):
                q = tr.predict(t + m.time_to_impact * k / 20)
                arc.points.append(Point(x=float(q[0]), y=float(q[1]), z=float(q[2])))
            out.append(arc)
            disk = Marker(header=header, ns='impact', id=tr.id, type=Marker.CYLINDER, lifetime=life)
            disk.pose.position = m.impact_point
            disk.pose.orientation.w = 1.0
            disk.scale.x = disk.scale.y = 2 * (1.5 + 2 * m.impact_sigma)   # blast radius + 2 sigma
            disk.scale.z = 0.05
            disk.color.r, disk.color.g, disk.color.b, disk.color.a = *col, 0.45
            out.append(disk)
            txt = Marker(header=header, ns='label', id=tr.id, type=Marker.TEXT_VIEW_FACING, lifetime=life)
            txt.pose.position = Point(x=m.position.x, y=m.position.y, z=m.position.z + 1.0)
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.6
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = *col, 1.0
            txt.text = f'#{tr.id} impact {m.time_to_impact:.1f}s  miss {m.miss_distance:.1f}m' + ('  THREAT' if threat else '')
            out.append(txt)
        return out

    def _log_perf(self):
        if self.stats['ms']:
            ms = np.array(self.stats['ms'])
            self.get_logger().info(f'perf: {self.stats["frames"] / 10.0:.1f} frames/s, '
                                   f'{ms.mean():.1f} ms/frame (p95 {np.percentile(ms, 95):.1f} ms), '
                                   f'{len(self.tracks)} blobs tracked')
            self.stats['frames'] = 0


def main(args=None):
    rclpy.init(args=args)
    node = MeteorTracker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
