#!/usr/bin/env python3
"""mission_controller: patrol around the station, report position, dodge meteors.

States
  PATROL   loop survey waypoints through Nav2 (NavigateToPose).
  EVADE    a tracked meteor is predicted to land within the danger radius: cancel
           the Nav2 goal and drive straight away from the predicted impact point
           with a direct /cmd_vel controller (Nav2 replanning is too slow for a
           few-second warning). Drives in reverse when that is quicker than turning.
  SHELTER  hold position until the threat has landed, then mark the impact zone as
           an obstacle for Nav2 (/hazards/obstacles) and resume PATROL.

Every 0.5 s it prints the rover's position relative to the station (lander),
computed from its own navigation estimate (wheel+IMU EKF, TF map->base_link), and
publishes it on /rover/station_pose. With print_truth:=true it also shows the
estimate's error against the simulator's ground truth (evaluation only).
"""
import json
import math

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from crater_interfaces.msg import MeteorTrackArray
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')
        self.declare_parameter('station_x', -2.6)
        self.declare_parameter('station_y', 0.0)
        # patrol loop around the station (map frame), flattened [x0, y0, x1, y1, ...]
        self.declare_parameter('waypoints', [6.0, 6.0, 6.0, -6.0, -11.0, -6.0, -11.0, 6.0])
        self.declare_parameter('danger_radius', 3.0)
        self.declare_parameter('escape_margin', 1.5)
        self.declare_parameter('evade_speed', 0.6)
        self.declare_parameter('print_period', 0.5)
        self.declare_parameter('print_truth', True)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.station = np.array([p('station_x'), p('station_y')])
        w = p('waypoints')
        self.waypoints = [np.array(w[i:i + 2]) for i in range(0, len(w) - 1, 2)]
        self.danger, self.margin, self.v_evade = p('danger_radius'), p('escape_margin'), p('evade_speed')
        self.print_truth = p('print_truth')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/rover/station_pose', 10)
        self.obst_pub = self.create_publisher(point_cloud2.PointCloud2, '/hazards/obstacles', 10)
        # for Mission Control (dashboard): state snapshot at 2 Hz + discrete events
        self.state_pub = self.create_publisher(String, '/rover/mission_state', 10)
        self.event_pub = self.create_publisher(String, '/rover/events', 50)
        self.create_subscription(MeteorTrackArray, '/meteors/tracks', self._tracks_cb, 10)
        self.create_subscription(Odometry, '/ground_truth/odom', self._truth_cb, 10)

        self.state = 'WAIT_NAV2'
        self.wp_idx = 0
        self.goal_handle = None
        self.goal_seq = 0
        self.retry_at = None
        self.threat = None            # dict: id, impact (np 2), tti_abs, escape (np 2)
        self.handled = set()          # meteor ids already dealt with
        self.impact_zones = []        # (x, y, r) known fresh craters
        self.tracks = {}
        self.truth = None
        self.pose = None              # (x, y, yaw) estimate
        self.counters = {'threats': 0, 'evasions_completed': 0, 'waypoints_reached': 0}

        self.create_timer(0.05, self._control)
        self.create_timer(p('print_period'), self._print)
        self.create_timer(1.0, self._publish_obstacles)
        self.create_timer(1.0, self._wait_nav2)       # also handles goal retries
        self.create_timer(0.5, self._publish_state)

    # ------------------------------------------------------------------ inputs
    def _truth_cb(self, m):
        self.truth = (m.pose.pose.position.x, m.pose.pose.position.y)

    def _tracks_cb(self, msg):
        now = self._now()
        self.tracks = {m.id: m for m in msg.tracks}
        if self.state not in ('PATROL', 'EVADE', 'SHELTER') or self.pose is None:
            return
        threats = [m for m in msg.tracks if m.is_threat and m.time_to_impact > 0 and m.id not in self.handled]
        if not threats:
            # keep the active threat's prediction fresh while it is still tracked
            if self.threat and self.threat['id'] in self.tracks:
                self._update_threat(self.tracks[self.threat['id']], now)
            return
        m = min(threats, key=lambda k: k.time_to_impact)
        if self.threat is None or self.threat['id'] != m.id:
            self._start_evade(m, now)
        else:
            self._update_threat(m, now)

    # ------------------------------------------------------------------ helpers
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _update_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except TransformException:
            return False
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (tf.transform.translation.x, tf.transform.translation.y, yaw)
        return True

    def _escape_point(self, impact, vel_xy):
        rover = np.array(self.pose[:2])
        away = rover - impact
        if np.linalg.norm(away) < 0.5:
            # rover sits on the impact point: run perpendicular to the meteor's ground track,
            # choosing the side closer to the current heading
            perp = np.array([-vel_xy[1], vel_xy[0]])
            if np.linalg.norm(perp) < 1e-3:
                perp = np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
            heading = np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])
            away = perp if perp @ heading >= 0 else -perp
        away = away / np.linalg.norm(away)
        return impact + away * (self.danger + self.margin)

    def _start_evade(self, m, now):
        impact = np.array([m.impact_point.x, m.impact_point.y])
        esc = self._escape_point(impact, np.array([m.velocity.x, m.velocity.y]))
        self.threat = {'id': m.id, 'impact': impact, 'tti_abs': now + m.time_to_impact,
                       'escape': esc, 'sigma': m.impact_sigma}
        self._cancel_nav()
        self.state = 'EVADE'
        self.counters['threats'] += 1
        self._event('warn',
            f'!!! INCOMING meteor #{m.id}: impact in {m.time_to_impact:.1f} s at '
            f'({impact[0]:.1f}, {impact[1]:.1f}) ±{m.impact_sigma:.2f} m, {m.miss_distance:.1f} m from rover '
            f'-> EVADING to ({esc[0]:.1f}, {esc[1]:.1f})')

    def _update_threat(self, m, now):
        impact = np.array([m.impact_point.x, m.impact_point.y])
        th = self.threat
        th['tti_abs'] = now + m.time_to_impact
        th['sigma'] = m.impact_sigma
        if np.linalg.norm(impact - th['impact']) > 0.5 and self.state == 'EVADE':
            th['impact'] = impact
            th['escape'] = self._escape_point(impact, np.array([m.velocity.x, m.velocity.y]))
        else:
            th['impact'] = impact

    # ------------------------------------------------------------------ Nav2
    def _wait_nav2(self):
        if self.retry_at is not None and self.state == 'PATROL' and self._now() > self.retry_at:
            self.retry_at = None
            self._send_goal()
        if self.state == 'WAIT_NAV2' and self.nav.server_is_ready() and self._update_pose():
            self._event('info', f'Nav2 ready. Starting patrol around the station '
                                f'({len(self.waypoints)} waypoints).')
            self.state = 'PATROL'
            self._send_goal()

    def _send_goal(self):
        wp = self.waypoints[self.wp_idx]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(wp[0]), float(wp[1])
        nxt = self.waypoints[(self.wp_idx + 1) % len(self.waypoints)]
        yaw = math.atan2(nxt[1] - wp[1], nxt[0] - wp[0])
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_seq += 1
        seq = self.goal_seq
        self.nav.send_goal_async(goal).add_done_callback(lambda f: self._goal_accepted(f, seq))

    def _goal_accepted(self, fut, seq):
        gh = fut.result()
        if seq != self.goal_seq or self.state != 'PATROL':
            if gh and gh.accepted:
                gh.cancel_goal_async()
            return
        if not gh or not gh.accepted:
            self.get_logger().warn('patrol goal rejected; retrying in 2 s')
            self.retry_at = self._now() + 2.0
            return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(lambda f: self._goal_done(f, seq))

    def _goal_done(self, fut, seq):
        if seq != self.goal_seq or self.state != 'PATROL':
            return
        status = fut.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.counters['waypoints_reached'] += 1
            self._event('info', f'reached patrol waypoint WP{self.wp_idx}')
        else:
            self._event('warn', f'patrol waypoint WP{self.wp_idx} not reached (status {status}); skipping')
        self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
        self._send_goal()

    def _cancel_nav(self):
        self.goal_seq += 1            # invalidates callbacks of the old goal
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    # ------------------------------------------------------------------ control loop
    def _control(self):
        if not self._update_pose() or self.state not in ('EVADE', 'SHELTER'):
            return
        now = self._now()
        th = self.threat
        x, y, yaw = self.pose
        if self.state == 'EVADE':
            err = th['escape'] - np.array([x, y])
            dist = float(np.linalg.norm(err))
            clear = np.linalg.norm(np.array([x, y]) - th['impact']) > self.danger + 0.5
            if dist < 0.4 or (clear and now > th['tti_abs']):
                self.cmd_pub.publish(Twist())
                self.state = 'SHELTER'
                self.counters['evasions_completed'] += 1
                self._event('info',
                    f'evasion manoeuvre complete: now {np.linalg.norm(np.array([x, y]) - th["impact"]):.1f} m '
                    f'from predicted impact; holding until meteor #{th["id"]} lands')
                return
            desired = math.atan2(err[1], err[0])
            a = wrap(desired - yaw)
            reverse = abs(a) > math.radians(100)
            if reverse:
                a = wrap(a + math.pi)
            tw = Twist()
            tw.angular.z = float(np.clip(2.0 * a, -1.0, 1.0))
            if abs(a) < math.radians(35):
                tw.linear.x = (-1 if reverse else 1) * self.v_evade * min(1.0, dist / 0.8)
            self.cmd_pub.publish(tw)
        elif self.state == 'SHELTER':
            self.cmd_pub.publish(Twist())
            if now > th['tti_abs'] + 1.0 and th['id'] not in self.tracks:
                self.impact_zones.append((float(th['impact'][0]), float(th['impact'][1]), 1.2))
                self.handled.add(th['id'])
                self._event('info',
                    f'meteor #{th["id"]} has landed near ({th["impact"][0]:.1f}, {th["impact"][1]:.1f}); '
                    f'impact zone added to the hazard map. Resuming patrol -> WP{self.wp_idx}')
                self.threat = None
                self.state = 'PATROL'
                self._send_goal()

    # ------------------------------------------------------------------ outputs
    def _event(self, level, text):
        """Log an event and forward it to Mission Control."""
        # separate call sites: rclpy forbids one call site from switching severity
        if level == 'warn':
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)
        self.event_pub.publish(String(data=json.dumps({'t': self._now(), 'level': level, 'text': text})))

    def _publish_state(self):
        th = self.threat
        state = {
            't': self._now(), 'state': self.state, 'wp_idx': self.wp_idx,
            'waypoints': [[float(w[0]), float(w[1])] for w in self.waypoints],
            'station': [float(self.station[0]), float(self.station[1])],
            'danger_radius': self.danger,
            'threat': None if th is None else {
                'id': th['id'], 'impact': [float(th['impact'][0]), float(th['impact'][1])],
                'time_to_impact': th['tti_abs'] - self._now(), 'sigma': float(th['sigma']),
                'escape': [float(th['escape'][0]), float(th['escape'][1])]},
            'impact_zones': [list(z) for z in self.impact_zones],
            'counters': self.counters,
        }
        self.state_pub.publish(String(data=json.dumps(state)))

    def _print(self):
        if self.pose is None:
            return
        x, y, yaw = self.pose
        rel = np.array([x, y]) - self.station
        dist = float(np.linalg.norm(rel))
        bearing = math.degrees(math.atan2(rel[1], rel[0]))
        ps = PoseStamped(header=Header(stamp=self.get_clock().now().to_msg(), frame_id='station'))
        ps.pose.position.x, ps.pose.position.y = float(rel[0]), float(rel[1])
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pose_pub.publish(ps)

        if self.state == 'PATROL':
            mode = f'PATROL -> WP{self.wp_idx}'
        elif self.state in ('EVADE', 'SHELTER'):
            mode = f'{self.state} (meteor #{self.threat["id"]}, impact in {self.threat["tti_abs"] - self._now():+.1f} s)'
        else:
            mode = self.state
        n_threat = sum(1 for m in self.tracks.values() if m.is_threat)
        line = (f'[t={self._now():7.1f}s] Rover rel. station: x={rel[0]:+7.2f} m  y={rel[1]:+7.2f} m  '
                f'dist={dist:6.2f} m  bearing={bearing:+5.0f}°  heading={math.degrees(yaw):+5.0f}° | '
                f'{mode} | objects tracked: {len(self.tracks)}, threats: {n_threat}')
        if self.print_truth and self.truth is not None:
            line += f' | nav error vs truth {math.hypot(self.truth[0] - x, self.truth[1] - y):.2f} m'
        print(line, flush=True)

    def _publish_obstacles(self):
        pts = []
        for cx, cy, r in self.impact_zones:
            for ox in np.arange(-r, r + 1e-6, 0.1):
                for oy in np.arange(-r, r + 1e-6, 0.1):
                    if ox * ox + oy * oy <= r * r:
                        pts.append((cx + ox, cy + oy, 0.1))
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        self.obst_pub.publish(point_cloud2.create_cloud_xyz32(header, pts))


def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())      # leave the rover stopped
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
