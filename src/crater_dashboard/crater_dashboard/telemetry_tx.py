#!/usr/bin/env python3
"""telemetry_tx: the rover's radio. Sends compact binary telemetry to Mission Control
over UDP (stdlib socket); every datagram is ACKed and retransmitted until confirmed.

Inputs (rover's own data only): /meteors/tracks, /odom, TF map->base_footprint,
/rover/mission_state, /rover/events.

Datagrams (see protocol.py):
  HELLO      at start and every 10 s: station, danger radius, patrol waypoints, link settings
  TELEMETRY  5 Hz: rover pose/speed, link stats, every tracked meteor (26 B each)
  MISSION    on change (and at least 1 Hz): state, counters, active threat
  EVENT      each mission event (alerts, evasions, landings, waypoints)
  CRATER     each impact zone added to the rover's hazard map
"""
import json
import math

import rclpy
from crater_interfaces.msg import MeteorTrackArray
from nav_msgs.msg import Odometry
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from . import protocol as P
from .link import ReliableSender


def stamp_s(st):
    return st.sec + st.nanosec * 1e-9


class TelemetryTx(Node):
    def __init__(self):
        super().__init__('telemetry_tx')
        for name, default in [('ground_host', '127.0.0.1'), ('ground_port', 5005), ('rate', 5.0),
                              ('link_loss', 0.10), ('link_delay_ms', 80.0), ('link_jitter_ms', 40.0),
                              ('link_corrupt', 0.01), ('station_x', -2.6), ('station_y', 0.0)]:
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.emu = {'loss': float(p('link_loss')), 'delay_ms': float(p('link_delay_ms')),
                    'jitter_ms': float(p('link_jitter_ms')), 'corrupt': float(p('link_corrupt'))}
        peer = (p('ground_host'), int(p('ground_port')))
        self.tx = ReliableSender(peer, emulator_cfg=self.emu)
        self.station = [float(p('station_x')), float(p('station_y'))]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tracks, self.speed = [], 0.0
        self.mission = None
        self.last_mission_payload, self.last_mission_t = None, -1e9
        self.craters_sent = 0

        self.create_subscription(MeteorTrackArray, '/meteors/tracks', lambda m: setattr(self, 'tracks', m.tracks), 10)
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'speed', m.twist.twist.linear.x), 10)
        self.create_subscription(String, '/rover/mission_state', self._mission_cb, 10)
        self.create_subscription(String, '/rover/events', self._event_cb, 50)
        self.create_timer(1.0 / float(p('rate')), self._telemetry)
        self.create_timer(10.0, self._hello)
        self.create_timer(10.0, self._log_stats)
        self.hello_sent = False
        self.get_logger().info(f'radio up: session {self.tx.session:#06x} -> udp://{peer[0]}:{peer[1]}, '
                               f'emulated link {self.emu}')

    def _t_ms(self):
        return int(self.get_clock().now().nanoseconds // 1_000_000)

    # ------------------------------------------------------------------ senders
    def _hello(self):
        m = self.mission or {}
        payload = P.encode_hello(m.get('station', self.station), m.get('danger_radius', 3.0),
                                 m.get('waypoints', []), self.emu)
        self.tx.send(P.HELLO, self._t_ms(), payload)
        self.hello_sent = True

    def _telemetry(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except TransformException:
            return
        if not self.hello_sent and self.mission is not None:
            self._hello()
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        gz = tf.transform.translation.z
        now = self.get_clock().now().nanoseconds * 1e-9
        meteors = []
        for m in self.tracks:
            alt = m.position.z - gz
            if alt < 0:
                continue
            has = m.time_to_impact > 0
            meteors.append({
                'id': m.id, 'x': m.position.x, 'y': m.position.y, 'alt': alt,
                'vx': m.velocity.x, 'vy': m.velocity.y, 'vz': m.velocity.z,
                'impact_x': m.impact_point.x if has else None, 'impact_y': m.impact_point.y if has else None,
                'tti': m.time_to_impact if has else None, 'sigma': m.impact_sigma if has else None,
                'threat': m.is_threat, 'age': max(0.0, now - stamp_s(m.last_seen)), 'n_obs': m.n_obs})
        payload = P.encode_telemetry(
            {'x': tf.transform.translation.x, 'y': tf.transform.translation.y, 'yaw': yaw, 'speed': self.speed},
            self.tx.link_stats(), meteors)
        self.tx.send(P.TELEMETRY, self._t_ms(), payload)
        self._mission(force=False)

    def _mission_cb(self, msg):
        first = self.mission is None
        self.mission = json.loads(msg.data)
        if first:
            self._hello()                                  # waypoints/station now known
        zones = self.mission.get('impact_zones', [])
        while self.craters_sent < len(zones):              # new impact zones in the hazard map
            x, y, r = zones[self.craters_sent]
            self.craters_sent += 1
            self.tx.send(P.CRATER, self._t_ms(), P.encode_crater(self.craters_sent, x, y, r))
        self._mission(force=False)

    def _mission(self, force):
        if self.mission is None:
            return
        m = self.mission
        payload = P.encode_mission(m['state'], m['wp_idx'], m.get('counters', {}), m.get('threat'))
        now = self.get_clock().now().nanoseconds * 1e-9
        # the threat countdown changes every call; compare without it to detect real changes
        key = (m['state'], m['wp_idx'], json.dumps(m.get('counters')),
               None if not m.get('threat') else m['threat']['id'])
        if force or key != self.last_mission_payload or now - self.last_mission_t >= 1.0:
            self.tx.send(P.MISSION, self._t_ms(), payload)
            self.last_mission_payload, self.last_mission_t = key, now

    def _event_cb(self, msg):
        e = json.loads(msg.data)
        self.tx.send(P.EVENT, int(e['t'] * 1000), P.encode_event(e['level'], e['text']))

    def _log_stats(self):
        s = self.tx.stats
        srtt = 'n/a' if self.tx.srtt is None else f'{self.tx.srtt * 1000:.0f} ms'
        self.get_logger().info(
            f'link: sent {s["sent"]} datagrams ({s["bytes"] / 1024:.1f} KiB incl. retransmissions), '
            f'retransmitted {s["retransmissions"]}, abandoned {s["abandoned"]}, in flight {len(self.tx.unacked)}, '
            f'SRTT {srtt}, RTO {self.tx.rto * 1000:.0f} ms')


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryTx()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.tx.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
