#!/usr/bin/env python3
"""dashboard_server: Mission Control ground station for the C.R.A.T.E.R rover.

Everything about the rover reaches this process ONLY through the radio link: compact
binary UDP datagrams from the rover's telemetry_tx (see protocol.py / link.py), each one
CRC-checked, ACKed (with a SACK bitmap) and de-duplicated. The ground station keeps the
latest mission picture and streams it to browsers with Server-Sent Events over a stdlib
HTTP server (no extra packages, no internet):

    GET /               the dashboard page (crater_dashboard/web/index.html)
    GET /stream         text/event-stream: `snapshot` events (10 Hz) + `heartbeat` (1 Hz)
    GET /snapshot.json  latest snapshot (debugging / curl)

Obstacles and the distances shown for each (computed here, on the ground):
  meteor  incoming, tracked by the rover's RGB-D camera
          now:       3D distance to the robot, 3D distance to the station
          predicted: impact point distance to the robot and to the station
  crater  landed meteors: CRATER datagrams (rover's hazard map), plus meteors that
          vanished from telemetry right at their predicted touchdown
          ground distance to the robot and to the station

Late (retransmitted) TELEMETRY older than the newest one applied only updates link
statistics; it never moves the rover backwards on the map.
Ground truth (/ground_truth/odom, /sim/impacts) is read over ROS only when sim_truth is
true, and only feeds the clearly labelled SIM TRUTH fields.
"""
import json
import math
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from . import protocol as P
from .link import ReliableReceiver

HISTORY_S = 90.0


def clean(v):
    """JSON-safe copy: NaN/inf -> None (the browser's JSON.parse rejects them)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_server')
        for name, default in [('host', '0.0.0.0'), ('port', 8080), ('telemetry_port', 5005),
                              ('sim_truth', True), ('station_x', -2.6), ('station_y', 0.0),
                              ('link_loss', 0.10), ('link_delay_ms', 80.0), ('link_jitter_ms', 40.0),
                              ('link_corrupt', 0.01)]:
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.emu = {'loss': float(p('link_loss')), 'delay_ms': float(p('link_delay_ms')),
                    'jitter_ms': float(p('link_jitter_ms')), 'corrupt': float(p('link_corrupt'))}

        self.lock = threading.Lock()
        self.snapshot, self.seq, self.snapshot_wall = None, 0, 0.0
        # ---- state reconstructed from the radio link
        self.hello = {'station': [float(p('station_x')), float(p('station_y'))], 'danger_radius': 3.0,
                      'waypoints': [], 'emu': None}
        self.telem = None                  # dict: t, wall, rover, link, meteors
        self.mission = None                # dict + t
        self.events = deque(maxlen=40)
        self.landed = []                   # craters: dict(id, x, y, t, source)
        self.last_impacts = {}             # meteor id -> (impact_x, impact_y, t_impact)
        self.history = {}                  # obstacle key -> deque[(t, d_robot, d_station)]
        self.trail, self.last_trail_t = deque(maxlen=400), -1e9
        self.rtf = 1.0                     # sim-time / wall-time rate, measured from telemetry
        self.stale_telemetry = 0           # late frames (older than the newest applied)
        self.counts = {k: 0 for k in P.TYPE_NAMES.values()}
        self.last_frame_bytes, self.last_frame_json_bytes = None, None
        self.throughput = deque(maxlen=60)

        self.rx = ReliableReceiver(int(p('telemetry_port')), self._on_packet, emulator_cfg=self.emu)
        self.get_logger().info(f'ground station listening on udp://0.0.0.0:{p("telemetry_port")} '
                               f'(ACK direction emulated: {self.emu})')
        # ---- SIM TRUTH (evaluation only)
        self.truth, self.impacts = None, deque(maxlen=20)
        if p('sim_truth'):
            self.create_subscription(Odometry, '/ground_truth/odom', self._truth_cb, 10)
            self.create_subscription(String, '/sim/impacts', lambda m: self.impacts.appendleft(json.loads(m.data)), 10)
        threading.Thread(target=self._snapshot_loop, daemon=True).start()
        threading.Thread(target=self._throughput_loop, daemon=True).start()

    def _truth_cb(self, m):
        self.truth = (m.pose.pose.position.x, m.pose.pose.position.y)

    # ------------------------------------------------------------------ radio link input
    def _on_packet(self, ptype, seq, t_ms, payload):
        t = t_ms / 1000.0
        name = P.TYPE_NAMES.get(ptype, '?')
        with self.lock:
            self.counts[name] = self.counts.get(name, 0) + 1
            if ptype == P.HELLO:
                self.hello = P.decode_hello(payload)
            elif ptype == P.TELEMETRY:
                self._on_telemetry(t, payload)
            elif ptype == P.MISSION:
                m = P.decode_mission(payload)
                if self.mission is None or t >= self.mission['t']:
                    m['t'] = t
                    self.mission = m
            elif ptype == P.EVENT:
                e = P.decode_event(payload)
                e['t'] = t
                self.events.append(e)
                self.events = deque(sorted(self.events, key=lambda k: k['t']), maxlen=40)
            elif ptype == P.CRATER:
                c = P.decode_crater(payload)
                self._add_crater(c['x'], c['y'], t, 'hazard map', hz_id=c['id'])

    def _on_telemetry(self, t, payload):
        rover, link, meteors = P.decode_telemetry(payload)
        size = len(payload) + P.OVERHEAD
        if self.telem is not None and t <= self.telem['t']:
            self.stale_telemetry += 1          # late retransmission: statistics only
            return
        wall = time.time()
        if self.telem is not None and wall > self.telem['wall'] and t > self.telem['t']:
            r = (t - self.telem['t']) / (wall - self.telem['wall'])
            self.rtf = min(3.0, max(0.05, 0.9 * self.rtf + 0.1 * r))
        rover['yaw'] = math.radians(rover['yaw'])
        self.telem = {'t': t, 'wall': wall, 'rover': rover, 'link': link, 'meteors': meteors}
        self.last_frame_bytes = size
        self.last_frame_json_bytes = len(json.dumps({'t': t, 'rover': rover, 'link': link, 'meteors': meteors}))
        # landed meteors: one that disappears around its predicted touchdown time
        current = {m['id'] for m in meteors}
        for m in meteors:
            if m.get('tti') is not None:
                self.last_impacts[m['id']] = (m['impact_x'], m['impact_y'], t + m['tti'])
        for mid in [k for k in self.last_impacts if k not in current]:
            ix, iy, t_imp = self.last_impacts.pop(mid)
            if t_imp - t < 3.0:
                self._add_crater(ix, iy, min(t, t_imp), 'tracker')
        if t - self.last_trail_t > 0.5:
            self.trail.append((round(rover['x'], 2), round(rover['y'], 2)))
            self.last_trail_t = t
        # distance history at telemetry rate (computed on the ground)
        for ob in self._obstacles(t, rover, meteors, t):
            h = self.history.setdefault(ob['key'], deque())
            h.append((round(t, 2), round(ob['d_robot'], 2), round(ob['d_station'], 2)))
            while h and t - h[0][0] > HISTORY_S:
                h.popleft()

    def _add_crater(self, x, y, t, source, hz_id=None):
        """Craters come from the rover's hazard map (CRATER datagrams, unique hz_id) or are
        inferred here from a meteor vanishing at its predicted touchdown. A tracker-inferred
        crater and a hazard-map crater within 2.5 m (blast radius + prediction error) are the
        same impact; two hazard-map craters are always distinct impacts."""
        for c in self.landed:
            if hz_id is not None and c.get('hz_id') == hz_id:
                return                                     # already known
            both_map = hz_id is not None and c.get('hz_id') is not None
            if not both_map and math.hypot(c['x'] - x, c['y'] - y) < 2.5:
                if hz_id is not None:                      # the rover's own map wins
                    c.update(x=float(x), y=float(y), source=source, hz_id=hz_id)
                return
        self.landed.append({'x': float(x), 'y': float(y), 't': float(t), 'source': source,
                            'hz_id': hz_id, 'id': len(self.landed) + 1})

    # ------------------------------------------------------------------ derived data
    def _obstacles(self, now, rover, meteors, t_frame):
        rx, ry = rover['x'], rover['y']
        sx, sy = self.hello['station']
        out = []
        for m in meteors:
            px, py, alt = m['x'], m['y'], m['alt']
            ob = {'key': f'M{m["id"]}', 'type': 'meteor', 'id': m['id'],
                  'status': 'THREAT' if m['threat'] else 'inbound', 'x': px, 'y': py, 'alt': alt,
                  'speed': math.sqrt(m['vx'] ** 2 + m['vy'] ** 2 + m['vz'] ** 2), 'vx': m['vx'], 'vy': m['vy'],
                  'd_robot': math.sqrt((px - rx) ** 2 + (py - ry) ** 2 + alt ** 2),
                  'd_station': math.sqrt((px - sx) ** 2 + (py - sy) ** 2 + alt ** 2),
                  'ground_d_robot': math.hypot(px - rx, py - ry), 'ground_d_station': math.hypot(px - sx, py - sy),
                  'age': (m['age'] or 0.0) + (now - t_frame), 'n_obs': m['n_obs'], 'is_threat': m['threat']}
            if m.get('impact_x') is not None:
                ix, iy = m['impact_x'], m['impact_y']
                ob.update({'impact_x': ix, 'impact_y': iy, 'tti': m['tti'] - (now - t_frame),
                           'sigma': m['sigma'], 'impact_d_robot': math.hypot(ix - rx, iy - ry),
                           'impact_d_station': math.hypot(ix - sx, iy - sy)})
            out.append(ob)
        for c in self.landed:
            d_robot, d_station = math.hypot(c['x'] - rx, c['y'] - ry), math.hypot(c['x'] - sx, c['y'] - sy)
            out.append({'key': f'C{c["id"]}', 'type': 'crater', 'id': c['id'], 'status': 'landed',
                        'x': c['x'], 'y': c['y'], 'alt': 0.0, 'd_robot': d_robot, 'd_station': d_station,
                        'ground_d_robot': d_robot, 'ground_d_station': d_station,
                        'age': now - c['t'], 'source': c['source']})
        return out

    # ------------------------------------------------------------------ snapshot (wall clock, 10 Hz)
    def _throughput_loop(self):
        while True:
            time.sleep(1.0)
            _, bps, _ = self.rx.rates(window=1.0)
            self.throughput.append(round(bps, 1))

    def _snapshot_loop(self):
        while True:
            time.sleep(0.1)
            try:
                self._build_snapshot()
            except Exception as e:                        # never let one bad frame kill the page
                self.get_logger().error(f'snapshot failed: {e!r}', throttle_duration_sec=5.0)

    def _build_snapshot(self):
        wall = time.time()
        with self.lock:
            if self.telem is None:
                return
            tel = self.telem
            now = tel['t'] + (wall - tel['wall']) * self.rtf       # rover time, extrapolated
            rover = tel['rover']
            sx, sy = self.hello['station']
            obstacles = self._obstacles(now, rover, tel['meteors'], tel['t'])
            rel = (rover['x'] - sx, rover['y'] - sy)
            mission = None
            if self.mission is not None:
                mission = dict(self.mission, waypoints=self.hello['waypoints'], station=self.hello['station'],
                               danger_radius=self.hello['danger_radius'])
                if mission.get('threat'):
                    th = dict(mission['threat'])
                    th['time_to_impact'] = th['time_to_impact'] - (now - self.mission['t'])
                    mission['threat'] = th
            pps, bps, avg = self.rx.rates()
            link = {
                'session': self.rx.session, 'last_packet_age': None if self.rx.last_rx is None else wall - self.rx.last_rx,
                'rx_pps': pps, 'rx_Bps': bps, 'avg_datagram': avg,
                'frame_bytes': self.last_frame_bytes, 'frame_json_bytes': self.last_frame_json_bytes,
                'ground': dict(self.rx.stats, stale_telemetry=self.stale_telemetry,
                               acks_dropped_by_emulator=self.rx.link.stats['dropped'],
                               acks_corrupted_by_emulator=self.rx.link.stats['corrupted']),
                'rover': tel['link'], 'counts': dict(self.counts),
                'emu_downlink': (self.hello.get('emu') or None), 'emu_ack': self.emu,
                'throughput': list(self.throughput), 'rtf': self.rtf,
            }
            snap = {
                'sim_time': now, 'wall_time': wall,
                'station': {'x': sx, 'y': sy},
                'rover': {'x': rover['x'], 'y': rover['y'], 'yaw': rover['yaw'], 'speed': rover['speed'],
                          'rel_x': rel[0], 'rel_y': rel[1], 'dist_station': math.hypot(*rel),
                          'bearing_deg': math.degrees(math.atan2(rel[1], rel[0])), 'trail': list(self.trail),
                          'data_age': now - tel['t']},
                'tracks_age': now - tel['t'],
                'obstacles': obstacles,
                'history': {k: list(v) for k, v in self.history.items()},
                'mission': mission,
                'events': sorted(self.events, key=lambda e: -e['t'])[:15],
                'link': link,
                'truth': {'nav_error': None if self.truth is None else
                          math.hypot(self.truth[0] - rover['x'], self.truth[1] - rover['y']),
                          'impacts': list(self.impacts)[:8]},
            }
        data = json.dumps(clean(snap), separators=(',', ':'), allow_nan=False)
        with self.lock:
            self.snapshot, self.snapshot_wall = data, time.time()
            self.seq += 1


def make_handler(node, page_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                with open(page_path, 'rb') as f:
                    self._send(200, 'text/html; charset=utf-8', f.read())
            elif path == '/snapshot.json':
                with node.lock:
                    data = node.snapshot or '{}'
                self._send(200, 'application/json', data.encode())
            elif path == '/stream':
                self._stream()
            else:
                self._send(404, 'text/plain', b'not found')

        def _stream(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            last_seq, last_hb = -1, 0.0
            try:
                self.wfile.write(b'retry: 1000\n\n')
                while True:
                    with node.lock:
                        seq, data, snap_wall = node.seq, node.snapshot, node.snapshot_wall
                    if data is not None and seq != last_seq:
                        self.wfile.write(b'event: snapshot\ndata: ' + data.encode() + b'\n\n')
                        self.wfile.flush()
                        last_seq = seq
                    now = time.time()
                    if now - last_hb > 1.0:
                        hb = json.dumps({'server_time': now, 'snapshot_age': now - snap_wall if snap_wall else None})
                        self.wfile.write(b'event: heartbeat\ndata: ' + hb.encode() + b'\n\n')
                        self.wfile.flush()
                        last_hb = now
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    page = os.path.join(get_package_share_directory('crater_dashboard'), 'web', 'index.html')
    host, port = node.get_parameter('host').value, int(node.get_parameter('port').value)
    server = None
    for candidate in range(port, port + 10):     # port taken (e.g. a second sim)? try the next ones
        try:
            server = ThreadingHTTPServer((host, candidate), make_handler(node, page))
            break
        except OSError as e:
            node.get_logger().warn(f'port {candidate} unavailable ({e.strerror}); trying {candidate + 1}')
    if server is None:
        node.get_logger().error(f'no free port in {port}-{port + 9}; dashboard not started')
        raise SystemExit(1)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://localhost:{server.server_address[1]}'
    node.get_logger().info('\n' + '=' * 60 + f'\n  MISSION CONTROL DASHBOARD  ->  open {url}\n' + '=' * 60)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass   # RCLError: context torn down by the signal handler mid-spin; normal on Ctrl+C
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
