#!/usr/bin/env python3
"""meteor_spawner: the *space environment* (simulation side, NOT onboard software).

Periodically throws a meteor at the rover's neighbourhood and reports where it really
landed. It is allowed to use ground truth (/ground_truth/odom, Gazebo poses); the
rover's own software never sees any of this, it only has its RGB-D camera.

Each meteor:
  * spawns 18-28 m from the rover, within +-35 deg of its heading (so the forward
    camera can see it coming), 12-18 m above the ground,
  * is aimed at the rover's predicted position + a random 0-5 m offset (about half
    the meteors are real threats), with a 7-9 s flight time:
        v0 = (target - p0 - 1/2 g T^2) / T,    g = (0, 0, -1.62)
  * is launched by the gz Python system gz_systems/meteor_launch.py (one-step
    impulse), then flies a real ballistic arc under lunar gravity,
  * on touchdown (seen in Gazebo's pose stream) is removed and replaced by a fresh
    crater; /sim/impacts reports the impact point, the rover's true distance and
    HIT (< blast radius + rover radius) or SAFE.

Framing: real meteoroids arrive at km/s and no rover could dodge them; this models
slower ballistic debris (impact ejecta), time-scaled for the demo.
"""
import json
import math
import os
import random
import sys
import threading

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from gen_world import crater_sdf, label_plugin, load_rock_meshes  # noqa: E402  (same look/labels as the world's)

G = 1.62


def meteor_sdf(name, p0, v0, r, mass, mesh):
    """Irregular rock visual (library mesh, scaled to ~2r) on a sphere collision, so the
    ballistic physics stay simple. `mesh` is an entry of the rock-mesh index."""
    I = 0.4 * mass * r * r
    ez = mesh['extent'][2]
    s = 2 * r                                       # footprint ~ diameter
    sz = 2 * r / max(ez, 0.3)                       # flat meshes are thickened a little
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <pose>{p0[0]:.3f} {p0[1]:.3f} {p0[2]:.3f} 0 0 0</pose>
    <link name="link">
      <inertial><mass>{mass}</mass><inertia><ixx>{I}</ixx><iyy>{I}</iyy><izz>{I}</izz></inertia></inertial>
      <collision name="c"><geometry><sphere><radius>{r}</radius></sphere></geometry></collision>
      <visual name="rock"><pose>0 0 {-ez * min(sz, s * 1.4) / 2:.3f} 0 0 0</pose>
        <geometry><mesh><uri>model://lunar_rocks/meshes/{mesh['file']}</uri>
          <scale>{s:.3f} {s:.3f} {min(sz, s * 1.4):.3f}</scale></mesh></geometry>
        <material><ambient>0.35 0.2 0.12 1</ambient><diffuse>0.4 0.25 0.15 1</diffuse>
          <emissive>0.9 0.35 0.05 1</emissive></material></visual>
    </link>
    {label_plugin('meteor')}
    <plugin filename="gz-sim-python-system-loader-system" name="gz::sim::systems::PythonSystemLoader">
      <module_name>meteor_launch</module_name>
      <mass>{mass}</mass><vx>{v0[0]:.4f}</vx><vy>{v0[1]:.4f}</vy><vz>{v0[2]:.4f}</vz>
    </plugin>
  </model>
</sdf>"""


class MeteorSpawner(Node):
    def __init__(self):
        super().__init__('meteor_spawner')
        for name, default in [('world', 'lunar_crater'), ('period', 20.0), ('first_delay', 35.0),
                              ('seed', 1), ('max_offset', 5.0), ('blast_radius', 1.5),
                              ('rover_radius', 0.5), ('max_meteors', 0)]:
            # dynamic typing: `meteor_period:=18` arrives as an int
            self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True))
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.world = p('world')
        self.period, self.first_delay = float(p('period')), float(p('first_delay'))
        self.max_offset, self.max_meteors = float(p('max_offset')), int(p('max_meteors'))
        self.hit_radius = float(p('blast_radius')) + float(p('rover_radius'))
        self.rng = random.Random(int(p('seed')))
        self.rock_meshes = load_rock_meshes()

        self.gz = GzNode()
        self.lock = threading.Lock()
        self.live = {}          # name -> dict(spawned meteor state)
        self.landed = []        # (name, x, y) detected by the gz pose callback
        self.gz.subscribe(Pose_V, f'/world/{self.world}/pose/info', self._gz_poses)

        self.rover = None       # (x, y, yaw, vx_world, vy_world, t)
        self.create_subscription(Odometry, '/ground_truth/odom', self._truth_cb, 10)
        self.report_pub = self.create_publisher(String, '/sim/impacts', 10)
        self.count = 0
        self.t_start = None
        self.next_spawn = None
        self.create_timer(0.1, self._tick)

    # ------------------------------------------------------------------ inputs
    def _truth_cb(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        v = m.twist.twist.linear.x
        self.rover = (m.pose.pose.position.x, m.pose.pose.position.y, yaw,
                      v * math.cos(yaw), v * math.sin(yaw))

    def _gz_poses(self, msg):
        """Gazebo pose stream (gz transport thread): detect meteor touchdown."""
        with self.lock:
            for pose in msg.pose:
                m = self.live.get(pose.name)
                if m and not m['landed'] and pose.position.z < m['r'] + 0.08:
                    m['landed'] = True
                    self.landed.append((pose.name, pose.position.x, pose.position.y))

    # ------------------------------------------------------------------ logic
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        now = self._now()
        if self.rover is None or now <= 0:
            return
        if self.t_start is None:
            self.t_start = now
            self.next_spawn = now + self.first_delay
        if now >= self.next_spawn and (self.max_meteors <= 0 or self.count < self.max_meteors):
            self._spawn(now)
            self.next_spawn = now + self.period * self.rng.uniform(0.8, 1.2)
        with self.lock:
            landed, self.landed = self.landed, []
        for name, x, y in landed:
            self._impact(name, x, y, now)
        # safety net: meteors that never registered a touchdown
        for name in [n for n, m in self.live.items() if now - m['t_spawn'] > m['T'] + 6.0]:
            self._remove(name)
            self.live.pop(name, None)

    def _spawn(self, now):
        rx, ry, ryaw, rvx, rvy = self.rover
        T = self.rng.uniform(7.0, 9.0)
        r = self.rng.uniform(0.25, 0.4)
        mass = round(2700 * 4 / 3 * math.pi * r ** 3, 1)        # basalt-ish density
        a, d = ryaw + math.radians(self.rng.uniform(-35, 35)), self.rng.uniform(18, 28)
        p0 = (rx + d * math.cos(a), ry + d * math.sin(a), self.rng.uniform(12, 18))
        off_r, off_a = self.max_offset * math.sqrt(self.rng.random()), self.rng.uniform(0, 2 * math.pi)
        tx = rx + rvx * T + off_r * math.cos(off_a)
        ty = ry + rvy * T + off_r * math.sin(off_a)
        v0 = ((tx - p0[0]) / T, (ty - p0[1]) / T, (r - p0[2] + 0.5 * G * T * T) / T)
        self.count += 1
        name = f'meteor_{self.count}'
        req = EntityFactory()
        req.sdf = meteor_sdf(name, p0, v0, r, mass, self.rng.choice(self.rock_meshes))
        for attempt in range(2):          # the create service can time out under heavy CPU load
            ok, rep = self.gz.request(f'/world/{self.world}/create', req, EntityFactory, Boolean, 5000)
            if ok and rep.data:
                break
        else:
            self.get_logger().error(f'failed to spawn {name} ({"rejected" if ok else "service timeout"})')
            return
        with self.lock:
            self.live[name] = {'t_spawn': now, 'T': T, 'r': r, 'aim': (tx, ty), 'offset': off_r,
                               'p0': p0, 'v0': v0, 'landed': False}
        self.get_logger().info(
            f'[SPACE] {name} inbound: from ({p0[0]:.1f},{p0[1]:.1f}, h={p0[2]:.1f}) m, '
            f'speed {math.sqrt(sum(c * c for c in v0)):.1f} m/s, aimed {off_r:.1f} m from rover\'s predicted '
            f'position, ETA {T:.1f} s')

    def _impact(self, name, x, y, now):
        m = self.live.pop(name, None)
        if m is None or self.rover is None:
            return
        rx, ry = self.rover[0], self.rover[1]
        dist = math.hypot(x - rx, y - ry)
        verdict = 'HIT' if dist < self.hit_radius else 'SAFE'
        rep = {'name': name, 'id': int(name.split('_')[1]), 'impact_time': now, 'flight_time': now - m['t_spawn'],
               'impact': [x, y], 'aim': list(m['aim']), 'aim_offset': m['offset'],
               'rover_at_impact': [rx, ry], 'rover_distance': dist, 'verdict': verdict}
        self.report_pub.publish(String(data=json.dumps(rep)))
        text = f'[SPACE] {name} IMPACT at ({x:.1f},{y:.1f}); rover {dist:.1f} m away -> {verdict}'
        if verdict == 'HIT':
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)
        self._remove(name)
        req = EntityFactory()
        req.sdf = ('<?xml version="1.0"?><sdf version="1.9">'
                   + crater_sdf(f'impact_crater_{name.split("_")[1]}', x, y,
                                self.rng.uniform(0.7, 1.0), self.rng) + '</sdf>')
        self.gz.request(f'/world/{self.world}/create', req, EntityFactory, Boolean, 3000)

    def _remove(self, name):
        req = Entity()
        req.name = name
        req.type = Entity.MODEL
        self.gz.request(f'/world/{self.world}/remove', req, Entity, Boolean, 3000)


def main():
    rclpy.init()
    node = MeteorSpawner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass   # RCLError: context torn down by the signal handler mid-spin; normal on Ctrl+C
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
