#!/usr/bin/env python3
"""Procedural lunar-plain world generator for C.R.A.T.E.R.

An open, airless plain: regolith, craters, scattered rocks (irregular procedural
meshes from tools/gen_rocks.py: boulders, fractured blocks, slabs, ejecta around
crater rims), albedo patches, the landing pad at the origin (rover spawn) and the
lander = the **station**. Meteors are not part of the world: they arrive from space
at run time (see meteor_spawner.py).

Usage:
    gen_world.py --seed 7 --out worlds/lunar_crater.sdf
"""
import argparse
import json
import math
import os
import random

FIELD_R = 20.0          # craters are scattered out to this radius [m]
ROCK_FIELD_R = 24.0     # rocks out to this radius [m]
PAD_KEEPOUT = 4.0       # no craters this close to the pad
ROCK_PAD_KEEPOUT = 2.6  # no rocks this close to the rover spawn
STATION_POS = (-2.6, 0.0)
# mission_controller patrol waypoints, in the map frame (= rover spawn pose); kept clear of rocks
PATROL_WPS = [(6, 6), (6, -6), (-11, -6), (-11, 6)]


# Semantic labels (gz-sim Label system) read by the rover's bounding-box camera to
# auto-label YOLO training images. Shared with meteor_spawner.py.
LABELS = {'meteor': 1, 'crater': 2, 'station': 3, 'rock': 4}


def label_plugin(name):
    return (f'<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
            f'<label>{LABELS[name]}</label></plugin>')


def material(r, g, b, spec=0.05, emissive=None):
    em = f'<emissive>{emissive[0]} {emissive[1]} {emissive[2]} 1</emissive>' if emissive else ''
    return (f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
            f'<specular>{spec} {spec} {spec} 1</specular>{em}</material>')


def crater_sdf(name, x, y, r, rng):
    """Dark bowl floor + lighter slope + a low raised ejecta rim. Shared with
    meteor_spawner.py, which places fresh craters where meteors hit."""
    n_rim = max(10, int(2 * math.pi * r / 0.35))
    rim = ''
    # irregular rim: low-frequency wobble of radius and height, plus jitter
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
    h0 = rng.uniform(0.06, 0.13)
    for k in range(n_rim):
        a = 2 * math.pi * k / n_rim
        rr = r * (1 + 0.06 * math.cos(2 * a + ph[0]) + 0.04 * math.cos(3 * a + ph[1]) + rng.uniform(-0.02, 0.03))
        h = max(0.03, h0 * (1 + 0.5 * math.cos(a + ph[2])) * rng.uniform(0.8, 1.2))
        w = 2 * math.pi * r / n_rim * 1.15
        pose = f'{rr*math.cos(a):.3f} {rr*math.sin(a):.3f} {h/2:.3f} 0 0 {a:.3f}'
        rim += (f'<collision name="rim_c{k}"><pose>{pose}</pose>'
                f'<geometry><box><size>0.25 {w:.3f} {h:.3f}</size></box></geometry></collision>'
                f'<visual name="rim_v{k}"><pose>{pose}</pose>'
                f'<geometry><box><size>0.3 {w:.3f} {h:.3f}</size></box></geometry>{material(0.5, 0.5, 0.48)}</visual>')
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <link name="link">
        <visual name="slope"><pose>0 0 0.004 0 0 0</pose>
          <geometry><cylinder><radius>{r*0.97:.3f}</radius><length>0.008</length></cylinder></geometry>
          {material(0.2, 0.2, 0.2)}</visual>
        <visual name="floor"><pose>0 0 0.009 0 0 0</pose>
          <geometry><cylinder><radius>{r*0.68:.3f}</radius><length>0.008</length></cylinder></geometry>
          {material(0.05, 0.05, 0.05, spec=0.0)}</visual>
        {rim}
      </link>
      {label_plugin('crater')}
    </model>"""


def load_rock_meshes():
    """Index of the rock-mesh library (tools/gen_rocks.py): source tree or installed share."""
    cands = [os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'models')]
    try:
        from ament_index_python.packages import get_package_share_directory
        cands.append(os.path.join(get_package_share_directory('crater_sim'), 'models'))
    except Exception:
        pass
    for c in cands:
        path = os.path.join(c, 'lunar_rocks', 'meshes', 'index.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError('rock meshes missing: run tools/gen_rocks.py')


def rock_size(rng):
    """Footprint [m]: mostly pebbles and small rocks, a long tail of boulders."""
    u = rng.random()
    return rng.uniform(0.2, 0.45) if u < 0.45 else rng.uniform(0.45, 0.9) if u < 0.85 else rng.uniform(0.9, 1.5)


def rock_sdf(name, x, y, size, rng, meshes, label=True):
    """One rock: a library mesh scaled (non-uniformly) to `size`, randomly rotated,
    tilted and partly buried. Returns (sdf, ground-truth dict)."""
    m = rng.choice(meshes)
    sx, sy, sz = size, size * rng.uniform(0.8, 1.25), size * rng.uniform(0.75, 1.3)
    ex, ey, ez = (m['extent'][0] * sx, m['extent'][1] * sy, m['extent'][2] * sz)
    roll, pitch, yaw = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-math.pi, math.pi)
    z = -rng.uniform(0.10, 0.35) * ez - 0.5 * max(abs(roll) * ey, abs(pitch) * ex) * 0.5
    g = rng.uniform(0.22, 0.55)
    col = (round(g * rng.uniform(0.97, 1.05), 3), round(g, 3), round(g * rng.uniform(0.9, 1.0), 3))
    top = z + ez
    coll = ''
    if top > 0.12:          # pebbles stay visual-only: the wheels roll over them
        coll = (f'<collision name="c"><pose>0 0 {ez / 2:.3f} 0 0 0</pose>'
                f'<geometry><box><size>{ex * 0.85:.3f} {ey * 0.85:.3f} {ez:.3f}</size></box></geometry></collision>')
    sdf = f"""
    <model name="{name}"><static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} {roll:.3f} {pitch:.3f} {yaw:.3f}</pose>
      <link name="link">{coll}
        <visual name="v"><geometry><mesh><uri>model://lunar_rocks/meshes/{m['file']}</uri>
          <scale>{sx:.3f} {sy:.3f} {sz:.3f}</scale></mesh></geometry>
          {material(*col, spec=0.04)}</visual>
      </link>
      {label_plugin('rock') if label else ''}
    </model>"""
    return sdf, {'name': name, 'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 3),
                 'extent': [round(ex, 3), round(ey, 3), round(ez, 3)], 'top_z': round(top, 3), 'mesh': m['file']}


def place_rocks(rng, n, rover_yaw, craters):
    """Scatter n rocks: ~70 % free field, the rest as ejecta clusters around crater rims
    and big boulders. Keeps the spawn, station and patrol waypoints clear."""
    meshes = load_rock_meshes()
    c, s = math.cos(rover_yaw), math.sin(rover_yaw)
    keep = [(0.0, 0.0, ROCK_PAD_KEEPOUT), (STATION_POS[0], STATION_POS[1], 1.8)]
    keep += [(c * wx - s * wy, s * wx + c * wy, 1.6) for wx, wy in PATROL_WPS]
    out, truth, big = [], [], []

    def ok(x, y, size):
        return (all(math.hypot(x - kx, y - ky) > kr + size / 2 for kx, ky, kr in keep)
                and all(math.hypot(x - t['x'], y - t['y']) > (size + max(t['extent'][:2])) / 2 + 0.15 for t in truth)
                and all(abs(math.hypot(x - cr['x'], y - cr['y']) - cr['radius']) > 0.1 for cr in craters))

    def add(x, y, size):
        sdf, gt = rock_sdf(f'rock_{len(truth)}', x, y, size, rng, meshes)
        out.append(sdf)
        truth.append(gt)
        if size > 0.8:
            big.append((x, y, size))

    n_field = int(n * 0.7)
    for _ in range(n_field * 30):
        if len(truth) >= n_field:
            break
        a, d, size = rng.uniform(0, 2 * math.pi), math.sqrt(rng.uniform(0.02, 1)) * ROCK_FIELD_R, rock_size(rng)
        x, y = d * math.cos(a), d * math.sin(a)
        if ok(x, y, size):
            add(x, y, size)
    anchors = [(cr['x'], cr['y'], cr['radius'] * 1.15) for cr in craters] + [(x, y, s * 0.9) for x, y, s in big]
    for _ in range(n * 30):
        if len(truth) >= n or not anchors:
            break
        ax, ay, ar = rng.choice(anchors)
        a, d, size = rng.uniform(0, 2 * math.pi), ar + rng.uniform(0.1, 1.6), rng.uniform(0.12, 0.45)
        x, y = ax + d * math.cos(a), ay + d * math.sin(a)
        if ok(x, y, size):
            add(x, y, size)
    return ''.join(out), truth


def albedo_patches(rng, n):
    out = ''
    for k in range(n):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(PAD_KEEPOUT, FIELD_R)
        r = rng.uniform(0.6, 2.5)
        g = rng.choice([rng.uniform(0.28, 0.34), rng.uniform(0.46, 0.52)])
        out += f"""
    <model name="albedo_{k}"><static>true</static>
      <pose>{d*math.cos(a):.3f} {d*math.sin(a):.3f} 0.002 0 0 0</pose>
      <link name="link"><visual name="v"><geometry><cylinder><radius>{r:.3f}</radius><length>0.002</length></cylinder></geometry>
        {material(g, g, g)}</visual></link></model>"""
    return out


def station():
    sx, sy = STATION_POS
    legs = ''
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        legs += (f'<visual name="leg{k}"><pose>{0.75*math.cos(a):.3f} {0.75*math.sin(a):.3f} 0.45 '
                 f'{-0.35*math.sin(a):.3f} {0.35*math.cos(a):.3f} 0</pose>'
                 f'<geometry><cylinder><radius>0.04</radius><length>1.0</length></cylinder></geometry>'
                 f'{material(0.7, 0.7, 0.7, spec=0.5)}</visual>')
    return f"""
    <model name="landing_pad"><static>true</static>
      <pose>0 0 0.001 0 0 0</pose>
      <link name="link"><visual name="v"><geometry><cylinder><radius>1.6</radius><length>0.002</length></cylinder></geometry>
        {material(0.55, 0.55, 0.52)}</visual></link></model>
    <model name="station"><static>true</static>
      <pose>{sx} {sy} 0 0 0 0</pose>
      <link name="link">
        <collision name="c"><pose>0 0 0.95 0 0 0</pose><geometry><box><size>1.4 1.4 1.0</size></box></geometry></collision>
        <visual name="body"><pose>0 0 0.95 0 0 0</pose><geometry><box><size>1.1 1.1 0.9</size></box></geometry>
          {material(0.85, 0.7, 0.2, spec=0.8)}</visual>
        <visual name="antenna"><pose>0 0 1.75 0 0 0</pose><geometry><cylinder><radius>0.3</radius><length>0.05</length></cylinder></geometry>
          {material(0.9, 0.9, 0.9, spec=0.6)}</visual>
        {legs}
      </link>
      {label_plugin('station')}
    </model>"""


def build(args):
    rng = random.Random(args.seed)
    placed, craters, truth = [(STATION_POS[0], STATION_POS[1], 1.0)], [], []
    for i in range(args.craters):
        r = rng.uniform(0.8, 2.2)
        for _ in range(400):
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(PAD_KEEPOUT + r, FIELD_R)
            x, y = d * math.cos(a), d * math.sin(a)
            if all(math.hypot(x - px, y - py) > r + pr + 1.5 for px, py, pr in placed):
                placed.append((x, y, r))
                craters.append(crater_sdf(f'crater_{i}', x, y, r, rng))
                truth.append({'name': f'crater_{i}', 'x': x, 'y': y, 'radius': r})
                break
    rocks, rock_truth = place_rocks(rng, args.rocks, args.rover_yaw, truth)

    el, az = math.radians(args.sun_elev), math.radians(args.sun_az)
    sun_dir = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
    world = f"""<?xml version="1.0" ?>
<!-- Generated by tools/gen_world.py --seed {args.seed} --sun-elev {args.sun_elev} --sun-az {args.sun_az} -->
<sdf version="1.9">
  <world name="lunar_crater">
    <physics name="4ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <gravity>0 0 -1.62</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>

    <scene>
      <ambient>{args.ambient} {args.ambient} {args.ambient} 1</ambient>
      <background>0 0 0 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 30 0 0 0</pose>
      <diffuse>1.0 0.98 0.95 1</diffuse>
      <specular>0.4 0.4 0.4 1</specular>
      <direction>{sun_dir[0]:.3f} {sun_dir[1]:.3f} {sun_dir[2]:.3f}</direction>
    </light>

    <gui fullscreen="0">
      <camera name="user_camera"><pose>-14 -14 11 0 0.5 0.785</pose></camera>
    </gui>

    <model name="regolith"><static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          {material(0.4, 0.4, 0.39, spec=0.02)}</visual>
      </link>
    </model>
    {station()}
    {albedo_patches(rng, args.patches)}
    {''.join(craters)}
    {rocks}

    <include>
      <uri>model://rover</uri>
      <name>crater_rover</name>
      <pose>0 0 0.25 0 0 {args.rover_yaw}</pose>
    </include>
  </world>
</sdf>
"""
    return world, truth, rock_truth


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'worlds', 'lunar_crater.sdf'))
    ap.add_argument('--craters', type=int, default=14)
    ap.add_argument('--patches', type=int, default=20)
    ap.add_argument('--rocks', type=int, default=80)
    ap.add_argument('--sun-elev', type=float, default=15.0, help='sun elevation [deg]; low = long shadows')
    ap.add_argument('--sun-az', type=float, default=35.0, help='sun azimuth [deg]')
    ap.add_argument('--ambient', type=float, default=0.12)
    ap.add_argument('--rover-yaw', type=float, default=0.0)
    ap.add_argument('--randomize', action='store_true',
                    help='draw sun, ambient light, crater/rock counts and rover yaw from --seed (dataset worlds)')
    args = ap.parse_args()
    if args.randomize:
        r = random.Random(args.seed * 7919)
        args.sun_elev, args.sun_az = r.uniform(6, 45), r.uniform(0, 360)
        args.ambient = r.uniform(0.05, 0.2)
        args.craters, args.patches = r.randint(8, 18), r.randint(10, 26)
        args.rocks = r.randint(110, 220)
        args.rover_yaw = r.uniform(-math.pi, math.pi)

    world, truth, rocks = build(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(world)
    gt_path = os.path.splitext(args.out)[0] + '_ground_truth.json'
    with open(gt_path, 'w') as f:
        json.dump({'seed': args.seed, 'station': STATION_POS, 'rover_yaw': args.rover_yaw,
                   'craters': truth, 'rocks': rocks}, f, indent=1)
    print(f'wrote {args.out} ({len(truth)} craters, {len(rocks)} rocks)\nwrote {gt_path}')


if __name__ == '__main__':
    main()
