#!/usr/bin/env python3
"""Score the meteor pipeline live: onboard predictions vs what really happened.

Listens to
  /meteors/tracks   onboard RGB-D tracker (what the rover believed)
  /sim/impacts      simulator ground truth (where each meteor really landed and
                    how far the rover really was: HIT / SAFE)
Each real impact is matched to every onboard track (fragment) whose final predicted
impact time and point agree with it; lead time comes from the earliest sighting.

Per meteor: detected?, detection lead time, impact-prediction error 3 s before
impact and at the last prediction, flagged as threat?, true outcome.

Usage (while the sim runs):
    ros2 run crater_sim eval_meteors.py --count 6          # stop after 6 impacts
    ros2 run crater_sim eval_meteors.py --duration 300 --json report.json
"""
import argparse
import json
import math
import time
from collections import defaultdict

import rclpy
from crater_interfaces.msg import MeteorTrackArray
from rclpy.node import Node
from std_msgs.msg import String


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--count', type=int, default=5, help='stop after this many impacts')
    ap.add_argument('--duration', type=float, default=600.0, help='wall-clock limit [s]')
    ap.add_argument('--json')
    args = ap.parse_args()

    rclpy.init()
    node = Node('eval_meteors', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
    history = defaultdict(list)    # track id -> [(t, impact_x, impact_y, t_impact_abs, is_threat, first_seen)]
    impacts = []

    def on_tracks(msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for m in msg.tracks:
            if m.time_to_impact > 0:
                fs = m.first_seen.sec + m.first_seen.nanosec * 1e-9
                history[m.id].append((t, m.impact_point.x, m.impact_point.y, t + m.time_to_impact,
                                      m.is_threat, fs, m.miss_distance))

    def on_impact(msg):
        rep = json.loads(msg.data)
        impacts.append(rep)
        print(f'  impact {len(impacts)}/{args.count}: {rep["name"]} -> {rep["verdict"]} '
              f'(rover {rep["rover_distance"]:.1f} m away)', flush=True)

    node.create_subscription(MeteorTrackArray, '/meteors/tracks', on_tracks, 50)
    node.create_subscription(String, '/sim/impacts', on_impact, 10)
    end = time.time() + args.duration
    print(f'waiting for {args.count} impacts ...', flush=True)
    while time.time() < end and len(impacts) < args.count:
        rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.shutdown()

    rows = []
    used = set()
    for rep in impacts:
        ix, iy = rep['impact']
        t_imp = rep['impact_time']
        # every onboard track (fragment) whose final prediction matches this impact
        tids = [tid for tid, h in history.items() if tid not in used
                and abs(h[-1][3] - t_imp) < 2.5 and math.hypot(h[-1][1] - ix, h[-1][2] - iy) < 4.0]
        used.update(tids)
        row = {'meteor': rep['name'], 'verdict': rep['verdict'], 'rover_distance': rep['rover_distance'],
               'aim_offset': rep['aim_offset'], 'detected': bool(tids), 'tracks': tids}
        if tids:
            h = sorted((e for tid in tids for e in history[tid]), key=lambda e: e[0])
            row['lead_time_s'] = t_imp - min(e[5] for e in h)
            row['flagged_threat'] = any(e[4] for e in h)
            early = [e for e in h if t_imp - e[0] >= 3.0]
            if early:
                e = early[-1]
                row['pred_err_3s_m'] = math.hypot(e[1] - ix, e[2] - iy)
            row['pred_err_final_m'] = math.hypot(h[-1][1] - ix, h[-1][2] - iy)
        rows.append(row)

    print(f'\n{"meteor":10s} {"det":>4s} {"lead s":>7s} {"err@3s":>7s} {"err fin":>8s} {"threat?":>8s} '
          f'{"aimed":>6s} {"rover dist":>10s} {"outcome":>8s}')
    for r in rows:
        f = lambda k, fmt: (fmt % r[k]) if k in r else '-'  # noqa: E731
        print(f'{r["meteor"]:10s} {"yes" if r["detected"] else "NO":>4s} {f("lead_time_s", "%.1f"):>7s} '
              f'{f("pred_err_3s_m", "%.2f"):>7s} {f("pred_err_final_m", "%.2f"):>8s} '
              f'{("yes" if r.get("flagged_threat") else "no") if r["detected"] else "-":>8s} '
              f'{r["aim_offset"]:5.1f}m {r["rover_distance"]:9.1f}m {r["verdict"]:>8s}')
    det = [r for r in rows if r['detected']]
    e3 = [r['pred_err_3s_m'] for r in det if 'pred_err_3s_m' in r]
    summary = {
        'impacts': len(rows), 'detected': len(det),
        'mean_lead_time_s': sum(r['lead_time_s'] for r in det) / len(det) if det else None,
        'mean_pred_err_3s_m': sum(e3) / len(e3) if e3 else None,
        'hits_on_rover': sum(1 for r in rows if r['verdict'] == 'HIT'),
    }
    print(f'\ndetected {summary["detected"]}/{summary["impacts"]}, mean lead time '
          f'{summary["mean_lead_time_s"] or 0:.1f} s, mean impact-prediction error 3 s before impact '
          f'{summary["mean_pred_err_3s_m"] or 0:.2f} m, rover hit {summary["hits_on_rover"]} time(s)')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'summary': summary, 'meteors': rows}, fh, indent=1)


if __name__ == '__main__':
    main()
