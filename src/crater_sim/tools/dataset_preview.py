#!/usr/bin/env python3
"""Contact sheet of auto-labelled dataset frames with their YOLO boxes drawn on, plus
per-class label counts. Used to check that the labels line up with the objects.

    dataset_preview.py ~/crater_ws/dataset/raw/train --out ~/crater_ws/dataset/preview/train.png
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

NAMES = ['rock', 'meteor', 'crater', 'station']
COLORS = [(90, 220, 120), (30, 140, 255), (200, 200, 200), (60, 200, 250)]   # BGR


def read_labels(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(int(p[0]), *map(float, p[1:5])) for p in (ln.split() for ln in f) if len(p) == 5]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('split', help='dataset split dir containing images/ and labels/')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--tile', type=int, default=320, help='tile width [px]')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--require', default='', help='only frames containing this class (e.g. rock)')
    args = ap.parse_args()

    images = sorted(glob.glob(os.path.join(args.split, 'images', '*.jpg')))
    lab = {im: os.path.join(args.split, 'labels', os.path.splitext(os.path.basename(im))[0] + '.txt') for im in images}
    counts, frames_with = np.zeros(len(NAMES), int), np.zeros(len(NAMES), int)
    for im in images:
        cls = [c for c, *_ in read_labels(lab[im])]
        for c in cls:
            counts[c] += 1
        for c in set(cls):
            frames_with[c] += 1
    print(f'{args.split}: {len(images)} images')
    for i, n in enumerate(NAMES):
        print(f'  {n:8s} {counts[i]:6d} boxes in {frames_with[i]:5d} images')

    pool = images
    if args.require:
        k = NAMES.index(args.require)
        pool = [im for im in images if any(c == k for c, *_ in read_labels(lab[im]))]
    pick = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    tw, th = args.tile, args.tile * 3 // 4
    rows = (len(pick) + args.cols - 1) // args.cols
    sheet = np.zeros((rows * th, args.cols * tw, 3), np.uint8)
    for k, im in enumerate(pick):
        img = cv2.resize(cv2.imread(im), (tw, th), interpolation=cv2.INTER_AREA)
        for c, cx, cy, w, h in read_labels(lab[im]):
            p0 = (int((cx - w / 2) * tw), int((cy - h / 2) * th))
            p1 = (int((cx + w / 2) * tw), int((cy + h / 2) * th))
            cv2.rectangle(img, p0, p1, COLORS[c], 1, cv2.LINE_AA)
            if w * tw > 18:
                cv2.putText(img, NAMES[c], (p0[0], max(9, p0[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            COLORS[c], 1, cv2.LINE_AA)
        cv2.putText(img, os.path.basename(im), (4, th - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        r, c = divmod(k, args.cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = img
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
