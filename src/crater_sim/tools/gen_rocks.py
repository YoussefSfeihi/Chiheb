#!/usr/bin/env python3
"""Procedural lunar rock meshes for C.R.A.T.E.R (pure numpy, no extra dependency).

Each rock starts as an icosphere and is deformed into one of three families:
  * boulder  rounded, lumpy (low-frequency radial noise, a couple of fracture planes)
  * block    angular, fractured (4-7 planar cuts -> flat faces and sharp edges)
  * slab     flat plate (squashed vertically, cut on the sides)
then gets a flattened base so it sits on the regolith. Meshes are normalised to a
max horizontal extent of 1 with the base at z = 0; the world generator scales them
(non-uniformly) to the rock's size in metres.

Writes <out>/rock_XX.obj and <out>/index.json ({file, family, extent}).

    gen_rocks.py --count 48 --seed 3 --out models/lunar_rocks/meshes
"""
import argparse
import json
import os

import numpy as np

FAMILIES = ('boulder', 'block', 'slab')


def icosphere(subdiv=3):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(verts), np.array(f, dtype=np.int64)


def _unit(rng, n=1):
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def make_rock(rng, family):
    """Return (vertices Nx3, faces Mx3) of one rock, base at z=0, max horizontal extent 1."""
    dirs, faces = icosphere(3)
    # radial lumps: a few low-frequency waves over the sphere + fine roughness
    r = np.ones(len(dirs))
    amp = {'boulder': 0.10, 'block': 0.06, 'slab': 0.05}[family]
    for d, fr, ph in zip(_unit(rng, 5), rng.uniform(1.5, 3.5, 5), rng.uniform(0, 2 * np.pi, 5)):
        r += amp * rng.uniform(0.4, 1.0) * np.cos(fr * dirs @ d + ph)
    for d, fr, ph in zip(_unit(rng, 6), rng.uniform(6, 11, 6), rng.uniform(0, 2 * np.pi, 6)):
        r += 0.012 * np.cos(fr * dirs @ d + ph)
    v = dirs * r[:, None]
    # anisotropic proportions
    sy = rng.uniform(0.6, 1.0)
    sz = {'boulder': rng.uniform(0.55, 0.85), 'block': rng.uniform(0.5, 0.9), 'slab': rng.uniform(0.2, 0.35)}[family]
    v *= np.array([1.0, sy, sz])
    # fracture planes: everything beyond the plane is projected onto it -> flat faces
    n_cuts = {'boulder': rng.integers(0, 3), 'block': rng.integers(4, 8), 'slab': rng.integers(2, 5)}[family]
    for n in _unit(rng, int(n_cuts)):
        if family == 'slab':                       # cut the rim, not the faces
            n[2] *= 0.2
            n /= np.linalg.norm(n)
        h = v @ n
        cut = h.max() * rng.uniform(0.55, 0.85)
        over = h > cut
        v[over] -= (h[over] - cut)[:, None] * n
    # flattened base (resting surface)
    z0 = v[:, 2].min()
    zb = z0 + rng.uniform(0.08, 0.18) * (v[:, 2].max() - z0)
    v[:, 2] = np.maximum(v[:, 2], zb)
    # normalise: centre in xy, base at z=0, max horizontal extent 1
    lo, hi = v.min(axis=0), v.max(axis=0)
    v -= np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    v /= max(hi[0] - lo[0], hi[1] - lo[1])
    return v, faces


def face_normals(v, f):
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(ln > 1e-12, ln, 1.0)


def write_obj(path, v, f, smooth):
    fn = face_normals(v, f)
    with open(path, 'w') as out:
        out.write('# C.R.A.T.E.R procedural rock (tools/gen_rocks.py)\n')
        out.writelines(f'v {x:.5f} {y:.5f} {z:.5f}\n' for x, y, z in v)
        if smooth:          # area-weighted vertex normals: rounded look
            vn = np.zeros_like(v)
            raw = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            for k in range(3):
                np.add.at(vn, f[:, k], raw)
            vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
            out.writelines(f'vn {x:.4f} {y:.4f} {z:.4f}\n' for x, y, z in vn)
            out.writelines(f'f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n' for a, b, c in f)
        else:               # one normal per face: faceted, fractured look
            out.writelines(f'vn {x:.4f} {y:.4f} {z:.4f}\n' for x, y, z in fn)
            out.writelines(f'f {a+1}//{i+1} {b+1}//{i+1} {c+1}//{i+1}\n' for i, (a, b, c) in enumerate(f))


def generate(out, count, seed):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        v, f = make_rock(rng, family)
        name = f'rock_{i:02d}.obj'
        write_obj(os.path.join(out, name), v, f, smooth=family == 'boulder')
        index.append({'file': name, 'family': family, 'extent': [round(float(e), 4) for e in v.max(axis=0) - v.min(axis=0)]})
    with open(os.path.join(out, 'index.json'), 'w') as fh:
        json.dump(index, fh, indent=1)
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--count', type=int, default=48)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'models', 'lunar_rocks', 'meshes'))
    args = ap.parse_args()
    index = generate(args.out, args.count, args.seed)
    print(f'wrote {len(index)} rock meshes to {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
