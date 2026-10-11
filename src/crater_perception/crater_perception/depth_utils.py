"""Depth helpers for Module 2 (pure numpy, unit-tested in test/test_depth_utils.py)."""
import math

import numpy as np


def depth_to_metres(depth, encoding):
    """32FC1 is already metres (Gazebo); 16UC1 is millimetres (real RGB-D sensors)."""
    if encoding == '16UC1':
        d = depth.astype(np.float32) / 1000.0
        d[depth == 0] = np.nan                     # 0 = no return
        return d
    return depth.astype(np.float32, copy=False)


def roi_depth(depth, box, roi_frac=0.6, min_range=0.1, max_range=60.0,
              min_valid_ratio=0.15, min_valid_px=20):
    """Robust depth of the object inside an image box.

    depth  HxW float array in metres (NaN / +-Inf allowed)
    box    (x0, y0, x1, y1) in pixels
    Only the central `roi_frac` of the box is used (its edges are mostly background).
    NaN, +-Inf, <= min_range and >= max_range (sensor clip) pixels are rejected; the
    depth is the median of the rest.

    -> (depth_m, depth_valid, n_valid, valid_ratio); depth_m is NaN when not valid.
    """
    H, W = depth.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * roi_frac / 2, (y1 - y0) * roi_frac / 2
    c0, c1 = max(0, int(math.floor(cx - hw))), min(W, int(math.ceil(cx + hw)))
    r0, r1 = max(0, int(math.floor(cy - hh))), min(H, int(math.ceil(cy + hh)))
    if c1 <= c0 or r1 <= r0:
        return float('nan'), False, 0, 0.0
    roi = depth[r0:r1, c0:c1]
    ok = np.isfinite(roi) & (roi > min_range) & (roi < max_range)
    n = int(ok.sum())
    ratio = n / roi.size
    # small boxes: accept fewer pixels, but never fewer than 3
    need = min(min_valid_px, max(3, roi.size // 2))
    if n < need or ratio < min_valid_ratio:
        return float('nan'), False, n, ratio
    return float(np.median(roi[ok])), True, n, ratio
