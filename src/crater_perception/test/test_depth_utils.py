import math

import numpy as np

from crater_perception.depth_utils import depth_to_metres, roi_depth


def scene(z=5.0, bg=30.0):
    d = np.full((100, 100), bg, np.float32)
    d[40:60, 40:60] = z            # the object
    return d


def test_median_inside_box():
    z, ok, n, _ = roi_depth(scene(), (40, 40, 60, 60))
    assert ok and abs(z - 5.0) < 1e-6 and n > 0


def test_roi_ignores_background_at_box_edges():
    # box 30 % larger than the object: the central 60 % is still all object
    z, ok, _, _ = roi_depth(scene(), (37, 37, 63, 63))
    assert ok and abs(z - 5.0) < 1e-6


def test_nan_inf_and_out_of_range_are_rejected():
    d = scene()
    # ROI = rows/cols 44..56; corrupt half of its rows, all with values that would
    # drag the median away from 5 m if they were kept
    d[44:46, 40:60] = np.nan
    d[46:47, 40:60] = np.inf
    d[47:48, 40:60] = -np.inf
    d[48:49, 40:60] = 0.05         # closer than min_range
    d[49:50, 40:60] = 60.0         # at the far clip
    z, ok, n, ratio = roi_depth(d, (40, 40, 60, 60))
    assert ok and abs(z - 5.0) < 1e-6 and 0.4 < ratio < 0.6


def test_median_is_robust_to_outliers():
    d = scene()
    d[45:47, 45:55] = 0.5          # a few spurious near returns
    z, ok, _, _ = roi_depth(d, (40, 40, 60, 60))
    assert ok and abs(z - 5.0) < 1e-6


def test_invalid_when_mostly_bad():
    d = scene()
    d[40:60, 40:60] = np.nan
    z, ok, n, ratio = roi_depth(d, (40, 40, 60, 60))
    assert not ok and math.isnan(z) and n == 0 and ratio == 0.0


def test_empty_or_outside_box():
    z, ok, n, _ = roi_depth(scene(), (150, 150, 160, 160))
    assert not ok and math.isnan(z) and n == 0
    z, ok, _, _ = roi_depth(scene(), (50, 50, 50, 50))
    assert not ok


def test_tiny_box_accepts_few_pixels():
    d = scene()
    z, ok, _, _ = roi_depth(d, (48, 48, 53, 53))
    assert ok and abs(z - 5.0) < 1e-6


def test_16uc1_millimetres():
    raw = np.array([[0, 5000], [1234, 65535]], np.uint16)
    m = depth_to_metres(raw, '16UC1')
    assert math.isnan(m[0, 0]) and abs(m[0, 1] - 5.0) < 1e-6 and abs(m[1, 0] - 1.234) < 1e-6
