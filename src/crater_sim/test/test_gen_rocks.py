import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from gen_rocks import FAMILIES, generate, make_rock  # noqa: E402


def test_rock_shape_invariants():
    rng = np.random.default_rng(0)
    for fam in FAMILIES * 4:
        v, f = make_rock(rng, fam)
        assert np.isfinite(v).all()
        assert f.min() >= 0 and f.max() < len(v)
        lo, hi = v.min(axis=0), v.max(axis=0)
        assert abs(lo[2]) < 1e-9                                  # base on the ground
        assert abs(max(hi[0] - lo[0], hi[1] - lo[1]) - 1) < 1e-9  # normalised footprint
        assert 0.05 < hi[2] < 1.2
        base = v[:, 2] < 1e-6
        assert base.sum() >= 3                                    # a flat resting face
        if fam == 'slab':
            assert hi[2] < 0.45


def test_deterministic(tmp_path):
    a = generate(str(tmp_path / 'a'), 6, seed=5)
    b = generate(str(tmp_path / 'b'), 6, seed=5)
    assert a == b
    assert (tmp_path / 'a' / 'rock_05.obj').read_text() == (tmp_path / 'b' / 'rock_05.obj').read_text()
