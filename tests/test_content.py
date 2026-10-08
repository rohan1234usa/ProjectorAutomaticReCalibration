"""Content sources: deterministic, in range, black border where asked, features crossing the middle."""

import numpy as np
import pytest

from sim.content import DENSITIES, flat, framed, inner_size, slide


def test_slide_is_deterministic_per_seed():
    size = (900, 280)
    a = slide(size, np.random.default_rng(5))
    b = slide(size, np.random.default_rng(5))
    c = slide(size, np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (280, 900, 3) and a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_slide_features_cross_the_middle():
    """Seams only show where content crosses the overlap, so the middle third must hold dark detail."""
    img = slide((1500, 450), np.random.default_rng(0))
    middle = img[:, 500:1000].mean(axis=2)
    assert (middle < 0.3).mean() > 0.005 and (middle > 0.85).mean() > 0.3
    assert np.ptp(img[:, 500:1000], axis=(0, 1)).min() > 0.5


def test_border_is_exactly_black():
    img = framed((200, 100), 0.1, lambda inner: flat(inner, 0.7))
    assert inner_size((200, 100), 0.1) == (180, 80)
    assert img[:10].max() == 0 and img[-10:].max() == 0 and img[:, :10].max() == 0 and img[:, -10:].max() == 0
    assert np.all(img[10:-10, 10:-10] == np.float32(0.7))


def test_bad_content_is_rejected():
    with pytest.raises(ValueError, match="density"):
        slide((10, 10), np.random.default_rng(0), "huge")
    assert set(DENSITIES) == {"low", "medium", "high"}
    for frac in (-0.05, 0.5):
        with pytest.raises(ValueError, match="border_frac"):
            framed((200, 100), frac, lambda inner: flat(inner, 0.5))
    with pytest.raises(ValueError, match="leaves no picture"):
        inner_size((10, 100), 0.1)  # a 10 px border on a 10 px wide picture
