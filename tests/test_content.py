"""Content sources: deterministic, in range, black border where asked, features crossing the middle."""

import numpy as np
import pytest

from sim.content import make_content


def test_slide_is_deterministic_per_seed():
    size = (900, 280)
    a = make_content({"type": "slide"}, size, np.random.default_rng(5))
    b = make_content({"type": "slide"}, size, np.random.default_rng(5))
    c = make_content({"type": "slide"}, size, np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (280, 900, 3) and a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_slide_features_cross_the_middle():
    """Seams only show where content crosses the overlap, so the middle third must hold dark detail."""
    img = make_content({"type": "slide"}, (1500, 450), np.random.default_rng(0))
    middle = img[:, 500:1000].mean(axis=2)
    assert (middle < 0.3).mean() > 0.005 and (middle > 0.85).mean() > 0.3
    assert np.ptp(img[:, 500:1000], axis=(0, 1)).min() > 0.5


def test_border_is_exactly_black():
    img = make_content({"type": "flat", "value": 0.7, "border_frac": 0.1}, (200, 100), np.random.default_rng(0))
    assert img[:10].max() == 0 and img[-10:].max() == 0 and img[:, :10].max() == 0 and img[:, -10:].max() == 0
    assert np.all(img[10:-10, 10:-10] == np.float32(0.7))


def test_bad_content_config_is_rejected():
    with pytest.raises(ValueError):
        make_content({"type": "slides"}, (10, 10), np.random.default_rng(0))
    with pytest.raises(ValueError):
        make_content({"type": "flat", "colour": 1}, (10, 10), np.random.default_rng(0))
