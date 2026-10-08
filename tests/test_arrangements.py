"""Arrangement presets: valid boxes, a content rect inside what the projectors cover, an overlap it crosses."""

import numpy as np
import pytest

from sim.arrangements import PRESETS, from_config, largest_rect_in_union, side_by_side
from sim.calibration import CalibrationSetup
from sim.planar import clip_convex, homography_from_points, is_convex, points_in_convex, raster_corners, rect_polygon

SCREEN = (4000.0, 1500.0)
RES = {"a": (1920, 1080), "b": (1920, 1080)}


def _setup(arrangement) -> CalibrationSetup:
    h = {n: homography_from_points(raster_corners(RES[n]), arrangement.corners[n]) for n in RES}
    return CalibrationSetup(h_cal=h, resolution=RES, content_rect_mm=arrangement.content_rect_mm)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_is_a_valid_overlapping_installation(preset):
    arr = from_config({"preset": preset}, SCREEN, RES)
    a, b = arr.corners["a"], arr.corners["b"]
    assert is_convex(a) and is_convex(b)
    x0, y0, x1, y1 = arr.content_rect_mm
    if preset != "corner":  # the corner arrangement fills its union's bounding box on purpose
        xs, ys = np.meshgrid(np.linspace(x0, x1, 81), np.linspace(y0, y1, 41))
        pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)
        assert np.all(points_in_convex(a, pts, margin=-1e-6) | points_in_convex(b, pts, margin=-1e-6))
    setup = _setup(arr)
    assert len(clip_convex(clip_convex(a, b), rect_polygon(*arr.content_rect_mm))) >= 3
    assert all(setup.inner_edges()[n] for n in RES)
    union = np.vstack([a, b])
    centre = (union.min(axis=0) + union.max(axis=0)) / 2
    assert np.allclose(centre, np.array(SCREEN) / 2, atol=1e-9)  # every pair is centred on the screen


def test_side_by_side_defaults_are_the_demo_installation():
    arr = from_config({"preset": "side_by_side"}, SCREEN, RES)
    assert np.allclose(arr.corners["a"], rect_polygon(205.0, 187.35, 2205.0, 1312.35))
    assert np.allclose(arr.corners["b"], rect_polygon(1795.0, 187.65, 3795.0, 1312.65))
    assert np.allclose(arr.content_rect_mm, (205.0, 187.65, 3795.0, 1312.35))


def test_stacked_overlap_is_a_horizontal_band():
    arr = from_config({"preset": "stacked"}, SCREEN, RES)
    a, b = arr.corners["a"], arr.corners["b"]
    assert b[:, 1].min() == pytest.approx(a[:, 1].max() - 200.0)  # B starts 200 mm above A's bottom
    assert arr.content_rect_mm[1] == pytest.approx(a[:, 1].min()) and arr.content_rect_mm[3] == pytest.approx(b[:, 1].max())


def test_different_sizes_are_vertically_centred():
    arr = from_config({"preset": "different_sizes"}, SCREEN, RES)
    a, b = arr.corners["a"], arr.corners["b"]
    assert np.ptp(b[:, 0]) == pytest.approx(1600.0) and np.ptp(b[:, 1]) == pytest.approx(900.0)
    assert a[:, 1].mean() == pytest.approx(b[:, 1].mean())


def test_rotated_content_rect_is_inside_and_nearly_as_large_as_possible():
    arr = from_config({"preset": "rotated", "angle_deg": 2.0}, SCREEN, RES)
    x0, y0, x1, y1 = arr.content_rect_mm
    b = arr.corners["b"]
    # B's rotated corners eat into the side-by-side rect; the search keeps almost the full width.
    assert x1 - x0 > 3400 and y1 - y0 > 1000
    assert y0 >= b[:, 1].min() - 1e-9  # never above B's highest corner


def test_numeric_rect_matches_the_exact_one_for_axis_aligned_boxes():
    exact = side_by_side(SCREEN, RES, 2000.0, 2000.0, 410.0, 0.3)
    numeric = largest_rect_in_union(list(exact.corners.values()), step_mm=1.0)
    assert np.allclose(numeric, exact.content_rect_mm, atol=1.0)
    x0, y0, x1, y1 = numeric
    e0, f0, e1, f1 = exact.content_rect_mm
    assert x0 >= e0 - 1e-9 and y0 >= f0 - 1e-9 and x1 <= e1 + 1e-9 and y1 <= f1 + 1e-9  # conservative


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"preset": "side_by_side", "width_mm": 3000}, "does not fit"),
        ({"preset": "side_by_side", "overlap_mm": 2500}, "overlap must be positive"),
        ({"preset": "rotated", "tilt_deg": 1}, "unknown keys"),
        ({"preset": "spiral"}, "preset must be one of"),
        ({"preset": "explicit", "corners_mm": {"a": [[0, 0], [1, 0], [1, 1], [0, 1]]}}, "needs corners_mm and content_rect_mm"),
    ],
)
def test_bad_arrangements_fail_clearly(cfg, message):
    with pytest.raises(ValueError, match=message):
        from_config(cfg, SCREEN, RES)
