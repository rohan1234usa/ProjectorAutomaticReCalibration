"""Simulator geometry: homographies, convex polygons, and the bilinear warp's sub-pixel accuracy."""

import numpy as np
import pytest

from sim.planar import (
    apply_h,
    box_mm,
    clip_convex,
    homography_from_points,
    is_convex,
    jacobian_det,
    points_in_convex,
    raster_corners,
    rect_polygon,
    segment_distance,
    signed_area,
    warp_linear,
)


def _random_quad(rng, center, size):
    """A convex quadrilateral: a square jittered by up to 15% of its size per corner."""
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * size / 2
    return center + base + rng.uniform(-0.15, 0.15, (4, 2)) * size


def test_homography_maps_corners_exactly():
    rng = np.random.default_rng(0)
    for _ in range(50):
        res = (int(rng.integers(320, 4096)), int(rng.integers(200, 2160)))
        dst = _random_quad(rng, np.array([2000.0, 800.0]), 2000.0)
        h = homography_from_points(raster_corners(res), dst)
        assert np.abs(box_mm(h, res) - dst).max() < 1e-9  # mm


def test_homography_inverse_round_trip():
    rng = np.random.default_rng(1)
    h = homography_from_points(raster_corners((1920, 1080)), _random_quad(rng, np.array([1000.0, 600.0]), 1500.0))
    pts = rng.uniform(0, 1920, (100, 2))
    assert np.abs(apply_h(np.linalg.inv(h), apply_h(h, pts)) - pts).max() < 1e-9


def test_jacobian_det_of_pure_scale_and_projective_map():
    s = np.diag([2.0, 3.0, 1.0])
    assert np.allclose(jacobian_det(s, np.array([5.0]), np.array([7.0])), 6.0)
    # Numerical check on a projective map: area of a tiny square's image / its area.
    h = homography_from_points(raster_corners((100, 100)), np.array([[0, 0], [120, 10], [110, 90], [-5, 100.0]]))
    u, v, eps = 30.0, 60.0, 1e-4
    quad = apply_h(h, np.array([[u, v], [u + eps, v], [u + eps, v + eps], [u, v + eps]]))
    assert jacobian_det(h, u, v) == pytest.approx(abs(signed_area(quad)) / eps**2, rel=1e-5)


def test_clip_convex_areas():
    a = rect_polygon(0, 0, 2, 2)
    assert abs(signed_area(clip_convex(a, rect_polygon(1, 1, 3, 3)))) == pytest.approx(1.0)
    assert len(clip_convex(a, rect_polygon(5, 5, 6, 6))) == 0
    # A small rotated square fully inside is returned unchanged in area.
    t = np.radians(30)
    small = np.array([[np.cos(t + k * np.pi / 2), np.sin(t + k * np.pi / 2)] for k in range(4)]) * 0.5 + 1
    assert abs(signed_area(clip_convex(small, a))) == pytest.approx(abs(signed_area(small)))
    # Touching edges collapse to an empty polygon rather than a degenerate sliver.
    assert len(clip_convex(a, rect_polygon(2, 0, 4, 2))) == 0


def test_clip_convex_random_quads_is_convex_and_inside_both():
    rng = np.random.default_rng(2)
    for _ in range(200):
        p = _random_quad(rng, np.array([0.0, 0.0]), 10.0)
        q = _random_quad(rng, rng.uniform(-8, 8, 2), rng.uniform(4, 14))
        o = clip_convex(p, q)
        if len(o) == 0:
            continue
        assert is_convex(o)
        assert points_in_convex(p, o, margin=-1e-9).all() and points_in_convex(q, o, margin=-1e-9).all()


def test_points_in_convex_and_segment_distance():
    sq = rect_polygon(0, 0, 10, 10)
    pts = np.array([[5, 5], [0, 5], [-0.1, 5], [5, 10.1]], dtype=float)
    assert points_in_convex(sq, pts).tolist() == [True, True, False, False]
    assert points_in_convex(sq[::-1], pts).tolist() == [True, True, False, False]  # orientation-free
    d = segment_distance(np.array([[0, 1.0], [5, 2.0], [12, 0.0]]), np.array([0, 0.0]), np.array([10, 0.0]))
    assert np.allclose(d, [1.0, 2.0, 2.0])


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_warp_is_continuous_subpixel(channels):
    """Canary: a 0.01 px shift must come out as 0.01 px (OpenCV < 5 rounds to 1/32 px)."""
    ramp = np.tile(np.arange(64, dtype=np.float32), (8, 1))
    src = ramp if channels == 1 else np.repeat(ramp[:, :, None], channels, axis=2)
    for shift in (0.01, 0.37):
        out = warp_linear(src, np.array([[1, 0, shift], [0, 1, 0], [0, 0, 1.0]]), (64, 8))
        row = out[4] if channels == 1 else out[4, :, 0]
        measured = np.mean(np.arange(64)[5:-5] - row[5:-5])
        assert measured == pytest.approx(shift, abs=1e-4)


def test_warp_rejects_types_that_round():
    with pytest.raises(TypeError):
        warp_linear(np.zeros((4, 4)), np.eye(3), (4, 4))
    with pytest.raises(ValueError):
        warp_linear(np.zeros((4, 4, 2), np.float32), np.eye(3), (4, 4))
