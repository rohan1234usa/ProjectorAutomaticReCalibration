"""The blending setup: blend weights for any overlap shape, and framebuffers built for H_cal."""

import numpy as np
import pytest

from sim.calibration import RAMPS
from sim.planar import apply_h, rect_polygon
from tests.scenes import corner_setup, make_setup, rotated_setup, side_by_side_setup


def _overlap_points(setup, n=20000, seed=0):
    """Random points inside both boxes and the content rect."""
    rng = np.random.default_rng(seed)
    a, b = setup.names
    lo = np.minimum(setup.box_mm(a).min(axis=0), setup.box_mm(b).min(axis=0))
    hi = np.maximum(setup.box_mm(a).max(axis=0), setup.box_mm(b).max(axis=0))
    pts = rng.uniform(lo, hi, (n, 2))
    w = setup.blend_at(pts)
    return pts[(w[a] > 0) & (w[b] > 0)]


@pytest.mark.parametrize("vertical_offset_mm", [0.0, 0.3, -0.3])
def test_side_by_side_blend_is_the_classic_1d_ramp(vertical_offset_mm):
    """Inside the content the ramp depends on x only, even when B sits 0.3 mm higher or lower."""
    setup = side_by_side_setup(overlap_mm=60.0, vertical_offset_mm=vertical_offset_mm)
    box_a, box_b = setup.box_mm("a"), setup.box_mm("b")
    x_left, x_right = box_b[:, 0].min(), box_a[:, 0].max()
    _, y0, _, y1 = setup.content_rect_mm
    x = np.linspace(x_left + 0.01, x_right - 0.01, 41)
    y = np.linspace(y0 + 0.01, y1 - 0.01, 31)
    pts = np.stack(np.meshgrid(x, y), axis=-1)
    w_a = setup.blend_at(pts)["a"]
    expected = RAMPS["cosine"]((x_right - x) / (x_right - x_left))
    assert np.abs(w_a - expected[None, :]).max() < 1e-12


def test_single_coverage_gets_full_weight():
    setup = side_by_side_setup()
    box_a, box_b = setup.box_mm("a"), setup.box_mm("b")
    pts = np.array([[box_a[0, 0] + 5, 130.0], [box_b[1, 0] - 5, 130.0]])
    w = setup.blend_at(pts)
    assert w["a"].tolist() == [1.0, 0.0] and w["b"].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("make", [side_by_side_setup, rotated_setup, corner_setup])
def test_weights_vanish_at_each_inner_edge(make):
    """Each projector fades to 0 at its edges inside the other's footprint (so they cannot show)."""
    setup = make()
    inner = setup.inner_edges()
    assert all(inner[n] for n in setup.names)
    for name in setup.names:
        for p, q in inner[name]:
            along = p + np.linspace(0.1, 0.9, 9)[:, None] * (q - p)
            normal = np.array([-(q - p)[1], (q - p)[0]]) / np.linalg.norm(q - p)
            for sign in (1.0, -1.0):  # step 1 micron off the edge, on whichever side is the overlap
                w = setup.blend_at(along + sign * 1e-3 * normal)
                inside_both = (w[setup.names[0]] > 0) & (w[setup.names[1]] > 0)
                assert np.all(w[name][inside_both] < 1e-4)


@pytest.mark.parametrize("make", [side_by_side_setup, rotated_setup, corner_setup])
def test_weights_are_bounded_and_complementary(make):
    setup = make()
    pts = _overlap_points(setup)
    w = setup.blend_at(pts)
    a, b = setup.names
    assert len(pts) > 100
    assert np.all((w[a] >= 0) & (w[a] <= 1)) and np.allclose(w[a] + w[b], 1.0)


def test_rotated_and_corner_arrangements_have_two_inner_edges_each():
    for setup in (rotated_setup(), corner_setup()):
        assert {n: len(e) for n, e in setup.inner_edges().items()} == {"a": 2, "b": 2}


def test_nested_boxes_outer_keeps_everything():
    outer, inner_box = rect_polygon(0, 0, 250, 140.625), rect_polygon(60, 30, 185, 100.3125)
    setup = make_setup({"a": outer, "b": inner_box}, (0, 0, 250, 140.625))
    w = setup.blend_at(np.array([[120.0, 60.0], [70.0, 40.0]]))
    assert w["a"].tolist() == [1.0, 1.0] and w["b"].tolist() == [0.0, 0.0]


def test_identical_boxes_share_equally():
    box = rect_polygon(0, 0, 250, 140.625)
    setup = make_setup({"a": box, "b": box.copy()}, (0, 0, 250, 140.625))
    w = setup.blend_at(np.array([[120.0, 60.0], [10.0, 10.0]]))
    assert np.allclose(w["a"], 0.5) and np.allclose(w["b"], 0.5)


def test_blend_map_matches_pointwise_rule():
    setup = rotated_setup()
    weights = setup.blend_weights("b")
    w, h = setup.resolution["b"]
    v, u = 70, 3
    point = apply_h(setup.h_cal["b"], np.array([u, v], dtype=float))
    assert weights.shape == (h, w) and weights.dtype == np.float32
    assert weights[v, u] == pytest.approx(setup.blend_at(point)["b"], abs=1e-6)


def test_framebuffer_samples_content_where_each_pixel_lands():
    """Content that is a linear function of screen x must come back exactly (bilinear is exact on it)."""
    setup = rotated_setup()
    size = setup.content_size()
    x0, _, x1, _ = setup.content_rect_mm
    x_mm = apply_h(setup.content_to_mm(size), np.stack(np.meshgrid(np.arange(size[0]), [0.0]), axis=-1))[0, :, 0]
    content = np.repeat(((x_mm - x0) / (x1 - x0)).astype(np.float32)[None, :, None], size[1], 0).repeat(3, 2)
    for name in setup.names:
        fb = setup.framebuffer(name, content)
        w, h = setup.resolution[name]
        u, v = np.meshgrid(np.arange(2, w - 2), np.arange(2, h - 2))
        landing = apply_h(setup.h_cal[name], np.stack([u, v], axis=-1).astype(float))
        expected = (landing[..., 0] - x0) / (x1 - x0)
        assert np.abs(fb[2:-2, 2:-2, 0] - expected).max() < 2e-5


def test_framebuffer_is_black_outside_the_content():
    setup = side_by_side_setup(vertical_offset_mm=3.0)  # rows of A above B's top lie outside the content
    content = np.ones((*setup.content_size()[::-1], 3), dtype=np.float32)
    fb = setup.framebuffer("a", content)
    assert fb[0].max() == 0.0 and fb[fb.shape[0] // 2, 3:-3].min() == pytest.approx(1.0)
