"""Ground truth: the offset is how far apart A and B put the same content, and treats A and B alike."""

import math

import numpy as np
import pytest

from sim.arrangements import from_config as arrangement
from sim.calibration import CalibrationSetup
from sim.planar import apply_h, homography_from_points, points_in_convex, raster_corners, translation
from sim.truth import coarse_pitch_mm, offset_mm, relative_homography

SCREEN = (4000.0, 1500.0)


def _setup(preset: str = "side_by_side", res_b=(1920, 1080)) -> CalibrationSetup:
    res = {"a": (1920, 1080), "b": res_b}
    arr = arrangement({"preset": preset}, SCREEN, res)
    h = {n: homography_from_points(raster_corners(res[n]), arr.corners[n]) for n in res}
    return CalibrationSetup(h_cal=h, resolution=res, content_rect_mm=arr.content_rect_mm)


def _moved(setup: CalibrationSetup, name: str, m: np.ndarray) -> dict[str, np.ndarray]:
    return {**setup.h_cal, name: m @ setup.h_cal[name]}


def _about(c: np.ndarray, local: np.ndarray) -> np.ndarray:
    return translation(*c) @ local @ translation(*(-c))


def test_aligned_is_zero_and_identity():
    setup = _setup()
    assert offset_mm(setup, setup.h_cal) == 0.0
    assert np.allclose(relative_homography(setup, setup.h_cal), np.eye(3))


def test_shift_offset_is_the_shift():
    setup = _setup()
    assert offset_mm(setup, _moved(setup, "b", translation(0.3, -1.1))) == pytest.approx(math.hypot(0.3, 1.1), abs=1e-12)


def test_rotation_offset_is_the_chord_at_the_farthest_overlap_vertex():
    setup = _setup()
    c = np.array([3000.0, 900.0])
    t = math.radians(0.07)
    rot = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    r = float(np.hypot(*(setup.overlap() - c).T).max())
    assert offset_mm(setup, _moved(setup, "b", _about(c, rot))) == pytest.approx(2 * r * math.sin(t / 2), rel=1e-12)


def test_moving_a_or_b_by_the_same_change_gives_the_same_offset():
    setup = _setup()
    zoom = _about(np.array([2000.0, 750.0]), np.diag([1.001, 1.001, 1.0]))
    assert offset_mm(setup, _moved(setup, "a", zoom)) == pytest.approx(offset_mm(setup, _moved(setup, "b", zoom)), rel=1e-12)


def test_moving_both_together_is_aligned():
    setup = _setup()
    both = {n: translation(5.0, 2.0) @ setup.h_cal[n] for n in setup.names}
    assert offset_mm(setup, both) == pytest.approx(0.0, abs=1e-12)


def test_keystone_offset_matches_brute_force():
    setup = _setup()
    k = np.array([[1, 0, 0], [0, 1, 0], [1.3e-6, -0.4e-6, 1.0]])
    h_act = _moved(setup, "b", _about(np.array([2795.0, 750.0]), k))
    poly = setup.overlap()
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    xs, ys = np.meshgrid(np.linspace(lo[0], hi[0], 801), np.linspace(lo[1], hi[1], 801))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    pts = pts[points_in_convex(poly, pts)]
    d_b = h_act["b"] @ np.linalg.inv(setup.h_cal["b"])
    brute = float(np.hypot(*(apply_h(d_b, pts) - pts).T).max())
    got = offset_mm(setup, h_act)
    assert brute <= got + 1e-12 and got - brute < 1e-4


def test_pitch_is_the_coarser_projector_at_the_overlap():
    assert coarse_pitch_mm(_setup()) == pytest.approx(2000.0 / 1920.0)
    finer_b = _setup("different_sizes", res_b=(1920, 1080))  # B is 1600 mm wide: 0.833 mm pixels
    assert coarse_pitch_mm(finer_b) == pytest.approx(2000.0 / 1920.0)
    coarse_b = _setup("different_sizes", res_b=(1280, 720))  # B: 1.25 mm pixels
    assert coarse_pitch_mm(coarse_b) == pytest.approx(1600.0 / 1280.0)
