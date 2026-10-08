"""Misalignment injection: directions follow the overlap, sizes are solved from the offset they cause."""

import math
from fractions import Fraction

import numpy as np
import pytest

from sim.arrangements import from_config as arrangement
from sim.calibration import CalibrationSetup
from sim.perturb import across, from_config, h_actual, pivot_point
from sim.planar import apply_h, homography_from_points, raster_corners, rect_polygon
from sim.truth import coarse_pitch_mm, offset_mm

SCREEN = (4000.0, 1500.0)
RES = {"a": (1920, 1080), "b": (1920, 1080)}
NOW = {"type": "step", "t0_s": 0}  # full strength from the start


def _setup(preset: str = "side_by_side") -> CalibrationSetup:
    arr = arrangement({"preset": preset}, SCREEN, RES)
    h = {n: homography_from_points(raster_corners(RES[n]), arr.corners[n]) for n in RES}
    return CalibrationSetup(h_cal=h, resolution=RES, content_rect_mm=arr.content_rect_mm)


def _offset_at(setup: CalibrationSetup, spec: dict, m: float = 1.0) -> float:
    (p,) = from_config({"b": {**spec, "schedule": NOW}}, setup)
    return offset_mm(setup, h_actual(setup, (p,), (m,)))


def test_across_points_away_from_the_partner():
    setup = _setup()
    assert np.allclose(across(setup, "b"), [1.0, 0.0]) and np.allclose(across(setup, "a"), [-1.0, 0.0])
    stacked = _setup("stacked")  # B below A: away from A is down
    assert np.allclose(across(stacked, "b"), [0.0, 1.0], atol=1e-12)


def test_shift_across_narrows_the_overlap_and_along_moves_down():
    setup = _setup()
    pitch = coarse_pitch_mm(setup)
    (p,) = from_config({"b": {"kind": "shift", "magnitude_px": 2, "schedule": NOW}}, setup)
    moved = apply_h(h_actual(setup, (p,), (1.0,))["b"] @ np.linalg.inv(setup.h_cal["b"]), np.array([0.0, 0.0]))
    assert np.allclose(moved, [2 * pitch, 0.0])
    (q,) = from_config({"b": {"kind": "shift", "magnitude_px": 2, "direction": "along", "schedule": NOW}}, setup)
    assert np.allclose(q.vector, [0.0, 1.0])


def test_pivots():
    setup = _setup()
    assert np.allclose(pivot_point(setup, "b", "centre"), [2795.0, 750.15])  # where B's centre pixel lands
    assert np.allclose(pivot_point(setup, "b", "far_corner"), [3795.0, 1312.65])  # B sits 0.3 mm low: BR is farthest
    assert np.allclose(pivot_point(setup, "b", [10, 20]), [10.0, 20.0])


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "shift", "magnitude_px": 1.5, "direction": [1, 1]},
        {"kind": "rotation", "magnitude_px": 1.5, "pivot": "centre"},
        {"kind": "rotation", "magnitude_px": 1.5, "pivot": "far_corner"},
        {"kind": "scale", "magnitude_px": 1.5},
        {"kind": "keystone", "magnitude_px": 1.5, "axis": "x"},
        {"kind": "keystone", "magnitude_px": 1.5, "axis": "y", "pivot": "overlap_centre"},
        {"kind": "rotation", "magnitude_px": -1.5},
        {"kind": "keystone", "magnitude_px": -1.5, "axis": "x"},  # a keystone is not symmetric in k
        {"kind": "keystone", "magnitude_px": -40, "axis": "x"},
    ],
)
def test_sizes_are_solved_from_the_offset_they_cause(spec):
    setup = _setup()
    size = abs(spec["magnitude_px"])
    assert _offset_at(setup, spec) == pytest.approx(size * coarse_pitch_mm(setup), abs=1e-9)


def test_natural_units_give_the_closed_form_offset():
    setup = _setup()
    (p,) = from_config({"b": {"kind": "rotation", "deg": 0.05, "pivot": "centre", "schedule": NOW}}, setup)
    poly = setup.overlap()
    r = float(np.hypot(*(poly - p.pivot).T).max())
    assert offset_mm(setup, h_actual(setup, (p,), (1.0,))) == pytest.approx(2 * r * math.sin(math.radians(0.05) / 2), rel=1e-12)
    (s,) = from_config({"b": {"kind": "scale", "factor": 1.001, "schedule": NOW}}, setup)
    assert offset_mm(setup, h_actual(setup, (s,), (1.0,))) == pytest.approx(0.001 * float(np.hypot(*(poly - s.pivot).T).max()))


def test_continuous_schedules_move_in_steps_of_002_px():
    setup = _setup()
    spec = {"kind": "shift", "magnitude_px": 1.0, "schedule": {"type": "drift", "t0_s": 600, "rate_per_h": 2.0}}  # 2 px/h
    (p,) = from_config({"b": spec}, setup)
    pitch = coarse_pitch_mm(setup)
    for t in (600, 601, 637, 1999, 4200):
        m = p.multiplier(Fraction(t))
        steps = offset_mm(setup, h_actual(setup, (p,), (m,))) / pitch / 0.02
        assert steps == pytest.approx(round(steps), abs=1e-9)
        assert abs(m - p.schedule.value(Fraction(t))) <= 0.01 + 1e-12  # rounding, never more than half a step


def test_perturbations_on_one_projector_compose_in_order():
    setup = _setup()
    specs = [{"kind": "rotation", "deg": 0.1, "schedule": NOW},
             {"kind": "shift", "magnitude_mm": 3.0, "direction": [0, 1], "schedule": NOW}]
    ps = from_config({"b": specs}, setup)
    got = h_actual(setup, ps, (1.0, 1.0))["b"]
    assert np.allclose(got, ps[1].matrix(1.0) @ ps[0].matrix(1.0) @ setup.h_cal["b"])


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"kind": "twist", "magnitude_px": 1}, "must be one of"),
        ({"kind": "rotation", "magnitude_px": 1, "direction": "across"}, "unknown keys"),
        ({"kind": "shift", "magnitude_px": 1, "pivot": "centre"}, "unknown keys"),
        ({"kind": "shift", "factor": 1.01}, "exactly one size"),
        ({"kind": "keystone", "magnitude_px": 1, "axis": "z"}, "must be one of"),
        ({"kind": "shift", "magnitude_px": 1, "direction": [0, 0]}, "non-zero"),
    ],
)
def test_bad_perturbations_fail_clearly(spec, message):
    with pytest.raises(ValueError, match=message):
        from_config({"b": {**spec, "schedule": NOW}}, _setup())


def test_a_forgotten_schedule_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="schedule is required"):
        from_config({"b": {"kind": "shift", "magnitude_px": 2}}, _setup())
    (off,) = from_config({"b": {"kind": "shift", "magnitude_px": 2, "schedule": {"type": "none"}}}, _setup())
    assert off.multiplier(Fraction(10**6)) == 0.0


def test_across_is_undefined_for_a_nested_projector():
    res = {"a": (1920, 1080), "b": (1920, 1080)}
    corners = {"a": rect_polygon(200, 100, 3800, 2125), "b": rect_polygon(1000, 600, 2000, 1162.5)}
    h = {n: homography_from_points(raster_corners(res[n]), corners[n]) for n in res}
    nested = CalibrationSetup(h_cal=h, resolution=res, content_rect_mm=(200.0, 100.0, 3800.0, 2125.0))
    with pytest.raises(ValueError, match="give a vector"):
        across(nested, "b")
