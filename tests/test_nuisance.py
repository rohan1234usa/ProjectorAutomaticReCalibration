"""Nuisances: each changes the picture as its physics says, and none of them ever misaligns anything."""

import copy
from fractions import Fraction

import numpy as np
import pytest

from sim.flicker import Flicker
from sim.frames import FrameSource
from sim.planar import apply_h
from sim.scenario import scenario_from_dict
from tests.scenes import TINY_BEZEL

STEP = {"type": "step", "t0_s": 2}


def _scenario(nuisances: list, **overrides) -> FrameSource:
    cfg = copy.deepcopy(TINY_BEZEL)
    cfg.update(duration_s=4, trusted_window_s=1, nuisances=nuisances,
               content={"items": [{"type": "flat", "value": 0.6, "hold_s": 4}]}, **overrides)
    return FrameSource(scenario_from_dict(cfg))


def _pixel(src: FrameSource, x_mm: float, y_mm: float, state=None) -> tuple[int, int]:
    camera = src.camera if state is None else src.camera_for(state)
    u, v = np.rint(apply_h(camera.h_mm_to_px, np.array([x_mm, y_mm]))).astype(int)
    return v, u


def test_lamp_dimming_scales_all_of_that_projectors_light():
    cfg_screen = {**TINY_BEZEL["screen"], "ambient": 0.0}  # no room light: only projector light remains
    src = _scenario([{"type": "lamp", "projector": "b", "gain": 0.85, "schedule": STEP}], screen=cfg_screen)
    before, after = src.expected(0), src.expected(5)  # t = 0 and 2.5 s
    only_b, only_a = _pixel(src, 480.0, 130.0), _pixel(src, 120.0, 130.0)
    assert after[only_b] / before[only_b] == pytest.approx(0.85, rel=1e-5)  # black level included
    assert after[only_a] == pytest.approx(before[only_a], rel=1e-6)


def test_room_light_step_raises_the_unlit_screen():
    src = _scenario([{"type": "room_light", "ambient": 0.05, "schedule": STEP}])
    unlit = _pixel(src, 40.0, 130.0)  # on the screen, outside both projectors
    assert src.expected(5)[unlit] / src.expected(0)[unlit] == pytest.approx(0.05 / 0.02, rel=1e-4)
    assert src.truth(5)["nuisances"]["ambient"] == pytest.approx(0.05)


def test_a_camera_bump_moves_everything_in_the_image_but_aligns_nothing_away():
    src = _scenario([{"type": "camera_bump", "shift_px": [5, -3], "schedule": STEP}])
    before, after = src.state(0), src.state(5)
    assert after.camera_key == (1.0,)
    centre = np.array(src.scene.markers.centres_mm[0])
    moved = apply_h(src.camera_for(after).h_mm_to_px, centre) - apply_h(src.camera_for(before).h_mm_to_px, centre)
    assert np.allclose(moved, [5.0, -3.0])
    truth = src.truth(5)
    assert truth["truth"]["aligned"] and truth["truth"]["offset_mm"] == 0.0
    assert truth["truth"]["camera_h_mm_to_px"] != src.truth(0)["truth"]["camera_h_mm_to_px"]


def test_a_person_walking_past_darkens_the_screen_and_hides_markers():
    person = {"type": "occluder", "t0_s": 2, "duration_s": 1, "height_mm": 260, "floor_mm": 300, "direction": "right"}
    src = _scenario([person])
    i = 5  # t = 2.5 s: halfway across, centred at x = 300 mm
    state = src.state(i)
    assert state.occluder and not src.state(1).occluder
    torso = _pixel(src, 300.0, 300.0 - 0.6 * 260)
    ratio = src.expected(i)[torso] / src.expected(1)[torso]
    assert 0.2 < ratio < 0.5  # reflectance 0.3 against the screen's 0.9
    visible = src.truth(i)["truth"]["markers_visible"]
    assert 6 in src.truth(1)["truth"]["markers_visible"] and 6 not in visible  # its legs hide the bottom pair
    assert src.truth(i)["truth"]["aligned"]


def test_flicker_cancels_when_the_exposure_spans_whole_periods():
    rows, exposure = 400, Fraction(1, 30)
    locked = Flicker("a", amplitude=0.1, frequency_hz=60.0, line_time_s=1e-5, seed=3, number=0)
    assert np.allclose(locked.row_gain(7, exposure, rows), 1.0, atol=1e-12)  # two whole periods: no flicker
    beating = Flicker("a", amplitude=0.1, frequency_hz=100.0, line_time_s=1e-5, seed=3, number=0)
    gain = beating.row_gain(7, exposure, rows)
    assert np.ptp(gain) > 0.005  # bands down the rolling-shutter frame
    global_shutter = Flicker("a", amplitude=0.1, frequency_hz=100.0, line_time_s=0.0, seed=3, number=0)
    assert np.ptp(global_shutter.row_gain(7, exposure, rows)) < 1e-12  # one gain per frame


def test_flicker_changes_from_frame_to_frame():
    """The camera's clock is not locked to the projector's: each frame catches another phase."""
    beating = Flicker("a", amplitude=0.1, frequency_hz=100.0, line_time_s=1e-5, seed=3, number=0)
    gains = [beating.row_gain(i, Fraction(1, 30), 400) for i in range(5)]
    assert all(np.abs(gains[i] - gains[i + 1]).max() > 1e-3 for i in range(4))
    assert np.array_equal(gains[2], beating.row_gain(2, Fraction(1, 30), 400))  # reproducible per frame


def test_flicker_scales_its_projectors_rows():
    src = _scenario([{"type": "flicker", "projector": "a", "amplitude": 0.1, "frequency_hz": 100, "line_time_s": 2e-5}],
                    screen={**TINY_BEZEL["screen"], "ambient": 0.0})
    plain = _scenario([], screen={**TINY_BEZEL["screen"], "ambient": 0.0})
    only_a_rows = slice(*sorted(int(_pixel(src, 120.0, y)[0]) for y in (90.0, 170.0)))
    col = int(_pixel(src, 120.0, 130.0)[1])
    ratio = src.expected(3)[only_a_rows, col] / plain.expected(3)[only_a_rows, col]
    assert np.ptp(ratio) > 1e-3 and abs(ratio.mean() - 1.0) < 0.02


def test_sharpening_adds_halos_but_keeps_the_level():
    plain = _scenario([])
    sharp = _scenario([{"type": "sharpening", "amount": 0.8, "sigma_px": 1.0}])
    a = plain.camera.decode(plain.frame(2))
    b = sharp.camera.decode(sharp.frame(2))
    assert abs(b.mean() - a.mean()) < 0.01 * a.mean()
    assert np.abs(np.diff(b, axis=1)).mean() > 1.3 * np.abs(np.diff(a, axis=1)).mean()


def test_every_nuisance_at_once_leaves_the_projectors_aligned():
    src = _scenario([
        {"type": "camera_bump", "shift_px": [2, 1], "rotation_deg": 0.1, "schedule": STEP},
        {"type": "lamp", "projector": "a", "gain": 0.9, "schedule": {"type": "ramp", "t0_s": 1, "duration_s": 2}},
        {"type": "room_light", "ambient": 0.04, "schedule": STEP},
        {"type": "occluder", "t0_s": 2, "duration_s": 1},
        {"type": "flicker", "projector": "b", "amplitude": 0.05},
        {"type": "sharpening"},
    ])
    lines = [src.truth(i) for i in range(len(src))]
    assert all(line["truth"]["aligned"] and line["truth"]["offset_mm"] == 0.0 for line in lines)
    assert lines[-1]["nuisances"]["lamp_gain"]["a"] == pytest.approx(0.9) and lines[-1]["nuisances"]["sharpening"]
    assert src.frame(5).shape == (520, 960)


@pytest.mark.parametrize(
    "nuisance, message",
    [
        ({"type": "lamp", "projector": "c", "schedule": STEP}, "must be one of"),
        ({"type": "lamp", "projector": "b", "gain": 0.85}, "missing keys \\['schedule'\\]"),
        ({"type": "occluder"}, "t0_s"),
        ({"type": "flicker", "projector": "a", "amplitude": 1.5}, "amplitude"),
        ({"type": "room_light", "ambient": 0.05, "colour": "warm", "schedule": STEP}, "unknown keys"),
    ],
)
def test_bad_nuisances_fail_clearly(nuisance, message):
    with pytest.raises(ValueError, match=message):
        _scenario([nuisance])
