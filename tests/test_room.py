"""The room and the gain screen: each projector's hotspot lies where the geometry puts it, the camera
sees each projector's light times its gain, and a matte screen (or none given) changes nothing."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sim.dataset import frame_hash
from sim.frames import FrameSource
from sim.planar import apply_h, points_in_convex
from sim.room import Room, ScreenGain
from sim.scenario import load_scenario, scenario_from_dict
from sim.sweep import deep_merge
from tests.scenes import TINY_BEZEL, tiny_shift_sweep

REPO = Path(__file__).resolve().parents[1]
P, C = np.array([100.0, 50.0, 400.0]), np.array([300.0, 200.0, 800.0])
# In the tiny bezel scene (A over x 80..330, B over 270..520, y 60..200 mm), these put each specular
# hotspot inside its own projector's single coverage: A's at (187.5, 172.5), B's at (412.5, 172.5).
PLACED = {"projectors": {"a": {"position_mm": [150, 130, 300]}, "b": {"position_mm": [450, 130, 300]}},
          "camera": {"position_mm": [300, 300, 900]}}


def _tiny(gain=None, **overrides):
    cfg = deep_merge(TINY_BEZEL, {**PLACED, **({"screen": {"gain": gain}} if gain else {})})
    return scenario_from_dict(deep_merge(cfg, overrides))


def test_default_positions_face_each_image_and_put_the_camera_below_the_screen():
    room = load_scenario(REPO / "scenarios" / "aligned_slides.yaml").scene.room
    assert np.allclose(room.projectors["a"], [1205.0, 749.85, 3200.0])  # 1.6 image widths in front of A's centre
    assert np.allclose(room.projectors["b"], [2795.0, 750.15, 3200.0])
    assert np.allclose(room.camera, [2000.0, 1875.0, 6000.0])
    assert room.gain is None and not room.active


@pytest.mark.parametrize("kind, foot", [("specular", [100 + 200 / 3, 100.0]), ("retro", [-100.0, -100.0])])
def test_each_hotspot_lies_where_the_geometry_puts_it(kind, foot):
    """Specular: where the line from the camera to the projector's mirror image crosses the screen;
    retro: where the line through the camera and the projector does. The brightest point agrees."""
    room = Room({"a": P, "b": P + [400.0, 0.0, 0.0]}, C, ScreenGain(1.8, 20.0, kind))
    assert np.allclose(room.hotspot_mm("a"), foot, atol=1e-9)
    assert float(room.gain_at("a", *foot)) == pytest.approx(1.8, abs=1e-12)
    xs, ys = np.meshgrid(np.arange(-400.0, 400.5), np.arange(-400.0, 400.5))
    k = np.unravel_index(np.argmax(room.gain_at("a", xs, ys)), xs.shape)
    assert np.hypot(xs[k] - foot[0], ys[k] - foot[1]) <= 1.0  # on a 1 mm grid


def test_the_gain_is_a_lobe_in_the_angle_to_the_mirror_direction():
    room = Room({"a": P, "b": P}, C, ScreenGain(2.0, 15.0))
    for x, y in [(0.0, 0.0), (37.0, -120.0), (260.0, 300.0), (-500.0, 80.0)]:
        r = np.array([x, y, 0.0]) - P * [1, 1, -1]  # from the mirror image P' = (px, py, -pz)
        c = C - [x, y, 0.0]
        alpha = np.degrees(np.arccos(r @ c / np.linalg.norm(r) / np.linalg.norm(c)))
        assert float(room.gain_at("a", x, y)) == pytest.approx(1 + np.exp(-0.5 * (alpha / 15.0) ** 2), rel=1e-9)


def test_no_gain_or_a_peak_of_one_is_a_matte_screen():
    for gain in (None, ScreenGain(1.0)):
        room = Room({"a": P, "b": P}, C, gain)
        assert not room.active and room.hotspot_mm("a") is None
        assert np.all(room.gain_at("a", np.linspace(-500.0, 500.0, 11), 30.0) == 1.0)
    level = Room({"a": C, "b": P}, C, ScreenGain(1.8, kind="retro"))
    assert level.hotspot_mm("a") is None  # retro, camera and projector at the same distance: no hotspot


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["screen"].update(gain={"peak": 0.9}), "at least 1"),
    (lambda c: c["screen"].update(gain={"peak": 1.5, "lobe_deg": 0}), "lobe_deg: must be positive"),
    (lambda c: c["screen"].update(gain={"peak": 1.5, "kind": "mirror"}), "must be one of"),
    (lambda c: c["screen"].update(gain={"peak": 1.5, "width": 3}), "unknown keys"),
    (lambda c: c["screen"].update(gain={"lobe_deg": 20}), "peak is required"),
    (lambda c: c["screen"].update(gain=1.8), "expected a mapping"),
    (lambda c: c["camera"].update(position_mm=[1, 2]), r"expected \[x, y, z\]"),
    (lambda c: c["camera"].update(position_mm=[1, 2, 3, 4]), r"expected \[x, y, z\]"),
    (lambda c: c["projectors"]["a"].update(position_mm=[1, 2, 0]), "z must be positive"),
    (lambda c: c["projectors"]["b"].update(position_mm=[1, 2, -5]), "z must be positive"),
    (lambda c: c["projectors"]["b"].update(position_mm=[1, "far", 5]), "expected a number"),
])
def test_positions_and_gain_refuse_mistakes(mutate, message):
    cfg = copy.deepcopy(TINY_BEZEL)
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        scenario_from_dict(cfg)


def test_each_projector_meets_the_screens_reflectance_times_its_gain_and_the_bezel_stays_matte():
    renderer = _tiny({"peak": 1.8}).scene.renderer()
    grid, room = renderer.grid, renderer.room
    above = int(apply_h(grid.mm_to_grid, np.array([0.0, -1.0]))[1])  # grid rows of bezel and wall above the screen
    for name in ("a", "b"):
        seen = renderer.reflectance_for(name)
        u, v = np.rint(apply_h(grid.mm_to_grid, room.hotspot_mm(name))).astype(int)
        # half a grid pixel from the foot moves alpha by ~0.03 degrees: (0.03 / 20)^2 / 2 x 0.8 ~ 1e-6
        assert seen[v, u] / renderer.reflectance[v, u] == pytest.approx(1.8, rel=1e-5)
        assert np.array_equal(seen[:above], renderer.reflectance[:above])
    matte = _tiny({"peak": 1.0}).scene.renderer()
    assert matte.reflectance_for("a") is matte.reflectance


@pytest.mark.parametrize("color", ["mono", "rgb"])
def test_the_camera_sees_each_projectors_light_times_its_gain(color):
    """Flat content without room light: the gain screen's image over its matte twin is, at camera pixels
    well inside one projector's single coverage, that projector's gain at the pixel's centre. Within
    1e-4: the lens blur (0.8 px, about 0.7 mm here) averages a lobe that changes over ~100 mm."""
    scene = {"screen": {"ambient": 0.0}, "camera": {"color": color}, "content": {"type": "flat", "value": 0.6, "border_frac": 0.0}}
    gained, matte = FrameSource(_tiny({"peak": 1.8}, **scene)), FrameSource(_tiny(None, **scene))
    image, twin = gained.expected(0), matte.expected(0)
    h, w = image.shape[:2]
    pixels = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(float)
    pts = apply_h(np.linalg.inv(gained.camera.h_mm_to_px), pixels)
    setup, room = gained.setup, gained.scenario.scene.room
    for name, other in (("a", "b"), ("b", "a")):
        alone = points_in_convex(setup.box_mm(name), pts, 5.0) & ~points_in_convex(setup.box_mm(other), pts, -5.0)
        want = room.gain_at(name, pts[alone][:, 0], pts[alone][:, 1])
        assert alone.sum() > 1000 and want.max() > 1.7  # the hotspot is in view
        assert np.allclose(image[alone] / twin[alone], want if color == "mono" else want[:, None], rtol=1e-4)


def test_unit_gain_or_positions_alone_change_no_frame():
    base = {**tiny_shift_sweep(), "name": "plain"}
    base.pop("sweep")
    hashes = []
    for extra in ({}, {"screen": {"gain": {"peak": 1.0}}}, PLACED):
        source = FrameSource(scenario_from_dict(deep_merge(base, extra)))
        hashes.append([frame_hash(source.frame(i)) for i in (0, 3, 7)])  # still, a slide change, shifted
    assert hashes[0] == hashes[1] == hashes[2]


def test_gain_changes_the_picture_and_never_the_truth():
    base = {**tiny_shift_sweep(), "name": "gain"}
    base.pop("sweep")
    plain, gained = (FrameSource(scenario_from_dict(deep_merge(base, x))) for x in ({}, {"screen": {"gain": {"peak": 2.0}}}))
    assert all(json.dumps(plain.truth(i)) == json.dumps(gained.truth(i)) for i in range(len(plain)))
    assert frame_hash(plain.frame(7)) != frame_hash(gained.frame(7))


@pytest.mark.slow
def test_the_demo_gain_screen_shows_each_hotspot_as_the_lobe_predicts():
    """gain_screen at peak 1.8 and 2.4, a flat gray frame. Net of the room light (which a gain screen does
    not concentrate here), the camera's brightness over the matte twin's is the lobe's gain, at each
    projector's hotspot and 600 mm toward its own edge, within 0.1%: the twin cancels vignetting, the
    blend and the black level, and the lens blur and a 24 mm patch average the lobe by about 1e-4."""
    path = REPO / "scenarios" / "gain_screen.yaml"
    matte = FrameSource(load_scenario(path, variant="peak=1__magnitude_px=0"))
    state = matte.state(400)
    assert matte.scenario.sequence.tag(state.segments[0][0]) == "flat" and len(state.segments) == 1
    room_light = matte.renderer.room_electrons(matte.camera) * np.float32(state.ambient)
    twin = matte.expected(400, state) - room_light
    h = matte.camera.h_mm_to_px

    def patch(img, xy):  # 21 x 21 camera pixels, about 24 mm
        u, v = np.rint(apply_h(h, np.asarray(xy, dtype=float))).astype(int)
        return float(img[v - 10:v + 11, u - 10:u + 11].mean())

    for peak in ("1.8", "2.4"):
        gained = FrameSource(load_scenario(path, variant=f"peak={peak}__magnitude_px=0"))
        image, room = gained.expected(400) - room_light, gained.scenario.scene.room
        for name, away in (("a", -600.0), ("b", 600.0)):
            foot = room.hotspot_mm(name)
            for xy in (foot, foot + [away, 0.0]):
                assert patch(image, xy) / patch(twin, xy) == pytest.approx(float(room.gain_at(name, *xy)), rel=1e-3)
