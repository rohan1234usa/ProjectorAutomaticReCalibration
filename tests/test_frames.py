"""Frames on demand: the same frame in any order or process, paired variants, linear-light mixing, mono."""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.dataset import frame_hash
from sim.frames import FrameSource
from sim.projector import REC709
from sim.scenario import load_scenarios, scenario_from_dict
from tests.scenes import TINY_BEZEL, tiny_shift_sweep

REPO = Path(__file__).resolve().parents[1]
ONSET = 6  # tiny_shift_sweep: frame 6 starts at 3.09 s, the first after the 3 s step


@pytest.fixture
def variants(tmp_path):
    path = tmp_path / "tiny_shift.yaml"
    path.write_text(yaml.safe_dump(tiny_shift_sweep(), sort_keys=False))
    return path, load_scenarios(path)


def test_frames_do_not_depend_on_render_order(variants):
    _, scenarios = variants
    forward = FrameSource(scenarios[1])
    hashes = [frame_hash(forward.frame(i)) for i in range(len(forward))]
    backward = FrameSource(scenarios[1])
    assert [frame_hash(backward.frame(i)) for i in reversed(range(len(backward)))][::-1] == hashes


def test_a_frame_rendered_in_another_process_is_identical(variants):
    path, scenarios = variants
    code = ("from sim.scenario import load_scenario; from sim.frames import FrameSource; from sim.dataset import frame_hash;"
            f"s = load_scenario({str(path)!r}, variant={scenarios[1].variant!r}); print(frame_hash(FrameSource(s).frame(7)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO)
    assert out.stdout.strip() == frame_hash(FrameSource(scenarios[1]).frame(7))


def test_variants_share_every_frame_until_the_onset(variants):
    _, scenarios = variants
    assert [s.variant for s in scenarios] == ["magnitude_px=0", "magnitude_px=1", "magnitude_px=4"]
    sources = [FrameSource(s) for s in scenarios]
    for i in range(len(sources[0])):
        distinct = {frame_hash(src.frame(i)) for src in sources}
        assert len(distinct) == (1 if i < ONSET else 3), f"frame {i}"


def test_a_change_inside_the_exposure_mixes_in_linear_light(variants):
    """Frame 3's exposure straddles the 1.6 s slide change: 30% slide 0, 70% slide 1."""
    _, scenarios = variants
    src = FrameSource(scenarios[0])
    segments = src.state(3).segments
    assert [(k, float(w)) for k, w in segments] == [((0, 0, 0), 0.3), ((0, 0, 1), 0.7)]
    before, after = src.expected(2), src.expected(4)  # wholly slide 0, wholly slide 1
    assert src.state(2).segments[0][0] == (0, 0, 0) and src.state(4).segments[0][0] == (0, 0, 1)
    mixed = src.expected(3)
    assert np.allclose(mixed, 0.3 * before + 0.7 * after, rtol=1e-5, atol=1e-3)


def test_truth_needs_no_render_and_reports_the_step(variants):
    _, scenarios = variants
    src = FrameSource(scenarios[2])  # 4 px across
    lines = [src.truth(i) for i in range(len(src))]
    assert not src._electrons.items  # nothing was rendered
    assert all(line["truth"]["offset_mm"] == 0.0 and line["truth"]["aligned"] for line in lines[:ONSET])
    assert all(abs(line["truth"]["offset_px"] - 4.0) < 1e-9 and not line["truth"]["aligned"] for line in lines[ONSET:])
    assert lines[ONSET]["perturbation"][0]["vector_mm"] == pytest.approx([4 * 250.0 / 240.0, 0.0])
    assert lines[3]["content"]["cut_in_exposure"] and lines[3]["content"]["tag"] == "deck_high+deck_low"
    assert lines[0]["truth"]["markers_visible"] == list(range(8))


def test_mono_is_the_rec709_luminance_of_rgb():
    mono = scenario_from_dict(TINY_BEZEL)
    rgb_cfg = copy.deepcopy(TINY_BEZEL)
    rgb_cfg["camera"]["color"] = "rgb"
    rgb = FrameSource(scenario_from_dict(rgb_cfg)).expected(0)
    luminance = rgb @ np.array(REC709, np.float32)
    got = FrameSource(mono).expected(0)
    assert got.shape == rgb.shape[:2]
    assert np.allclose(got, luminance, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("gain", [None, {"peak": 1.8}])
def test_components_sum_to_one_pass(gain):
    """Light adds linearly: the per-source images (room, bezel light, each projector, drawn with shared
    scratch buffers) sum to one camera pass over the total radiance, to float32 rounding; on a gain
    screen too, where each projector's light meets its own reflectance map."""
    cfg = copy.deepcopy(TINY_BEZEL)
    cfg["screen"]["bezel"]["light"] = 0.01
    if gain:
        cfg["screen"]["gain"] = gain
    cfg.update(duration_s=1, content={"items": [{"type": "deck", "slides": 2, "hold_s": 0.5, "border_frac": 0.1}]})
    scenario = scenario_from_dict(cfg)
    source = FrameSource(scenario)
    renderer, screen = source.renderer, scenario.scene.screen
    assert not np.array_equal(scenario.content_image(0), scenario.content_image(1))
    for i in (0, 1):  # a new slide: the second render reuses every buffer the first one drew into
        one_pass = scenario.scene.camera.expected_electrons(
            renderer.screen_radiance(scenario.content_image(i)), renderer.grid, renderer.quality.camera_supersample,
            border=screen.wall_reflectance * screen.ambient)
        assert np.allclose(source.expected(i), one_pass, rtol=2e-6, atol=0.01)
