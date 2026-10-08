"""Every scenario in scenarios/ loads, its variants are distinct, and its timeline respects the detector's.

Adding a test idea means adding a YAML file, so each file is checked here: it parses, every sweep
variant builds, and no perturbation starts inside the detector's trusted window (TRUSTED_WINDOW_S
in detector.yaml). The slow test renders the first frame of every scenario.
"""

from fractions import Fraction
from pathlib import Path

import pytest

from detector.config import DetectorConfig
from sim.frames import FrameSource
from sim.scenario import load_scenarios

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = sorted(p for p in (REPO / "scenarios").glob("*.yaml") if not p.name.startswith("_"))
DETECTOR = DetectorConfig.from_yaml(REPO / "detector.yaml")


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_scenario_and_all_its_variants_load(path):
    variants = load_scenarios(path)
    names = [v.variant for v in variants]
    assert len(set(names)) == len(names)
    for v in variants:
        assert v.timing.sample_every == Fraction(str(DETECTOR.SAMPLE_EVERY_S))
        if v.perturbations:
            assert v.timing.trusted_window >= DETECTOR.TRUSTED_WINDOW_S
            assert all(p.schedule.onset is None or p.schedule.onset >= DETECTOR.TRUSTED_WINDOW_S for p in v.perturbations)


def _event_frames(scenario) -> list[int]:
    """Frame 0 and the first frame at or after every perturbation and nuisance onset."""
    timing = scenario.timing
    onsets = [p.schedule.onset for p in scenario.perturbations if p.schedule.onset is not None]
    onsets += scenario.nuisances.onsets
    frames = {0}
    for onset in onsets:
        i = max(0, -((timing.phase - onset) // timing.sample_every))  # ceil((onset - phase) / sample)
        if i < timing.n_frames:
            frames.add(int(i))
    return sorted(frames)


@pytest.mark.slow
@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_event_frames_render(path):
    """The first and last variant of every scenario render at frame 0 and at each onset."""
    variants = load_scenarios(path, quality="fast")
    for scenario in {id(v): v for v in (variants[0], variants[-1])}.values():
        source = FrameSource(scenario)
        w, h = scenario.scene.camera.resolution
        for i in _event_frames(scenario):
            frame = source.frame(i)
            assert frame.shape == (h, w) and frame.dtype.name == "uint16", (scenario.variant, i)
            assert 0 < frame.mean() < 60000
