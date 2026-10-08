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


@pytest.mark.slow
@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_first_frame_renders(path):
    scenario = load_scenarios(path, quality="fast")[0]
    frame = FrameSource(scenario).frame(0)
    w, h = scenario.scene.camera.resolution
    assert frame.shape == (h, w) and frame.dtype.name == "uint16"
