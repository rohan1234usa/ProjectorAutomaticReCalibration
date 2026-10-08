"""Scenario YAML: repo scenarios load, extends and sweeps resolve, randomness is reproducible, mistakes fail clearly."""

import copy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.scenario import load_scenario, load_scenarios, scenario_from_dict
from tests.scenes import TINY_SCENARIO

REPO = Path(__file__).resolve().parents[1]


def test_phase1_scenario_still_loads():
    s = load_scenario(REPO / "scenarios" / "aligned_side_by_side.yaml", quality="fast")
    assert s.name == "aligned_side_by_side" and set(s.scene.projectors) == {"a", "b"}
    assert s.scene.setup.content_size() == (3446, 1080)  # the content rect at the 1.04 mm projector pitch
    assert s.timing.n_frames == 1 and s.scene.markers is None and s.perturbations == ()


def test_randomness_is_reproducible_from_the_seed():
    a, b = scenario_from_dict(TINY_SCENARIO), scenario_from_dict(TINY_SCENARIO)
    other = scenario_from_dict({**TINY_SCENARIO, "seed": 3})
    assert np.array_equal(a.content_image(), b.content_image())
    assert not np.array_equal(a.content_image(), other.content_image())


def test_projectors_are_placed_by_name_not_by_listing_order():
    cfg = copy.deepcopy(TINY_SCENARIO)
    cfg["projectors"] = {"b": {"resolution": [240, 135]}, "a": {"resolution": [240, 135]}}
    cfg["arrangement"]["width_a_mm"] = cfg["arrangement"].pop("width_mm")
    cfg["arrangement"]["width_b_mm"] = 300.0  # B wider: its box must be the right-hand one
    setup = scenario_from_dict(cfg).scene.setup
    box_a, box_b = setup.box_mm("a"), setup.box_mm("b")
    assert setup.names == ("a", "b")
    assert box_a[:, 0].min() < box_b[:, 0].min()
    assert np.ptp(box_b[:, 0]) == pytest.approx(300.0) and np.ptp(box_a[:, 0]) == pytest.approx(250.0)


def test_numbers_without_a_decimal_point_are_numbers():
    """YAML reads 3e-4 as the string '3e-4'; the scenario must still read it as 0.0003."""
    cfg = copy.deepcopy(TINY_SCENARIO)
    cfg["screen"]["ambient"] = yaml.safe_load("x: 3e-4")["x"]
    assert cfg["screen"]["ambient"] == "3e-4"
    assert scenario_from_dict(cfg).scene.screen.ambient == 0.0003


def test_timing_uses_exact_fractions():
    cfg = {**copy.deepcopy(TINY_SCENARIO), "duration_s": 10, "sample_every_s": 0.5}
    cfg["camera"].update(phase_s=0.0123, exposure_s="1/30")
    s = scenario_from_dict(cfg)
    assert s.timing.time(3) == Fraction(3, 2) + Fraction(123, 10000)
    assert s.timing.exposure == Fraction(1, 30) and s.timing.n_frames == 20


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))  # sweep keys expand in the order written
    return path


def test_extends_and_sweep_expand_into_paired_variants(tmp_path):
    base = {k: v for k, v in TINY_SCENARIO.items() if k != "name"}
    _write(tmp_path, "_base.yaml", base)
    scenario = {
        "extends": "_base.yaml",
        "duration_s": 4,
        "trusted_window_s": 1,
        "perturbation": {"b": {"kind": "shift", "magnitude_px": 1, "schedule": {"type": "step", "t0_s": 2}}},
        "sweep": {"perturbation.b.magnitude_px": [0, 0.5, 1], "perturbation.b.direction": ["across", "along"]},
    }
    variants = load_scenarios(_write(tmp_path, "shifts.yaml", scenario), quality="fast")
    names = [v.variant for v in variants]
    assert names == ["magnitude_px=0__direction=across", "magnitude_px=0.5__direction=across",
                     "magnitude_px=0.5__direction=along", "magnitude_px=1__direction=across",
                     "magnitude_px=1__direction=along"]  # zero size twice is the same scenario: kept once
    assert {v.name for v in variants} == {"shifts"} and {v.scene.seed for v in variants} == {TINY_SCENARIO["seed"]}
    assert "sweep" not in variants[1].data and "extends" not in variants[1].data
    assert variants[3].data["perturbation"]["b"]["magnitude_px"] == 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("camera"), "missing keys"),
        (lambda c: c.update(colour="red"), "unknown keys"),
        (lambda c: c["arrangement"].update(width_mm=2000), "does not fit"),
        (lambda c: c["arrangement"].update(preset="hexagonal"), "preset must be one of"),
        (lambda c: c["arrangement"].update(preset="stacked"), "unknown keys"),
        (lambda c: c["camera"].pop("resolution"), "resolution is required"),
        (lambda c: c["projectors"]["a"].update(gama=2.2), "unknown keys"),
        (lambda c: c["screen"].pop("size_mm"), "size_mm is required"),
        (lambda c: c["projectors"]["b"].pop("resolution"), "projectors.b: resolution is required"),
        (lambda c: c["projectors"].update(c=c["projectors"].pop("b")), "named a and b"),
        (lambda c: c["screen"].update(ambient=True), "expected a number"),
        (lambda c: c.update(duration_s=10, content={"items": [{"type": "black", "hold_s": 2}]}), "set loop: true"),
        (lambda c: c.update(nuisances=[{"type": "camera_bump"}]), "Phase 2b"),
        (lambda c: c.update(perturbation={"b": {"kind": "shift", "magnitude_px": 1, "schedule": {"type": "step", "t0_s": 5}}}),
         "trusted window"),
        (lambda c: c.update(perturbation={"b": {"kind": "shift", "magnitude_px": 1, "deg": 1}}), "exactly one size"),
        (lambda c: c.update(perturbation={"c": {"kind": "shift", "magnitude_px": 1}}), "unknown keys"),
        (lambda c: c.update(trusted_window_s=0, perturbation={"b": {"kind": "shift", "magnitude_px": 1}}), "schedule is required"),
        (lambda c: c.update(reference={"available": True, "lag_s": -0.1}), "cannot show content before"),
    ],
)
def test_mistakes_fail_with_a_clear_message(mutate, message):
    cfg = copy.deepcopy(TINY_SCENARIO)
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        scenario_from_dict(cfg)


def test_content_need_only_last_until_the_last_exposure_ends():
    """A 6 s held slide for a 6 s run is enough: the last exposure ends at 5.5 s + 1/30 s."""
    cfg = {**copy.deepcopy(TINY_SCENARIO), "duration_s": 6, "content": {"items": [{"type": "held", "hold_s": 6}]}}
    assert scenario_from_dict(cfg).timing.n_frames == 12


def test_display_lag_delays_the_content():
    cfg = {**copy.deepcopy(TINY_SCENARIO), "duration_s": 10, "content": {"items": [
        {"type": "black", "hold_s": 2}, {"type": "flat", "value": 0.5, "hold_s": 8}]}}
    now = scenario_from_dict(cfg)
    late = scenario_from_dict({**cfg, "reference": {"available": True, "lag_s": 0.6}})

    def items(s):  # which content item each of the first 7 frames shows
        return [s.sequence.segments(s.timing.time(i), s.timing.exposure)[0][0][0] for i in range(7)]

    assert items(now) == [0, 0, 0, 0, 1, 1, 1]  # the flat field from 2 s
    assert items(late) == [0, 0, 0, 0, 0, 0, 1]  # ... shown 0.6 s later
    assert late.reference["lag_s"] == Fraction(3, 5)


def test_seeds_are_read_exactly():
    big = 2**53 + 1
    assert scenario_from_dict({**copy.deepcopy(TINY_SCENARIO), "seed": big}).scene.seed == big
    assert scenario_from_dict({**copy.deepcopy(TINY_SCENARIO), "seed": "1e3"}).scene.seed == 1000
