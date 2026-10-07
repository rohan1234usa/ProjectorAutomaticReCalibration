"""Scenario YAML: the repo scenario loads, randomness is reproducible, and mistakes fail clearly."""

import copy
from pathlib import Path

import numpy as np
import pytest

from sim.scenario import load_scene, scene_from_dict
from tests.test_visualize import TINY

REPO = Path(__file__).resolve().parents[1]


def test_repo_scenario_loads():
    scene = load_scene(REPO / "scenarios" / "aligned_side_by_side.yaml", quality="fast")
    assert scene.name == "aligned_side_by_side" and set(scene.projectors) == {"a", "b"}
    assert scene.setup.content_size() == (3446, 1080)  # the content rect at the 1.04 mm projector pitch


def test_randomness_is_reproducible_from_the_seed():
    a, b = scene_from_dict(TINY), scene_from_dict(TINY)
    assert np.array_equal(a.content_image(), b.content_image())
    assert a.frame_rng(3).random() == b.frame_rng(3).random() != a.frame_rng(4).random()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c.pop("camera"), "missing keys"),
        (lambda c: c.update(colour="red"), "unknown keys"),
        (lambda c: c["arrangement"].pop("width_mm"), "needs width_mm"),
        (lambda c: c["arrangement"].update(preset="stacked"), "preset must be"),
        (lambda c: c["camera"].pop("resolution"), "resolution is required"),
        (lambda c: c["projectors"]["a"].update(gama=2.2), "unknown keys"),
    ],
)
def test_mistakes_fail_with_a_clear_message(mutate, message):
    cfg = copy.deepcopy(TINY)
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        scene_from_dict(cfg)
