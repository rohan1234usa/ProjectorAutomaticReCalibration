"""scripts/visualize.py: writes one PNG; labels and level patches follow the true geometry."""

import copy

import cv2
import numpy as np
import yaml

from scripts import visualize
from sim.planar import points_in_convex
from sim.scenario import scenario_from_dict
from tests.scenes import TINY_SCENARIO


def _rotated_tiny() -> dict:
    """B rotated 5 degrees about its centre: the overlap is no longer a vertical band."""
    cfg = copy.deepcopy(TINY_SCENARIO)
    a = np.array([[80, 60], [330, 60], [330, 200.625], [80, 200.625]], dtype=float)
    b = a + [190.0, 0.0]
    c = b.mean(axis=0)
    t = np.radians(5.0)
    b = (b - c) @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T + c
    cfg["arrangement"] = {"preset": "explicit", "corners_mm": {"a": a.tolist(), "b": b.tolist()},
                          "content_rect_mm": [90.0, 75.0, 500.0, 190.0]}
    return cfg


def _run(tmp_path, cfg: dict) -> dict:
    scenario = tmp_path / f"{cfg['name']}.yaml"
    scenario.write_text(yaml.safe_dump(cfg))
    return visualize.main([str(scenario), "--out", str(tmp_path / "out")])


def test_visualize_writes_view_and_levels(tmp_path):
    summary = _run(tmp_path, TINY_SCENARIO)
    view = cv2.imread(summary["view"])
    assert view.shape[1] == 480 and view.shape[0] > 2 * 208  # frame panel + geometry panel
    lv = {k: v["mean_e"] for k, v in summary["levels"].items()}
    assert lv["unlit"] < lv["black_a"] < lv["black_overlap"]
    assert lv["black_b"] < lv["black_overlap"]


def test_patches_sit_inside_their_true_regions_for_a_rotated_overlap():
    scenario = scenario_from_dict(_rotated_tiny())
    scene = scenario.scene
    a, b = scene.setup.names
    box_a, box_b = scene.setup.box_mm(a), scene.setup.box_mm(b)
    pts, masks = visualize.region_masks(scene, visualize.border_mm(scenario))
    centre = (300.0, 130.0)
    expected = {"black_a": (True, False), "black_b": (False, True), "black_overlap": (True, True),
                "unlit": (False, False), "overlap": (True, True), "only_a": (True, False)}
    for region, (in_a, in_b) in expected.items():
        point, depth = visualize.deepest_point(pts, masks[region], centre)
        assert depth > 2.0, region
        assert bool(points_in_convex(box_a, point)) == in_a and bool(points_in_convex(box_b, point)) == in_b, region


def test_missing_regions_report_null(tmp_path):
    cfg = copy.deepcopy(TINY_SCENARIO)
    cfg["content"] = {"type": "held"}  # no black border: no black patches to measure
    summary = _run(tmp_path, cfg)
    assert summary["levels"]["unlit"] is not None
    assert summary["levels"]["black_a"] is None and summary["levels_display_255"]["black_overlap"] is None


def test_patches_the_camera_cannot_see_report_null(tmp_path):
    cfg = copy.deepcopy(TINY_SCENARIO)
    cfg["name"] = "zoomed"
    cfg["camera"]["margin"] = -0.2  # frames only the middle of the screen: its unlit margins are off-frame
    summary = _run(tmp_path, cfg)
    assert summary["levels"]["unlit"] is None
    assert summary["levels"]["black_overlap"] is not None


def test_hotspots_are_drawn_and_reported_only_on_a_gain_screen(tmp_path):
    matte = _run(tmp_path, TINY_SCENARIO)
    assert "hotspot_mm" not in matte
    cfg = copy.deepcopy(TINY_SCENARIO)
    cfg["name"] = "tiny_gain"
    cfg["screen"]["gain"] = {"peak": 1.8}
    cfg["projectors"]["a"]["position_mm"] = [150, 130, 300]
    cfg["camera"]["position_mm"] = [300, 300, 900]
    gained = _run(tmp_path, cfg)
    assert np.allclose(gained["hotspot_mm"]["a"], [187.5, 172.5])  # P + (C - P) pz / (pz + cz)
    assert set(gained["hotspot_mm"]) == {"a", "b"}
