"""scripts/visualize.py: writes one PNG; labels and level patches follow the true geometry."""

import copy

import cv2
import numpy as np
import yaml

from scripts import visualize
from sim.planar import points_in_convex
from sim.scenario import scene_from_dict

TINY = {
    "name": "tiny",
    "seed": 2,
    "quality": "fast",
    "screen": {"size_mm": [600, 260], "reflectance": 0.9, "ambient": 0.0003},
    "arrangement": {"preset": "side_by_side", "width_mm": 250, "overlap_mm": 60, "vertical_offset_mm": 0.3},
    "projectors": {"a": {"resolution": [240, 135]}, "b": {"resolution": [240, 135]}},
    "blend": {"shape": "cosine"},
    "content": {"type": "slide", "border_frac": 0.1},
    "camera": {"preset": "whole_screen", "resolution": [480, 208]},
}


def _rotated_tiny() -> dict:
    """B rotated 5 degrees about its centre: the overlap is no longer a vertical band."""
    cfg = copy.deepcopy(TINY)
    a = np.array([[80, 60], [330, 60], [330, 200.625], [80, 200.625]], dtype=float)
    b = a + [190.0, 0.0]
    c = b.mean(axis=0)
    t = np.radians(5.0)
    b = (b - c) @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]).T + c
    cfg["arrangement"] = {"preset": "explicit", "corners_mm": {"a": a.tolist(), "b": b.tolist()},
                          "content_rect_mm": [90.0, 75.0, 500.0, 190.0]}
    return cfg


def test_visualize_writes_view_and_levels(tmp_path):
    scenario = tmp_path / "tiny.yaml"
    scenario.write_text(yaml.safe_dump(TINY))
    summary = visualize.main([str(scenario), "--out", str(tmp_path / "out")])
    view = cv2.imread(summary["view"])
    assert view.shape[1] == 480 and view.shape[0] > 2 * 208  # frame panel + geometry panel
    lv = {k: v["mean_e"] for k, v in summary["levels"].items()}
    assert lv["unlit"] < lv["black_a"] < lv["black_overlap"]
    assert lv["black_b"] < lv["black_overlap"]


def test_patches_sit_inside_their_true_regions_for_a_rotated_overlap():
    scene = scene_from_dict(_rotated_tiny())
    a, b = scene.setup.names
    box_a, box_b = scene.setup.box_mm(a), scene.setup.box_mm(b)
    pts, masks = visualize.region_masks(scene)
    centre = (300.0, 130.0)
    expected = {"black_a": (True, False), "black_b": (False, True), "black_overlap": (True, True),
                "unlit": (False, False), "overlap": (True, True), "only_a": (True, False)}
    for region, (in_a, in_b) in expected.items():
        point, depth = visualize.deepest_point(pts, masks[region], centre)
        assert depth > 2.0, region
        assert bool(points_in_convex(box_a, point)) == in_a and bool(points_in_convex(box_b, point)) == in_b, region


def test_missing_regions_report_null(tmp_path):
    cfg = copy.deepcopy(TINY)
    cfg["content"] = {"type": "slide"}  # no black border: no black patches to measure
    scenario = tmp_path / "noborder.yaml"
    scenario.write_text(yaml.safe_dump(cfg))
    summary = visualize.main([str(scenario), "--out", str(tmp_path / "out")])
    assert summary["levels"]["unlit"] is not None
    assert summary["levels"]["black_a"] is None and summary["levels_display_255"]["black_overlap"] is None
