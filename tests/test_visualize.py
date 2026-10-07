"""scripts/visualize.py on a tiny scenario: writes one PNG, and the measured levels are ordered."""

import cv2
import yaml

from scripts import visualize

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


def test_visualize_writes_view_and_levels(tmp_path):
    scenario = tmp_path / "tiny.yaml"
    scenario.write_text(yaml.safe_dump(TINY))
    summary = visualize.main([str(scenario), "--out", str(tmp_path / "out")])
    view = cv2.imread(summary["view"])
    assert view.shape[1] == 480 and view.shape[0] > 208
    lv = {k: v["mean_e"] for k, v in summary["levels"].items()}
    assert lv["unlit"] < lv["black_a"] < lv["black_overlap"]
    assert lv["black_b"] < lv["black_overlap"]
