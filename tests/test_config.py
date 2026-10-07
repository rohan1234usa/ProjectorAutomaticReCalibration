"""DetectorConfig: defaults match the contract in CLAUDE.md section 10, YAML loading is strict."""

from pathlib import Path

import pytest

from detector.config import DetectorConfig, tolerance_from_viewing_distance

REPO = Path(__file__).resolve().parents[1]

SECTION_10 = {
    "ADJUSTABLE_INTERVAL_IN_SECONDS": 30,
    "SAFETY_CHECK_S": 900,
    "TRUSTED_WINDOW_S": 600,
    "SAMPLE_EVERY_S": 0.5,
    "POOL_FRAMES": 60,
    "TOLERANCE_MM": 1.7,
    "TRIGGER_MM": 0.85,
    "AGREE_MM": 1.4,
    "MIN_OFFSET_MM": 1.5,
    "INLIER_MM": 0.7,
    "YES_VOTES": (3, 4),
    "CLEAR_RATIO": 0.5,
    "HOLD_MAX_INTERVALS": 2,
    "CORE_MIN_WEIGHT": 0.2,
    "TILE_MM": 64,
    "CTRL_MARGIN_MM": 40,
    "EDGE_PIECE_MM": 80,
    "CANVAS_PX_PER_MM": 2.0,
    "MOTION_LEVEL": 0.02,
    "MAX_CLIPPED": 0.002,
    "DARK_LEVEL": 0.003,
    "MIN_TILE_EDGES": 0.02,
    "VARIETY_MAX_NCC": 0.98,
    "FIDUCIAL_MOVE_PX": 0.3,
    "MIN_PEAK_SNR": 6.0,
    "NULL_FACTOR": 1.5,
    "ECHO_FLOOR_CAMERA_PX": 2.5,
    "MIN_GOOD_TILES": 3,
    "MIN_TILE_FRAMES": 20,
    "MIN_SEAM_FRAMES": 20,
    "MIN_EDGE_SNR": 4.0,
    "LINE_INLIER_FRAC": 0.6,
    "FLAT_MAX_DEV": 0.01,
    "LAMP_WARN": 0.03,
    "REFERENCE_MODE": "auto",
    "WIENER_LAMBDA": 0.01,
    "SOURCE_RING": 10,
    "MATCH_MIN_NCC": 0.9,
}


def test_defaults_match_section_10():
    cfg = DetectorConfig()
    assert {k: getattr(cfg, k) for k in SECTION_10} == SECTION_10
    assert set(cfg.to_dict()) == set(SECTION_10)


def test_repo_detector_yaml_equals_defaults():
    assert DetectorConfig.from_yaml(REPO / "detector.yaml") == DetectorConfig()


def test_claude_md_section_10_equals_defaults():
    """The contract printed in CLAUDE.md section 10 must match the code, value for value."""
    import re

    import yaml

    text = (REPO / "CLAUDE.md").read_text()
    block = re.search(r"```yaml\n(.*?)```", text[text.index("## 10. Config"):], re.S).group(1)
    assert DetectorConfig.from_dict(yaml.safe_load(block)) == DetectorConfig()
    assert set(yaml.safe_load(block)) == set(SECTION_10)


def test_yaml_round_trip(tmp_path):
    import yaml

    cfg = DetectorConfig(TOLERANCE_MM=2.5, YES_VOTES=(2, 3), REFERENCE_MODE="on")
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()))
    assert DetectorConfig.from_yaml(path) == cfg


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="TOLERANCE_M"):
        DetectorConfig.from_dict({"TOLERANCE_M": 1.0})
    with pytest.raises(ValueError, match="HASH_MATCH_BITS"):  # dropped with matched-frame SSIM
        DetectorConfig.from_dict({"HASH_MATCH_BITS": 6})


@pytest.mark.parametrize(
    "override",
    [{"YES_VOTES": [5, 4]}, {"YES_VOTES": [0, 4]}, {"CORE_MIN_WEIGHT": 0.6}, {"POOL_FRAMES": 2.5},
     {"TOLERANCE_MM": -1}, {"CLEAR_RATIO": 1.0}, {"MIN_GOOD_TILES": True}, {"REFERENCE_MODE": "maybe"},
     {"REFERENCE_MODE": 1}, {"LINE_INLIER_FRAC": 0.0}, {"MATCH_MIN_NCC": 1.5}],
)
def test_invalid_values_rejected(override):
    with pytest.raises(ValueError):
        DetectorConfig.from_dict(override)


def test_types_are_coerced():
    cfg = DetectorConfig.from_dict({"TILE_MM": 64, "POOL_FRAMES": 30.0, "YES_VOTES": [2, 5], "REFERENCE_MODE": "off"})
    assert isinstance(cfg.TILE_MM, float) and isinstance(cfg.POOL_FRAMES, int)
    assert cfg.YES_VOTES == (2, 5) and cfg.REFERENCE_MODE == "off"


def test_tolerance_from_viewing_distance():
    # 1 arcminute at 6 m is 1.745 mm: the origin of the 1.7 mm default.
    assert tolerance_from_viewing_distance(6000.0) == pytest.approx(1.7453, abs=1e-4)
