"""Datasets on disk: round trip, byte-identical reruns, nothing but the setup in setup.json, and the checker."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import check_dataset, make_dataset
from sim.dataset import frame_hash, read_dataset, write_dataset
from sim.scenario import load_scenarios
from tests.scenes import tiny_shift_sweep

# What setup.json may contain, recursively: the blending setup, the marker layout, locked camera
# settings. Anything else (camera pose, optics, light levels, geometry after drift) is truth.
ALLOWED = {
    "version": None,
    "screen": {"size_mm": None},
    "projectors": {"*": {"resolution": None, "h_cal_px_to_mm": None, "gamma_assumed": None}},
    "content_rect_mm": None,
    "blend": {"rule": None, "shape": None, "space": None, "black_uplift": None},
    "markers": {"dictionary": None, "size_mm": None, "quiet_zone_cells": None,
                "items": [{"id": None, "centre_mm": None, "rotation_deg": None}]},
    "camera": {"resolution": None, "color": None, "bit_depth": None, "gamma": None, "pedestal_dn": None,
               "gain_dn_per_e": None, "full_well_e": None, "read_noise_e": None, "exposure_s": None},
    "reference": {"available": None},
}


def _within(value, allowed, where="setup"):
    if allowed is None:
        assert not isinstance(value, dict), f"{where}: unexpected nested keys"
        return
    if isinstance(allowed, list):
        for i, item in enumerate(value):
            _within(item, allowed[0], f"{where}[{i}]")
        return
    for key, sub in value.items():
        rule = allowed.get(key, allowed.get("*", "missing"))
        assert rule != "missing", f"{where}.{key} is not something the detector may read"
        _within(sub, rule, f"{where}.{key}")


@pytest.fixture
def sweep_file(tmp_path):
    path = tmp_path / "tiny_shift.yaml"
    path.write_text(yaml.safe_dump(tiny_shift_sweep(), sort_keys=False))
    return path


def test_write_and_read_back(sweep_file, tmp_path):
    scenario = load_scenarios(sweep_file)[1]
    summary = write_dataset(scenario, tmp_path / "ds", frames="sample", every=5)
    ds = read_dataset(tmp_path / "ds")
    assert summary["n_frames"] == len(ds.metadata) == 12
    stored = [line["i"] for line in ds.metadata if line["png"]]
    assert {0, 5, 10}.issubset(stored) and 6 in stored  # every 5th frame, plus the onset (an event)
    for i in (0, 1, 6, 7):  # from PNGs and re-rendered, each checked against its hash
        assert frame_hash(ds.frame(i)) == ds.metadata[i]["frame_sha256"]
    assert ds.scenario().variant == scenario.variant


def test_reruns_are_byte_identical_whatever_the_parallelism(sweep_file, tmp_path):
    for out, jobs in (("run1", "1"), ("run2", "2")):
        make_dataset.main([str(sweep_file), str(tmp_path / out), "--frames", "sample", "--every", "4", "--jobs", jobs])
    files = sorted(p.relative_to(tmp_path / "run1") for p in (tmp_path / "run1").rglob("*") if p.is_file())
    compared = [f for f in files if f.name not in ("timing.json", "dataset.json")]
    assert any(f.suffix == ".png" for f in compared) and any(f.name == "metadata.jsonl" for f in compared)
    for f in compared:
        assert (tmp_path / "run1" / f).read_bytes() == (tmp_path / "run2" / f).read_bytes(), str(f)


def test_frames_none_writes_truth_only(sweep_file, tmp_path):
    write_dataset(load_scenarios(sweep_file)[2], tmp_path / "ds", frames="none")
    ds = read_dataset(tmp_path / "ds")
    assert not (tmp_path / "ds" / "frames").exists()
    assert all(line["frame_sha256"] is None and line["png"] is None for line in ds.metadata)
    assert ds.metadata[-1]["truth"]["offset_px"] == pytest.approx(4.0)


def test_setup_json_holds_only_what_the_detector_may_read(sweep_file, tmp_path):
    write_dataset(load_scenarios(sweep_file)[2], tmp_path / "ds", frames="none")
    setup = json.loads((tmp_path / "ds" / "setup.json").read_text())
    _within(setup, ALLOWED)
    assert len(setup["markers"]["items"]) == 8 and setup["camera"]["color"] == "mono"
    h_cal = np.array(setup["projectors"]["b"]["h_cal_px_to_mm"])
    assert h_cal.shape == (3, 3)


def test_checker_passes_a_good_sweep_and_catches_a_wrong_offset(sweep_file, tmp_path, capsys):
    out = tmp_path / "sweep"
    make_dataset.main([str(sweep_file), str(out), "--frames", "sample"])
    assert check_dataset.main([str(out)])
    meta = out / "magnitude_px=1" / "metadata.jsonl"
    original = meta.read_text()
    lines = [json.loads(s) for s in original.splitlines()]
    lines[8]["truth"]["offset_mm"] += 1e-5
    meta.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert not check_dataset.main([str(out)])
    meta.write_text(original)
    # The check reads what was asked for from scenario.yaml: a dataset whose truth matches a
    # different size than the one requested must fail, even though it is self-consistent.
    spec = out / "magnitude_px=4" / "scenario.yaml"
    spec.write_text(spec.read_text().replace("magnitude_px: 4", "magnitude_px: 4.01"))
    assert not check_dataset.main([str(out)])
