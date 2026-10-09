"""compare_datasets: two runs of the same scenario are the same dataset, and a difference is located."""

import json
import shutil

import cv2
import numpy as np
import pytest
import yaml

from scripts import compare_datasets, make_dataset
from scripts.dataset_files import pixel_hash, read_png
from sim.dataset import dumps, frame_hash
from tests.scenes import tiny_shift_sweep


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The tiny three-variant sweep written twice, in one process and split over two."""
    root = tmp_path_factory.mktemp("compare")
    path = root / "tiny_shift.yaml"
    path.write_text(yaml.safe_dump(tiny_shift_sweep(), sort_keys=False))
    for out, jobs in (("run1", "1"), ("run2", "2")):
        make_dataset.main([str(path), str(root / out), "--frames", "sample", "--every", "4", "--jobs", jobs])
    return root, path


def _copy(runs, name):
    root, _ = runs
    shutil.copytree(root / "run2", root / name)
    return root / name


def _rewrite(meta, change):
    lines = [json.loads(s) for s in meta.read_text().splitlines()]
    for line in lines:
        change(line)
    meta.write_text("".join(dumps(line) + "\n" for line in lines))  # the simulator's own formatting


def test_two_runs_of_the_same_sweep_are_the_same_dataset(runs):
    root, _ = runs
    results, summary = compare_datasets.compare(root / "run1", root / "run2")
    assert summary["ok"] and summary["variants"] == 3 and summary["differing"] == 0
    assert all(r["pngs"]["same_files"] >= 3 for r in results)  # frames 0, 4 and 8 at least
    assert compare_datasets.main([str(root / "run1"), str(root / "run2")])


def test_a_changed_metadata_line_is_reported_by_frame_and_field(runs):
    copy = _copy(runs, "offset")
    meta = copy / "magnitude_px=1" / "metadata.jsonl"
    lines = meta.read_text().splitlines()
    line = json.loads(lines[8])
    line["truth"]["offset_mm"] += 1e-5
    lines[8] = dumps(line)
    meta.write_text("\n".join(lines) + "\n")
    results, summary = compare_datasets.compare(runs[0] / "run1", copy)
    changed = {r["variant"]: r for r in results}["magnitude_px=1"]
    assert not summary["ok"] and summary["differing"] == 1
    assert changed["metadata"]["differing_frames"] == 1
    assert changed["metadata"]["first"] == {"i": 8, "fields": ["truth.offset_mm"]}


def test_missing_and_altered_frames_are_reported_by_path(runs):
    copy = _copy(runs, "pngs")
    (copy / "magnitude_px=4" / "frames" / "000004.png").unlink()
    png = copy / "magnitude_px=4" / "frames" / "000008.png"
    cv2.imwrite(str(png), read_png(png) + np.uint16(1))  # one DN brighter everywhere
    results, _ = compare_datasets.compare(runs[0] / "run1", copy)
    pngs = {r["variant"]: r for r in results}["magnitude_px=4"]["pngs"]
    assert pngs["only_in_a"] == ["frames/000004.png"] and pngs["differing"] == ["frames/000008.png"]


def test_a_partial_sweep_lists_the_variants_it_lacks(runs, tmp_path):
    root, path = runs
    make_dataset.main([str(path), str(tmp_path / "partial"), "--frames", "sample", "--every", "4",
                       "--variants", "magnitude_px=4"])
    _, summary = compare_datasets.compare(root / "run1", tmp_path / "partial")
    assert not summary["ok"] and summary["only_in_a"] == ["magnitude_px=0", "magnitude_px=1"]
    assert summary["variants.json"] != "same"


def test_ignored_fields_hide_a_known_schema_difference(runs):
    copy = _copy(runs, "schema")
    for variant in ("magnitude_px=0", "magnitude_px=1", "magnitude_px=4"):
        _rewrite(copy / variant / "metadata.jsonl", lambda line: line["nuisances"].pop("lamp_gain"))
    _, summary = compare_datasets.compare(runs[0] / "run1", copy)
    assert not summary["ok"]
    results, summary = compare_datasets.compare(runs[0] / "run1", copy, ignore=("nuisances",))
    assert summary["ok"], results


def test_a_sweep_root_is_not_compared_with_a_single_dataset(runs):
    root, _ = runs
    with pytest.raises(SystemExit):
        compare_datasets.main([str(root / "run1"), str(root / "run1" / "magnitude_px=0")])


@pytest.mark.parametrize("shape", [(5, 7), (4, 6, 3)])
def test_the_checkers_frame_hash_is_the_simulators(shape):
    frame = np.random.default_rng(3).integers(0, 65536, size=shape, dtype=np.uint16)
    assert pixel_hash(frame) == frame_hash(frame)
