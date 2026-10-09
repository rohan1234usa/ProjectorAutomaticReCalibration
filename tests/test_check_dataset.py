"""check_dataset: a dataset is held to what its scenario asked for, and a corrupted one fails, saying where.

The checker re-derives everything from scenario.yaml and setup.json with its own code, so each
test here corrupts one recorded value the simulator wrote and expects the check to name it.
"""

import copy
import json
import shutil

import cv2
import numpy as np
import pytest
import yaml

from scripts import check_dataset, make_dataset
from scripts.check_frames import frames_to_render
from scripts.dataset_files import read_png
from tests.scenes import TINY_BEZEL, tiny_shift_sweep

NUISANCES = [
    {"type": "lamp", "projector": "b", "gain": 0.85, "schedule": {"type": "ramp", "t0_s": 1, "duration_s": 2}},
    {"type": "room_light", "ambient": 0.05, "schedule": {"type": "step", "t0_s": 2}},
    {"type": "camera_bump", "shift_px": [3, -2], "rotation_deg": 0.05, "schedule": {"type": "step", "t0_s": 3}},
    {"type": "occluder", "t0_s": 1.5, "duration_s": 1, "height_mm": 260, "floor_mm": 300},
    {"type": "flicker", "projector": "a", "amplitude": 0.05},
    {"type": "sharpening"},
]


@pytest.fixture
def sweep_file(tmp_path):
    path = tmp_path / "tiny_shift.yaml"
    path.write_text(yaml.safe_dump(tiny_shift_sweep(), sort_keys=False))
    return path


@pytest.fixture(scope="module")
def sample_sweep(tmp_path_factory):
    """The tiny three-variant sweep with every frame hashed and frames 0, 4, 6 (the onset) and 8 stored."""
    root = tmp_path_factory.mktemp("checked")
    path = root / "tiny_shift.yaml"
    path.write_text(yaml.safe_dump(tiny_shift_sweep(), sort_keys=False))
    make_dataset.main([str(path), str(root / "sweep"), "--frames", "sample", "--every", "4"])
    return root / "sweep"


def _write(tmp_path, cfg, *extra):
    path = tmp_path / f"{cfg['name']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    make_dataset.main([str(path), str(tmp_path / cfg["name"]), *extra])
    return tmp_path / cfg["name"]


def _edit(meta, i, change):
    lines = [json.loads(s) for s in meta.read_text().splitlines()]
    change(lines[i])
    meta.write_text("".join(json.dumps(line) + "\n" for line in lines))


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


def test_checker_covers_every_kind_and_continuous_schedules(tmp_path):
    """Rotation, scale and keystone, stepped or ramped or drifting: closed forms and brute force all pass.

    Truth only (--frames none), so it runs in seconds. A corrupted schedule value or offset must fail.
    """
    cfg = tiny_shift_sweep()
    cfg["perturbation"] = {"b": {"kind": "rotation", "magnitude_px": 2, "pivot": "far_corner",
                                 "schedule": {"type": "step", "t0_s": 3}}}
    cfg["sweep"] = {
        "perturbation.b.kind": ["rotation", "scale", "keystone"],
        "perturbation.b.magnitude_px": [0, -1.5, 2],
        "perturbation.b.schedule": [{"type": "step", "t0_s": 3}, {"type": "ramp", "t0_s": 3, "duration_s": 2},
                                    {"type": "drift", "t0_s": 3, "rate_per_h": 1800}],
    }
    path = tmp_path / "kinds.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = tmp_path / "kinds"
    make_dataset.main([str(path), str(out)])
    assert check_dataset.main([str(out)])
    stepped = check_dataset.check(out / "kind=scale__magnitude_px=2__schedule=step-3")
    ramped = check_dataset.check(out / "kind=keystone__magnitude_px=-1.5__schedule=ramp-3-2")
    assert stepped["closed_form_frames"] == 12 and ramped["brute_force_frames"] > 0  # a partial keystone: brute force

    meta = out / "kind=rotation__magnitude_px=2__schedule=drift-3-1800" / "metadata.jsonl"
    original = meta.read_text()
    lines = [json.loads(s) for s in original.splitlines()]
    lines[-1]["perturbation"][0]["applied"] *= 1.5  # off the schedule by far more than a quantum
    meta.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert not check_dataset.main([str(out)])
    meta.write_text(original)
    meta = out / "kind=keystone__magnitude_px=-1.5__schedule=ramp-3-2" / "metadata.jsonl"
    lines = [json.loads(s) for s in meta.read_text().splitlines()]
    lines[7]["truth"]["offset_mm"] += 0.01
    meta.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert not check_dataset.main([str(out)])


def test_frames_are_rendered_again_and_must_match_their_hashes(sample_sweep, tmp_path):
    result = check_dataset.check(sample_sweep / "magnitude_px=4", 3)
    # 3 spread frames (0, 6, 11) and 3 of the 4 stored ones (0, 6, 8): 4 frames in all
    assert result["ok"] and result["rerendered"] == 4 and result["pngs_checked"] == 4
    assert check_dataset.main([str(sample_sweep), "--rerender", "2", "--strict"])
    wrong = tmp_path / "wrong_hash"
    shutil.copytree(sample_sweep / "magnitude_px=4", wrong)
    _edit(wrong / "metadata.jsonl", 5, lambda line: line.update(frame_sha256="0" * 64))
    assert "frame 5: the re-rendered frame does not match the recorded hash" in check_dataset.check(wrong, 12)["problems"]


def test_a_stored_frame_that_changed_is_caught(sample_sweep, tmp_path):
    copied = tmp_path / "png"
    shutil.copytree(sample_sweep / "magnitude_px=1", copied)
    png = copied / "frames" / "000004.png"
    cv2.imwrite(str(png), read_png(png) + np.uint16(1))  # one DN brighter everywhere
    (copied / "frames" / "000008.png").unlink()
    problems = check_dataset.check(copied)["problems"]
    assert "frame 4: frames/000004.png does not match the frame's recorded hash" in problems
    assert "frame 8: frames/000008.png is missing or unreadable" in problems


def test_a_rotation_recorded_as_a_shift_of_equal_offset_fails(tmp_path):
    """The closed-form offset alone would pass: only the request-derived h_actual catches it."""
    cfg = {**tiny_shift_sweep(), "name": "rotation"}
    cfg.pop("sweep")
    cfg["perturbation"] = {"b": {"kind": "rotation", "magnitude_px": 2, "pivot": "far_corner",
                                 "schedule": {"type": "step", "t0_s": 3}}}
    out = _write(tmp_path, cfg)
    assert check_dataset.main([str(out)])
    h_cal = np.array(json.loads((out / "setup.json").read_text())["projectors"]["b"]["h_cal_px_to_mm"])

    def swap(line):
        d = line["truth"]["offset_mm"]
        line["truth"]["h_actual"]["b"] = (np.array([[1.0, 0.0, d], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ h_cal).tolist()

    _edit(out / "metadata.jsonl", 9, swap)
    result = check_dataset.check(out)
    assert result["max_offset_error_mm"] < 1e-6  # the offset still matches the request
    assert "frame 9: h_actual of b is not its requested moves applied to h_cal" in result["problems"]
    assert "frame 9: h_rel is not D_B D_A^-1 of the recorded h_actual" in result["problems"]


@pytest.mark.parametrize("i, change, field", [
    (8, lambda line: line["nuisances"]["lamp_gain"].update(b=0.9), "nuisances.lamp_gain"),
    (8, lambda line: line["nuisances"].update(ambient=0.02), "nuisances.ambient"),
    (8, lambda line: line["nuisances"].update(camera_bump=[0.0]), "nuisances.camera_bump"),
    (3, lambda line: line["nuisances"].update(occluder=False), "nuisances.occluder"),
    (2, lambda line: line["truth"]["markers_visible"].pop(), "markers_visible"),
    (1, lambda line: line.update(t_s=0.51), "t_s"),
    (3, lambda line: line["content"].update(segments=[["0/0/0", 1.0]]), "content.segments"),
    (3, lambda line: line["content"].update(tag="deck_high"), "content.tag"),
])
def test_the_timeline_is_rechecked(tmp_path, i, change, field):
    """Lamp, room light, knock, passer-by, markers, frame times and content, each from the scenario alone."""
    cfg = {**copy.deepcopy(TINY_BEZEL), "name": "nuisances", "duration_s": 5, "trusted_window_s": 1,
           "nuisances": NUISANCES, "content": {"loop": True, "items": [{"type": "deck", "slides": 2, "hold_s": 1.2}]}}
    out = _write(tmp_path, cfg)
    assert check_dataset.check(out)["ok"]
    _edit(out / "metadata.jsonl", i, change)
    problems = check_dataset.check(out)["problems"]
    assert any(p.startswith(f"frame {i}: {field}") for p in problems), problems


def test_setup_json_must_agree_with_the_scenario(sample_sweep, tmp_path):
    copied = tmp_path / "setup"
    shutil.copytree(sample_sweep / "magnitude_px=0", copied)
    setup = json.loads((copied / "setup.json").read_text())
    setup["reference"]["available"] = True
    (copied / "setup.json").write_text(json.dumps(setup))
    assert "setup.json: reference.available is not False, as the scenario says" in check_dataset.check(copied)["problems"]


def test_a_pairing_that_cannot_run_is_reported_and_strict_fails_it(sweep_file, tmp_path, capsys):
    out = tmp_path / "sweep"
    make_dataset.main([str(sweep_file), str(out)])  # --frames none: no frame hashes to pair
    assert check_dataset.main([str(out)])
    report = [json.loads(s) for s in capsys.readouterr().out.splitlines()]
    paired = next(r for r in report if "paired_groups" in r)
    assert paired["paired_skipped_no_hashes"] == [["magnitude_px=0", "magnitude_px=1", "magnitude_px=4"]]
    assert not check_dataset.main([str(out), "--strict"])
    assert not check_dataset.main([str(out / "magnitude_px=1"), "--rerender", "2", "--strict"])  # nothing to render against


def test_rerendering_takes_at_most_k_stored_frames():
    """A dataset written with --frames all stores every frame; a spot check must stay a spot check."""
    every = [{"i": i, "png": f"frames/{i:06d}.png"} for i in range(2400)]
    assert len(frames_to_render(every, 10)) == 10  # the same ten spread frames, stored or not
    sparse = [{"i": i, "png": f"frames/{i:06d}.png" if i in (0, 600, 1230) else None} for i in range(2400)]
    assert set(frames_to_render(sparse, 3)) == {0, 1200, 2399} | {0, 600, 1230}  # no more stored than k: all


def test_a_shift_is_held_to_its_size_beside_a_rotation(tmp_path):
    """With two kinds at once the offset has no closed form, but the shift's own vector still has one."""
    cfg = {**tiny_shift_sweep(), "name": "mixed"}
    cfg.pop("sweep")
    step = {"type": "step", "t0_s": 3}
    cfg["perturbation"] = {"b": [{"kind": "shift", "magnitude_px": 1, "schedule": step},
                                 {"kind": "rotation", "magnitude_px": 1, "pivot": "centre", "schedule": step}]}
    out = _write(tmp_path, cfg)
    result = check_dataset.check(out)
    assert result["ok"] and result["brute_force_frames"] == 6  # frames 6-11, measured on the verified h_actual
    _edit(out / "metadata.jsonl", 9, lambda line: line["perturbation"][0].update(
        vector_mm=[1.1 * v for v in line["perturbation"][0]["vector_mm"]]))
    assert any(p.startswith("frame 9: shift of") for p in check_dataset.check(out)["problems"])


def test_a_partial_keystone_is_tied_to_the_full_strength_one(tmp_path):
    """A ramped keystone's k is m x the strength that gives the requested size at m = 1, in every frame."""
    cfg = {**tiny_shift_sweep(), "name": "keystone"}
    cfg.pop("sweep")
    cfg["perturbation"] = {"b": {"kind": "keystone", "magnitude_px": 2, "pivot": "centre",
                                 "schedule": {"type": "ramp", "t0_s": 3, "duration_s": 2}}}
    out = _write(tmp_path, cfg)
    assert check_dataset.check(out)["ok"]
    _edit(out / "metadata.jsonl", 8, lambda line: line["perturbation"][0].update(
        k_per_mm=1.01 * line["perturbation"][0]["k_per_mm"]))
    assert any("keystone of b has k / m" in p for p in check_dataset.check(out)["problems"])


def test_an_empty_dataset_fails_with_a_message(sample_sweep, tmp_path):
    copied = tmp_path / "empty"
    shutil.copytree(sample_sweep / "magnitude_px=0", copied)
    (copied / "metadata.jsonl").write_text("")
    result = check_dataset.check(copied)
    assert not result["ok"] and result["problems"] == ["metadata.jsonl: no frames"]
