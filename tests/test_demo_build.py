"""The demo site's renderer sharing, captions, data files and build (demo/renders.py, demo/site.py)."""

import re

import pytest
import yaml

from demo import gallery, manifest
from demo.renders import Family
from demo.site import MARKER, SCHEMA, _reuse, build, check_target, read_data, write_data
from sim.dataset import frame_hash
from sim.frames import FrameSource
from sim.scenario import load_scenarios, scenario_from_dict
from tests.scenes import tiny_shift_sweep


@pytest.fixture
def tiny_sweep(tmp_path):
    path = tmp_path / "tiny_shift.yaml"
    path.write_text(yaml.safe_dump(tiny_shift_sweep(), sort_keys=False))
    return {s.variant: s for s in load_scenarios(path)}


def test_a_shared_renderer_draws_each_variant_bit_for_bit(tiny_sweep):
    """The base's renderer with a variant's landing geometry gives that variant's own frame, and its truth."""
    names = list(tiny_sweep)
    with Family(tiny_sweep, names[0], "tiny_shift") as fam:
        for variant in names[1:]:
            own = FrameSource(tiny_sweep[variant])
            for i in (0, 7):  # before and after the step
                assert frame_hash(fam.frame(variant, i)) == frame_hash(own.frame(i)), (variant, i)
                assert fam.facts(variant, i)["offset_mm"] == own.truth(i)["truth"]["offset_mm"]
        assert not fam.electrons(names[1], 7).flags.writeable  # shared from the memo: read-only
    with pytest.raises(RuntimeError, match="closed"):
        fam.frame(names[0], 0)


def test_a_family_refuses_a_variant_that_needs_its_own_renderer(tiny_sweep):
    names = list(tiny_sweep)
    cfg = {k: v for k, v in tiny_shift_sweep().items() if k != "sweep"}
    other = scenario_from_dict({**cfg, "screen": {**cfg["screen"], "ambient": 0.05}}, variant="brighter_room")
    with Family({**tiny_sweep, "brighter_room": other}, names[0], "tiny_shift") as fam:
        assert fam.shares(names[2]) and not fam.shares("brighter_room")
        with pytest.raises(ValueError, match="beyond its perturbation"):
            fam.state("brighter_room", 0)


def test_nuisance_captions_come_from_the_scenario():
    """The camera bump of aligned_nuisances is (3, -2) px: the caption says 3.6 px, not 3."""
    variants = {s.variant: s for s in load_scenarios(manifest.SCENARIOS / f"{manifest.NUISANCE}.yaml")}
    bump = gallery.caption(variants["nuisances=camera_bump"].nuisances, 0.02, 0)
    assert "3.6 px (+3, -2)" in bump and "0.05°" in bump and "hidden" not in bump
    every = gallery.caption(variants[manifest.ALL_NUISANCES].nuisances, 0.02, 1)
    for words in ("knocked", "lamp fades to 85%", "from 0.02 to 0.05", "crosses", "flickers", "sharpening",
                  "1 marker is hidden"):
        assert words in every, words


def test_data_files_round_trip_and_stale_ones_are_not_reused(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        (root / "data").mkdir(parents=True)
    data = {"meta": {"schema": SCHEMA, "built": "2026-10-10T12:00:00+00:00"}, "x": [1.5, None, "</script>"]}
    write_data(old, "tests", data)
    assert read_data(old, "tests") == data
    (old / "data" / "junit.xml").write_text("<testsuites/>")
    assert _reuse(old, new, ["tests"], ["data/junit.xml"], "tests", lambda m: None) == {"tests": data}
    assert (new / "data" / "tests.js").read_text() == (old / "data" / "tests.js").read_text()
    write_data(old, "tests", {**data, "meta": {"schema": SCHEMA - 1}})
    assert _reuse(old, new, ["tests"], [], "tests", lambda m: None) is None
    assert read_data(old, "missing") is None


def test_build_refuses_to_replace_a_folder_it_did_not_make(tmp_path):
    (tmp_path / "keep.txt").write_text("mine")
    with pytest.raises(SystemExit):
        check_target(tmp_path)
    (tmp_path / MARKER).write_text("")
    check_target(tmp_path)  # a previous build: fine to replace


@pytest.mark.slow
def test_full_build_at_fast_quality(tmp_path):
    out = tmp_path / "site"
    build(out, samples="fast", tests="skip", log=lambda msg: None)
    for page in ("index", "samples", "algorithm", "tests"):
        text = (out / f"{page}.html").read_text()
        assert '<header class="site-header">' in text and "<!-- header -->" not in text
        for script in re.findall(r'<script src="([^"]+)"', text):
            assert (out / script).exists(), (page, script)
    for name in ("samples", "algorithm"):
        data = read_data(out, name)
        assert data["meta"]["schema"] == SCHEMA and data["meta"]["quality"] == "fast"
        for src in set(re.findall(r'"src": ?"(img/[^"]+)"', (out / "data" / f"{name}.js").read_text())):
            assert (out / src).exists(), src
    overview = read_data(out, "overview")
    assert overview["tests"] is None and overview["hero"]["size_px"] == 8.0  # no test data to reuse
    assert "window.DEMO[\"tests\"] = null;" in (out / "data" / "tests.js").read_text()  # the page says so
    assert (out / MARKER).exists()
    build(out, samples="skip", tests="skip", log=lambda msg: None)  # pages only: the figures are carried over
    assert read_data(out, "samples")["meta"]["quality"] == "fast"
