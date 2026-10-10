"""The demo site (demo/): its readers of the documents and the test run, its frame choices and its pictures."""

import textwrap

import cv2
import numpy as np
import pytest
import yaml

from demo import cepstrum, evidence, figures, images, manifest, markdown, sources, testrun
from demo.overlays import edge_pieces, rotate_cw
from demo.renders import Family
from sim.planar import centroid, points_in_convex, rect_polygon
from sim.scenario import load_scenarios, scenario_from_dict
from tests.scenes import TINY_BEZEL

REPO = sources.REPO


def test_markdown_lists_tables_and_escaping():
    text = textwrap.dedent("""\
        ## A heading

        A paragraph with `code <x>`, **bold**, *italic* and a < sign.

        1. **First.** Item text
           that continues.
           - nested one;
           - nested two.

           A second paragraph of the first item.
        2. Second item.

        | a | b |
        |---|---|
        | 1 | `2` |
        """)
    html = markdown.to_html(text)
    assert '<h2 id="a-heading">A heading</h2>' in html
    assert "<code>code &lt;x&gt;</code>" in html and "a &lt; sign" in html
    assert "<strong>bold</strong>" in html and "<em>italic</em>" in html
    assert html.count("<ol>") == 1 and html.count("<li>") == 4
    assert "<ul><li>nested one;</li><li>nested two.</li></ul>" in html
    assert "<p>A second paragraph of the first item.</p>" in html
    assert "<td>1</td><td><code>2</code></td>" in html


def test_markdown_bold_around_code_and_safe_links():
    """Bold that holds a code span pairs up; link targets are escaped once, and only web, mail or relative links."""
    html = markdown.inline("3. **`aligned_slides`** reads *`x`* [ok](https://a.org/?p=1&q=2) [no](javascript:alert)")
    assert "<strong><code>aligned_slides</code></strong>" in html and "<em><code>x</code></em>" in html
    assert "**" not in html and '<a href="https://a.org/?p=1&amp;q=2">ok</a>' in html
    assert "javascript" not in html.split("ok</a>")[1].split(">")[0] and "<a" not in html.split("ok</a>")[1]
    assert markdown.inline("`<b>` and **bold `*not em*`**") == \
        "<code>&lt;b&gt;</code> and <strong>bold <code>*not em*</code></strong>"


def test_build_order_states_follow_the_status_column():
    """Done when its status says so; the first phase that is not done is next, the rest later (a done
    phase may come after the next one, as 2b and 3 may run in parallel)."""
    table = "\n".join([
        "## 8. Build order", "", "| Phase | Scope | Done condition | Depends on | Status |", "|---|---|---|---|---|",
        "| 1 One | s | d | — | **done 2026-10-06**: numbers |", "| 2 Two | s | d | 1 | |",
        "| 3 Three | s | d | 1 | done 2026-10-08 |", "| 4 Four | s | d | 2 | |", "", "## 9. Next"])
    phases = sources.phases(table)
    assert [(p["number"], p["title"], p["state"]) for p in phases] == [
        ("1", "One", "done"), ("2", "Two", "next"), ("3", "Three", "done"), ("4", "Four", "later")]


def test_build_order_table_from_claude_md():
    phases = sources.phases((REPO / "CLAUDE.md").read_text())
    states = [p["state"] for p in phases]
    assert phases and phases[0]["number"] == "1" and states[0] == "done"
    rest = [s for s in states if s != "done"]
    assert rest == [] or (rest[0] == "next" and set(rest[1:]) <= {"later"})
    assert all(p["status"] for p in phases if p["state"] == "done")
    assert all(p["title"] and p["scope"] and p["done_condition"] for p in phases)


def test_config_values_and_sections_match_detector_yaml():
    text = (REPO / "detector.yaml").read_text()
    cfg = sources.config(text)
    assert cfg["values"] == yaml.safe_load(text)
    keys = [item["key"] for section in cfg["sections"] for item in section["items"]]
    assert sorted(keys) == sorted(cfg["values"]) and len(keys) == len(set(keys))
    assert all(section["name"] for section in cfg["sections"])


def test_findings_split_into_dated_entries():
    entries = sources.findings((REPO / "docs" / "findings.md").read_text())
    assert entries and all(e["date"][:4].isdigit() for e in entries)
    assert len({e["anchor"] for e in entries}) == len(entries)


JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="4">
<testcase classname="tests.test_x" name="test_ok[side_by_side]" time="0.25"/>
<testcase classname="tests.test_x" name="test_bad" time="1.5"><failure message="assert 1 == 2">trace</failure></testcase>
<testcase classname="tests.test_x" name="test_skip" time="0"><skipped message="no camera"/></testcase>
<testcase classname="tests.test_y" name="test_err" time="0.1"><error message="fixture broke"/></testcase>
</testsuite></testsuites>"""


def test_junit_cases_are_parsed_and_summarized(tmp_path):
    (tmp_path / "junit.xml").write_text(JUNIT)
    cases = testrun.parse_junit(tmp_path / "junit.xml")
    assert [(c["file"], c["func"], c["params"], c["outcome"]) for c in cases] == [
        ("test_x", "test_ok", "side_by_side", "passed"), ("test_x", "test_bad", "", "failed"),
        ("test_x", "test_skip", "", "skipped"), ("test_y", "test_err", "", "error")]
    assert "assert 1 == 2" in cases[1]["message"]
    tests = {"test_x": {"doc": "X.", "area": "Other", "functions": {"test_ok": {"doc": "It works.\n\nMore.", "slow": True}}},
             "test_y": {"doc": "Y.", "area": "Other", "functions": {}}}
    summary = testrun.summarize(cases, tests, {"slow_not_run": ["a::b"]})
    assert summary["totals"] == {"passed": 1, "failed": 1, "error": 1, "skipped": 1, "total": 4, "slow_not_run": 1,
                                 "seconds": 1.9}
    assert cases[0]["title"] == "It works." and cases[0]["slow"]
    assert cases[1]["title"] == "Bad"


def test_test_files_are_read_without_importing_them(tmp_path):
    (tmp_path / "test_z.py").write_text('"""Module doc."""\nimport pytest\n\n\n@pytest.mark.slow\n'
                                        'def test_one():\n    """Doc one."""\n    assert True\n\n\ndef test_two_words():\n    pass\n')
    read = testrun.read_tests(tmp_path)["test_z"]
    assert read["doc"] == "Module doc." and read["area"] == "Other"
    one, two = read["functions"]["test_one"], read["functions"]["test_two_words"]
    assert one["slow"] and one["doc"] == "Doc one." and one["source"].startswith("@pytest.mark.slow")
    assert not two["slow"] and testrun.humanize("test_two_words") == "Two words"


def test_every_chosen_frame_shows_what_its_caption_says():
    assert manifest.validate() == []


def test_crossing_finds_a_step_and_says_none_without_one():
    xs = np.arange(0.0, 60.0, 0.5)
    step = np.where(xs < 31.3, 10.0, 30.0)
    step[(xs > 30) & (xs < 32)] = 20.0 + 10.0 * (xs[(xs > 30) & (xs < 32)] - 31.0)  # a ramp through 20 at 31
    assert evidence.crossing(xs, step) == pytest.approx(31.0, abs=1e-9)
    assert evidence.crossing(xs, np.full_like(xs, 5.0)) is None  # flat: no edge, never 0


def test_sample_mm_reads_float32_only():
    """The demo's one warp refuses other types, which OpenCV would round to 1/32 px."""
    image = np.arange(12.0, dtype=np.float32).reshape(3, 4)
    got = images.sample_mm(image, np.eye(3), np.array([1.5]), np.array([1.0]))
    assert got.dtype == np.float32 and got[0, 0] == pytest.approx(5.5)
    for bad in (image.astype(np.float64), image.astype(np.uint16), np.dstack([image, image])):
        with pytest.raises(TypeError):
            images.sample_mm(bad, np.eye(3), np.array([1.0]), np.array([1.0]))


def test_display_curves_and_shrinking():
    white = 1000.0
    ramp = np.linspace(0, white, 50, dtype=np.float32)
    shown = images.natural(ramp, white)
    assert shown[0] == 0 and shown[-1] == 255 and np.all(np.diff(shown.astype(int)) >= 0)
    line = np.zeros((40, 40), np.float32)
    line[:, 17] = 1.0  # one pixel wide
    assert images.shrink_max(line, 4)[:, 4].min() == 1.0
    assert images.shrink(line, 10)[:, 4].max() < 0.5  # a block average all but erases it
    assert images.crop(line, (-5, -5, 10, 10)).shape == (10, 10)


def test_zoomed_rotation_matches_opencv():
    img = np.zeros((4, 7), np.uint8)
    img[1, 5] = 255
    turned = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    u, v = rotate_cw(np.array([[5.0, 1.0]]), img.shape[0])[0]
    assert turned[int(v), int(u)] == 255


def test_side_by_side_edge_pieces():
    setup = scenario_from_dict(TINY_BEZEL).scene.setup
    pieces = edge_pieces(setup, piece_mm=20.0)
    inner = [p for p in pieces if p["kind"] == "inner"]
    # A's right edge lies inside B and B's left edge inside A; everything else faces the unlit screen.
    assert {p["owner"] for p in inner} == {"a", "b"}
    assert all(abs(p["normal"][0]) > 0.99 for p in inner)
    assert all((p["normal"][0] > 0) == (p["owner"] == "a") for p in inner)
    for p in pieces:
        assert abs(np.hypot(*p["normal"]) - 1) < 1e-9


@pytest.mark.parametrize("preset", manifest.PRESETS)
def test_edge_pieces_of_every_arrangement(preset):
    """Pieces tile each outline, point outward, and are inner only inside the other box and the content."""
    scenario = {s.variant: s for s in load_scenarios(manifest.SCENARIOS / "arrangements.yaml")}[
        manifest.SHOTS[f"arrangement_{preset}"].variant]
    setup = scenario.scene.setup
    content = rect_polygon(*setup.content_rect_mm)
    pieces = edge_pieces(setup)
    for name, other in (("a", "b"), ("b", "a")):
        box, mine = setup.box_mm(name), [p for p in pieces if p["owner"] == name]
        perimeter = sum(np.hypot(*(box[(k + 1) % 4] - box[k])) for k in range(4))
        assert sum(np.hypot(*(np.subtract(p["p1"], p["p0"]))) for p in mine) == pytest.approx(perimeter, abs=0.1)
        for p in mine:
            mid, normal = np.array(p["mid"]), np.array(p["normal"])
            assert np.dot(normal, mid - centroid(box)) > 0  # outward
            in_other = bool(points_in_convex(setup.box_mm(other), mid, margin=0.5))
            in_content = bool(points_in_convex(content, mid, margin=-0.5))
            assert p["kind"] == ("outer" if not in_other else "inner" if in_content else "margin")
        assert any(p["kind"] == "outer" for p in mine) and any(p["kind"] == "inner" for p in mine)
    if preset == "rotated":  # the turned box pokes out of the content inside the other box
        assert any(p["kind"] == "margin" for p in pieces)


def _pools(echo_px: int, n: int = 56, tiles: int = 40, seed: int = 3) -> dict[str, np.ndarray]:
    """Log spectra of random texture tiles; the core ones carry an echo at `echo_px` along x."""
    rng = np.random.default_rng(seed)
    pools = {k: np.zeros((n, n)) for k in ("core", "ctrl_a", "ctrl_b")}
    for _ in range(tiles):
        for k in pools:
            s = cv2.GaussianBlur(rng.normal(size=(n, n + 16)), (0, 0), 1.0)
            tile = s[:, 8 : 8 + n]
            if k == "core" and echo_px:
                tile = 0.5 * tile + 0.5 * s[:, 8 - echo_px : 8 - echo_px + n]
            pools[k] += cepstrum.log_spectrum(tile) / tiles
    return pools


def test_teaching_cepstrum_finds_an_echo_and_not_its_aligned_twin():
    cfg = sources.config((REPO / "detector.yaml").read_text())["values"]
    echo = cepstrum.score(_pools(6), 2.5, cfg)
    assert echo["verdict"] == "resolved" and abs(echo["radius_px"] - 6) < 0.5 and abs(echo["at"][1]) < 0.5
    assert cepstrum.score(_pools(0), 2.5, cfg)["verdict"] == "none"


def test_a_tiny_frame_and_its_overlay(tmp_path):
    scenario = scenario_from_dict(TINY_BEZEL)
    with Family({scenario.variant: scenario}, scenario.variant) as fam:
        fig = figures.frame(tmp_path, fam, "tiny", scenario.variant, 0, width=400)
    assert (tmp_path / fig["img"]["src"]).stat().st_size > 1000
    lay = fig["layers"]
    for key in ("a", "b", "overlap"):
        poly = np.array(lay[key])
        assert len(poly) >= 3
        assert np.all(poly >= 0) and np.all(poly[:, 0] <= lay["w"]) and np.all(poly[:, 1] <= lay["h"])
    assert fig["facts"]["aligned"] and len(lay["markers"]) == 8
