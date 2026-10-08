"""Bezel markers: layout, exact painting, and that a camera finds them where they truly are.

The demo-scale checks are the Phase 2 done condition: at ambient 0.02 all 8 markers are found
in single noisy frames, and their centres (from edge-line fits, tests/markers.py) land within
0.2 camera px of the truth. In a dark room (ambient 0.0003) none are found.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from sim.fiducials import CELLS, MarkerSet, auto_layout, cell_matrix, check_on_bezel, paint, visible
from sim.frames import FrameSource
from sim.planar import apply_h, rect_polygon
from sim.scenario import load_scenario, scenario_from_dict
from sim.screen import Bezel, Screen, ScreenGrid
from tests.markers import centre_of, detect, edge_lines, to_8bit
from tests.scenes import TINY_BEZEL

REPO = Path(__file__).resolve().parents[1]
DEMO_SCREEN = Screen((4000.0, 1500.0), bezel=Bezel(width_mm=150.0))


def test_cell_matrices_are_distinct_aruco_codes():
    m = [cell_matrix(i) for i in range(8)]
    assert all(c.shape == (CELLS, CELLS) for c in m)
    assert all(c[0].all() and c[-1].all() and c[:, 0].all() and c[:, -1].all() for c in m)  # black border
    assert len({c.tobytes() for c in m}) == 8


def test_auto_layout_corners_and_overlap_pairs():
    overlap = rect_polygon(1795.0, 187.5, 2205.0, 1312.5)  # the demo's vertical overlap band
    markers = MarkerSet(auto_layout(DEMO_SCREEN, overlap, 80.0, 1))
    check_on_bezel(markers, DEMO_SCREEN)
    c = np.array(markers.centres_mm)
    assert np.allclose(c[:4], [[-75, -75], [4075, -75], [4075, 1575], [-75, 1575]])
    assert np.allclose(c[4:], [[1795, -75], [2205, -75], [1795, 1575], [2205, 1575]])  # above and below its edges
    horizontal = rect_polygon(1000.0, 650.0, 3000.0, 850.0)  # a stacked arrangement's band
    side = np.array(auto_layout(DEMO_SCREEN, horizontal, 80.0, 1))[4:]
    assert np.allclose(side, [[-75, 650], [-75, 850], [4075, 650], [4075, 850]])


def test_layout_refuses_a_bezel_too_narrow_for_the_markers():
    narrow = Screen((4000.0, 1500.0), bezel=Bezel(width_mm=100.0))
    with pytest.raises(ValueError, match="bezel of at least"):
        auto_layout(narrow, rect_polygon(1795.0, 187.5, 2205.0, 1312.5), 80.0, 1)


def test_paint_is_exact_area_coverage():
    """Paper and ink areas come out exact, wherever the marker sits relative to the grid."""
    grid = ScreenGrid.covering_extent((-150.0, -150.0, 0.0, 0.0), 1.37)  # the top-left bezel corner
    for centre in ((-75.0, -75.0), (-74.63, -75.21)):
        markers = MarkerSet((centre,), size_mm=80.0)
        refl = np.full(grid.shape, 0.05, np.float32)
        paint(refl, grid, markers, background=0.05)
        ink_cells = cell_matrix(0).sum()
        cell_area = (markers.cell_mm * grid.px_per_mm) ** 2
        expected = (0.8 - 0.05) * (markers.footprint_mm * grid.px_per_mm) ** 2 + (0.04 - 0.8) * ink_cells * cell_area
        assert float(np.sum(refl.astype(np.float64) - 0.05)) == pytest.approx(expected, rel=1e-5)
        assert refl.min() >= 0.04 - 1e-6 and refl.max() <= 0.8 + 1e-6


def _measure(scenario, frames: list[int]) -> tuple[list[list[int]], np.ndarray]:
    """Ids found in each frame, and every found marker's centre error in camera px."""
    source = FrameSource(scenario)
    camera, markers, screen = scenario.scene.camera, scenario.scene.markers, scenario.scene.screen
    paper_e = markers.paper * screen.ambient * camera.electrons_per_unit_radiance
    ids, errors = [], []
    for i in frames:
        electrons = camera.decode(source.frame(i))
        found = detect(to_8bit(electrons, paper_e))
        ids.append(sorted(found))
        for k, corners in found.items():
            truth = apply_h(camera.h_mm_to_px, np.array(markers.centres_mm[k]))
            errors.append(float(np.hypot(*(centre_of(edge_lines(electrons, corners)) - truth))))
    return ids, np.array(errors)


def test_markers_found_in_a_tiny_frame():
    scenario = scenario_from_dict(TINY_BEZEL)
    camera, markers = scenario.scene.camera, scenario.scene.markers
    assert visible(markers, camera.h_mm_to_px, camera.resolution) == list(range(8))
    ids, errors = _measure(scenario, [0])
    assert ids == [list(range(8))]
    assert errors.max() < 0.3  # 6-px cells here; the demo camera's 11.6-px cells reach < 0.2 (slow test)


@pytest.mark.slow
def test_markers_found_at_demo_scale():
    """The done condition: 8/8 at ambient 0.02 in single noisy frames, centre error < 0.2 px."""
    scenario = load_scenario(REPO / "scenarios" / "aligned_slides.yaml")
    ids, errors = _measure(scenario, [0, 1, 500, 1701, 3599])
    assert ids == [list(range(8))] * 5
    assert errors.max() < 0.2, f"centre errors: max {errors.max():.3f} px"


@pytest.mark.slow
def test_markers_are_not_found_in_a_dark_room():
    """At ambient 0.0003 the room-lit paper gives ~4 electrons: no marker in a single frame."""
    scenario = load_scenario(REPO / "scenarios" / "aligned_slides.yaml", quality="fast")
    data = copy.deepcopy(scenario.data)
    data["screen"]["ambient"] = 0.0003
    dark = scenario_from_dict(data)
    camera = dark.scene.camera
    paper_e = dark.scene.markers.paper * 0.0003 * camera.electrons_per_unit_radiance
    found = detect(to_8bit(camera.decode(FrameSource(dark).frame(0)), paper_e))
    assert found == {}
