"""Rendered screen light: seamless when aligned, black level visible, light conserved.

All checks use noiseless screen irradiance or radiance. The camera is tested separately.
Tolerances are relative brightness (seams) or millimetres (edge positions), each with its
physical reason.
"""

import numpy as np
import pytest

from sim.content import flat, slide
from sim.planar import homography_from_points, raster_corners, rect_polygon, translation, warp_linear
from sim.projector import REC709, project
from sim.screen import ScreenGrid
from tests.scenes import (
    SCREEN_MM,
    corner_setup,
    grid_masks,
    make_renderer,
    notch_points,
    rotated_setup,
    sample_grid,
    side_by_side_setup,
)


def _overlap_width_px(setup) -> float:
    box_a, box_b = setup.box_mm("a"), setup.box_mm("b")
    return (box_a[:, 0].max() - box_b[:, 0].min()) / setup.finest_pitch_mm("a")


@pytest.mark.parametrize("quality", ["fast", "standard"])
@pytest.mark.parametrize("vertical_offset_mm", [0.0, 0.3])
def test_side_by_side_flat_white_is_seamless(quality, vertical_offset_mm):
    """The only seam left is the bilinear reconstruction error of the cosine ramp.

    Per projector it is at most max|w''|/8 = (pi^2/2)/(8 W^2), with W the overlap width in pixels;
    the two projectors' errors can add. That is 3.7e-4 for this 58-px overlap and 8e-6 at the
    demo scene's 394 px.
    """
    setup = side_by_side_setup(vertical_offset_mm=vertical_offset_mm)
    r = make_renderer(setup, quality=quality, black_level=0.0, ambient=0.0)
    irr = r.screen_irradiance(flat(setup.content_size(), 1.0))
    masks = grid_masks(setup, r.grid, margin_px=3 * r.quality.screen_samples_per_px)
    w = _overlap_width_px(setup)
    assert np.abs(irr[masks["union"]] - 1.0).max() <= 2 * (np.pi**2 / 2) / (8 * w**2)
    assert masks["overlap"].sum() > 1000


def test_linear_ramp_leaves_a_bump_at_the_inner_edges():
    """Why the cosine ramp is the default.

    A linear ramp still has slope 1/W where it meets an inner edge. The projector's pixel blur
    then spills a bump of up to 1/(4W) across the edge, while the overlap interior stays exact.
    """
    setup = side_by_side_setup(shape="linear")
    r = make_renderer(setup, black_level=0.0, ambient=0.0)
    irr = r.screen_irradiance(flat(setup.content_size(), 1.0))
    masks = grid_masks(setup, r.grid, margin_px=3 * r.quality.screen_samples_per_px)
    w = _overlap_width_px(setup)
    worst = np.abs(irr[masks["union"]] - 1.0).max()
    assert np.abs(irr[masks["overlap"]] - 1.0).max() < 1e-6
    assert 2 * (np.pi**2 / 2) / (8 * w**2) < worst <= 1 / (4 * w)


@pytest.mark.parametrize("make", [rotated_setup, corner_setup])
def test_rotated_and_corner_overlaps_are_seamless_away_from_notches(make):
    """Two inner edges per projector means kinks in the distance field.

    The bilinear error at a kink is about a quarter of the jump in the weight's slope: measured
    5.4e-3 here, about 1/W, so about 1e-3 at demo scale. Near a notch (an overlap corner on the
    image outline) the weight must go from 0 to 1 in no distance at all. No blend can avoid the
    speck that leaves, and it stays within 6 pixels of the notch.
    """
    setup = make()
    r = make_renderer(setup, black_level=0.0, ambient=0.0)
    irr = r.screen_irradiance(flat(setup.content_size(), 1.0))
    masks = grid_masks(setup, r.grid, margin_px=3 * r.quality.screen_samples_per_px)
    pts = r.grid.centres_mm()
    notch_px = np.min([np.hypot(*(pts - v).transpose(2, 0, 1)) for v in notch_points(setup)], axis=0)
    notch_px /= setup.finest_pitch_mm("a")
    err = np.abs(irr - 1.0)
    assert err[masks["union"] & (notch_px > 8)].max() < 8e-3
    assert notch_px[masks["union"] & (err > 0.01)].max(initial=0.0) < 6.0
    assert np.abs(irr[masks["only_a"] | masks["only_b"]] - 1.0).max() < 1e-6


def test_overlap_adds_exactly_one_black_level():
    """On flat gray, the overlap is brighter than single coverage by exactly one black level.

    Blending cannot remove black: both projectors emit it over their whole raster.
    """
    setup = side_by_side_setup()
    r = make_renderer(setup)
    rad = r.screen_radiance(flat(setup.content_size(), 0.5))
    box_a, box_b = setup.box_mm("a"), setup.box_mm("b")
    y = box_a[:, 1].mean()
    overlap = sample_grid(r.grid, rad, (box_a[:, 0].max() + box_b[:, 0].min()) / 2, y)
    only_a = sample_grid(r.grid, rad, (box_a[:, 0].min() + box_b[:, 0].min()) / 2, y)
    only_b = sample_grid(r.grid, rad, (box_a[:, 0].max() + box_b[:, 0].max()) / 2, y)
    one_black = r.screen.reflectance * r.projectors["b"].black_level * r.projectors["b"].brightness  # 6.0e-4
    assert np.allclose(overlap - only_a, one_black, atol=1e-5)
    assert np.allclose(only_a, only_b, atol=1e-7)


@pytest.mark.parametrize("quality, edge_tol_mm", [("fast", 0.05), ("standard", 0.02), ("fine", 0.02)])
def test_black_content_shows_each_raster(quality, edge_tol_mm):
    """With black content each raster glows at its black level. Its 50% edge sits on the box edge."""
    setup = side_by_side_setup()
    r = make_renderer(setup, quality=quality)
    rad = r.screen_radiance(flat(setup.content_size(), 0.0))
    s, p = r.screen, r.projectors["a"]
    box_a, box_b = setup.box_mm("a"), setup.box_mm("b")
    y = box_a[:, 1].mean()
    unlit = s.reflectance * s.ambient
    single = s.reflectance * (s.ambient + p.black_level)
    double = s.reflectance * (s.ambient + 2 * p.black_level)
    assert sample_grid(r.grid, rad, box_a[0, 0] / 2, y) == pytest.approx(unlit, rel=1e-5)
    assert sample_grid(r.grid, rad, (box_a[0, 0] + box_b[0, 0]) / 2, y) == pytest.approx(single, rel=1e-5)
    assert sample_grid(r.grid, rad, (box_a[1, 0] + box_b[0, 0]) / 2, y) == pytest.approx(double, rel=1e-5)
    # 50% crossing of the black-level step across A's left edge and B's right edge, along one row.
    row = int(round(y * r.grid.px_per_mm - 0.5))
    profile = rad[row]
    x_mm = (np.arange(profile.size) + 0.5) / r.grid.px_per_mm
    half = (unlit + single) / 2
    for true_x, rising in ((box_a[0, 0], True), (box_b[1, 0], False)):
        near = np.abs(x_mm - true_x) < 3.0
        xs, vs = x_mm[near], profile[near]
        i = np.nonzero((vs[:-1] < half) & (vs[1:] >= half) if rising else (vs[:-1] >= half) & (vs[1:] < half))[0][0]
        crossing = xs[i] + (half - vs[i]) * (xs[i + 1] - xs[i]) / (vs[i + 1] - vs[i])
        assert abs(crossing - true_x) < edge_tol_mm


def test_coarse_projector_edge_light_is_not_clipped():
    """A projector with 4x coarser pixels than the grid's finest still spills half a pixel of light.

    Bilinear reconstruction ramps from full at the last pixel centre to zero half a pixel past the
    box edge; the warp window must cover that whole ramp, however coarse the projector.
    """
    res = (60, 34)  # 250 mm wide -> 4.17 mm per pixel
    box = rect_polygon(100.0, 50.0, 350.0, 191.67)
    h = homography_from_points(raster_corners(res), box)
    grid = ScreenGrid.covering(SCREEN_MM, 2.0 / 1.0417)  # sized for a 1.04 mm/px partner
    out = np.zeros((*grid.shape, 3), np.float32)
    project(np.ones((res[1], res[0], 3), np.float32), h, h, grid, out)
    pitch = 250.0 / res[0]
    x = (np.arange(grid.shape[1]) + 0.5) / grid.px_per_mm
    row = out[int(round(120.0 * grid.px_per_mm - 0.5)), :, 0]
    tail = (x > 350.0) & (x < 350.0 + pitch / 2)
    assert np.allclose(row[tail], 0.5 - (x[tail] - 350.0) / pitch, atol=1e-4)
    assert np.all(row[x > 350.0 + pitch / 2 + 0.01] == 0.0)


def test_zoom_conserves_light():
    """Zooming a projector 1% spreads the same light over 1.01^2 times the area."""
    setup = side_by_side_setup()
    r = make_renderer(setup, black_level=0.0, ambient=0.0)
    box_a = setup.box_mm("a")
    c = box_a.mean(axis=0)
    zoom = translation(*c) @ np.diag([1.01, 1.01, 1.0]) @ translation(*-c)
    irr = r.screen_irradiance(flat(setup.content_size(), 1.0), {"a": zoom @ setup.h_cal["a"]})
    x = (box_a[0, 0] + setup.box_mm("b")[0, 0]) / 2  # middle of A's single-coverage region
    assert sample_grid(r.grid, irr, x, c[1]) == pytest.approx(1 / 1.01**2, rel=1e-4)


def _alone(r, name: str, content: np.ndarray) -> np.ndarray:
    """Radiance if projector `name` showed the content alone at full weight (no blend partner)."""
    light = r.projectors[name].emitted_light(r.setup.framebuffer(name, content), np.ones_like(r.blend[name]))
    out = np.zeros(r.grid.shape, np.float32)
    project(light, r.setup.h_cal[name], r.setup.h_cal[name], r.grid, out)
    return (out + np.float32(r.screen.ambient)) * r.reflectance


def test_aligned_slide_blend_adds_no_error_and_misalignment_does():
    """In the overlap, the blend must be no worse than either projector showing the slide alone.

    Errors are measured against the ideal: the content sampled once, directly on the screen.
    Pointwise |w_A e_A + w_B e_B| <= w_A |e_A| + w_B |e_B|. Shifting B by one pixel must then
    show up as a clearly larger error: the ghost the detector will look for.
    """
    setup = side_by_side_setup()
    r = make_renderer(setup, black_level=0.0, ambient=0.0)
    size = setup.content_size()
    content = slide(size, np.random.default_rng(3))
    grid: ScreenGrid = r.grid
    to_content = np.linalg.inv(setup.content_to_mm(size)) @ np.linalg.inv(grid.mm_to_grid)
    linear = np.power(warp_linear(content, to_content, grid.shape[::-1], inverse=True), 2.2)
    ideal = r.screen.reflectance * (linear @ np.array(REC709, np.float32))  # mono: Rec. 709 luminance
    overlap = grid_masks(setup, grid, margin_px=3 * r.quality.screen_samples_per_px)["overlap"]

    def overlap_error(radiance: np.ndarray) -> float:
        return float(np.abs(radiance - ideal)[overlap].mean())

    blended = overlap_error(r.screen_radiance(content))
    alone = max(overlap_error(_alone(r, n, content)) for n in setup.names)
    shifted = overlap_error(r.screen_radiance(content, {"b": translation(setup.finest_pitch_mm("b"), 0) @ setup.h_cal["b"]}))
    assert blended <= 1.02 * alone
    assert shifted > 2.5 * blended


def test_black_level_uplift_makes_black_uniform_until_a_projector_moves():
    """With compensation, single coverage gets the partner's black too: no brighter overlap on black."""
    from dataclasses import replace

    setup = replace(side_by_side_setup(), black_uplift=True)
    r = make_renderer(setup)
    black = flat(setup.content_size(), 0.0)
    rad = r.screen_radiance(black)
    box_a, box_b = setup.box_mm("a"), setup.box_mm("b")
    y = box_a[:, 1].mean()
    overlap = sample_grid(r.grid, rad, (box_a[:, 0].max() + box_b[:, 0].min()) / 2, y)
    only_a = sample_grid(r.grid, rad, (box_a[:, 0].min() + box_b[:, 0].min()) / 2, y)
    only_b = sample_grid(r.grid, rad, (box_a[:, 0].max() + box_b[:, 0].max()) / 2, y)
    assert only_a == pytest.approx(overlap, rel=1e-5) and only_b == pytest.approx(overlap, rel=1e-5)
    # After B moves 3 mm right, A's lift no longer meets B's edge: a dark gap opens at B's old edge.
    moved = r.screen_radiance(black, {"b": translation(3.0, 0) @ setup.h_cal["b"]})
    gap = sample_grid(r.grid, moved, box_b[:, 0].min() + 1.5, y)
    assert gap < overlap - 0.5 * r.screen.reflectance * r.projectors["b"].black_level
