"""Camera: framing geometry, exposure and vignetting, noise statistics, pixel integration, encoding."""

import numpy as np
import pytest

from sim.camera import DN_MAX, Camera
from sim.planar import apply_h
from sim.screen import ScreenGrid

SCREEN = (600.0, 260.0)


def _flat_grid(value: float, px_per_mm: float = 2.0) -> tuple[ScreenGrid, np.ndarray]:
    grid = ScreenGrid.covering(SCREEN, px_per_mm)
    return grid, np.full((*grid.shape, 3), value, dtype=np.float32)


def test_whole_screen_framing():
    margin, keystone = 0.05, 0.02
    cam = Camera.whole_screen(SCREEN, (640, 300), margin=margin, keystone=keystone)
    corners = apply_h(cam.h_mm_to_px, np.array([[0, 0], [SCREEN[0], 0], [SCREEN[0], SCREEN[1]], [0, SCREEN[1]]]))
    top_width, bottom_width = corners[1, 0] - corners[0, 0], corners[2, 0] - corners[3, 0]
    assert top_width == pytest.approx(bottom_width * (1 - keystone))
    # The screen plus a 5% margin on each side exactly fills the limiting dimension of the frame.
    scale = min(640 / (SCREEN[0] * 1.1), 300 / (SCREEN[1] * 1.1))
    assert bottom_width == pytest.approx(SCREEN[0] * scale)
    assert corners[3, 1] - corners[0, 1] == pytest.approx(SCREEN[1] * scale)
    assert corners[:, 0].min() > -0.5 and corners[:, 0].max() < 639.5
    assert corners[:, 1].min() > -0.5 and corners[:, 1].max() < 299.5


@pytest.mark.parametrize("supersample", [1, 2])
def test_flat_field_exposure_and_vignetting(supersample):
    cam = Camera.whole_screen(SCREEN, (641, 301), vignetting=0.15)  # odd size: a pixel sits at the exact centre
    grid, radiance = _flat_grid(0.4)
    e = cam.expected_electrons(radiance, grid, supersample=supersample, border=0.4)[..., 1]
    assert e[150, 320] == pytest.approx(0.4 * cam.well_fill_at_white * cam.full_well_e, rel=1e-4)
    corner = e[0, 0] / e[150, 320]
    assert 0.85 < corner < 0.86  # 15% vignetting at the very corner; pixel 0 sits half a pixel inside


@pytest.mark.parametrize("radiance", [0.5, 0.0003])
def test_noise_has_shot_plus_read_variance(radiance):
    cam = Camera.whole_screen(SCREEN, (640, 480))
    grid, rad = _flat_grid(radiance)
    expected = cam.expected_electrons(rad, grid)
    frame = cam.capture(rad, grid, np.random.default_rng(7))
    sl = (slice(140, 340), slice(220, 420))  # central 200 x 200 pixels, all 3 channels
    resid = (cam.decode(frame) - expected)[sl].ravel().astype(np.float64)
    variance = float(np.mean(expected[sl]) + cam.read_noise_e**2)
    assert abs(resid.mean()) < 5 * np.sqrt(variance / resid.size)
    assert resid.var() == pytest.approx(variance, rel=0.03)


def test_capture_is_deterministic_given_the_seed():
    cam = Camera.whole_screen(SCREEN, (320, 160))
    grid, rad = _flat_grid(0.2)
    a = cam.capture(rad, grid, np.random.default_rng(3))
    b = cam.capture(rad, grid, np.random.default_rng(3))
    c = cam.capture(rad, grid, np.random.default_rng(4))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.parametrize("x_edge", [301.3, 301.55, 301.8, 302.05])
def test_edges_land_where_they_should_at_every_supersampling(x_edge):
    """A vertical step must image at its true camera position for k = 1 and k = 2.

    This catches a half-pixel slip in the sub-pixel mapping (0.25 px at k = 2) and edge
    positions snapping to pixels when the screen grid is finer than the camera (aliasing).
    Edge position = centroid of the profile's derivative, unbiased for symmetric blur.
    """
    cam = Camera.whole_screen(SCREEN, (480, 208), keystone=0.0, vignetting=0.0)
    grid = ScreenGrid.covering(SCREEN, 2.0)  # 2.75 grid pixels per camera pixel
    left_edges = np.arange(grid.shape[1]) / grid.px_per_mm
    covered = np.clip((left_edges + 1 / grid.px_per_mm - x_edge) * grid.px_per_mm, 0.0, 1.0)
    rad = np.repeat((0.1 + 0.5 * covered).astype(np.float32)[None, :, None], grid.shape[0], 0).repeat(3, 2)
    row = 104
    true_u = apply_h(cam.h_mm_to_px, np.array([x_edge, 130.0]))[0]
    for k in (1, 2):
        profile = cam.optical_image(rad, grid, supersample=k, border=0.1)[row, :, 1].astype(np.float64)
        i0 = int(round(true_u))
        g = np.diff(profile[i0 - 8 : i0 + 9])
        centroid = i0 - 8 + 0.5 + np.sum(np.arange(g.size) * g) / g.sum()
        assert abs(centroid - true_u) < 0.02, f"k={k}"


def test_encoding_pedestal_and_saturation():
    cam = Camera.whole_screen(SCREEN, (64, 32), read_noise_e=0.0)
    assert cam.encode(np.zeros((2, 2, 3), np.float32)).max() == cam.pedestal_dn
    assert cam.encode(np.full((2, 2, 3), 1e6, np.float32)).min() == DN_MAX
    e = np.array([[[0.0, 123.4, 15000.0]]], np.float32)
    assert np.allclose(cam.decode(cam.encode(e)), e, atol=0.5 / cam.gain_dn_per_e)
    gamma_cam = Camera.whole_screen(SCREEN, (64, 32), gamma=2.2)
    assert np.allclose(gamma_cam.decode(gamma_cam.encode(e)), e, rtol=1e-3, atol=2.0)
