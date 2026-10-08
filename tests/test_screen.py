"""Surfaces: the reflectance map mixes materials by exact area, including where screen meets wall."""

import numpy as np
import pytest

from sim.screen import Bezel, Screen, ScreenGrid, surfaces


def test_a_pixel_straddling_screen_and_wall_mixes_them_by_area():
    """No bezel: the grid's last row is 40% screen, 60% wall."""
    screen = Screen((600.0, 260.0), reflectance=0.9, wall_reflectance=0.33)
    grid = ScreenGrid.covering_extent(screen.extent_mm, 1.44)
    refl, lamp = surfaces(grid, screen)
    c = grid.coverage(0.0, 260.0, 1)[-1]
    assert 0 < c < 1
    assert refl[-1, 0] == pytest.approx(0.9 * c + 0.33 * (1 - c), rel=1e-6)
    assert refl[: grid.shape[0] - 1, : grid.shape[1] - 1].min() == pytest.approx(0.9)
    assert np.abs(lamp).max() < 1e-6  # no bezel, so its lamp lights nothing


def test_bezel_screen_and_wall_area_weights():
    screen = Screen((400.0, 200.0), reflectance=0.9, wall_reflectance=0.3, bezel=Bezel(width_mm=20.37, reflectance=0.05))
    grid = ScreenGrid.covering_extent(screen.extent_mm, 0.731)
    refl, lamp = surfaces(grid, screen)
    area_px = 1.0 / grid.px_per_mm**2  # mm^2 per grid pixel
    w, h = screen.size_mm
    bezel_area = (w + 2 * 20.37) * (h + 2 * 20.37) - w * h
    grid_area = grid.shape[0] * grid.shape[1] / grid.px_per_mm**2
    expected = 0.9 * w * h + 0.05 * bezel_area + 0.3 * (grid_area - w * h - bezel_area)
    assert float(refl.astype(np.float64).sum()) * area_px == pytest.approx(expected, rel=1e-6)
    assert float(lamp.astype(np.float64).sum()) * area_px == pytest.approx(bezel_area, rel=1e-6)
