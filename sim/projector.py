"""A projector: a grid of tiny light sources whose light lands on the screen through a lens.

Physics, in the order light goes through it:

1. Code value to light. Each pixel receives a gamma-encoded code value v in [0, 1], like any
   video signal. The projector turns it into light with a power law, light ~ v**gamma
   (gamma ~ 2.2), scaled by its brightness and per-colour balance.
2. Black level. Even at v = 0 a real projector leaks light; a contrast of 1500:1 means black
   is 1/1500 of white. The leak covers the whole raster whatever the picture, so blending
   cannot remove it. It is added after the blend weight.
3. Where the light lands. The lens maps pixel coordinates to screen millimetres with a
   homography. Each pixel's light spreads over a small footprint. We model pixel aperture
   plus lens blur by bilinear interpolation between neighbouring pixels, a "tent" one pixel
   wide on each side. That puts the raster's 50% edge exactly on the box edge and represents
   sub-pixel shifts smoothly.
4. Light is conserved. A pixel emits a fixed amount of light. If the projector is zoomed or
   tilted so the pixel's footprint grows, the same light spreads over more area and the
   irradiance drops. Irradiance at calibration is the reference (``brightness``). After a
   perturbation it is scaled by |det J_cal| / |det J_actual|, the ratio of footprint areas,
   which is exact for a homography.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.planar import box_mm, jacobian_det, translation, warp_linear
from sim.screen import ScreenGrid


@dataclass(frozen=True)
class Projector:
    name: str
    resolution: tuple[int, int]  # (width, height) in pixels
    gamma: float = 2.2
    brightness: float = 1.0  # full-white irradiance on the screen at calibration
    black_level: float = 1.0 / 1500.0  # light emitted for code 0, as a fraction of full white
    color_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if min(self.resolution) < 1 or self.gamma <= 0 or self.brightness <= 0:
            raise ValueError(f"projector {self.name}: bad resolution, gamma or brightness")
        if not 0 <= self.black_level < 1 or len(self.color_balance) != 3 or min(self.color_balance) <= 0:
            raise ValueError(f"projector {self.name}: need 0 <= black_level < 1 and 3 positive color gains")

    def emitted_light(self, framebuffer: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Irradiance each pixel puts on the screen at the calibrated geometry, (h, w, 3) float32.

        `framebuffer` holds the code values the calibration software sends, (h, w, 3) in [0, 1].
        `blend` holds the per-pixel blend weights in linear light, (h, w).
        """
        w, h = self.resolution
        if framebuffer.shape != (h, w, 3) or blend.shape != (h, w):
            raise ValueError(f"projector {self.name}: framebuffer/blend do not match resolution {w}x{h}")
        light = np.power(np.clip(framebuffer, 0.0, 1.0), np.float32(self.gamma), dtype=np.float32)
        light *= blend[..., None]
        light *= np.asarray(self.color_balance, dtype=np.float32) * np.float32((1.0 - self.black_level) * self.brightness)
        light += np.float32(self.black_level * self.brightness)
        return light


def area_ratio(h_cal: np.ndarray, h_actual: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Per-pixel footprint-area ratio |det J_cal| / |det J_actual|, (h, w) float32."""
    w, h = resolution
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    return (np.abs(jacobian_det(h_cal, u, v)) / np.abs(jacobian_det(h_actual, u, v))).astype(np.float32)


def project(light: np.ndarray, h_cal: np.ndarray, h_actual: np.ndarray, grid: ScreenGrid, out: np.ndarray) -> None:
    """Add one projector's irradiance onto the screen grid `out` (rows, cols, 3), in place.

    `h_actual` is where the projector's pixels really land now. In Phase 1 it equals `h_cal`;
    a drift changes only `h_actual`, because the framebuffer was built for `h_cal`.
    """
    h, w = light.shape[:2]
    if not np.array_equal(h_actual, h_cal):
        light = light * area_ratio(h_cal, h_actual, (w, h))[..., None]
    r0, r1, c0, c1 = grid.window(box_mm(h_actual, (w, h)))
    if r1 <= r0 or c1 <= c0:
        return
    m = translation(-c0, -r0) @ grid.mm_to_grid @ h_actual
    out[r0:r1, c0:c1] += warp_linear(light, m, (c1 - c0, r1 - r0))


def side_by_side(
    screen_size_mm: tuple[float, float],
    resolution_a: tuple[int, int],
    resolution_b: tuple[int, int],
    width_a_mm: float,
    width_b_mm: float,
    overlap_mm: float,
    vertical_offset_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Arrangement preset: A on the left, B on the right, overlapping by `overlap_mm`.

    Pixels are square, so each image's height follows from its width and resolution. B sits
    `vertical_offset_mm` lower than A. The pair is centred on the screen. Returns both boxes
    (TL, TR, BR, BL corners in mm) and the content rect (x0, y0, x1, y1): the full combined
    width over the height both projectors cover.
    """
    if not 0 < overlap_mm < min(width_a_mm, width_b_mm):
        raise ValueError("overlap must be positive and smaller than both image widths")
    w_screen, h_screen = screen_size_mm
    ha = width_a_mm * resolution_a[1] / resolution_a[0]
    hb = width_b_mm * resolution_b[1] / resolution_b[0]
    total_w = width_a_mm + width_b_mm - overlap_mm
    xa = (w_screen - total_w) / 2
    xb = xa + width_a_mm - overlap_mm
    top = min(0.0, vertical_offset_mm)
    bottom = max(ha, vertical_offset_mm + hb)
    ya = (h_screen - (bottom - top)) / 2 - top
    yb = ya + vertical_offset_mm

    def corners(x: float, y: float, w: float, h: float) -> np.ndarray:
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)

    content = (xa, max(ya, yb), xa + total_w, min(ya + ha, yb + hb))
    return corners(xa, ya, width_a_mm, ha), corners(xb, yb, width_b_mm, hb), content
