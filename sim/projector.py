"""A projector: a grid of tiny light sources whose light lands on the screen through a lens.

Physics, in the order light goes through it:

1. Code value to light. Each pixel receives a gamma-encoded code value v in [0, 1], like any
   video signal. The projector turns it into light with a power law, light ~ v**gamma
   (gamma ~ 2.2), scaled by its brightness and per-colour balance.
2. Mono. The camera records luminance only (CLAUDE.md decision 7), and luminance is a fixed
   linear mix of the three primaries' light (Rec. 709: 0.2126 R + 0.7152 G + 0.0722 B). Every
   later step is linear in light, so the mix is taken right here, after the power law (which is
   not linear) and before anything else. That carries one channel instead of three through the
   rest of the chain. RGB stays available.
3. Black level. Even at v = 0 a real projector leaks light; a contrast of 1500:1 means black
   is 1/1500 of white. The leak covers the whole raster whatever the picture, so blending
   cannot remove it. It is added after the blend weight.
4. Where the light lands. The lens maps pixel coordinates to screen millimetres with a
   homography. Each pixel's light spreads over a small footprint. We model pixel aperture
   plus lens blur by bilinear interpolation between neighbouring pixels, a "tent" one pixel
   wide on each side. That puts the raster's 50% edge exactly on the box edge and represents
   sub-pixel shifts smoothly.
5. Light is conserved. A pixel emits a fixed amount of light. If the projector is zoomed or
   tilted so the pixel's footprint grows, the same light spreads over more area and the
   irradiance drops. Irradiance at calibration is the reference (``brightness``). After a
   perturbation it is scaled by |det J_cal| / |det J_actual|, the ratio of footprint areas,
   which is exact for a homography.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.planar import box_mm, jacobian_det, raster_corners, translation, warp_linear
from sim.screen import ScreenGrid

REC709 = (0.2126, 0.7152, 0.0722)  # luminance weights of the R, G, B primaries' linear light


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

    def emitted_light(self, framebuffer: np.ndarray, blend: np.ndarray, mono: bool = True) -> np.ndarray:
        """Irradiance each pixel puts on the screen at the calibrated geometry, float32.

        `framebuffer` holds the code values the calibration software sends, (h, w, 3) in [0, 1].
        `blend` holds the per-pixel blend weights in linear light, (h, w). The result is (h, w)
        luminance when `mono`, else (h, w, 3).
        """
        w, h = self.resolution
        if framebuffer.shape != (h, w, 3) or blend.shape != (h, w):
            raise ValueError(f"projector {self.name}: framebuffer/blend do not match resolution {w}x{h}")
        light = np.power(np.clip(framebuffer, 0.0, 1.0), np.float32(self.gamma), dtype=np.float32)
        channel_gain = np.asarray(self.color_balance) * (1.0 - self.black_level) * self.brightness
        if mono:
            light = light @ (np.asarray(REC709) * channel_gain).astype(np.float32)
            light *= blend
        else:
            light *= blend[..., None]
            light *= channel_gain.astype(np.float32)
        light += np.float32(self.black_level * self.brightness)
        return light


def area_ratio(h_cal: np.ndarray, h_actual: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Per-pixel footprint-area ratio |det J_cal| / |det J_actual|, (h, w) float32."""
    w, h = resolution
    u = np.arange(w, dtype=np.float64)[None, :]
    v = np.arange(h, dtype=np.float64)[:, None]
    return (np.abs(jacobian_det(h_cal, u, v)) / np.abs(jacobian_det(h_actual, u, v))).astype(np.float32)


def project(light: np.ndarray, h_cal: np.ndarray, h_actual: np.ndarray, grid: ScreenGrid, out: np.ndarray) -> None:
    """Add one projector's irradiance onto the screen grid `out` ((rows, cols) or (rows, cols, 3)), in place.

    `h_actual` is where the projector's pixels really land now. When aligned it equals `h_cal`;
    a drift changes only `h_actual`, because the framebuffer was built for `h_cal`.
    """
    h, w = light.shape[:2]
    if not np.array_equal(h_actual, h_cal):
        ratio = area_ratio(h_cal, h_actual, (w, h))
        light = light * (ratio if light.ndim == 2 else ratio[..., None])
    # Bilinear reconstruction spreads light half a projector pixel past the box edge, so the
    # window must reach that far for this projector's largest pixel (|det J| peaks at a corner).
    u, v = raster_corners((w, h)).T
    max_pitch_mm = float(np.sqrt(np.abs(jacobian_det(h_actual, u, v)).max()))
    pad = math.ceil(0.5 * max_pitch_mm * grid.px_per_mm) + 2
    r0, r1, c0, c1 = grid.window(box_mm(h_actual, (w, h)), pad_px=pad)
    if r1 <= r0 or c1 <= c0:
        return
    m = translation(-c0, -r0) @ grid.mm_to_grid @ h_actual
    out[r0:r1, c0:c1] += warp_linear(light, m, (c1 - c0, r1 - r0))
