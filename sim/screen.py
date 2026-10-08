"""The screen: a flat matte surface that turns light falling on it into light the camera sees.

Physics. A matte (Lambertian) screen reflects a fixed fraction of the light landing on it,
equally in all directions. So the radiance the camera records at a point is

    radiance = reflectance * (room light + light from every projector at that point).

Light from the two projectors simply adds, because projectors are incoherent sources. That
sum is why the overlap needs blending: each projector must supply only part of the picture
there. It is also why each projector's *black level* adds up. Black level is the faint light a
projector emits even when showing black. The unlit screen is darker than one projector's
black, and one black is darker than two overlapping blacks. Those small steps are what make
each projector's raster faintly visible in dark content.

Units. Irradiance is measured in multiples of one projector's full-white irradiance, so
``ambient = 0.0003`` means the room light reaching the screen is 1/3300 of a projector's white.

The ScreenGrid is the simulator's internal raster of the screen in millimetres. All light is
summed there before the camera looks at it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Screen:
    size_mm: tuple[float, float]  # (width, height); coordinates run from (0, 0) at top-left
    reflectance: float = 0.9  # fraction of incident light a matte white screen returns
    ambient: float = 0.0003  # room-light irradiance on the screen (dark room)
    surround: float = 0.0001  # radiance of the wall around the screen, as the camera sees it

    def __post_init__(self) -> None:
        if min(self.size_mm) <= 0:
            raise ValueError(f"screen size must be positive, got {self.size_mm}")
        if not 0 < self.reflectance <= 1 or self.ambient < 0 or self.surround < 0:
            raise ValueError("need 0 < reflectance <= 1, ambient >= 0, surround >= 0")

    @property
    def unlit_radiance(self) -> float:
        return self.reflectance * self.ambient

    def radiance(self, irradiance: np.ndarray) -> np.ndarray:
        """Radiance from the projectors' total irradiance; modifies `irradiance` in place."""
        irradiance += np.float32(self.ambient)
        irradiance *= np.float32(self.reflectance)
        return irradiance


@dataclass(frozen=True)
class ScreenGrid:
    """A raster of the screen: grid pixel (r, c) covers [c/s, (c+1)/s] x [r/s, (r+1)/s] mm."""

    px_per_mm: float
    shape: tuple[int, int]  # (rows, cols)

    @classmethod
    def covering(cls, size_mm: tuple[float, float], px_per_mm: float) -> ScreenGrid:
        w, h = size_mm
        return cls(px_per_mm, (math.ceil(h * px_per_mm - 1e-9), math.ceil(w * px_per_mm - 1e-9)))

    @property
    def mm_to_grid(self) -> np.ndarray:
        """Homography from screen mm to grid pixel coordinates (pixel centres at integers)."""
        s = self.px_per_mm
        return np.array([[s, 0.0, -0.5], [0.0, s, -0.5], [0.0, 0.0, 1.0]])

    def to_mm(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen mm (x, y) of grid pixel centres."""
        s = self.px_per_mm
        return (np.asarray(cols) + 0.5) / s, (np.asarray(rows) + 0.5) / s

    def centers_mm(self) -> np.ndarray:
        """(rows, cols, 2) array of every grid pixel centre in mm."""
        x, y = self.to_mm(np.arange(self.shape[0])[:, None], np.arange(self.shape[1])[None, :])
        return np.stack(np.broadcast_arrays(x, y), axis=-1)

    def window(self, poly_mm: np.ndarray, pad_px: int = 2) -> tuple[int, int, int, int]:
        """Grid rows/cols (r0, r1, c0, c1) covering a polygon's bounding box, padded, clipped."""
        g = np.asarray(poly_mm, dtype=np.float64) * self.px_per_mm - 0.5
        rows, cols = self.shape
        c0 = max(0, int(math.floor(g[:, 0].min())) - pad_px)
        c1 = min(cols, int(math.ceil(g[:, 0].max())) + pad_px + 1)
        r0 = max(0, int(math.floor(g[:, 1].min())) - pad_px)
        r1 = min(rows, int(math.ceil(g[:, 1].max())) + pad_px + 1)
        return r0, r1, c0, c1
