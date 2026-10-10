"""The screen, its bezel and the wall: surfaces that turn the light falling on them into light the camera sees.

Physics. A matte (Lambertian) surface reflects a fixed fraction of the light landing on it,
equally in all directions. So the radiance the camera records at a point is

    radiance = reflectance(point) * (room light + light from every projector at that point).

Light from the two projectors simply adds, because projectors are incoherent sources. That
sum is why the overlap needs blending: each projector must supply only part of the picture
there. It is also why each projector's *black level* adds up. Black level is the faint light a
projector emits even when showing black. The unlit screen is darker than one projector's
black, and one black is darker than two overlapping blacks. Those small steps are what make
each projector's raster faintly visible in dark content.

Surfaces. The screen (reflectance about 0.9) is framed by a *bezel*, a dark border (about 0.05)
that absorbs projector overspill. Printed fiducial markers sit on the bezel: white paper with
black ink cells (``sim/fiducials.py``). Beyond the bezel is the wall. Everything is lit by the
room light, and the bezel may carry a small lamp of its own (``Bezel.light``) so the markers
stay visible in a dark room. Reflectance belongs to the surface alone, so it is painted once
into a static *reflectance map* on the screen grid, by exact area coverage: a grid pixel
straddling two materials gets the area-weighted mix of their reflectances, which keeps every
material edge at its true sub-pixel position.

Units. Irradiance is measured in multiples of one projector's full-white irradiance, so
``ambient = 0.02`` (a dim lecture hall, the default) means the room light reaching the screen
is 1/50 of a projector's white. ``0.0003`` is a dark room.

The ScreenGrid is the simulator's internal raster in millimetres. It covers the screen and the
bezel; all light is summed there before the camera looks at it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from sim.cfg import check_keys, num, pair


@dataclass(frozen=True)
class Bezel:
    width_mm: float = 0.0  # 0 = no bezel: the wall starts at the screen's edge
    reflectance: float = 0.05  # black velvet frame
    light: float = 0.0  # irradiance of the bezel's own light on the bezel, x one projector's white

    def __post_init__(self) -> None:
        if self.width_mm < 0 or not 0 < self.reflectance <= 1 or self.light < 0:
            raise ValueError("bezel: need width_mm >= 0, 0 < reflectance <= 1, light >= 0")


@dataclass(frozen=True)
class Screen:
    size_mm: tuple[float, float]  # (width, height); coordinates run from (0, 0) at top-left
    reflectance: float = 0.9  # fraction of incident light a matte white screen returns
    ambient: float = 0.02  # room-light irradiance on every surface (dim lecture hall)
    wall_reflectance: float = 0.3  # the wall around the bezel, lit by the room light only
    bezel: Bezel = field(default_factory=Bezel)

    def __post_init__(self) -> None:
        if min(self.size_mm) <= 0:
            raise ValueError(f"screen size must be positive, got {self.size_mm}")
        if not 0 < self.reflectance <= 1 or self.ambient < 0 or not 0 <= self.wall_reflectance <= 1:
            raise ValueError("screen: need 0 < reflectance <= 1, ambient >= 0, 0 <= wall_reflectance <= 1")

    @property
    def extent_mm(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of screen plus bezel: the region the screen grid covers."""
        b = self.bezel.width_mm
        w, h = self.size_mm
        return -b, -b, w + b, h + b


@dataclass(frozen=True)
class ScreenGrid:
    """A raster of the screen: grid pixel (r, c) covers x0 + [c, c+1]/s by y0 + [r, r+1]/s mm."""

    px_per_mm: float
    shape: tuple[int, int]  # (rows, cols)
    origin_mm: tuple[float, float] = (0.0, 0.0)  # (x0, y0): screen mm of the grid's top-left corner

    @classmethod
    def covering(
        cls, size_mm: tuple[float, float], px_per_mm: float, origin_mm: tuple[float, float] = (0.0, 0.0)
    ) -> ScreenGrid:
        w, h = size_mm
        shape = (math.ceil(h * px_per_mm - 1e-9), math.ceil(w * px_per_mm - 1e-9))
        return cls(px_per_mm, shape, (float(origin_mm[0]), float(origin_mm[1])))

    @classmethod
    def covering_extent(cls, extent_mm: tuple[float, float, float, float], px_per_mm: float) -> ScreenGrid:
        x0, y0, x1, y1 = extent_mm
        return cls.covering((x1 - x0, y1 - y0), px_per_mm, (x0, y0))

    @property
    def mm_to_grid(self) -> np.ndarray:
        """Homography from screen mm to grid pixel coordinates (pixel centres at integers)."""
        s = self.px_per_mm
        ox, oy = self.origin_mm
        return np.array([[s, 0.0, -s * ox - 0.5], [0.0, s, -s * oy - 0.5], [0.0, 0.0, 1.0]])

    def to_mm(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Screen mm (x, y) of grid pixel centres."""
        s = self.px_per_mm
        ox, oy = self.origin_mm
        return ox + (np.asarray(cols) + 0.5) / s, oy + (np.asarray(rows) + 0.5) / s

    def centres_mm(self) -> np.ndarray:
        """(rows, cols, 2) array of every grid pixel centre in mm."""
        x, y = self.to_mm(np.arange(self.shape[0])[:, None], np.arange(self.shape[1])[None, :])
        return np.stack(np.broadcast_arrays(x, y), axis=-1)

    def window(self, poly_mm: np.ndarray, pad_px: int = 2) -> tuple[int, int, int, int]:
        """Grid rows/cols (r0, r1, c0, c1) covering a polygon's bounding box, padded, clipped."""
        g = (np.asarray(poly_mm, dtype=np.float64) - np.asarray(self.origin_mm)) * self.px_per_mm - 0.5
        rows, cols = self.shape
        c0 = max(0, math.floor(g[:, 0].min()) - pad_px)
        c1 = min(cols, math.ceil(g[:, 0].max()) + pad_px + 1)
        r0 = max(0, math.floor(g[:, 1].min()) - pad_px)
        r1 = min(rows, math.ceil(g[:, 1].max()) + pad_px + 1)
        return r0, r1, c0, c1

    def coverage(self, lo_mm: float, hi_mm: float, axis: Literal["x", "y"], start: int = 0,
                 stop: int | None = None) -> np.ndarray:
        """Fraction of each grid column (axis "x") or row ("y") in [start, stop) inside [lo, hi] mm."""
        i = {"x": 0, "y": 1}[axis]
        n = self.shape[1 - i]
        stop = n if stop is None else stop
        s = self.px_per_mm
        edge = self.origin_mm[i] + np.arange(start, stop, dtype=np.float64) / s
        return np.clip((np.minimum(edge + 1.0 / s, hi_mm) - np.maximum(edge, lo_mm)) * s, 0.0, 1.0)

    def rect_coverage(self, rect_mm: tuple[float, float, float, float]) -> tuple[slice, slice, np.ndarray]:
        """Exact area fraction (float32) of each grid pixel inside an axis-aligned rect, on its padded window."""
        x0, y0, x1, y1 = rect_mm
        r0, r1, c0, c1 = self.window(np.array([[x0, y0], [x1, y1]]), pad_px=1)
        cy = self.coverage(y0, y1, "y", r0, r1).astype(np.float32)
        cx = self.coverage(x0, x1, "x", c0, c1).astype(np.float32)
        return slice(r0, r1), slice(c0, c1), np.outer(cy, cx)


def surfaces(grid: ScreenGrid, screen: Screen) -> tuple[np.ndarray, np.ndarray]:
    """Reflectance map (screen, bezel, wall) and bezel-light mask on `grid`, both (rows, cols) float32.

    Exact area coverage: refl = rho_wall (1 - c_extent) + rho_bezel (c_extent - c_screen)
    + rho_screen c_screen, with c the area fraction of each grid pixel inside that rectangle.
    The bezel-light mask is c_extent - c_screen: where the bezel's own light shines.
    """
    w, h = screen.size_mm
    refl = np.full(grid.shape, screen.bezel.reflectance, dtype=np.float32)
    light = np.ones(grid.shape, dtype=np.float32)
    rs, cs, cov = grid.rect_coverage((0.0, 0.0, w, h))
    refl[rs, cs] += np.float32(screen.reflectance - screen.bezel.reflectance) * cov
    light[rs, cs] -= cov
    # Beyond the extent is wall. A grid built on the extent has at most its last row and column
    # partly outside, so only rows and columns with coverage e below 1 are touched. So far such a
    # pixel holds screen (refl - bezel is the screen's share, exactly 0 where there is none) and
    # bezel for the rest; the part outside the extent becomes wall:
    #     refl = (refl - bezel) + bezel e + wall (1 - e),    light = (light - 1) + e.
    x0, y0, x1, y1 = screen.extent_mm
    cy, cx = grid.coverage(y0, y1, "y"), grid.coverage(x0, x1, "x")
    wall, bezel, one = np.float32(screen.wall_reflectance), np.float32(screen.bezel.reflectance), np.float32(1.0)
    for r in np.nonzero(cy < 1.0)[0]:
        e = (cy[r] * cx).astype(np.float32)
        refl[r] = (refl[r] - bezel) + (bezel * e + wall * (one - e))
        light[r] = (light[r] - one) + e
    full_rows = np.nonzero(cy >= 1.0)[0]
    for c in np.nonzero(cx < 1.0)[0]:
        e = np.float32(cx[c])
        refl[full_rows, c] = (refl[full_rows, c] - bezel) + (bezel * e + wall * (one - e))
        light[full_rows, c] = (light[full_rows, c] - one) + e
    return refl, light


def from_config(cfg: Mapping[str, Any]) -> Screen:
    """Parse a scenario's ``screen`` block (its ``bezel.markers`` are read by ``sim/fiducials.py``, its
    ``gain`` by ``sim/room.py``)."""
    check_keys(cfg, {"size_mm", "reflectance", "ambient", "wall_reflectance", "bezel", "gain"}, "screen")
    if "size_mm" not in cfg:
        raise ValueError("screen: size_mm is required")
    bezel_cfg = dict(cfg.get("bezel", {}) or {})
    check_keys(bezel_cfg, {"width_mm", "reflectance", "light", "markers"}, "screen.bezel")
    bezel = Bezel(**{k: num(v, f"screen.bezel.{k}") for k, v in bezel_cfg.items() if k != "markers"})
    values = {k: num(v, f"screen.{k}") for k, v in cfg.items() if k not in ("size_mm", "bezel", "gain")}
    return Screen(size_mm=pair(cfg["size_mm"], "screen.size_mm"), bezel=bezel, **values)
