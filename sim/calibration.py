"""What the calibration software knows and sends: the "blending setup".

After calibration the software holds three things:
  * for each projector, the homography it measured from projector pixels to screen mm (H_cal);
  * the rectangle of screen the content should fill (the content rect C);
  * a blend map per projector.

From these it builds each projector's framebuffer. For every projector pixel it looks up the
content at the screen point where that pixel lands (geometric correction), then multiplies by
that pixel's blend weight (photometric blending). Nothing changes until the next calibration.
So when a projector physically drifts, its framebuffer is still made for the old geometry and
the picture lands in the wrong place. That mismatch is exactly what the detector looks for.
Besides camera frames, this information is the one thing the detector may read. The harness
hands it over as plain data, since the detector never imports the simulator.

Blend rule. Where both projectors cover a point, their weights must add up to 1 in linear
light, so the sum is seamless. Each projector must also fade to exactly 0 at each of its edges
that lies inside the other projector's footprint *and* inside the content; otherwise that
edge would show as a step. Edges on the outline of the displayed picture need no fade. So,
with O = A ∩ B ∩ C (a convex polygon) and O's edges tagged by the box they lie on:

    d_A = distance to O's edges that lie only on A's outline   (A's "inner edges")
    d_B = the same for B
    t   = d_A / (d_A + d_B),    w_A = S(t),    w_B = 1 - S(t)

S is a cosine ramp (smooth at both ends) or a linear one. A point covered by one projector
gets weight 1 from it. This is the classic distance-to-edge blend, restricted to the edges
that need it. Measuring distance to *all* of a projector's edges would pinch the ramp near the
overlap's shared top and bottom edges, which real side-by-side blends do not do.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from sim.planar import (
    Segment,
    apply_h,
    box_mm,
    boundary_distance,
    clip_convex,
    distance_to_segments,
    edges,
    jacobian_det,
    points_in_convex,
    raster_corners,
    rect_polygon,
    warp_linear,
)

RAMPS = {
    "linear": lambda t: t,
    "cosine": lambda t: 0.5 - 0.5 * np.cos(np.pi * t),
}
_ON_EDGE_MM = 1e-6  # tolerance for "this overlap edge lies on that outline" (float rounding only)


@dataclass(frozen=True, eq=False)
class CalibrationSetup:
    h_cal: Mapping[str, np.ndarray]  # projector px -> screen mm, as measured at calibration
    resolution: Mapping[str, tuple[int, int]]  # (width, height) per projector
    content_rect_mm: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    blend_shape: str = "cosine"

    def __post_init__(self) -> None:
        if len(self.h_cal) != 2 or set(self.h_cal) != set(self.resolution):
            raise ValueError("a calibration setup needs exactly two projectors with resolutions")
        if self.blend_shape not in RAMPS:
            raise ValueError(f"blend_shape must be one of {sorted(RAMPS)}, got {self.blend_shape!r}")
        x0, y0, x1, y1 = self.content_rect_mm
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"content rect must have positive size, got {self.content_rect_mm}")

    @property
    def names(self) -> tuple[str, str]:
        a, b = self.h_cal
        return a, b

    def box_mm(self, name: str) -> np.ndarray:
        return box_mm(self.h_cal[name], self.resolution[name])

    def pixel_pitch_mm(self, name: str) -> float:
        """Side of the projector's smallest pixel footprint on the screen."""
        u, v = raster_corners(self.resolution[name]).T
        return float(np.sqrt(np.abs(jacobian_det(self.h_cal[name], u, v)).min()))

    def overlap(self) -> np.ndarray:
        """The calibrated overlap inside the content: box A ∩ box B ∩ content rect (convex polygon)."""
        a, b = self.names
        return clip_convex(clip_convex(self.box_mm(a), self.box_mm(b)), rect_polygon(*self.content_rect_mm))

    def inner_edges(self) -> dict[str, list[Segment]]:
        """For each projector, the overlap edges where it must fade to zero."""
        a, b = self.names
        boxes = {a: self.box_mm(a), b: self.box_mm(b)}
        content = rect_polygon(*self.content_rect_mm)
        overlap = self.overlap()
        inner: dict[str, list[Segment]] = {a: [], b: []}
        for p, q in edges(overlap):
            mid = (p + q) / 2
            on = {n: boundary_distance(boxes[n], mid) < _ON_EDGE_MM for n in (a, b)}
            if boundary_distance(content, mid) < _ON_EDGE_MM:
                continue  # on the outline of the displayed picture: nobody fades here
            for n, other in ((a, b), (b, a)):
                if on[n] and not on[other]:
                    inner[n].append((p, q))
        return inner

    def blend_at(self, pts_mm: np.ndarray) -> dict[str, np.ndarray]:
        """Blend weight of each projector at screen points (..., 2), in linear light."""
        a, b = self.names
        pts = np.asarray(pts_mm, dtype=np.float64)
        inside = {n: points_in_convex(self.box_mm(n), pts) for n in (a, b)}
        inner = self.inner_edges()
        da, db = distance_to_segments(pts, inner[a]), distance_to_segments(pts, inner[b])
        t = np.full(da.shape, 0.5)
        fa, fb = np.isfinite(da), np.isfinite(db)
        total = np.where(fa & fb, da + db, 1.0)
        ok = fa & fb & (total > 0)
        t[ok] = da[ok] / total[ok]
        t[~fa & fb] = 1.0  # A has no inner edge (B nested inside A): A keeps everything
        t[fa & ~fb] = 0.0  # B has no inner edge (A nested inside B): B keeps everything
        ramp = RAMPS[self.blend_shape](t)
        both = inside[a] & inside[b]
        return {
            a: np.where(both, ramp, inside[a].astype(np.float64)),
            b: np.where(both, 1.0 - ramp, inside[b].astype(np.float64)),
        }

    def blend_weights(self, name: str) -> np.ndarray:
        """Blend weight of every pixel of projector `name`, (h, w) float32 (the blend map)."""
        w, h = self.resolution[name]
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pts = apply_h(self.h_cal[name], np.stack([u, v], axis=-1))
        return self.blend_at(pts)[name].astype(np.float32)

    def content_size(self) -> tuple[int, int]:
        """Native content resolution: the content rect sampled at the finest projector pitch."""
        pitch = min(self.pixel_pitch_mm(n) for n in self.names)
        x0, y0, x1, y1 = self.content_rect_mm
        return max(1, round((x1 - x0) / pitch)), max(1, round((y1 - y0) / pitch))

    def content_to_mm(self, content_size: tuple[int, int]) -> np.ndarray:
        """Homography from content pixel coordinates to screen mm (content fills the rect)."""
        wc, hc = content_size
        x0, y0, x1, y1 = self.content_rect_mm
        px, py = (x1 - x0) / wc, (y1 - y0) / hc
        return np.array([[px, 0.0, x0 + 0.5 * px], [0.0, py, y0 + 0.5 * py], [0.0, 0.0, 1.0]])

    def to_setup_dict(self, gamma_assumed: Mapping[str, float]) -> dict:
        """What the calibration software can report: geometry, content rect and the blend rule.

        The blend weights follow from these: the rule is "distance to each projector's inner
        edges", shaped by the ramp and applied in linear light assuming the projector gamma.
        """
        return {
            "projectors": {
                n: {
                    "resolution": [int(v) for v in self.resolution[n]],
                    "h_cal_px_to_mm": np.asarray(self.h_cal[n], dtype=np.float64).tolist(),
                    "gamma_assumed": float(gamma_assumed[n]),
                }
                for n in self.names
            },
            "content_rect_mm": [float(v) for v in self.content_rect_mm],
            "blend": {"rule": "inner_edge_distance", "shape": self.blend_shape, "space": "linear", "black_uplift": False},
        }

    def framebuffer(self, name: str, content: np.ndarray) -> np.ndarray:
        """Code values sent to projector `name`: content resampled where each pixel lands at H_cal.

        Pixels landing outside the content rect show black (code 0). The blend weight is applied
        by the projector model in linear light, which is what ideal blending software achieves.
        """
        hc, wc = content.shape[:2]
        projector_to_content = np.linalg.inv(self.content_to_mm((wc, hc))) @ self.h_cal[name]
        src = np.ascontiguousarray(content, dtype=np.float32)
        return warp_linear(src, projector_to_content, self.resolution[name], inverse=True)
