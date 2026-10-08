"""Ground truth: how far apart the two projectors now put the same piece of content.

At calibration each projector p was measured to land pixel u at H_cal,p(u), and the calibration
software built its framebuffer from that. If the projector has since moved, pixel u really
lands at H_act,p(u). So the content point the software meant to show at screen point x now
appears at

    D_p(x) = H_act,p(H_cal,p^-1(x)),

the projector's *displacement map* (identity while aligned). In the overlap both projectors show
the same content, so a viewer sees two copies of it, |D_B(x) - D_A(x)| apart. The ground-truth
offset is the largest such separation over P, the calibrated overlap inside the content rect:

    offset_mm = max over x in P of |D_B(x) - D_A(x)|.

This treats A and B alike: moving A by some transform gives the same offset as moving B by it.
The brief's relative homography h_rel = H_actB H_calB^-1 H_calA H_actA^-1 = D_B D_A^-1 maps where
A shows a content point to where B shows it, and is recorded too; but taking the maximum of
|h_rel(y) - y| over y in P would sample a slightly different region, D_A(P), and break that
symmetry (by 1e-3 mm for a 0.1% zoom).

For shift, rotation and scale the separation |D_B - D_A| is the length of an affine function, a
convex function, so its maximum over a convex polygon sits at a vertex and the vertices give it
exactly. Projective changes (keystone) are sampled densely along the edges and inside P, then
refined by a short local search.

Offsets are also reported in projector pixels, using one pitch for everything: the coarser
projector's pixel size, sqrt|det J_cal|, at P's centroid.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from sim.calibration import CalibrationSetup
from sim.planar import apply_h, centroid, jacobian_det, points_in_convex

ALIGNED_MM = 0.02  # offsets below this count as aligned (float noise, sub-quantum drift)


def coarse_pitch_mm(setup: CalibrationSetup) -> float:
    """Side of the coarser projector's pixel at the calibrated overlap's centroid."""
    c = centroid(setup.overlap())
    pitches = []
    for name in setup.names:
        u, v = apply_h(np.linalg.inv(setup.h_cal[name]), c)
        pitches.append(float(np.sqrt(abs(jacobian_det(setup.h_cal[name], u, v)))))
    return max(pitches)


def displacement_maps(setup: CalibrationSetup, h_actual: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """D_p = H_act,p H_cal,p^-1 for each projector (screen mm -> screen mm; exactly identity when unmoved)."""
    out = {}
    for n in setup.names:
        h = np.asarray(h_actual.get(n, setup.h_cal[n]))
        out[n] = np.eye(3) if np.array_equal(h, setup.h_cal[n]) else h @ np.linalg.inv(setup.h_cal[n])
    return out


def relative_homography(setup: CalibrationSetup, h_actual: Mapping[str, np.ndarray]) -> np.ndarray:
    a, b = setup.names
    d = displacement_maps(setup, h_actual)
    h = d[b] @ np.linalg.inv(d[a])
    return h / h[2, 2]


def _is_affine(h: np.ndarray) -> bool:
    return abs(h[2, 0]) < 1e-15 and abs(h[2, 1]) < 1e-15


def _samples(poly: np.ndarray, per_edge: int = 128, grid: int = 33) -> np.ndarray:
    pts = [poly]
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        pts.append(p + np.linspace(0.0, 1.0, per_edge)[1:-1, None] * (q - p))
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid))
    inner = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    pts.append(inner[points_in_convex(poly, inner)])
    return np.vstack(pts)


def separation_mm(d_a: np.ndarray, d_b: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """|D_B(x) - D_A(x)| at points x."""
    return np.hypot(*(apply_h(d_b, pts) - apply_h(d_a, pts)).T)


def max_separation(d_a: np.ndarray, d_b: np.ndarray, poly: np.ndarray) -> float:
    """max over the convex polygon `poly` of |D_B(x) - D_A(x)|."""
    if _is_affine(d_a) and _is_affine(d_b):
        return float(separation_mm(d_a, d_b, poly).max())  # convex in x: the maximum is at a vertex
    pts = _samples(poly)
    f = separation_mm(d_a, d_b, pts)
    best = float(f.max())
    # Local refinement from the best samples, along compass directions and along P's edges.
    edge_dirs = [q - p for p, q in zip(poly, np.roll(poly, -1, axis=0))]
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    dirs += [s * e / np.hypot(*e) for e in edge_dirs for s in (1.0, -1.0)]
    for start in pts[np.argsort(f)[-3:]]:
        x, fx = start.copy(), float(separation_mm(d_a, d_b, start[None])[0])
        step = 0.05 * float(np.hypot(*(poly.max(axis=0) - poly.min(axis=0))))
        while step > 1e-7:
            cand = np.array([x + step * d for d in dirs])
            cand = cand[points_in_convex(poly, cand, margin=-1e-12)]
            fc = separation_mm(d_a, d_b, cand) if len(cand) else np.zeros(0)
            if len(fc) and fc.max() > fx:
                x, fx = cand[int(np.argmax(fc))], float(fc.max())
            else:
                step /= 2
        best = max(best, fx)
    return best


def offset_mm(setup: CalibrationSetup, h_actual: Mapping[str, np.ndarray]) -> float:
    """Largest separation of A's and B's copies of the same content over the calibrated overlap."""
    a, b = setup.names
    d = displacement_maps(setup, h_actual)
    return max_separation(d[a], d[b], setup.overlap())
