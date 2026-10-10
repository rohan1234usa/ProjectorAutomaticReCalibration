"""Double contours, found the way CLAUDE.md 4.3 plans to find them: a teaching illustration.

When B has moved, every content feature in the overlap shows twice: O = a·S(x) + b·S(x - d). In
the Fourier domain that is S's spectrum times (a + b e^(-iωd)), a ripple of period 1/d, and the
log spectrum turns the product into a sum: log|O| = log|S| + log|a + b e^(-iωd)|. Transforming
the log spectrum back (the cepstrum) puts the ripple at one place, quefrency d: a peak at the
offset, whatever the content.

Text has its own repetitions -- line spacing, letter pitch -- which also make peaks. So each
core tile (where both blend weights are at least CORE_MIN_WEIGHT) is compared with control tiles
just outside the overlap, where A or B shines alone and the same kind of content shows no echo:

  diff = z(core) - (z(ctrl_a) + z(ctrl_b)) / 2,   z = robust z-score (median, 1.4826 MAD)

A peak counts if it stands MIN_PEAK_SNR above the noise and NULL_FACTOR above the null, the
largest |z(ctrl_a) - z(ctrl_b)|. A disk of ECHO_FLOOR_CAMERA_PX around the origin is blanked:
closer echoes merge with the origin peak (the floor). Tiles are cut on the camera grid.

Two simplifications make this an illustration, not ``detector/echo.py`` (Phase 6a). It knows from
the simulator which frames hold slides and pools the log spectra of all core tiles (and of all
control tiles) over the sixteen slides shown after B moved, instead of scoring each tile and
asking the tiles to agree on one smooth field. And its "unresolved" rule is its own: a peak that
passes both tests but is only the rising flank of something higher inside the floor is the rim of an
echo too close to the origin to measure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo import images as im
from demo.figures import px_per_mm, save
from demo.manifest import ECHO_SIZES, SHOTS
from demo.renders import Family
from scripts.visualize import border_mm
from sim.planar import apply_h, boundary_distance, centroid, points_in_convex, rect_polygon

WINDOW = 12  # cepstrum half-width searched and shown, camera px
WALK_MM = 600  # how far a control tile is looked for along the blend gradient


def tile_size(fam: Family, tile_mm: float) -> tuple[int, float]:
    """Tile side in camera px (even) for TILE_MM at the overlap centre, and the camera px per mm there."""
    scale = px_per_mm(fam.scene.camera.h_mm_to_px, fam.scene.setup.overlap().mean(axis=0))
    return int(2 * round(tile_mm * scale / 2)), scale


def tiles(fam: Family, n: int, cfg: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
    """Top-left corners (camera px) of core tiles and of each one's control tiles beside the overlap.

    A tile is tested at its four corners and its centre, many candidate positions at once.
    """
    setup, h = fam.scene.setup, fam.scene.camera.h_mm_to_px
    inv = np.linalg.inv(h)
    x0, y0, x1, y1 = setup.content_rect_mm
    shot = SHOTS["echo_0"]
    m = border_mm(fam.scenarios[shot.variant], shot.frame) + 8.0
    picture = rect_polygon(x0 + m, y0 + m, x1 - m, y1 - m)
    corners = np.array([[0, 0], [n, 0], [n, n], [0, n], [n / 2, n / 2]], dtype=np.float64)

    def mm(top_left: np.ndarray) -> np.ndarray:
        """Screen mm of the five test points of tiles with these top-left corners: (..., 5, 2)."""
        return apply_h(inv, np.asarray(top_left, dtype=np.float64)[..., None, :] + corners)

    def core(top_left: np.ndarray) -> np.ndarray:
        pts = mm(top_left)
        w = setup.blend_at(pts)
        ok = points_in_convex(picture, pts) & (np.minimum(w["a"], w["b"]) >= cfg["CORE_MIN_WEIGHT"])
        return ok.all(axis=-1)

    ov = apply_h(h, setup.overlap())
    u_lo, u_hi, v_lo, v_hi = int(ov[:, 0].min()), int(ov[:, 0].max()), int(ov[:, 1].min()), int(ov[:, 1].max())
    us = np.arange(u_lo, u_hi - n)
    grid = np.stack(np.meshgrid(us[::4], np.arange(v_lo, v_hi)), axis=-1)  # (rows, columns, 2)
    rows = np.nonzero(core(grid).any(axis=1))[0]
    if not len(rows):
        raise ValueError(f"no {n} px tile fits where both blend weights reach {cfg['CORE_MIN_WEIGHT']}")
    others = {"ctrl_a": setup.box_mm("b"), "ctrl_b": setup.box_mm("a")}  # a control lies outside the other box
    steps = np.arange(0.0, WALK_MM, 1.0)
    out: dict[str, list[tuple[int, int]]] = {"core": [], "ctrl_a": [], "ctrl_b": []}
    for v in range(v_lo + int(rows[0]), v_hi - n, n):
        fits = core(np.stack([us, np.full_like(us, v)], axis=-1))
        if not fits.any():
            continue
        first = int(np.argmax(fits))  # the keystone tilts the core's sides: each row starts where it fits
        for k in range(first, len(us), n):
            if not fits[k]:
                break
            out["core"].append((int(us[k]), v))
            centre = mm(np.array([us[k], v]))[4]
            for name, other in others.items():
                own = name[-1]
                g = setup.blend_at(centre[None] + [[1.0, 0.0], [0.0, 1.0]])[own] - setup.blend_at(centre[None])[own]
                g = g / np.hypot(*g)
                starts = apply_h(h, centre + steps[:, None] * g) - n / 2  # walk along the gradient of its own weight
                pts = mm(starts)
                clear = (~points_in_convex(other, pts)).all(axis=-1) & (
                    boundary_distance(other, pts).min(axis=-1) >= cfg["CTRL_MARGIN_MM"])
                if clear.any():
                    j = int(np.argmax(clear))
                    if points_in_convex(picture, pts[j]).all():
                        out[name].append((int(round(starts[j, 0])), int(round(starts[j, 1]))))
    return out


def log_spectrum(tile: np.ndarray) -> np.ndarray:
    """Linear high-pass (the tile minus its blur), Hann window, log power."""
    t = tile.astype(np.float64)
    hp = t - cv2.GaussianBlur(t, (0, 0), 3.0, borderType=cv2.BORDER_REFLECT)
    w = np.outer(np.hanning(len(t)), np.hanning(len(t)))
    p = np.abs(np.fft.fft2(hp * w)) ** 2
    return np.log(p + 1e-6 * p.mean())


def zscore(c: np.ndarray) -> np.ndarray:
    med = np.median(c)
    return (c - med) / (1.4826 * np.median(np.abs(c - med)))


def _refine(win: np.ndarray, y: int, x: int) -> np.ndarray:
    """Sub-pixel peak position (parabola through each axis' three samples), relative to the window centre."""
    def vertex(a: float, b: float, c: float) -> float:
        d = a - 2 * b + c
        return 0.0 if d >= 0 else 0.5 * (a - c) / d

    dx = vertex(win[y, x - 1], win[y, x], win[y, x + 1]) if 0 < x < win.shape[1] - 1 else 0.0
    dy = vertex(win[y - 1, x], win[y, x], win[y + 1, x]) if 0 < y < win.shape[0] - 1 else 0.0
    return np.array([x - WINDOW + dx, y - WINDOW + dy])


def score(pools: dict[str, np.ndarray], floor_px: float, cfg: dict[str, Any]) -> dict[str, Any]:
    """Core-minus-control cepstrum near the origin, its best peak outside the floor, the null, the verdict.

    resolved: the peak passes both tests and is a peak of its own, outside the floor.
    unresolved: it passes, but a neighbour nearer the origin is higher: it is the rim of an echo
    inside the floor, too close to the origin to measure (present, size unknown).
    none: nothing passes; the echo, if any, is below the floor (the detector's None, upper bound = floor).
    """
    cep = {k: zscore(np.fft.fftshift(np.abs(np.fft.ifft2(v)))) for k, v in pools.items()}
    diff = cep["core"] - (cep["ctrl_a"] + cep["ctrl_b"]) / 2
    null_map = np.abs(cep["ctrl_a"] - cep["ctrl_b"])
    c = len(diff) // 2
    win = diff[c - WINDOW : c + WINDOW + 1, c - WINDOW : c + WINDOW + 1]
    yy, xx = np.mgrid[-WINDOW : WINDOW + 1, -WINDOW : WINDOW + 1]
    outside = np.hypot(xx, yy) > floor_px
    k = int(np.argmax(np.where(outside, win, -np.inf)))
    y, x = np.unravel_index(k, win.shape)
    peak = float(win[y, x])
    null = float(null_map[c - WINDOW : c + WINDOW + 1, c - WINDOW : c + WINDOW + 1][outside].max())
    at = _refine(win, y, x)
    if at[0] < 0 or (at[0] == 0 and at[1] < 0):
        at = -at  # the cepstrum is symmetric: the echo's sign is not known
    accepted = peak >= cfg["MIN_PEAK_SNR"] and peak >= cfg["NULL_FACTOR"] * null
    r0 = np.hypot(x - WINDOW, y - WINDOW)
    rim = any(win[y + dy, x + dx] > peak and np.hypot(x + dx - WINDOW, y + dy - WINDOW) < r0
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx or dy) and 0 <= y + dy < len(win) and 0 <= x + dx < len(win))
    radius = float(np.hypot(*at))
    verdict = ("unresolved" if rim else "resolved") if accepted else "none"
    return {"peak": round(peak, 2), "at": [round(float(at[0]), 2), round(float(at[1]), 2)],
            "radius_px": round(radius, 2), "null": round(null, 2),
            "threshold": round(max(cfg["MIN_PEAK_SNR"], cfg["NULL_FACTOR"] * null), 2), "accepted": accepted,
            "verdict": verdict, "window": win}


def illustrate(out: Path, fam: Family, cfg: dict[str, Any]) -> dict[str, Any]:
    """Pool every core and control tile over the slides after the onset, for each shift size; score each."""
    n, scale = tile_size(fam, cfg["TILE_MM"])
    floor_px = max(cfg["ECHO_FLOOR_CAMERA_PX"], cfg["MIN_OFFSET_MM"] * scale)
    grid = tiles(fam, n, cfg)
    shots = {size: SHOTS[f"echo_{size}"] for size in ECHO_SIZES}
    frames = shots[ECHO_SIZES[0]].frames
    if any(s.frames != frames for s in shots.values()):
        raise ValueError("cepstrum: every shift size must pool the same frames")
    pools = {size: {k: np.zeros((n, n)) for k in grid} for size in ECHO_SIZES}
    for i in frames:  # frames outside, sizes inside: A's image and the picture are rendered once per frame
        for size, shot in shots.items():
            e = fam.electrons(shot.variant, i)
            for k, corners in grid.items():
                for u, v in corners:
                    pools[size][k] += log_spectrum(e[v : v + n, u : u + n])
    results = []
    c = centroid(fam.scene.setup.overlap())
    for size, shot in shots.items():
        res = score({k: pools[size][k] / (len(grid[k]) * len(frames)) for k in grid}, floor_px, cfg)
        facts = fam.facts(shot.variant, frames[0])
        moved = {p: apply_h(facts["camera_h"], apply_h(d, c)) for p, d in facts["displacement"].items()}
        echo = moved["b"] - moved["a"]  # where B's copy of the content sits relative to A's, camera px
        results.append({"size_px": float(size), "offset_mm": round(facts["offset_mm"], 4),
                        "echo_px": [round(float(echo[0]), 2), round(float(echo[1]), 2)],
                        "estimate_mm": round(res["radius_px"] / scale, 3) if res["verdict"] == "resolved" else None,
                        **{k: res[k] for k in ("peak", "at", "null", "threshold", "verdict", "window")}})
    yy, xx = np.mgrid[-WINDOW : WINDOW + 1, -WINDOW : WINDOW + 1]
    blank = np.hypot(xx, yy) <= floor_px  # the method ignores the floor; so does the picture's colour scale
    vmax = max(float(r["window"][~blank].max()) for r in results)
    for r in results:
        shown = np.sqrt(np.clip(np.where(blank, 0.0, r.pop("window")), 0.0, None) / vmax)
        r["img"] = save(out, f"cepstrum_{r['size_px']:g}", im.ramp(shown, 1.0), "png")
    return {"tile_px": n, "px_per_mm": round(scale, 4), "floor_px": round(floor_px, 3),
            "floor_mm": round(floor_px / scale, 3), "window": WINDOW, "tiles": {k: len(v) for k, v in grid.items()},
            "frames": len(frames), "results": results, "row": _row(fam)}


def _row(fam: Family, length: int = 256) -> dict[str, Any]:
    """One camera row through the overlap (aligned), for the 1-D toy: the one with the most fine detail (text)."""
    shot = SHOTS["echo_0"]
    e = fam.electrons(shot.variant, shot.frame) / np.float32(fam.white)
    ov = apply_h(fam.scene.camera.h_mm_to_px, fam.scene.setup.overlap())
    u0 = int(ov[:, 0].mean()) - length // 2
    rows = range(int(ov[:, 1].min()) + 40, int(ov[:, 1].max()) - 40)
    v = max(rows, key=lambda r: float(np.var(np.diff(e[r, u0 : u0 + length]))))  # many sharp strokes, not one big edge
    return {"values": [round(float(x), 4) for x in e[v, u0 : u0 + length]]}
