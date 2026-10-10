"""Algorithm figures: the evidence each planned method looks at, measured on simulator frames.

These are teaching illustrations, not the detector (CLAUDE.md section 4; built from Phase 3 on).
Each uses ground truth the detector will not have -- the camera's true view, the aligned twin of
a sweep variant -- to show plainly what the method relies on:

- Rectification maps the camera image onto a millimetre grid of the screen, so both boxes become
  the shapes the blending software placed (here with the true view; the detector solves it from
  the bezel markers).
- Boundary analysis finds each projector's edges. In a black frame only the black level shows,
  about 9 electrons per projector on 270 from the room light, so frames are averaged and the
  room light taken away; then each raster is a faint plateau, and the overlap twice as bright.
  (Pictures are shrunk before they are stretched: clipping noise first would bias the grey.)
  When B moves, B's edges move and A's stay put, in dark frames and in bright ones alike.
- The hotspot: the blend makes a(x) + b(x) = 1 across the overlap. Move B by u and its ramp moves
  with it, so the overlap's brightness changes by b(x - u) - b(x): a dip or a bump the shape of
  the ramp's slope. Room light and black level dilute it by L / (L + room + 2 black).
- Offset borders: where the picture's own edge crosses the overlap, A and B each draw a copy of
  it; when they part, the one edge becomes two half-height steps.
- Reference mode reads the frame sent to the projectors, which the camera sees a moment later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from demo import images as im
from demo.figures import WIDE, save
from demo.manifest import SHOTS, TWIN, along
from demo.renders import Family
from scripts.visualize import border_mm
from sim.planar import apply_h, clip_convex, rect_polygon, warp_linear


def _r(a: np.ndarray, d: int = 4) -> list[float]:
    return [round(float(v), d) for v in np.asarray(a).ravel()]


def crossing(xs: np.ndarray, profile: np.ndarray, n: int = 20) -> float | None:
    """Where a step profile crosses halfway between its two ends (linear interpolation), or None."""
    lo, hi = float(np.mean(profile[:n])), float(np.mean(profile[-n:]))
    half = (lo + hi) / 2
    s = np.sign(profile - half)
    idx = np.nonzero(s[:-1] != s[1:])[0]
    if abs(hi - lo) < 1e-9 or not len(idx):
        return None
    k = idx[np.argmin(np.abs(idx - len(xs) / 2))]  # the crossing nearest the middle of the window
    x0, x1, y0, y1 = xs[k], xs[k + 1], profile[k], profile[k + 1]
    return float(x0 + (half - y0) * (x1 - x0) / (y1 - y0))


def rectified(out: Path, fam: Family, variant: str, i: int, px_per_mm: float = 1.0) -> dict[str, Any]:
    """The camera frame resampled onto a millimetre grid of screen and bezel."""
    e = fam.electrons(variant, i)
    facts = fam.facts(variant, i)
    x0, y0, x1, y1 = fam.scene.screen.extent_mm
    w, h = int(round((x1 - x0) * px_per_mm)), int(round((y1 - y0) * px_per_mm))
    to_mm = np.array([[1 / px_per_mm, 0.0, x0 + 0.5 / px_per_mm], [0.0, 1 / px_per_mm, y0 + 0.5 / px_per_mm],
                      [0.0, 0.0, 1.0]])
    canvas = warp_linear(np.ascontiguousarray(e, dtype=np.float32), facts["camera_h"] @ to_mm, (w, h), inverse=True)

    def px(poly: np.ndarray) -> list[list[float]]:
        return ((np.asarray(poly) - [x0, y0]) * px_per_mm).round(1).tolist()

    setup, markers = fam.scene.setup, fam.scene.markers
    boxes = facts["boxes_mm"]
    return {"img": save(out, "rectified", im.shrink(im.natural(canvas, fam.white), WIDE)), "w": w, "h": h,
            "px_per_mm": px_per_mm, "origin_mm": [x0, y0],
            "layers": {"w": w, "h": h, "a": px(boxes["a"]), "b": px(boxes["b"]),
                       "overlap": px(clip_convex(boxes["a"], boxes["b"])),
                       "content": px(rect_polygon(*setup.content_rect_mm)), "hotspots": [], "labels": [],
                       "markers": [{"id": k, "poly": px(markers.square(k)), "visible": True}
                                   for k in range(len(markers.centres_mm))]}}


def dark_raster(out: Path, fam: Family) -> dict[str, Any]:
    """Black frames: one alone and the mean of twelve, room light removed, stretched to show the black level."""
    shot, one = SHOTS["dark_before"], SHOTS["black"].frame
    room = fam.room(TWIN, shot.frame)
    single = fam.electrons(TWIN, one) - room
    mean = fam.mean_electrons(TWIN, shot.frames) - room
    lo, hi = -6.0, 24.0
    cam = fam.facts(TWIN, one)["camera_h"]
    xs, ys = np.arange(0.0, fam.scene.screen.size_mm[0] + 0.1, 2.0), np.arange(350.0, 1150.1, 4.0)
    prof = {name: im.sample_mm(img, cam, xs, ys).mean(axis=0) for name, img in (("single", single), ("mean", mean))}

    def level(lo_mm: float, hi_mm: float, profile: np.ndarray) -> float:
        return round(float(np.median(profile[(xs >= lo_mm) & (xs <= hi_mm)])), 2)

    a, b = fam.scene.setup.box_mm("a"), fam.scene.setup.box_mm("b")
    regions = {"unlit": (20.0, a[:, 0].min() - 20), "A only": (a[:, 0].min() + 100, b[:, 0].min() - 100),
               "overlap": (b[:, 0].min() + 40, a[:, 0].max() - 40), "B only": (a[:, 0].max() + 100, b[:, 0].max() - 100)}
    u, v = np.rint(apply_h(cam, [sum(regions["A only"]) / 2, fam.scene.screen.size_mm[1] / 2])).astype(int)
    patch = (single[v - 100 : v + 100, u - 100 : u + 100], mean[v - 100 : v + 100, u - 100 : u + 100])
    room_on_screen = float(np.median(room[v - 100 : v + 100, u - 100 : u + 100]))
    return {"single": save(out, "dark_single", im.stretch(im.shrink(single, WIDE), lo, hi)),
            "mean": save(out, "dark_mean", im.stretch(im.shrink(mean, WIDE), lo, hi)),
            "frames": list(shot.frames), "stretch_e": [lo, hi],
            "profile": {"x_mm": _r(xs, 1), "single": _r(prof["single"], 2), "mean": _r(prof["mean"], 2)},
            "levels": {k: level(*v, prof["mean"]) for k, v in regions.items()},
            "noise_e": {"single": round(float(patch[0].std()), 2), "mean": round(float(patch[1].std()), 2)},
            "room_e": round(room_on_screen, 1), "black_e": round(fam.white * fam.black, 2)}


def black_frame(out: Path, fam: Family, name: str, variant: str, i: int) -> dict[str, Any]:
    """One black frame with the room light removed, stretched like the twelve-frame mean above."""
    net = fam.electrons(variant, i) - fam.room(variant, i)
    xs, ys = np.arange(0.0, fam.scene.screen.size_mm[0] + 0.1, 4.0), np.arange(250.0, 1250.1, 4.0)
    prof = im.sample_mm(net, fam.facts(variant, i)["camera_h"], xs, ys).mean(axis=0)
    return {"img": save(out, name, im.stretch(im.shrink(net, WIDE), -6.0, 24.0)),
            "profile": {"x_mm": _r(xs, 1), "net_e": _r(prof, 2)},
            "uplift": bool(fam.scene.setup.black_uplift)}


def edge_profiles(fam: Family, images: dict[str, np.ndarray], cam: np.ndarray, edges: list[tuple[str, float]],
                  half: float = 30.0, step: float = 0.5) -> list[dict[str, Any]]:
    """Profiles across vertical edges at x (averaged over the middle of the screen), and each one's crossing."""
    ys = np.arange(400.0, 1100.1, 2.0)
    out = []
    for name, x in edges:
        xs = np.arange(x - half, x + half + 1e-9, step)
        prof = {k: im.sample_mm(img, cam, xs, ys).mean(axis=0) for k, img in images.items()}
        pos = {k: crossing(xs, p) for k, p in prof.items()}
        out.append({"name": name, "x_mm": x, "xs": _r(xs, 2), **{k: _r(p, 2) for k, p in prof.items()},
                    "crossing": {k: None if v is None else round(v, 3) for k, v in pos.items()}})
    return out


def boundary(fam: Family) -> dict[str, Any]:
    """B's edges move with B and A's stay put: in black frames (black level) and on a slide (picture edge)."""
    moved = SHOTS["dark_after_8"]
    room = fam.room(TWIN, moved.frame)
    dark = {"aligned": fam.mean_electrons(TWIN, moved.frames) - room,
            "moved": fam.mean_electrons(moved.variant, moved.frames) - room}
    a, b = fam.scene.setup.box_mm("a"), fam.scene.setup.box_mm("b")
    cam = fam.facts(TWIN, moved.frame)["camera_h"]
    raster = edge_profiles(fam, dark, cam, [("A outer edge", float(a[:, 0].min())), ("A inner edge", float(a[:, 0].max())),
                                            ("B inner edge", float(b[:, 0].min())), ("B outer edge", float(b[:, 0].max()))])
    i = SHOTS["shift_8"].frame
    border = border_mm(fam.scenarios[TWIN], i)
    x0, _, x1, _ = fam.scene.setup.content_rect_mm
    bright = {"aligned": fam.electrons(TWIN, i) / np.float32(fam.white),
              "moved": fam.electrons(moved.variant, i) / np.float32(fam.white)}
    picture = edge_profiles(fam, bright, cam, [("A picture edge", x0 + border), ("B picture edge", x1 - border)], 20.0, 0.25)
    shift = fam.facts(moved.variant, moved.frame)["boxes_mm"]["b"] - b
    return {"raster": raster, "picture": picture, "true_shift_mm": _r(shift.mean(axis=0), 4),
            "frames": list(moved.frames), "frame": i}


def hotspot(out: Path, fam: Family) -> dict[str, Any]:
    """Flat grey, B moved 2 px across vs the aligned twin: the overlap's brightness, measured and modelled."""
    i = SHOTS["hotspot_0"].frame
    v0, v1 = SHOTS["hotspot_0"].variant, SHOTS["hotspot_2"].variant
    clean = {v: fam.expected(v, i) for v in (v0, v1)}
    noisy = {v: fam.electrons(v, i) for v in (v0, v1)}
    f0, f1 = fam.facts(v0, i), fam.facts(v1, i)
    cam = f0["camera_h"]
    ratio = clean[v1] / np.maximum(clean[v0], 1.0) - 1.0
    ratio_noisy = noisy[v1] / np.maximum(noisy[v0], 1.0) - 1.0
    overlap = clip_convex(f0["boxes_mm"]["a"], f0["boxes_mm"]["b"])
    xs, ys = np.arange(overlap[:, 0].min() - 150, overlap[:, 0].max() + 150.1, 1.0), np.arange(300.0, 1200.1, 3.0)
    measured = im.sample_mm(ratio, cam, xs, ys).mean(axis=0)
    measured_noisy = im.sample_mm(ratio_noisy, cam, xs, ys).mean(axis=0)
    setup, scene = fam.scene.setup, fam.scene
    v = (f1["boxes_mm"]["b"] - f0["boxes_mm"]["b"]).mean(axis=0)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx, gy], axis=-1)
    b_now, b_cal = setup.blend_at(pts - v)["b"], setup.blend_at(pts)["b"]
    flat = next(item for item in fam.scenarios[v0].sequence.items if item.kind == "flat")
    p = scene.projectors["b"]
    light = float(flat.value) ** p.gamma * (1 - p.black_level) * p.brightness
    blacks = sum(q.black_level * q.brightness for q in scene.projectors.values())
    factor = light / (light + f0["ambient"] + blacks)
    model = ((b_now - b_cal) * factor).mean(axis=0)
    mid = fam.scene.screen.size_mm[1] / 2  # the ratio does not change down the overlap: a band will do
    corners = np.array([[xs[0], mid - 160], [xs[-1], mid - 160], [xs[-1], mid + 160], [xs[0], mid + 160]])
    span = apply_h(cam, corners)
    window = (int(span[:, 0].min()), int(span[:, 1].min()), int(span[:, 0].max()) + 1, int(span[:, 1].max()) + 1)
    vmax = 0.01
    dip = float(measured.min())
    x_lo, x_hi = float(overlap[:, 0].min()), float(overlap[:, 0].max())
    inside = (xs > x_lo + 5) & (xs < x_hi - 5)  # away from B's moved raster edge, which the ramp model leaves out

    def rms(profile: np.ndarray) -> float:
        return float(np.sqrt(np.mean((profile - model)[inside] ** 2)))
    return {"map": save(out, "hotspot_ratio", im.diverging(im.crop(ratio, window), vmax), "png"),
            "window": list(window), "vmax": vmax, "frame": i, "shift_mm": _r(v, 4),
            "profile": {"x_mm": _r(xs, 1), "measured": _r(measured, 6), "noisy": _r(measured_noisy, 6),
                        "model": _r(model, 6)},
            "dip_pct": round(100 * dip, 4), "per_px_pct": round(100 * dip / f1["offset_px"], 4),
            "model_rms": rms(measured), "noisy_rms": rms(measured_noisy), "factor": round(factor, 4),
            "light": round(light, 4), "overlap_x_mm": [x_lo, x_hi],
            "offset_mm": round(f1["offset_mm"], 4)}


def offset_border(out: Path, fam: Family) -> dict[str, Any]:
    """The picture's bottom edge inside the overlap: one step when aligned, two half-steps when B moved down."""
    i = SHOTS["shift_along_8"].frame
    y_edge = fam.scene.setup.content_rect_mm[3] - border_mm(fam.scenarios[TWIN], i)
    facts = fam.facts(TWIN, i)
    overlap = clip_convex(facts["boxes_mm"]["a"], facts["boxes_mm"]["b"])
    xc = float(overlap[:, 0].mean())
    ys, xs = np.arange(y_edge - 20, y_edge + 20.01, 0.25), np.arange(xc - 120, xc + 120.1, 2.0)
    cam = facts["camera_h"]
    variants = {"aligned": TWIN, "4 px along": along("4"), "8 px along": along("8")}
    profiles: dict[str, Any] = {}
    crops = {}
    u, v = (cam @ [xc, y_edge, 1.0])[:2] / (cam @ [xc, y_edge, 1.0])[2]
    window = (int(u) - 70, int(v) - 40, int(u) + 70, int(v) + 40)
    for label, variant in variants.items():
        e = fam.electrons(variant, i)
        profiles[label] = _r(im.sample_mm(e / np.float32(fam.white), cam, xs, ys).mean(axis=1), 5)
        crops[label] = save(out, f"border_{label.replace(' ', '_')}", im.crop(im.natural(e, fam.white), window), "png")
    return {"y_mm": _r(ys, 2), "y_edge_mm": round(y_edge, 2), "profiles": profiles, "crops": crops,
            "window": list(window), "frame": i,
            "shift_mm": round(float(fam.facts(along("8"), i)["offset_mm"]), 4)}


def reference(out: Path, fam: Family) -> dict[str, Any]:
    """The picture sent to the projectors next to the camera frame that shows it a moment later."""
    shot = SHOTS["reference"]
    scenario = fam.scenarios[shot.variant]
    sent = np.rint(np.clip(scenario.content_image(shot.frame), 0.0, 1.0) * 255).astype(np.uint8)
    camera = im.natural(fam.electrons(shot.variant, shot.frame), fam.white)
    ring = fam.source.source(shot.frame, ring=4)
    return {"source": save(out, "reference_source", im.shrink(sent, 1200)),
            "camera": save(out, "reference_camera", im.shrink(camera, 1200)),
            "t_s": round(float(scenario.timing.time(shot.frame)), 4), "lag_s": float(scenario.reference["lag_s"]),
            "ring": [{"sent_s": round(t, 4), "tag": scenario.sequence.tag(key)} for t, key, _ in ring]}
