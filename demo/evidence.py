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
- The hotspot: the blend makes a(x) + b(x) = 1 across the overlap. Move B and its ramp moves
  with it, so the overlap's brightness changes by b(D_B^-1(x)) - b(x), where D_B is where B's
  calibrated picture now lands: a dip or a bump the shape of the ramp's slope. Room light and
  black level dilute it by L / (L + room + 2 black).
- Offset borders: where the picture's own edge crosses the overlap, A and B each draw a copy of
  it; when they part, the one edge becomes two half-height steps.
- Reference mode reads the frame sent to the projectors, which the camera sees a moment later.

The profiles run across a vertical overlap with A on the left (side by side): the functions that
draw them check that and refuse any other arrangement, rather than slice the wrong regions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from demo import images as im
from demo.figures import WIDE, px_per_mm, save
from demo.manifest import SHOTS
from demo.overlays import layers
from demo.renders import Family
from scripts.visualize import border_mm, deepest_point, region_masks
from sim.planar import apply_h, centroid, clip_convex, is_vertical, warp_linear
from sim.state import frame_state

MARGIN_MM = 50.0  # profiles keep this far inside the boxes' top and bottom edges


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


def side_by_side(fam: Family, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Boxes A and B, after checking that the overlap is vertical with A on the left."""
    a, b = fam.scene.setup.box_mm("a"), fam.scene.setup.box_mm("b")
    if not (is_vertical(fam.scene.setup.overlap()) and centroid(a)[0] < centroid(b)[0]):
        raise ValueError(f"{what}: draws profiles across a vertical overlap with A on the left; "
                         f"{fam.stem} {fam.base} is not arranged that way")
    return a, b


def _rows(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    """Screen rows that cross both boxes, MARGIN_MM inside their top and bottom edges."""
    top, bottom = max(a[:, 1].min(), b[:, 1].min()), min(a[:, 1].max(), b[:, 1].max())
    return np.arange(top + MARGIN_MM, bottom - MARGIN_MM + 1e-9, step)


def rectified(out: Path, fam: Family, variant: str, i: int, per_mm: float = 1.0) -> dict[str, Any]:
    """The camera frame resampled onto a grid of `per_mm` pixels per millimetre of screen and bezel."""
    e = fam.electrons(variant, i)
    facts = fam.facts(variant, i)
    x0, y0, x1, y1 = fam.scene.screen.extent_mm
    w, h = int(round((x1 - x0) * per_mm)), int(round((y1 - y0) * per_mm))
    to_mm = np.array([[1 / per_mm, 0.0, x0 + 0.5 / per_mm], [0.0, 1 / per_mm, y0 + 0.5 / per_mm],
                      [0.0, 0.0, 1.0]])  # canvas pixel centre (i, j) -> screen mm
    canvas = warp_linear(np.ascontiguousarray(e, dtype=np.float32), facts["camera_h"] @ to_mm, (w, h), inverse=True)
    lay = layers(fam, {**facts, "camera_h": np.linalg.inv(to_mm)}, labels=False, size=(w, h))
    return {"img": save(out, "rectified", im.shrink(im.natural(canvas, fam.white), WIDE)), "layers": lay}


def dark_raster(out: Path, fam: Family, twin: str) -> dict[str, Any]:
    """Black frames: one alone and the mean of several, room light removed, stretched to show the black level."""
    a, b = side_by_side(fam, "dark_raster")
    shot, one = SHOTS["dark_before"], SHOTS["black"].frame
    room = fam.room(twin, shot.frame)
    single = fam.electrons(twin, one) - room
    mean = fam.mean_electrons(twin, shot.frames) - room
    lo, hi = -6.0, 24.0
    cam = fam.facts(twin, one)["camera_h"]
    xs, ys = np.arange(0.0, fam.scene.screen.size_mm[0] + 0.1, 2.0), _rows(a, b, 4.0)
    prof = {name: im.sample_mm(img, cam, xs, ys).mean(axis=0) for name, img in (("single", single), ("mean", mean))}
    pts, masks = region_masks(fam.scene)
    centre = tuple(np.asarray(fam.scene.screen.size_mm) / 2)
    scale = px_per_mm(cam, np.array(centre))
    levels: dict[str, float | None] = {}
    patches: dict[str, tuple[slice, slice]] = {}
    for label, region in (("unlit", "unlit"), ("A only", "only_a"), ("overlap", "overlap"), ("B only", "only_b")):
        found = deepest_point(pts, masks[region], centre)
        if found is None:
            levels[label] = None
            continue
        u, v = np.rint(apply_h(cam, found[0])).astype(int)
        half = int(min(100, max(1, 0.5 * found[1] * scale)))  # a square inside the region's inscribed circle
        patches[label] = (slice(v - half, v + half + 1), slice(u - half, u + half + 1))
        levels[label] = round(float(np.median(mean[patches[label]])), 2)
    if "A only" not in patches:
        raise ValueError("dark_raster: no region lit by A alone to measure the noise in")
    patch = patches["A only"]
    return {"single": save(out, "dark_single", im.stretch(im.shrink(single, WIDE), lo, hi)),
            "mean": save(out, "dark_mean", im.stretch(im.shrink(mean, WIDE), lo, hi)),
            "count": len(shot.frames), "first": shot.frames[0], "last": shot.frames[-1], "stretch_e": [lo, hi],
            "profile": {"x_mm": _r(xs, 1), "single": _r(prof["single"], 2), "mean": _r(prof["mean"], 2)},
            "levels": levels,
            "noise_e": {"single": round(float(single[patch].std()), 2), "mean": round(float(mean[patch].std()), 2)},
            "room_e": round(float(np.median(room[patch])), 1), "black_e": round(fam.white * fam.black, 2)}


def black_frame(out: Path, fam: Family, name: str, variant: str, i: int) -> dict[str, Any]:
    """One black frame with the room light removed, stretched like the mean of the dark frames."""
    net = fam.electrons(variant, i) - fam.room(variant, i)
    return save(out, name, im.stretch(im.shrink(net, WIDE), -6.0, 24.0))


def edge_profiles(images: dict[str, np.ndarray], cam: np.ndarray, edges: list[tuple[str, float]], ys: np.ndarray,
                  half: float = 30.0, step: float = 0.5) -> list[dict[str, Any]]:
    """Profiles across vertical edges at x (averaged over the rows ys), and each one's crossing."""
    out = []
    for name, x in edges:
        xs = np.arange(x - half, x + half + 1e-9, step)
        prof = {k: im.sample_mm(img, cam, xs, ys).mean(axis=0) for k, img in images.items()}
        pos = {k: crossing(xs, p) for k, p in prof.items()}
        out.append({"name": name, "x_mm": x, "xs": _r(xs, 2), **{k: _r(p, 2) for k, p in prof.items()},
                    "crossing": {k: None if v is None else round(v, 3) for k, v in pos.items()}})
    return out


def boundary(fam: Family, twin: str) -> dict[str, Any]:
    """B's edges move with B and A's stay put: in black frames (black level) and on a slide (picture edge)."""
    a, b = side_by_side(fam, "boundary")
    moved = SHOTS["dark_after_8"]
    room = fam.room(twin, moved.frame)
    dark = {"aligned": fam.mean_electrons(twin, moved.frames) - room,
            "moved": fam.mean_electrons(moved.variant, moved.frames) - room}
    cam = fam.facts(twin, moved.frame)["camera_h"]
    ys = _rows(a, b, 2.0)
    raster = edge_profiles(dark, cam, [("A outer edge", float(a[:, 0].min())), ("A inner edge", float(a[:, 0].max())),
                                       ("B inner edge", float(b[:, 0].min())), ("B outer edge", float(b[:, 0].max()))], ys)
    i = SHOTS["shift_8"].frame
    border = border_mm(fam.scenarios[twin], i)
    x0, _, x1, _ = fam.scene.setup.content_rect_mm
    bright = {"aligned": fam.electrons(twin, i) / np.float32(fam.white),
              "moved": fam.electrons(moved.variant, i) / np.float32(fam.white)}
    picture = edge_profiles(bright, cam, [("A picture edge", x0 + border), ("B picture edge", x1 - border)], ys, 20.0, 0.25)
    facts = fam.facts(moved.variant, moved.frame)
    c = centroid(fam.scene.setup.overlap())
    shift = apply_h(facts["displacement"]["b"], c) - c  # how far B's picture moved there
    return {"raster": raster, "picture": picture, "true_shift_mm": _r(shift, 4), "count": len(moved.frames),
            "size_px": round(facts["offset_px"], 2)}


def _flat_light(fam: Family, variant: str, i: int) -> tuple[float, float]:
    """(content light L of the flat grey shown in frame i, the two projectors' black levels together).

    Both come from the projector's own light model, ``Projector.emitted_light``: its first pixel
    row at full blend weight, its second at zero, which leaves the black level alone.
    """
    scenario = fam.scenarios[variant]
    segments = frame_state(scenario, i).segments
    levels = np.unique(scenario.sequence.image(segments[0][0]))
    if len(segments) != 1 or len(levels[levels > 0]) != 1:
        raise ValueError(f"hotspot: frame {i} of {variant} does not show one flat grey (inside a black border)")
    lights, blacks = {}, 0.0
    for name, p in fam.scene.projectors.items():
        w, h = p.resolution
        blend = np.zeros((h, w), np.float32)
        blend[0] = 1.0
        light = p.emitted_light(np.full((h, w, 3), float(levels.max()), np.float32), blend)
        lights[name], blacks = float(light[0, 0] - light[1, 0]), blacks + float(light[1, 0])
    return lights["b"], blacks  # B is the projector that moves


def hotspot(out: Path, fam: Family) -> dict[str, Any]:
    """Flat grey, B moved 2 px across vs the aligned twin: the overlap's brightness, measured and modelled."""
    side_by_side(fam, "hotspot")
    i = SHOTS["hotspot_0"].frame
    v0, v1 = SHOTS["hotspot_0"].variant, SHOTS["hotspot_2"].variant
    clean = {v: fam.expected(v, i) for v in (v0, v1)}
    noisy = {v: fam.electrons(v, i) for v in (v0, v1)}
    f0, f1 = fam.facts(v0, i), fam.facts(v1, i)
    cam = f0["camera_h"]
    ratio = clean[v1] / np.maximum(clean[v0], np.float32(1.0)) - np.float32(1.0)
    ratio_noisy = noisy[v1] / np.maximum(noisy[v0], np.float32(1.0)) - np.float32(1.0)
    overlap = clip_convex(f0["boxes_mm"]["a"], f0["boxes_mm"]["b"])
    xs, ys = np.arange(overlap[:, 0].min() - 150, overlap[:, 0].max() + 150.1, 1.0), np.arange(300.0, 1200.1, 3.0)
    measured = im.sample_mm(ratio, cam, xs, ys).mean(axis=0)
    measured_noisy = im.sample_mm(ratio_noisy, cam, xs, ys).mean(axis=0)
    setup = fam.scene.setup
    pts = np.stack(np.meshgrid(xs, ys), axis=-1)
    weights = {n: setup.blend_at(pts)[n] for n in setup.names}  # at calibration (the twin is aligned)
    moved = {n: setup.blend_at(apply_h(np.linalg.inv(f1["displacement"][n]), pts))[n] for n in setup.names}
    light, blacks = _flat_light(fam, v0, i)
    factor = light / (light + f0["ambient"] + blacks)
    model = (sum(moved[n] - weights[n] for n in setup.names) * factor).mean(axis=0)
    mid = fam.scene.screen.size_mm[1] / 2  # the ratio does not change down the overlap: a band will do
    corners = np.array([[xs[0], mid - 160], [xs[-1], mid - 160], [xs[-1], mid + 160], [xs[0], mid + 160]])
    span = apply_h(cam, corners)
    window = (int(span[:, 0].min()), int(span[:, 1].min()), int(span[:, 0].max()) + 1, int(span[:, 1].max()) + 1)
    vmax = 0.01
    extreme = float(measured[np.argmax(np.abs(measured))])
    x_lo, x_hi = float(overlap[:, 0].min()), float(overlap[:, 0].max())
    inside = (xs > x_lo + 5) & (xs < x_hi - 5)  # away from B's moved raster edge, which the ramp model leaves out

    def rms(profile: np.ndarray) -> float:
        return float(np.sqrt(np.mean((profile - model)[inside] ** 2)))
    return {"map": save(out, "hotspot_ratio", im.diverging(im.crop(ratio, window), vmax), "png"), "vmax": vmax,
            "profile": {"x_mm": _r(xs, 1), "measured": _r(measured, 6), "noisy": _r(measured_noisy, 6),
                        "model": _r(model, 6)},
            "change": "dip" if extreme < 0 else "rise", "change_pct": round(100 * abs(extreme), 4),
            "per_px_pct": round(100 * abs(extreme) / f1["offset_px"], 4), "size_px": round(f1["offset_px"], 2),
            "model_rms": rms(measured), "noisy_rms": rms(measured_noisy), "factor": round(factor, 4),
            "overlap_x_mm": [x_lo, x_hi], "offset_mm": round(f1["offset_mm"], 4)}


def offset_border(out: Path, fam: Family, twin: str) -> dict[str, Any]:
    """The picture's bottom edge inside the overlap: one step when aligned, two half-steps when B moved down."""
    side_by_side(fam, "offset_border")
    shots = [SHOTS["shift_along_4"], SHOTS["shift_along_8"]]
    i = shots[0].frame
    if any(s.frame != i for s in shots):
        raise ValueError("offset_border: the along shots must show the same frame")
    y_edge = fam.scene.setup.content_rect_mm[3] - border_mm(fam.scenarios[twin], i)
    facts = fam.facts(twin, i)
    overlap = clip_convex(facts["boxes_mm"]["a"], facts["boxes_mm"]["b"])
    xc = float(overlap[:, 0].mean())
    ys, xs = np.arange(y_edge - 20, y_edge + 20.01, 0.25), np.arange(xc - 120, xc + 120.1, 2.0)
    cam = facts["camera_h"]
    u, v = apply_h(cam, [xc, y_edge])
    window = (int(u) - 70, int(v) - 40, int(u) + 70, int(v) + 40)
    profiles = []
    for variant in (twin, *(s.variant for s in shots)):
        e = fam.electrons(variant, i)
        size = fam.facts(variant, i)["offset_px"]
        label = "aligned" if size == 0 else f"{size:.3g} px along"
        values = im.sample_mm(e / np.float32(fam.white), cam, xs, ys).mean(axis=1)
        crop = im.natural(im.crop(e, window), fam.white)
        profiles.append({"label": label, "values": _r(values, 5),
                         "crop": save(out, f"border_{label.replace(' ', '_')}", crop, "png")})
    return {"y_mm": _r(ys, 2), "y_edge_mm": round(y_edge, 2), "profiles": profiles}


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
