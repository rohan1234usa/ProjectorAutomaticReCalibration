"""Sample-data figures: what the simulated camera records, and the truth behind each frame.

Every picture here is a real simulator frame (CLAUDE.md section 5) at the quality the build asks
for, shrunk for the web. Each comes with its ground truth -- the true offset between the
projectors, where each box lands -- which the detector will never see; the page draws it on
top so a reader can tell projector edges from content edges.

The misalignment series uses the sweeps' pairing: every variant of ``shift_sweep`` shares its
content and its noise, so subtracting the aligned twin's frame leaves only what B's move
changed. Where A shines alone the difference is exactly zero, which the figure reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo import images as im
from demo.manifest import SHOTS, SIZES
from demo.overlays import layers
from demo.renders import Family
from sim import fiducials
from sim.dataset import frame_hash
from sim.planar import apply_h, clip_convex, signed_area
from sim.truth import coarse_pitch_mm

WIDE = 1600  # whole frames on the pages
SMALL = 960  # gallery frames
CROP = 260  # overlap crops, camera px (shown enlarged)


def save(out: Path, name: str, img: np.ndarray, ext: str = "jpg") -> dict[str, Any]:
    name = name.replace(".", "p")
    return {"src": f"img/{name}.{ext}", **im.save(out / "img" / f"{name}.{ext}", img)}


def display(fam: Family, e: np.ndarray, mode: str) -> np.ndarray:
    return im.log(e, fam.white, fam.black) if mode == "log" else im.natural(e, fam.white)


def px_per_mm(h_mm_to_px: np.ndarray, at_mm: np.ndarray) -> float:
    """Camera pixels per screen millimetre along x at a screen point."""
    at = np.asarray(at_mm, dtype=np.float64)
    return float(np.linalg.norm(apply_h(h_mm_to_px, at + [0.5, 0.0]) - apply_h(h_mm_to_px, at - [0.5, 0.0])))


def public(facts: dict[str, Any]) -> dict[str, Any]:
    """The facts a caption shows (numbers rounded, no matrices)."""
    return {"frame": facts["frame"], "t_s": facts["t_s"], "tag": facts["tag"],
            "offset_mm": round(facts["offset_mm"], 4), "aligned": facts["aligned"], "ambient": facts["ambient"],
            "lamp": {k: round(v, 4) for k, v in facts["lamp"].items()}, "camera_bump": facts["camera_bump"],
            "people": facts["people"], "markers_visible": len(facts["markers_visible"])}


def mask(fam: Family, poly_mm: np.ndarray, camera_h: np.ndarray, erode: int = 0) -> np.ndarray:
    """Camera pixels inside a screen polygon (optionally shrunk by `erode` px)."""
    w, h = fam.scene.camera.resolution
    m = np.zeros((h, w), np.uint8)
    if len(poly_mm) >= 3:
        cv2.fillPoly(m, [np.rint(apply_h(camera_h, poly_mm) * 16).astype(np.int32)], 1, cv2.LINE_8, shift=4)
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return m.astype(bool)


def frame(out: Path, fam: Family, name: str, variant: str, i: int, mode: str = "natural", width: int = WIDE,
          log_too: bool = False, e: np.ndarray | None = None) -> dict[str, Any]:
    """One frame: picture(s), its facts, and its true geometry for the overlay."""
    e = fam.electrons(variant, i) if e is None else e
    facts = fam.facts(variant, i)
    fig = {"img": save(out, name, im.shrink(display(fam, e, mode), width)), "facts": public(facts),
           "layers": layers(fam, facts)}
    if log_too:
        fig["img_log"] = save(out, name + "_log", im.shrink(display(fam, e, "log"), width))
    return fig


def installation(fam: Family) -> dict[str, Any]:
    """The installation's numbers, read from the scenario, for the page's technical details."""
    scenario = fam.scenarios[fam.base]
    scene, setup, timing = fam.scene, fam.scene.setup, scenario.timing
    a, b = setup.box_mm("a"), setup.box_mm("b")
    overlap = setup.overlap()
    onsets = [p.schedule.onset for s in fam.scenarios.values() for p in s.perturbations if p.schedule.onset is not None]
    projector = scene.projectors["a"]
    return {
        "reflectance": scene.screen.reflectance, "ambient": scene.screen.ambient, "bezel_mm": scene.screen.bezel.width_mm,
        "markers": len(scene.markers.centres_mm), "marker_mm": scene.markers.size_mm, "dictionary": fiducials.DICTIONARY,
        "projector_px": list(setup.resolution["a"]), "image_mm": round(float(np.hypot(*(a[1] - a[0]))), 1),
        "pitch_mm": round(coarse_pitch_mm(setup), 4), "overlap_mm": round(float(np.ptp(overlap[:, 0])), 1),
        "b_lower_mm": round(float(b[0, 1] - a[0, 1]), 2), "blend": setup.blend_shape,
        "black_ratio": round(1.0 / projector.black_level), "camera_margin": scenario.data["camera"].get("margin"),
        "camera_keystone": scenario.data["camera"].get("keystone"),
        "camera_px_per_mm": round(px_per_mm(scene.camera.h_mm_to_px, overlap.mean(axis=0)), 3),
        "exposure_s": float(timing.exposure), "sample_every_s": float(timing.sample_every),
        "trusted_window_s": float(timing.trusted_window), "onset_s": float(min(onsets)) if onsets else None,
    }


def textured_windows(e: np.ndarray, overlap_px: np.ndarray, count: int = 2) -> list[tuple[int, int, int, int]]:
    """The `count` most textured CROP-sized windows centred on the overlap, along its long axis, apart."""
    lo, hi = overlap_px.min(axis=0), overlap_px.max(axis=0)
    axis = 1 if hi[1] - lo[1] >= hi[0] - lo[0] else 0  # scan down a tall overlap, across a wide one
    mid = int(round(overlap_px[:, 1 - axis].mean()))

    def window(s: int) -> tuple[int, int, int, int]:
        return (mid - CROP // 2, s, mid + CROP // 2, s + CROP) if axis else (s, mid - CROP // 2, s + CROP, mid + CROP // 2)

    lap = np.abs(cv2.Laplacian(np.asarray(e, dtype=np.float32), cv2.CV_32F))
    scored = sorted(((float(im.crop(lap, window(s)).mean()), s)
                     for s in range(int(lo[axis]) + 8, int(hi[axis]) - 8 - CROP, 10)), reverse=True)
    chosen: list[int] = []
    for _, s in scored:
        if all(abs(s - c) >= CROP for c in chosen):
            chosen.append(s)
        if len(chosen) == count:
            break
    return [window(s) for s in sorted(chosen)]


def shift_series(out: Path, fam: Family) -> dict[str, Any]:
    """From invisible to obvious: B moved 0 ... 8 px across, the same slide, the same noise."""
    shots = {s: SHOTS[f"shift_{s}"] for s in SIZES}
    along = SHOTS["shift_along_8"]
    twin_shot = shots[SIZES[0]]
    i = twin_shot.frame
    if twin_shot.moved or any(s.frame != i for s in (*shots.values(), along)):
        raise ValueError("shift_series: every step must show the same frame, the first one aligned")
    twin = fam.electrons(twin_shot.variant, i)
    base = fam.facts(twin_shot.variant, i)
    overlap_px = apply_h(base["camera_h"], clip_convex(base["boxes_mm"]["a"], base["boxes_mm"]["b"]))
    windows = textured_windows(twin, overlap_px)
    scale = 0.25 * fam.white  # the difference that shows at full colour
    dim = (im.natural(twin, fam.white) * 0.35).astype(np.uint8)
    dim = im.shrink(dim, dim.shape[1] // 2)
    b_then = mask(fam, base["boxes_mm"]["b"], base["camera_h"])

    def step(key: str, variant: str) -> dict[str, Any]:
        e = fam.electrons(variant, i)
        facts = fam.facts(variant, i)
        diff = np.abs(e - twin)
        b_either = b_then | mask(fam, facts["boxes_mm"]["b"], facts["camera_h"])
        b_either = cv2.dilate(b_either.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)  # and its blur
        a_only = mask(fam, facts["boxes_mm"]["a"], facts["camera_h"], 3) & ~b_either  # A alone in both frames
        alpha = np.sqrt(np.clip(im.shrink_max(diff, 2) / scale, 0.0, 1.0))
        return {"offset_mm": round(facts["offset_mm"], 4), "offset_px": round(facts["offset_px"], 4),
                "crops": [save(out, f"{key}_crop{n}", im.natural(im.crop(e, w), fam.white), "png")
                          for n, w in enumerate(windows)],
                "diff": save(out, f"{key}_diff", im.tint(dim, alpha, (134, 182, 239))),
                "a_only_max_e": round(float(diff[a_only].max()), 3)}

    steps = [{"size_px": float(s), **step(f"shift_{s}", shot.variant)} for s, shot in shots.items()]
    biggest = steps[-1]
    moved_along = step("shift_along_8", along.variant)
    return {"frame": i, "t_s": base["t_s"], "steps": steps,
            "along": {"size_px": round(moved_along["offset_px"], 2), **moved_along},
            "diff_scale_e": round(scale, 1), "a_only_max_e": max(s["a_only_max_e"] for s in steps),
            "hero": {"size_px": biggest["size_px"], "offset_mm": biggest["offset_mm"], "moved": biggest["crops"][0]}}


def blend_map(out: Path, fam: Family, name: str, step_mm: float = 4.0) -> dict[str, Any]:
    """Who lights each point: A's orange and B's blue mixed by their blend weights (a + b = 1 in the overlap)."""
    setup, (sw, sh) = fam.scene.setup, fam.scene.screen.size_mm
    xs, ys = np.arange(step_mm / 2, sw, step_mm), np.arange(step_mm / 2, sh, step_mm)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1)
    w = setup.blend_at(pts)
    img = np.full((*w["a"].shape, 3), 13.0)
    lit = (w["a"] + w["b"]) > 0
    img[lit] = w["a"][lit, None] * np.float64(im.A) + w["b"][lit, None] * np.float64(im.B)
    img = np.rint(img).astype(np.uint8)
    overlap = setup.overlap()
    return {**save(out, name, img, "png"), "overlap_m2": round(abs(signed_area(overlap)) / 1e6, 3),
            "overlap_vertices": len(overlap)}


def content_item(out: Path, fam: Family, key: str, label: str, intended: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """One kind of content, with the numbers the frame router would look at (CLAUDE.md 4.3).

    The routing shown is the brief's thresholds applied to the overlap: skip when the frame differs
    from the previous one by more than MOTION_LEVEL, dark when its mean, net of the room light, is
    below DARK_LEVEL. Textured versus flat needs per-tile edge density (Phase 3), so content that is
    neither skipped nor dark keeps the kind it was chosen for, or "lit" if it was meant to be dark.
    """
    shot = SHOTS[key]
    i = shot.frame
    e = fam.electrons(shot.variant, i)
    mode = "log" if intended == "dark" or shot.tag in ("black", "dark", "video_dark") else "natural"
    facts = fam.facts(shot.variant, i)
    inside = mask(fam, clip_convex(facts["boxes_mm"]["a"], facts["boxes_mm"]["b"]), facts["camera_h"], 4)
    room = fam.room(shot.variant, i)
    fig = {"label": label, "intended": intended, "mode": mode,
           "img": save(out, f"content_{key}", im.shrink(display(fam, e, mode), SMALL)),
           "overlap_mean": round(float(e[inside].mean() / fam.white), 5),
           "overlap_mean_net": round(float((e - room)[inside].mean() / fam.white), 5)}
    if intended == "skip" or shot.tag == "video_dark":
        before = fam.electrons(shot.variant, i - 1)
        fig["motion"] = round(float(np.abs(e - before).mean() / fam.white), 5)
    if fig.get("motion", 0.0) > cfg["MOTION_LEVEL"]:
        fig["routing"] = "skip"
    elif fig["overlap_mean_net"] < cfg["DARK_LEVEL"]:
        fig["routing"] = "dark"
    else:
        fig["routing"] = intended if intended in ("textured", "flat") else "lit"
    return fig


def inputs(fam: Family) -> dict[str, Any]:
    """What the detector may read (setup.json) next to what it never sees (one metadata line).

    The line is the one ``make_dataset`` writes for the 8 px variant, so the family must render that
    variant itself: FrameSource.truth describes its own scenario only.
    """
    shot = SHOTS["shift_8"]
    if fam.base != shot.variant:
        raise ValueError(f"inputs: the family renders {fam.base}, the metadata line needs {shot.variant}")
    line = fam.source.truth(shot.frame)
    line["frame_sha256"], line["png"] = frame_hash(fam.frame(fam.base, shot.frame)), None
    return {"setup": fam.source.setup_dict(), "line": line, "variant": fam.base, "frame": shot.frame}
