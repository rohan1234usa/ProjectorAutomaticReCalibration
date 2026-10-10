"""Render every figure of the demo, one simulator family at a time, and return the pages' data.

A renderer holds about a gigabyte, so families are opened one after another and released;
frames needed across families (a "before" picture for a nuisance shown in another variant) are
kept as arrays. The shift sweep's renderer serves all thirteen of its variants (see
``demo/renders.py``); its base is the 8 px across variant, whose metadata line the samples page
shows as the example of ground truth.

Before anything is rendered, :func:`demo.manifest.validate` confirms that every chosen frame still
shows what its caption will say.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from demo import cepstrum, evidence, figures, gallery, timelines
from demo.manifest import ALL_NUISANCES, NUISANCE, PRESETS, SHIFT, SHOTS, validate
from demo.renders import Family

# Content kinds, with the kind of frame each was chosen to show (CLAUDE.md 4.3: skip / dark / flat / textured).
CONTENT = [
    ("slides_low", "Text slide, sparse", "textured"),
    ("installation", "Text slide, medium", "textured"),
    ("slides_high", "Text slide, dense", "textured"),
    ("held", "One slide held for half an hour", "textured"),
    ("photo", "Photo-like still", "textured"),
    ("stripes", "Stripes of a known period", "textured"),
    ("letterbox", "Letterboxed photo", "textured"),
    ("blank_overlap", "Nothing textured inside the overlap", "flat"),
    ("flat", "Flat grey", "flat"),
    ("video", "Video, two video frames in one exposure", "skip"),
    ("fast_pan", "Video, fast pan", "skip"),
    ("dark_still", "Dark film, still", "dark"),
    ("dark_video", "Dark film, moving", "dark"),
    ("black", "Black", "dark"),
]

# Why each nuisance cannot fake a YES (what it does, with its numbers, comes from the scenario file).
NUISANCES = {
    "camera_bump": ("Camera bump", "Both boxes and every marker move together: the markers re-solve the view and "
                    "the joint fit's shared camera term absorbs it."),
    "lamp": ("Lamp dimming", "The overlap's brightness tilts the way a dimmer lamp tilts it, not the way a move "
             "bends it: a lamp warning, never a YES."),
    "room_light": ("Room light", "Everything brightens together, A and B alike."),
    "occluder": ("Someone walking past", "Frames that change this much are skipped."),
    "flicker": ("Flicker banding", "The rolling shutter turns the flicker into bands that move from frame to frame; "
                "locking the exposure to the refresh removes them."),
    "sharpening": ("In-camera sharpening", "Sharpening draws halos along every edge, a fake second contour. They are "
                   "the same inside and outside the overlap, so the control tiles cancel them."),
    "all": ("All at once", "Truth is still aligned."),
}

Log = Callable[[str], None]


def _shift_family(out: Path, quality: str | None, cfg: dict[str, Any], log: Log) -> tuple[dict, dict, dict]:
    samples: dict[str, Any] = {}
    algorithm: dict[str, Any] = {}
    content: dict[str, Any] = {}
    twin = SHOTS["installation"].variant
    with Family.load(SHIFT, SHOTS["shift_8"].variant, quality) as fam:
        samples["installation"] = figures.frame(out, fam, "installation", twin, SHOTS["installation"].frame, log_too=True)
        samples["installation"]["about"] = figures.installation(fam)
        algorithm["pitch_mm"] = samples["installation"]["about"]["pitch_mm"]
        log("installation")
        samples["shift"] = figures.shift_series(out, fam)
        log("shift series")
        for key, label, kind in CONTENT:
            if SHOTS[key].scenario == SHIFT:
                content[key] = figures.content_item(out, fam, key, label, kind, cfg)
        samples["inputs"] = figures.inputs(fam)
        samples["dark"] = evidence.dark_raster(out, fam, twin)
        algorithm["rectified"] = evidence.rectified(out, fam, twin, SHOTS["installation"].frame)
        algorithm["boundary"] = evidence.boundary(fam, twin)
        algorithm["border"] = evidence.offset_border(out, fam, twin)
        algorithm["reference"] = evidence.reference(out, fam)
        log("shift-sweep evidence")
        algorithm["echo"] = cepstrum.illustrate(out, fam, cfg)
        log("cepstrum")
    return samples, algorithm, content


def _content(out: Path, quality: str | None, cfg: dict[str, Any], content: dict[str, Any], samples: dict,
             algorithm: dict, log: Log) -> None:
    """Every other content kind, grouped by the renderer that draws it; plus the gain screen and black uplift."""
    groups: dict[tuple[str, str], list[str]] = {}
    for key, _, _ in CONTENT:
        shot = SHOTS[key]
        if shot.scenario != SHIFT:
            groups.setdefault((shot.scenario, shot.variant), []).append(key)
    labels = {key: (label, kind) for key, label, kind in CONTENT}
    flat, uplift_off = SHOTS["flat"], SHOTS["uplift_off"]
    for (stem, variant), keys in groups.items():
        with Family.load(stem, variant, quality) as fam:
            for key in keys:
                content[key] = figures.content_item(out, fam, key, *labels[key], cfg)
            if (stem, variant) == (flat.scenario, flat.variant):  # the matte twin of the gain screen
                samples["gain"] = {"matte": gallery.gain(out, fam, "gain_matte", flat.frame, content["flat"]["img"])}
                algorithm["hotspot"] = evidence.hotspot(out, fam)
            if (stem, variant) == (uplift_off.scenario, uplift_off.variant):
                algorithm["uplift"] = {"off": evidence.black_frame(out, fam, "uplift_off", variant, uplift_off.frame)}
        log(f"content: {stem}")
    shot = SHOTS["uplift_on"]
    with Family.of(shot, quality) as fam:
        algorithm["uplift"]["on"] = evidence.black_frame(out, fam, "uplift_on", shot.variant, shot.frame)
    shot = SHOTS["gain_peak"]
    with Family.of(shot, quality) as fam:
        samples["gain"]["peak"] = gallery.gain(out, fam, "gain_peak", shot.frame)
    with Family.of(SHOTS["zoomed_0"], quality) as fam:
        samples["zoomed"] = gallery.zoomed(out, fam, (SHOTS["zoomed_0"], SHOTS["zoomed_8"]))
    log("gain screen, black uplift, zoomed camera")


def _nuisance(out: Path, fam: Family, key: str, variant: str, before: dict[str, Any], during: int,
              crop: bool = False) -> dict[str, Any]:
    """One nuisance's figure: before and during, what changed, and what the scenario says happened."""
    now = gallery.capture(fam, variant, during)
    fig = gallery.pair(out, fam, key, before, now, crop)
    hidden = len(fam.scene.markers.centres_mm) - now["facts"]["markers_visible"]
    fig["what"] = gallery.caption(fam.scenarios[variant].nuisances, fam.scene.screen.ambient, hidden)
    return fig


def _nuisances(out: Path, quality: str | None, log: Log) -> list[dict[str, Any]]:
    keep: dict[int, dict[str, Any]] = {}
    figs: dict[str, Any] = {}
    quiet = SHOTS["occluder_before"]  # nothing else happens in this variant outside its person's windows
    flicker, lamp = SHOTS["flicker_during"], SHOTS["lamp_before"]  # flicker is on from frame 0: before comes from quiet
    with Family.load(NUISANCE, quiet.variant, quality) as fam:
        for i in (*quiet.frames, lamp.frame):
            keep[i] = gallery.capture(fam, quiet.variant, i)
        figs["occluder"] = _nuisance(out, fam, "occluder", quiet.variant, keep[quiet.frame], SHOTS["occluder_during"].frame)
    log("nuisance: occluder")
    plan = {  # nuisance: (variant, before frame or a kept frame, during frame, crop)
        "camera_bump": (SHOTS["bump_during"].variant, SHOTS["bump_before"].frame, SHOTS["bump_during"].frame, False),
        "lamp": (SHOTS["lamp_during"].variant, keep[lamp.frame], SHOTS["lamp_during"].frame, False),
        "room_light": (SHOTS["room_during"].variant, SHOTS["room_before"].frame, SHOTS["room_during"].frame, False),
        "flicker": (flicker.variant, keep[flicker.frames[0]], flicker.frames[0], False),
        "sharpening": (SHOTS["sharpening_during"].variant, keep[quiet.frame], SHOTS["sharpening_during"].frame, True),
        "all": (ALL_NUISANCES, SHOTS["all_before"].frame, SHOTS["all_during"].frame, False),
    }
    for key, (variant, before, during, crop) in plan.items():
        with Family.load(NUISANCE, variant, quality) as fam:
            b = gallery.capture(fam, variant, before) if isinstance(before, int) else before
            figs[key] = _nuisance(out, fam, key, variant, b, during, crop)
            if key == "flicker":  # the bands move from one frame to the next
                nxt = flicker.frames[1]
                figs[key]["next"] = gallery.change_map(out, fam, "nuisance_flicker_next_diff", keep[nxt]["e"],
                                                       gallery.capture(fam, variant, nxt)["e"])[0]
        log(f"nuisance: {key}")
    return [{**figs[k], "label": NUISANCES[k][0], "why": NUISANCES[k][1]} for k in NUISANCES]


def render(out: Path, quality: str | None, cfg: dict[str, Any], log: Log = print) -> tuple[dict, dict]:
    """Render everything into `out`/img; return (samples data, algorithm data)."""
    problems = validate()
    if problems:
        raise RuntimeError("the sample manifest no longer matches the scenarios:\n  " + "\n  ".join(problems))
    t0 = time.perf_counter()

    def step(what: str) -> None:
        log(f"  {time.perf_counter() - t0:6.1f} s  {what}")

    samples, algorithm, content = _shift_family(out, quality, cfg, step)
    samples["arrangements"], algorithm["arrangements"] = [], []
    for preset in PRESETS:
        shot = SHOTS[f"arrangement_{preset}"]
        with Family.of(shot, quality) as fam:
            fig, shape = gallery.arrangement(out, fam, preset, shot.frame, cfg["EDGE_PIECE_MM"])
        samples["arrangements"].append(fig)
        algorithm["arrangements"].append(shape)
        step(f"arrangement: {preset}")
    _content(out, quality, cfg, content, samples, algorithm, step)
    samples["content"] = [content[key] for key, _, _ in CONTENT]
    samples["nuisances"] = _nuisances(out, quality, step)
    samples["timelines"] = {"shift": timelines.shift_steps(), "drift": timelines.drift(cfg["TOLERANCE_MM"]),
                            "rotation": timelines.rotations(), "lanes": timelines.nuisance_lanes()}
    step("timelines")
    return samples, algorithm
