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
from demo.manifest import ALL_NUISANCES, NUISANCE, PRESETS, SHIFT, SHOTS, TWIN, across, validate
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

NUISANCES = {
    "camera_bump": ("Camera bump", "The camera is knocked 3 px and 0.05°. Both boxes and every marker move together; "
                    "the markers re-solve the view and the joint fit's shared camera term absorbs it."),
    "lamp": ("Lamp dimming", "Projector B's lamp fades to 85% over two minutes. The overlap's brightness tilts the "
             "way a dimmer lamp tilts it, not the way a move bends it: a lamp warning, never a YES."),
    "room_light": ("Room light", "The room light steps from 0.02 to 0.05 of projector white. Everything brightens "
                   "together, A and B alike."),
    "occluder": ("Someone walking past", "A person crosses in front of the screen for six seconds and hides a marker. "
                 "Frames that change this much are skipped."),
    "flicker": ("Flicker banding", "Projector A flickers at 100 Hz and the rolling shutter turns it into bands that "
                "move from frame to frame; locking the exposure to the refresh removes it."),
    "sharpening": ("In-camera sharpening", "Sharpening left on draws halos along every edge, a fake second contour. "
                   "It is the same inside and outside the overlap, so the control tiles cancel it."),
    "all": ("All at once", "Bump, lamp, room light, a passer-by, flicker and sharpening together: truth is still "
            "aligned."),
}

Log = Callable[[str], None]


def _shift_family(out: Path, quality: str | None, cfg: dict[str, Any], log: Log) -> tuple[dict, dict, dict]:
    samples: dict[str, Any] = {}
    algorithm: dict[str, Any] = {}
    content: dict[str, Any] = {}
    with Family.load(SHIFT, across("8"), quality) as fam:
        samples["installation"] = figures.frame(out, fam, "installation", TWIN, SHOTS["installation"].frame,
                                                log_too=True)
        log("installation")
        samples["shift"] = figures.shift_series(out, fam)
        log("shift series")
        for key, label, kind in CONTENT:
            if SHOTS[key].scenario == SHIFT:
                content[key] = figures.content_item(out, fam, key, label, kind, cfg)
        samples["inputs"] = figures.inputs(fam)
        algorithm["rectified"] = evidence.rectified(out, fam, TWIN, SHOTS["installation"].frame)
        algorithm["dark"] = evidence.dark_raster(out, fam)
        algorithm["boundary"] = evidence.boundary(fam)
        algorithm["border"] = evidence.offset_border(out, fam)
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
    for (stem, variant), keys in groups.items():
        with Family.load(stem, variant, quality) as fam:
            for key in keys:
                content[key] = figures.content_item(out, fam, key, *labels[key], cfg)
            if (stem, variant) == (SHOTS["flat"].scenario, SHOTS["flat"].variant):
                samples["gain"] = {"matte": gallery.gain(out, fam, "gain_matte", SHOTS["gain_matte"].frame)}
                algorithm["hotspot"] = evidence.hotspot(out, fam)
            if (stem, variant) == (SHOTS["uplift_off"].scenario, SHOTS["uplift_off"].variant):
                algorithm["uplift"] = {"off": evidence.black_frame(out, fam, "uplift_off", variant,
                                                                   SHOTS["uplift_off"].frame)}
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


def _nuisances(out: Path, quality: str | None, log: Log) -> list[dict[str, Any]]:
    keep: dict[int, dict[str, Any]] = {}
    figs: dict[str, Any] = {}
    quiet = SHOTS["occluder_before"]  # nothing else happens in this variant outside its person's windows
    flicker, lamp = SHOTS["flicker_during"], SHOTS["lamp_before"]  # flicker is on from frame 0: before comes from quiet
    with Family.load(NUISANCE, quiet.variant, quality) as fam:
        for i in (*quiet.frames, lamp.frame):
            keep[i] = gallery.capture(fam, quiet.variant, i)
        figs["occluder"] = gallery.pair(out, fam, "occluder", keep[quiet.frame],
                                        gallery.capture(fam, quiet.variant, SHOTS["occluder_during"].frame))
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
            figs[key] = gallery.pair(out, fam, key, b, gallery.capture(fam, variant, during), crop)
            if key == "flicker":  # the bands move from one frame to the next
                figs[key]["next"] = gallery.pair(out, fam, "flicker_next", keep[flicker.frames[1]],
                                                 gallery.capture(fam, variant, flicker.frames[1]))
        log(f"nuisance: {key}")
    order = ("camera_bump", "lamp", "room_light", "occluder", "flicker", "sharpening", "all")
    return [{**figs[k], "label": NUISANCES[k][0], "why": NUISANCES[k][1]} for k in order]


def render(out: Path, quality: str | None, cfg: dict[str, Any], log: Log = print) -> tuple[dict, dict]:
    """Render everything into `out`/img; return (samples data, algorithm data)."""
    problems = validate()
    if problems:
        raise RuntimeError("the sample manifest no longer matches the scenarios:\n  " + "\n  ".join(problems))
    t0 = time.perf_counter()

    def step(what: str) -> None:
        log(f"  {time.perf_counter() - t0:6.1f} s  {what}")

    samples, algorithm, content = _shift_family(out, quality, cfg, step)
    samples["arrangements"] = []
    for preset in PRESETS:
        with Family.load("arrangements", f"arrangement={preset}__magnitude_px=0", quality) as fam:
            samples["arrangements"].append(gallery.arrangement(out, fam, preset, SHOTS[f"arrangement_{preset}"].frame,
                                                               cfg["EDGE_PIECE_MM"]))
        step(f"arrangement: {preset}")
    _content(out, quality, cfg, content, samples, algorithm, step)
    samples["content"] = [content[key] for key, _, _ in CONTENT]
    samples["nuisances"] = _nuisances(out, quality, step)
    samples["timelines"] = {"shift": timelines.shift_steps(), "drift": timelines.drift(cfg["TOLERANCE_MM"]),
                            "rotation": timelines.rotations(), "lanes": timelines.nuisance_lanes()}
    step("timelines")
    algorithm["arrangements"] = {a["preset"]: a["geometry"] for a in samples["arrangements"]}
    return samples, algorithm
