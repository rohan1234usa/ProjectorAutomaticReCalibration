"""Which frames the demo shows, and a check that each still shows what its figure says.

A figure's caption makes a claim about its frame: a text slide is up, projector B has already
moved, the lamp is at full dim, a person is walking past. The frames were chosen by reading the
scenario timelines (CLAUDE.md section 7; frame i is exposed at 0.0123 + 0.5 i s, and every
perturbation in the sweeps starts at 615 s, frame 1230). Should a scenario file change, those
claims could silently go stale, so :func:`validate` re-reads every chosen frame's state -- no
rendering, a few milliseconds -- and lists every mismatch. The build stops on any.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sim.scenario import Scenario, load_scenarios
from sim.state import frame_state
from sim.truth import ALIGNED_MM, offset_mm

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"

SHIFT = "shift_sweep"
TWIN = "magnitude_px=0__direction=across"  # the aligned twin of every shift_sweep variant
SIZES = ("0", "0.25", "0.5", "1", "2", "4", "8")  # projector px
ECHO_SIZES = ("0", "1", "2", "4", "8")
# One frame from the middle of each of the 16 distinct slides shown after the onset (615 s).
ECHO_FRAMES = (1262, 1331, 1405, 1479, 1565, 1639, 1713, 1787, 1861, 1947, 2021, 2095, 2169, 2243, 2329, 2383)
PRESETS = ("side_by_side", "stacked", "rotated", "corner", "different_sizes", "large_overlap")
NUISANCE = "aligned_nuisances"
ALL_NUISANCES = "nuisances=camera_bump+lamp+room_light+occluder+flicker+sharpening"


def across(size: str) -> str:
    return f"magnitude_px={size}__direction=across"


def along(size: str) -> str:
    return f"magnitude_px={size}__direction=along"


@dataclass(frozen=True)
class Shot:
    """Frames of one scenario variant, and what must be true of them."""

    scenario: str  # file stem in scenarios/
    variant: str
    frames: tuple[int, ...]
    tag: str | None = None  # content tag of every picture in the exposure
    moved: bool | None = None  # B has fully moved (offset > 0) / the projectors are aligned
    ambient: float | None = None
    lamp_b: float | None = None
    bumped: bool | None = None
    people: bool | None = None
    item: int | None = None  # content item index (tells the two video clips apart)
    straddle: bool | None = None  # the exposure holds two pictures
    distinct: bool = False  # every frame shows a different picture
    tag_prefix: str | None = None  # every picture's tag starts with this
    notes: dict = field(default_factory=dict)

    @property
    def frame(self) -> int:
        return self.frames[0]


SHOTS: dict[str, Shot] = {
    "installation": Shot(SHIFT, TWIN, (1250,), tag="deck_medium", moved=False),
    **{f"shift_{s}": Shot(SHIFT, across(s) if s != "0" else TWIN, (1300,), tag="deck_high", moved=s != "0")
       for s in SIZES},
    "shift_along_4": Shot(SHIFT, along("4"), (1300,), tag="deck_high", moved=True),
    "shift_along_8": Shot(SHIFT, along("8"), (1300,), tag="deck_high", moved=True),
    **{f"arrangement_{p}": Shot("arrangements", f"arrangement={p}__magnitude_px=0", (100,), tag="deck_medium",
                                moved=False) for p in PRESETS},
    "slides_low": Shot(SHIFT, TWIN, (1400,), tag="deck_low", moved=False),
    "slides_high": Shot(SHIFT, TWIN, (1300,), tag="deck_high", moved=False),
    "held": Shot("held_slide", "magnitude_px=0", (100,), tag="held_medium", moved=False),
    "photo": Shot("slow_drift", "slow_drift", (700,), tag="photo", moved=False),
    "video": Shot("aligned_video", "aligned_video", (100,), tag="video_photo", item=0, straddle=True),
    "fast_pan": Shot("aligned_video", "aligned_video", (400,), tag="video_photo", item=1),
    "dark_still": Shot("dark_film", "magnitude_px=0__black_uplift=false", (500,), tag="dark", moved=False),
    "dark_video": Shot("dark_film", "magnitude_px=0__black_uplift=false", (100,), tag="video_dark", moved=False),
    "stripes": Shot("repetition_limit", "magnitude_mm=0", (30,), tag="stripes", moved=False),
    "letterbox": Shot("boundary_hidden", "magnitude_px=0", (20,), tag="photo_letterbox", moved=False),
    "blank_overlap": Shot("blank_band", "magnitude_px=0", (20,), tag="photo_blank", moved=False),
    "flat": Shot("gain_screen", "peak=1__magnitude_px=0", (400,), tag="flat", moved=False),
    "black": Shot(SHIFT, TWIN, (375,), tag="black", moved=False),
    "dark_before": Shot(SHIFT, TWIN, tuple(range(371, 383)), tag="black", moved=False),
    "dark_after_twin": Shot(SHIFT, TWIN, tuple(range(1517, 1529)), tag="black", moved=False),
    "dark_after_8": Shot(SHIFT, across("8"), tuple(range(1517, 1529)), tag="black", moved=True),
    "uplift_off": Shot("dark_film", "magnitude_px=0__black_uplift=false", (610,), tag="black", moved=False),
    "uplift_on": Shot("dark_film", "magnitude_px=0__black_uplift=true", (610,), tag="black", moved=False),
    "bump_before": Shot(NUISANCE, "nuisances=camera_bump", (1828,), bumped=False),
    "bump_during": Shot(NUISANCE, "nuisances=camera_bump", (1832,), bumped=True),
    "lamp_before": Shot(NUISANCE, "nuisances=occluder", (2071,), people=False, lamp_b=1.0),
    "lamp_during": Shot(NUISANCE, "nuisances=lamp", (2071,), lamp_b=0.85),
    "room_before": Shot(NUISANCE, "nuisances=room_light", (2028,), ambient=0.02),
    "room_during": Shot(NUISANCE, "nuisances=room_light", (2032,), ambient=0.05),
    "occluder_before": Shot(NUISANCE, "nuisances=occluder", (1828, 1829), people=False),
    "occluder_during": Shot(NUISANCE, "nuisances=occluder", (1836,), people=True),
    "flicker_during": Shot(NUISANCE, "nuisances=flicker", (1828, 1829)),
    "sharpening_during": Shot(NUISANCE, "nuisances=sharpening", (1828,)),
    "all_before": Shot(NUISANCE, ALL_NUISANCES, (1828,), bumped=False, ambient=0.02, lamp_b=1.0, people=False),
    "all_during": Shot(NUISANCE, ALL_NUISANCES, (2638,), bumped=True, ambient=0.05, lamp_b=0.85, people=True),
    "gain_matte": Shot("gain_screen", "peak=1__magnitude_px=0", (400,), tag="flat", moved=False),
    "gain_peak": Shot("gain_screen", "peak=2.4__magnitude_px=0", (400,), tag="flat", moved=False),
    "zoomed_0": Shot("camera_zoomed", "magnitude_px=0", (1300,), tag="deck_high", moved=False),
    "zoomed_8": Shot("camera_zoomed", "magnitude_px=8", (1300,), tag="deck_high", moved=True),
    "hotspot_0": Shot("gain_screen", "peak=1__magnitude_px=0", (1250,), tag="flat", moved=False),
    "hotspot_2": Shot("gain_screen", "peak=1__magnitude_px=2", (1250,), tag="flat", moved=True),
    **{f"echo_{s}": Shot(SHIFT, across(s) if s != "0" else TWIN, ECHO_FRAMES, moved=s != "0", distinct=True,
                            tag_prefix="deck")
       for s in ECHO_SIZES},
    "reference": Shot(SHIFT, TWIN, (1300,), tag="deck_high", moved=False),
}


def _scenarios(cache: dict[str, dict[str, Scenario]], stem: str) -> dict[str, Scenario]:
    if stem not in cache:
        cache[stem] = {s.variant: s for s in load_scenarios(SCENARIOS / f"{stem}.yaml")}
    return cache[stem]


def problems(key: str, shot: Shot, scenario: Scenario) -> list[str]:
    """What is not as the shot claims, frame by frame."""
    out = []
    pictures = [frame_state(scenario, i).segments for i in shot.frames]
    if shot.distinct and len({seg for seg in pictures}) != len(pictures):
        out.append(f"{key}: {shot.scenario} {shot.variant}: frames {shot.frames} repeat a picture")
    for i in shot.frames:
        st = frame_state(scenario, i)
        where = f"{key}: {shot.scenario} {shot.variant} frame {i}"
        tags = {scenario.sequence.tag(k) for k, _ in st.segments}
        if shot.tag is not None and tags != {shot.tag}:
            out.append(f"{where}: shows {sorted(tags)}, not {shot.tag}")
        if shot.tag_prefix is not None and not all(t.startswith(shot.tag_prefix) for t in tags):
            out.append(f"{where}: shows {sorted(tags)}, not only {shot.tag_prefix}*")
        if shot.item is not None and {k[0] for k, _ in st.segments} != {shot.item}:
            out.append(f"{where}: not content item {shot.item}")
        if shot.straddle is not None and (len(st.segments) > 1) != shot.straddle:
            out.append(f"{where}: exposure holds {len(st.segments)} pictures")
        if shot.moved is not None:
            offset = offset_mm(scenario.scene.setup, st.h_actual)
            full = bool(st.applied) and all(a == 1.0 for a in st.applied)
            if shot.moved and not (full and offset >= ALIGNED_MM):
                out.append(f"{where}: B not fully moved yet (applied {st.applied}, offset {offset:.3f} mm)")
            if not shot.moved and offset >= ALIGNED_MM:
                out.append(f"{where}: projectors misaligned by {offset:.3f} mm, expected aligned")
        if shot.ambient is not None and abs(st.ambient - shot.ambient) > 1e-9:
            out.append(f"{where}: room light {st.ambient}, expected {shot.ambient}")
        if shot.lamp_b is not None and abs(st.gains["b"] - shot.lamp_b) > 1e-9:
            out.append(f"{where}: lamp B at {st.gains['b']}, expected {shot.lamp_b}")
        if shot.bumped is not None and any(st.camera_key) != shot.bumped:
            out.append(f"{where}: camera bump {st.camera_key}")
        if shot.people is not None and bool(st.people) != shot.people:
            out.append(f"{where}: {'nobody' if not st.people else 'someone'} in front of the screen")
    return out


def validate(shots: dict[str, Shot] = SHOTS) -> list[str]:
    """Every mismatch between the manifest and the scenario files (empty when all hold)."""
    cache: dict[str, dict[str, Scenario]] = {}
    out = []
    for key, shot in shots.items():
        variants = _scenarios(cache, shot.scenario)
        if shot.variant not in variants:
            out.append(f"{key}: {shot.scenario} has no variant {shot.variant!r}")
            continue
        out += problems(key, shot, variants[shot.variant])
    return out
