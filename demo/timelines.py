"""Ground truth over time, without rendering a single frame.

A frame's state says where each projector's pixels land and what the room is doing
(``sim/state.py``); the true offset follows from the landing geometry alone (``sim/truth.py``).
So whole timelines -- 2,400 frames of a sweep, the 15,600 of a two-hour drift -- cost well under
a millisecond per frame, and every frame is read: a crossing is found where it happens, not on a
coarser grid. The charts show exactly what each scenario asks the detector to notice (or, for
nuisances, to ignore).

Series are stored as the points where the value changes, which is all a step chart needs.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from demo.manifest import ALL_NUISANCES, NUISANCE, SCENARIOS, SHIFT, SHOTS, SIZES
from sim.scenario import Scenario, load_scenarios
from sim.state import frame_state
from sim.truth import coarse_pitch_mm, offset_mm

LANE_MARGIN_S = 60  # the nuisance lanes start this long before the first nuisance and end after the last


def _variants(stem: str) -> dict[str, Scenario]:
    return {s.variant: s for s in load_scenarios(SCENARIOS / f"{stem}.yaml")}


def offsets(scenario: Scenario) -> list[list[float]]:
    """[t_s, offset_mm] at the first frame and every frame where the offset changes, and the last frame."""
    setup, cache, out = scenario.scene.setup, {}, []
    n = scenario.timing.n_frames
    for i in range(n):
        st = frame_state(scenario, i)
        key = b"".join(st.h_actual[k].tobytes() for k in setup.names)
        if key not in cache:
            cache[key] = offset_mm(setup, st.h_actual)
        value = round(cache[key], 5)
        if not out or out[-1][1] != value or i == n - 1:
            out.append([round(float(st.t), 3), value])
    return out


def _onset(scenarios: list[Scenario]) -> float:
    onsets = [p.schedule.onset for s in scenarios for p in s.perturbations if p.schedule.onset is not None]
    if not onsets:
        raise ValueError(f"{scenarios[0].name}: no perturbation has an onset")
    return float(min(onsets))


def shift_steps() -> dict[str, Any]:
    variants = _variants(SHIFT)
    chosen = {s: variants[SHOTS[f"shift_{s}"].variant] for s in SIZES}
    twin = chosen[SIZES[0]]
    series = [{"name": f"{s} px", "size_px": float(s), "points": offsets(v)} for s, v in chosen.items()]
    return {"series": series, "pitch_mm": round(coarse_pitch_mm(twin.scene.setup), 5),
            "duration_s": float(twin.timing.duration), "trusted_window_s": float(twin.timing.trusted_window),
            "onset_s": _onset(list(chosen.values()))}


def drift(tolerance_mm: float) -> dict[str, Any]:
    scenario = next(iter(_variants("slow_drift").values()))
    points = offsets(scenario)
    crossing = next((t for t, v in points if v > tolerance_mm), None)
    rate = scenario.perturbations[0].schedule.rate_per_h
    return {"points": points, "crossing_s": crossing, "rate_px_per_h": rate,
            "duration_s": float(scenario.timing.duration), "onset_s": _onset([scenario])}


def rotations() -> list[dict[str, Any]]:
    """Each rotation variant at its last frame: the angle it took for each offset, about each pivot."""
    out = []
    for name, scenario in _variants("rotation_sweep").items():
        st = frame_state(scenario, scenario.timing.n_frames - 1)
        tags = [p.tag(a) for p, a in zip(scenario.perturbations, st.applied, strict=True)]
        off = offset_mm(scenario.scene.setup, st.h_actual)
        size, _, pivot = name.partition("__pivot=")
        if off > 0:
            out.append({"size_px": float(size.split("=")[1]), "pivot": pivot, "deg": round(abs(tags[0]["deg"]), 5),
                        "offset_mm": round(off, 4), "offset_px": round(off / coarse_pitch_mm(scenario.scene.setup), 4)})
    if not out:
        raise ValueError("rotation_sweep: no variant has turned B by its last frame")
    return sorted(out, key=lambda r: (r["pivot"], r["size_px"]))


def _frame_at(scenario: Scenario, t: Fraction) -> int:
    """The first frame whose exposure starts at or after t (clamped to the run)."""
    timing = scenario.timing
    i = math.ceil((t - timing.phase) / timing.sample_every)
    return min(max(i, 0), timing.n_frames - 1)


def nuisance_lanes(step: int = 2) -> dict[str, Any]:
    """The all-nuisances variant around its nuisances: each one's state over time, and the true offset (zero)."""
    scenario = _variants(NUISANCE)[ALL_NUISANCES]
    nz = scenario.nuisances
    spans = [(x.schedule.onset, x.schedule.onset + x.schedule.duration) for x in (*nz.bumps, *nz.lamps, *nz.rooms)
             if x.schedule.onset is not None] + [(o.t0, o.t0 + o.duration) for o in nz.occluders]
    if not spans:
        raise ValueError(f"{NUISANCE} {ALL_NUISANCES}: no nuisance with an onset")
    lo = _frame_at(scenario, min(s for s, _ in spans) - LANE_MARGIN_S)
    hi = _frame_at(scenario, max(e for _, e in spans) + LANE_MARGIN_S)
    setup = scenario.scene.setup
    lanes: dict[str, list[list[float]]] = {"offset_mm": [], "lamp_b": [], "room_light": [], "camera_bump": [],
                                           "person": []}
    for i in range(lo, hi + 1, step):
        st = frame_state(scenario, i)
        t = round(float(st.t), 3)
        values = {"offset_mm": round(offset_mm(setup, st.h_actual), 6), "lamp_b": round(st.gains["b"], 4),
                  "room_light": round(st.ambient, 4), "camera_bump": round(max(st.camera_key or (0.0,)), 2),
                  "person": 1.0 if st.people else 0.0}
        for k, v in values.items():
            if not lanes[k] or lanes[k][-1][1] != v:
                lanes[k].append([t, v])
    end = round(float(scenario.timing.time(hi)), 3)
    for points in lanes.values():
        if points[-1][0] != end:
            points.append([end, points[-1][1]])
    return {"lanes": lanes, "t0": round(float(scenario.timing.time(lo)), 3), "t1": end}
