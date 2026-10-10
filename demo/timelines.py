"""Ground truth over time, without rendering a single frame.

A frame's state says where each projector's pixels land and what the room is doing
(``sim/state.py``); the true offset follows from the landing geometry alone (``sim/truth.py``).
So whole timelines -- 2,400 frames of a sweep, the 15,600 of a two-hour drift -- cost
milliseconds per frame, and the charts can show exactly what each scenario asks the detector to
notice (or, for nuisances, to ignore).

Series are stored as the points where the value changes, which is all a step chart needs.
"""

from __future__ import annotations

from typing import Any

from demo.manifest import ALL_NUISANCES, NUISANCE, SCENARIOS, SHIFT, SIZES, TWIN, across
from sim.scenario import Scenario, load_scenarios
from sim.state import frame_state
from sim.truth import coarse_pitch_mm, offset_mm


def _variants(stem: str) -> dict[str, Scenario]:
    return {s.variant: s for s in load_scenarios(SCENARIOS / f"{stem}.yaml")}


def offsets(scenario: Scenario, step: int = 1) -> list[list[float]]:
    """[t_s, offset_mm] at every `step`-th frame where the offset changes (and the last frame)."""
    setup, cache, out = scenario.scene.setup, {}, []
    n = scenario.timing.n_frames
    for i in [*range(0, n, step), n - 1]:
        st = frame_state(scenario, i)
        key = b"".join(st.h_actual[k].tobytes() for k in setup.names)
        if key not in cache:
            cache[key] = offset_mm(setup, st.h_actual)
        value = round(cache[key], 5)
        if not out or out[-1][1] != value or i == n - 1:
            out.append([round(float(st.t), 3), value])
    return out


def shift_steps() -> dict[str, Any]:
    variants = _variants(SHIFT)
    pitch = coarse_pitch_mm(variants[TWIN].scene.setup)
    series = [{"name": f"{s} px", "size_px": float(s), "points": offsets(variants[across(s) if s != "0" else TWIN], 2)}
              for s in SIZES]
    timing = variants[TWIN].timing
    return {"series": series, "pitch_mm": round(pitch, 5), "duration_s": float(timing.duration),
            "trusted_window_s": float(timing.trusted_window), "onset_s": 615.0}


def drift(tolerance_mm: float) -> dict[str, Any]:
    scenario = next(iter(_variants("slow_drift").values()))
    points = offsets(scenario, 20)
    crossing = next((t for t, v in points if v > tolerance_mm), None)
    return {"points": points, "crossing_s": crossing, "tolerance_mm": tolerance_mm,
            "pitch_mm": round(coarse_pitch_mm(scenario.scene.setup), 5), "duration_s": float(scenario.timing.duration),
            "trusted_window_s": float(scenario.timing.trusted_window)}


def rotations() -> list[dict[str, Any]]:
    """Each rotation variant once B has turned: the angle it took for each offset, about each pivot."""
    out = []
    for name, scenario in _variants("rotation_sweep").items():
        st = frame_state(scenario, 1300)
        tags = [p.tag(a) for p, a in zip(scenario.perturbations, st.applied, strict=True)]
        off = offset_mm(scenario.scene.setup, st.h_actual)
        size, _, pivot = name.partition("__pivot=")
        if off > 0:
            out.append({"variant": name, "size_px": float(size.split("=")[1]), "pivot": pivot,
                        "deg": round(abs(tags[0]["deg"]), 5), "offset_mm": round(off, 4),
                        "offset_px": round(off / coarse_pitch_mm(scenario.scene.setup), 4)})
    return sorted(out, key=lambda r: (r["pivot"], r["size_px"]))


def nuisance_lanes(lo: int = 1700, hi: int = 2800, step: int = 2) -> dict[str, Any]:
    """The all-nuisances variant: each nuisance's state over time, and the true offset (zero)."""
    scenario = _variants(NUISANCE)[ALL_NUISANCES]
    setup = scenario.scene.setup
    lanes: dict[str, list[list[float]]] = {"offset_mm": [], "lamp_b": [], "room_light": [], "camera_bump": [],
                                           "person": []}
    for i in range(lo, hi, step):
        st = frame_state(scenario, i)
        t = round(float(st.t), 3)
        values = {"offset_mm": round(offset_mm(setup, st.h_actual), 6), "lamp_b": round(st.gains["b"], 4),
                  "room_light": round(st.ambient, 4), "camera_bump": round(max(st.camera_key or (0.0,)), 2),
                  "person": 1.0 if st.people else 0.0}
        for k, v in values.items():
            if not lanes[k] or lanes[k][-1][1] != v:
                lanes[k].append([t, v])
    end = round(float(scenario.timing.time(hi - 1)), 3)
    for points in lanes.values():
        points.append([end, points[-1][1]])
    return {"lanes": lanes, "t0": round(float(scenario.timing.time(lo)), 3), "t1": end}
