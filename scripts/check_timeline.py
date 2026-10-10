"""The dataset checker's own timeline: what a scenario says is happening at each frame, apart from sim/.

``scripts/check_dataset.py`` holds a dataset's metadata to what its scenario.yaml asked for. The
geometry is in ``scripts/check_geometry.py``; this module re-derives the timeline the plain way,
from the scenario text alone:

  time        frame i is exposed during [phase_s + i x sample_every_s, ... + exposure_s), in exact
              fractions of a second (YAML's 0.0123 is 123/10000; "1/30" is 1/30);
  schedules   the strength m(t) of a perturbation or nuisance: step, staircase, drift, ramp,
              bump_then_hold, oscillate (sim/schedule.py's table says what each means);
  nuisances   the room light (the base ambient plus each change times its schedule), each
              projector's lamp (1 + (gain - 1) m(t), multiplied over lamps), each camera knock
              (m(t) in steps of 0.01), whether anyone is passing (t0 <= t < t0 + duration_s),
              which projectors flicker, and whether sharpening is on;
  content     which pictures the projectors show during the exposure, and for what share of it:
              items play in order (looped or not), each picture held for hold_s (a video frame
              for 1/fps), and the projectors show what was sent lag_s earlier. A key
              item/loop/index names a picture; the loop is 0 for kinds that replay (flat, black,
              video);
  camera      the recorded camera is the frame-0 camera knocked by each bump's recorded strength;
  markers     with nobody passing, exactly the markers whose paper lies inside that camera's
              frame; with someone passing, some of them.

``Timeline.check`` compares one metadata line with all of this. A field the line lacks (a dataset
written before the field existed) is not a problem but a skipped check, and is reported as one.
"""

from __future__ import annotations

import bisect
import math
from fractions import Fraction
from typing import Any

import numpy as np

from scripts.check_geometry import knocked, markers_in_view

BUMP_QUANTUM = 0.01
TOL = 1e-9  # recorded states are computed in floating point from the same exact times
VARIES = ("deck", "held", "photo", "dark", "stripes")  # kinds that draw new pictures every loop

ContentKey = tuple[int, int, int]


def exact(value: Any) -> Fraction:
    """A YAML number or text as the exact decimal or fraction written."""
    return Fraction(str(value))


def perturbation_specs(scenario: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """(projector, spec) for every perturbation, in the simulator's order: a, then b, then as listed."""
    out = []
    perturbation = scenario.get("perturbation")
    if isinstance(perturbation, dict):
        for name in ("a", "b"):
            items = perturbation.get(name)
            if items is not None:
                out += [(name, s) for s in ([items] if isinstance(items, dict) else items)]
    return out


def expected_multiplier(schedule: Any, t: Fraction) -> float:
    """The schedule's multiplier m(t), before any quantization."""
    if schedule in (None, "none") or schedule.get("type") == "none" or t < exact(schedule.get("t0_s", 0)):
        return 0.0
    kind, dt = schedule["type"], t - exact(schedule.get("t0_s", 0))
    if kind == "step":
        return 1.0
    if kind == "staircase":
        levels = schedule["levels"]
        return float(levels[min(int(dt // exact(schedule["hold_s"])), len(levels) - 1)])
    if kind == "drift":
        return float(exact(schedule["rate_per_h"]) * dt / 3600)
    if kind == "ramp":
        return float(min(Fraction(1), dt / exact(schedule["duration_s"])))
    if kind == "bump_then_hold":
        peak, hold, tau = (float(exact(schedule.get(k, 1))) for k in ("peak", "hold", "settle_s"))
        return hold + (peak - hold) * (math.exp(-float(dt) / tau) if tau > 0 else 0.0)
    return 0.5 * (1 - math.cos(2 * math.pi * float(dt / exact(schedule["period_s"]))))  # oscillate


def quantized(value: float, quantum: float) -> list[float]:
    """value in whole quanta; both neighbours when it lies on the halfway point (rounding may go either way)."""
    q = value / quantum
    if abs(q - math.floor(q) - 0.5) < 1e-6:
        return [math.floor(q) * quantum, math.ceil(q) * quantum]
    return [round(q) * quantum]


def _item(cfg: dict[str, Any], default_hold: Fraction) -> dict[str, Any]:
    kind = cfg["type"]
    blank = cfg.get("blank_overlap")
    suffix = ("_letterbox" if exact(cfg.get("letterbox", 0)) else "") + \
             ("_blank" if blank is not None and blank is not False else "")
    if kind == "video":
        fps = exact(cfg.get("fps", 30))
        duration = exact(cfg["duration_s"]) if "duration_s" in cfg else default_hold
        cut = cfg.get("cut_s", 8)
        return {"kind": kind, "hold": 1 / fps, "count": int(duration * fps), "fps": fps,
                "cut": None if cut is None else exact(cut), "tags": [f"video_{cfg.get('style', 'photo')}"],
                "suffix": suffix}
    hold = exact(cfg["hold_s"]) if "hold_s" in cfg else default_hold
    count = (int(exact(cfg.get("slides", 1))) if kind == "deck"
             else int(exact(cfg.get("pictures", 1))) if kind in ("photo", "dark", "stripes") else 1)
    densities = cfg.get("densities", [cfg.get("density", "medium")])
    tags = [f"{kind}_{d}" for d in densities] if kind in ("deck", "held") else [kind]
    return {"kind": kind, "hold": hold, "count": count, "tags": tags, "suffix": suffix}


class Content:
    """The pictures a scenario sends over time, from its content block."""

    def __init__(self, cfg: Any, default_hold: Fraction, lag: Fraction) -> None:
        if isinstance(cfg, dict) and "items" in cfg:
            items, self.loop = cfg["items"], bool(cfg.get("loop", False))
        elif isinstance(cfg, dict):
            items, self.loop = [cfg], False
        else:
            items, self.loop = list(cfg), False
        self.items = [_item(c, default_hold) for c in items]
        self.starts, t = [], Fraction(0)
        for item in self.items:
            self.starts.append(t)
            t += item["hold"] * item["count"]
        self.cycle, self.lag = t, lag

    def locate(self, t: Fraction) -> tuple[ContentKey, Fraction]:
        """The picture being sent at content time t, and when it is replaced."""
        t = max(t, Fraction(0))
        loop = int(t // self.cycle) if self.loop else 0
        offset = t - loop * self.cycle
        i = bisect.bisect_right(self.starts, offset) - 1
        item = self.items[i]
        index = min(int((offset - self.starts[i]) // item["hold"]), item["count"] - 1)
        ends = loop * self.cycle + self.starts[i] + (index + 1) * item["hold"]
        if not self.loop and i == len(self.items) - 1 and index == item["count"] - 1:
            ends = max(ends, t) + self.cycle  # the last picture of a run that does not loop stays up
        return (i, loop if item["kind"] in VARIES else 0, index), ends

    def segments(self, t: Fraction, exposure: Fraction) -> list[tuple[ContentKey, Fraction]]:
        """Pictures shown during the exposure [t, t + exposure), with their shares of it."""
        out: list[tuple[ContentKey, Fraction]] = []
        now, end = t - self.lag, t - self.lag + exposure
        while now < end:
            key, change = self.locate(now)
            nxt = min(change, end)
            if out and out[-1][0] == key:
                out[-1] = (key, out[-1][1] + (nxt - now) / exposure)
            else:
                out.append((key, (nxt - now) / exposure))
            now = nxt
        return out

    def _scene(self, item: dict[str, Any], k: int) -> int:
        return 0 if item["cut"] is None else int((Fraction(k) / item["fps"]) // item["cut"])

    def _same_shot(self, a: ContentKey, b: ContentKey) -> bool:
        item = self.items[a[0]]
        if a[0] != b[0] or item["kind"] != "video" or b[2] != a[2] + 1:
            return False
        return self._scene(item, a[2]) == self._scene(item, b[2])

    def record(self, t: Fraction, exposure: Fraction) -> dict[str, Any]:
        """The content block of a metadata line for the exposure starting at t."""
        segments = self.segments(t, exposure)
        keys = [key for key, _ in segments]
        tags = {self.items[i]["tags"][index % len(self.items[i]["tags"])] + self.items[i]["suffix"]
                for i, _, index in keys}
        return {"segments": [["/".join(str(k) for k in key), float(w)] for key, w in segments],
                "tag": "+".join(sorted(tags)), "frames_in_exposure": len(segments),
                "cut_in_exposure": any(not self._same_shot(a, b) for a, b in zip(keys, keys[1:], strict=False))}


def _same_content(key: str, value: Any, want: Any) -> bool:
    if key != "segments":
        return value == want
    return len(value) == len(want) and all(v[0] == w[0] and abs(v[1] - w[1]) <= 1e-12
                                           for v, w in zip(value, want, strict=False))


class Timeline:
    """What the scenario says about every frame apart from the projectors' geometry."""

    def __init__(self, scenario: dict[str, Any], setup: dict[str, Any], first: dict[str, Any]) -> None:
        camera = scenario.get("camera", {})
        self.phase, self.exposure = exact(camera.get("phase_s", 0)), exact(camera.get("exposure_s", "1/30"))
        self.sample = exact(scenario.get("sample_every_s", "1/2"))
        duration = exact(scenario.get("duration_s", self.sample))
        n = max(0, -((self.phase - duration) // self.sample))
        lag = exact((scenario.get("reference") or {}).get("lag_s", 0))
        shown_until = self.phase + (n - 1) * self.sample + self.exposure - lag  # what an item without hold_s lasts
        self.content = Content(scenario["content"], shown_until, lag)
        screen = scenario["screen"]
        self.base = float(exact(screen.get("ambient", 0.02)))
        self.bezel_light = float(exact((screen.get("bezel") or {}).get("light", 0.0)))
        self.nuisances = list(scenario.get("nuisances") or [])
        self.setup, self.n_frames = setup, n
        self.camera0 = np.array(first["truth"]["camera_h_mm_to_px"])
        self.knocked_at_start = any((first.get("nuisances") or {}).get("camera_bump") or [])
        self._in_view: dict[bytes, list[int]] = {}

    def _of(self, kind: str) -> list[dict[str, Any]]:
        return [n for n in self.nuisances if n["type"] == kind]

    def nuisance_state(self, t: Fraction) -> dict[str, Any]:
        ambient = self.base + sum((float(exact(n["ambient"])) - self.base) * expected_multiplier(n["schedule"], t)
                                  for n in self._of("room_light"))
        lamps = {"a": 1.0, "b": 1.0}
        for n in self._of("lamp"):
            lamps[n["projector"]] *= 1.0 + (float(exact(n.get("gain", 0.85))) - 1.0) * expected_multiplier(n["schedule"], t)
        return {
            "ambient": ambient, "bezel_light": self.bezel_light, "lamp_gain": lamps,
            "camera_bump": [quantized(expected_multiplier(n["schedule"], t), BUMP_QUANTUM) for n in self._of("camera_bump")],
            "occluder": any(exact(n["t0_s"]) <= t < exact(n["t0_s"]) + exact(n.get("duration_s", 6))
                            for n in self._of("occluder")),
            "flicker": sorted({n["projector"] for n in self._of("flicker")}),
            "sharpening": bool(self._of("sharpening")),
        }

    def check(self, line: dict[str, Any], problems: list[str], skipped: set[str]) -> None:
        """Compare one metadata line with the timeline; append problems, note checks the line cannot take."""
        i = line["i"]
        t = self.phase + i * self.sample
        if line["t_s"] != float(t):
            problems.append(f"frame {i}: t_s is {line['t_s']}, phase_s + i x sample_every_s is {float(t)}")
        state = self.nuisance_state(t)
        got = line.get("nuisances", {})
        for key, want in state.items():
            if key not in got:
                skipped.add(f"nuisances.{key}")
                continue
            value = got[key]
            if key == "camera_bump":
                same = len(value) == len(want) and all(any(abs(v - c) <= TOL for c in cs) for v, cs in zip(value, want, strict=False))
            elif key == "lamp_gain":
                same = set(value) == set(want) and all(abs(value[n] - want[n]) <= TOL for n in want)
            elif key in ("ambient", "bezel_light"):
                same = abs(value - want) <= TOL
            else:
                same = value == want
            if not same:
                problems.append(f"frame {i}: nuisances.{key} is {value}, the scenario says {want}")
        for key, want in self.content.record(t, self.exposure).items():
            value = line.get("content", {}).get(key)
            if value is None:
                skipped.add(f"content.{key}")
            elif not _same_content(key, value, want):
                problems.append(f"frame {i}: content.{key} is {value}, the scenario says {want}")
        self._camera_and_markers(line, state["occluder"], problems, skipped)

    def _camera_and_markers(self, line: dict[str, Any], occluded: bool, problems: list[str], skipped: set[str]) -> None:
        i, truth = line["i"], line["truth"]
        h = np.array(truth["camera_h_mm_to_px"])
        bumps = (line.get("nuisances") or {}).get("camera_bump")
        if bumps is None or self.knocked_at_start:
            skipped.add("truth.camera_h_mm_to_px")
        else:
            want = self.camera0
            for knock, m in zip(self._of("camera_bump"), bumps, strict=False):
                if m:
                    shift = [float(exact(v)) for v in knock.get("shift_px", [0, 0])]
                    want = knocked(want, self.setup["camera"]["resolution"], shift,
                                   float(exact(knock.get("rotation_deg", 0))), m)
            if not np.allclose(h, want, rtol=1e-12, atol=1e-9):
                problems.append(f"frame {i}: the camera is not frame 0's camera knocked by {bumps}")
        key = h.tobytes()
        if key not in self._in_view:
            self._in_view[key] = markers_in_view(self.setup, h)
        seen, view = truth["markers_visible"], self._in_view[key]
        if (set(seen) - set(view)) if occluded else (seen != view):
            problems.append(f"frame {i}: markers_visible is {seen}; in view{' (someone passing)' if occluded else ''}: {view}")
