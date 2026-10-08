"""Schedules: how strongly a perturbation or nuisance acts at each moment.

A schedule turns a time t (exact Fraction seconds) into a dimensionless multiplier m(t) of the
change it drives: 0 means "not happening", 1 means "the configured magnitude". The kinds model
what real installations do:

  none            m = 0: nothing happens.
  step            0 before t0, 1 from t0 on: a projector bumped once, a light switched.
  staircase       levels[k] for t in [t0 + k hold, t0 + (k+1) hold), the last level held after:
                  a slow walk through known offsets.
  drift           rate_per_h x hours since t0: thermal creep or a sagging mount.
  ramp            0 to 1 linearly over duration_s from t0, then 1: a lamp dimming gradually.
  bump_then_hold  jumps to `peak` at t0, then relaxes to `hold` with time constant settle_s:
                  a knock that partly springs back.
  oscillate       (1 - cos(2 pi (t - t0) / period)) / 2: a heating/cooling cycle.

Drift, ramp, bump_then_hold and oscillate change continuously. The perturbation applying them
quantizes the resulting displacement (to 0.02 projector pixels), so the renderer sees a finite
set of geometries and its caches stay effective.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from sim.cfg import check_keys, choice, num, seconds

KINDS = ("none", "step", "staircase", "drift", "ramp", "bump_then_hold", "oscillate")
CONTINUOUS = frozenset({"drift", "ramp", "bump_then_hold", "oscillate"})
_KEYS = {
    "none": set(),
    "step": {"t0_s"},
    "staircase": {"t0_s", "levels", "hold_s"},
    "drift": {"t0_s", "rate_per_h"},
    "ramp": {"t0_s", "duration_s"},
    "bump_then_hold": {"t0_s", "peak", "hold", "settle_s"},
    "oscillate": {"t0_s", "period_s"},
}


@dataclass(frozen=True)
class Schedule:
    kind: str = "none"
    t0: Fraction = Fraction(0)
    levels: tuple[float, ...] = ()
    hold: Fraction = Fraction(0)  # staircase step length
    rate_per_h: float = 0.0
    duration: Fraction = Fraction(0)  # ramp length
    peak: float = 1.0
    hold_value: float = 1.0
    settle_s: float = 0.0
    period: Fraction = Fraction(0)

    @property
    def continuous(self) -> bool:
        return self.kind in CONTINUOUS

    @property
    def onset(self) -> Fraction | None:
        """When the multiplier first leaves 0 (None if it never does)."""
        return None if self.kind == "none" else self.t0

    def value(self, t: Fraction) -> float:
        if self.kind == "none" or t < self.t0:
            return 0.0
        dt = t - self.t0
        if self.kind == "step":
            return 1.0
        if self.kind == "staircase":
            return float(self.levels[min(int(dt // self.hold), len(self.levels) - 1)])
        if self.kind == "drift":
            return self.rate_per_h * float(dt) / 3600.0
        if self.kind == "ramp":
            return 1.0 if dt >= self.duration else float(dt / self.duration)
        if self.kind == "bump_then_hold":
            decay = math.exp(-float(dt) / self.settle_s) if self.settle_s > 0 else 0.0
            return self.hold_value + (self.peak - self.hold_value) * decay
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * float(dt / self.period)))  # oscillate


def from_config(cfg: Mapping[str, Any] | str | None, where: str) -> Schedule:
    """Parse ``{type: step, t0_s: 615}`` and friends; ``none``/absent means no schedule."""
    if cfg is None or cfg == "none" or (isinstance(cfg, Mapping) and cfg.get("type") == "none" and len(cfg) == 1):
        return Schedule()
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{where}: expected a mapping like {{type: step, t0_s: 615}}, got {cfg!r}")
    kind = choice(cfg.get("type"), KINDS, f"{where}.type")
    rest = {k: v for k, v in cfg.items() if k != "type"}
    check_keys(rest, _KEYS[kind], f"{where} ({kind})")
    missing = _KEYS[kind] - set(rest) - {"hold", "peak"}
    if missing:
        raise ValueError(f"{where} ({kind}): missing keys {sorted(missing)}")
    t0 = seconds(rest.get("t0_s", 0), f"{where}.t0_s")
    if kind == "staircase":
        levels = tuple(num(v, f"{where}.levels") for v in rest["levels"])
        hold = seconds(rest["hold_s"], f"{where}.hold_s")
        if not levels or hold <= 0:
            raise ValueError(f"{where}: staircase needs levels and hold_s > 0")
        return Schedule(kind, t0, levels=levels, hold=hold)
    if kind == "drift":
        return Schedule(kind, t0, rate_per_h=num(rest["rate_per_h"], f"{where}.rate_per_h"))
    if kind == "ramp":
        duration = seconds(rest["duration_s"], f"{where}.duration_s")
        if duration <= 0:
            raise ValueError(f"{where}: ramp needs duration_s > 0")
        return Schedule(kind, t0, duration=duration)
    if kind == "bump_then_hold":
        settle = num(rest["settle_s"], f"{where}.settle_s")
        if settle < 0:
            raise ValueError(f"{where}: settle_s must be >= 0")
        return Schedule(kind, t0, peak=num(rest.get("peak", 1.0), f"{where}.peak"),
                        hold_value=num(rest.get("hold", 1.0), f"{where}.hold"), settle_s=settle)
    if kind == "oscillate":
        period = seconds(rest["period_s"], f"{where}.period_s")
        if period <= 0:
            raise ValueError(f"{where}: oscillate needs period_s > 0")
        return Schedule(kind, t0, period=period)
    return Schedule(kind, t0)
