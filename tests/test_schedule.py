"""Schedules: the multiplier each kind gives over time, and clear errors for bad ones."""

import math
from fractions import Fraction as F

import pytest

from sim.schedule import from_config


def test_none_and_step():
    assert from_config(None, "s").value(F(1000)) == 0.0
    step = from_config({"type": "step", "t0_s": 615}, "s")
    assert step.value(F(61499, 100)) == 0.0 and step.value(F(615)) == 1.0 and step.onset == 615
    assert not step.continuous


def test_staircase_holds_its_last_level():
    s = from_config({"type": "staircase", "t0_s": 10, "levels": [0.25, 0.5, 1], "hold_s": 5}, "s")
    assert [s.value(F(t)) for t in (9, 10, 14, 15, 20, 99)] == [0.0, 0.25, 0.25, 0.5, 1.0, 1.0]


def test_drift_ramp_bump_and_oscillate():
    drift = from_config({"type": "drift", "t0_s": 600, "rate_per_h": 2.0}, "s")
    assert drift.value(F(600 + 1800)) == pytest.approx(1.0) and drift.continuous
    ramp = from_config({"type": "ramp", "t0_s": 100, "duration_s": 50}, "s")
    assert [ramp.value(F(t)) for t in (99, 125, 150, 400)] == [0.0, 0.5, 1.0, 1.0]
    bump = from_config({"type": "bump_then_hold", "t0_s": 10, "peak": 3.0, "hold": 1.0, "settle_s": 2.0}, "s")
    assert bump.value(F(10)) == 3.0 and bump.value(F(12)) == pytest.approx(1 + 2 * math.exp(-1))
    osc = from_config({"type": "oscillate", "t0_s": 0, "period_s": 1200}, "s")
    assert osc.value(F(0)) == 0.0 and osc.value(F(600)) == pytest.approx(1.0) and osc.value(F(300)) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"type": "jump", "t0_s": 1}, "must be one of"),
        ({"type": "step"}, "missing keys"),
        ({"type": "step", "t0_s": 1, "hold_s": 2}, "unknown keys"),
        ({"type": "staircase", "t0_s": 1, "levels": [], "hold_s": 2}, "levels and hold_s"),
        ({"type": "ramp", "t0_s": 1, "duration_s": 0}, "duration_s > 0"),
        ("sometimes", "expected a mapping"),
    ],
)
def test_bad_schedules_fail_clearly(cfg, message):
    with pytest.raises(ValueError, match=message):
        from_config(cfg, "schedule")
