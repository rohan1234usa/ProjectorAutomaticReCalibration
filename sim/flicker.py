"""Projector flicker, and the bands it paints on a rolling-shutter camera.

Physics. A projector's light is not steady: lamp ballasts, LED drivers and DLP colour wheels
modulate it at a fixed frequency f (here a sinusoid, 1 + a sin(2 pi f t + phi)). A camera pixel
row collects light over its exposure window [s, s + tau]. Its gain is the average of the
modulation over that window,

    gain(s) = 1 + a [cos(2 pi f s + phi) - cos(2 pi f (s + tau) + phi)] / (2 pi f tau),

which is exactly 1 when tau is a whole number of modulation periods: that is why the exposure
is locked to a multiple of the refresh period (CLAUDE.md decision 7), and why flicker appears
only when that rule is broken. A rolling shutter starts each row a line time later than the one
above (s = row x t_line), so a broken exposure shows up as horizontal bands; a global shutter
(t_line = 0) gives the same gain to the whole frame. The camera's clock is not locked to the
projector's, and frames are half a second apart, so where in the modulation cycle a frame starts
(phi) is effectively random: it is drawn per frame from the seed, spawn key (2, flicker, frame).
That is what makes the bands move and the brightness change from frame to frame.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from sim.cfg import check_keys, num


@dataclass(frozen=True)
class Flicker:
    projector: str
    amplitude: float  # a: peak relative modulation of the projector's light
    frequency_hz: float  # f
    line_time_s: float  # rolling-shutter delay between consecutive rows (0: global shutter)
    seed: int
    number: int  # which flicker of the scenario, for its random stream

    def phase(self, frame: int) -> float:
        """Where in the modulation cycle frame `frame`'s first row starts, radians."""
        rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(2, self.number, frame)))
        return float(rng.uniform(0.0, 2 * math.pi))

    def row_gain(self, frame: int, exposure: Fraction, rows: int) -> np.ndarray:
        """Gain of every camera row of frame `frame`, (rows,) float32."""
        tau = float(exposure)
        w = 2 * math.pi * self.frequency_hz
        phi = self.phase(frame)
        start = self.line_time_s * np.arange(rows, dtype=np.float64)
        gain = 1.0 + self.amplitude * (np.cos(w * start + phi) - np.cos(w * (start + tau) + phi)) / (w * tau)
        return gain.astype(np.float32)


_KEYS = {"type", "projector", "amplitude", "frequency_hz", "line_time_s"}


def from_config(cfg: Mapping[str, Any], projectors: tuple[str, ...], seed: int, number: int, where: str) -> Flicker:
    """``{type: flicker, projector: a, amplitude: 0.02, frequency_hz: 100, line_time_s: 1.0e-5}``."""
    check_keys(cfg, _KEYS, where)
    projector = cfg.get("projector")
    if projector not in projectors:
        raise ValueError(f"{where}.projector: must be one of {list(projectors)}, got {projector!r}")
    amplitude = num(cfg.get("amplitude", 0.02), f"{where}.amplitude")
    frequency = num(cfg.get("frequency_hz", 100.0), f"{where}.frequency_hz")
    if not 0 <= amplitude < 1 or frequency <= 0:
        raise ValueError(f"{where}: need 0 <= amplitude < 1 and frequency_hz > 0")
    return Flicker(projector, amplitude, frequency, num(cfg.get("line_time_s", 0.0), f"{where}.line_time_s"),
                   seed, number)
