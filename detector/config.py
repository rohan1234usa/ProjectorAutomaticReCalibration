"""Every tunable of the detector, in one frozen dataclass.

Why one place: the detector must be reproducible and its thresholds sweepable, so each
number it uses is named here. The names are the contract between the detector, the
evaluation harness and YAML files (CLAUDE.md section 10); the defaults are starting
guesses that evaluation will move.

Where the main threshold comes from: people with normal vision resolve detail down to
about one arcminute. Seen from the closest seat, at distance D, one arcminute covers
D * tan(1') millimetres of screen. A misalignment smaller than that cannot be seen, so
TOLERANCE_MM starts there (D = 6 m gives 1.745 mm) and is later learned from operators
pressing the recalibrate button.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

ARCMINUTE_RAD = math.radians(1.0 / 60.0)


def tolerance_from_viewing_distance(distance_mm: float) -> float:
    """Smallest on-screen offset in mm that a viewer at `distance_mm` can resolve (~1 arcminute)."""
    if distance_mm <= 0:
        raise ValueError(f"viewing distance must be positive, got {distance_mm}")
    return distance_mm * math.tan(ARCMINUTE_RAD)


@dataclass(frozen=True)
class DetectorConfig:
    """Detector tunables. Field names match CLAUDE.md section 10 exactly."""

    ADJUSTABLE_INTERVAL_IN_SECONDS: float = 30.0  # boundary check cadence
    SAFETY_CHECK_S: float = 900.0  # timed artifact check regardless of boundary result
    TRUSTED_WINDOW_S: float = 600.0  # after calibration: refine boxes, learn references
    POOL_FRAMES: int = 60  # usable frames pooled per artifact check
    TOLERANCE_MM: float = 1.7  # about 1 arcmin at 6 m; swept in evaluation
    TRIGGER_MM: float = 0.85  # boundary offset that triggers the artifact check
    AGREE_MM: float = 1.4  # layers "agree" within this
    YES_VOTES: tuple[int, int] = (3, 4)  # K of N intervals
    CLEAR_RATIO: float = 0.5  # back to NO only below CLEAR_RATIO * TOLERANCE_MM
    CORE_MIN_WEIGHT: float = 0.2  # core = overlap points where both blend weights are at least this
    TILE_MM: float = 64.0
    CTRL_MARGIN_MM: float = 40.0
    EDGE_PIECE_MM: float = 80.0
    CANVAS_PX_PER_MM: float = 2.0
    HASH_MATCH_BITS: int = 6
    MIN_PEAK_SNR: float = 6.0
    MIN_GOOD_TILES: int = 3

    def __post_init__(self) -> None:
        positive = (
            "ADJUSTABLE_INTERVAL_IN_SECONDS", "SAFETY_CHECK_S", "TOLERANCE_MM", "TRIGGER_MM",
            "AGREE_MM", "TILE_MM", "CTRL_MARGIN_MM", "EDGE_PIECE_MM", "CANVAS_PX_PER_MM",
            "MIN_PEAK_SNR", "POOL_FRAMES", "MIN_GOOD_TILES",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.TRUSTED_WINDOW_S < 0 or self.HASH_MATCH_BITS < 0:
            raise ValueError("TRUSTED_WINDOW_S and HASH_MATCH_BITS must be non-negative")
        k, n = self.YES_VOTES
        if not 1 <= k <= n:
            raise ValueError(f"YES_VOTES must be (K, N) with 1 <= K <= N, got {self.YES_VOTES}")
        if not 0 < self.CLEAR_RATIO < 1:
            raise ValueError(f"CLEAR_RATIO must be in (0, 1), got {self.CLEAR_RATIO}")
        # Both weights >= CORE_MIN_WEIGHT is only satisfiable when it is <= 0.5 (a + b = 1).
        if not 0 < self.CORE_MIN_WEIGHT <= 0.5:
            raise ValueError(f"CORE_MIN_WEIGHT must be in (0, 0.5], got {self.CORE_MIN_WEIGHT}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorConfig:
        """Build from a mapping; unknown keys are an error (they are usually typos)."""
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(f"unknown detector config keys: {unknown}")
        values = {name: _coerce(name, value, known[name].default) for name, value in data.items()}
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DetectorConfig:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: expected a mapping of config keys")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["YES_VOTES"] = list(self.YES_VOTES)
        return out


def _coerce(name: str, value: Any, default: Any) -> Any:
    """Convert a YAML value to the type of the field's default, refusing lossy conversions."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: booleans are not valid here")
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            raise ValueError(f"{name}: expected a list of {len(default)} integers, got {value!r}")
        return tuple(_coerce(name, v, d) for v, d in zip(value, default))
    if isinstance(default, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{name}: expected an integer, got {value!r}")
        return value
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number, got {value!r}")
    return float(value)
