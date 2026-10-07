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
REFERENCE_MODES = ("auto", "on", "off")


def tolerance_from_viewing_distance(distance_mm: float) -> float:
    """Smallest on-screen offset in mm that a viewer at `distance_mm` can resolve (~1 arcminute)."""
    if distance_mm <= 0:
        raise ValueError(f"viewing distance must be positive, got {distance_mm}")
    return distance_mm * math.tan(ARCMINUTE_RAD)


@dataclass(frozen=True)
class DetectorConfig:
    """Detector tunables. Field names match CLAUDE.md section 10 exactly."""

    # -- cadence --
    ADJUSTABLE_INTERVAL_IN_SECONDS: float = 30.0  # boundary check cadence
    SAFETY_CHECK_S: float = 900.0  # timed overlap check regardless of boundary result
    TRUSTED_WINDOW_S: float = 600.0  # after calibration: refine boxes, fill baseline pools
    SAMPLE_EVERY_S: float = 0.5  # frame sampling period for the pools
    POOL_FRAMES: int = 60  # usable frames pooled per overlap check
    # -- thresholds in mm --
    TOLERANCE_MM: float = 1.7  # about 1 arcmin at 6 m; swept in evaluation
    TRIGGER_MM: float = 0.85  # boundary offset that triggers the overlap check
    AGREE_MM: float = 1.4  # layers "agree" within this
    MIN_OFFSET_MM: float = 1.5  # smallest echo the cepstrum can separate from its origin peak
    INLIER_MM: float = 0.7  # RANSAC inlier distance for the smooth motion field
    # -- decision --
    YES_VOTES: tuple[int, int] = (3, 4)  # K of N intervals
    CLEAR_RATIO: float = 0.5  # back to NO only below CLEAR_RATIO * TOLERANCE_MM
    HOLD_MAX_INTERVALS: int = 2  # intervals a boundary-vs-artifacts disagreement may hold the answer
    # -- geometry on the mm canvas --
    CORE_MIN_WEIGHT: float = 0.2  # core = overlap points where both blend weights are at least this
    TILE_MM: float = 64.0
    CTRL_MARGIN_MM: float = 40.0
    EDGE_PIECE_MM: float = 80.0
    CANVAS_PX_PER_MM: float = 2.0
    # -- frame routing --
    MOTION_LEVEL: float = 0.02  # mean |frame difference| above this (x white) = motion or a cut: skip
    MAX_CLIPPED: float = 0.002  # fraction of saturated pixels above this: skip
    DARK_LEVEL: float = 0.003  # region mean below this (x white) = dark frame
    MIN_TILE_EDGES: float = 0.02  # edge density a tile needs to count as textured
    VARIETY_MAX_NCC: float = 0.98  # blind mode keeps a frame only if it differs from the last kept
    FIDUCIAL_MOVE_PX: float = 0.3  # marker motion above this = camera moved: re-solve, never YES
    # -- evidence quality --
    MIN_PEAK_SNR: float = 6.0  # a cepstral or kernel peak must stand this far above the noise
    NULL_FACTOR: float = 1.5  # ... and above NULL_FACTOR x the control-vs-control null
    ECHO_FLOOR_CAMERA_PX: float = 2.5  # the cepstrum cannot resolve echoes below this many camera px
    MIN_GOOD_TILES: int = 3  # fewer valid tiles -> "can't tell"
    MIN_TILE_FRAMES: int = 20  # textured frames an echo tile needs
    MIN_SEAM_FRAMES: int = 20  # lit (or dark) frames a seam block needs
    MIN_EDGE_SNR: float = 4.0  # pooled SNR an edge piece needs to count this interval
    LINE_INLIER_FRAC: float = 0.6  # tiles must agree on one smooth field at least this often
    FLAT_MAX_DEV: float = 0.01  # a hotspot block is "flat" if its sub-blocks agree within this
    LAMP_WARN: float = 0.03  # fitted lamp gain off by more than this = lamp warning
    # -- reference mode --
    REFERENCE_MODE: str = "auto"  # auto: use source frames when the feed supplies them; on; off
    WIENER_LAMBDA: float = 0.01  # Wiener regularisation, as a fraction of the mean source power
    SOURCE_RING: int = 10  # source frames kept for matching (the camera lags by a few frames)
    MATCH_MIN_NCC: float = 0.9  # a source frame must match the camera frame at least this well

    def __post_init__(self) -> None:
        positive = (
            "ADJUSTABLE_INTERVAL_IN_SECONDS", "SAFETY_CHECK_S", "SAMPLE_EVERY_S", "POOL_FRAMES",
            "TOLERANCE_MM", "TRIGGER_MM", "AGREE_MM", "MIN_OFFSET_MM", "INLIER_MM", "HOLD_MAX_INTERVALS",
            "TILE_MM", "CTRL_MARGIN_MM", "EDGE_PIECE_MM", "CANVAS_PX_PER_MM", "MOTION_LEVEL", "DARK_LEVEL",
            "MIN_TILE_EDGES", "FIDUCIAL_MOVE_PX", "MIN_PEAK_SNR", "NULL_FACTOR", "ECHO_FLOOR_CAMERA_PX",
            "MIN_GOOD_TILES", "MIN_TILE_FRAMES", "MIN_SEAM_FRAMES", "MIN_EDGE_SNR", "FLAT_MAX_DEV",
            "LAMP_WARN", "WIENER_LAMBDA", "SOURCE_RING",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.TRUSTED_WINDOW_S < 0:
            raise ValueError("TRUSTED_WINDOW_S must be non-negative")
        k, n = self.YES_VOTES
        if not 1 <= k <= n:
            raise ValueError(f"YES_VOTES must be (K, N) with 1 <= K <= N, got {self.YES_VOTES}")
        unit_interval = ("CLEAR_RATIO", "MAX_CLIPPED", "VARIETY_MAX_NCC", "LINE_INLIER_FRAC", "MATCH_MIN_NCC")
        for name in unit_interval:
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0, 1), got {getattr(self, name)}")
        # Both weights >= CORE_MIN_WEIGHT is only satisfiable when it is <= 0.5 (a + b = 1).
        if not 0 < self.CORE_MIN_WEIGHT <= 0.5:
            raise ValueError(f"CORE_MIN_WEIGHT must be in (0, 0.5], got {self.CORE_MIN_WEIGHT}")
        if self.REFERENCE_MODE not in REFERENCE_MODES:
            raise ValueError(f"REFERENCE_MODE must be one of {REFERENCE_MODES}, got {self.REFERENCE_MODE!r}")

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
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{name}: expected a string, got {value!r}")
        return value
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
