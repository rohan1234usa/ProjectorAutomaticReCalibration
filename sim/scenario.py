"""Scenarios: one YAML file per test idea, turned into the objects that render it.

Scenarios are declarative so that adding a test means adding a YAML file, not code. A scenario
describes an installation that stays fixed (screen and bezel with its markers, two projectors
and their arrangement, the calibration's blend, the camera) and what happens over time:

  duration_s, sample_every_s   the run and the camera's sampling period; frame i is exposed
                               during [phase_s + i sample_every_s, ... + exposure_s)
  trusted_window_s             the detector refines its baseline this long after calibration,
                               so no perturbation may start before it
  content                      what is shown (``sim/sequence.py``)
  perturbation                 how projectors drift (``sim/perturb.py``)
  nuisances                    camera bump, lamp dimming, room light, occluder, flicker, sharpening
                               (``sim/nuisance.py``); they never misalign, so truth stays aligned
  reference                    whether the source feed exists, and the display lag

``extends`` and ``sweep`` are resolved first (``sim/sweep.py``). Each module parses and checks
its own block, so a typo fails with a message naming the key. Every number goes through one
coercing reader, because YAML reads ``1e-3`` as a string.

Randomness: every random draw comes from the scenario seed through numpy SeedSequence spawn
keys -- content pictures (0, item, loop, index), camera noise (1, frame) -- so any frame can be
re-rendered on its own, in any process, and sweep variants share their noise.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sim import arrangements, camera, fiducials, nuisance, perturb, projector, screen
from sim.calibration import RAMPS, CalibrationSetup
from sim.camera import Camera
from sim.cfg import check_keys, choice, integer, num, require, seconds
from sim.fiducials import MarkerSet
from sim.nuisance import Nuisances
from sim.perturb import Perturbation
from sim.pictures import ContentGeometry
from sim.planar import apply_h, homography_from_points, raster_corners
from sim.projector import Projector
from sim.render import QUALITY, Quality, Renderer
from sim.screen import Screen
from sim.sequence import Sequence
from sim.sequence import from_config as sequence_from_config
from sim.sweep import expand, load_yaml

_REQUIRED_KEYS = {"screen", "arrangement", "projectors", "content", "camera"}
_TOP_KEYS = _REQUIRED_KEYS | {
    "name", "description", "seed", "quality", "blend", "duration_s", "sample_every_s", "trusted_window_s",
    "perturbation", "nuisances", "reference",
}


@dataclass(frozen=True, eq=False)
class Scene:
    """The installation: everything that stays fixed while a scenario runs."""

    name: str
    screen: Screen
    projectors: dict[str, Projector]
    setup: CalibrationSetup
    camera: Camera
    markers: MarkerSet | None
    quality: Quality
    seed: int
    camera_preset: str = "whole_screen"

    def renderer(self) -> Renderer:
        return Renderer(self.screen, self.projectors, self.setup, self.camera, self.quality, self.markers)


@dataclass(frozen=True)
class Timing:
    duration: Fraction
    sample_every: Fraction
    phase: Fraction  # camera clock offset: keeps exposures off video-frame boundaries
    exposure: Fraction
    trusted_window: Fraction

    @property
    def n_frames(self) -> int:
        """Frames whose exposure starts before the run ends: ceil((duration - phase) / sample_every)."""
        return max(0, -((self.phase - self.duration) // self.sample_every))

    def time(self, i: int) -> Fraction:
        """Start of frame i's exposure."""
        return self.phase + i * self.sample_every


@dataclass(frozen=True, eq=False)
class Scenario:
    name: str
    variant: str
    data: dict[str, Any]  # the resolved scenario (no extends, no sweep): written as scenario.yaml
    scene: Scene
    timing: Timing
    sequence: Sequence
    perturbations: tuple[Perturbation, ...]
    reference: dict[str, Any]
    nuisances: Nuisances = Nuisances()

    def content_image(self, i: int = 0) -> np.ndarray:
        """The first picture shown during frame i."""
        return self.sequence.image(self.sequence.segments(self.timing.time(i), self.timing.exposure)[0][0])

    def setup_dict(self) -> dict[str, Any]:
        """Exactly what the detector may read (CLAUDE.md section 3): never any ground truth."""
        scene = self.scene
        gammas = {n: p.gamma for n, p in scene.projectors.items()}
        return {
            "version": 1,
            "screen": {"size_mm": [float(v) for v in scene.screen.size_mm]},
            **scene.setup.to_setup_dict(gammas),
            "markers": None if scene.markers is None else scene.markers.to_setup_dict(),
            "camera": {**scene.camera.setup_dict(), "exposure_s": float(self.timing.exposure)},
            "reference": {"available": bool(self.reference["available"])},
        }


def load_scenarios(path: str | Path, quality: str | None = None) -> list[Scenario]:
    """Every variant of a scenario file, its ``extends`` and ``sweep`` resolved."""
    data = load_yaml(path)
    name = str(data.get("name", Path(path).stem))
    data["name"] = name
    variants = expand(data, canonical)
    return [scenario_from_dict(d, quality=quality, variant=v if len(variants) > 1 else name) for v, d in variants]


def load_scenario(path: str | Path, quality: str | None = None, variant: str | None = None) -> Scenario:
    scenarios = load_scenarios(path, quality)
    if variant is None and len(scenarios) == 1:
        return scenarios[0]
    for s in scenarios:
        if s.variant == variant:
            return s
    raise ValueError(f"{Path(path).name}: pick a variant from {[s.variant for s in scenarios]}")


def canonical(data: Mapping[str, Any]) -> dict[str, Any]:
    """The scenario with zero-size perturbations removed: sweep variants equal in effect match."""
    out = copy.deepcopy(dict(data))
    pert = out.get("perturbation")
    if isinstance(pert, Mapping):
        for name in list(pert):
            specs = pert[name]
            specs = [specs] if isinstance(specs, Mapping) else list(specs or [])
            kept = [s for s in specs if not _zero_size(s)]
            if kept:
                pert[name] = kept
            else:
                del pert[name]
        if not pert:
            out["perturbation"] = "none"
    return out


def _zero_size(spec: Any) -> bool:
    if not isinstance(spec, Mapping):
        return False
    for key, null in (("magnitude_px", 0.0), ("magnitude_mm", 0.0), ("deg", 0.0), ("factor", 1.0)):
        if key in spec:
            try:
                return num(spec[key], key) == null
            except ValueError:
                return False
    return False


def scenario_from_dict(
    data: Mapping[str, Any], quality: str | None = None, name: str | None = None, variant: str | None = None
) -> Scenario:
    """Build one scenario variant from resolved data (no ``extends`` or ``sweep`` left)."""
    if not isinstance(data, Mapping):
        raise ValueError("scenario: expected a mapping of keys")
    check_keys(data, _TOP_KEYS, "scenario")
    require(data, _REQUIRED_KEYS, "scenario")
    data = copy.deepcopy(dict(data))
    name = str(data.get("name", name or "scenario"))
    data["name"] = name
    if quality is not None:
        data["quality"] = quality
    seed = integer(data.get("seed", 0), "seed")
    screen_cfg = dict(data["screen"])
    the_screen = screen.from_config(screen_cfg)
    projectors = projector.from_config(data["projectors"])
    res = {n: p.resolution for n, p in projectors.items()}
    arrangement = arrangements.from_config(data["arrangement"], the_screen.size_mm, res)
    blend = dict(data.get("blend", {}))
    check_keys(blend, {"shape", "black_uplift"}, "blend")
    uplift = blend.get("black_uplift", False)
    if not isinstance(uplift, bool):
        raise ValueError("blend.black_uplift: expected true or false")
    setup = CalibrationSetup(
        h_cal={n: homography_from_points(raster_corners(res[n]), arrangement.corners[n]) for n in ("a", "b")},
        resolution=res,
        content_rect_mm=arrangement.content_rect_mm,
        blend_shape=choice(blend.get("shape", "cosine"), tuple(RAMPS), "blend.shape"),
        black_uplift=uplift,
    )
    inner = setup.inner_edges()
    if not inner["a"] and not inner["b"]:
        raise ValueError("arrangement: neither projector fades anywhere; the boxes do not overlap inside the content")
    markers_cfg = screen_cfg.get("bezel", {}).get("markers") if isinstance(screen_cfg.get("bezel"), Mapping) else None
    markers = None if markers_cfg is None else fiducials.from_config(markers_cfg, the_screen, setup.overlap())
    cam, preset, exposure, phase = camera.from_config(data["camera"], the_screen.size_mm, setup.overlap())
    if markers is not None:
        seen = fiducials.visible(markers, cam.h_mm_to_px, cam.resolution)
        hidden = sorted(set(range(len(markers.centres_mm))) - set(seen))
        if preset == "whole_screen" and hidden:
            raise ValueError(f"markers {hidden} fall outside the camera frame")
        if len(seen) < 4:
            raise ValueError(f"camera: only markers {seen} are in view; the detector needs at least 4")
    if preset == "zoomed":
        corners = apply_h(cam.h_mm_to_px, setup.overlap())
        w, h = cam.resolution
        if corners.min() < 0 or corners[:, 0].max() > w - 1 or corners[:, 1].max() > h - 1:
            raise ValueError("camera: the zoomed field does not hold the whole overlap; lower px_per_mm")
    quality_name = data.get("quality", "standard")
    if quality_name not in QUALITY:
        raise ValueError(f"quality must be one of {sorted(QUALITY)}, got {quality_name!r}")
    scene = Scene(name, the_screen, projectors, setup, cam, markers, QUALITY[quality_name], seed, preset)

    sample = seconds(data.get("sample_every_s", "1/2"), "sample_every_s")
    timing = Timing(
        duration=seconds(data.get("duration_s", sample), "duration_s"),
        sample_every=sample,
        phase=phase,
        exposure=exposure,
        trusted_window=seconds(data.get("trusted_window_s", 600), "trusted_window_s"),
    )
    if sample <= 0 or timing.duration <= 0 or not 0 < exposure <= sample or not 0 <= phase < sample:
        raise ValueError("timing: need sample_every_s > 0, duration_s > 0, "
                         "0 < exposure_s <= sample_every_s and 0 <= phase_s < sample_every_s")
    reference = dict(data.get("reference", {}) or {})
    check_keys(reference, {"available", "lag_s"}, "reference")
    available = reference.get("available", False)
    if not isinstance(available, bool):
        raise ValueError("reference.available: expected true or false")
    lag = seconds(reference.get("lag_s", 0), "reference.lag_s")
    if lag < 0:
        raise ValueError("reference.lag_s: the display cannot show content before it is sent")
    reference = {"available": available, "lag_s": lag}
    # The projectors show content(t - lag); the content must last until the last exposure ends.
    shown_until = timing.time(timing.n_frames - 1) + exposure - lag
    size = setup.content_size()
    x0, _, x1, _ = setup.content_rect_mm
    geometry = ContentGeometry(px_per_mm=size[0] / (x1 - x0),
                               overlap_px=apply_h(np.linalg.inv(setup.content_to_mm(size)), setup.overlap()))
    sequence = sequence_from_config(data["content"], seed, size, shown_until, geometry, lag=lag)
    perturbations = perturb.from_config(data.get("perturbation"), setup)
    for p in perturbations:
        if p.schedule.onset is not None and p.schedule.onset < timing.trusted_window:
            raise ValueError(f"perturbation of {p.projector} starts at {float(p.schedule.onset):g} s, inside the "
                             f"{float(timing.trusted_window):g} s trusted window after calibration")
    nuisances = nuisance.from_config(data.get("nuisances"), setup.names, the_screen.size_mm, seed)
    for onset in nuisances.onsets:
        if onset < timing.trusted_window:
            raise ValueError(f"a nuisance starts at {float(onset):g} s, inside the {float(timing.trusted_window):g} s "
                             "trusted window after calibration")
    return Scenario(name, variant or name, data, scene, timing, sequence, perturbations, reference, nuisances)
