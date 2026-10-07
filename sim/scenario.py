"""Scenarios: one YAML file per test idea, turned into the objects that render it.

Scenarios are declarative so that adding a test means adding a YAML file, not code. Phase 1
reads the static part of a scenario and builds a Scene that renders one frame. That part is:
screen, arrangement, projectors, blend, content, camera, quality and seed. Phase 2 adds time,
perturbations, nuisances and per-frame ground truth on top of the same keys.

Randomness: every random draw comes from the scenario seed through numpy SeedSequence spawn
keys (content: (0,), frame i: (1, i)), so any frame can be re-rendered on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sim.calibration import CalibrationSetup
from sim.camera import Camera
from sim.content import make_content
from sim.planar import homography_from_points, raster_corners
from sim.projector import Projector, side_by_side
from sim.render import QUALITY, Quality, Renderer
from sim.screen import Screen

_TOP_KEYS = {"name", "description", "seed", "quality", "screen", "arrangement", "projectors", "blend", "content", "camera"}
_PROJECTOR_KEYS = {"resolution", "gamma", "brightness", "black_level", "color_balance"}
_CAMERA_OPTICS = {"psf_sigma_px", "exposure", "full_well_e", "read_noise_e", "pedestal_dn", "vignetting", "gamma"}


@dataclass(frozen=True, eq=False)
class Scene:
    name: str
    screen: Screen
    projectors: dict[str, Projector]
    setup: CalibrationSetup
    camera: Camera
    content: dict[str, Any]
    quality: Quality
    seed: int

    def renderer(self) -> Renderer:
        return Renderer(self.screen, self.projectors, self.setup, self.camera, self.quality)

    def content_image(self) -> np.ndarray:
        rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(0,)))
        return make_content(self.content, self.setup.content_size(), rng)

    def frame_rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(1, index)))


def load_scene(path: str | Path, quality: str | None = None) -> Scene:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return scene_from_dict(data, quality=quality, default_name=Path(path).stem)


def scene_from_dict(data: Mapping[str, Any], quality: str | None = None, default_name: str = "scene") -> Scene:
    _check_keys(data, _TOP_KEYS, "scenario")
    screen_cfg = dict(data["screen"])
    _check_keys(screen_cfg, {"size_mm", "reflectance", "ambient", "surround"}, "screen")
    screen = Screen(size_mm=_pair(screen_cfg.pop("size_mm"), float), **screen_cfg)

    projectors = {}
    for name, cfg in dict(data["projectors"]).items():
        _check_keys(cfg, _PROJECTOR_KEYS, f"projectors.{name}")
        cfg = dict(cfg)
        resolution = _pair(cfg.pop("resolution"), int)
        if "color_balance" in cfg:
            cfg["color_balance"] = tuple(float(c) for c in cfg["color_balance"])
        projectors[name] = Projector(name=name, resolution=resolution, **cfg)
    if len(projectors) != 2:
        raise ValueError("scenario: exactly two projectors are supported")

    corners, content_rect = _arrangement(dict(data["arrangement"]), screen, projectors)
    blend = dict(data.get("blend", {}))
    _check_keys(blend, {"shape"}, "blend")
    setup = CalibrationSetup(
        h_cal={n: homography_from_points(raster_corners(p.resolution), corners[n]) for n, p in projectors.items()},
        resolution={n: p.resolution for n, p in projectors.items()},
        content_rect_mm=content_rect,
        blend_shape=blend.get("shape", "cosine"),
    )

    camera_cfg = dict(data["camera"])
    _check_keys(camera_cfg, {"preset", "resolution", "margin", "keystone"} | _CAMERA_OPTICS, "camera")
    if camera_cfg.pop("preset", "whole_screen") != "whole_screen":
        raise ValueError("camera: only the whole_screen preset exists in Phase 1")
    camera = Camera.whole_screen(screen.size_mm, _pair(camera_cfg.pop("resolution"), int), **camera_cfg)

    quality_name = quality or data.get("quality", "standard")
    if quality_name not in QUALITY:
        raise ValueError(f"quality must be one of {sorted(QUALITY)}, got {quality_name!r}")
    return Scene(
        name=str(data.get("name", default_name)),
        screen=screen,
        projectors=projectors,
        setup=setup,
        camera=camera,
        content=dict(data["content"]),
        quality=QUALITY[quality_name],
        seed=int(data.get("seed", 0)),
    )


def _arrangement(
    cfg: dict[str, Any], screen: Screen, projectors: Mapping[str, Projector]
) -> tuple[dict[str, np.ndarray], tuple[float, float, float, float]]:
    """Expand an arrangement preset into each projector's box corners and the content rect."""
    preset = cfg.pop("preset", None)
    if preset == "side_by_side":
        _check_keys(cfg, {"width_mm", "width_a_mm", "width_b_mm", "overlap_mm", "vertical_offset_mm"}, "arrangement")
        a, b = projectors
        width = cfg.get("width_mm")
        box_a, box_b, rect = side_by_side(
            screen.size_mm,
            projectors[a].resolution,
            projectors[b].resolution,
            float(cfg.get("width_a_mm", width)),
            float(cfg.get("width_b_mm", width)),
            float(cfg["overlap_mm"]),
            float(cfg.get("vertical_offset_mm", 0.0)),
        )
        return {a: box_a, b: box_b}, rect
    if preset == "explicit":
        _check_keys(cfg, {"corners_mm", "content_rect_mm"}, "arrangement")
        corners = {n: np.asarray(c, dtype=np.float64) for n, c in cfg["corners_mm"].items()}
        if set(corners) != set(projectors) or any(c.shape != (4, 2) for c in corners.values()):
            raise ValueError("arrangement.corners_mm: 4 corners (TL, TR, BR, BL) for each projector")
        x0, y0, x1, y1 = (float(v) for v in cfg["content_rect_mm"])
        return corners, (x0, y0, x1, y1)
    raise ValueError(f"arrangement.preset must be side_by_side or explicit, got {preset!r}")


def _check_keys(cfg: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")


def _pair(value: Any, kind: type) -> tuple[Any, Any]:
    first, second = value
    return kind(first), kind(second)
