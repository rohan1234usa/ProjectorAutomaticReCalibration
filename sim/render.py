"""The renderer: content -> each projector's light on the screen -> what the camera records.

The whole physical chain in one place, in the order light travels:

1. The calibration software builds each projector's framebuffer from the content, using the
   geometry it measured at calibration (CalibrationSetup).
2. Each projector turns its framebuffer into light (gamma, blend weight, black level).
3. That light lands on the screen through the projector's *actual* geometry: the calibrated
   one when aligned, a different one after a drift. Light from both projectors adds in linear
   units on a millimetre grid of the screen, plus room light, times the screen's reflectance.
4. The camera photographs the screen.

Quality presets trade fidelity for speed. In ``standard`` the screen grid samples each
projector pixel 2 x 2 times and the camera integrates over 2 x 2 sub-pixels. ``fast`` uses
1.5 x 1.5 screen samples and a single camera sample, for tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from sim.calibration import CalibrationSetup
from sim.camera import Camera
from sim.projector import Projector, project
from sim.screen import Screen, ScreenGrid


@dataclass(frozen=True)
class Quality:
    screen_samples_per_px: float  # screen-grid samples per (finest) projector pixel, per axis
    camera_supersample: int  # camera sub-pixels per pixel, per axis


QUALITY = {
    "fast": Quality(screen_samples_per_px=1.5, camera_supersample=1),
    "standard": Quality(screen_samples_per_px=2.0, camera_supersample=2),
}


@dataclass(frozen=True, eq=False)
class RenderResult:
    frame: np.ndarray  # (h, w, 3) uint16 camera frame
    radiance: np.ndarray  # (rows, cols, 3) float32 screen radiance on `grid`
    grid: ScreenGrid


class Renderer:
    """Renders frames for one installation: screen, two projectors, their calibration, a camera."""

    def __init__(
        self,
        screen: Screen,
        projectors: Mapping[str, Projector],
        setup: CalibrationSetup,
        camera: Camera,
        quality: Quality,
    ) -> None:
        if set(projectors) != set(setup.names):
            raise ValueError(f"projectors {sorted(projectors)} do not match calibration {sorted(setup.names)}")
        for name, p in projectors.items():
            if tuple(p.resolution) != tuple(setup.resolution[name]):
                raise ValueError(f"projector {name}: resolution differs from the calibration setup")
        self.screen, self.projectors, self.setup = screen, dict(projectors), setup
        self.camera, self.quality = camera, quality
        pitch = min(setup.pixel_pitch_mm(n) for n in setup.names)
        self.grid = ScreenGrid.covering(screen.size_mm, quality.screen_samples_per_px / pitch)
        # The blend maps depend only on the calibration, so they are computed once.
        self.blend = {n: setup.blend_weights(n) for n in setup.names}

    def screen_irradiance(self, content: np.ndarray, h_actual: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Total projector irradiance on the screen grid, (rows, cols, 3) float32."""
        total = np.zeros((*self.grid.shape, 3), dtype=np.float32)
        for name in self.setup.names:
            light = self.projectors[name].emitted_light(self.setup.framebuffer(name, content), self.blend[name])
            h_cal = self.setup.h_cal[name]
            project(light, h_cal, (h_actual or {}).get(name, h_cal), self.grid, total)
        return total

    def screen_radiance(self, content: np.ndarray, h_actual: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        return self.screen.radiance(self.screen_irradiance(content, h_actual))

    def render(
        self,
        content: np.ndarray,
        rng: np.random.Generator,
        h_actual: Mapping[str, np.ndarray] | None = None,
    ) -> RenderResult:
        radiance = self.screen_radiance(content, h_actual)
        frame = self.camera.capture(
            radiance, self.grid, rng, supersample=self.quality.camera_supersample, surround=self.screen.surround
        )
        return RenderResult(frame=frame, radiance=radiance, grid=self.grid)
