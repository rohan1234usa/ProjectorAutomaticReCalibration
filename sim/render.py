"""The renderer: content -> each projector's light on the screen -> what the camera records.

The whole physical chain in one place, in the order light travels:

1. The calibration software builds each projector's framebuffer from the content, using the
   geometry it measured at calibration (CalibrationSetup).
2. Each projector turns its framebuffer into light (gamma, blend weight, black level).
3. That light lands on the screen through the projector's *actual* geometry: the calibrated
   one when aligned, a different one after a drift. Light from both projectors adds in linear
   units on a millimetre grid of the screen and bezel, together with the room light and the
   bezel's own lamp, times the static reflectance map (screen, bezel, marker paper and ink).
4. The camera photographs it; beyond the grid it sees the wall.

Components. Steps 3 and 4 are linear in light, so the camera image of the whole scene is the
sum of the camera images of each light source alone, each times its strength:

    E = ambient * E_room + lamp * E_lamp + sum_p gain_p * E_p

E_room is the image of the surfaces under unit room light (with the wall around them), E_lamp
under the bezel's unit lamp, and E_p under projector p's light alone. Each component is rendered
once and reused until what it depends on changes. A held slide then costs one render plus noise
per frame, and a lamp dimming or a room-light step costs nothing. Components are always summed
by :func:`compose` in a fixed order, so a frame is bit-identical however it was reached. Noise,
clipping and encoding are not linear and come after the sum.

Quality presets trade fidelity for speed. In ``standard`` the screen grid samples each
projector pixel 2 x 2 times and the camera integrates over 2 x 2 sub-pixels. ``fast`` uses
1.5 x 1.5 screen samples and a single camera sample, for tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sim.calibration import CalibrationSetup
from sim.camera import Camera
from sim.fiducials import MarkerSet, paint
from sim.projector import Projector, project
from sim.screen import Screen, ScreenGrid, surfaces


@dataclass(frozen=True)
class Quality:
    screen_samples_per_px: float  # screen-grid samples per (finest) projector pixel, per axis
    camera_supersample: int  # camera sub-pixels per pixel, per axis


QUALITY = {
    "fast": Quality(screen_samples_per_px=1.5, camera_supersample=1),
    "standard": Quality(screen_samples_per_px=2.0, camera_supersample=2),
}


def readonly(a: np.ndarray) -> np.ndarray:
    """Freeze a cached array, so no later step can change it in place by mistake."""
    a.setflags(write=False)
    return a


def compose(terms: Sequence[tuple[float, np.ndarray]]) -> np.ndarray:
    """Weighted sum of component images in the given order: the one way frames are summed.

    A weight of exactly 1 is added without multiplying (x * 1 == x in floating point), which
    makes the common case six times faster without changing a single bit.
    """
    weight, image = terms[0]
    out = image * np.float32(weight)
    for weight, image in terms[1:]:
        if weight == 1.0:
            out += image
        else:
            out += np.float32(weight) * image
    return out


class Renderer:
    """Renders one installation: screen with bezel and markers, two projectors, their calibration, a camera."""

    def __init__(
        self,
        screen: Screen,
        projectors: Mapping[str, Projector],
        setup: CalibrationSetup,
        camera: Camera,
        quality: Quality,
        markers: MarkerSet | None = None,
    ) -> None:
        if set(projectors) != set(setup.names):
            raise ValueError(f"projectors {sorted(projectors)} do not match calibration {sorted(setup.names)}")
        for name, p in projectors.items():
            if tuple(p.resolution) != tuple(setup.resolution[name]):
                raise ValueError(f"projector {name}: resolution differs from the calibration setup")
        self.screen, self.projectors, self.setup = screen, dict(projectors), setup
        self.camera, self.quality, self.markers = camera, quality, markers
        self.mono = camera.color == "mono"
        pitch = min(setup.pixel_pitch_mm(n) for n in setup.names)
        self.grid = ScreenGrid.covering_extent(screen.extent_mm, quality.screen_samples_per_px / pitch)
        # The blend maps and the surfaces depend only on the installation, so they are built once.
        self.blend = {n: setup.blend_weights(n) for n in setup.names}
        refl, lamp = surfaces(self.grid, screen)
        if markers is not None:
            paint(refl, self.grid, markers, screen.bezel.reflectance)
        self.reflectance = readonly(refl)
        self.lamp_mask = readonly(lamp)

    def _per_pixel(self, a: np.ndarray) -> np.ndarray:
        """A grid map shaped to multiply a light image (with a channel axis when rendering RGB)."""
        return a if self.mono else a[..., None]

    # -- light --------------------------------------------------------------------------------
    def projector_light(self, name: str, content: np.ndarray) -> np.ndarray:
        """Light leaving each pixel of projector `name` for this content (before any lamp gain)."""
        framebuffer = self.setup.framebuffer(name, content)
        return self.projectors[name].emitted_light(framebuffer, self.blend[name], mono=self.mono)

    def projector_irradiance(self, name: str, light: np.ndarray, h_actual: np.ndarray | None = None) -> np.ndarray:
        """That light on the screen grid, landing through `h_actual` (default: calibrated)."""
        h_cal = self.setup.h_cal[name]
        out = np.zeros(self.grid.shape if self.mono else (*self.grid.shape, 3), dtype=np.float32)
        project(light, h_cal, h_cal if h_actual is None else h_actual, self.grid, out)
        return out

    def screen_irradiance(self, content: np.ndarray, h_actual: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Total projector irradiance on the screen grid (for physics checks)."""
        total = None
        for name in self.setup.names:
            irr = self.projector_irradiance(name, self.projector_light(name, content), (h_actual or {}).get(name))
            total = irr if total is None else total + irr
        return total

    def screen_radiance(self, content: np.ndarray, h_actual: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Radiance of every grid pixel: reflectance x (room light + bezel lamp + projectors)."""
        irr = self.screen_irradiance(content, h_actual)
        irr += np.float32(self.screen.ambient)
        if self.screen.bezel.light > 0:
            irr += np.float32(self.screen.bezel.light) * self._per_pixel(self.lamp_mask)
        irr *= self._per_pixel(self.reflectance)
        return irr

    # -- camera components: expected electrons ------------------------------------------------
    def projector_electrons(self, name: str, light: np.ndarray, h_actual: np.ndarray | None, camera: Camera) -> np.ndarray:
        """Camera image of projector `name`'s light alone (read-only)."""
        irr = self.projector_irradiance(name, light, h_actual)
        irr *= self._per_pixel(self.reflectance)
        return readonly(camera.expected_electrons(irr, self.grid, self.quality.camera_supersample))

    def room_electrons(self, camera: Camera) -> np.ndarray:
        """Camera image of the surfaces and the wall under unit room light (read-only)."""
        rad = self.reflectance if self.mono else np.repeat(self.reflectance[..., None], 3, axis=2)
        k = self.quality.camera_supersample
        return readonly(camera.expected_electrons(rad, self.grid, k, border=self.screen.wall_reflectance))

    def lamp_electrons(self, camera: Camera) -> np.ndarray:
        """Camera image of the bezel under its own unit lamp (read-only)."""
        rad = self.reflectance * self.lamp_mask
        rad = rad if self.mono else np.repeat(rad[..., None], 3, axis=2)
        return readonly(camera.expected_electrons(rad, self.grid, self.quality.camera_supersample))

    def expected_electrons(self, content: np.ndarray, h_actual: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Noiseless frame of one still content image, summed from its components."""
        terms = [(self.screen.ambient, self.room_electrons(self.camera))]
        if self.screen.bezel.light > 0:
            terms.append((self.screen.bezel.light, self.lamp_electrons(self.camera)))
        for name in self.setup.names:
            light = self.projector_light(name, content)
            terms.append((1.0, self.projector_electrons(name, light, (h_actual or {}).get(name), self.camera)))
        return compose(terms)

    def render(
        self, content: np.ndarray, rng: np.random.Generator, h_actual: Mapping[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """One noisy uint16 camera frame of a still content image."""
        return self.camera.encode(self.camera.add_noise(self.expected_electrons(content, h_actual), rng))
