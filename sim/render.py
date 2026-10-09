"""The renderer: content -> each projector's light on the screen -> what the camera records.

The whole physical chain in one place, in the order light travels:

1. The calibration software builds each projector's framebuffer from the content, using the
   geometry it measured at calibration (CalibrationSetup).
2. Each projector turns its framebuffer into light (gamma, blend weight, black level).
3. That light lands on the screen through the projector's *actual* geometry: the calibrated
   one when aligned, a different one after a drift. Light from both projectors adds in linear
   units on a millimetre grid of the screen and bezel, together with the room light and the
   bezel's own light, times the static reflectance map (screen, bezel, marker paper and ink).
4. The camera photographs it; beyond the grid it sees the wall.

Components. Steps 3 and 4 are linear in light, so the camera image of the whole scene is the
sum of the camera images of each light source alone, each times its strength:

    E = ambient * E_room + bezel_light * E_bezel + sum_p gain_p * E_p

E_room is the image of the surfaces under unit room light (with the wall around them), E_bezel
under the bezel's own unit light, and E_p under projector p's light alone (gain_p: its lamp). On a
gain screen E_p also carries the screen's gain toward the camera for p's light (``sim/room.py``):
p's light meets its own reflectance map, and the room and bezel light the matte one.
The Renderer draws these components; ``sim/frames.py`` caches each one until what it depends on
changes, so a held slide costs one render plus noise per frame, and a lamp dimming or a
room-light step costs nothing. Components are always summed by :func:`compose` in a fixed order,
so a frame is bit-identical however it was reached. Noise, clipping and encoding are not linear
and come after the sum.

Quality presets trade fidelity for speed. In ``standard`` the screen grid samples each
projector pixel 2 x 2 times and the camera integrates over 2 x 2 sub-pixels. ``fast`` uses
1.5 x 1.5 screen samples and a single camera sample, for tests. ``fine`` uses 3 x 3 of each, to
check that a result does not depend on the sampling: 2.25 times the screen grid of ``standard``
(64 rather than 28 Mpx at demo scale), and a new picture takes about 1.5 times as long.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from sim.calibration import CalibrationSetup
from sim.camera import Camera
from sim.fiducials import MarkerSet, paint
from sim.projector import Projector, project
from sim.room import Room, reflectance_toward_camera
from sim.screen import Screen, ScreenGrid, surfaces


@dataclass(frozen=True)
class Quality:
    screen_samples_per_px: float  # screen-grid samples per (finest) projector pixel, per axis
    camera_supersample: int  # camera sub-pixels per pixel, per axis


QUALITY = {
    "fast": Quality(screen_samples_per_px=1.5, camera_supersample=1),
    "standard": Quality(screen_samples_per_px=2.0, camera_supersample=2),
    "fine": Quality(screen_samples_per_px=3.0, camera_supersample=3),
}


def readonly(a: np.ndarray) -> np.ndarray:
    """Freeze a cached array, so no later step can change it in place by mistake."""
    a.setflags(write=False)
    return a


def _scaled(weight: float | np.ndarray, image: np.ndarray) -> np.ndarray:
    w = np.float32(weight) if np.isscalar(weight) else np.asarray(weight, dtype=np.float32)
    return w * image


def compose(terms: Sequence[tuple[float | np.ndarray, np.ndarray]]) -> np.ndarray:
    """Weighted sum of component images in the given order: the one way frames are summed.

    A weight is a number, or an array broadcast over the image (a per-row flicker gain). A weight
    of exactly 1 is added without multiplying (x * 1 == x in floating point), which makes the
    common case six times faster without changing a single bit.
    """
    weight, image = terms[0]
    out = _scaled(weight, image)
    for weight, image in terms[1:]:
        if np.isscalar(weight) and weight == 1.0:
            out += image
        else:
            out += _scaled(weight, image)
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
        room: Room | None = None,
    ) -> None:
        if set(projectors) != set(setup.names):
            raise ValueError(f"projectors {sorted(projectors)} do not match calibration {sorted(setup.names)}")
        for name, p in projectors.items():
            if tuple(p.resolution) != tuple(setup.resolution[name]):
                raise ValueError(f"projector {name}: resolution differs from the calibration setup")
        self.screen, self.projectors, self.setup = screen, dict(projectors), setup
        self.camera, self.quality, self.markers, self.room = camera, quality, markers, room
        self.mono = camera.color == "mono"
        pitch = setup.finest_pitch_mm()
        self.grid = ScreenGrid.covering_extent(screen.extent_mm, quality.screen_samples_per_px / pitch)
        # The blend maps and the surfaces depend only on the installation, so they are built once.
        self.blend = {n: setup.blend_weights(n) for n in setup.names}
        self.uplift = {}  # extra light per projector pixel when the software lifts single-coverage black
        if setup.black_uplift:
            a, b = setup.names
            for name, partner in ((a, b), (b, a)):
                level = projectors[partner].black_level * projectors[partner].brightness
                self.uplift[name] = readonly(setup.uplift_mask(name) * np.float32(level))
        refl, bezel = surfaces(self.grid, screen)
        if markers is not None:
            paint(refl, self.grid, markers, screen.bezel.reflectance)
        self.reflectance = readonly(refl)
        self.bezel_mask = readonly(bezel)
        self._reflectance_for: dict[str, np.ndarray] = {}  # per projector, on a gain screen only
        # Reused between frames (values never carry over): a zeroed grid that each projector's
        # light is drawn into and cleared from, each projector's warp window (A's and B's differ
        # in shape, so each keeps its own), and the camera's large temporaries.
        self._grid_buffer: np.ndarray | None = None
        self._project_workspace: dict[str, dict] = {n: {} for n in setup.names}
        self._workspace: dict = {}

    def reflectance_for(self, name: str) -> np.ndarray:
        """The reflectance map projector `name`'s light meets, as the camera sees it: the matte map itself
        unless the screen has gain, then the map with the screen's gain for that projector (built once)."""
        if self.room is None or not self.room.active:
            return self.reflectance
        if name not in self._reflectance_for:
            self._reflectance_for[name] = readonly(
                reflectance_toward_camera(self.grid, self.screen, self.reflectance, self.room, name))
        return self._reflectance_for[name]

    def _per_pixel(self, a: np.ndarray) -> np.ndarray:
        """A grid map shaped to multiply a light image (with a channel axis when rendering RGB)."""
        return a if self.mono else a[..., None]

    # -- light --------------------------------------------------------------------------------
    def projector_light(self, name: str, content: np.ndarray) -> np.ndarray:
        """Light leaving each pixel of projector `name` for this content (before any lamp gain)."""
        framebuffer = self.setup.framebuffer(name, content)
        light = self.projectors[name].emitted_light(framebuffer, self.blend[name], mono=self.mono)
        if name in self.uplift:
            light += self.uplift[name] if self.mono else self.uplift[name][..., None]
        return light

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
        """Radiance of every grid pixel: reflectance x (room light + bezel light + projectors).

        On a gain screen each projector's light meets its own reflectance map (``reflectance_for``).
        """
        if self.room is not None and self.room.active:
            rad = None
            for name in self.setup.names:
                irr = self.projector_irradiance(name, self.projector_light(name, content), (h_actual or {}).get(name))
                irr *= self._per_pixel(self.reflectance_for(name))
                rad = irr if rad is None else rad + irr
            diffuse = np.float32(self.screen.ambient) * self.reflectance  # room and bezel light: no gain
            if self.screen.bezel.light > 0:
                diffuse += np.float32(self.screen.bezel.light) * self.bezel_mask * self.reflectance
            rad += self._per_pixel(diffuse)
            return rad
        irr = self.screen_irradiance(content, h_actual)
        irr += np.float32(self.screen.ambient)
        if self.screen.bezel.light > 0:
            irr += np.float32(self.screen.bezel.light) * self._per_pixel(self.bezel_mask)
        irr *= self._per_pixel(self.reflectance)
        return irr

    # -- camera components: expected electrons ------------------------------------------------
    def projector_electrons(self, name: str, light: np.ndarray, h_actual: np.ndarray | None, camera: Camera) -> np.ndarray:
        """Camera image of projector `name`'s light alone (read-only, a new array).

        The light is drawn into a persistent zeroed grid, times the reflectance where it landed
        (zero times reflectance is zero elsewhere), photographed, and wiped again.
        """
        if self._grid_buffer is None:
            self._grid_buffer = np.zeros(self.grid.shape if self.mono else (*self.grid.shape, 3), dtype=np.float32)
        irr = self._grid_buffer
        h_cal = self.setup.h_cal[name]
        window = project(light, h_cal, h_cal if h_actual is None else h_actual, self.grid, irr,
                         self._project_workspace[name])
        try:
            if window is not None:
                r0, r1, c0, c1 = window
                irr[r0:r1, c0:c1] *= self._per_pixel(self.reflectance_for(name)[r0:r1, c0:c1])
            electrons = camera.expected_electrons(irr, self.grid, self.quality.camera_supersample,
                                                  workspace=self._workspace)
        finally:
            if window is not None:
                irr[r0:r1, c0:c1] = 0.0
        return readonly(electrons)

    def room_electrons(self, camera: Camera) -> np.ndarray:
        """Camera image of the surfaces and the wall under unit room light (read-only)."""
        rad = self.reflectance if self.mono else np.repeat(self.reflectance[..., None], 3, axis=2)
        k = self.quality.camera_supersample
        return readonly(camera.expected_electrons(rad, self.grid, k, border=self.screen.wall_reflectance,
                                                  workspace=self._workspace))

    def bezel_light_electrons(self, camera: Camera) -> np.ndarray:
        """Camera image of the bezel under its own unit light (read-only)."""
        rad = self.reflectance * self.bezel_mask
        rad = rad if self.mono else np.repeat(rad[..., None], 3, axis=2)
        return readonly(camera.expected_electrons(rad, self.grid, self.quality.camera_supersample,
                                                  workspace=self._workspace))
