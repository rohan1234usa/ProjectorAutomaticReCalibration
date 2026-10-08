"""The camera: a fixed, locked camera photographing the screen.

Physics, in the order light goes through it:

1. Geometry. A pinhole camera looking at a flat screen maps it to the image with a homography
   (mm to camera pixels). The ``whole_screen`` preset frames the screen with a 5% margin and a
   slight keystone (top edge a little narrower), as a camera mounted below the screen's centre
   would see it. The bezel and its markers fall inside that margin.
2. Optics. The lens blurs every point into a small spot, the point-spread function, modelled as
   a Gaussian of ``psf_sigma_px``. It also darkens the image towards the corners (vignetting,
   the cos^4 law: falloff (1 + a r^2)^-2). The blur happens before the sensor samples the image,
   and it is what keeps fine screen detail from aliasing. So part of it (up to 0.8 sub-pixel) is
   applied on the screen grid before resampling, and the rest in camera space. Gaussian
   variances add, so the total stays ``psf_sigma_px``; the screen-space part uses the local
   scale at the image centre, which keystone changes by only ~2% across the frame.
3. Pixels. Each sensor pixel collects all the light falling on its square. We render at k times
   the resolution and average k x k blocks, which is exactly that collection.
4. Photons to numbers. Light frees electrons in proportion to radiance; ``well_fill_at_white``
   is the fraction of the full well that radiance 1.0 (one projector's white on a perfect
   reflector) fills during one exposure. The count fluctuates randomly: shot noise has
   variance equal to the mean, and readout adds read noise. Electrons are scaled to 16-bit
   numbers on top of a fixed pedestal, so noise below zero is not clipped away. Exposure, gain
   and white balance are locked, so all of this stays constant over time.

Steps 1-3 and the electron scaling are linear in light: the image of a sum of radiances is the
sum of their images. The renderer relies on that to cache one image per light source and add
them per frame (``sim/render.py``). Noise, clipping and encoding are not linear and come last.
The camera is mono by default (16-bit linear luminance, CLAUDE.md decision 7); every function
here accepts (rows, cols) or (rows, cols, 3) radiance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np

from sim.planar import apply_h, homography_from_points, jacobian_det, warp_linear
from sim.screen import ScreenGrid

DN_MAX = 65535
COLORS = ("mono", "rgb")


@dataclass(frozen=True, eq=False)
class Camera:
    resolution: tuple[int, int]  # (width, height) in pixels
    h_mm_to_px: np.ndarray = field(repr=False)  # screen mm -> camera pixel coordinates
    psf_sigma_px: float = 0.8
    well_fill_at_white: float = 0.75  # fraction of full well reached by radiance 1.0 (one projector's white)
    full_well_e: float = 20000.0
    read_noise_e: float = 3.0
    pedestal_dn: int = 256
    vignetting: float = 0.15  # fraction of light lost in the image corners
    gamma: float = 1.0  # 1.0 = linear output (the dataset format); else output = signal**(1/gamma)
    color: str = "mono"  # mono: 16-bit luminance frames (h, w); rgb: (h, w, 3)

    def __post_init__(self) -> None:
        if min(self.resolution) < 1 or self.well_fill_at_white <= 0 or self.full_well_e <= 0 or self.gamma <= 0:
            raise ValueError("camera: bad resolution, well_fill_at_white, full well or gamma")
        if not 0 <= self.vignetting < 1 or self.psf_sigma_px < 0 or self.read_noise_e < 0:
            raise ValueError("camera: need 0 <= vignetting < 1, psf_sigma_px >= 0, read_noise_e >= 0")
        if not 0 <= self.pedestal_dn < DN_MAX:
            raise ValueError("camera: pedestal out of range")
        if self.color not in COLORS:
            raise ValueError(f"camera: color must be one of {COLORS}, got {self.color!r}")

    @classmethod
    def whole_screen(
        cls,
        screen_size_mm: tuple[float, float],
        resolution: tuple[int, int],
        margin: float = 0.05,
        keystone: float = 0.02,
        **optics: float | str,
    ) -> Camera:
        """Camera framing the whole screen plus `margin` (fraction of each side) on every side.

        `keystone` narrows the image of the screen's top edge by that fraction, about the
        vertical centre line, as seen from a camera below the screen's centre.
        """
        sw, sh = screen_size_mm
        cw, ch = resolution
        scale = min(cw / (sw * (1 + 2 * margin)), ch / (sh * (1 + 2 * margin)))
        half_w, half_h = sw * scale / 2, sh * scale / 2
        cx, cy = (cw - 1) / 2, (ch - 1) / 2
        top = half_w * (1 - keystone)
        image = np.array([[cx - top, cy - half_h], [cx + top, cy - half_h],
                          [cx + half_w, cy + half_h], [cx - half_w, cy + half_h]])
        screen = np.array([[0.0, 0.0], [sw, 0.0], [sw, sh], [0.0, sh]])
        return cls(resolution=resolution, h_mm_to_px=homography_from_points(screen, image), **optics)

    @property
    def gain_dn_per_e(self) -> float:
        return (DN_MAX - self.pedestal_dn) / self.full_well_e

    @property
    def electrons_per_unit_radiance(self) -> float:
        """Electrons a pixel at the image centre collects for radiance 1.0 (simulator-side only)."""
        return self.well_fill_at_white * self.full_well_e

    def px_per_mm_at_centre(self) -> float:
        """Camera pixels per screen millimetre at the image centre (local linear scale)."""
        w, h = self.resolution
        centre_mm = apply_h(np.linalg.inv(self.h_mm_to_px), np.array([(w - 1) / 2, (h - 1) / 2]))
        return float(np.sqrt(abs(jacobian_det(self.h_mm_to_px, centre_mm[0], centre_mm[1]))))

    def vignetting_map(self) -> np.ndarray:
        """Fraction of light reaching each pixel, (h, w) float32, read-only: 1 at the centre."""
        return self._vignetting

    @cached_property
    def _vignetting(self) -> np.ndarray:
        # The camera never changes, so the map is built once (it costs ~45 ms at 3840 x 1600).
        w, h = self.resolution
        x = (np.arange(w) - (w - 1) / 2)[None, :]
        y = (np.arange(h) - (h - 1) / 2)[:, None]
        r2 = (x**2 + y**2) / ((w / 2) ** 2 + (h / 2) ** 2)
        a = (1.0 - self.vignetting) ** -0.5 - 1.0  # so that the corners get 1 - vignetting
        vignetting = ((1.0 + a * r2) ** -2).astype(np.float32)
        vignetting.setflags(write=False)
        return vignetting

    def optical_image(self, radiance: np.ndarray, grid: ScreenGrid, supersample: int = 1, border: float = 0.0) -> np.ndarray:
        """Noiseless radiance each pixel records, float32: geometry, PSF, pixel area, vignetting.

        `border` is the radiance seen beyond the grid (the wall). The input is not modified.
        """
        k = int(supersample)
        if k < 1:
            raise ValueError("supersample must be a positive integer")
        w, h = self.resolution
        # Anti-aliasing share of the lens blur, applied on the screen grid before resampling.
        sigma_pre = min(self.psf_sigma_px, 0.8 / k)  # camera pixels
        sigma_pre_grid = sigma_pre * grid.px_per_mm / self.px_per_mm_at_centre()
        if sigma_pre_grid >= 0.25:
            radiance = cv2.GaussianBlur(radiance, (0, 0), sigma_pre_grid)
        else:
            sigma_pre = 0.0  # grid coarser than the sub-pixels: resampling cannot alias
        # Camera pixel (u, v) covers sub-pixels k*u .. k*u + k-1; its centre is k*u + (k-1)/2.
        to_sub = np.array([[k, 0.0, (k - 1) / 2], [0.0, k, (k - 1) / 2], [0.0, 0.0, 1.0]])
        grid_to_sub = to_sub @ self.h_mm_to_px @ np.linalg.inv(grid.mm_to_grid)
        img = warp_linear(radiance, grid_to_sub, (k * w, k * h), border=border)
        sigma_rest = np.sqrt(max(self.psf_sigma_px**2 - sigma_pre**2, 0.0))
        if sigma_rest > 0:
            img = cv2.GaussianBlur(img, (0, 0), sigma_rest * k)
        if k > 1:
            assert img.shape[:2] == (k * h, k * w)  # INTER_AREA is an exact block mean only then
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        v = self.vignetting_map()
        img *= v if img.ndim == 2 else v[..., None]
        return img

    def expected_electrons(
        self, radiance: np.ndarray, grid: ScreenGrid, supersample: int = 1, border: float = 0.0
    ) -> np.ndarray:
        img = self.optical_image(radiance, grid, supersample, border)
        img *= np.float32(self.electrons_per_unit_radiance)
        return img

    def add_noise(self, electrons: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Shot noise (variance = mean) plus read noise, as a new array. Draws one normal per value."""
        noise = rng.standard_normal(electrons.shape, dtype=np.float32)
        sigma = np.sqrt(np.maximum(electrons, 0.0) + np.float32(self.read_noise_e**2))
        return electrons + sigma * noise

    def capture(
        self, radiance: np.ndarray, grid: ScreenGrid, rng: np.random.Generator, supersample: int = 1, border: float = 0.0
    ) -> np.ndarray:
        """One frame from a screen radiance: uint16, with shot and read noise."""
        return self.encode(self.add_noise(self.expected_electrons(radiance, grid, supersample, border), rng))

    def encode(self, electrons: np.ndarray) -> np.ndarray:
        """Electrons -> 16-bit numbers (linear with pedestal if gamma == 1, else gamma-encoded)."""
        if self.gamma == 1.0:
            dn = self.pedestal_dn + electrons * np.float32(self.gain_dn_per_e)
        else:
            dn = DN_MAX * np.clip(electrons / np.float32(self.full_well_e), 0.0, 1.0) ** np.float32(1.0 / self.gamma)
        return np.clip(np.rint(dn), 0, DN_MAX).astype(np.uint16)

    def decode(self, frame: np.ndarray) -> np.ndarray:
        """16-bit numbers -> electrons (inverse of `encode`, up to quantisation and clipping)."""
        dn = frame.astype(np.float32)
        if self.gamma == 1.0:
            return (dn - np.float32(self.pedestal_dn)) / np.float32(self.gain_dn_per_e)
        return np.float32(self.full_well_e) * (dn / DN_MAX) ** np.float32(self.gamma)

    def setup_dict(self) -> dict:
        """Locked camera settings the detector knows (CLAUDE.md 4.1): never the pose or the optics."""
        return {
            "resolution": [int(v) for v in self.resolution],
            "color": self.color,
            "bit_depth": 16,
            "gamma": float(self.gamma),
            "pedestal_dn": int(self.pedestal_dn),
            "gain_dn_per_e": float(self.gain_dn_per_e),
            "full_well_e": float(self.full_well_e),
            "read_noise_e": float(self.read_noise_e),
        }
