"""Where things stand in the room, and how a gain screen sends each projector's light back.

Positions. The screen is the plane z = 0 in screen millimetres: x to the right and y down from
the screen's top-left corner, z out of the screen toward the room. Each projector's lens and the
camera's lens sit at a point (x, y, z) with z > 0. Nothing about the images depends on these
points: where a projector's pixels land, and where the camera sees them, are the homographies
(``sim/projector.py``, ``sim/camera.py``) that calibration measured. The points matter only for
the screen's gain, which depends on angles. They are facts about the installation that the
detector never reads (setup.json does not carry them).

Gain screens. A matte screen (``sim/screen.py``) returns light equally in every direction, so it
looks equally bright from every seat. A gain screen concentrates the light it returns: a silvered
or pearlescent surface returns most of it near the mirror direction (``specular``), a glass-bead
surface back toward where it came from (``retro``). Its gain G is how bright the screen looks to
the camera relative to a matte white screen under the same light, a function of the angle alpha
between the direction to the camera and the direction the screen sends the most light:

    G(alpha) = 1 + (peak - 1) exp(-(alpha / lobe_deg)^2 / 2),

a lobe in angle: ``peak`` on its axis, falling to 1 (matte) a few lobe widths away. For projector
p's light arriving at screen point X, the axis is the mirror image of the incoming ray, as if the
light came from P' = (px, py, -pz) behind the screen (specular), or the ray back to the projector
(retro). So the camera sees a bright hotspot for each projector: where the line from the camera to
P' crosses the screen, P_xy + (C_xy - P_xy) pz / (pz + cz) (specular), or where the line through
the camera and the projector does (retro). The projectors stand apart, so their hotspots do too,
and the overlap between them lies in the tails of both lobes, differently for A and for B.

Why the detector cares. The shading is static, so it is part of the calibrated baseline; but it
differs between the projectors across the overlap, which a hotspot fit that assumes a flat screen
must absorb (its lamp-gain and trend terms), and it changes the black-level steps the boundary
layer measures. ``scenarios/gain_screen.yaml`` asks whether either can mistake it for a shift.

The gain multiplies light where it is reflected: each projector's light meets its own
"reflectance toward the camera" map, the matte map plus (G - 1) times the screen's share of each
grid pixel times the screen's reflectance (``reflectance_toward_camera``), so the bezel, marker
paper and wall stay matte and a pixel on the screen's edge gains only for its screen part.

Simplifications. Room light and the bezel's own light arrive from every direction, so they keep
gain 1 (a real gain screen also rejects some room light arriving off axis). A knocked camera keeps
its position: a knock moves the image by pixels and the angles by hundredths of a degree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.calibration import CalibrationSetup
from sim.cfg import check_keys, choice, num
from sim.planar import centroid
from sim.screen import Screen, ScreenGrid

KINDS = ("specular", "retro")
THROW_RATIO = 1.6  # default projector distance, in widths of its image
CAMERA_BELOW = 1.25  # default camera height, in screen heights below the screen's top edge
CAMERA_DISTANCE = 1.5  # default camera distance, in screen widths
CHUNK_ROWS = 128  # grid rows per step when building a map (bounds the float64 temporaries)


@dataclass(frozen=True)
class ScreenGain:
    peak: float = 1.0  # gain on the lobe's axis; 1 is matte
    lobe_deg: float = 20.0  # angular width (one standard deviation) of the lobe
    kind: str = "specular"  # specular: around the mirror direction; retro: back toward the source

    def __post_init__(self) -> None:
        if not self.peak >= 1.0:
            raise ValueError(f"screen.gain.peak: must be at least 1 (1 is matte), got {self.peak}")
        if not self.lobe_deg > 0:
            raise ValueError(f"screen.gain.lobe_deg: must be positive, got {self.lobe_deg}")
        if self.kind not in KINDS:
            raise ValueError(f"screen.gain.kind: must be one of {list(KINDS)}, got {self.kind!r}")

    @property
    def active(self) -> bool:
        return self.peak != 1.0


@dataclass(frozen=True, eq=False)
class Room:
    projectors: dict[str, np.ndarray]  # each projector's lens, (x, y, z) in screen mm
    camera: np.ndarray  # the camera's lens, (x, y, z) in screen mm
    gain: ScreenGain | None = None  # None: a matte screen

    def __post_init__(self) -> None:
        for where, p in [*((f"projectors.{n}", p) for n, p in self.projectors.items()), ("camera", self.camera)]:
            if np.shape(p) != (3,) or not p[2] > 0:
                raise ValueError(f"{where}.position_mm: need [x, y, z] with z > 0 (in front of the screen)")

    @property
    def active(self) -> bool:
        """True if the screen is not matte: only then does any light depend on the positions."""
        return self.gain is not None and self.gain.active

    def gain_at(self, name: str, x_mm: Any, y_mm: Any) -> np.ndarray:
        """The screen's gain toward the camera for projector `name`'s light at screen points (x, y)."""
        x, y = np.broadcast_arrays(np.asarray(x_mm, dtype=np.float64), np.asarray(y_mm, dtype=np.float64))
        if not self.active:
            return np.ones(x.shape)
        p, c = self.projectors[name], self.camera
        cx, cy, cz = c[0] - x, c[1] - y, c[2]  # toward the camera
        sign = 1.0 if self.gain.kind == "specular" else -1.0
        rx, ry, rz = sign * (x - p[0]), sign * (y - p[1]), p[2]  # specular: away from P'; retro: back to P
        cross = np.sqrt((ry * cz - rz * cy) ** 2 + (rz * cx - rx * cz) ** 2 + (rx * cy - ry * cx) ** 2)
        alpha = np.degrees(np.arctan2(cross, rx * cx + ry * cy + rz * cz))  # stable for small angles
        return 1.0 + (self.gain.peak - 1.0) * np.exp(-0.5 * (alpha / self.gain.lobe_deg) ** 2)

    def hotspot_mm(self, name: str) -> np.ndarray | None:
        """Where on the screen's plane projector `name`'s light looks brightest to the camera (None: nowhere)."""
        if not self.active:
            return None
        p, c = self.projectors[name], self.camera
        denominator = p[2] + c[2] if self.gain.kind == "specular" else p[2] - c[2]
        if denominator == 0:
            return None  # retro, with camera and projector at the same distance: the line never meets the screen
        return p[:2] + (c[:2] - p[:2]) * (p[2] / denominator)


def default_positions(screen: Screen, setup: CalibrationSetup) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Each projector on the normal through its image's centre at 1.6 image widths; the camera below
    the screen (as the whole-screen camera's keystone implies) at 1.5 screen widths."""
    projectors = {}
    for name in setup.names:
        box = setup.box_mm(name)  # TL, TR, BR, BL
        x, y = centroid(box)
        projectors[name] = np.array([x, y, THROW_RATIO * float(np.hypot(*(box[1] - box[0])))])
    w, h = screen.size_mm
    return projectors, np.array([w / 2, CAMERA_BELOW * h, CAMERA_DISTANCE * w])


def position(value: Any, where: str) -> np.ndarray:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 3:
        raise ValueError(f"{where}: expected [x, y, z] in mm, got {value!r}")
    p = np.array([num(v, where) for v in value])
    if not p[2] > 0:
        raise ValueError(f"{where}: z must be positive (in front of the screen), got {p[2]:g}")
    return p


def gain_from_config(cfg: Any) -> ScreenGain | None:
    """Parse ``screen.gain``: {peak, lobe_deg (default 20), kind (default specular)}; absent or null is matte."""
    if cfg is None:
        return None
    if not isinstance(cfg, Mapping):
        raise ValueError("screen.gain: expected a mapping like {peak: 1.8, lobe_deg: 20, kind: specular}")
    check_keys(cfg, {"peak", "lobe_deg", "kind"}, "screen.gain")
    if "peak" not in cfg:
        raise ValueError("screen.gain: peak is required (1 is matte)")
    return ScreenGain(num(cfg["peak"], "screen.gain.peak"), num(cfg.get("lobe_deg", 20.0), "screen.gain.lobe_deg"),
                      choice(cfg.get("kind", "specular"), KINDS, "screen.gain.kind"))


def from_config(data: Mapping[str, Any], screen: Screen, setup: CalibrationSetup) -> Room:
    """The room of a scenario: ``projectors.<p>.position_mm``, ``camera.position_mm``, ``screen.gain``."""
    projectors, camera = default_positions(screen, setup)
    for name in setup.names:
        value = ((data.get("projectors") or {}).get(name) or {}).get("position_mm")
        if value is not None:
            projectors[name] = position(value, f"projectors.{name}.position_mm")
    value = (data.get("camera") or {}).get("position_mm")
    if value is not None:
        camera = position(value, "camera.position_mm")
    return Room(projectors, camera, gain_from_config((data.get("screen") or {}).get("gain")))


def reflectance_toward_camera(grid: ScreenGrid, screen: Screen, reflectance: np.ndarray, room: Room,
                              name: str) -> np.ndarray:
    """The reflectance map projector `name`'s light meets, as the camera sees it (float32, a new array).

    Where a grid pixel holds screen (area share c), the screen returns G times what a matte screen
    would toward the camera, so the pixel gains (G - 1) c rho_screen; bezel, paper and wall keep
    their reflectance. G is taken at each pixel's centre: it varies over centimetres, the pixel is
    a fraction of a millimetre.
    """
    out = reflectance.copy()
    rs, cs, cover = grid.rect_coverage((0.0, 0.0, *screen.size_mm))
    rows, cols = np.arange(rs.start, rs.stop), np.arange(cs.start, cs.stop)
    for k in range(0, len(rows), CHUNK_ROWS):
        r = rows[k:k + CHUNK_ROWS]
        x, y = grid.to_mm(r[:, None], cols[None, :])
        extra = (room.gain_at(name, x, y) - 1.0) * cover[k:k + len(r)] * screen.reflectance
        out[r[0]:r[-1] + 1, cs] += extra.astype(np.float32)
    return out
