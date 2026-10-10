"""Turning linear camera electrons into pictures a person can read, and saving them small.

Camera frames are linear: electrons in proportion to light. A projected picture spans about
1500:1 between the projector's black level and its white, and a plain display curve puts the
black level in the bottom few of 256 grey levels, where nobody can see it. So each figure picks
the curve that shows what it is about:

  natural   y = (x / white)^(1/2.2), roughly how the room looks to an eye;
  log       ``scripts.visualize.log_display``: content and the faint black-level raster at once;
  stretch   a linear window [lo, hi] of electrons, for the black level alone once the room
            light has been taken away.

Frames are shrunk by averaging blocks of pixels (INTER_AREA), which is what a coarser camera
would record. Difference maps are shrunk by each block's maximum instead: a change one pixel
wide would otherwise be averaged into invisibility.

Colour maps follow the site's palette: one hue from dark to light for magnitudes, two opposed
hues around a neutral grey for signed changes. Figures sit on a dark background in both site
themes, like photographs of a dim lecture hall.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scripts.visualize import log_display

# Palette (RGB), for figures on the dark figure background.
A = (217, 89, 38)  # projector A: orange
B = (57, 135, 229)  # projector B: blue
OVERLAP = (25, 158, 112)  # aqua
MARKER = (213, 81, 129)  # magenta
NEUTRAL = (56, 56, 53)  # the diverging midpoint on a dark figure
BRIGHTER = (230, 103, 103)  # red: brighter than before
DARKER = B  # blue: darker than before
BLUE_RAMP = ((0.0, (13, 13, 13)), (0.25, (24, 79, 149)), (0.55, (57, 135, 229)), (0.8, (134, 182, 239)),
             (1.0, (205, 226, 251)))


def natural(electrons: np.ndarray, white: float) -> np.ndarray:
    """8-bit display values with a 2.2 power curve, white = one projector's full white."""
    x = np.clip(electrons / np.float32(white), 0.0, 1.0)
    return np.rint(255.0 * x ** (1.0 / 2.2)).astype(np.uint8)


def log(electrons: np.ndarray, white: float, black: float) -> np.ndarray:
    """The visualize log curve: content and black level visible together."""
    return log_display(electrons / np.float32(white), black)


def stretch(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear window: lo -> 0, hi -> 255, clipped."""
    return np.clip(np.rint(255.0 * (values - lo) / (hi - lo)), 0, 255).astype(np.uint8)


def shrink(img: np.ndarray, width: int) -> np.ndarray:
    """Average blocks down to `width` pixels across (no change if already narrower)."""
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def shrink_max(img: np.ndarray, factor: int) -> np.ndarray:
    """Each factor x factor block's maximum: keeps thin features a block average would erase."""
    h, w = img.shape[:2]
    h2, w2 = h // factor, w // factor
    blocks = img[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor, *img.shape[2:])
    return blocks.max(axis=(1, 3))


def crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """img[v0:v1, u0:u1] for box (u0, v0, u1, v1), clamped to the image."""
    h, w = img.shape[:2]
    u0, v0, u1, v1 = box
    u0, u1 = max(0, min(w, u0)), max(0, min(w, u1))
    v0, v1 = max(0, min(h, v0)), max(0, min(h, v1))
    return img[v0:v1, u0:u1]


def rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[..., None], 3, axis=2) if gray.ndim == 2 else gray


def tint(gray: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """A colour laid over a grey picture with per-pixel opacity alpha in [0, 1]."""
    base = rgb(gray).astype(np.float32)
    a = np.clip(alpha, 0.0, 1.0)[..., None].astype(np.float32)
    return np.rint(base * (1 - a) + np.float32(color) * a).astype(np.uint8)


def ramp(values: np.ndarray, vmax: float, stops=BLUE_RAMP) -> np.ndarray:
    """One-hue magnitude map: 0 -> the darkest stop, vmax -> the lightest (piecewise linear)."""
    t = np.clip(np.asarray(values, dtype=np.float32) / np.float32(vmax), 0.0, 1.0)
    xs = [s[0] for s in stops]
    out = np.stack([np.interp(t, xs, [s[1][c] for s in stops]) for c in range(3)], axis=-1)
    return np.rint(out).astype(np.uint8)


def diverging(values: np.ndarray, vmax: float, neg=DARKER, pos=BRIGHTER, mid=NEUTRAL) -> np.ndarray:
    """Signed map: -vmax -> neg, 0 -> the neutral grey, +vmax -> pos."""
    t = np.clip(np.asarray(values, dtype=np.float32) / np.float32(vmax), -1.0, 1.0)[..., None]
    mid_c, neg_c, pos_c = (np.float32(c) for c in (mid, neg, pos))
    out = np.where(t < 0, mid_c + (neg_c - mid_c) * -t, mid_c + (pos_c - mid_c) * t)
    return np.rint(out).astype(np.uint8)


def sample_mm(image: np.ndarray, h_mm_to_px: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples of a camera image at the screen points (x, y) for x in xs, y in ys: (len(ys), len(xs))."""
    gx, gy = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    pts = np.stack([gx, gy, np.ones_like(gx)], axis=-1) @ np.asarray(h_mm_to_px).T
    u = (pts[..., 0] / pts[..., 2]).astype(np.float32)
    v = (pts[..., 1] / pts[..., 2]).astype(np.float32)
    src = np.ascontiguousarray(image, dtype=np.float32)
    return cv2.remap(src, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def save(path: Path, img: np.ndarray, quality: int = 88) -> dict:
    """Write a PNG or JPEG (by suffix) from RGB or grey; returns its size for the page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, buf = cv2.imencode(path.suffix, data, params)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())
    return {"w": int(img.shape[1]), "h": int(img.shape[0]), "bytes": len(buf)}
