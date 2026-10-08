"""Photo-like and synthetic test pictures: natural textures, dark film, stripes, letterbox, blank overlap.

Why each exists:

* Photo-like. Natural images have most of their contrast at coarse scales: their amplitude
  spectrum falls roughly as 1/f**1.2. The detector's echo test pools the log spectra of real content,
  so the simulator's "photos" are random fields with that spectrum, tone-mapped into a believable
  range, plus a few soft-edged objects (natural images also have edges). No external assets.
* Dark film. The same statistics mapped to low code values with a few highlights: a dim scene,
  where each projector's black level shows its raster edge (the boundary layer's dark frames).
* Stripes. A pure periodic pattern of known period and angle: the worst case for the echo test,
  because the cepstrum of a period-P pattern peaks at P exactly as a P-sized misalignment would.
  The grating is sinusoidal in *light* (the projector undoes the code's gamma), so it has one
  spatial frequency on the screen, and two copies half a period apart cancel where they are
  blended half and half.
* Letterbox. The picture between black bars: the content border no longer reaches the raster
  edge, so the boundary layer must find the edge through the black level alone.
* Blank overlap. Flat inside the overlap, textured elsewhere: nothing for the echo test to see,
  so the hotspot fit and the boundary must carry the check.

All functions return gamma-encoded code values in [0, 1] as (h, w, 3) float32, like every
content picture (``sim/content.py``).
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def pink_field(shape: tuple[int, int], rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    """Zero-mean, unit-variance random field whose amplitude spectrum falls as 1/f**beta."""
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1.0
    spectrum = (rng.standard_normal(f.shape) + 1j * rng.standard_normal(f.shape)) / f**beta
    spectrum[0, 0] = 0.0
    field = np.fft.irfft2(spectrum, s=(h, w))
    return (field / field.std()).astype(np.float32)


def photo(size: tuple[int, int], rng: np.random.Generator, mean: float = 0.45, contrast: float = 0.2,
          objects: int = 10) -> np.ndarray:
    """A photo-like picture: a 1/f luminance field with mild colour, plus soft-edged objects."""
    w, h = size
    lum = pink_field((h, w), rng, beta=1.2)  # natural images: amplitude ~ 1/f**1.0-1.4
    tint = pink_field((h, w), rng, beta=1.4)
    img = np.empty((h, w, 3), np.float32)
    for c, k in enumerate((0.06, 0.0, -0.06)):  # a warm/cool cast that varies across the picture
        img[..., c] = mean + contrast * lum + k * tint
    for _ in range(objects):
        _soft_ellipse(img, rng, value=rng.uniform(0.05, 0.95, 3).astype(np.float32))
    return np.clip(img, 0.0, 1.0)


def dark(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A dim film scene: low code values (about 0.6% of white in light on average), with a few highlights."""
    img = photo(size, rng, mean=0.10, contrast=0.05, objects=0)
    for _ in range(3):
        _soft_ellipse(img, rng, value=rng.uniform(0.4, 0.8, 3).astype(np.float32), max_frac=0.06)
    return img


def _soft_ellipse(img: np.ndarray, rng: np.random.Generator, value: np.ndarray, max_frac: float = 0.18) -> None:
    """Blend an ellipse of random place, size and angle into `img`, in place."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    ax, ay = rng.uniform(0.02, max_frac) * h, rng.uniform(0.02, max_frac) * h
    blend_ellipse(img, cx, cy, ax, ay, rng.uniform(0, math.pi), value)


def blend_ellipse(img: np.ndarray, cx: float, cy: float, ax: float, ay: float, angle: float,
                  colour: np.ndarray) -> None:
    """Blend an ellipse into `img` in place: centre (cx, cy), semi-axes (ax, ay) px turned by `angle`.

    The edge is soft over about one pixel across the short axis (an antialiased object edge).
    """
    h, w = img.shape[:2]
    reach = max(ax, ay)
    x0, x1 = int(max(0, cx - reach - 2)), int(min(w, cx + reach + 3))
    y0, y1 = int(max(0, cy - reach - 2)), int(min(h, cy + reach + 3))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    c, s = math.cos(angle), math.sin(angle)
    u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
    r = np.sqrt((u / ax) ** 2 + (v / ay) ** 2)
    alpha = np.clip((1.0 - r) * min(ax, ay) + 0.5, 0.0, 1.0)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - alpha) + colour * alpha


def stripes(size: tuple[int, int], period_px: float, angle_deg: float = 0.0, low: float = 0.15,
            high: float = 0.85, phase: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """A grating of `period_px` content pixels, sinusoidal in light; angle 0 = vertical stripes.

    `low` and `high` are the code values at the dark and bright crests; the light between them
    follows a cosine, encoded back to code values with `gamma`. A scenario passes projector A's
    gamma (``sim/pictures.py``): content is encoded once for both projectors, so if B's gamma
    differed the stripes would be sinusoidal in A's light only.
    """
    w, h = size
    t = math.radians(angle_deg)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    angle = 2 * math.pi * (xx * math.cos(t) + yy * math.sin(t)) / period_px + phase
    lo, hi = low**gamma, high**gamma
    light = lo + (hi - lo) * 0.5 * (1 + np.cos(angle))
    value = (light ** (1.0 / gamma)).astype(np.float32)
    return np.repeat(value[..., None], 3, axis=2)


def letterbox(picture: np.ndarray, bar_frac: float) -> np.ndarray:
    """Black bars of `bar_frac` of the height at the top and the bottom."""
    if not 0 <= bar_frac < 0.5:
        raise ValueError(f"letterbox: bar fraction must be in [0, 0.5), got {bar_frac}")
    out = picture.copy()
    bar = round(bar_frac * picture.shape[0])
    if bar:
        out[:bar] = 0.0
        out[-bar:] = 0.0
    return out


def blank_inside(picture: np.ndarray, polygon_px: np.ndarray, value: float = 0.5) -> np.ndarray:
    """Flat `value` inside a polygon (content pixel coordinates), with an anti-aliased edge."""
    h, w = picture.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon_px) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return (picture * (1 - alpha) + np.float32(value) * alpha).astype(np.float32)
