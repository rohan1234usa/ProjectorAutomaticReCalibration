"""Content: the still pictures the calibration software is asked to show.

Content is authored like any video signal. It is an RGB image of gamma-encoded code values in
[0, 1] that fills the content rect; the projector's gamma later turns codes into light. This
module provides:
  * flat fields (gray or colour), to test that the blend is seamless;
  * black, to see the black level;
  * procedural text slides at three text densities (low, medium, high), with graphics.

Seams and ghosts only show where content crosses the overlap, so test content must cross it.
A slide's text lines and its flat gray band span the whole width, so they cross an overlap of
any shape. The charts are laid out for the side-by-side demo: a bar chart near each side edge
and a line chart in the middle third, where that overlap is. Every feature is drawn as obvious
slide content (text, a framed chart, bars): a bare diagonal line across a slide was once
mistaken for a projector boundary. Density changes how much text there is and how small it is:
dense text is repetitive, fine detail, the hardest case for the echo test.

An optional black border (:func:`framed`, ``border_frac``) keeps the picture away from the
raster edges, so each projector's faint black-level raster shows around it.

All drawing happens on 8-bit canvases, because OpenCV only antialiases text and lines there.
The result is then scaled to [0, 1].
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WORDS = (
    "alignment blend overlap screen lens zoom focus throw keystone raster pixel gamma lumen "
    "contrast black level camera fiducial homography offset drift warm thermal lamp laser "
    "calibration baseline schedule venue stage audience seat aisle balcony rehearsal show cue "
    "playback signal cable frame rate refresh scaler edge corner tile grid content slide video "
    "sharp soft bright dim uniform color white point shift rotation scale tilt check report"
).split()
# Body text per density: number of lines, capital height and first baseline / line pitch (in % of
# the slide height). All densities leave the gray band (55-62%) and the charts (66-94%) in place.
DENSITIES = {
    "low": (4, 4.2, 24.0, 8.5),
    "medium": (6, 2.8, 21.0, 6.0),
    "high": (12, 1.9, 17.0, 3.2),
}


def flat(size: tuple[int, int], value: float | tuple[float, float, float] = 0.5) -> np.ndarray:
    w, h = size
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = np.asarray(value, dtype=np.float32)
    return img


def black(size: tuple[int, int]) -> np.ndarray:
    return flat(size, 0.0)


def _text_scale(height_px: float, thickness: int = 1) -> float:
    (_, cap), _ = cv2.getTextSize("H", _FONT, 1.0, thickness)
    return height_px / cap


def _put(img: np.ndarray, text: str, org: tuple[float, float], height: float, color: tuple[int, int, int]) -> None:
    """Draw text whose capital letters are `height` px tall, baseline at `org`."""
    scale = _text_scale(height)
    thickness = max(1, round(scale * 1.4))
    cv2.putText(img, text, (round(org[0]), round(org[1])), _FONT, scale, color, thickness, cv2.LINE_AA)


def _line_of_words(rng: np.random.Generator, width_px: float, height: float) -> str:
    scale = _text_scale(height)
    thickness = max(1, round(scale * 1.4))
    words: list[str] = []
    while True:
        candidate = " ".join([*words, str(rng.choice(_WORDS))])
        if words and cv2.getTextSize(candidate, _FONT, scale, thickness)[0][0] > width_px:
            return " ".join(words)
        words = candidate.split(" ")


def slide(size: tuple[int, int], rng: np.random.Generator, density: str = "medium") -> np.ndarray:
    """A presentation slide at text density low/medium/high; its text and gray band span the width."""
    if density not in DENSITIES:
        raise ValueError(f"slide density must be one of {sorted(DENSITIES)}, got {density!r}")
    lines, text_h, first, pitch = DENSITIES[density]
    w, h = size
    u = h / 100.0  # layout unit: 1% of the slide height
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = (235, 233, 228)  # warm white background
    img[: round(12 * u)] = (46, 82, 140)  # title bar
    title = "Projector Alignment Review - Main Hall"
    scale = _text_scale(5.0 * u)
    tw = cv2.getTextSize(title, _FONT, scale, max(1, round(scale * 1.4)))[0][0]
    _put(img, title, ((w - tw) / 2, 8.5 * u), 5.0 * u, (250, 250, 250))
    for i in range(lines):  # body text lines spanning the full width
        line = _line_of_words(rng, 0.9 * w, text_h * u)
        _put(img, "- " + line, (0.04 * w, first * u + i * pitch * u), text_h * u, (30, 30, 34))
    img[round(55 * u) : round(62 * u)] = (128, 128, 128)  # flat mid-gray band
    for k, x0 in enumerate((0.05, 0.75)):  # two small bar charts, near the left and right edges
        for j in range(5):
            bh = rng.uniform(8, 24) * u
            x = round((x0 + j * 0.04) * w)
            colour = ((220, 120, 40), (60, 150, 80), (70, 110, 200), (200, 180, 50), (150, 80, 170))[(j + k) % 5]
            cv2.rectangle(img, (x, round(92 * u - bh)), (x + round(0.025 * w), round(92 * u)), colour, -1)
    _line_chart(img, rng, x0=round(0.30 * w), x1=round(0.70 * w), y0=round(66 * u), y1=round(94 * u), u=u)
    return img.astype(np.float32) / 255.0


def _line_chart(img: np.ndarray, rng: np.random.Generator, x0: int, x1: int, y0: int, y1: int, u: float) -> None:
    """A framed line chart: a zig-zag series with circle markers, axes and ticks, in the slide's centre.

    Side by side, its sloped segments and closed markers cross the overlap, where ghosting shows best.
    """
    thin = max(1, round(0.25 * u))
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 96), thin)
    for k in range(1, 4):  # horizontal grid lines
        y = round(y0 + k * (y1 - y0) / 4)
        cv2.line(img, (x0, y), (x1, y), (205, 205, 210), thin)
    n = 9
    xs = np.linspace(x0 + 0.06 * (x1 - x0), x1 - 0.06 * (x1 - x0), n)
    ys = y1 - (0.15 + 0.7 * rng.uniform(0, 1, n)) * (y1 - y0)
    pts = np.stack([xs, ys], axis=1).round().astype(np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 44), max(1, round(0.4 * u)), cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), round(1.2 * u), (200, 60, 50), -1, cv2.LINE_AA)
    for x in xs:  # x-axis ticks
        cv2.line(img, (round(x), y1), (round(x), y1 - round(1.5 * u)), (90, 90, 96), thin)


def border_px(frac: float, size: tuple[int, int]) -> int:
    """Width in content pixels of the black border on every side: `frac` x the height."""
    if not 0.0 <= frac < 0.5:
        raise ValueError(f"content: border_frac must be in [0, 0.5), got {frac}")
    return round(frac * size[1])


def inner_size(size: tuple[int, int], frac: float) -> tuple[int, int]:
    """(width, height) of the picture inside a black border of `frac` x the height."""
    w, h = size
    b = border_px(frac, size)
    if min(w - 2 * b, h - 2 * b) < 1:
        raise ValueError(f"content: border_frac {frac} leaves no picture inside {size}")
    return w - 2 * b, h - 2 * b


def framed(size: tuple[int, int], frac: float, draw: Callable[[tuple[int, int]], np.ndarray]) -> np.ndarray:
    """A content picture of `size` (width, height): `draw(inner size)` inside a black border."""
    w, h = size
    b = border_px(frac, size)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[b : h - b, b : w - b] = draw(inner_size(size, frac))
    return img
