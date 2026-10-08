"""Content items: the kinds of picture a scenario can show, and how each is drawn.

A scenario's content is a list of items (``sim/sequence.py`` plays them in time). Each item is
one kind of picture, held for ``hold_s`` seconds per picture:

  deck     ``slides`` text slides; ``densities`` cycles low/medium/high slide by slide
  held     one slide held for its whole ``hold_s`` (the 20-minute held slide)
  flat     a flat field (``value``: gray level or RGB)
  black    black
  photo    ``pictures`` photo-like stills (1/f texture and soft objects)
  dark     ``pictures`` dim film stills
  stripes  ``pictures`` sinusoidal gratings of ``period_mm`` on the screen at ``angle_deg``, each
           at a random phase (the echo test needs varied content, even repetitive content)
  video    a synthetic clip (``sim/video.py``): ``duration_s`` at ``fps``, cuts every ``cut_s``,
           a panning background, ``objects`` gliding at ``speed_px_per_s``, ``fade_s`` fades,
           ``style`` photo or dark

Options for every kind:
  border_frac    a black border inside the content rect, so the black-level raster shows around it
  letterbox      black bars of this fraction of the height at the top and bottom: the content
                 border then sits inside the raster, so only the black level shows its edge
  blank_overlap  a flat value (or true for 0.5) inside the calibrated overlap: nothing textured
                 there, so the echo test is blind and the hotspot fit and the boundary must work

The overlap and the screen scale come from the installation (:class:`ContentGeometry`); the
detector learns neither from the pictures. Slides, flat fields and black are drawn exactly as
``sim/content.py`` draws them, so earlier datasets stay bit-identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from sim import textures
from sim.cfg import check_keys, choice, integer, num, pair, seconds
from sim.content import DENSITIES, border_px, make_content
from sim.planar import clip_convex, rect_polygon
from sim.video import STYLES, Clip, VideoFrames

KINDS = ("deck", "held", "flat", "black", "photo", "dark", "stripes", "video")
_COMMON = {"type", "hold_s", "border_frac", "letterbox", "blank_overlap"}
_KEYS = {
    "deck": _COMMON | {"slides", "density", "densities"},
    "held": _COMMON | {"density"},
    "flat": _COMMON | {"value"},
    "black": _COMMON,
    "photo": _COMMON | {"pictures"},
    "dark": _COMMON | {"pictures"},
    "stripes": _COMMON | {"period_mm", "angle_deg", "low", "high", "pictures"},
    "video": (_COMMON - {"hold_s"}) | {"duration_s", "fps", "cut_s", "pan_px_per_s", "objects", "speed_px_per_s",
                                       "fade_s", "style"},
}


@dataclass(frozen=True, eq=False)
class ContentGeometry:
    """What pictures may know of the installation, in content pixels."""

    px_per_mm: float  # content pixels per screen millimetre
    overlap_px: np.ndarray  # the calibrated overlap ∩ content rect, as a polygon in content pixels


@dataclass(frozen=True)
class Item:
    kind: str
    hold: Fraction  # how long each picture stays up (a video frame's period for video)
    count: int  # pictures in the item
    densities: tuple[str, ...] = ("medium",)
    value: float | tuple[float, ...] = 0.5
    border_frac: float = 0.0
    letterbox: float = 0.0
    blank_overlap: float | None = None
    stripes: tuple[float, float, float, float] | None = None  # period (content px), angle, low, high
    clip: Clip | None = None

    @property
    def duration(self) -> Fraction:
        return self.hold * self.count

    @property
    def varies(self) -> bool:
        """True if pictures differ between loops (random slides and stills); the rest replay."""
        return self.kind in ("deck", "held", "photo", "dark", "stripes")

    def density(self, index: int) -> str:
        return self.densities[index % len(self.densities)]

    def tag(self, index: int) -> str:
        base = {"deck": f"deck_{self.density(index)}", "held": f"held_{self.density(index)}"}.get(self.kind, self.kind)
        if self.kind == "video":
            base = f"video_{self.clip.style}"
        return base + ("_letterbox" if self.letterbox else "") + ("_blank" if self.blank_overlap is not None else "")


def parse(cfg: Mapping[str, Any], where: str, default_hold: Fraction, geometry: ContentGeometry) -> Item:
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{where}: expected a mapping like {{type: deck, ...}}")
    kind = choice(cfg.get("type"), KINDS, f"{where}.type")
    check_keys(cfg, _KEYS[kind], where)
    options: dict[str, Any] = {"border_frac": num(cfg.get("border_frac", 0.0), f"{where}.border_frac"),
                               "letterbox": num(cfg.get("letterbox", 0.0), f"{where}.letterbox")}
    if not 0 <= options["letterbox"] < 0.5:
        raise ValueError(f"{where}.letterbox: must be in [0, 0.5)")
    blank = cfg.get("blank_overlap", None)
    if blank is not None and blank is not False:
        options["blank_overlap"] = 0.5 if blank is True else num(blank, f"{where}.blank_overlap")
    if kind == "video":
        fps = seconds(cfg.get("fps", 30), f"{where}.fps")
        duration = seconds(cfg["duration_s"], f"{where}.duration_s") if "duration_s" in cfg else default_hold
        frames = duration * fps
        if fps <= 0 or frames < 1 or frames.denominator != 1:
            raise ValueError(f"{where}: duration_s x fps must be a whole number of frames")
        cut = cfg.get("cut_s", 8)
        clip = Clip(
            fps=fps,
            cut_s=None if cut is None else seconds(cut, f"{where}.cut_s"),
            pan_px_per_s=pair(cfg.get("pan_px_per_s", [120, 0]), f"{where}.pan_px_per_s"),
            objects=integer(cfg.get("objects", 6), f"{where}.objects"),
            speed_px_per_s=num(cfg.get("speed_px_per_s", 300), f"{where}.speed_px_per_s"),
            fade_s=seconds(cfg.get("fade_s", 0), f"{where}.fade_s"),
            style=choice(cfg.get("style", "photo"), tuple(STYLES), f"{where}.style"),
        )
        return Item(kind, 1 / fps, int(frames), clip=clip, **options)
    hold = seconds(cfg["hold_s"], f"{where}.hold_s") if "hold_s" in cfg else default_hold
    if hold <= 0:
        raise ValueError(f"{where}: hold_s must be positive")
    if kind in ("deck", "held"):
        if "density" in cfg and "densities" in cfg:
            raise ValueError(f"{where}: give density or densities, not both")
        densities = cfg.get("densities", [cfg.get("density", "medium")])
        densities = tuple(choice(d, tuple(DENSITIES), f"{where}.density") for d in densities)
        count = integer(cfg.get("slides", 1), f"{where}.slides") if kind == "deck" else 1
        if count < 1:
            raise ValueError(f"{where}: slides must be at least 1")
        return Item(kind, hold, count, densities=densities, **options)
    if kind in ("photo", "dark", "stripes"):
        count = integer(cfg.get("pictures", 1), f"{where}.pictures")
        if count < 1:
            raise ValueError(f"{where}: pictures must be at least 1")
        if kind != "stripes":
            return Item(kind, hold, count, **options)
        period = num(cfg.get("period_mm", 20.0), f"{where}.period_mm") * geometry.px_per_mm
        if period < 2:
            raise ValueError(f"{where}: period_mm is below two content pixels")
        stripes = (period, num(cfg.get("angle_deg", 0.0), f"{where}.angle_deg"),
                   num(cfg.get("low", 0.15), f"{where}.low"), num(cfg.get("high", 0.85), f"{where}.high"))
        return Item(kind, hold, count, stripes=stripes, **options)
    value: float | tuple[float, ...] = 0.0
    if kind == "flat":
        raw = cfg.get("value", 0.5)
        value = tuple(num(v, f"{where}.value") for v in raw) if isinstance(raw, (list, tuple)) else num(raw, f"{where}.value")
    return Item(kind, hold, 1, value=value, **options)


def draw(item: Item, index: int, rng: np.random.Generator, size: tuple[int, int], geometry: ContentGeometry,
         video: VideoFrames | None = None) -> np.ndarray:
    """Picture `index` of `item` at content `size`, (h, w, 3) code values in [0, 1]."""
    if item.kind in ("deck", "held", "flat", "black"):
        cfg: dict[str, Any] = {"border_frac": item.border_frac}
        if item.kind in ("deck", "held"):
            cfg.update(type="slide", density=item.density(index))
        elif item.kind == "flat":
            cfg.update(type="flat", value=item.value)
        else:
            cfg.update(type="black")
        img = make_content(cfg, size, rng)  # exactly as in Phase 2a
    else:
        w, h = size
        b = border_px({"border_frac": item.border_frac}, size)
        inner = (w - 2 * b, h - 2 * b)
        if item.kind == "photo":
            picture = textures.photo(inner, rng)
        elif item.kind == "dark":
            picture = textures.dark(inner, rng)
        elif item.kind == "stripes":
            picture = textures.stripes(inner, *item.stripes, phase=float(rng.uniform(0, 2 * np.pi)))
        else:
            picture = video.frame(index)
        img = np.zeros((h, w, 3), dtype=np.float32)
        img[b : h - b, b : w - b] = picture
    w, h = size
    border = border_px({"border_frac": item.border_frac}, size)
    bar = max(border, round(item.letterbox * h))
    if item.blank_overlap is not None:  # flat only where the picture is: not over its border or bars
        area = rect_polygon(border - 0.5, bar - 0.5, w - border - 0.5, h - bar - 0.5)
        region = clip_convex(geometry.overlap_px, area)
        if len(region):
            img = textures.blank_inside(img, region, item.blank_overlap)
    if item.letterbox:
        img = textures.letterbox(img, item.letterbox)
    return img


def video_frames(item: Item, item_index: int, size: tuple[int, int], seed: int) -> VideoFrames:
    """The frame source for a video item, at the picture size inside its border."""
    w, h = size
    b = border_px({"border_frac": item.border_frac}, size)
    return VideoFrames(item.clip, (w - 2 * b, h - 2 * b), seed, (item_index,))
