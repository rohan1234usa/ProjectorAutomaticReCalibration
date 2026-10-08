"""Content over time: what the calibration software is asked to show at each moment.

A scenario's ``content`` is a list of items played in order, optionally looped:

  deck    a slide deck: ``slides`` slides, each held ``hold_s`` seconds. ``densities`` cycles the
          text density slide by slide (e.g. [low, medium, high]); ``density`` fixes it.
  held    one slide held for ``hold_s`` seconds (the 20-minute held slide).
  flat    a flat field (``value``: gray level or RGB) for ``hold_s`` seconds.
  black   black for ``hold_s`` seconds.

Every item takes ``border_frac``, a black border inside the content rect. (Phase 2b adds video,
photos, stripes, letterbox, blank-overlap and dark film.) Written as
``content: {loop: true, items: [...]}``; a bare list plays once; a single mapping with a
``type`` is one picture held for the whole run.

Exposure. The camera integrates light over [t, t + exposure_s). If the picture changes inside
that window, the frame holds both pictures, weighted by how long each was up:
:meth:`Sequence.segments` returns those (key, weight) pairs and the renderer mixes them in
linear light. Times are exact Fractions, so whether a change falls inside the window never
depends on rounding.

Randomness. Every picture is drawn from its own SeedSequence spawn key (0, item, loop, index):
the same whichever order frames are rendered in, and a looped deck shows new slides each loop.

Display latency. The projectors show the content that was sent ``lag_s`` earlier (the scenario's
``reference.lag_s``): the content timeline is the sending clock, and the frame exposed at t shows
content(t - lag_s). The reference feed (Phase 2b) hands out source frames on the sending clock.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from sim.cfg import check_keys, choice, integer, num, seconds
from sim.content import DENSITIES, make_content

ContentKey = tuple[int, int, int]  # (item, loop, picture index)
Segments = tuple[tuple[ContentKey, Fraction], ...]
_KINDS = ("deck", "held", "flat", "black")
_ITEM_KEYS = {
    "deck": {"type", "slides", "hold_s", "density", "densities", "border_frac"},
    "held": {"type", "hold_s", "density", "border_frac"},
    "flat": {"type", "hold_s", "value", "border_frac"},
    "black": {"type", "hold_s", "border_frac"},
}


@dataclass(frozen=True)
class Item:
    kind: str
    hold: Fraction  # how long each picture stays up
    count: int  # pictures in the item
    densities: tuple[str, ...] = ("medium",)
    value: float | tuple[float, ...] = 0.5
    border_frac: float = 0.0

    @property
    def duration(self) -> Fraction:
        return self.hold * self.count

    @property
    def varies(self) -> bool:
        """True if pictures differ between loops (random slides); flat and black never do."""
        return self.kind in ("deck", "held")

    def density(self, index: int) -> str:
        return self.densities[index % len(self.densities)]

    def tag(self, index: int) -> str:
        return f"{self.kind}_{self.density(index)}" if self.varies else self.kind


@dataclass(eq=False)
class Sequence:
    items: tuple[Item, ...]
    loop: bool
    seed: int
    size: tuple[int, int]  # content resolution (width, height)
    lag: Fraction = Fraction(0)
    _starts: list[Fraction] = field(init=False, repr=False)
    _cache: OrderedDict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("content: at least one item is needed")
        self._starts, t = [], Fraction(0)
        for item in self.items:
            self._starts.append(t)
            t += item.duration
        self._cache = OrderedDict()

    @property
    def cycle(self) -> Fraction:
        return self._starts[-1] + self.items[-1].duration

    def check_covers(self, duration: Fraction) -> None:
        if not self.loop and duration > self.cycle:
            raise ValueError(f"content lasts {float(self.cycle):g} s but must last {float(duration):g} s; "
                             "set loop: true")

    def _locate(self, t: Fraction) -> tuple[ContentKey, Fraction]:
        """The picture shown at display time t, and when it is replaced."""
        t = max(t, Fraction(0))
        loop = int(t // self.cycle) if self.loop else 0
        offset = t - loop * self.cycle
        i = bisect.bisect_right(self._starts, offset) - 1
        item = self.items[i]
        index = min(int((offset - self._starts[i]) // item.hold), item.count - 1)
        ends = loop * self.cycle + self._starts[i] + (index + 1) * item.hold
        if not self.loop and i == len(self.items) - 1 and index == item.count - 1:
            ends = max(ends, t) + self.cycle  # the last picture of a non-looping run stays up
        key = (i, loop if item.varies else 0, index)
        return key, ends

    def segments(self, t: Fraction, exposure: Fraction) -> Segments:
        """Pictures shown during the exposure [t, t + exposure), with their share of it."""
        start, end = t - self.lag, t - self.lag + exposure
        out: list[tuple[ContentKey, Fraction]] = []
        now = start
        while now < end:
            key, change = self._locate(now)
            nxt = min(change, end)
            if out and out[-1][0] == key:
                out[-1] = (key, out[-1][1] + (nxt - now) / exposure)
            else:
                out.append((key, (nxt - now) / exposure))
            now = nxt
        return tuple(out)

    def tag(self, key: ContentKey) -> str:
        return self.items[key[0]].tag(key[2])

    def image(self, key: ContentKey) -> np.ndarray:
        """The content picture for `key` (read-only, cached)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        item = self.items[key[0]]
        rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(0, *key)))
        cfg: dict[str, Any] = {"border_frac": item.border_frac}
        if item.varies:
            cfg.update(type="slide", density=item.density(key[2]))
        elif item.kind == "flat":
            cfg.update(type="flat", value=item.value)
        else:
            cfg.update(type="black")
        img = make_content(cfg, self.size, rng)
        img.setflags(write=False)
        self._cache[key] = img
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return img


def _item(cfg: Mapping[str, Any], where: str, default_hold: Fraction) -> Item:
    kind = choice(cfg.get("type"), _KINDS, f"{where}.type")
    check_keys(cfg, _ITEM_KEYS[kind], where)
    hold = seconds(cfg["hold_s"], f"{where}.hold_s") if "hold_s" in cfg else default_hold
    if hold <= 0:
        raise ValueError(f"{where}: hold_s must be positive")
    border = num(cfg.get("border_frac", 0.0), f"{where}.border_frac")
    if kind in ("deck", "held"):
        if "density" in cfg and "densities" in cfg:
            raise ValueError(f"{where}: give density or densities, not both")
        densities = cfg.get("densities", [cfg.get("density", "medium")])
        densities = tuple(choice(d, tuple(DENSITIES), f"{where}.density") for d in densities)
        count = integer(cfg.get("slides", 1), f"{where}.slides") if kind == "deck" else 1
        if count < 1:
            raise ValueError(f"{where}: slides must be at least 1")
        return Item(kind, hold, count, densities=densities, border_frac=border)
    value: float | tuple[float, ...] = 0.0
    if kind == "flat":
        raw = cfg.get("value", 0.5)
        value = tuple(num(v, f"{where}.value") for v in raw) if isinstance(raw, (list, tuple)) else num(raw, f"{where}.value")
    return Item(kind, hold, 1, value=value, border_frac=border)


def from_config(cfg: Any, seed: int, size: tuple[int, int], duration: Fraction, lag: Fraction = Fraction(0)) -> Sequence:
    """Parse ``content``: {loop, items: [...]}, a bare list (played once), or one picture mapping.

    `duration` is how long the content must last (on the content clock); `lag` is the display
    latency: the projectors show what was sent `lag` seconds earlier.
    """
    if isinstance(cfg, Mapping) and "items" in cfg:
        check_keys(cfg, {"items", "loop"}, "content")
        items_cfg, loop = cfg["items"], cfg.get("loop", False)
        if not isinstance(loop, bool):
            raise ValueError("content.loop: expected true or false")
    elif isinstance(cfg, Mapping):
        picture = dict(cfg)
        if picture.get("type") == "slide":  # Phase 1 spelling of a held slide
            picture["type"] = "held"
        items_cfg, loop = [picture], False
    else:
        items_cfg, loop = cfg, False
    if not isinstance(items_cfg, list) or not items_cfg:
        raise ValueError("content: expected a non-empty list of items")
    items = tuple(_item(c, f"content.items[{i}]", duration) for i, c in enumerate(items_cfg))
    sequence = Sequence(items, loop, seed, size, lag)
    sequence.check_covers(duration)
    return sequence
