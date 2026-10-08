"""Content over time: which picture the calibration software is sending at each moment.

A scenario's ``content`` is a list of items played in order, optionally looped
(``content: {loop: true, items: [...]}``); a bare list plays once, and a single mapping with a
``type`` is one picture held for the whole run. The kinds of item and how each picture is drawn
are in ``sim/pictures.py``: slide decks, held slides, flat fields, black, photos, dark film,
stripes and video.

Exposure. The camera integrates light over [t, t + exposure_s). If the picture changes inside
that window -- a slide change, a cut, or simply the next video frame -- the frame holds both
pictures, weighted by how long each was up: :meth:`Sequence.segments` returns those (key, weight)
pairs and the renderer mixes them in linear light. Times are exact Fractions, so whether a
change falls inside the window never depends on rounding.

Randomness. Every still picture is drawn from its own SeedSequence spawn key (0, item, loop,
index): the same whichever order frames are rendered in, and a looped deck shows new slides each
loop. Video frames are a function of the scene seed (3, item, scene) and the frame index.

Clocks. The content timeline is the clock the video signal is *sent* on. The projectors show
the content that was sent ``lag_s`` earlier, so the frame exposed at t shows content(t - lag_s).
:meth:`Sequence.sent_before` lists the pictures sent by a given time: the reference feed.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from sim import pictures
from sim.cfg import check_keys
from sim.pictures import ContentGeometry, Item

ContentKey = tuple[int, int, int]  # (item, loop, picture index)
Segments = tuple[tuple[ContentKey, Fraction], ...]
_EPSILON = Fraction(1, 10**9)  # far shorter than any picture: steps just before a picture's start


@dataclass(eq=False)
class Sequence:
    items: tuple[Item, ...]
    loop: bool
    seed: int
    size: tuple[int, int]  # content resolution (width, height)
    geometry: ContentGeometry
    lag: Fraction = Fraction(0)
    _starts: list[Fraction] = field(init=False, repr=False)
    _cache: OrderedDict = field(init=False, repr=False)
    _videos: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("content: at least one item is needed")
        self._starts, t = [], Fraction(0)
        for item in self.items:
            self._starts.append(t)
            t += item.duration
        self._cache, self._videos = OrderedDict(), {}

    @property
    def cycle(self) -> Fraction:
        return self._starts[-1] + self.items[-1].duration

    def check_covers(self, duration: Fraction) -> None:
        if not self.loop and duration > self.cycle:
            raise ValueError(f"content lasts {float(self.cycle):g} s but must last {float(duration):g} s; "
                             "set loop: true")

    def _locate(self, t: Fraction) -> tuple[ContentKey, Fraction, Fraction]:
        """The picture being sent at content time t, when it started and when it is replaced."""
        t = max(t, Fraction(0))
        loop = int(t // self.cycle) if self.loop else 0
        offset = t - loop * self.cycle
        i = bisect.bisect_right(self._starts, offset) - 1
        item = self.items[i]
        index = min(int((offset - self._starts[i]) // item.hold), item.count - 1)
        start = loop * self.cycle + self._starts[i] + index * item.hold
        ends = start + item.hold
        if not self.loop and i == len(self.items) - 1 and index == item.count - 1:
            ends = max(ends, t) + self.cycle  # the last picture of a non-looping run stays up
        return (i, loop if item.varies else 0, index), start, ends

    def segments(self, t: Fraction, exposure: Fraction) -> Segments:
        """Pictures shown during the exposure [t, t + exposure), with their share of it."""
        start, end = t - self.lag, t - self.lag + exposure
        out: list[tuple[ContentKey, Fraction]] = []
        now = start
        while now < end:
            key, _, change = self._locate(now)
            nxt = min(change, end)
            if out and out[-1][0] == key:
                out[-1] = (key, out[-1][1] + (nxt - now) / exposure)
            else:
                out.append((key, (nxt - now) / exposure))
            now = nxt
        return tuple(out)

    def sent_before(self, t: Fraction, count: int) -> list[tuple[Fraction, ContentKey]]:
        """The last `count` pictures sent at or before content time t, newest first, with send times."""
        out: list[tuple[Fraction, ContentKey]] = []
        now = t
        while len(out) < count and now >= 0:
            key, start, _ = self._locate(now)
            out.append((max(start, Fraction(0)), key))
            now = start - _EPSILON
        return out

    def same_shot(self, a: ContentKey, b: ContentKey) -> bool:
        """True if `b` follows `a` within one video scene: motion, not a cut or a new picture."""
        item = self.items[a[0]]
        if a[0] != b[0] or item.kind != "video" or b[2] != a[2] + 1:
            return False
        return item.clip.scene_of(a[2])[0] == item.clip.scene_of(b[2])[0]

    def tag(self, key: ContentKey) -> str:
        return self.items[key[0]].tag(key[2])

    def image(self, key: ContentKey) -> np.ndarray:
        """The content picture for `key` (read-only, cached)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        item = self.items[key[0]]
        rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(0, *key)))
        video = None
        if item.kind == "video":
            if key[0] not in self._videos:
                self._videos[key[0]] = pictures.video_frames(item, key[0], self.size, self.seed)
            video = self._videos[key[0]]
        img = pictures.draw(item, key[2], rng, self.size, self.geometry, video)
        img.setflags(write=False)
        self._cache[key] = img
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return img


def from_config(cfg: Any, seed: int, size: tuple[int, int], duration: Fraction, geometry: ContentGeometry,
                lag: Fraction = Fraction(0)) -> Sequence:
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
    items = tuple(pictures.parse(c, f"content.items[{i}]", duration, geometry) for i, c in enumerate(items_cfg))
    sequence = Sequence(items, loop, seed, size, geometry, lag)
    sequence.check_covers(duration)
    return sequence
