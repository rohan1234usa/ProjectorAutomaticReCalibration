"""Synthetic video: objects gliding over a panning textured background, with cuts and fades.

Why. Lectures show video too, and video breaks what a still slide allows: the picture moves
between samples, cuts replace it entirely, and a camera exposure can straddle two video frames
and record both at once. That double image looks like misalignment, so the detector's motion
gate must skip such frames, and the simulator must produce them faithfully.

A clip is a pure function of (clip parameters, seed, frame index k): frame k shows the scene at
time k / fps, whatever order frames are asked for. Every ``cut_s`` seconds a cut starts a new
scene, with a new background and new objects drawn from that scene's own seed. Within a scene
the background pans at a constant speed and objects glide across the frame, bouncing off its
edges. The background is a 1/f texture that is periodic by construction (it is synthesised by
an inverse FFT), so panning wraps around it without a seam. ``fade_s`` fades every scene in from
black and back out at its cuts, as film scenes do; ``style: dark`` uses dim film backgrounds.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from sim.textures import blend_ellipse, pink_field

STYLES = {"photo": (0.45, 0.2), "dark": (0.10, 0.05)}  # background mean and contrast, in code values
WARM_CAST = 0.05  # red raised and blue lowered by this (code values) at the photo style's contrast


@dataclass(frozen=True)
class Clip:
    fps: Fraction = Fraction(30)
    cut_s: Fraction | None = Fraction(8)  # None: one scene for the whole clip
    pan_px_per_s: tuple[float, float] = (120.0, 0.0)  # content pixels per second
    objects: int = 6
    speed_px_per_s: float = 300.0
    fade_s: Fraction = Fraction(0)  # fade out before and in after each cut
    style: str = "photo"

    def scene_of(self, k: int) -> tuple[int, Fraction]:
        """(scene number, time since the scene began) of video frame k."""
        t = Fraction(k) / self.fps
        if self.cut_s is None:
            return 0, t
        scene = int(t // self.cut_s)
        return scene, t - scene * self.cut_s


class VideoFrames:
    """Frames of one clip at one content size; a scene's background and objects are drawn once."""

    def __init__(self, clip: Clip, size: tuple[int, int], seed: int, key: tuple[int, ...]) -> None:
        if clip.style not in STYLES:
            raise ValueError(f"video: style must be one of {sorted(STYLES)}, got {clip.style!r}")
        self.clip, self.size, self.seed, self.key = clip, size, seed, key
        self._scenes: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _scene(self, scene: int) -> tuple[np.ndarray, np.ndarray]:
        if scene not in self._scenes:
            rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(3, *self.key, scene)))
            w, h = self.size
            mean, contrast = STYLES[self.clip.style]
            lum = pink_field((h, w), rng, beta=1.2)
            background = np.repeat((mean + contrast * lum)[..., None], 3, axis=2)
            cast = WARM_CAST * contrast / STYLES["photo"][1]  # weaker in dim film
            background[..., 0] += cast
            background[..., 2] -= cast
            n = self.clip.objects
            objects = np.column_stack([
                rng.uniform(0, w, n), rng.uniform(0, h, n),  # start position
                rng.uniform(0, 2 * math.pi, n),  # heading
                rng.uniform(0.03, 0.12, n) * h, rng.uniform(0.03, 0.12, n) * h,  # semi-axes
                rng.uniform(0.05, 0.95, (n, 3)),  # colour
            ])
            self._scenes[scene] = (np.clip(background, 0.0, 1.0).astype(np.float32), objects)
            while len(self._scenes) > 2:
                self._scenes.popitem(last=False)
        return self._scenes[scene]

    def frame(self, k: int) -> np.ndarray:
        """Video frame k, (h, w, 3) code values in [0, 1]."""
        scene, t = self.clip.scene_of(k)
        background, objects = self._scene(scene)
        w, h = self.size
        dx, dy = (round(v * float(t)) for v in self.clip.pan_px_per_s)
        img = np.roll(background, (dy % h, dx % w), axis=(0, 1))  # a new array: safe to draw on
        for x0, y0, heading, ax, ay, r, g, b in objects:
            travel = self.clip.speed_px_per_s * float(t)
            x = _bounce(x0 + travel * math.cos(heading), w)
            y = _bounce(y0 + travel * math.sin(heading), h)
            blend_ellipse(img, x, y, ax, ay, 0.0, np.array([r, g, b], np.float32))
        if self.clip.fade_s > 0 and self.clip.cut_s is not None:
            level = min(Fraction(1), t / self.clip.fade_s, (self.clip.cut_s - t) / self.clip.fade_s)
            img *= np.float32(max(0.0, float(level)))
        return img


def _bounce(p: float, length: float) -> float:
    """Position after bouncing between 0 and `length` (a triangle wave)."""
    p = p % (2 * length)
    return p if p <= length else 2 * length - p

