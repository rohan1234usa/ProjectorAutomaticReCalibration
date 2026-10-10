"""Frames from the simulator, rendered the way the harness renders them, one family at a time.

A FrameSource keeps a renderer: the screen grid, the blend maps, the room's own image. Building
one takes about half a second and a gigabyte, so the demo keeps one alive at a time. The
variants of a sweep that differ only in how projector B moved can share it: their frames show
the same pictures, room and camera, and only where each projector's pixels land differs. The
base variant's state with the other variant's landing geometry therefore renders that variant's
frame bit for bit (the noise is drawn from the frame index alone, ``sim/frames.py``). Variants
that differ in anything else -- arrangement, nuisances, the screen's gain -- need their own
renderer, and :meth:`Family.state` refuses to share across them.

``facts`` gives what a figure needs about a frame -- the true offset, where each box lands, the
camera's view -- for the base and each variant that shares its renderer. It is read from
``FrameSource.truth``, the metadata line a dataset stores, so a caption and a dataset never
disagree.
"""

from __future__ import annotations

import dataclasses
import gc
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from demo.manifest import SCENARIOS, Shot
from scripts.visualize import white_electrons
from sim.frames import FrameSource
from sim.scenario import Scenario, load_scenarios
from sim.state import FrameState, frame_state
from sim.truth import displacement_maps

MEMO = 4  # decoded frames kept: figures often show the same frame several ways


class Family:
    """One scenario file's renderer, built for `base` and shared with perturbation-only variants."""

    def __init__(self, scenarios: dict[str, Scenario], base: str, stem: str = "") -> None:
        self.stem = stem or scenarios[base].name
        self.scenarios = scenarios
        self.base = base
        self._source: FrameSource | None = FrameSource(self.scenarios[base])
        self.scene = self._source.scene
        self.white = white_electrons(self.scene)
        self.black = self.scene.projectors["a"].black_level
        self._frames: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    @classmethod
    def load(cls, stem: str, base: str, quality: str | None = None, folder: Path = SCENARIOS) -> Family:
        """The family of scenarios/<stem>.yaml, rendering `base` (and the variants that share it)."""
        return cls({s.variant: s for s in load_scenarios(folder / f"{stem}.yaml", quality)}, base, stem)

    @classmethod
    def of(cls, shot: Shot, quality: str | None = None) -> Family:
        return cls.load(shot.scenario, shot.variant, quality)

    def __enter__(self) -> Family:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def source(self) -> FrameSource:
        if self._source is None:
            raise RuntimeError(f"{self.stem}: the family is closed (its renderer was released)")
        return self._source

    def close(self) -> None:
        """Release the renderer (about a gigabyte); the family cannot render after this."""
        self._source = None
        self._frames.clear()
        gc.collect()

    # -- rendering ------------------------------------------------------------------------------
    def shares(self, variant: str) -> bool:
        """True if `variant` differs from the base only in its perturbation."""
        a, b = dict(self.scenarios[self.base].data), dict(self.scenarios[variant].data)
        a.pop("perturbation", None)
        b.pop("perturbation", None)
        return a == b

    def state(self, variant: str, i: int) -> FrameState:
        """The state the base renderer needs to draw `variant`'s frame i.

        It is the base's state with `variant`'s landing geometry: everything that decides the
        picture is right, but its ``scheduled`` and ``applied`` still describe the base's
        perturbation. Ask ``frame_state(scenario, i)`` for those.
        """
        base = self.source.state(i)
        if variant == self.base:
            return base
        if not self.shares(variant):
            raise ValueError(f"{self.stem}: {variant} differs from {self.base} beyond its perturbation")
        return dataclasses.replace(base, h_actual=frame_state(self.scenarios[variant], i).h_actual)

    def frame(self, variant: str, i: int) -> np.ndarray:
        """The 16-bit camera frame, exactly as FrameSource(variant).frame(i) gives it."""
        return self.source.frame(i, self.state(variant, i))

    def electrons(self, variant: str, i: int) -> np.ndarray:
        """The frame decoded to electrons (read-only: the last few are kept and shared)."""
        key = (variant, i)
        if key in self._frames:
            self._frames.move_to_end(key)
            return self._frames[key]
        e = self.scene.camera.decode(self.frame(variant, i))
        e.flags.writeable = False
        self._frames[key] = e
        while len(self._frames) > MEMO:
            self._frames.popitem(last=False)
        return e

    def expected(self, variant: str, i: int) -> np.ndarray:
        """Noiseless electrons (no shot or read noise)."""
        return self.source.expected(i, self.state(variant, i))

    def mean_electrons(self, variant: str, frames: tuple[int, ...]) -> np.ndarray:
        total = np.zeros(self.electrons(variant, frames[0]).shape, np.float64)
        for i in frames:
            total += self.electrons(variant, i)
        return (total / len(frames)).astype(np.float32)

    def room(self, variant: str, i: int) -> np.ndarray:
        """Electrons from room light (and the bezel's own light) alone in this frame: the unlit level."""
        return self.source.unlit(self.state(variant, i))

    # -- what a figure may say about a frame -----------------------------------------------------------
    def facts(self, variant: str, i: int) -> dict[str, Any]:
        """True offset, boxes, camera view and content of `variant`'s frame i (no render)."""
        st = self.state(variant, i)
        line = self.source.truth(i, st)
        truth = line["truth"]
        return {
            "variant": variant,
            "frame": i,
            "t_s": round(line["t_s"], 4),
            "tag": line["content"]["tag"],
            "offset_mm": truth["offset_mm"],
            "offset_px": truth["offset_px"],
            "aligned": truth["aligned"],
            "boxes_mm": {n: np.array(box) for n, box in truth["boxes_mm"].items()},
            "displacement": displacement_maps(self.scene.setup, st.h_actual),  # calibrated mm -> mm now
            "camera_h": np.array(truth["camera_h_mm_to_px"]),
            "markers_visible": truth["markers_visible"],
            "ambient": st.ambient,
            "lamp": dict(st.gains),
            "camera_bump": list(st.camera_key),
            "people": bool(st.people),
        }
