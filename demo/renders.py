"""Frames from the simulator, rendered the way the harness renders them, one family at a time.

A FrameSource keeps a renderer: the screen grid, the blend maps, the room's own image. Building
one takes about half a second and a gigabyte, so the demo keeps one alive at a time. The
variants of a sweep that differ only in how projector B moved can share it: their frames show
the same pictures, room and camera, and only where each projector's pixels land differs. The
base variant's state with the other variant's landing geometry therefore renders that variant's
frame bit for bit (the noise is drawn from the frame index alone, ``sim/frames.py``). Variants
that differ in anything else -- arrangement, nuisances, the screen's gain -- need their own
renderer, and :meth:`Family.frame` refuses to share across them.

``facts`` gives what a figure needs about a frame -- the true offset, where each box lands, the
camera's view -- for any variant, from its own state and without a second renderer.
"""

from __future__ import annotations

import dataclasses
import gc
from pathlib import Path
from typing import Any

import numpy as np

from demo.manifest import SCENARIOS, Shot
from scripts.visualize import white_electrons
from sim.frames import FrameSource
from sim.planar import box_mm
from sim.scenario import Scenario, load_scenarios
from sim.state import FrameState, frame_state
from sim.truth import ALIGNED_MM, offset_mm


class Family:
    """One scenario file's renderer, built for `base` and shared with perturbation-only variants."""

    def __init__(self, scenarios: dict[str, Scenario], base: str, stem: str = "") -> None:
        self.stem = stem or scenarios[base].name
        self.scenarios = scenarios
        self.base = base
        self.source = FrameSource(self.scenarios[base])
        self.scene = self.source.scene
        self.white = white_electrons(self.scene)
        self.black = self.scene.projectors["a"].black_level
        self._room: dict[tuple, tuple[np.ndarray, Any]] = {}

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

    def close(self) -> None:
        self.source = None  # type: ignore[assignment]
        self._room.clear()
        gc.collect()

    # -- rendering ------------------------------------------------------------------------------
    def shares(self, variant: str) -> bool:
        """True if `variant` differs from the base only in its perturbation."""
        a, b = dict(self.scenarios[self.base].data), dict(self.scenarios[variant].data)
        a.pop("perturbation", None)
        b.pop("perturbation", None)
        return a == b

    def state(self, variant: str, i: int) -> FrameState:
        """The state the base renderer needs to draw `variant`'s frame i."""
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
        return self.scene.camera.decode(self.frame(variant, i))

    def expected(self, variant: str, i: int) -> np.ndarray:
        """Noiseless electrons (no shot or read noise)."""
        return self.source.expected(i, self.state(variant, i))

    def mean_electrons(self, variant: str, frames: tuple[int, ...]) -> np.ndarray:
        total = None
        for i in frames:
            e = self.electrons(variant, i).astype(np.float64)
            total = e if total is None else total + e
        return (total / len(frames)).astype(np.float32)

    def room(self, variant: str, i: int) -> np.ndarray:
        """Electrons from room light (and the bezel's own light) alone in this frame: the unlit level."""
        st = self.state(variant, i)
        if st.camera_key not in self._room:
            camera, renderer = self.source.camera_for(st), self.source.renderer
            bezel = self.scene.screen.bezel.light
            lamp = renderer.bezel_light_electrons(camera) * np.float32(bezel) if bezel > 0 else 0.0
            self._room[st.camera_key] = (renderer.room_electrons(camera).astype(np.float32), lamp)
        room, lamp = self._room[st.camera_key]
        return room * np.float32(st.ambient) + lamp

    # -- what a figure may say about a frame -----------------------------------------------------------
    def facts(self, variant: str, i: int) -> dict[str, Any]:
        """True offset, boxes, camera view and content of `variant`'s frame i (no render)."""
        scenario = self.scenarios[variant]
        st = frame_state(scenario, i)
        setup = self.scene.setup
        off = offset_mm(setup, st.h_actual)
        render_state = self.state(variant, i)
        camera = self.source.camera_for(render_state)
        return {
            "variant": variant,
            "frame": i,
            "t_s": round(float(st.t), 4),
            "tag": "+".join(sorted({scenario.sequence.tag(k) for k, _ in st.segments})),
            "offset_mm": off,
            "offset_px": off / self.source.pitch_mm,
            "aligned": off < ALIGNED_MM,
            "boxes_mm": {n: box_mm(st.h_actual[n], setup.resolution[n]) for n in setup.names},
            "camera_h": camera.h_mm_to_px,
            "markers_visible": self.source.markers_visible(render_state),
            "ambient": st.ambient,
            "lamp": dict(st.gains),
            "camera_bump": list(st.camera_key),
            "people": bool(st.people),
        }
