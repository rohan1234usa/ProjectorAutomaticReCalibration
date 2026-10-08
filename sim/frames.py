"""Frames on demand: the camera frame and the ground truth at any frame index, deterministically.

A FrameSource renders frame i of a scenario when asked, in any order, in any process, and always
gives the same 16-bit frame. That lets the evaluation harness feed the detector without
datasets on disk, while a stored hash proves the frame is the one the dataset described.

Each frame starts from its *state*: the pictures shown during its exposure, where each
projector's pixels land, and the room light. The noiseless image is the weighted sum of cached
camera components (``sim/render.py``): the surfaces under room light, and each projector's light
for those pictures through its current geometry. A component is rendered once and reused while
its inputs stay the same, which for a held slide means one render and then only noise. Noise is
drawn from SeedSequence(seed, spawn_key=(1, i)) and from nothing else, so frames before a
perturbation's onset are identical across the variants of a sweep.

Ground truth comes from the state alone, without rendering (``sim/truth.py``).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from sim import fiducials
from sim.perturb import h_actual as perturbed_geometry
from sim.planar import box_mm
from sim.render import compose, readonly
from sim.scenario import Scenario
from sim.sequence import Segments
from sim.truth import ALIGNED_MM, coarse_pitch_mm, offset_mm, relative_homography


class _LRU:
    """A small least-recently-used cache: values are rebuilt by `make` when evicted."""

    def __init__(self, size: int) -> None:
        self.size, self.items = size, OrderedDict()

    def get(self, key: Hashable, make: Callable[[], Any]) -> Any:
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        value = make()
        self.items[key] = value
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return value


@dataclass(frozen=True, eq=False)
class FrameState:
    index: int
    t: Fraction  # start of the exposure, seconds
    segments: Segments  # pictures shown during the exposure, with their share of it
    scheduled: tuple[float, ...]  # each perturbation's schedule value
    applied: tuple[float, ...]  # ... after quantization: what the geometry uses
    h_actual: dict[str, np.ndarray]
    ambient: float


class FrameSource:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.scene = scenario.scene
        self.setup = self.scene.setup
        self.camera = self.scene.camera
        self.renderer = self.scene.renderer()
        self.pitch_mm = coarse_pitch_mm(self.setup)
        self._light = _LRU(8)
        self._electrons = _LRU(6)
        self._room: np.ndarray | None = None
        self._lamp: np.ndarray | None = None
        self._offsets: dict[bytes, tuple[float, list]] = {}
        markers = self.scene.markers
        self._visible = [] if markers is None else fiducials.visible(markers, self.camera.h_mm_to_px, self.camera.resolution)

    def __len__(self) -> int:
        return self.scenario.timing.n_frames

    def time(self, i: int) -> Fraction:
        if not 0 <= i < len(self):
            raise IndexError(f"frame {i} outside 0..{len(self) - 1}")
        return self.scenario.timing.time(i)

    def state(self, i: int) -> FrameState:
        t = self.time(i)
        perturbations = self.scenario.perturbations
        scheduled = tuple(p.schedule.value(t) for p in perturbations)
        applied = tuple(p.multiplier(t) for p in perturbations)
        return FrameState(
            index=i,
            t=t,
            segments=self.scenario.sequence.segments(t, self.scenario.timing.exposure),
            scheduled=scheduled,
            applied=applied,
            h_actual=perturbed_geometry(self.setup, perturbations, applied),
            ambient=self.scene.screen.ambient,
        )

    # -- rendering ------------------------------------------------------------------------------
    def _light_for(self, name: str, segments: Segments) -> np.ndarray:
        def make() -> np.ndarray:
            if len(segments) == 1:
                image = self.scenario.sequence.image(segments[0][0])
                return readonly(self.renderer.projector_light(name, image))
            parts = [(float(w), self._light_for(name, ((key, Fraction(1)),))) for key, w in segments]
            return readonly(compose(parts))  # a change inside the exposure: mixed in linear light

        return self._light.get((name, segments), make)

    def _projector(self, name: str, segments: Segments, h: np.ndarray) -> np.ndarray:
        def make() -> np.ndarray:
            return self.renderer.projector_electrons(name, self._light_for(name, segments), h, self.camera)

        return self._electrons.get((name, segments, h.tobytes()), make)

    def expected(self, i: int, state: FrameState | None = None) -> np.ndarray:
        """Noiseless expected electrons of frame i, float32."""
        state = state or self.state(i)
        if self._room is None:
            self._room = self.renderer.room_electrons(self.camera)
        terms = [(state.ambient, self._room)]
        lamp = self.scene.screen.bezel.light
        if lamp > 0:
            if self._lamp is None:
                self._lamp = self.renderer.lamp_electrons(self.camera)
            terms.append((lamp, self._lamp))
        for name in self.setup.names:
            terms.append((1.0, self._projector(name, state.segments, state.h_actual[name])))
        return compose(terms)

    def frame(self, i: int, state: FrameState | None = None) -> np.ndarray:
        """Camera frame i: 16-bit, (h, w) mono or (h, w, 3) RGB."""
        electrons = self.expected(i, state)
        rng = np.random.default_rng(np.random.SeedSequence(self.scene.seed, spawn_key=(1, i)))
        return self.camera.encode(self.camera.add_noise(electrons, rng))

    # -- ground truth -----------------------------------------------------------------------------
    def truth(self, i: int, state: FrameState | None = None) -> dict[str, Any]:
        """Frame i's metadata line (everything but the frame hash): never needs a render."""
        state = state or self.state(i)
        geometry = b"".join(state.h_actual[n].tobytes() for n in self.setup.names)
        if geometry not in self._offsets:
            rel = relative_homography(self.setup, state.h_actual)
            self._offsets[geometry] = (offset_mm(self.setup, state.h_actual), rel.tolist())
        offset, rel = self._offsets[geometry]
        sequence = self.scenario.sequence
        tags = sorted({sequence.tag(key) for key, _ in state.segments})
        perturbations = []
        for p, scheduled, applied in zip(self.scenario.perturbations, state.scheduled, state.applied):
            entry = {"projector": p.projector, "scheduled": scheduled, "applied": applied, **p.tag(applied)}
            if p.kind != "shift":
                entry["pivot_mm"] = [float(v) for v in p.pivot]
            perturbations.append(entry)
        return {
            "i": i,
            "t_s": float(state.t),
            "content": {
                "tag": "+".join(tags),
                "segments": [["/".join(str(k) for k in key), float(w)] for key, w in state.segments],
                "cut_in_exposure": len(state.segments) > 1,
            },
            "truth": {
                "aligned": offset < ALIGNED_MM,
                "offset_mm": offset,
                "offset_px": offset / self.pitch_mm,
                "h_rel": rel,
                "h_actual": {n: state.h_actual[n].tolist() for n in self.setup.names},
                "boxes_mm": {n: box_mm(state.h_actual[n], self.setup.resolution[n]).tolist() for n in self.setup.names},
                "camera_h_mm_to_px": self.camera.h_mm_to_px.tolist(),
                "markers_visible": list(self._visible),
            },
            "perturbation": perturbations,
            "nuisances": {"ambient": state.ambient, "bezel_light": self.scene.screen.bezel.light},
        }

    def setup_dict(self) -> dict[str, Any]:
        return self.scenario.setup_dict()
