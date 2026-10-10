"""Frames on demand: the camera frame and the ground truth at any frame index, deterministically.

A FrameSource renders frame i of a scenario when asked, in any order, in any process, and always
gives the same 16-bit frame. That lets the evaluation harness feed the detector without
datasets on disk, while a stored hash proves the frame is the one the dataset described.

Each frame starts from its state (``sim/state.py``). The noiseless image is the weighted sum of
cached camera components (``sim/render.py``): the surfaces under room light, the bezel under its
lamp, and each projector's light for the pictures shown, through its current geometry and the
camera's current view. Lamp dimming and room light are weights; projector flicker is a weight per
camera row (``sim/flicker.py``). A person crossing in front of the screen replaces part of the
image with a surface of lower reflectance lit by the same light. Then come shot and read noise,
drawn from SeedSequence(seed, spawn_key=(1, i)) and from nothing else, so frames before any
change are identical across the variants of a sweep; then in-camera sharpening, if left on.

A component is rendered once and reused while its inputs stay the same, which for a held slide
means one render and then only noise. Ground truth comes from the state alone.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from fractions import Fraction
from typing import Any

import numpy as np

from sim import fiducials
from sim.camera import Camera
from sim.nuisance import occluder_mask
from sim.planar import box_mm, clip_convex, rect_polygon
from sim.render import compose, readonly
from sim.scenario import Scenario
from sim.sequence import EPSILON, ContentKey, Segments
from sim.state import FrameState, frame_state
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
        self._views: dict[tuple, dict[str, Any]] = {}  # per camera knock: camera, room and lamp images
        self._offsets: dict[bytes, tuple[float, list]] = {}

    def __len__(self) -> int:
        return self.scenario.timing.n_frames

    def time(self, i: int) -> Fraction:
        if not 0 <= i < len(self):
            raise IndexError(f"frame {i} outside 0..{len(self) - 1}")
        return self.scenario.timing.time(i)

    def state(self, i: int) -> FrameState:
        self.time(i)
        return frame_state(self.scenario, i)

    # -- rendering ------------------------------------------------------------------------------
    def _view(self, key: tuple) -> dict[str, Any]:
        """The camera after its knocks so far, with its own images of the room light and bezel light."""
        if key not in self._views:
            camera = self.camera
            for bump, m in zip(self.scenario.nuisances.bumps, key, strict=True):
                if m:
                    camera = camera.moved(bump.image_transform(m, camera.resolution))
            self._views[key] = {"camera": camera}
        return self._views[key]

    def _room(self, key: tuple) -> np.ndarray:
        view = self._view(key)
        if "room" not in view:
            view["room"] = self.renderer.room_electrons(view["camera"])
        return view["room"]

    def _bezel_light(self, key: tuple) -> np.ndarray:
        view = self._view(key)
        if "bezel" not in view:
            view["bezel"] = self.renderer.bezel_light_electrons(view["camera"])
        return view["bezel"]

    def _light_for(self, name: str, segments: Segments) -> np.ndarray:
        def make() -> np.ndarray:
            if len(segments) == 1:
                image = self.scenario.sequence.image(segments[0][0])
                return readonly(self.renderer.projector_light(name, image))
            parts = [(float(w), self._light_for(name, ((key, Fraction(1)),))) for key, w in segments]
            return readonly(compose(parts))  # a change inside the exposure: mixed in linear light

        return self._light.get((name, segments), make)

    def _projector(self, name: str, segments: Segments, h: np.ndarray, key: tuple) -> np.ndarray:
        def make() -> np.ndarray:
            camera = self._view(key)["camera"]
            return self.renderer.projector_electrons(name, self._light_for(name, segments), h, camera)

        return self._electrons.get((name, segments, h.tobytes(), key), make)

    def _projector_terms(self, state: FrameState) -> list[tuple[Any, np.ndarray]]:
        terms = []
        rows = self.camera.resolution[1]
        for name in self.setup.names:
            weight: Any = state.gains[name]
            bands = [f.row_gain(state.index, self.scenario.timing.exposure, rows)
                     for f in self.scenario.nuisances.flickers if f.projector == name]
            if bands:
                band = np.prod(bands, axis=0) * np.float32(weight)
                weight = band[:, None] if self.renderer.mono else band[:, None, None]
            terms.append((weight, self._projector(name, state.segments, state.h_actual[name], state.camera_key)))
        return terms

    def expected(self, i: int, state: FrameState | None = None) -> np.ndarray:
        """Noiseless expected electrons of frame i, float32."""
        state = state or self.state(i)
        key = state.camera_key
        terms = [(state.ambient, self._room(key))]
        bezel_light = self.scene.screen.bezel.light
        if bezel_light > 0:
            terms.append((bezel_light, self._bezel_light(key)))
        projectors = self._projector_terms(state)
        electrons = compose(terms + projectors)
        if state.people:
            camera = self._view(key)["camera"]
            # A person is lit like the screen behind them (projector light, room light) but
            # reflects only their own reflectance of it; projector images already carry the
            # screen's reflectance, so they are rescaled. Later people pass in front. (On a gain
            # screen those images also carry the screen's gain, which a person lacks: issue #3.)
            lit = compose(projectors)
            uniform = camera.vignetting_map() * np.float32(camera.electrons_per_unit_radiance * state.ambient)
            for reflectance, polygons in state.people:
                mask = occluder_mask(polygons, camera.h_mm_to_px, camera.resolution)
                person = lit * np.float32(reflectance / self.scene.screen.reflectance)
                person += (uniform if self.renderer.mono else uniform[..., None]) * np.float32(reflectance)
                m = mask if self.renderer.mono else mask[..., None]
                electrons += m * (person - electrons)
        return electrons

    def unlit(self, state: FrameState) -> np.ndarray:
        """Noiseless electrons from the room light and the bezel's own light alone: both projectors off.

        The same cached images and weights :meth:`expected` starts from, so a figure can take the
        room light away exactly as the frame added it.
        """
        key = state.camera_key
        terms = [(state.ambient, self._room(key))]
        bezel_light = self.scene.screen.bezel.light
        if bezel_light > 0:
            terms.append((bezel_light, self._bezel_light(key)))
        return compose(terms)

    def frame(self, i: int, state: FrameState | None = None) -> np.ndarray:
        """Camera frame i: 16-bit, (h, w) mono or (h, w, 3) RGB."""
        electrons = self.expected(i, state)
        rng = np.random.default_rng(np.random.SeedSequence(self.scene.seed, spawn_key=(1, i)))
        electrons = self.camera.add_noise(electrons, rng)
        if self.scenario.nuisances.sharpening is not None:
            electrons = self.scenario.nuisances.sharpening.apply(electrons)
        return self.camera.encode(electrons)

    # -- ground truth and the reference feed --------------------------------------------------------
    def markers_visible(self, state: FrameState) -> list[int]:
        markers = self.scene.markers
        if markers is None:
            return []
        camera = self._view(state.camera_key)["camera"]
        seen = fiducials.visible(markers, camera.h_mm_to_px, camera.resolution)
        hidden = {i for i in seen for part in state.occluder
                  if len(clip_convex(rect_polygon(*markers.footprint(i)), part)) >= 3}
        return [i for i in seen if i not in hidden]

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
        for p, scheduled, applied in zip(self.scenario.perturbations, state.scheduled, state.applied, strict=True):
            entry = {"projector": p.projector, "scheduled": scheduled, "applied": applied, **p.tag(applied)}
            if p.kind != "shift":
                entry["pivot_mm"] = [float(v) for v in p.pivot]
            perturbations.append(entry)
        nz = self.scenario.nuisances
        return {
            "i": i,
            "t_s": float(state.t),
            "content": {
                "tag": "+".join(tags),
                "segments": [["/".join(str(k) for k in key), float(w)] for key, w in state.segments],
                "frames_in_exposure": len(state.segments),
                "cut_in_exposure": any(not sequence.same_shot(a, b)
                                       for (a, _), (b, _) in zip(state.segments, state.segments[1:], strict=False)),
            },
            "truth": {
                "aligned": offset < ALIGNED_MM,
                "offset_mm": offset,
                "offset_px": offset / self.pitch_mm,
                "h_rel": rel,
                "h_actual": {n: state.h_actual[n].tolist() for n in self.setup.names},
                "boxes_mm": {n: box_mm(state.h_actual[n], self.setup.resolution[n]).tolist() for n in self.setup.names},
                "camera_h_mm_to_px": self._view(state.camera_key)["camera"].h_mm_to_px.tolist(),
                "markers_visible": self.markers_visible(state),
            },
            "perturbation": perturbations,
            "nuisances": {
                "ambient": state.ambient,
                "bezel_light": self.scene.screen.bezel.light,
                "lamp_gain": dict(state.gains),
                "camera_bump": list(state.camera_key),
                "occluder": bool(state.occluder),
                "flicker": sorted({f.projector for f in nz.flickers}),
                "sharpening": nz.sharpening is not None,
            },
        }

    def source(self, i: int, ring: int = 10) -> list[tuple[float, ContentKey, np.ndarray]]:
        """Reference feed at frame i: the last `ring` pictures sent before the exposure ends, newest first.

        Each is (send time in seconds, content key, picture). The frame shows what was sent
        ``reference.lag_s`` earlier, which is among them when the ring is long enough. The key is
        the simulator's own label for the picture: the harness must strip it before the detector
        sees the feed, which has to match pictures by their content alone.
        """
        if not self.scenario.reference["available"]:
            raise ValueError(f"{self.scenario.variant}: the scenario has no source feed (reference.available)")
        end = self.time(i) + self.scenario.timing.exposure - EPSILON  # the exposure is [t, t + exposure)
        sent = self.scenario.sequence.sent_before(end, ring)
        return [(float(t), key, self.scenario.sequence.image(key)) for t, key in sent]

    def camera_for(self, state: FrameState) -> Camera:
        """The camera as it sees this frame: moved by any knocks so far."""
        return self._view(state.camera_key)["camera"]

    def setup_dict(self) -> dict[str, Any]:
        return self.scenario.setup_dict()
