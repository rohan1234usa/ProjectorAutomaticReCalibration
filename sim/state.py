"""The state of a frame: everything that decides what the camera records during one exposure.

Frame i is exposed during [t, t + exposure_s). Its state collects, from the scenario's timeline:

  * the pictures shown during the exposure and their shares of it (``sim/sequence.py``);
  * where each projector's pixels land (``sim/perturb.py``), taken at the exposure's start --
    a projector does not move noticeably within 33 ms;
  * the room light, each projector's lamp gain, how far the camera has been knocked, and the
    outline of anyone crossing in front of the screen (``sim/nuisance.py``).

Nothing here renders, so a state is cheap; ground truth needs nothing else. Camera knocks are
quantized to 1% of their size: a knock that settles over a few seconds still gives the renderer
only a handful of views to cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from sim.perturb import h_actual as perturbed_geometry
from sim.scenario import Scenario
from sim.sequence import Segments

BUMP_QUANTUM = 0.01


@dataclass(frozen=True, eq=False)
class FrameState:
    index: int
    t: Fraction  # start of the exposure, seconds
    segments: Segments  # pictures shown during the exposure, with their share of it
    scheduled: tuple[float, ...]  # each perturbation's schedule value
    applied: tuple[float, ...]  # ... after quantization: what the geometry uses
    h_actual: dict[str, np.ndarray]
    ambient: float  # room light
    gains: dict[str, float]  # each projector's lamp, relative to calibration
    camera_key: tuple[float, ...]  # each camera bump's (quantized) strength; all zero = undisturbed
    people: tuple[tuple[float, tuple[np.ndarray, ...]], ...]  # each person passing: (reflectance,
    # silhouette polygons in screen mm), in the order listed (later ones in front); empty when nobody passes

    @property
    def occluder(self) -> tuple[np.ndarray, ...]:
        """Every silhouette polygon in the frame, of everyone passing."""
        return tuple(polygon for _, polygons in self.people for polygon in polygons)


def frame_state(scenario: Scenario, i: int) -> FrameState:
    t = scenario.timing.time(i)
    setup = scenario.scene.setup
    perturbations = scenario.perturbations
    scheduled = tuple(p.schedule.value(t) for p in perturbations)
    applied = tuple(p.multiplier(t) for p in perturbations)
    nz = scenario.nuisances
    base = scenario.scene.screen.ambient
    ambient = base + sum((room.ambient - base) * room.schedule.value(t) for room in nz.rooms)
    gains = {n: 1.0 for n in setup.names}
    for lamp in nz.lamps:
        gains[lamp.projector] *= lamp.value(t)
    camera_key = tuple(round(b.schedule.value(t) / BUMP_QUANTUM) * BUMP_QUANTUM for b in nz.bumps)
    people = []
    for person in nz.occluders:
        polygons = person.outline(t, scenario.scene.screen.size_mm[0])
        if polygons is not None:
            people.append((person.reflectance, tuple(polygons)))
    return FrameState(
        index=i,
        t=t,
        segments=scenario.sequence.segments(t, scenario.timing.exposure),
        scheduled=scheduled,
        applied=applied,
        h_actual=perturbed_geometry(setup, perturbations, applied),
        ambient=ambient,
        gains=gains,
        camera_key=camera_key,
        people=tuple(people),
    )
