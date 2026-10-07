# Projector Misalignment Detector — project brief

This repository builds a camera-based script that answers one question about two
edge-blended projectors forming a single combined image: **have they drifted out of
alignment since the last calibration?** Output is `YES` (misaligned, recalibrate) or `NO`,
plus the estimated offset and a confidence.

No hardware is available yet. Everything is developed and proven against a **simulator**
that renders the projectors, the screen and the camera, and generates **ground-truth
datasets** (some runs aligned, some misaligned by known amounts). The detector must never
read simulator internals; it only sees camera frames and the calibration software's
blending setup, exactly as it would on real hardware.

Read this whole file before writing code. If a simulation result contradicts a design
choice below, do not silently redesign: write the finding with numbers to
`docs/findings.md` and raise it.

---

## 1. Decisions already made (do not reopen)

1. **Only the combined image is ever shown.** The projectors are never shown one at a
   time and no test pattern is projected at runtime. Each projector's footprint on the
   screen must be measured passively from the combined image.
2. **Any arrangement that overlaps is possible.** Side by side, one above the other, one
   rotated relative to the other, meeting at a corner, different sizes, or one mostly on
   top of the other. Never assume a vertical band. The overlap is *measured* as the
   intersection of the two projector footprints, and every other region (core tiles,
   control areas, edge pieces) is derived from it.
3. **The state immediately after calibration is ground truth.** Everything captured then
   (footprints, profiles, reference frames) is what later checks compare against. A new
   calibration replaces the baseline.
4. **Boundary analysis is the primary tool.** Each projector's footprint (a convex
   quadrilateral on the screen, the "box") is measured after calibration and re-measured
   every `ADJUSTABLE_INTERVAL_IN_SECONDS`. The comparison of current boxes to calibrated
   boxes is content-independent and is what drives the alignment likelihood.
5. **The overlap artifact check is secondary and triggered.** When boundary analysis
   reports an offset above threshold (or on a slow timed safety check), the detector
   inspects the overlap for the four visible symptoms: double contours, brightness
   hotspots, color fringing, offset borders.
6. **SSIM is confirmation only, never primary.** SSIM against the post-calibration image
   changes whenever the *content* changes, even with the projectors perfectly aligned. It
   is used only when the current frame is recognized as the same content as a stored
   post-calibration reference frame (perceptual-hash match).
7. **Camera is fixed and locked.** Exposure, gain, white balance and focus are locked;
   in-camera sharpening and denoise are off. Printed fiducial markers on the screen frame
   give the camera→screen homography. Frames are kept in RGB and converted to linear light.
8. **Units are millimeters on the screen.** Projector pixels can differ in size between
   the two projectors, so offsets are measured in mm and reported also in "coarser
   projector pixels" for readability.
9. **YES means a human would notice.** The threshold starts from the viewing geometry
   (≈1 arcminute at the closest seat, i.e. `D · tan(1′)` mm) and is later learned from
   manual recalibrate-button presses. In simulation it is a swept parameter.

---

## 2. Terminology (use these names in code and docs)

| Term | Meaning |
| --- | --- |
| **box** | One projector's lit footprint on the screen: a convex quadrilateral, 4 corners in mm. `box_a`, `box_b`. |
| **overlap** | `intersect(box_a, box_b)`: a convex polygon. Any shape. |
| **core** | Points of the overlap where both blend weights are ≥ `CORE_MIN_WEIGHT`. Tiled with `TILE_MM` squares. |
| **control areas** | Regions just outside the overlap lit by only one projector: `only_a`, `only_b`. Used to tell content patterns from real ghosting. |
| **edge pieces** | The overlap's boundary cut into pieces of `EDGE_PIECE_MM`; each piece knows which projector owns it and its outward normal. |
| **outer boundary** | The combined image's outline. It is made of segments each owned by one projector. Always visible via that projector's raster edge (black level against the unlit screen) and, when content there isn't black, via the content border. |
| **inner edges** | Edges of one box that lie inside the other box. Faded to zero by blending in bright content; visible only via black level in dark content. |
| **blend map** | Each projector's 2-D fade weight over the screen, `a(x) + b(x) = 1` in linear light inside the overlap. From the blending setup, else estimated from distance to each projector's edge. |
| **baseline** | Everything captured right after calibration (section 4.5). |
| **symptom** | One of the four overlap artifacts: double contours, hotspots, color fringing, offset borders. |
| **offset** | Largest displacement anywhere in the overlap between current and calibrated state, in mm. |
| **relative homography** | 3×3 matrix mapping box A's frame to box B's; identity when aligned. 6 free params (shift, rotation, scale, shear) fitted first; 2 keystone params only if a pattern remains. |

---

## 3. Repository layout

```
projector_align/
  sim/                 # simulator: never imported by detector/
    screen.py          # screen size (mm), fiducial positions
    projector.py       # projector: resolution, homography px→mm, blend map, gamma, black level, color balance
    content.py         # content sources: slides, text, photos, video sequences, flat, dark, letterboxed
    camera.py          # camera: homography mm→px, PSF blur, noise, gamma, vignetting, optional CA / sharpening
    perturb.py         # misalignment injection: shift, rotation, scale, keystone; schedules
    nuisance.py        # camera bump, lamp dimming, room light, occluder, flicker
    scenario.py        # declarative scenario (YAML) → frame generator with per-frame ground truth
    dataset.py         # write/read datasets: frames (PNG, 16-bit linear) + metadata.jsonl
  detector/            # the real algorithm: sees only camera frames + blending setup
    config.py          # all tunables as a dataclass, loadable from YAML
    rectify.py         # fiducials → homography → rectified linear RGB canvas (mm grid)
    geometry.py        # Boxes, OverlapGeometry (overlap, core tiles, control areas, edge pieces, border crossings)
    boundary.py        # PRIMARY: measure boxes at calibration; per-interval re-measure; relative homography fit
    artifacts.py       # SECONDARY: double contours, hotspots, color fringing, offset borders
    ssim_matched.py    # CONFIRMATION: perceptual-hash reference library, SSIM terms on matched frames
    baseline.py        # capture/save/load the post-calibration baseline
    fusion.py          # combine boundary + artifact evidence into one offset estimate + confidence
    decision.py        # threshold, hysteresis, K-of-N voting, YES/NO + warnings
    runner.py          # the loop: baseline → every ADJUSTABLE_INTERVAL_IN_SECONDS → check → decide
  eval/
    metrics.py         # detection rate vs offset, FPR, latency, boundary availability, per-type confusion
    sweep.py           # run detector over a dataset with threshold sweeps
    report.py          # CSV + PNG plots + per-scenario diagnostic images
  scenarios/           # YAML scenario files (one per test idea; adding a test = adding a file)
  scripts/
    make_dataset.py    # python -m scripts.make_dataset scenarios/xxx.yaml out/xxx
    run_detector.py    # python -m scripts.run_detector out/xxx --config detector.yaml
    evaluate.py        # python -m scripts.evaluate out/xxx/results.jsonl
    visualize.py       # quick-look overlays for a frame: boxes, overlap, tiles, diff/SSIM maps
  tests/               # pytest; unit + property + regression (fixed seeds)
  docs/
    findings.md        # dated findings with numbers
    research/          # exported research doc (Markdown) for reference
CLAUDE.md
pyproject.toml
```

Hard rule: `detector/` imports nothing from `sim/`. The harness (`eval/`, `scripts/`) is the
only place both meet.

---

## 4. The detector, in order of trust

### 4.1 Rectification (`rectify.py`)
- Detect the 4+ fiducials, solve camera→screen homography (mm grid, `CANVAS_PX_PER_MM`).
- Convert to linear light (undo camera gamma). Keep R, G, B.
- Detect fiducial motion between checks; if the camera moved, re-solve and continue
  (camera motion must never produce a YES).

### 4.2 Boundary analysis — PRIMARY (`boundary.py`)
**At calibration** (`measure_boxes_at_start`):
- Prior: the boxes the blending setup reports (the calibration software knows where it
  put each frame). Treat as a starting guess, not truth.
- Evidence, from the combined image only:
  1. **Outer boundary segments.** Fit straight lines to the combined image's outline.
     Each segment belongs to one projector. Use both the content border (strong, when the
     content there isn't black) and the raster edge via black level (faint, always
     present — needs frame averaging).
  2. **Inner edges from dark frames.** When the overlap is dark, each projector's black
     level outlines its full frame, including edges hidden under the other projector.
     Average many dark frames.
  3. **Blend-ramp profiles.** In lit flat frames the fade across each overlap edge locates
     the owner's edge.
- Fit two convex quadrilaterals to all evidence (RANSAC lines → corners by intersection).
  Store corner uncertainty from the line-fit residuals.
- Keep refining during a trusted window (`TRUSTED_WINDOW_S`) after calibration as more
  dark/lit frames arrive.

**Every `ADJUSTABLE_INTERVAL_IN_SECONDS`** (`boundary_check`):
- Re-measure whichever edges are currently visible (outer segments nearly always; inner
  edges only in dark frames; border crossings when content there is lit).
- Each piece of evidence constrains motion only *across* its own edge. Require evidence at
  ≥2 distinct angles before fitting; otherwise return `None` ("boundary not visible this
  interval"), never a guess. Arrangements with rotated edges make this easier.
- Fit the relative homography B-vs-A against the calibrated boxes: 6 params first, 8 only
  if residuals show a pattern. Report `offset_mm` = max corner displacement of the overlap
  polygon, plus confidence from residuals and corner uncertainty.
- Log a **boundary availability** flag every interval; the evaluation harness measures
  how often the primary tool could answer, per content type.

### 4.3 Overlap artifact check — SECONDARY, triggered (`artifacts.py`)
Runs when `boundary_check` reports `offset_mm > TRIGGER_MM` (default = 0.5 × threshold),
or on the timed safety check every `SAFETY_CHECK_S` regardless. Pools `POOL_FRAMES` usable
frames (skip motion/cut frames, clipped frames), then evaluates:

1. **Double contours.** Cepstral echo peak in the core tiles, minus the same statistic in
   the nearest control areas (content patterns appear in both; a real ghost only inside).
   Returns offset vector per tile; tiles must agree on one smooth 6-param model.
2. **Brightness hotspots.** 2-D block-mean brightness of the overlap, normalized per frame,
   vs the baseline map. Shape classifies cause: band along one overlap edge = shift across
   that edge; gradient along the overlap = rotation; one projector's whole side = lamp
   (warning, not misalignment).
3. **Color fringing.** Echo offset per channel (R, G, B) minus the lens's own per-channel
   offset measured at calibration. Same offset in all channels = projectors moved;
   channel spread = convergence fault (warning).
4. **Offset borders.** Where the combined image's border runs through the overlap, the
   step between A's and B's border position vs the calibrated step. Absent in arrangements
   where no border crosses the overlap.

Each symptom returns `offset_mm` or `None` (content can't show it). `agreeing` = number of
symptoms independently above threshold.

### 4.4 Matched-frame SSIM — CONFIRMATION only (`ssim_matched.py`)
- During the trusted window after calibration, store perceptual hashes + frames of what
  was on screen (`ReferenceLibrary`).
- When a current frame hashes within `HASH_MATCH_BITS` of a stored one, compute SSIM
  *terms separately* (luminance, contrast, structure) over the overlap after removing a
  global gain/offset. Structure drop ⇒ geometric; luminance drop with structure intact ⇒
  intensity. Estimate shift from the difference image via the gradient (optical-flow
  style) method.
- Never run SSIM on unmatched content. Never let it alone produce a YES.

### 4.5 Baseline (`baseline.py`)
Captured after every calibration: homography, `box_a`, `box_b`, `OverlapGeometry`,
residual relative homography (≈ identity), edge-profile pools (lit and dark), hotspot map,
per-channel lens offset, border steps, reference library, noise floor (SSIM/PSNR between
two independent captures of the same frame), timestamp.

### 4.6 Fusion and decision (`fusion.py`, `decision.py`)
- Primary estimate = boundary `offset_mm`. If artifacts ran: agree within `AGREE_MM` ⇒
  take the max and raise confidence; artifacts say more than boundary with ≥2 symptoms
  agreeing ⇒ take artifacts; boundary says more than artifacts ⇒ suspect edge fit or
  camera, hold and re-check next interval.
- Decision: `offset_mm > TOLERANCE_MM` votes YES; YES after `YES_VOTES = (3, 4)`; back to
  NO only below `CLEAR_RATIO × TOLERANCE_MM`. `None` ⇒ hold previous answer.
- Output per interval: `answer`, `offset_mm`, `offset_px`, `confidence`, `boundary_available`,
  `symptoms`, `warnings` (lamp, convergence, camera-moved), all logged as JSONL.

---

## 5. The simulator (`sim/`)

Purpose: generate camera frames with known truth. Fidelity matters more than speed, but
keep a `fast` quality preset for tests.

- **Screen**: width/height mm, fiducial positions (4 corners of the frame), unlit screen
  reflectance (so black level shows), optional screen gain/vignetting.
- **Projector** (×2): resolution, homography px→mm (set from an arrangement preset or
  explicit corners), blend map (from distance to its own edges, linear or cosine ramp),
  gamma, brightness, black level (non-zero!), color balance, optional DLP flicker/banding.
- **Content**: library of sources — slide decks (text-heavy, graphics), photos, video
  clips (motion, cuts), flat gray, flat colors, black, letterboxed (black borders), stripes
  of known period (to test repetition limits). Sequences with timestamps.
- **Rendering**: for each frame, warp content through each projector's homography, apply
  blend, gamma, black level; sum in linear light on the screen (mm grid); then camera.
- **Camera**: resolution, homography mm→px (presets: whole screen + 5% margin; zoomed on
  overlap + margins), PSF blur (Gaussian σ in px), shot + read noise, exposure/gain (locked),
  gamma, vignetting, optional chromatic aberration, optional in-camera sharpening (to
  reproduce that failure mode). Outputs 16-bit RGB.
- **Perturbation** (`perturb.py`): applied to A, B or both as a modification of the
  projector homography: shift (dx, dy mm), rotation (θ about a point), scale (s), keystone
  (h31, h32). Schedules: `step(t0)`, `drift(rate per hour)`, `bump_then_hold`,
  `oscillate(period)` (thermal), `none`.
- **Nuisances** (`nuisance.py`): camera bump (changes camera homography, not projectors),
  lamp dimming of one projector, room-light step, occluder (a person-shaped blob for N
  frames), flicker. These must never cause YES.
- **Ground truth per frame**: true relative homography, true `offset_mm` (max over the
  true overlap polygon), true `offset_px`, perturbation type tags, aligned flag
  (`offset_mm == 0`), nuisance tags, content type tag, timestamp. Written to
  `metadata.jsonl`; frames as PNG. Scenarios are deterministic given `seed`.

Scenarios are YAML files in `scenarios/`. Adding a test idea must mean adding a YAML file,
not code. Example keys: `arrangement`, `projectors`, `camera`, `content`, `perturbation`,
`nuisances`, `duration_s`, `frame_rate`, `seed`.

---

## 6. Evaluation harness (`eval/`)

For every dataset, run the detector with the interval `ADJUSTABLE_INTERVAL_IN_SECONDS` and
produce:

- **Detection rate vs true offset** (a psychometric-style curve) per misalignment type,
  per arrangement, per content type.
- **False positive rate** on aligned runs, with and without nuisances.
- **Latency**: intervals from onset to the first YES.
- **Boundary availability rate**: fraction of intervals where the primary tool could
  answer, per content type. This is the key number for the "boundary first" decision.
- **Offset accuracy**: estimated vs true offset (mean abs error in mm and px).
- **Symptom confusion**: which symptoms fired for which true type.
- **Threshold sweep**: all of the above across `TOLERANCE_MM` values.
- **Diagnostic images** per scenario: box overlays on the rectified frame, overlap/core/
  control masks, edge profiles vs baseline, difference maps, SSIM term maps, cepstrum
  peaks. These are for humans; keep them easy to open.

Reports: `results.jsonl`, `summary.csv`, PNG plots, `report.md` per run.

---

## 7. Initial scenario catalogue

Create these YAML scenarios first; each exists to answer a specific question.

| Scenario | Question it answers |
| --- | --- |
| `aligned_slides` | FPR on static text-heavy content, no perturbation. |
| `aligned_video` | FPR under motion, cuts, dark scenes. |
| `aligned_nuisances` | Camera bump, lamp dimming, room light, occluder ⇒ still NO? |
| `shift_sweep` | Steps of 0.25, 0.5, 1, 2, 4, 8 px shift across and along the overlap edge; detection curve. |
| `rotation_sweep` | Small rotations of B about its center and about a far corner. |
| `scale_keystone` | Zoom and tilt perturbations; does the 6→8 param escalation work? |
| `slow_drift` | 2 px over 2 hours; latency and timed safety check. |
| `arrangements` | Same shift sweep across side-by-side, stacked, rotated, corner, different-size, large-overlap. |
| `boundary_hidden` | Letterboxed content + bright overlap: how often is the boundary unavailable, and does the overlap check catch what the boundary can't? |
| `repetition_limit` | Stripes of period P with shifts near P/2 and P: measure the aliasing limit. |
| `recurring_frames` | Title slide shown after calibration and again later: does matched SSIM help? |
| `camera_zoomed` | Camera on the overlap only vs whole screen: smallest detectable offset. |

---

## 8. Build order (each phase ends with tests passing and a short `docs/findings.md` entry)

1. **Scaffold + simulator core.** `pyproject.toml`, package layout, config dataclasses.
   Render an aligned side-by-side scene through the camera; `scripts/visualize.py` shows
   it. Done: a human can look at the PNG and see a seamless image with a faint black-level
   raster around it.
2. **Perturbation + dataset generator.** Scenario YAML → frames + `metadata.jsonl` with
   per-frame truth. Done: `shift_sweep` and `aligned_slides` datasets generate
   deterministically; a test asserts truth offsets match the injected homographies.
3. **Rectification + geometry.** Fiducials → homography → mm canvas; `OverlapGeometry`
   from two boxes of any shape. Done: property tests on random convex quads (overlap
   polygon, core tiles, control areas, edge pieces all valid for rotated/corner cases).
4. **Boundary analysis (primary).** Calibration measurement + per-interval re-measure +
   relative homography fit. Done: on `shift_sweep` and `rotation_sweep`, estimated offset
   within 0.5 px of truth whenever `boundary_available`; `None` (not a wrong number) when
   edges are hidden.
5. **Runner + decision.** Baseline capture, interval loop, hysteresis/voting, JSONL output.
   Done: `aligned_*` datasets produce zero YES; `shift_sweep` at 4 px produces YES within
   `YES_VOTES` intervals.
6. **Overlap artifact check.** The four symptoms + fusion. Done: `boundary_hidden` shows
   the artifact check catching offsets the boundary layer could not see; nuisances still
   produce no YES.
7. **Evaluation harness.** Metrics, sweeps, plots, diagnostic images, `report.md`.
   Done: one command evaluates every scenario and writes the detection curves.
8. **Confirmation + learning.** Matched-frame SSIM, reference library, threshold learning
   from simulated "button presses" (ground truth offsets above a hidden human threshold).
9. **Stretch.** Reference mode (video feed known) via multi-frame Wiener deconvolution;
   multi-projector grids; camera zoomed on overlap.

Do not start phase N+1 before phase N's "done" condition holds.

---

## 9. Engineering conventions

- Python ≥ 3.11, `numpy`, `opencv-python-headless`, `scikit-image` (SSIM), `pyyaml`,
  `pytest`, `matplotlib` (plots only). No GUI dependencies; everything runs headless.
- Type hints everywhere; `@dataclass` for configs and results; no global mutable state in
  `detector/` (the runner owns state).
- Determinism: every random draw takes a `numpy.random.Generator` seeded from the scenario.
- Linear light inside; gamma only at the camera output and content input.
- Every module has a docstring that explains the physics it models or exploits, in plain
  words, before the code. A reader with no computer-vision background should understand
  *why* each step exists.
- Keep modules under ~300 lines; split rather than grow.
- Tests: unit tests per function, property tests for geometry, regression tests on fixed
  seeds with tolerances stated in mm. A test that asserts a symptom fires must also assert
  it does *not* fire on the matching aligned scenario.
- Logging: structured JSONL per interval. Never print tensors.
- Commit per phase with a message that states which "done" condition was met and the
  numbers that show it.

---

## 10. Config (names are the contract; defaults are starting guesses)

```yaml
ADJUSTABLE_INTERVAL_IN_SECONDS: 30      # boundary check cadence
SAFETY_CHECK_S: 900                     # timed artifact check regardless of boundary result
TRUSTED_WINDOW_S: 600                   # after calibration: refine boxes, learn references
POOL_FRAMES: 60                         # usable frames pooled per artifact check
TOLERANCE_MM: 1.7                       # ≈ 1 arcmin at 6 m; swept in evaluation
TRIGGER_MM: 0.85                        # boundary offset that triggers the artifact check
AGREE_MM: 1.4                           # layers "agree" within this
YES_VOTES: [3, 4]                       # K of N intervals
CLEAR_RATIO: 0.5
CORE_MIN_WEIGHT: 0.2
TILE_MM: 64
CTRL_MARGIN_MM: 40
EDGE_PIECE_MM: 80
CANVAS_PX_PER_MM: 2.0
HASH_MATCH_BITS: 6
MIN_PEAK_SNR: 6.0
MIN_GOOD_TILES: 3
```

---

## 11. Out of scope for now

Real camera drivers, projector control, reading the recalibrate button, GUIs, grids of
more than two projectors (keep the geometry general so it can come later), stereo/3D
content, curved screens.
