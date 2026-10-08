# Projector Misalignment Detector — project brief

This repository builds a camera-based script that answers one question about two
edge-blended projectors forming a single combined image: **have they drifted out of
alignment since the last calibration?** Output is `YES` (misaligned, press recalibrate) or
`NO`, plus the estimated offset and a confidence. Today a professor checks this by eye in a
lecture hall and presses the calibration software's recalibrate button; the script replaces
that visual check.

The design comes from the original pseudocode, `docs/research/pseudocode.md` (Oct 1, 2026).
This brief keeps that document's methods for checking the overlap, its frame routing,
decision rule, edge cases and stress tests, and generalizes it in three ways:
- the two projectors may sit in any overlapping arrangement, at angles to each other;
- the camera sees the whole picture;
- boundary analysis of each projector's rectangle is the primary tool.

No hardware is available yet. Everything is developed and proven against a **simulator**
that renders the projectors, the screen and the camera, and generates **ground-truth
datasets** (some runs aligned, some misaligned by known amounts). The detector must never
read simulator internals. It only sees what it would see on real hardware:
- camera frames;
- the calibration software's blending setup;
- in reference mode, the video frames sent to the projectors.

Read this whole file before writing code. If a simulation result contradicts a design
choice below, do not silently redesign: write the finding with numbers to
`docs/findings.md` and raise it.

**Status:** Phases 1 and 2 (the simulator) are done; Phase 3 (detector inputs and geometry) is
next (section 8).

---

## 1. Decisions already made (do not reopen)

1. **Only the combined image is ever shown.** The projectors are never shown one at a
   time and no test pattern is projected at runtime. Each projector's footprint on the
   screen must be measured passively from the combined image.
2. **Any arrangement that overlaps is possible.** Side by side, one above the other, one
   rotated relative to the other, meeting at a corner, different sizes, or one mostly on
   top of the other. Never assume a vertical band. The overlap is *measured* as the
   intersection of the two projector footprints. Every other region (core tiles, control
   tiles, hotspot blocks, edge pieces) is derived from it.
3. **The state immediately after calibration is ground truth.** Everything captured then
   (footprints, profiles, pools) is what later checks compare against. A new calibration
   replaces the baseline and resets the answer to NO.
4. **Boundary analysis is the primary tool.** Each projector's footprint (a convex
   quadrilateral on the screen, the "box") is measured after calibration and re-measured
   every `ADJUSTABLE_INTERVAL_IN_SECONDS`, from the parts of its outline visible in the
   combined image: its outer edges against the unlit screen, and its inner edges in dark
   frames. The comparison of current boxes to calibrated boxes is content-independent and
   is what drives the alignment likelihood.
5. **The overlap check is secondary and triggered.** It runs when boundary analysis reports
   an offset above `TRIGGER_MM`, or on a slow timed safety check. It inspects the overlap
   for three signs of misalignment, using the pseudocode's methods:
   - **double contours**: a cepstral echo in core tiles that is missing from the control
     tiles beside the overlap;
   - **brightness hotspots**: the overlap is brighter or darker than at calibration. This
     is fitted as a shift across the overlap edge, kept separate from lamp gain and trend;
   - **offset borders**: where a content border crosses the overlap, A's and B's copies of
     it no longer coincide.

   Color fringing is not checked.
6. **SSIM is not used.** SSIM against the post-calibration image changes whenever the
   *content* changes, even with the projectors perfectly aligned. Its intended role,
   confirmation on recurring content, is taken by reference mode (decision 10).
7. **Camera is fixed and locked.**
   - Exposure, gain, white balance and focus are locked. Exposure is a multiple of the
     projector refresh period, so there is no flicker banding.
   - In-camera sharpening and denoise are off.
   - Frames are 16-bit linear **luminance** (mono).
   - Printed passive ArUco markers on the screen bezel give the camera→screen homography.
     They sit at the four corners and above and below the overlap. Being passive, they need
     room light (about 0.02 of projector white or more) or a little light of their own.
8. **Units are millimeters on the screen.** Projector pixels can differ in size between
   the two projectors, so offsets are measured in mm and reported also in "coarser
   projector pixels" for readability.
9. **YES means a human would notice.** The threshold starts from the viewing geometry
   (≈1 arcminute at the closest seat, i.e. `D · tan(1′)` mm) and is later learned from
   manual recalibrate-button presses. In simulation it is a swept parameter.
10. **Two input modes.**
    - *Blind mode* uses only the camera. It is the default and is built first.
    - *Reference mode* also reads the video frame being sent to the projectors (a screen
      capture or an HDMI splitter). It measures where each projector places the content by
      multi-frame Wiener deconvolution. It has no resolution floor, is harder to fool, and
      is built right after blind mode.
11. **Every frame type is useful, and "can't tell" beats a guess.** Frames are routed by
    kind:
    - motion, cuts and clipped frames are skipped;
    - textured regions feed the echo and the edge fits;
    - flat regions feed the hotspot fit;
    - dark regions outline each projector's raster through its black level.

    A measurement that cannot see returns `None` (or an upper bound), never 0. `None` holds
    the previous answer.
12. **Self-referencing.** Measurements compare projector A to projector B. Boxes are fitted
    jointly with a shared camera-motion term, so a bumped camera cannot fake a YES.
    Nuisances must never produce YES: camera bump, lamp dimming, room light, people walking
    past, flicker.

---

## 2. Terminology (use these names in code and docs)

| Term | Meaning |
| --- | --- |
| **box** | One projector's lit footprint on the screen: a convex quadrilateral, 4 corners in mm. `box_a`, `box_b`. |
| **overlap** | `intersect(box_a, box_b)`: a convex polygon, any shape. The pseudocode's "band" is the side-by-side special case. |
| **core** | Points of the overlap where both blend weights are ≥ `CORE_MIN_WEIGHT`. Tiled with `TILE_MM` squares (core tiles). |
| **control tiles** | For each core tile: a tile just outside the overlap in the region lit by A only (`ctrl_a`), and one in the region lit by B only (`ctrl_b`). Found by walking along the blend gradient. They show the same kind of content with no echo. A pattern found inside and outside the overlap is content; one found only inside is misalignment. |
| **control areas** | `only_a`, `only_b`: the regions lit by one projector only. |
| **edge pieces** | Box outlines cut into pieces of `EDGE_PIECE_MM`. Each piece knows which projector owns it, its outward normal, and how it can be seen (content border, black level, blend ramp). |
| **outer boundary** | The combined image's outline. It is made of segments each owned by one projector. Always visible via that projector's raster edge (black level against the unlit screen) and, when content there isn't black, via the content border. |
| **inner edges** | Edges of one box that lie inside the other box (and inside the content). Faded to zero by blending in bright content; visible only via black level in dark content. |
| **blend map** | Each projector's 2-D fade weight over the screen, `a(x) + b(x) = 1` in linear light inside the overlap. From the blending setup, else estimated from distance to each projector's inner edges. |
| **blending setup** | What the calibration software knows and the detector may read: each projector's calibrated homography and resolution, the content rect, the blend shape (hence the blend maps), and the marker layout. |
| **content rect** | The rectangle of screen the calibration software fills with content. Projector pixels outside it show black. |
| **echo** | The second copy of the content in the overlap when misaligned: `O ≈ a·S(x − d_A) + b·S(x − d_B)`. The **cepstrum** `|F⁻¹{log|F{O}|}|` turns it into a peak at the offset `d = d_B − d_A`. |
| **hotspot map** | Per-block brightness of the overlap relative to the single-projector flanks, compared with its value at calibration. This is the pseudocode's seam profile, in 2-D. |
| **kernel** | Reference mode: the Wiener-deconvolution kernel between the source frame and the camera image of a tile. It peaks where the projector puts the content. |
| **source frame** | Reference mode: the video frame sent to the projectors, with a timestamp. The camera lags it by a few frames. |
| **pools** | Per-tile and per-block accumulators filled frame by frame and evaluated on demand: log spectra, cross spectra, brightness ratios, edge profiles. |
| **frame kind** | `skip` (motion, cut, clipped, camera moved), `dark`, `flat` or `textured`. Decided per frame and per region. |
| **floor / unresolved** | The floor is the smallest offset a method can resolve. An echo detected below its floor is `unresolved` (present, size unknown). No detection returns `None` with `upper_bound = floor`. |
| **field** | The smooth motion model fitted to local measurements: shift (2 params), shift + rotation (3), affine (6), homography (8). |
| **relative homography** | 3×3 matrix mapping box A's frame to box B's; identity when aligned. 6 free params (shift, rotation, scale, shear) fitted first; 2 keystone params only if a pattern remains. |
| **baseline** | Everything captured right after calibration (section 4.5). |
| **symptom** | One of the three overlap signs: double contours, hotspots, offset borders. |
| **offset** | Largest displacement anywhere in the calibrated overlap between the current and the calibrated state, in mm. |
| **bezel** | The frame around the screen surface; it carries the markers. |

---

## 3. Repository layout

The tree's root is this repository's root; commands run from here. `[done]` marks what
exists.

```
CLAUDE.md
README.md               # short public overview and quick start                     [done]
pyproject.toml          # uv project, no package build: `uv sync`, `uv run pytest`   [done]
detector.yaml           # section 10 defaults                                       [done]
sim/                    # simulator: never imported by detector/; imports nothing from detector/
  cfg.py                # strict YAML readers: unknown keys, numbers ("1e-3"), exact Fraction times [done]
  planar.py             # homographies, convex polygons, the one image warp (sim's own geometry)  [done]
  screen.py             # screen, bezel, wall, room light; the static reflectance map             [done]
  fiducials.py          # ArUco marker layout and exact rendering on the bezel                    [done]
  projector.py          # resolution, homography px→mm, gamma, black level, colour balance, mono  [done]
  calibration.py        # the blending setup: H_cal, content rect, blend maps, black uplift, framebuffers [done]
  arrangements.py       # presets: side by side, stacked, rotated, corner, different sizes, large overlap [done]
  content.py            # still images: flat, black, text slides at three densities               [done]
  textures.py           # photo-like textures, dark film, stripes, letterbox, blank-overlap content [done]
  video.py              # synthetic video: moving objects, pans, cuts, fades                      [done]
  pictures.py           # content item kinds (deck, held, flat, black, photo, dark, stripes, video) [done]
  sequence.py           # content over time; exposure straddles; what was sent when (reference)  [done]
  camera.py             # homography mm→px, PSF, pixel integration, vignetting, noise, 16-bit; whole/zoomed [done]
  schedule.py           # how a change unfolds: step, staircase, drift, ramp, bump_then_hold, oscillate [done]
  perturb.py            # misalignment injection on h_actual: shift, rotation, scale, keystone    [done]
  nuisance.py           # camera bump, lamp dimming, room light, occluder, sharpening             [done]
  flicker.py            # per-projector flicker and rolling-shutter banding                      [done]
  state.py              # FrameState: everything that decides one exposure (no rendering)       [done]
  truth.py              # per-frame ground truth: offset_mm/px (symmetric), relative homography  [done]
  render.py             # the render chain as cached linear components: content → light → camera [done]
  frames.py             # FrameSource: on-demand deterministic frames with render caches         [done]
  sweep.py              # scenario files: extends (inheritance) and sweeps (variants)            [done]
  scenario.py           # declarative scenario (YAML) → objects                                  [done]
  dataset.py            # write/read datasets: metadata.jsonl, setup.json, optional 16-bit PNG frames [done]
detector/               # the real algorithm: camera frames + blending setup (+ source frames) only
  inputs.py             # CameraFrame, BlendingSetup, SourceFrame: the detector's input contract
  config.py             # all tunables as a dataclass, loadable from YAML                         [done]
  rectify.py            # markers → homography → rectified linear canvas (mm grid); camera motion
  polygon.py            # convex polygon helpers (the detector's own)
  blending.py           # blend maps from the blending setup
  geometry.py           # OverlapGeometry: overlap, core + control tiles, hotspot blocks, edge pieces, border crossings
  classify.py           # frame routing: skip / dark / flat / textured
  pools.py              # EchoPool, SeamPool, edge accumulators
  edges.py              # sub-pixel edge-piece measurement (content border, black level, blend ramp)
  field.py              # smooth motion-field fits (2/3/6/8 params), RANSAC with sign handling
  boundary.py           # PRIMARY: boxes at calibration; per-interval re-measure; joint A/B fit
  echo.py               # SECONDARY: double contours (cepstrum minus control tiles)
  seam.py               # SECONDARY: hotspot-map fit (shift across edges, lamp gain, trend)
  borders.py            # SECONDARY: offset borders
  artifacts.py          # SECONDARY orchestrator: pools → symptoms → offset, agreeing count
  source.py             # reference mode: source ring buffer, lag matching
  kernel.py             # reference mode: Wiener kernels, kernel offset
  baseline.py           # capture/save/load the post-calibration baseline
  fusion.py             # combine boundary + overlap evidence into one offset estimate + confidence
  decision.py           # threshold, hysteresis, K-of-N voting, YES/NO + warnings
  runner.py             # the loop: baseline → every interval → check → decide → JSONL
  results.py            # result dataclasses
eval/
  feed.py               # the only module importing both sim/ and detector/: FrameSource → detector inputs
  metrics.py            # detection rate vs offset, FPR, latency, availability, per-type confusion
  sweep.py              # run the detector over datasets with threshold sweeps
  report.py             # CSV + PNG plots + per-scenario diagnostic images + report.md
scenarios/              # YAML scenario files (one per test idea; adding a test = adding a file); _lecture_hall.yaml is the shared base
scripts/
  make_dataset.py       # python -m scripts.make_dataset scenarios/xxx.yaml out/xxx [--frames all|sample|none]  [done]
  check_dataset.py      # python -m scripts.check_dataset out/xxx: recorded truth vs injected offsets, paired variants [done]
  check_geometry.py     # the checker's own plane geometry, independent of sim/                  [done]
  run_detector.py       # python -m scripts.run_detector out/xxx --config detector.yaml
  evaluate.py           # python -m scripts.evaluate out/xxx/results.jsonl
  visualize.py          # quick look at one frame, with the true geometry drawn on top            [done]
tests/                  # pytest; unit + property + regression (fixed seeds); -m slow for demo scale [done: Phases 1, 2]
docs/
  findings.md           # dated findings with numbers
  research/pseudocode.md  # the original pseudocode this brief generalizes
```

Hard rules:
- `detector/` imports nothing from `sim/`, and `sim/` imports nothing from `detector/`.
  `tests/test_imports.py` enforces both.
- The harness (`eval/`, `scripts/`) is the only place the two meet. `eval/feed.py` is the
  only module that turns simulator output into detector input.
- The detector never sees ground truth.

---

## 4. The detector, in order of trust

### 4.1 Rectification (`rectify.py`)
- Detect the bezel markers (≥ 4 must be visible).
- Solve the camera→screen homography from marker **centres**, onto a mm grid at
  `CANVAS_PX_PER_MM`. Detected marker corners carry an inward bias (Phase 2: 0.65 px on
  average and up to 1 px unrefined, 0.27 px with sub-pixel refinement); centres cancel it.
  Phase 2 also found that ArUco's own sub-pixel corners give single-frame centres too noisy
  for 0.2 px (p95 0.32 px), while straight-line fits to each marker's four outer edges give
  p95 0.09 px. ArUco needs an 8-bit image scaled to the marker paper's level, not the frame's
  maximum (`docs/findings.md`, 2026-10-07).
- Frames are already linear (camera gamma 1). Subtract the pedestal and scale to relative
  radiance; pedestal and gain are known camera settings and part of the detector's inputs.
- Track marker motion between frames. If the markers move more than `FIDUCIAL_MOVE_PX`:
  re-solve, raise a `camera_moved` warning and reset the pools. Camera motion must never
  produce a YES.

### 4.2 Boundary analysis — PRIMARY (`boundary.py`, `edges.py`, `field.py`)
**At calibration** (`measure_boxes_at_start`):
- Prior: the boxes the blending setup reports (the calibration software knows where it put
  each frame). Treat them as a starting guess, not truth.
- Evidence comes from the combined image only, as edge pieces:
  1. **Outer boundary segments.** Each belongs to one projector. Use both the content
     border (strong, when the content there isn't black) and the raster edge via black
     level (faint but always present; needs averaging over frames and along the edge).
  2. **Inner edges from dark frames.** When the overlap is dark, each projector's black
     level outlines its full frame, including edges hidden under the other projector.
  3. **Blend-ramp profiles.** In lit flat frames, the fade across each inner edge locates
     the owner's edge.
  4. **Border crossings.** Where a content border crosses the overlap, each projector's
     copy of it locates that projector.
- Fit two convex quadrilaterals to all the evidence (robust line fits, corners by
  intersection). Store corner uncertainty from the fit residuals.
- Keep refining during the trusted window (`TRUSTED_WINDOW_S`) after calibration, as more
  dark and lit frames arrive.

**Every `ADJUSTABLE_INTERVAL_IN_SECONDS`** (`boundary_check`):
- Re-measure whichever pieces are visible now:
  - each piece constrains motion only *across* its own edge, so it gives one observation
    n̂·u, with σ from its edge-fit covariance;
  - pieces below `MIN_EDGE_SNR` are not used.
- Require evidence at ≥ 2 distinct angles per box before fitting. Otherwise return `None`
  ("boundary not visible this interval"), never a guess.
- Fit A and B **jointly**, with a shared camera-motion term (shift, optionally rotation),
  by weighted least squares with robust (Huber) weights. This generalizes the pseudocode's
  "(ΔB − ΔA)": residual homography error and camera bumps cancel, and only relative motion
  remains.
- Fit the relative homography B-vs-A against the calibrated boxes: 6 params first, 8 only
  if the residuals show a pattern.
- Report `offset_mm` = the largest displacement over the calibrated overlap polygon, plus a
  confidence from the residuals and corner uncertainty.
- Precision budget (a planning estimate, to confirm in Phase 4):
  - a black-level edge piece of 80 mm localizes to ≈ 0.11 mm in one frame and ≈ 0.02 mm
    over 30 frames;
  - a content-border piece localizes to ≈ 0.004 mm;
  - the whole-screen camera, at ≈ 0.87 camera px per mm, resolves corners to ≲ 0.05 mm per
    interval, ten times finer than the 0.5 px target.
- Log a **boundary availability** flag every interval. The evaluation harness measures how
  often the primary tool could answer, per content type.

### 4.3 Overlap check — SECONDARY, triggered (`artifacts.py`, `echo.py`, `seam.py`, `borders.py`)
Runs when `boundary_check` reports `offset_mm > TRIGGER_MM` (default 0.5 × tolerance), or on
the timed safety check every `SAFETY_CHECK_S` regardless. These are the pseudocode's
methods, generalized from a band to any overlap.

**Routing and pools** (`classify.py`, `pools.py`)
- Frames are sampled every `SAMPLE_EVERY_S`. A frame is skipped when:
  - the mean absolute difference from the previous sample exceeds `MOTION_LEVEL` (motion or
    a cut; the camera may have blended two video frames, which looks like a ghost);
  - more than `MAX_CLIPPED` of its pixels saturate;
  - the markers moved.
- Each region (tile, block, piece) is then `dark` (mean below `DARK_LEVEL` × white),
  `textured` (edge density above `MIN_TILE_EDGES`) or `flat`.
- In blind mode a frame is kept only if it differs from the last kept one (NCC below
  `VARIETY_MAX_NCC`): the echo needs varied content.
- Pools fill continuously, bounded to `POOL_FRAMES`, and are evaluated when the check runs.
  A tile needs `MIN_TILE_FRAMES` textured frames; a block needs `MIN_SEAM_FRAMES` flat (or
  dark) frames.

**1. Double contours** (`echo.py`)
- Per core tile, pool the log magnitude spectrum of the tile after a high-pass and a Hann
  window; likewise for its control tiles. The high-pass must be *linear*, so the two-copy
  model stays intact.
- Cepstrum = |IFFT(mean log spectrum)|, then a robust z-score (median, 1.4826·MAD).
- `diff = z(core) − mean z(controls)`. Content patterns appear in both and cancel; the echo
  appears only in the core.
- Zero a disk around the origin, of radius equal to the tile's floor. Take the maximum and
  refine it sub-pixel.
- Accept the peak only if it is above `MIN_PEAK_SNR` and above `NULL_FACTOR` × the
  control-vs-control null, max|z(ctrl_a) − z(ctrl_b)|.
- Tiles are cut on the **camera grid**, not the rectified canvas: resampling locks cepstral
  peaks to multiples of the camera pitch. Peaks are converted to mm with the camera
  homography's local Jacobian.
- **Floor** = max(`MIN_OFFSET_MM`, `ECHO_FLOOR_CAMERA_PX` / local camera px per mm). That is
  ≈ 2.9 mm for the whole-screen camera and ≈ 1.4 mm zoomed (planning estimates).
  - A peak below the floor is `unresolved`.
  - No peak returns `None` with `upper_bound = floor`. Never 0: that would veto a real
    1.7–3 mm offset.
- The echo's sign is ambiguous (+d or −d). Tiles must agree on one smooth field:
  - fitted by `field.py` with RANSAC over sign choices, inlier distance `INLIER_MM`;
  - an inlier fraction of at least `LINE_INLIER_FRAC` is required;
  - the global sign is fixed by the hotspot fit or the boundary estimate;
  - tiles that disagree mean content, not drift: return `None`.
- Control tiles are found by walking along the blend gradient, from the tile centre to
  `CTRL_MARGIN_MM` beyond the opposite inner edge. If one side is missing, one control is
  used. If both are missing (one box inside the other), the echo is off.

**2. Brightness hotspots** (`seam.py`)
- Per block of the overlap (`TILE_MM`): the ratio of its mean to the mean of flat flank
  blocks just outside the overlap, pooled over flat frames. Blocks must pass a flatness
  gate (`FLAT_MAX_DEV`). The ratio is compared with its baseline value.
- Model: Δr(x) = −∇b(x)·u(x) + (g − 1)·b(x) + trend(x). Shifting B moves its ramp, so the
  overlap dims or brightens; a dimmer lamp (gain g) tilts it instead.
- Fit u with `field.py` from these observations, plus g and the trend. Each observation is
  1-D: it sees only motion across the overlap edge.
- Sensitivity (planning estimate): 0.40% brightness per projector pixel of shift, for the
  demo's cosine ramp over 394 px. The pseudocode's example: 2 px in a 384 px linear blend
  is about 0.5%. Block noise is ≈ 0.03% per flat frame.
- |g − 1| > `LAMP_WARN` → lamp warning, not misalignment.
- The sign of the brightness change also fixes the echo's sign.
- In dark frames the same blocks see each projector's black-level outline. Those
  inner-edge positions go to the boundary layer (4.2), not here.

**3. Offset borders** (`borders.py`)
- Where the combined image's content border runs through the overlap: the step between
  A's and B's border position, against the calibrated step.
- `None` in arrangements or content where no border crosses the overlap.

Each symptom returns `offset_mm`, `None`, or `unresolved` with a floor.
`artifacts.offset_mm` is the max over symptoms (the pseudocode's "either can raise the
alarm"). `agreeing` is the number of symptoms independently above `TOLERANCE_MM`.

### 4.4 Reference mode (`source.py`, `kernel.py`)
- Enabled when the source feed is available (`REFERENCE_MODE`: auto, on or off).
- Keep the last `SOURCE_RING` source frames, with timestamps. The camera lags the video
  output by a few frames, so:
  - map each candidate to the camera grid through the blending setup (content → mm →
    camera, in linear light);
  - pick the one with the best NCC on downsampled images;
  - reject matches below `MATCH_MIN_NCC` (e.g. a cut straddling the exposure).
- Per core tile and control tile, pool the cross spectrum Σ O·conj(S) and the power
  Σ|S|². Wiener kernel: k = IFFT(Σ O·conj(S) / (Σ|S|² + `WIENER_LAMBDA` · mean power)).
- `ctrl_a`'s kernel shows where projector A alone puts the content; `ctrl_b`'s shows where
  B does. Offset = centroid(k_b) − centroid(k_a), so camera motion cancels.
- The core tile's kernel must be consistent with a·u_A + b·u_B. It splits into two blobs
  only for large offsets.
- There is no resolution floor (planning estimate: within ≈ 0.1 mm for 0.5–4 mm offsets at
  whole-screen sampling). In reference mode, double contours become a sub-pixel symptom.
  Repetitive content fools it only at whole-period shifts, where the kernel is periodic too
  and only the boundary can tell (`docs/findings.md`, 2026-10-07; `repetition_limit`).

### 4.5 Baseline (`baseline.py`)
Captured after every calibration and refined during `TRUSTED_WINDOW_S`:
- camera homography and marker centres;
- `box_a`, `box_b`, with corner covariance;
- `OverlapGeometry` (tiles, controls, blocks, pieces) and the blend maps;
- the residual relative field (≈ identity) and per-tile residual offsets;
- per-piece edge positions (lit and dark), with counts;
- per-block hotspot ratios, with counts;
- border steps;
- lamp gain ratio and white level;
- noise model (variance vs mean);
- reference-mode kernel offsets;
- timestamp and config hash.

### 4.6 Fusion and decision (`fusion.py`, `decision.py`)
- Primary estimate = boundary `offset_mm`. If the overlap check ran:
  - layers agree within `AGREE_MM` ⇒ take the max and raise confidence;
  - the overlap check says more than the boundary, with ≥ 2 symptoms agreeing ⇒ take the
    overlap check;
  - the boundary says more than the overlap check ⇒ suspect the edge fit or the camera,
    hold and re-check next interval, for at most `HOLD_MAX_INTERVALS`.
- The overlap check contradicts the boundary only when it was informative: it resolved
  tiles, and the boundary offset is above the echo floor + `AGREE_MM`. Otherwise the
  boundary decides alone.
- Decision:
  - `offset_mm > TOLERANCE_MM` votes YES; YES after `YES_VOTES = (3, 4)`;
  - back to NO only below `CLEAR_RATIO × TOLERANCE_MM`;
  - `None` ⇒ hold the previous answer;
  - only a recalibration resets the baseline and the answer.
- Output per interval, logged as JSONL: `answer`, `offset_mm`, `offset_px`, `confidence`,
  `boundary_available`, `symptoms`, `mode` (blind or reference), and `warnings` (lamp,
  camera moved, focus).
- The log matters as much as the answer: comparing logged offsets with the professor's own
  calls is how the thresholds get set.

---

## 5. The simulator (`sim/`)

Purpose: generate camera frames with known truth. Fidelity matters more than speed, but
keep a `fast` quality preset for tests (`standard` is the default, `fine` checks that a result
does not depend on sampling). Everything below is built (Phases 1 and 2) except screen gain.

- **Screen**: width/height mm, reflectance, room light (ambient), and the wall around it (a
  reflectance lit by the room light).
  - A bezel around the screen (`screen.bezel`: `width_mm`, reflectance, optional `light` of
    its own) carries 8 ArUco DICT_4X4_50 markers. Each is 80 mm with a one-cell white quiet
    zone, which detection needs. They sit at the 4 bezel corners plus two above and two below
    the overlap (beside it, for a horizontal overlap), so a camera zoomed on a vertical
    overlap still sees ≥ 4 (the zoomed preset is refused for `stacked` and `large_overlap`).
  - A static reflectance map renders screen, bezel, wall, paper and ink by exact area coverage
    (per material, so marker edges sit at their true sub-pixel positions).
  - **Default ambient: 0.02** of projector white, a dim lecture hall. Measured in Phase 2:
    8/8 markers found in single frames, centre error p95 0.09 px; at the dark-room 0.0003 none
    are found in a single frame. The `dark_room` scenario covers the dark case.
  - Optional screen gain and vignetting: a high-gain screen is brighter near its hotspot (not
    built yet: it needs a gain map per projector; `docs/findings.md`, 2026-10-07).
- **Projector** (×2):
  - resolution, homography px→mm (from an arrangement preset or explicit corners), gamma,
    brightness, black level (non-zero!), colour balance;
  - light is conserved under zoom and keystone (|det J| ratio);
  - flicker and rolling-shutter banding are nuisances (below).
- **Blending setup**: the calibration software's state (H_cal per projector, content rect,
  blend maps, optional black-level uplift). Blend rule: each projector fades to 0 at its inner
  edges inside the content (distance to those edges, cosine or linear ramp).
  `docs/findings.md` (2026-10-06) explains why not "distance to all own edges".
- **Arrangements** (`arrangements.py`): side by side, stacked, rotated, corner, different
  sizes, large overlap, or explicit corners; each pair is centred on the screen.
- **Content** (`pictures.py`, `sequence.py`): a library of procedural sources (no external
  assets), as sequences with timestamps:
  - slide decks (text density low, medium or high, plus graphics);
  - photo-like textures;
  - video clips with motion, pans and cuts. An exposure spanning two video frames blends
    them in linear light;
  - dark film scenes;
  - flat gray, flat colours, black;
  - letterboxed (black bars);
  - stripes of known period, sinusoidal in light;
  - blank-overlap content (flat inside the overlap, textured elsewhere);
  - one slide held for 20 minutes.
- **Rendering**, per frame:
  1. the calibration software builds each framebuffer from the content at H_cal;
  2. each projector turns it into light (gamma, blend weight, black level);
  3. the light lands through the *actual* homography and adds in linear light on the
     screen mm grid;
  4. then the camera.
- **Camera**:
  - resolution and homography mm→px. Presets: whole screen + 5% margin; zoomed on the overlap
    + control strips + the markers at its ends, 1.74 px/mm by default, sensor in portrait for a
    vertical overlap;
  - Gaussian PSF, part of it applied on the screen grid before resampling to prevent
    aliasing;
  - pixel integration by supersampling, shot + read noise, locked exposure, pedestal,
    vignetting, optional gamma;
  - optional in-camera sharpening (a nuisance), to reproduce that failure mode: its halos can
    mimic double contours;
  - output 16-bit linear luminance. Rec. 709 weights are applied right after the projector
    light; RGB stays an option. This roughly halves render time and memory.
- **Perturbation** (`perturb.py`, `schedule.py`):
  - applied to A, B or both as a change of the actual homography, `h_actual = M(t) · h_cal`;
  - kinds: shift (`across` or `along` the moving projector's inner edges, or a vector),
    rotation and scale (about its centre, its far corner, the overlap centre or a point),
    keystone; each sized by the offset it causes (`magnitude_px` or `magnitude_mm`), or by
    `deg` / `factor`;
  - schedules: `none`, `step(t0)`, `staircase(levels, hold_s)`, `drift(rate per hour)`,
    `ramp`, `bump_then_hold`, `oscillate(period)` (thermal). Continuous ones are quantized to
    0.02 px so render caches stay effective. A schedule is required.
- **Nuisances** (`nuisance.py`). These must never cause YES:
  - camera bump: changes the camera homography; the markers move with it;
  - lamp dimming of one projector (e.g. 15%, black level included);
  - room-light step;
  - occluder: a person-shaped silhouette crossing the screen for N frames, which may cover
    markers;
  - flicker: per-projector brightness modulation, with rolling-shutter banding when exposure
    is not a multiple of the modulation period (its phase changes from frame to frame);
  - in-camera sharpening left on by mistake;
  - black-level uplift outside the overlap, if the blending software compensates (part of
    the blending setup, `blend.black_uplift`).
- **Ground truth per frame** (`truth.py`), written to `metadata.jsonl`:
  - true relative homography `h_rel = H_actB·H_calB⁻¹·H_calA·H_actA⁻¹`;
  - true `offset_mm`: how far apart A and B now put the same content, the largest
    |D_B(x) − D_A(x)| over the calibrated overlap ∩ content rect, with D_p = H_act,p·H_cal,p⁻¹
    (symmetric in A and B; exact at the vertices for shift, rotation and scale, densely sampled
    and refined for keystone);
  - `offset_px`, using the coarser projector pitch;
  - `aligned` (`offset_mm` < 0.02 mm);
  - perturbation and nuisance tags, content tag, timestamp, camera homography.
- **Frame delivery** (`frames.py`, `dataset.py`):
  - A `FrameSource` renders frame i on demand, seeded from the scenario seed
    (`SeedSequence(seed, spawn_key=(1, i))`), so any frame reproduces in any process.
  - The optical chain is linear in light, so a frame is a weighted sum of cached camera
    images, one per light source (room light, bezel light, each projector's light for the
    pictures shown and its current geometry). Each is rendered once and reused while its
    inputs stay the same, so an unchanged frame costs only its noise (about 25 ms at demo
    scale in one process, against about 0.13 s for a new slide and 0.16 s for a video frame).
  - `make_dataset` always writes `scenario.yaml`, `setup.json` (exactly what the detector
    may read) and `metadata.jsonl`. Frames are written as 16-bit PNG only on request
    (`--frames all|sample|none`; ≈ 8 MB per mono frame).
  - Reference mode: the projectors show what was sent `reference.lag_s` earlier, and
    `FrameSource.source(i)` lists the pictures sent before frame i's exposure ends.
  - `scripts/check_dataset.py` checks a dataset's truth against what its scenario asked for,
    with its own geometry code.

Scenarios are YAML files in `scenarios/`. Adding a test idea must mean adding a YAML file,
not code. Keys: `name`, `description`, `seed`, `quality`, `screen` (with `bezel` and its
`markers`), `arrangement`, `projectors`, `blend`, `content` (a sequence), `camera` (with
`color`, `exposure_s`, `phase_s`), `perturbation`, `nuisances`, `reference`, `duration_s`,
`sample_every_s`, `trusted_window_s` (no perturbation or scheduled nuisance may start before
it), `extends` (deep-merge
a base file such as `_lecture_hall.yaml`) and `sweep` (dotted keys to lists of values; one
dataset variant per combination). Sweep variants share the seed, so they are paired: same
content and noise, and bit-identical frames before a perturbation's onset.

---

## 6. Evaluation harness (`eval/`)

For every dataset, run the detector with the interval `ADJUSTABLE_INTERVAL_IN_SECONDS` and
produce:

- **Detection rate vs true offset** (a psychometric-style curve) per misalignment type, per
  arrangement and per content type: overall and **per signal** (boundary, echo, hotspots,
  borders, kernel).
- **False positive rate** on aligned runs, with and without nuisances.
- **Latency**: intervals from onset to the first YES.
- **Boundary availability rate**: the fraction of intervals where the primary tool could
  answer, per content type. This is the key number for the "boundary first" decision. Also
  the **"can't tell" rate** of the whole detector.
- **Offset accuracy**: estimated vs true offset (mean abs error in mm and px), per signal.
- **Blind vs reference**: the same datasets in both modes.
- **Symptom confusion**: which symptoms fired for which true type.
- **Threshold sweep**: all of the above across `TOLERANCE_MM` values.
- **Diagnostic images** per scenario: box overlays on the rectified frame, overlap / core /
  control masks, edge profiles vs baseline, hotspot maps, cepstra and kernels. These are
  for humans; keep them easy to open.

Reports: `results.jsonl`, `summary.csv`, PNG plots, `report.md` per run.

---

## 7. Scenario catalogue

These YAML scenarios exist in `scenarios/` (built in Phase 2), each to answer a specific
question. They extend `_lecture_hall.yaml`: ambient 0.02, whole-screen camera, mono,
`sample_every_s: 0.5`, nothing perturbed before the 600 s trusted window.

| Scenario | Question it answers |
| --- | --- |
| `aligned_slides` | FPR on static text-heavy decks (three text densities, 20 min). |
| `held_slide` | One slide for 20 min: does the detector cope with zero content variety (echo pools starve; boundary and hotspots carry on)? |
| `aligned_video` | FPR under motion, fast motion, cuts, dark scenes. |
| `aligned_nuisances` | Camera bump, 15% lamp dimming, room-light step, occluder, flicker, in-camera sharpening ⇒ still NO? |
| `shift_sweep` | Steps of 0.25, 0.5, 1, 2, 4, 8 px shift across and along the overlap edge; detection curve. |
| `rotation_sweep` | Small rotations of B about its centre and about a far corner. |
| `scale_keystone` | Zoom and tilt perturbations; does the 6→8 param escalation work? |
| `slow_drift` | 2 px over 2 hours; latency and timed safety check. |
| `arrangements` | The shift step (0, 0.5, 1, 2, 4 px across) in side-by-side, stacked, rotated, corner, different-size and large-overlap arrangements. |
| `boundary_hidden` | Letterboxed photos and video + bright overlap: how often is the boundary unavailable, and does the overlap check catch what the boundary can't? (The bars cross the overlap, so the boundary probably stays visible: `docs/findings.md`, 2026-10-07, 2b item 8.) |
| `blank_band` | Nothing textured inside the overlap: the hotspot fit and the boundary must carry the check. |
| `dark_film` | Dark scenes and fades: raster edges via black level. |
| `repetition_limit` | Stripes of period P with shifts near P/2 and P: measure the aliasing limit (blind echo vs reference kernel). |
| `camera_zoomed` | Camera on the overlap + control strips only vs whole screen: smallest detectable offset per signal (paired with `shift_sweep`'s across steps). |
| `dark_room` | Ambient 0.0003 with and without bezel light: can the markers be found, and what does averaging buy? |

---

## 8. Build order

Each phase ends with tests passing, a `docs/findings.md` entry and a commit. Do not start a
phase before the done conditions of the phases it depends on hold.

| Phase | Scope | Done condition | Depends on | Status |
|---|---|---|---|---|
| 1 Scaffold + simulator core | pyproject, packages, config dataclass; screen, projector, blending setup, content, camera, render chain; `scripts/visualize.py` | A human sees a seamless image with a faint black-level raster around it | — | **done 2026-10-06**: seam 3.2e-6, raster edge within 0.0001 mm, 0.43 s per frame, 68 tests |
| 2a Simulator datasets | bezel + markers, mono camera, arrangement presets, slide decks and held/flat/black sequences, perturbations + schedules, truth, FrameSource + caches, `make_dataset`, `check_dataset` | `shift_sweep` and `aligned_slides` generate twice with identical metadata and frame hashes; every frame's `offset_mm` within 1e-6 mm of the injected value; markers found 8/8 at ambient 0.02 with centre error < 0.2 px | 1 | **done 2026-10-07**: `--jobs 1` and `--jobs 4` runs byte-identical (34,800 frames each); offset error ≤ 8e-14 mm; marker centres p95 0.092 px, max 0.125 px; 45 ms per unchanged frame; 203 tests (7 slow) |
| 2b Simulator library | video, textures (photo, dark film, stripes, letterbox, blank overlap), nuisances + flicker, black-level uplift, zoomed camera preset, reference feed, the rest of the §7 catalogue | every §7 scenario loads and renders its event frames; `aligned_video`, `aligned_nuisances`, `camera_zoomed` and `boundary_hidden` generate twice identically; the zoomed camera finds the 4 overlap markers with centre error < 0.2 px; each nuisance has a physics test, with truth still aligned; a video straddle equals the linear-light mix; source timestamps lag the display by `lag_s` | 2a | **done 2026-10-07**, refined 2026-10-08: `--jobs 4` and `--jobs 5` runs byte-identical (52,800 frames each); zoomed marker centres max 0.089 px; lamp, room light and bump exact; flicker moves frame to frame; 280 tests (20 slow) |
| 3 Detector inputs + geometry | `inputs.py`, `rectify.py`, `polygon.py`, `blending.py`, `geometry.py`, `classify.py`, `results.py` | 200 random convex quad pairs (rotated, corner, nested): overlap, core tiles, controls and pieces valid; blend weights equal the simulator's within 1e-6; rectification error ≤ 0.05 mm; camera bump re-solved within 0.1 mm | 2a | |
| 4 Boundary (primary) | `edges.py`, `field.py`, `boundary.py`; minimal `eval/feed.py`, `eval/metrics.py`, `scripts/run_detector.py` | On `shift_sweep` and `rotation_sweep`: offset within 0.2 mm of truth whenever available (spec: 0.5 px); `None`, not a wrong number, when edges are hidden; availability ≥ 95% on `aligned_slides` and `aligned_video` | 2a, 3 (`aligned_video`: 2b) | |
| 5 Runner + decision | `baseline.py`, `pools.py`, `decision.py`, `runner.py`, JSONL output | `aligned_*` including nuisances: zero YES over ≥ 100 intervals; 4 px → YES within `YES_VOTES` intervals; 2 px → YES; 1 px → NO | 4 | |
| 6a Echo | `echo.py`, developed on synthetic overlap data `O = a·S + b·warp(S)` | At 1.74 camera px/mm: 1.5–4 mm within 0.3 mm. At 0.87: ≥ 3 mm within 0.3 mm, and 1.5–2.5 mm flagged `unresolved`. Aligned twins: no detection over 50 seeds | 3 | |
| 6b Hotspots + borders | `seam.py`, `borders.py`, synthetic first | Flat frames: 0.5 px across-shift within 0.2 px; along-shift → no offset; lamp −5% → warning with offset < 0.3 mm; aligned text crossing blocks → no offset | 3 | |
| 6c Overlap check + fusion | `artifacts.py`, `fusion.py`, runner hook, safety check | `boundary_hidden` and `camera_zoomed`: 2 px caught by the overlap check where the boundary returns `None` (if Phase 4 finds the boundary visible there, as predicted, first add content whose overlap is never dark); nuisances still produce no YES | 5, 6a, 6b | |
| 7 Reference mode | `source.py`, `kernel.py` | Whole-screen `shift_sweep`: kernel offset within 0.15 mm at 0.25–4 px; aligned < 0.1 mm; camera lag of 3 frames with cuts: wrong matches rejected | 6c | |
| 8 Evaluation harness | `eval/sweep.py`, `eval/report.py`, `scripts/evaluate.py`, diagnostic images | One command evaluates every scenario and writes the detection curves, FPR, latency, availability, confusion, blind vs reference, and `report.md` | 6c | |
| 9 Learning + stretch | threshold learning from simulated recalibrate presses (truth offsets above a hidden human threshold); grids of more than two projectors | Learned threshold within 10% of the hidden one (provisional) | 8 | |

Phase 3 needs only 2a, so 2b and 3 can run in parallel, and so can 6a/6b alongside 4/5
(they need only the geometry).

---

## 9. Engineering conventions

- **Tooling:** Python ≥ 3.11 (3.13 in `.venv`), managed with `uv`: `uv sync`,
  `uv run pytest`, `uv run python -m scripts.<name>`.
- **Dependencies:**
  - `numpy`, `opencv-python-headless` ≥ 5.0 (continuous sub-pixel warps for float32),
    `pyyaml`, `pytest`;
  - `matplotlib` for plots only, added when first used;
  - no GUI dependencies; everything runs headless.
- **Code shape:**
  - type hints everywhere;
  - `@dataclass` for configs and results;
  - no global mutable state in `detector/` (the runner owns state);
  - keep modules under ~300 lines; split rather than grow.
- **Determinism:** every random draw takes a `numpy.random.Generator`, seeded from the
  scenario through `SeedSequence` spawn keys.
- **Light:** linear inside; gamma only at the content input (and at the camera output, if
  enabled). Camera frames are 16-bit linear luminance.
- **Docstrings:** every module has a docstring that explains the physics it models or
  exploits, in plain words, before the code. A reader with no computer-vision background
  should understand *why* each step exists.
- **Warps:** every warp goes through one wrapper per package that accepts only float32 with
  1, 3 or 4 channels. Other types round sub-pixel positions to 1/32 px.
- **Tests:**
  - unit tests per function, property tests for geometry;
  - regression tests on fixed seeds, with tolerances stated in mm (or relative brightness,
    with the physical reason);
  - a test that asserts a symptom fires must also assert it does *not* fire on the
    matching aligned twin;
  - a measurement that cannot see must be tested to return `None`, not 0.
- **Logging:** structured JSONL per interval. Never print tensors.
- **Commits:** one per phase (Phase 2: one per milestone, 2a and 2b), with a message that
  states which "done" condition was met and the numbers that show it.

---

## 10. Config (names are the contract; defaults are starting guesses)

```yaml
# cadence
ADJUSTABLE_INTERVAL_IN_SECONDS: 30      # boundary check cadence
SAFETY_CHECK_S: 900                     # timed overlap check regardless of boundary result
TRUSTED_WINDOW_S: 600                   # after calibration: refine boxes, fill baseline pools
SAMPLE_EVERY_S: 0.5                     # frame sampling period for the pools
POOL_FRAMES: 60                         # usable frames pooled per overlap check
# thresholds in mm
TOLERANCE_MM: 1.7                       # ≈ 1 arcmin at 6 m (≈ 1.6 px at 1.04 mm/px; pseudocode: 2 px); swept
TRIGGER_MM: 0.85                        # boundary offset that triggers the overlap check
AGREE_MM: 1.4                           # layers "agree" within this
MIN_OFFSET_MM: 1.5                      # smallest echo the cepstrum can separate from its origin peak
INLIER_MM: 0.7                          # RANSAC inlier distance for the smooth motion field
# decision
YES_VOTES: [3, 4]                       # K of N intervals
CLEAR_RATIO: 0.5                        # back to NO only below CLEAR_RATIO × TOLERANCE_MM
HOLD_MAX_INTERVALS: 2                   # intervals a boundary-vs-overlap disagreement may hold
# geometry on the mm canvas
CORE_MIN_WEIGHT: 0.2                    # ≈ the pseudocode's CORE_FRACTION 0.6 for a linear ramp
TILE_MM: 64
CTRL_MARGIN_MM: 40
EDGE_PIECE_MM: 80
CANVAS_PX_PER_MM: 2.0
# frame routing
MOTION_LEVEL: 0.02                      # mean |frame difference| (× white) above this: skip
MAX_CLIPPED: 0.002                      # saturated fraction above this: skip
DARK_LEVEL: 0.003                       # region mean (× white) below this: dark
MIN_TILE_EDGES: 0.02                    # edge density a tile needs to count as textured
VARIETY_MAX_NCC: 0.98                   # blind mode keeps a frame only if it differs from the last kept
FIDUCIAL_MOVE_PX: 0.3                   # marker motion above this: camera moved, re-solve, never YES
# evidence quality
MIN_PEAK_SNR: 6.0                       # robust z-score a cepstral or kernel peak needs
NULL_FACTOR: 1.5                        # ... and this × the control-vs-control null
ECHO_FLOOR_CAMERA_PX: 2.5               # the cepstrum cannot resolve echoes below this many camera px
MIN_GOOD_TILES: 3                       # fewer valid tiles -> "can't tell"
MIN_TILE_FRAMES: 20                     # textured frames an echo tile needs
MIN_SEAM_FRAMES: 20                     # flat (or dark) frames a hotspot block needs
MIN_EDGE_SNR: 4.0                       # pooled SNR an edge piece needs to count
LINE_INLIER_FRAC: 0.6                   # tiles must agree on one smooth field this often
FLAT_MAX_DEV: 0.01                      # a block is flat if its sub-blocks agree within this
LAMP_WARN: 0.03                         # fitted lamp gain off by more than this: lamp warning
# reference mode
REFERENCE_MODE: auto                    # auto | on | off
WIENER_LAMBDA: 0.01                     # Wiener regularization, × the mean source power
SOURCE_RING: 10                         # source frames kept for lag matching
MATCH_MIN_NCC: 0.9                      # a source frame must match the camera frame this well
```

`detector.yaml` holds these defaults with comments; `tests/test_config.py` keeps the
dataclass, the YAML file and this list identical.

---

## 11. Out of scope for now, and open hardware questions

Out of scope:
- real camera drivers, projector control, reading the recalibrate button, GUIs;
- grids of more than two projectors (keep the geometry general so it can come later);
- stereo/3D content, curved screens;
- SSIM;
- colour fringing / convergence checks (RGB stays a simulator option only).

Open questions for the real installation (from the pseudocode; the simulator sweeps them
meanwhile):
- Where exactly is the overlap, and what shape are the blend ramps: linear, cosine or
  custom, applied in linear light or in gamma? The blending software should report both.
- Can the script read the video sent to the projectors? Yes unlocks reference mode.
- Camera resolution, lens and mount: whole screen at ≥ 0.87 camera px per mm, or zoomed on
  the overlap?
- How large an offset does the professor notice from the seats? That sets `TOLERANCE_MM`.
- Is black-level compensation enabled outside the overlap?
- Does the recalibrate button emit a signal the script can read?
- Can printed markers go on the screen bezel, and is there enough room light to see them?
