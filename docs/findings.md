# Findings

Dated entries with numbers, newest at the bottom. A result that contradicts a design choice
in CLAUDE.md is written here and raised, never silently designed around.

## 2026-10-06 — Phase 1: the simulator core renders a seamless aligned side-by-side scene

**Done condition met.** `uv run python -m scripts.visualize scenarios/aligned_side_by_side.yaml --out out/phase1`
writes `out/phase1/view.png`. The scene:
- two 1920×1080 projectors side by side, each 2000 mm wide (1.04 mm/px);
- a 410 mm (393.6 px) overlap, with B 0.3 mm lower, so the two pixel grids are not aligned;
- a text slide with a 5% black border;
- a 3840×1600 camera seeing the whole screen with a 2% keystone.

What the image shows:
- The slide's text, diagonal, gray band and circle continue across the overlap with no seam.
- Each projector's black-level raster glows faintly around the picture.
- The overlap's doubled black shows as a brighter patch in the top and bottom border bands.

All 68 tests pass in 0.8 s.

The display uses a log curve, y = ln(1+x/b)/ln(1+1/b), with x = radiance relative to one
projector's white and b = black level (1/1500). Plain sRGB would put the black level at about
3/255, which is invisible. Levels measured in the frame:

| patch | electrons | display (/255) |
|---|---|---|
| unlit screen | 3.7 | 12 |
| A black / B black | 12.0 / 12.0 | 30 / 30 |
| overlap black | 21.8 | 43 |

### Numbers

**Seam.** On noiseless screen irradiance (flat white, black level 0, standard quality), the
largest deviation anywhere in the picture is 3.2e-6. The bound set by the cosine ramp's
curvature is 2·(π²/2)/(8W²) = 8.0e-6 for W = 393.6 px.

**Black level is not blended away.** On 50% gray, the overlap minus single coverage is
6.0005e-4 radiance. That is exactly one black level times reflectance (6.0003e-4): the overlap
is 0.31% brighter. How much brighter depends on the fill:

| fill code | overlap brighter by |
|---|---|
| 0.92 | 0.08% |
| 0.5 | 0.31% |
| 0.3 | 0.93% |
| 0.1 | 9.6% |

So an aligned blend is seamless only on content well above black. The slide's darkest fill
(the title bar, about 0.8% in luminance) is at the edge of visibility. This is the same effect
§4.2 relies on to see inner edges in dark frames.

**Raster edge position.** The 50% point of the black-level step lies within 0.0001 mm of the
true box edge (standard quality). The tests hold it to 0.02 mm (standard) and 0.05 mm (fast).

**Raster SNR in one frame.**
- The step from unlit screen to one projector's black is 8.3 e⁻, against 4.6 e⁻ of per-pixel
  noise: per-pixel SNR 1.8 (1.4 for the difference of two pixels).
- The step from single to overlap black is 9.8 e⁻ against 5.5 e⁻: SNR 1.8.
- This confirms §4.2: edges seen through black level need averaging over frames and along
  the edge.

**Runtime.**
- Blend maps take 0.4 s, computed once per calibration.
- One frame takes 0.43 s at standard quality (2 screen samples per projector pixel, camera
  supersampling k = 2) and 0.26 s at fast.
- The visualize run peaks at 2.0 GB of memory.

**OpenCV 5.0.0 interpolation.**
- Float32 bilinear warps with 1, 3 or 4 channels are continuous: a 0.01 px shift comes back
  as 0.009999 px.
- Float64 and 2-channel warps still round to 1/32 px: 0.37 px comes back as 0.375.
- Every warp goes through `sim.planar.warp_linear`, which refuses those types. The canary
  test is `test_warp_is_continuous_subpixel`.

### Departures from the literal brief (raised, not silently changed)

1. **Blend map.** "Distance to its own edges" is applied only to the edges that must fade:
   - those are each projector's edges that lie inside the other projector *and* inside the
     content;
   - found by tagging the edges of the overlap ∩ content by the box they lie on.

   The literal all-edges rule pinches the ramp near the overlap's shared top and bottom edges.
   At demo geometry it would change weights by up to 0.68, and by more than 0.01 over 56% of
   the overlap's height (45% of its area). Using inner edges gives the classic 1-D ramp for
   side by side, and still works for rotated and corner arrangements.
2. **Layout.** The root of the tree in CLAUDE.md §3, `projector_align/`, is taken to be the
   repo root. Only then do the brief's commands (`python -m scripts.make_dataset scenarios/…`)
   and `docs/` paths resolve.
3. **Extra modules:**
   - `sim/planar.py`: simulator-only geometry, independent of the detector so truth cannot
     share its bugs;
   - `sim/calibration.py`: the blending setup, i.e. what the detector may read;
   - `sim/render.py`: the render chain.

### Other observations

**Camera aliasing (found and fixed).**
- The camera was sampling the finer screen grid before applying the lens blur. Edge
  positions snapped by up to 0.27 px (k = 1) and 0.06 px (k = 2).
- The fix applies the anti-aliasing share of the PSF (up to 0.8 sub-pixel) on the screen grid
  before resampling; the total PSF is unchanged. Edge error dropped to ≤ 0.003 px (k = 1) and
  ≤ 0.010 px (k = 2, with keystone and vignetting).
- Dropping the (k−1)/2 sub-pixel offset would cost 0.27 px.
- Both problems are guarded by `test_edges_land_where_they_should_at_every_supersampling`.

**Linear vs cosine ramp.** A linear ramp still has a slope where it meets an inner edge, and
pixel blur spills that slope across the raster edge. The result is a bump of up to 1/(4W): 1.4
to 3.1e-3 at W = 58 px, and ≤ 6.4e-4 at demo scale. That is why cosine is the default.

**Rotated (5°) and corner arrangements** (58 px overlaps, tests only):
- Away from notches the seam is ≤ 5.4e-3. That error comes from kinks in the distance field
  and scales as 1/W.
- At a notch (an overlap corner on the image outline) any blend must jump from 0 to 1. That
  leaves a speck of up to 2–3% within 5 px of the notch.
- A real content rect drawn inside the union keeps most notches off the picture.

**Aligned slide.**
- In the overlap, blending adds no error beyond each projector's own resampling: the blended
  error is 0.94× (fast) and 0.91× (standard) that of either projector shown alone.
- Shifting B by 1 px multiplies the overlap error by 3.6–4.2. That is the ghost the detector
  has to find.

### Deferred

- **Fiducial markers (ArUco):** added at the start of Phase 2, before any dataset is written,
  because Phase 3 needs them in the frames.
- **Presets and optional effects:**
  - the fine quality preset;
  - other arrangement presets and the zoomed camera;
  - chromatic aberration and in-camera sharpening;
  - screen gain;
  - DLP flicker;
  - black-level uplift;
  - room-bounce light, which raises the unlit level and adds shot noise but leaves the raster
    step itself unchanged.

## 2026-10-07 — Replan after the original pseudocode

The original pseudocode (Oct 1, 2026) is now in `docs/research/pseudocode.md`, and CLAUDE.md
has been rewritten around it.

### What the pseudocode changed, and what it did not

**Kept from CLAUDE.md** (confirmed with you):
- boundary analysis of each projector's rectangle is primary, with the rectangles inferred
  from the non-overlap sections;
- the overlap check is secondary and triggered;
- any arrangement, with the rectangles at angles;
- a camera that sees the whole picture;
- mm units.

**Taken from the pseudocode:**
- the concrete overlap-check methods: the cepstral echo with control tiles beside the
  overlap, the seam (hotspot) fit that separates shift from lamp gain and trend, and dark
  raster edges;
- frame routing (skip / dark / flat / textured) and pooling;
- reference mode with Wiener kernels, now a first-class phase right after blind mode;
- the decision rule (3 of 4 votes, clear below 0.5 × tolerance, `None` holds);
- the edge-case table and the stress tests (held slide, blank band, dark film, 15% lamp dim).

**Dropped:**
- matched-frame SSIM: it changes with content, and reference mode does its job;
- colour fringing;
- `HASH_MATCH_BITS`.

The camera becomes mono (16-bit linear luminance).

### The "diagonal overlap" in the Phase 1 image was content, not geometry

In `out/phase1/view.png` the overlap is the full-height vertical band in the middle (410 mm).
When aligned, the blend makes it invisible across the lit slide, and it showed only as
doubled black in the border. The diagonal line was a feature drawn in the slide: two
rectangles cannot overlap along a diagonal while their union stays a rectangle.

Two changes so this cannot be misread again:
- `scripts/visualize.py` now adds a second panel with the true geometry drawn on the frame:
  box A, box B, the overlap polygon and the content rect;
- the slide's bare diagonal and its centre circle are replaced by a framed line chart that
  still crosses the overlap.

### Planning estimates (to confirm in the phase that builds each piece)

These come from read-only experiments run during replanning, on the existing simulator and
on synthetic overlap data. They are not yet verified by tests:

| Quantity | Estimate | Consequence |
|---|---|---|
| Blind echo floor, whole-screen camera (0.87 camera px/mm) | resolves only ≥ 2.5–3 mm; 1.5–2 mm read as ~2.3–2.6 mm | The blind overlap check can *confirm* but not *detect* threshold-level offsets (1.7 mm) with this camera. The boundary layer must carry detection; zoom (floor ≈ 1.4 mm) or reference mode closes the gap. |
| Echo on the rectified canvas | peaks lock to multiples of the camera pitch | Echo tiles are cut on the camera grid. |
| Robust z of aligned twins | 7.5–12.6, above `MIN_PEAK_SNR` 6 | A fixed threshold alone would false-alarm; hence `NULL_FACTOR` × control-vs-control null. |
| Black-level edge, 80 mm piece | 0.11 mm per frame, 0.02 mm over 30 frames; content border 0.004 mm | The boundary layer has ample precision at whole-screen resolution. |
| Hotspot sensitivity, cosine ramp over 394 px | 0.40% per projector px; block noise 0.03% per flat frame | Detectable on flat content. |
| Reference kernel offset | 0.5 → 0.47, 1 → 1.01, 2 → 1.96, 4 → 3.89 mm | No resolution floor. |
| Marker detection, passive ArUco 80 mm | ambient 0.0003: 0 of 6 found in single frames; 0.02: 6/6, homography error 0.12 mm; corners biased ~0.9 px inward | Default ambient 0.02; fit from marker centres; a `dark_room` scenario. |
| Raster per-pixel SNR at ambient 0.02 | 0.38 (vs 1.5 at 0.0003) | Still recoverable by averaging along edges and over frames. |
| Mono vs RGB render | 0.22 s and 1.05 GB vs 0.38 s and 2.28 GB; 16-bit PNG 7.8 MB vs 24 MB | Mono is the default; frames are written to disk only on request. |

### Config

- Removed `HASH_MATCH_BITS`.
- Added the knobs the methods use: `SAMPLE_EVERY_S`, `MIN_OFFSET_MM`, `INLIER_MM`,
  `HOLD_MAX_INTERVALS`, routing levels, evidence-quality gates, and the reference-mode
  parameters (CLAUDE.md §10).
- `detector.yaml`, the dataclass and the tests are kept identical.

## 2026-10-07 — Phase 2a: datasets with ground truth

**Done condition met.** Commands:
- `make_dataset scenarios/aligned_slides.yaml … --frames sample`, run twice;
- `make_dataset scenarios/shift_sweep.yaml … --frames sample`, once with `--jobs 1` and once
  with `--jobs 4`;
- `check_dataset` on the first run.

Results:
- **Reproducible:** run 1 (`--jobs 1`) and run 2 (`--jobs 4`) are byte-identical in every file
  except `timing.json` and `dataset.json`. That covers `metadata.jsonl`, `setup.json`,
  `scenario.yaml`, the PNGs and `variants.json`, so all 34,800 frame hashes per run match
  (3600 + 13 × 2400).
  - Run 2 already used the code with the review fixes listed below.
  - A third run with the committed code reproduces run 1 again (`aligned_slides`, and
    `shift_sweep` 8 px across).
- **Truth equals what was asked:** `check_dataset` recomputes the overlap, the pitch
  (1.041667 mm) and each requested offset from `setup.json` and `scenario.yaml` with its own
  geometry code. On every frame:
  - `offset_mm` is within 8.0e-14 mm of the requested size × its schedule;
  - `offset_px` agrees with it to the last bit;
  - every shift points the requested way;
  - the moved projector's `h_actual` is its shift of `h_cal`.
- **Paired variants:** the 15,990 frames before the onset (1230 per variant) are bit-identical
  across all 13 variants.
- **Markers:** 8/8 found in every sampled noisy frame at ambient 0.02. Centre error is
  p50 0.038, p95 0.092, max 0.125 camera px (40 marker sightings in 5 frames).

Phase 2 is split into two milestones (decided before starting). 2a, this entry, meets the
done condition. 2b adds video, textures, nuisances, the zoomed camera, the reference feed and
the rest of the §7 catalogue.

### What 2a built

- **Installation:**
  - a bezel (150 mm, reflectance 0.05) with 8 ArUco DICT_4X4_50 markers (80 mm plus a
    one-cell quiet zone): 4 in the corners, 2 above the overlap, 2 below;
  - a static reflectance map painted by exact area coverage, per material;
  - the wall, lit by the room light;
  - default ambient 0.02.
- **Mono:** Rec. 709 luminance taken right after the projector's gamma; RGB stays an option.
- **Six arrangement presets** (side by side, stacked, rotated, corner, different sizes, large
  overlap), each defaulted for the demo screen and centred on it.
- **Time:**
  - exact `Fraction` times and a camera phase;
  - slide decks at three text densities, a held slide, flat, black;
  - an exposure that straddles a picture change mixes both pictures in linear light.
- **Perturbations:** shift, rotation, scale and keystone on A, B or both. They are sized by
  the offset they cause and run on seven schedules; continuous schedules are quantized to
  0.02 px.
- **Truth per frame:**
  - symmetric `offset_mm` and `offset_px`;
  - `h_rel` and `h_actual`;
  - the actual boxes, the camera homography, the visible markers, and the content segments.
- **Frames and datasets:**
  - `FrameSource` renders any frame on demand from cached linear components;
  - datasets hold `scenario.yaml`, `setup.json`, `metadata.jsonl`, optional 16-bit PNGs,
    `dataset.json` and `timing.json`;
  - `make_dataset` and `check_dataset`;
  - scenario files support `extends`, and `sweep` over dotted keys with paired seeds.
- **Scenarios:** `_lecture_hall.yaml` (the base), `aligned_slides`, `shift_sweep` (13
  variants), `rotation_sweep` (9), `scale_keystone` (9). The Phase 1 file still works.

### Numbers

**Markers** (whole-screen camera, 0.868 px/mm, cell 11.6 px, ambient 0.02; paper gives about
240 e⁻ against 15.5 e⁻ of noise):

| Centre estimator | p50 (px) | p95 (px) | max (px) |
|---|---|---|---|
| ArUco SUBPIX corners, diagonal intersection, single noisy frame | 0.156 | 0.315 | 0.431 |
| Line fits to the square's 4 outer edges (`tests/markers.py`), single noisy frame | 0.038 | 0.092 | 0.125 |
| Same, noiseless | — | — | 0.041 |

- Corner bias on a noiseless frame:
  - ArUco corners sit 0.65 px inward on average, up to 0.98 px, without refinement;
  - with SUBPIX refinement, 0.27 px (0.25–0.32).
- The 0.9 px in CLAUDE.md §4.1 matches the unrefined case. Centres cancel the bias either
  way.
- ArUco needs an 8-bit image scaled to the paper's level (paper → 200). With a bright slide
  up, `dn >> 8` or a min-max stretch finds no markers at all (planning review).
- Dark room (ambient 0.0003): 0/8 found in single frames, as planned.
- Tiny test scene (1.23 px/mm, 6 px cells): 8/8, centre error < 0.3 px.

**Rendering** (mono, demo scale, standard quality, this Mac):
- Each new light source (a new slide or a new geometry) costs one camera pass of about 20 ms.
- The first frame takes 0.8 s, which includes the room component and both projectors.
- A new slide takes about 0.4 s: generating the content, two projector components, then
  noise.
- A geometry change takes about 0.17 s.
- An unchanged frame takes 45–51 ms. That is mostly noise (23 ms), summing the components
  (3.5 ms after skipping ×1 multiplies) and encoding.
- Truth costs 0.07 ms per frame.
- Linear superposition differs from a single pass by ≤ 3.6e-7 relative, and OpenCV's results
  do not depend on its thread count (1, 2, 4 or 10 threads; planning review).
- The screen grid with the bezel is 3456 × 8256 at standard quality.
- Phase 1's scene measures as before: unlit 3.6 e⁻, one black 12.0 e⁻, overlap 22.0 e⁻.

**Datasets:**
- `aligned_slides`: 3600 frames in 137–162 s (33–39 ms of rendering per frame), 56 MB, of
  which 6 PNGs.
- `shift_sweep`: 13 × 2400 frames, 76–91 s per variant. It takes 1064 s serially and 492 s
  with `--jobs 4` on 10 cores, and fills 602 MB with 65 PNGs.
- `metadata.jsonl` holds about 1.3 KB per frame.

**Offset definition.** The ground truth is the largest separation of A's and B's copies of the
same content over the calibrated overlap: max over x∈P of |D_B(x) − D_A(x)|, with
D_p = H_act,p·H_cal,p⁻¹.
- The brief's wording, read as max |h_rel(y) − y| over y∈P, samples a slightly different
  region. It is then not symmetric: a 0.1% zoom of A about the screen centre gives
  0.59795 mm, and the same zoom of B gives 0.59855 mm. The symmetric form gives 0.59855 mm
  for both.
- `h_rel` is still recorded.
- The detector's own `offset_mm` (§4.2) should use the symmetric form, so that truth and
  estimate mean the same thing.

### Departures and choices (raised, not silently changed)

1. **Marker centres.** Single-frame ArUco centres miss 0.2 px; edge-line fits meet it.
   - The done condition is checked with the edge-line estimator. It lives in `tests/` as a
     measuring instrument, not detector code.
   - For Phase 3, `rectify.py` should use edge-line fits, or a median over frames, and scale
     to the paper level. CLAUDE.md §4.1 now says so.
2. **Offset definition:** the symmetric form above.
3. **`aligned_slides`** reads "20 min" as 20 minutes *checked*: 600 s of trusted window plus
   1200 s, so 1800 s in all.
   - Densities cycle slide by slide within one dataset. Per-density numbers come from the
     content tags.
   - Slides hold 37 s, so changes wander through the 30 s check intervals.
4. **Shift direction.** `across` is the length-weighted normal of the moving projector's
   inner edges, positive *away* from its partner: the overlap narrows and darkens. `along` is
   that normal turned +90°.
5. **Sizes.** Every perturbation can be sized by the offset it causes (`magnitude_px`), so
   sweeps line up on one axis. Rotation and scale also take `deg` and `factor`.
6. **Renames:**
   - `Camera.exposure` → `well_fill_at_white`, because `exposure_s` is now the time window;
   - `Screen.surround` (a fixed radiance) → `wall_reflectance`, lit by the room light. The
     Phase 1 scenario uses 0.33, which reproduces its old 0.0001.
7. **Frames on disk:**
   - `--frames none` writes truth only and renders nothing;
   - `sample` renders and hashes every frame, but stores a PNG only every 600 frames and at
     each geometry change. Every 60 frames would have filled about 4.5 GB per `shift_sweep`
     run.
8. **Corner preset:** the content rect is the union's bounding box. No rectangle inside the
   L-shaped union crosses the overlap usefully.
9. **Rotated preset:** the pair is re-centred after B turns.
10. **Sweeps:** variants equal in effect (any zero-size perturbation) are kept once, so
    `shift_sweep` has 13 variants, not 14. Names carry the swept values, e.g.
    `magnitude_px=2__direction=across`.
11. **Modules not in §3:**
    - `sim/cfg.py`: YAML reads `1e-3` as a string, so every number goes through one reader;
    - `sim/schedule.py`: schedules shared with the 2b nuisances;
    - `sim/sweep.py`;
    - `scripts/check_dataset.py`;
    - `tests/markers.py`.
12. **Deferred:**
    - to 2b: video, textures, nuisances and flicker, black-level uplift, screen gain, the
      zoomed camera, the reference feed;
    - to Phase 5 or 9: recalibration events (an H_cal change mid-run).

### Code review before committing

A review of the change found nine problems. All are fixed, each with a test.

1. **`check_dataset` checked the simulator against itself.** Its "injected" offset was rebuilt
   from the simulator's own tags, so a wrong size passed. A keystone asked for −4 px came out
   at 3.78 px and still passed. It now reads what was asked for from `scenario.yaml` and
   recomputes geometry and pitch itself.
2. **Negative keystone sizes were solved as positive.** A keystone is not symmetric in k:
   −4 px came out at 3.97 px. The solver now searches on the requested side.
3. **`reference.lag_s` was parsed but ignored.** The projectors now show content(t − lag).
4. **Screen and wall in one grid pixel.** Where a pixel held both (only without a bezel), the
   reflectance was 0.30 instead of 0.48. The fix is bit-identical for the demo's bezel grids.
5. **`across` for a nested projector** was NaN; it is now a clear error ("give a vector").
6. **Content length.** Content had to last until `duration_s` + exposure instead of the end of
   the last exposure, which refused a 6 s held slide for a 6 s run.
7. **A perturbation without a `schedule` was silently never applied.** A schedule is now
   required; `{type: none}` switches a perturbation off.
8. **The pairing check** now compares only variants that differ in nothing but their
   perturbation, so sweeps over the scene (ambient, arrangement, camera) are not paired.
9. **Seeds** are read as exact integers (2⁵³ + 1 stays itself).

## 2026-10-07 — Phase 2b: the rest of the simulator (library, nuisances, zoomed camera, reference feed)

**Done condition met.**
- **Every §7 scenario loads and renders its event frames.** That is the first and last variant
  of each of the 16 files, at frame 0 and at every perturbation or nuisance onset (`pytest -m
  slow`).
- **Reproducible:** `camera_zoomed` (6 variants), `aligned_nuisances` (7), `boundary_hidden`
  (2) and `aligned_video` (1) were generated twice, with `--jobs 4` and `--jobs 5`. Every file
  except `timing.json` and `dataset.json` is byte-identical, which covers 48,000 frames per run.
  `aligned_video` was split over frame ranges, 4 ways and 5 ways.
- **Checks pass:** `check_dataset` passes on all of them.
  - Truth stays aligned throughout the nuisance runs.
  - Elsewhere every frame's offset equals the requested shift.
  - Paired variants share all 9,840 of their pre-onset frames.
- **Zoomed camera:** markers 4–7 are found in every sampled frame at 1.73 px/mm at the image
  centre, with centre error p50 0.032, p95 0.071, max 0.073 px (20 sightings in 5 frames).
- **Nuisance physics** (tests on tiny scenes), with truth aligned and `offset_mm` exactly 0
  throughout:
  - lamp dimming to 0.85 scales that projector's light by exactly 0.85, black level included;
  - a room-light step from 0.02 to 0.05 raises the unlit screen 2.5×;
  - a camera knock of (5, −3) px moves every marker by exactly that;
  - a person walking past darkens the screen behind them to 0.3/0.9 of its level (their
    reflectance against the screen's, under the same light) and hides the markers they cover;
  - flicker vanishes when the exposure spans whole periods, and makes rolling-shutter bands
    when it doesn't;
  - sharpening adds halos without changing the level: at a raster edge, a bright fringe and a
    dark one of 2.2% of the step each (amount 0.8, σ 1 px).
- **Video:** an exposure that straddles two video frames mixes them in exact shares (7/10 and
  3/10).
- **Reference feed:** sends are timestamped on the sending clock, and the projectors show
  content(t − `lag_s`).

### What 2b built

- `sim/pictures.py` — content item kinds and how each is drawn:
  - photo-like stills (1/f^1.2 texture and soft objects);
  - dark film stills;
  - sinusoidal stripes of a given period in screen mm;
  - video;
  - options on every item: `border_frac`, `letterbox`, `blank_overlap`.
- `sim/textures.py` and `sim/video.py`:
  - clips at any fps, with cuts, wrap-around pans, bouncing objects and fades;
  - the background is periodic by construction, so pans have no seam.
- `sim/nuisance.py` and `sim/flicker.py`: camera bump, lamp dimming, room light, occluder,
  in-camera sharpening, and flicker. Flicker is a closed-form row gain that averages the
  modulation over each row's exposure window.
- `sim/state.py`: the frame state.
- `sim/frames.py`:
  - composition with per-projector gains and per-row flicker weights;
  - the occluder, and sharpening after noise;
  - per-camera-knock component caches;
  - the reference feed, `FrameSource.source(i)`.
- **Black-level uplift** (`blend.black_uplift`): each projector adds its partner's black level
  where the partner does not reach. It is built at calibration and reported in `setup.json`.
- **The zoomed camera preset:**
  - it frames the overlap across, and the full height plus the bezel along, with the sensor's
    long side along the overlap (portrait for side by side);
  - it is validated to hold the whole overlap and at least 4 markers.
- **Scenario files and datasets:**
  - each module parses its own block (screen, projectors, camera);
  - `key: null` in a scenario removes a key inherited from its base;
  - long single-variant datasets render in parallel over frame ranges, with byte-identical
    output.
- **The rest of the §7 catalogue:**
  - `held_slide`, `aligned_video`, `aligned_nuisances` (7 variants), `slow_drift`;
  - `arrangements` (30 variants: 6 presets × 5 sizes), `boundary_hidden`, `blank_band`,
    `dark_film`;
  - `repetition_limit` (8), `camera_zoomed` (6), `dark_room` (2).

### Numbers

**Rendering** (demo scale, standard quality; CPU time per frame in each worker):

| Dataset | Frames per variant | ms per frame | Wall time of a run (`--jobs 4`) |
|---|---|---|---|
| `aligned_nuisances` (7 variants) | 3600 | 39–47 | 367 s |
| `camera_zoomed` (6 variants) | 2400 | 33–55 | 239 s |
| `boundary_hidden` (2 variants; half photos, half video) | 2400 | 159–171 | 424 s |
| `aligned_video` (split 4 ways) | 3600 | 735 | 677 s |

Every video frame is new content, and every exposure holds two video frames, so each frame
renders both projectors anew. Sizes per run: 397, 288, 91 and 55 MB, mostly the sample PNGs.

- **Phase 2a datasets still re-render bit-identically.** Frames and truth of `aligned_slides`
  and `shift_sweep` match their stored hashes. Only `metadata.jsonl` gained the nuisance fields
  (`lamp_gain`, `camera_bump`, `occluder`, `flicker`, `sharpening`).
- **Stripes at a half-period shift.** The grating is sinusoidal in light, so with 8 mm stripes
  and B shifted 4 mm the two copies cancel to flat gray where the blend weights are equal (the
  middle of the overlap). Elsewhere a residual contrast of |a − b| remains. This is the aliasing
  the echo test must not take for alignment, and the case `repetition_limit` exists for.

### Departures and choices (raised, not silently changed)

1. **Photo statistics.** The texture's amplitude spectrum falls as 1/f^1.2, the middle of the
   natural-image range (1.0–1.4). With 1/f, a 2 px pan changed a frame by 12% on average.
2. **Flicker** is a sinusoid per projector whose phase is drawn per frame from the seed
   (spawn key (2, flicker, frame)), since the camera's clock is not locked to the projector's.
   Real DLP colour-wheel modulation is not sinusoidal; the closed-form average is exact for
   this model.
3. **Occluder:**
   - a flat silhouette (head, body, legs) in front of the screen;
   - lit like the screen behind it, with reflectance 0.3;
   - it covers the bezel and wall too;
   - no cast shadow.
4. **Camera bumps** are quantized to 1% of their size, so a bump that settles gives few cached
   views.
5. **Screen gain** (a high-gain screen's hotspot) is not built. It would need a gain map per
   projector, and none of the planned detector phases needs it yet.
6. **`dark_room`** has no perturbation and lasts 5 minutes: it only asks whether markers can be
   found, with or without the bezel light, and what averaging buys.
7. **`aligned_nuisances`** runs one nuisance per variant, plus one variant with all of them.
   Flicker and sharpening are installation properties, present from calibration on; the others
   start after the trusted window.
8. **`boundary_hidden` probably does not hide the boundary from the whole-screen camera.** This
   is a prediction to check in Phase 4, not a redesign.
   - The letterbox bars cross the overlap, so both inner edges show through black level: about
     9 e⁻ against 17 e⁻ of noise per pixel, which is SNR ≈ 6 per frame along a 117-row bar.
   - The left and right outer edges are strong content borders.
   - If so, the brief's question ("how often is the boundary unavailable?") gets the answer
     "rarely", and Phase 6c's done condition will need content whose overlap is never dark.
9. **Scenarios that use the reference feed:** `aligned_video` (lag 0.1 s, i.e. 3 video frames,
   with cuts) and `repetition_limit` (lag 0.1 s). `dark_film` sweeps black-level uplift on and
   off.

### Code review before committing (2b)

A review found ten problems and a minor one. One of them, `boundary_hidden`, is the prediction
in item 8 above. The others are fixed, with tests wherever they concern code:

1. **Flicker was frozen.** Frames are 0.5 s apart, exactly 50 cycles of 100 Hz, so every frame
   caught the same phase and the bands never moved: the flicker went into the baseline and
   could never trouble the detector. Each frame now draws its own phase (spawn key (2, flicker,
   frame)), since the camera's clock is not locked to the projector's.
2. **`repetition_limit` held one static picture.** The variety gate would have starved the echo
   pools. It now shows 40 stripe pictures at random phases, with the source feed on. Its
   comment no longer claims the reference kernel can see a whole-period shift: for periodic
   content the kernel is periodic too.
3. **`blank_overlap` painted gray over the black border and the letterbox bars** inside the
   overlap. It now fills only the picture area.
4. **Stripes were sinusoidal in code values.** After gamma 2.2 that meant a second harmonic at
   20%, so a half-period shift did not cancel. They are now sinusoidal in light.
5. **The reference feed was anchored at the exposure's start.** With no lag it missed the
   newest picture shown during the exposure. It now lists what was sent before the exposure
   ends.
6. **A camera bump, lamp or room-light nuisance without a schedule** silently did nothing. A
   schedule is now required, as for perturbations.
7. **`cut_in_exposure` was true for every video frame**, since each exposure spans two. It now
   flags only real changes (a cut, a new picture). The new `frames_in_exposure` counts the
   pictures.
8. **No scenario used uplift, the reference feed or a lag.** See item 9 above.
9. **This entry still had placeholders** for the run numbers; they now hold the measured
   values.
10. **`visualize.py` placed its level patches with the camera before any knock.**

## 2026-10-08 — Phase 2 refinement before the pull request

Three read-only reviews went over everything Phase 2 changed: the rendering and geometry
code, the content, timeline, dataset and script files, and the docs, scenarios and tests.
Their findings are fixed below. None of the changes alters a frame that was already generated:
- **Stored datasets still re-render bit-identically.** Sampled frames of 8 stored datasets were
  re-rendered after the last change and match their stored sha256: `aligned_slides`, two
  `shift_sweep` variants, `aligned_video`, two `aligned_nuisances` variants (including frames
  with a person passing), `boundary_hidden` and the earlier `camera_zoomed`.
- **Content pictures are unchanged.** The content refactor below leaves all 1,703 pictures the
  catalogue draws identical, video frames included (every item sampled, hashed before and
  after).
- **Variant names are unchanged.** All 99 names in the catalogue stay the same, so existing
  dataset directories keep their names.
- **Tests:** 280 (20 slow), all passing.

### Performance

- **Buffer reuse.** Each frame used to allocate several grid- and sensor-sized temporaries,
  hundreds of MB in all. They are now kept between frames:
  - a zeroed screen grid, which each projector's light is drawn into and wiped from;
  - a warp window per projector (A's and B's differ in shape);
  - the camera's blurred, warped and supersampled images.

  Values cannot change, because every operation writes all of its output. One process now
  renders a video frame in 160 ms instead of 178 ms.
- **Parallel workers are limited by memory traffic.** With 4 workers, each video frame takes
  0.97–1.03 s in each worker, so 4 workers render 3.9–4.1 frames/s against 5.6–6.2 for one
  process.
  - Profiling showed every operation uniformly 3–4× slower, not one hot spot.
  - Limiting BLAS to one thread changed nothing (4.3 frames/s).
  - This explains the 735 ms per frame in the 2b table: it is CPU time per frame inside each of
    4 workers.
  - Static content still gains from workers: the new `camera_zoomed` ran in 251 s with
    `--jobs 4`, at 45–47 ms per frame per worker. `make_dataset --help` now says so.
- **Cost per frame now** (demo scale, standard quality, one process):

  | Frame | Cost |
  |---|---|
  | unchanged (noise and encoding only) | 24 ms |
  | new slide | 126 ms |
  | video | 163 ms |

- **The `fine` preset** (Phase 1's deferred item) uses 3 × 3 screen samples per projector
  pixel and 3 × 3 camera sub-pixels:
  - its screen grid is 64 Mpx, against 28 Mpx for `standard`;
  - a new slide takes 190 ms, against 126 ms.

### `camera_zoomed` is now paired with `shift_sweep`

- It now has `shift_sweep`'s seed (21), content (decks and black) and onset, so the zoomed and
  whole-screen cameras compare shift for shift. Before, it had its own seed, photos and steps.
- **Steps:** 0, 0.25, 0.5, 1, 1.5, 2, 4 and 8 px across, 8 variants. 1.5 px (1.56 mm) sits
  just above the zoomed echo floor, max(`MIN_OFFSET_MM`, 2.5 px / 1.74 px/mm) = 1.5 mm.
- **Reproducible:** regenerated twice, with `--jobs 4` and `--jobs 5`.
  - Every file under `out/p2b` except `timing.json` and `dataset.json` is byte-identical,
    which now covers 52,800 frames per run.
  - `check_dataset` passes.
  - The 8 variants share all 9,840 of their pre-onset frames.
- **Markers:** 4–7 are found in all 5 sampled frames. Centre error is p50 0.034, p95 0.072,
  max 0.089 px (20 sightings), against 0.032 / 0.071 / 0.073 px with the old content.

### Fixes, each with a test

1. **Occluders.** Each person passing keeps their own reflectance. Before, when two people
   were in the frame at once, the last one's reflectance applied to both. No stored frame had
   two people at once.
2. **`check_dataset` re-evaluates every schedule kind itself:** drift, ramp, bump_then_hold
   and oscillate as well as step and staircase. Before, it trusted the recorded `scheduled`
   value.
   - Its own geometry moved to `scripts/check_geometry.py`, so both files stay under 300
     lines.
   - A new test sweeps rotation, scale and keystone, each stepped, ramped and drifting. It
     covers the closed forms and the brute force, and a corrupted line must fail.
3. **Stripes** use projector A's gamma from the installation instead of an assumed 2.2. Content
   is encoded once for both projectors, so if B's gamma differed, the stripes would be
   sinusoidal in A's light only.
4. **Video items** refuse four bad settings, each with its own message: `fps: 0`, a fps that is
   not a number, `cut_s: 0`, and `fade_s` without cuts (which was silently ignored).
5. **Datasets:**
   - `dataset.json` records `git_dirty`: whether `sim/` differed from the recorded commit;
   - a missing PNG raises `FileNotFoundError` naming it;
   - `variants.json` keeps the variants an earlier run wrote, in the sweep's order;
   - `--every 0` is refused.
6. **`visualize.py`:**
   - it places its labels and level patches where the boxes land in the frame shown (it used
     the calibrated boxes);
   - it names its output after the variant and frame, so views no longer overwrite each
     other.
7. **Sweep labels.** A list of mappings is labelled by each mapping's `type`, `kind` or
   `preset`. Before, it took each mapping's first value.
8. **New tests for existing behaviour:**
   - the cached components sum to one camera pass over the total radiance, bezel light
     included, also when buffers are reused;
   - the reference feed runs `lag_s` ahead of the display;
   - a bezel light of 0.02 makes all 8 markers findable at ambient 0.0003, where none are found
     without it;
   - sharpening rings at an edge, 2.2% of the step on each side;
   - the occluder darkens by exactly 0.3/0.9;
   - every scheduled nuisance starts after the trusted window;
   - the slow catalogue test also renders one variant per installation (every arrangement
     preset);
   - ArUco's own sub-pixel corners miss the 0.2 px target that the edge-line fits meet.

### Clean-ups (no change in behaviour)

- **One implementation each:**
  - `content.framed` draws every content kind inside its black border;
  - `textures.blend_ellipse` serves photos and video;
  - `sim.dataset.process_pool` serves `make_dataset` and the frame-range split;
  - `planar.local_scale` and `planar.is_vertical` replace four and two inline copies;
  - `CalibrationSetup.finest_pitch_mm` replaces `pixel_pitch_mm`, which quietly meant the
    finest pitch;
  - the projector names are always (a, b), which fixes the direction of `h_rel`.
- **Renamed:**
  - the bezel's own light (it was called "lamp", the word for projector lamps);
  - `Camera.to_setup_dict`, `ScreenGrid.centres_mm`;
  - the schedule fields `step` and `settled`;
  - `tests/markers.refined_corners`.
- **Removed:**
  - `Renderer.render` and `Renderer.expected_electrons`, a second frame path without
    nuisances;
  - `Screen.unlit_radiance` and `wall_radiance`, `Scene.camera_preset`, `Perturbation.spec`,
    and `blank_inside`'s `margin_px`;
  - the Phase 1 alias `type: slide` (`held` is the same thing).
- **Corrected docs:**
  - the 2b entry's flicker departure (the phase is drawn per frame);
  - the occluder ratio;
  - CLAUDE.md's reference-mode claim: repetitive content fools the kernel at whole-period
    shifts;
  - the zoomed camera needs a vertical overlap;
  - `boundary_hidden` probably keeps the boundary visible;
  - the slide charts are laid out for side by side.

## 2026-10-09 — Phase 2c: screen gain, dataset verification, the source feed on the shift sweeps

Phase 2c finishes what Phase 2 deferred and left unverified. An audit of `main` after PR #2 found
the simulator complete except for screen gain, and found that the "generate twice identically"
done conditions had been checked by hand and that `check_dataset` trusted parts of what it
checked. 2c builds the gain screen, the tools that check those claims, and the source feed that
Phase 7 needs on `shift_sweep`; the remaining real-life disruptions are GitHub issues #3–#18.

**Done condition met.**
- **Gain screen:** at peak 1.8 and 2.4, in a flat gray frame of `gain_screen`, the camera's
  brightness net of room light over the matte twin's equals the lobe's gain at each projector's
  hotspot and 600 mm toward its own edge, within 1.0e-4 (table below; the test holds it to 0.1%).
- **Matte is unchanged:** peak 1, or positions without gain, render bit-identical frames (tiny
  scenes: still, slide change, shifted); every stored 2a/2b dataset passes `check_dataset
  --rerender` before and after the gain code, with identical results: 32 variants, 87,600
  metadata lines, 766 frames re-rendered to their recorded hashes, 169 PNGs re-hashed.
- **Generate twice identically, now by a tool:** `gain_screen` (6 variants, 14,400 frames),
  `shift_sweep` (13, 31,200) and `camera_zoomed` (8, 19,200), each generated with `--jobs 4`
  and again with `--jobs 5`, are the same datasets by `compare_datasets`: every file but
  `timing.json`, environment included, 64,800 frames and 135 PNGs. All three pass
  `check_dataset --rerender 10 --strict` with nothing skipped.
- **The checker catches what it used to trust:** a rotation recorded as a shift of equal offset
  (the offset alone still matched), a wrong lamp gain, room light, knock or passer-by flag, a
  dropped marker, a wrong frame time, a wrong picture or tag, a stale or missing PNG, a wrong
  frame hash on re-render, and `setup.json` disagreeing with the scenario on the source feed.
- **The source feed:** on `shift_sweep` and `camera_zoomed` it changes exactly the frames
  predicted, 37 per variant (481 and 296 in all), in `content.segments`, `content.tag` and
  `frame_sha256` only; truth is unchanged and all 105 stored PNGs are byte-identical.

### What 2c built

- **`sim/room.py`: positions and the gain screen.**
  - `projectors.<p>.position_mm` and `camera.position_mm` place the lenses in the room (x, y on
    the screen, z toward the room). Defaults: each projector 1.6 image widths in front of its
    image's centre; the camera 1.5 screen widths away and 1.25 screen heights below the screen's
    top edge, as the whole-screen camera's keystone implies. Demo: A (1205, 749.85, 3200), B
    (2795, 750.15, 3200), camera (2000, 1875, 6000) mm.
  - `screen.gain: {peak, lobe_deg, kind}`: G = 1 + (peak − 1)·exp(−½(α/lobe_deg)²), α the angle
    between the direction to the camera and the mirror image of the incoming ray (`specular`) or
    the ray back to the projector (`retro`). Each projector has its own hotspot: specular at
    P_xy + (C_xy − P_xy)·pz/(pz + cz).
  - Each projector's light meets its own "reflectance toward the camera" map, built once per
    projector when the screen has gain: the matte map plus (G − 1)·ρ_screen times the screen's
    share of each grid pixel, so the bezel, marker paper and wall stay matte and a pixel on the
    screen's edge gains only for its screen part. It replaces the reflectance multiply in the
    projector components, so a held slide still costs only noise; room and bezel light keep
    the matte map.
  - The positions never reach `setup.json` (a test holds the allowlist), and nothing reads them
    on a matte screen.
- **`scripts/compare_datasets.py`** (with `scripts/dataset_files.py`, the checkers' own reading
  of a dataset): two runs are the same dataset when every file but `timing.json` agrees, the
  environment in `dataset.json` aside (reported, never counted). Differing metadata lines are
  diffed field by field ("37 frames differ, first frame 74, in content.segments, frame_sha256");
  `--ignore-fields` handles datasets written before a field existed.
- **`check_dataset`**, split into `check_geometry.py`, `check_timeline.py` and `check_frames.py`
  (stored, re-rendered and paired frames) so each file stays under 300 lines and only
  re-rendering imports `sim/`:
  - every frame's `h_actual` is rebuilt from the request and the recorded multipliers (shifts by
    their recorded vector, rotations and scales by m × the requested size about the requested
    pivot, keystones along the requested axis with the recorded strength of the requested sign),
    so the brute-force offset is measured on a verified homography, and a keystone at full
    strength is also measured by brute force;
  - `h_rel` from `h_actual`; `setup.json`'s source feed and exposure against the scenario; the
    frame count;
  - the timeline from the scenario text alone: frame times as exact fractions, the nuisance
    state (room light, lamps, knocks quantized to 1%, passers-by, flicker, sharpening), the
    pictures shown with their shares, tags and cuts (lag included), the knocked camera, and the
    visible markers (all of those in view when nobody passes, a subset when someone does);
  - stored PNGs re-hashed; `--rerender K` renders K spread frames and up to K stored ones again;
  - a check that cannot run (no hashes, fields a dataset predates) is reported under `skipped`
    or `paired_skipped_no_hashes` instead of passing silently; `--strict` fails on it.
- **Catalogue:** `scenarios/gain_screen.yaml` (6 variants: peak 1, 1.8, 2.4 × shift 0, 2 px);
  `shift_sweep` and `camera_zoomed` gain the source feed (lag 0.1 s), and stay paired.
- **Visualizer:** a cross at each projector's hotspot on a gain screen, and `hotspot_mm` in its
  summary.

### Numbers

**Gain screen** (`gain_screen`, frame 400, flat gray 0.5, noiseless; 21 × 21 camera-pixel
patches ≈ 24 mm; A and B are mirror images, so B's numbers equal A's):

| peak | where | lobe G | measured, net of room light | error | measured, with room light |
|---|---|---|---|---|---|
| 1.8 | hotspot (1481.5, 1141.3) | 1.8000 | 1.7999 | −4.0e-5 | 1.7328 |
| 1.8 | 600 mm toward A's edge | 1.5760 | 1.5758 | −7.3e-5 | 1.5275 |
| 2.4 | hotspot | 2.4000 | 2.3999 | −5.2e-5 | 2.2823 |
| 2.4 | 600 mm toward A's edge | 2.0079 | 2.0077 | −1.0e-4 | 1.9231 |

- **Room light dilutes the hotspot.** Gain applies to projector light only, and the room light
  (0.02 of white, against 0.22 for gray 0.5) is not gained, so the camera sees 1.73 where the
  screen's gain is 1.80. A first version of the demo test compared hotspot ratios with the lobe
  and missed by 0.7%; this was the reason, and the test now subtracts the room-light image.
- **The overlap is a trough of both lobes.** At peak 1.8, A's gain falls from 1.636 to 1.438
  across the overlap (x = 1795 to 2205 mm) while B's rises from 1.438 to 1.636; at the centre
  both are 1.546. On flat gray the camera sees 1.58 × the matte level at the overlap's edges and
  1.50 at its centre, smooth and symmetric: no seam, but a 5% dip that the hotspot fit's lamp-gain
  and trend terms must absorb (it is static, so it is part of the baseline). At the far edges of
  the picture A's gain is 1.17 (x = 205) and B's 1.02.
- **Cost:** a reflectance map is 114 MB per projector at demo scale `standard` (3456 × 8256
  float32), built in 0.34 s, once per process; a gain frame with new content took 0.69 s
  (including one map) against 0.26 s matte. `gain_screen` generated in 227 s with `--jobs 4`
  (6 × 2400 frames) and in 252 s with `--jobs 5`; for comparison, the paired `shift_sweep`
  took 504 and 447 s, and `camera_zoomed` 277 and 251 s.

**Stored datasets, before and after the gain code** (`check_dataset --rerender 20`, `aligned_video`
`--rerender 10`):

| dataset | variants | lines | re-rendered | PNGs | result |
|---|---|---|---|---|---|
| `out/p2/run1/aligned_slides` | 1 | 3,600 | 25 | 6 | ok, 7 checks skipped (pre-2b schema) |
| `out/p2/run1/shift_sweep` | 13 | 31,200 | 312 | 65 | ok, same 7 skipped; 15,990 paired frames |
| `out/p2b/run1/camera_zoomed` | 8 | 19,200 | 192 | 40 | ok, nothing skipped; 9,840 paired frames |
| `out/p2b/run1/aligned_nuisances` | 7 | 25,200 | 175 | 42 | ok, nothing skipped |
| `out/p2b/run1/boundary_hidden` | 2 | 4,800 | 48 | 10 | ok, nothing skipped; 2,460 paired frames |
| `out/p2b/run1/aligned_video` | 1 | 3,600 | 14 | 6 | ok, nothing skipped |

- The new independent checks found no error in any stored dataset: the nuisance states of all 7
  `aligned_nuisances` variants, the 0.1 s lag of `aligned_video`'s content (exposures spanning
  two video frames, cuts and fades), and every knocked camera and visible-marker list agree with
  the scenario text.
- The 2a datasets predate six metadata fields (`content.frames_in_exposure` and five nuisance
  fields), and the camera check needs one of them: the checker reports those seven checks as
  skipped, and `--rerender` lists the six as fields the current code adds.
- `compare_datasets` confirms the hand comparisons of Phase 2: `out/p2/run1` and `run2` are the
  same datasets (34,800 frames; `run3`'s `aligned_slides` too), and so are `out/p2b/run1` and
  `run2` (52,800 frames).

**The source feed on the shift sweeps.** Content changes at T = 191c + {37, 74, 111, 148, 185,
191} s, and frame 2T is exposed from T + 0.0123 s for 1/30 s. Without the feed's lag that
exposure showed the new picture; with the projectors 0.1 s behind the sender it shows the old
one, and no exposure straddles a change either way (0.046 s < 0.1 s). That predicts 37 frames
per variant in 1200 s (six changes in each of six cycles, and one at 1183 s), none of them a
stored PNG frame (0, 600, 1200, 1230, 1800). `compare_datasets` against the stored runs finds
exactly those in all 21 variants: frames 74, 148, 222, 296, 370, 382, …, 2292, 2366, differing
in `content.segments`, `content.tag` and `frame_sha256` only. The PNGs are byte-identical, and
`scenario.yaml` and `setup.json` differ only in the reference block. (`out/p2/run1/shift_sweep`
predates 2b, so that comparison ignores `content.frames_in_exposure` and `nuisances`.) The
checker re-derives the same content from the scenario text, lag included, on every frame.

### Departures and choices (raised, not silently changed)

1. **Gain as a reflectance map per projector, not a gain map multiplied after it.** The same
   physics, but exact on pixels that straddle the screen's edge (only the screen part gains),
   and it replaces a multiply instead of adding one. `projector_irradiance` stays gain-free,
   because irradiance is the light that lands; `screen_radiance` applies the gain, so the
   components still sum to one camera pass (tested with and without gain).
2. **Room and bezel light keep gain 1**, and the bezel, paper and wall stay matte. A real gain
   screen also rejects room light arriving off axis; issue #5 proposes an `ambient_gain`.
3. **A passer-by is lit by the gained projector light** (the occluder rescales the projector
   components); a person is not the screen. Issue #3 proposes fixing it with the shadows.
4. **The demo test measures the gain net of room light**, to 0.1%, instead of the planned 2% on
   hotspot ratios, which room light dilutes (above).
5. **`check_frames.py`** holds the PNG, re-render and pairing checks, apart from
   `check_dataset.py`, so that file stays under 300 lines and the one import of `sim/` is
   isolated (a test holds the checker scripts free of `sim/` at module level).
6. **The checker now uses each frame's exact time**, phase_s + i × sample_every_s, rather than the
   recorded float `t_s` read back as a decimal; `t_s` itself is checked against it.
7. **A knocked camera is checked** against the frame-0 camera turned and shifted by each bump's
   recorded strength. Frame 0's own camera is the reference: its pose is truth that no input
   records.
8. **The mistake tests for the new keys** live in `tests/test_room.py`, beside the model, rather
   than in `tests/test_scenario.py`.

### Review pass before merging

A second read of everything 2c changed found four gaps in the checker and two small crashes. All
are fixed, each with a test:

1. **A shift beside another kind went unchecked.** A shift's recorded vector was held to its size
   and direction only when every active perturbation was a shift. Beside a rotation, only the
   brute-force offset ran, and it confirms that the recorded offset and `h_actual` agree, not
   that the shift is the size asked for. Every active shift is now checked, whatever else is active.
2. **A partial keystone was only self-consistent.** A ramped keystone's strength k was checked
   against `h_actual` and its brute-force offset, both written from the same k. Now k / m must
   be the same in every frame, which ties each partial frame to the full-strength one, whose size
   is checked against the request.
3. **`--rerender K` re-rendered everything on a `--frames all` dataset**, since every frame has a
   PNG. It now renders up to K of the stored frames, evenly spread. A `--frames sample` dataset
   stores 5 or 6 PNGs per variant, so with K = 10 or 20, as in every run above, the frames chosen
   are the same as before.
4. **An empty `metadata.jsonl` crashed the checker**; it now fails with a message.
5. **The visualizer crashed on a retro-reflective screen seen from a projector's own distance**,
   where that projector has no hotspot; it now leaves the hotspot out.
6. **Clean-ups:** an unused constant is gone, and the pairing check moved beside the other
   frame-hash checks.

**The whole catalogue, truth only.** Every scenario was written with `--frames none` and checked
with the strengthened checker:
- 107 variants and 277,201 frames all pass, with nothing skipped;
- the worst closed-form offset error is 9.9e-13 mm;
- the 4,680 keystone frames of `scale_keystone`, measured by brute force, agree within
  1.4e-14 mm, and their strength per unit m is constant;
- the pairing check reports, rather than passes, the groups it cannot pair without frame hashes.

**Code review on PR #19** (high effort, posted as inline comments) found seven things; six are
fixed, each with a test where it concerns code:

1. **The shift direction was checked against the line between the box centres**, not against
   the inner-edge normal that defines `across`. They differ by 20.05° for projector B in the
   `rotated` preset and by 22.38° in `corner`. So an `along` shift there would have been reported
   as wrong (|cos| 0.34–0.38 against a 0.1 limit), and `across` passed only narrowly (0.92 against
   0.9). The checker now derives the inner-edge normal with its own geometry
   (`Geometry.across`) and holds every shift to its direction within 1e-9. The whole catalogue's
   truth passes the exact check, the rotated and corner arrangements included.
2. **A passer-by on a gain screen inherits the screen's gain.** This is left to issue #3: no
   scenario combines the two yet.
3. **`--rerender` accepted a negative K** and then rendered nothing; it is now refused.
4. **The RGB gain branch of `screen_radiance` was untested.** The components-sum test now runs it.
5. **Two recursive JSON differs** (the comparison's and the re-render's) are now one, in
   `scripts/dataset_files.py`.
6. **Each perturbation's fixed geometry was recomputed on every frame**: its pivot, reach,
   angle or factor, and direction. It is now computed once per dataset (`Request`).
7. **Open issues were described as done** in CLAUDE.md and in this entry; the wording now says
   they propose the change.

**Docs.**
- CLAUDE.md §4.3 gave the zoomed echo floor as ≈ 1.4 mm. Its own formula gives 1.5 mm, because
  `MIN_OFFSET_MM` binds there: 2.5 camera px are 1.44 mm at 1.74 px/mm. `camera_zoomed.yaml`
  already said 1.5 mm; §4.3 now does too.
- §5 lists the disruptions still to build, with their issue numbers; §11 asks what screen the real
  installation has.

**Tests:** 344 (322 default + 22 slow), and ruff is clean.

### Issues filed for the rest

Real-life disruptions that are not misalignment (decision 12): occluder cast shadows (#3),
projector defocus (#4), room-bounce light (#5), flicker and sharpening on a schedule with a DLP
waveform (#6), dynamic iris (#7), non-uniform room light (#8), camera creep and vibration (#9),
one-projector overlays and dropouts (#10), a presenter standing for minutes (#11), room-light
flicker (#12), reference-feed timing jitter (#13). Leftovers: recalibration mid-run (#14),
hygiene from the audit (#15: zoomed refusal by name, a bounded camera-view cache, gain-map
memory, pixel comparison). Scenarios: never-dark overlap content for `boundary_hidden` (#16),
a thermal cycle with A or both perturbed (#17), `aligned_nuisances` paired with
`aligned_slides` (#18).

## 2026-10-10 — Demo site: what its illustrations measured, and two routing thresholds to revisit

`python -m scripts.make_site --serve` builds a static site into `out/site` and serves it on
localhost (`demo/`, `scripts/make_site.py`; README "Demo site"). It shows:
- sample frames with their ground truth drawn on top;
- the planned detector explained step by step, with five small interactive models;
- the test suite as run at build time, test by test, next to the §8 done conditions and this file.

A standard-quality build takes about 63 s: 19 s for the default suite (354 tests, 23 slow ones
deselected) and 39 s of rendering, with a 2.9 GB peak, 97 images and 8.4 MB. `demo/manifest.py`
re-checks every chosen frame against its scenario before rendering, without rendering, so a changed
scenario stops the build instead of mislabelling a picture. `demo/` uses `sim/` and the harness
helper `scripts/visualize.py`, never `detector/`; `tests/test_imports.py` holds it to that, also
through what importing it loads, and holds both packages to never importing it.

Its algorithm figures compute, with ground truth at hand, what the detector will compute. They
are teaching illustrations, not detector code, but their numbers are worth keeping. All of them
come from the whole-screen camera at standard quality, against the aligned twin of each sweep
variant.

### Numbers

**Pairing.** In `shift_sweep` frame 1300, where A shines alone in both frames, every variant
(0.25 to 8 px across, 8 px along) differs from its aligned twin by exactly 0 electrons. The
noise draws are shared, so a difference map shows only what B's move changed.

**Hotspot** (`gain_screen`, matte, flat grey 0.5, B 2 px across, frame 1250).
- The overlap dims by 0.7268% at its middle, 0.363% per projector pixel.
- The model is b(D⁻¹(x)) − b(x) times L / (L + room + 2·black) = 0.911, with nothing fitted: D is
  where B's calibrated picture now lands (the truth's displacement map; x − u for this shift), and
  L and the black levels come from the projector's own light model. It gives 0.7269% and matches
  the noiseless profile to 2.1e-7 rms inside the overlap, 5 mm clear of its edges.
- One noisy exposure, averaged along the overlap, stays within 2.3e-6 rms of the model.
- §4.3's planning figure of 0.40% per pixel assumed no room light; at ambient 0.02 the dilution
  of 0.911 gives 0.364%.
- B's raster edge leaves a −0.18% sliver at x ≈ 1796 mm that the ramp model leaves out.

**Echo** (`demo/cepstrum.py`). 30 core tiles of 56 camera px (TILE_MM at 0.864 px/mm), their
control tiles, and the 16 distinct slides after the onset, pooled and scored once with the
config's thresholds. The floor is 2.5 camera px, or 2.89 mm.

| B moved | true offset | best peak beyond the floor | needed (MIN_PEAK_SNR, 1.5 × null) | verdict |
|---|---|---|---|---|
| 0 | 0 | 9.9 | 27.4 | none |
| 1 px | 1.04 mm | 17.9 | 24.0 | none |
| 2 px | 2.08 mm | 40.5, on the floor's rim | 21.3 | unresolved |
| 4 px | 4.17 mm | 99.6 | 22.0 | resolved at 4.24 mm |
| 8 px | 8.33 mm | 92.1 | 27.4 | resolved at 8.21 mm |

The null test is what rejects the aligned twin, as the 2026-10-07 replan expected. Two parts of
the design turned out to matter even for an illustration:
- **The null, and enough content.** In a trial at fast quality, pooling only the 5 slides of one
  loop over half the core tiles, the aligned twin scored 22.5 against a null of 10.7, a false
  detection. 16 slides and every core tile were needed.
- **Agreement between tiles.** Scored tile by tile instead (fast quality, 16 tiles, 16 slides),
  8 px passed in only 2 tiles, and the aligned twin's one passing tile pointed elsewhere. So the
  per-tile agreement step (RANSAC, Phase 6a) carries real weight. The tests needed to set its
  inlier fraction belong to Phase 6a.

"Unresolved" here means a peak that passes but is only the flank of something higher inside the
floor; that rule is the illustration's, and Phase 6a should define its own.

**Boundary evidence** (black frames 1517–1528 against 371–382, mean of 12, room light removed;
per-pixel noise 16.8 e⁻ in one frame, 4.9 e⁻ in the mean).
- Black level, at the middle of each region: 8.9 e⁻ where A shines alone, 8.8 e⁻ where B does,
  18.0 e⁻ in the overlap, on 265 e⁻ of room light.
- With B moved 8.33 mm, the 50% crossings of B's raster edges, averaged down the boxes' height,
  moved 8.32 and 8.32 mm. A's moved 0.000. B's picture edge on a slide moved 8.333 mm.

**Truth timelines.** `slow_drift` crosses TOLERANCE_MM (1.7 mm, 1.63 px) at 6468.0 s (frame 12936),
1.63 h after its onset at 600 s. That is the 1 px/h rate, quantized to 0.02 px. Every frame is read
for this; an earlier build sampled every 20th and reported 6470 s.

### Raised, not changed (for Phase 3, `classify.py`)

1. **DARK_LEVEL (0.003 of white) sits below the room light (0.02).** Overlap means in the content
   library, as measured and then net of the room light:

   | frame | as measured | room light removed |
   |---|---|---|
   | black (`shift_sweep` 375) | 0.0213 | 0.0013 |
   | dark film, moving (`dark_film` 100) | 0.0235 | 0.0036 |
   | dark film, still (`dark_film` 500) | 0.0399 | 0.0200 |

   Read literally ("region mean below DARK_LEVEL × white"), no region is ever dark at ambient
   0.02, so the inner edges would never be seen. Judged net of the unlit level that the screen
   outside both rasters shows, black frames are dark, but neither dark-film frame is, the moving
   one by only 0.0006. Phase 3 should define "dark" relative to that level and decide whether dim
   film counts.
2. **MOTION_LEVEL (0.02 of white) can never catch dark scenes.**
   - Moving dark film changes 0.0035 of white between samples, a sixth of the threshold, so it is
     never skipped.
   - Photo-style video changes 0.056 and 0.076, so it is.
   - A threshold relative to the region's own level would treat both alike.

The demo applies both thresholds net of the room light and labels the result. It changes no
config value.

### Review pass before the pull request

A review of the demo (PR #20) changed what some figures said; the numbers above are after it.
- **Edge pieces.** CLAUDE.md §2 calls an edge inner only inside the other box *and* inside the
  content. Inside the other box but outside the content, both projectors show black, so the
  edge's black-level step shows in every frame. `rotated` has 5 such pieces, which had been
  counted as inner: 118 outer, 33 inner, 5 margin. The other five presets have none.
- **Pixels in the boundary toy** now use each arrangement's own coarser pitch: 0.729 mm for
  `stacked` and 0.833 mm for `corner`, not side by side's 1.042 mm.
- **Captions** of the nuisances come from the scenario file. The camera bump is (+3, −2) px,
  3.6 px, which a hand-written caption had called 3 px.
- **Provenance.** Each data file records when, from which commit and at which quality it was
  made, so a page rebuilt with `--samples skip --tests skip` no longer credits reused figures or
  test results to the new build.
- **Speed.** The cepstrum's tile search tests many positions at once: the same tiles at fast and
  standard quality, in 0.1 s instead of 3.2 s. Rendering takes 39 s instead of 42 s, and the
  peak is 2.9 GB instead of 3.2 GB; the 2.5 GB first reported for it was low.
