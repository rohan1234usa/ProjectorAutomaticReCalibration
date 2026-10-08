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
  - a person walking past darkens the screen behind them to 0.2–0.5 and hides the markers they
    cover;
  - flicker vanishes when the exposure spans whole periods, and makes rolling-shutter bands
    when it doesn't;
  - sharpening adds halos without changing the level.
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
2. **Flicker** is modelled as a sinusoid per projector, with a fixed random phase from the seed
   and no per-frame randomness. Real DLP colour-wheel modulation is not sinusoidal; the
   closed-form average is exact for this model.
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
   found, with or without the bezel lamp, and what averaging buys.
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
9. **Placeholders in this entry.**
10. **`visualize.py` placed its level patches with the camera before any knock.**
