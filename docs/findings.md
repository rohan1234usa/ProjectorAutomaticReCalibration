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
- the slide's bare diagonal is replaced by a framed line chart that still crosses the
  overlap.

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
