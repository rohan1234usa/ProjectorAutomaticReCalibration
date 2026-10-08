# Edge-Blended Projector Misalignment Detector — Pseudocode

*Original design document, Oct 1, 2026, by Rohan. Transcribed from the PDF "Stacked Projector
Misalignment Detector — Pseudocode". Identifiers that the PDF extraction had split across lines
are reassembled; the pipeline figure is described in words. CLAUDE.md is the current brief: it
keeps this document's methods for the overlap check, its frame routing, decision rule, edge cases
and stress tests, and puts boundary analysis of the projector rectangles first.*

## Problem and assumptions

The script watches the overlap band during normal use and answers one question: do the two
projectors still agree where their images overlap? It returns **YES** (misaligned, press
recalibrate) or **NO**, replacing the professor's visual check.

The design rests on four assumptions:

- **Partial overlap, edge-blended.** Each projector covers part of the screen. In a shared overlap
  band both show the same pixels, faded with complementary blend ramps so brightness stays even.
  The check concentrates on this band.
- **Passive camera.** A fixed camera sees the band plus a strip on each side of it. The script
  never changes what is projected.
- **Recalibration is ground truth.** Right after the recalibrate button is pressed, the system is
  aligned. The script captures its baseline at that moment.
- **Two input modes.** *Blind mode* uses only the camera. *Reference mode* also reads the video
  frame being sent to the projectors, if the hardware allows it. It is more sensitive and harder
  to fool.

All offsets below are in projector pixels, on a rectified canvas covering the band and its side
strips.

## Core idea: misalignment shows up inside the overlap band

The projectors share pixels only in the overlap band, so that is where they can be compared. When
they drift apart, the band shows two copies of the picture offset by **d**, each faded by its
blend ramp.

```
O(x) ≈ a(x)·S(x − d_A) + b(x)·S(x − d_B),        a(x) + b(x) = 1
```

O is the rectified camera image, S is the content, a and b are the blend weights across the band
in linear light, and the misalignment is **d = d_B − d_A**. The echo is strongest in the middle of
the band, where a ≈ b. Rotation or keystone drift makes **d** change along the band, so it is
measured in tiles stacked down the band.

Three properties make this work on live content:

- **Self-referencing.** The check compares projector A to projector B, not to a stored position.
  A bumped camera shifts both copies equally and cannot fake a YES.
- **Built-in control.** The strips beside the band are lit by one projector each, so they show the
  same kind of content with no echo. A pattern found inside and outside the band is content; one
  found only inside is misalignment.
- **Every frame type is useful.** Textured frames feed the echo, flat frames reveal the blend
  seam, and dark frames outline each projector's raster edge through its black level.

The cepstrum turns the echo into a single peak at **q = d**, because the log turns content × echo
into content + echo:

```
C(q) = | F⁻¹{ log |F{O}| } |
```

Three signals can measure **d**; the script uses whichever the hardware and the current content
allow.

| Signal | Needs | Misalignment looks like | Limit |
|---|---|---|---|
| Echo in the band (cepstrum) | Camera only; textured content in the band | A peak at **d** inside the band that is missing from the control strips | Can't resolve offsets under about 1.5 px |
| Seam profile | Camera only; any content, flat and dark frames best | The band turns brighter or darker than at calibration; in dark frames the raster edges move | Sees only the shift across the seam, not along it; weak for small shifts in a wide band (2 px in a 384 px linear blend is about 0.5% brightness) |
| Kernel (deconvolution) | Camera + the video frame sent to the projectors | Each projector places the content at a different spot; the band's kernel splits into two blobs | Needs a capture of the video signal and frame sync |

## Pipeline overview

*Figure, described:* "A YES needs a smooth offset down the overlap band, repeated over time. Blind
mode uses the camera only; reference mode adds the source frame (dashed)." Frames flow left to
right along the top: **Camera frame** (every 0.5 s) → **Rectify** (linear-light canvas) →
**Usable?** (no: moving or clipped → skip) → yes → **Pool band tiles** (plus control strips; in
reference mode the **Source frame** feeds in here) → after 60 usable frames → measurements flow
right to left along the bottom: **Measure offset** (echo, seam, kernel) → **Fit offset line**
(smooth along band) → **Vote** (3 of last 4) → **YES / NO** (holds if unsure). **Recalibrate**
(button pressed) → **Capture baseline** (known-good state), which feeds "subtract baseline" into the
line fit and "reset to NO" into the vote. Only the recalibrate button resets the baseline and the
answer. 11 steps, 1 gate.

## Pseudocode 1: configuration and baseline

Setup runs at install and again every time the recalibrate button is pressed, because that is the
only moment the system is known to be aligned. The camera only needs to see the band and a strip
on each side, so it can be zoomed in for more pixels per projector pixel. Every number below is a
starting guess to tune on real hardware.

```python
# ---------------- GEOMETRY (example: two 1920x1080 projectors side by side, 384 px overlap) ----
BAND_W, BAND_H = 384, 1080          # overlap band, in projector pixels (read from the blending setup)
CTRL_W = 256                        # single-projector strip kept on each side, for control tiles
ROI_W, ROI_H = CTRL_W + BAND_W + CTRL_W, BAND_H   # the only region the camera must see
BAND_X0, BAND_X1 = CTRL_W, CTRL_W + BAND_W       # band columns inside the ROI canvas
BLEND_A, BLEND_B = load_blend_ramps()   # each projector's weight across the band; sum to 1 in
                                        # linear light (from the blending software; else assume
                                        # linear ramps)

# ---------------- TILES ----------------
BAND_TILES = 6          # tiles stacked down the band; rotation shows as offsets that vary along it
CORE_FRACTION = 0.6     # echo tiles use the middle 60% of the band, where a and b are both large

# ---------------- THRESHOLDS ----------------
MIN_OFFSET_PX = 1.5     # smallest offset the echo test can resolve (blind mode)
TOLERANCE_PX = 2.0      # offset that should trigger YES -- ask the professor
POOL_FRAMES = 60        # usable frames pooled per measurement
SAMPLE_EVERY_S = 0.5    # no need to look at every video frame
MIN_TILE_FRAMES = 20    # an echo tile needs this many textured frames to count
MIN_SEAM_FRAMES = 20    # a seam profile needs this many lit (or dark) frames
MIN_PEAK_SNR = 6.0      # a peak must stand this far above the noise (robust z-score)
MIN_GOOD_TILES = 3      # fewer valid band tiles -> "can't tell yet"
YES_VOTES = (3, 4)      # YES needs 3 of the last 4 measurements over tolerance
CLEAR_RATIO = 0.5       # back to NO only below 0.5 x tolerance (hysteresis)
REFERENCE_MODE = source_feed_available()   # can we read the video sent to the projectors?

# ---------------- CAMERA ----------------
def init_camera():
    cam = open_camera()
    cam.lock(exposure=True, gain=True, white_balance=True, focus=True)  # auto modes break the math
    cam.set_exposure(multiple_of(PROJECTOR_REFRESH_PERIOD))             # avoids flicker banding
    return cam

# ---------------- BAND GEOMETRY ----------------
def find_band_homography(frame):
    # Printed markers on the screen frame above and below the band: passive, so they work during
    # live content.
    markers_cam = detect_fiducials(frame)                        # 4 points, camera pixels
    return homography(markers_cam, MARKER_POSITIONS_ON_CANVAS)  # camera px -> ROI canvas px
                                                                 # (measured once)

def rectify(frame, Hmat):
    lin = to_linear_light(to_gray(frame))   # undo camera gamma so the blended copies ADD
    return warp_perspective(lin, Hmat, (ROI_W, ROI_H))

# ---------------- BASELINE: after every recalibration ----------------
def on_recalibration_complete(cam):
    Hmat = find_band_homography(cam.read())
    echo_pools, seam_pools = collect_pools(cam, Hmat, 3 * POOL_FRAMES)  # same as runtime, longer
    tiles = [measure_tile(echo_pools[i]) for i in range(BAND_TILES)]
    baseline = Baseline(
        homography = Hmat,
        residual_line = fit_band_field(tiles) or ZERO_LINE,  # offset left right after calibration
        seam = seam_pools,                                    # the band's brightness profile when aligned
    )
    save(baseline)
    reset_decision_state()                                    # start fresh at NO
    return baseline
```

## Pseudocode 2: measuring the offset

Each band tile pools many frames, then yields one offset. Every tile has two control tiles just
outside the band, one per projector, so content patterns can be told apart from a real echo. Each
frame is routed by type: textured frames feed the echo, and every lit or dark frame feeds the seam
profile.

```python
# ---------------- ROUTE EACH FRAME ----------------
def classify(img, prev_img):
    if mean_abs_diff(img, prev_img) > MOTION_LEVEL: return "skip"  # motion or a cut: frames blend
    if fraction_clipped(img) > MAX_CLIPPED: return "skip"          # saturation breaks the sum model
    if mean(band(img)) < DARK_LEVEL: return "dark"                 # black level outlines the raster edges
    return "lit"

# ---------------- POOLS ----------------
class EchoPool:                  # one tile: in the band core, or a control tile beside it
    def __init__(self):
        self.sum_log_spec = zeros(TILE_SHAPE)              # blind mode
        self.cross = zeros(TILE_SHAPE, dtype=complex)      # reference mode: sum of O * conj(S)
        self.power = zeros(TILE_SHAPE)                     # reference mode: sum of |S|^2
        self.count = 0
    def add(self, cam_tile, src_tile=None):
        O = fft2(prep(cam_tile))
        self.sum_log_spec += log(abs(O) + EPS)
        if src_tile is not None:
            S = fft2(prep(src_tile))
            self.cross += O * conj(S)
            self.power += abs(S) ** 2
        self.count += 1

class SeamPool:                  # long-run brightness profile across the band, one per band tile
    def __init__(self):
        self.lit, self.dark = zeros(ROI_W), zeros(ROI_W)
        self.n_lit = self.n_dark = 0
    def add(self, strip, kind):
        prof = mean(strip, axis="rows")                    # average down the tile: one value per column
        if kind == "lit":
            self.lit += prof / mean(prof); self.n_lit += 1  # normalise out overall brightness
        else:
            self.dark += prof; self.n_dark += 1

def prep(tile):
    # A LINEAR high-pass keeps the two-copy model intact; the window stops tile edges faking peaks.
    return highpass(tile - mean(tile)) * hann2d(TILE_SHAPE)

def collect_pools(cam, Hmat, n_frames):
    echo = [{"band": EchoPool(), "ctrl_a": EchoPool(), "ctrl_b": EchoPool()}
            for _ in range(BAND_TILES)]
    seam = [SeamPool() for _ in range(BAND_TILES)]
    prev, last_kept, used = None, None, 0
    while used < n_frames:
        img = rectify(cam.read(), Hmat)
        kind = classify(img, prev) if prev is not None else "skip"
        if kind != "skip" and (REFERENCE_MODE or differs_from(img, last_kept)):  # blind mode needs variety
            src = matching_source_frame(img) if REFERENCE_MODE else None
            for i in range(BAND_TILES):
                seam[i].add(row_strip(img, i), kind)
                if kind == "lit" and edge_density(band_core(img, i)) > MIN_TILE_EDGES:
                    # src crops are None in blind mode
                    echo[i]["band"].add(band_core(img, i), band_core(src, i))
                    echo[i]["ctrl_a"].add(left_strip(img, i), left_strip(src, i))    # lit by A only
                    echo[i]["ctrl_b"].add(right_strip(img, i), right_strip(src, i))  # lit by B only
            used, last_kept = used + 1, img
        prev = img
        sleep(SAMPLE_EVERY_S)
    return echo, seam

def matching_source_frame(img):
    # The camera lags the video output by a few frames: pick the buffered frame that matches best.
    candidates = [to_roi_canvas(s) for s in source_ring_buffer.last(10)]  # cropped, resized, linear light
    return max(candidates, key=lambda s: ncc(downsample(img), downsample(s)))

# ---------------- ECHO / KERNEL: one band tile ----------------
def measure_tile(pools):
    if pools["band"].count < MIN_TILE_FRAMES:
        return TileResult(valid=False)
    return kernel_offset(pools) if REFERENCE_MODE else echo_offset(pools)

def cepstrum(pool):
    return fftshift(abs(ifft2(pool.sum_log_spec / pool.count)))

def zscore(c):
    return (c - median(c)) / (1.4826 * mad(c))

def echo_offset(pools):   # BLIND MODE
    band = zscore(cepstrum(pools["band"]))
    ctrl = mean([zscore(cepstrum(pools[k])) for k in ("ctrl_a", "ctrl_b")])
    diff = band - ctrl                 # content patterns appear on both sides; the echo only in the band
    zero_disk(diff, center(diff), radius=MIN_OFFSET_PX)   # the origin always peaks; ignore it
    loc = argmax2d(diff)
    d = subpixel_peak(diff, loc) - center(diff)           # sign is ambiguous: +d or -d
    if diff[loc] < MIN_PEAK_SNR:
        d = (0, 0)                                        # textured band, no echo: aligned here
    return TileResult(offset=d, snr=diff[loc], valid=True)

def kernel_offset(pools):   # REFERENCE MODE
    k_a = wiener_kernel(pools["ctrl_a"])     # where projector A alone puts the content
    k_b = wiener_kernel(pools["ctrl_b"])     # where projector B alone puts the content
    k_band = wiener_kernel(pools["band"])    # both: one blob if aligned, two if not
    d = subpixel_peak(k_b) - subpixel_peak(k_a)           # B relative to A; camera motion cancels out
    agrees = count_blobs(k_band) == 2 or norm(d) < 2 * MIN_OFFSET_PX   # the band should confirm it
    snr = min(peak_snr(k_a), peak_snr(k_b))
    return TileResult(offset=d, snr=snr, valid=agrees and snr >= MIN_PEAK_SNR)

def wiener_kernel(pool):
    return fftshift(real(ifft2(pool.cross / (pool.power + WIENER_LAMBDA))))  # multi-frame deconvolution

# ---------------- SEAM PROFILE: offset across the seam only ----------------
def seam_offset(pool, base):
    estimates = []
    if pool.n_lit >= MIN_SEAM_FRAMES and base.n_lit >= MIN_SEAM_FRAMES:
        ratio = (pool.lit / pool.n_lit) / (base.lit / base.n_lit)  # cancels vignetting and screen gain
        # Shifting B by dx moves its ramp: the band dims or brightens. A dimmer lamp tilts it instead.
        # Fit both, keep dx: ratio(x) ~ trend(x) * (A(x) + g * B(x - dx)) / (A(x) + B(x))
        dx, g = least_squares_fit(ratio, BLEND_A, BLEND_B, params=["dx", "g", "trend"])
        estimates.append(dx)
    if pool.n_dark >= MIN_SEAM_FRAMES and base.n_dark >= MIN_SEAM_FRAMES:
        # In dark frames each projector's black level outlines its raster: B's left edge sits inside
        # A's image. How far that edge moved since calibration is the offset across the seam.
        estimates.append((edge_position(pool.dark, "B") - edge_position(base.dark, "B"))
                         - (edge_position(pool.dark, "A") - edge_position(base.dark, "A")))
    return median(estimates) if estimates else None
```

## Pseudocode 3: decision and main loop

A YES needs three things: enough band tiles have evidence, their offsets fall on a smooth line
down the band, and the offset stays over tolerance across several measurements. Echo and seam are
combined so either can raise the alarm; voting filters out one-off spikes.

```python
# ---------------- COMBINE BAND TILES ----------------
def fit_band_field(results):
    good = [r for r in results if r.valid]
    if len(good) < MIN_GOOD_TILES:
        return None                       # not enough evidence yet
    align_signs(good)                     # blind echo gives +/-d; pick signs that agree with
                                          # neighbours and the seam
    # Shift, rotation and keystone all give an offset that changes smoothly down the band.
    line, inlier_frac = ransac_fit_line([r.y for r in good], [r.offset for r in good])  # d(y) = d0 + y*d1
    if inlier_frac < 0.6:
        return None                       # tiles disagree: likely content, not drift
    return line

def misalignment_px(line, baseline):
    # Largest change from the calibrated state anywhere along the band, in projector pixels.
    return max(norm(line(y) - baseline.residual_line(y)) for y in band_sample_rows())

def combine(echo_px, seam_px):
    found = [abs(v) for v in (echo_px, seam_px) if v is not None]
    return max(found) if found else None  # None = can't tell this round

# ---------------- DECISION WITH HYSTERESIS ----------------
state, votes = "NO", deque(maxlen=YES_VOTES[1])

def decide(offset):
    global state
    if offset is None:
        return state                      # can't tell: hold the last answer rather than guess
    votes.append(offset > TOLERANCE_PX)
    if state == "NO" and sum(votes) >= YES_VOTES[0]:
        state = "YES"
    elif state == "YES" and offset < CLEAR_RATIO * TOLERANCE_PX:
        state = "NO"; votes.clear()
    return state

# ---------------- MAIN LOOP ----------------
def main():
    cam = init_camera()
    baseline = load_baseline() or on_recalibration_complete(cam)
    on_event("recalibration_done", lambda: on_recalibration_complete(cam))  # if the button is readable
    while True:
        frame = cam.read()
        if fiducials_moved(frame, baseline.homography):
            baseline.homography = find_band_homography(frame)  # camera bumped: re-rectify only
        echo_pools, seam_pools = collect_pools(cam, baseline.homography, POOL_FRAMES)
        tiles = [measure_tile(echo_pools[i]) for i in range(BAND_TILES)]
        line = fit_band_field(tiles)
        echo_px = misalignment_px(line, baseline) if line else None
        seams = [seam_offset(seam_pools[i], baseline.seam[i]) for i in range(BAND_TILES)]
        seams = [s for s in seams if s is not None]
        seam_px = percentile([abs(s) for s in seams], 80) if len(seams) >= MIN_GOOD_TILES else None
        offset = combine(echo_px, seam_px)
        answer = decide(offset)
        publish(answer)                               # the YES / NO output
        log(now(), answer, echo_px, seam_px)          # keep the numbers for tuning
```

The log matters as much as the answer: comparing logged offsets against the professor's own calls
is how TOLERANCE_PX and the other thresholds get set.

## Edge cases and failure modes

The biggest risk is a false YES from content that looks like an echo. The control strips, the
smooth-line check and voting each guard against it.

| Situation | What goes wrong | How the design handles it |
|---|---|---|
| Repetitive content (text lines, grids, tables) | A false cepstral peak at the pattern's spacing | The same pattern shows in the control strips and cancels; tiles must agree down the band; reference mode is immune |
| Nothing textured inside the band (a slide margin, a sky) | No echo signal | The seam profile carries the check; flat content is ideal for it |
| One slide left up for a long time | Blind echo can't pool varied content | Seam profile still updates; otherwise the last answer holds |
| Motion or scene cuts | The camera blends two video frames, which looks like a ghost | Skipped by the motion gate |
| Dark scenes and fades | No texture for the echo | Routed to the dark seam profile: each projector's black level outlines its raster edge |
| Camera bumped | The band lands in the wrong place on the canvas | Echo and kernel compare the projectors to each other, so no false YES; fiducials restore the mapping |
| Camera auto-exposure or gamma | Blended copies and ramps stop adding linearly | Lock exposure, gain and white balance; convert to linear light |
| Projector flicker or DLP color banding | Stripes create false peaks and seam ripples | Exposure set to a multiple of the refresh period |
| One projector dimmer (lamp aging) | The band's brightness tilts, which could pass for a seam shift | The seam fit separates gain from shift; echo position is unaffected |
| One projector out of focus | The band looks soft but has no double edges | Correctly NO for misalignment; reference mode can flag it as a separate warning |
| Drift along the seam (vertical, for side-by-side) | The seam profile can't see it | Echo and kernel measure both directions |
| Offset under about 1.5 px | Blind echo can't separate it from the origin | Zoom the camera on the band, or use reference mode |

## Open questions and next steps

The first two questions set the geometry everything else hangs on; the rest pick the mode and the
config values.

- Where exactly is the overlap band, and how wide is it? The blending software should report it;
  it sets BAND_W and the camera's region of interest.
- What shape are the blend ramps (linear, cosine, custom), and are they applied in linear light or
  gamma? The seam model needs them.
- Can the script read the video sent to the projectors (screen capture on the source PC, or an
  HDMI splitter into a capture card)? Yes unlocks reference mode.
- Camera resolution, lens and mount: can it be zoomed on the band so each projector pixel spans at
  least 2 camera pixels?
- How large an offset does the professor notice from the seats? That sets TOLERANCE_PX.
- Is black-level compensation enabled outside the band? It changes how dark frames look, though
  the baseline comparison still works.
- Does the recalibrate button emit a signal the script can read, so the baseline refreshes
  automatically?
- Can printed markers go on the screen frame above and below the band?

The algorithm can be built and tested before any hardware arrives:

1. **Simulate.** Take slides and video clips and render the band as O = a(x)·S(x) + b(x)·warp(S),
   with the blend ramps and a small shift or rotation. Add camera perspective, blur, noise and
   gamma.
2. **Sweep.** Run offsets from 0 to 5 px, across and along the seam, plus small rotations. Record
   how often each signal says YES at each offset.
3. **Stress test.** Text-heavy slides, dark film scenes, fast motion, a blank band, one slide held
   for 20 minutes, one projector dimmed by 15%.
4. **Move to hardware.** Capture a baseline right after a real recalibration, then compare logged
   offsets with the professor's own calls to set the thresholds.
