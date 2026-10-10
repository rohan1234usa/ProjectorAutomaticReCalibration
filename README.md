# Projector Automatic Re-Calibration

A camera watches two edge-blended projectors that form one combined image, and answers one
question during normal use: **have they drifted out of alignment since the last
calibration?** The answer is `YES` (press recalibrate) or `NO`, with the estimated offset in
millimetres.

There is no hardware yet. Everything is built and tested against a physical simulator of the
projectors, the screen and the camera, which also generates ground-truth datasets with known
misalignments.

- [CLAUDE.md](CLAUDE.md) is the project brief: the decisions, the detector design, the
  simulator, the build order and the config contract.
- [docs/research/pseudocode.md](docs/research/pseudocode.md) is the original design this work
  generalizes.
- [docs/findings.md](docs/findings.md) is a dated log of results with numbers.

## Status

Phases 1 and 2, the simulator, are done. It renders two edge-blended projectors in any of six
arrangements, on a matte or gain screen framed by a bezel with printed fiducial markers, through
a locked 16-bit mono camera that sees the whole screen or zooms on the overlap. Over time it:
- plays slides, photos, stripes and synthetic video;
- moves the projectors by known amounts (shift, rotation, scale, keystone, on any schedule);
- adds nuisances that must never cause a YES (camera knocks, lamp dimming, room light, people
  walking past, flicker, sharpening, black-level compensation);
- records ground truth for every frame.

`make_dataset` turns any of the 17 scenario files into a reproducible dataset. Two checkers
verify datasets with their own code, apart from the simulator: `check_dataset` holds every
frame's truth to what its scenario asked for, and `compare_datasets` proves two runs identical.
The disruptions that are still to be simulated are tracked as
[GitHub issues](https://github.com/rohan1234usa/ProjectorAutomaticReCalibration/issues).

A demo site shows all of this in a browser: sample frames with their ground truth, the planned
detector explained step by step, and the test results (see [Demo site](#demo-site) below).

Next is Phase 3, where the detector itself starts: its inputs and geometry (CLAUDE.md §8).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Everything runs from the repository root.

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run python -m scripts.visualize scenarios/aligned_side_by_side.yaml --out out/phase1
```

The last command writes `out/phase1/view.png` with two panels:
- the simulated camera frame, shown with a log display so the 1/1500 black level is visible;
- the same frame with each projector's footprint, their overlap and, with a bezel, the markers
  drawn on it.

It also prints the render time and the light levels it measured. `--variant` and `--frame`
pick any frame of any scenario, written as `view_<variant>_frame<i>.png`.

Generate a dataset, then check its ground truth against what was injected:

```bash
uv run python -m scripts.make_dataset scenarios/shift_sweep.yaml out/shift_sweep --frames sample --jobs 4
```

```bash
uv run python -m scripts.check_dataset out/shift_sweep --rerender 10
```

`--rerender 10` also renders ten frames of each variant again, and up to ten of the stored ones,
and compares them with the recorded hashes. To prove that a second run makes the same dataset,
generate it again and compare the two:

```bash
uv run python -m scripts.make_dataset scenarios/shift_sweep.yaml out/shift_sweep_again --frames sample --jobs 2
```

```bash
uv run python -m scripts.compare_datasets out/shift_sweep out/shift_sweep_again
```

`--frames none` writes the ground truth alone in seconds. Any frame can be re-rendered from a
dataset's `scenario.yaml`; with `--frames sample` or `all`, its stored hash proves it is the
same frame. The slow, demo-scale tests run with `uv run pytest -m slow`.

## Demo site

A static site shows sample frames from the simulator with their ground truth, explains the
planned detector step by step with small interactive models, and lists the test results test by
test. Build it and serve it on localhost:

```bash
uv run python -m scripts.make_site --serve
```

Then open http://localhost:8000. The build renders about a hundred figures (about a minute at
standard quality), runs the default test suite and reads CLAUDE.md, docs/findings.md and
detector.yaml into `out/site/`. `--samples fast` renders quicker. `--tests all` adds the slow
tests. `--samples skip --tests skip` keeps the last build's figures and results, to refresh the
pages alone. The pages also open straight from disk, without a server.

## Layout

| Path | What it holds |
|---|---|
| `sim/` | The simulator: screen, bezel and markers, projectors and their arrangements, where they stand and the screen's gain, the calibration software's blending setup, content over time, perturbations, camera, render chain, ground truth and datasets. |
| `detector/` | The detector, which only sees camera frames and the blending setup. So far it holds its configuration. |
| `scripts/` | Command-line tools: make, view, check and compare datasets; build the demo site. The checkers keep their own code, apart from `sim/`. |
| `demo/` | The demo site: which frames it shows, the figures drawn from them, the readers of the docs and the test run, and the hand-written pages. It uses `sim/`, never `detector/`. |
| `scenarios/` | One YAML file per test idea; `_lecture_hall.yaml` is the installation they share. |
| `tests/` | The test suite. |

`detector/` and `sim/` never import each other, and neither imports `demo/`.
