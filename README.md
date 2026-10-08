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

Phases 1 and 2a are done. The simulator renders two edge-blended projectors in any of six
arrangements, on a screen framed by a bezel with printed fiducial markers, through a locked
16-bit mono camera. It plays slide decks over time, moves the projectors by known amounts
(shift, rotation, scale, keystone, on any schedule), and records ground truth for every frame.
`make_dataset` turns a scenario file into a reproducible dataset.

Next are Phase 2b (video, textures, nuisances, the zoomed camera, the reference feed) and
Phase 3, where the detector itself starts (CLAUDE.md §8).

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
- the same frame with each projector's footprint, their overlap and the markers drawn on it.

It also prints the render time and the light levels it measured. `--variant` and `--frame`
pick any frame of any scenario.

Generate a dataset, then check its ground truth against what was injected:

```bash
uv run python -m scripts.make_dataset scenarios/shift_sweep.yaml out/shift_sweep --frames sample --jobs 4
```

```bash
uv run python -m scripts.check_dataset out/shift_sweep
```

`--frames none` writes the ground truth alone in seconds. Any frame can be re-rendered from a
dataset's `scenario.yaml`, and its stored hash proves it is the same frame. The slow,
demo-scale tests run with `uv run pytest -m slow`.

## Layout

| Path | What it holds |
|---|---|
| `sim/` | The simulator: screen, bezel and markers, projectors and their arrangements, the calibration software's blending setup, content over time, perturbations, camera, render chain, ground truth and datasets. |
| `detector/` | The detector, which only sees camera frames and the blending setup. So far it holds its configuration. |
| `scripts/` | Command-line tools. |
| `scenarios/` | One YAML file per test idea; `_lecture_hall.yaml` is the installation they share. |
| `tests/` | The test suite. |

`detector/` and `sim/` never import each other.
