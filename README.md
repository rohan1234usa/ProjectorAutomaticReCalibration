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

Phase 1, the simulator core, is done. It renders two aligned projectors through a simulated
camera: a seamless combined image, with each projector's faint black-level raster around it.
Phase 2, scenario datasets with ground truth, is next. The detector itself starts in Phase 3
(CLAUDE.md §8).

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
- the same frame with each projector's footprint and their overlap drawn on it.

It also prints the render time and the light levels it measured.

## Layout

| Path | What it holds |
|---|---|
| `sim/` | The simulator: screen, projectors, the calibration software's blending setup, content, camera and render chain. |
| `detector/` | The detector, which only sees camera frames and the blending setup. So far it holds its configuration. |
| `scripts/` | Command-line tools. |
| `scenarios/` | One YAML file per test idea. |
| `tests/` | The test suite. |

`detector/` and `sim/` never import each other.
