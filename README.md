# V3Kite.jl

[![Build Status](https://github.com/OpenSourceAWE/V3Kite.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/OpenSourceAWE/V3Kite.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

Julia package for simulation and validation of the TU Delft V3 leading edge
inflatable (LEI) kite. Built on
[SymbolicAWEModels.jl](https://github.com/OpenSourceAWE/SymbolicAWEModels.jl);
ships calibration, model setup, CSV replay, and bundled V3 geometry.

## Installation

```bash
git clone https://github.com/OpenSourceAWE/V3Kite.jl
cd V3Kite.jl
./bin/install
./bin/run_julia
```

## Quick Start

```bash
./bin/run_julia
```

```julia
include("examples/menu.jl")
```

Pick an example from the menu. See `V3SimConfig` in `src/simulation.jl` for
the simulation options; bundled geometry and flight data live at
`v3_data_path()`.

> **First run is slow** (compilation + system build). Subsequent runs are
> fast — keep the same REPL open across runs.
You can make the first run much faster by using a custom system image. You can create it with the script `bin/create_system_image`. This requires up to 64GB RAM (for example 16GB physical RAM and a 48 GB swap file) and 30 min of time (on a laptop with a Ryzen 7840U CPU).

## Examples

All examples share one project — `julia --project=examples`. Run the
interactive launcher with `examples/menu.jl`, or pick one from the table:

| # | Script | What it does |
|---|--------|--------------|
| 1 | `v3kite.jl` | Hello-world: heading PID + winch PID + 3D replay. Start here. |
| 2 | `examples_2d/reel_out_v3.jl` | Single reel-out maneuver, 2D ControlPlots. |
| 3 | `realtime.jl` | Keyboard-controlled simulation (arrows steer/depower, ESC stops). |
| 4 | `open_loop.jl` | Settle in power zone, then ramped open-loop steering. |
| 5 | **`flight_replay.jl`** | Replays real EKF flight data through the simulator. See below. |
| 6 | **`batch_run_circles.jl`** | Parameter sweep of circular-flight runs. See below. |
| 7 | `batch_load_circles.jl` | Loads a batch directory; writes metrics CSV + scatter plot. |
| 8 | `batch_run_zenith_then_circles.jl` | Two-phase variant (zenith hold → circles). |
| 9 | `load_and_plot.jl` | Post-processes any saved log: timeseries, 3D replay, line stretch. |

Utilities (no simulation): `photogrammetry_aoa.jl`, `plot_wind_sources.jl`,
`depower_drum_model.jl`.

### Flight replay

`flight_replay.jl` slices a maneuver from an EKF H5 by UTC, settles the wing
into the recorded conditions, then steps the simulator while feeding
recorded steering/depower/tether inputs. A second `SymbolicAWEModel` driven
straight from the EKF state is plotted alongside. Outputs land in
`processed_data/`; PDFs go to `output/` when `SAVE_FIGS=true`. Toggles for
maneuver, year, feedback gains, and tape reductions are at the top of the
script.

### Batch sweeps

`batch_run_circles.jl` runs a grid of circular-flight sims (settle → ramp
steering → early-stop on course-rate convergence). Logs land in
`processed_data/<batch_tag>/`; failures get listed in `failed_runs.txt`.
Edit `defaults`/`sweeps`/`combine_all` at the bottom of the script to
define the grid. Then:

```bash
julia --project=examples examples/batch_load_circles.jl
```

prompts for a batch directory and emits `circles_batch_analysis.csv` plus a
`|u_s · v_a|` vs `|χ̇|` scatter.

## Calibration

| Constant | Value | Meaning |
|----------|-------|---------|
| `V3_STEERING_L0_BASE` | 1.6 m | Neutral steering tape length |
| `V3_DEPOWER_L0_BASE` | 0.2 m | Neutral depower tape length |
| `V3_STEERING_GAIN` | 1.4 m | Max differential at 100% steering |
| `V3_DEPOWER_GAIN` | 5.0 m | 0–100% depower stroke |

Tape reductions are applied via `V3GeomAdjustConfig` and `set_steering!` /
`set_depower!` — see their docstrings.

## Visualization Extension

When GLMakie is loaded, extra plotters become available, e.g.:

```julia
plot_body_frame_local(sys_struct; dir=:front)
```

## Testing

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Related Packages

- [SymbolicAWEModels.jl](https://github.com/aenarete/SymbolicAWEModels.jl) — symbolic kite modeling
- [VortexStepMethod.jl](https://github.com/aenarete/VortexStepMethod.jl) — aerodynamics
- [KiteUtils.jl](https://github.com/aenarete/KiteUtils.jl) — shared utilities

## License

This project is licensed under the MPL-2.0 License. The documentation is licensed under the CC-BY-4.0 License. Please see the below `Copyright notice`.

## Copyright notice

Technische Universiteit Delft hereby disclaims all copyright interest in the package “V3Kite.jl” (model of the V3 kite) written by the Author(s).

Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering, Technische Universiteit Delft.

See the copyright notices in the source files, and the list of authors in [AUTHORS.md](AUTHORS.md).
