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
| 1 | `v3kite.jl` | Hello-world: heading PID + 3D replay. Start here. |
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
straight from the EKF state is plotted alongside (solid: simulation, dashed:
flight). Outputs land in `processed_data/`; PDFs go to `output/` when
`SAVE_FIGS=true`. Toggles for maneuver, year, feedback gains, and tape
reductions are at the top of the script.

The figures below replay a 9-second straight-to-right-turn segment of the V3
kite (Oct. 2025) from measured inputs — the validation from our Torque 2026
paper.

#### Coupled model

<p align="center"><img src="docs/figures/coupled_model_replay.png" width="60%" alt="Coupled model replay"></p>

VSM aerodynamics (10 sections, 36 panels, 19 polars at Re = 10⁶ with canopy
billowing) coupled to a 44-point, 95-segment structural model with Dyneema
tethers. Symbolic ODEs via `ModelingToolkit.jl`, `FBDF` stiff solver.

#### Trajectory and time series: drag slightly underestimated

<p align="center"><img src="docs/figures/flight_replay_trajectory.png" width="60%" alt="Replay trajectory"></p>

<p align="center"><img src="docs/figures/flight_replay_panels.png" width="90%" alt="Replay panels"></p>

Orientations match closely at first; the simulated kite then accelerates a
little, diverging in path through the turn. Course χ tracks well, but tether
force F_t and apparent wind v_app slightly exceed measured values — a minor
underestimation of residual drag. A ~1.5° steering offset balances left/right
turns, compensating for the kite's aerodynamic asymmetry.

#### Turn rate gain agrees closely

`G_k = χ̇ / (v_a · u_s)` relates course rate to airspeed and steering, fitted
by least squares over the right turn:

<p align="center"><img src="docs/figures/flight_replay_yaw_rate_course.png" width="90%" alt="Yaw rate vs course"></p>

#### Wing shape matches photogrammetry

Depower tapes are calibrated on the straight-flight frame (7182) to match the
photogrammetric mean angle of attack. Spanwise twist, simulation (blue) vs
photogrammetry (orange), for straight flight (7182, left) and a right turn
(7362, right):

<p align="center">
<table><tr>
<td><img src="docs/figures/twist_dist_frame_7182.png" width="100%"></td>
<td><img src="docs/figures/twist_dist_frame_7362.png" width="100%"></td>
</tr></table>
</p>

The asymmetric twist from differential steering is reproduced; mean twist in
the turn is slightly underestimated.

### Batch sweeps

`batch_run_circles.jl` runs a grid of circular-flight sims (settle → ramp
steering → early-stop on course-rate convergence). Logs land in
`processed_data/<batch_tag>/`; failures get listed in `failed_runs.txt`.
Edit `defaults`/`sweeps`/`combine_all` at the bottom of the script to
define the grid. Then:

```bash
julia --project=examples examples/batch_load_circles.jl
```

prompts for a batch directory and emits `circles_batch_analysis.csv` plus the
plots below.

`|u_s · v_a|` vs `|χ̇|`, one dot per run, colored by swept parameter; line is
`G_k` fit on the default runs:

<p align="center"><img src="docs/figures/circles_batch_usva_vs_course_rate.png" width="100%" alt="Batch scatter"></p>

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

- [SymbolicAWEModels.jl](https://github.com/OpenSourceAWE/SymbolicAWEModels.jl) — symbolic kite modeling
- [VortexStepMethod.jl](https://github.com/OpenSourceAWE/VortexStepMethod.jl) — aerodynamics
- [AtmosphericModels.jl](https://github.com/OpenSourceAWE/AtmosphericModels.jl) — wind shear and turbulence models  
- [KiteUtils.jl](https://github.com/OpenSourceAWE/KiteUtils.jl) — shared utilities

## License

This project is licensed under the MPL-2.0 License. The documentation is licensed under the CC-BY-4.0 License. Please see the below `Copyright notice`.

## Copyright notice

Technische Universiteit Delft hereby disclaims all copyright interest in the package “V3Kite.jl” (model of the V3 kite) written by the Author(s).

Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering, Technische Universiteit Delft.

See the copyright notices in the source files, and the list of authors in [AUTHORS.md](AUTHORS.md).
