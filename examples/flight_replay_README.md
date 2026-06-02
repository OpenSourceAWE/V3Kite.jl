# Flight Replay Adjusted: Settings Reference

This document explains all top-level settings in `examples/flight_replay_adjusted.jl`.

## What this script does

The script replays recorded flight test data through the physics model, compares simulated vs logged behavior, and generates plots.

Main flow:

1. Load and slice flight data by UTC time window.
2. Build and optionally settle the kite model.
3. Replay steering/depower/tether commands over time (or by path distance).
4. Save logs and generate comparison plots.

## Settings at a glance

| Setting | Type | Units | Default in script | What it controls |
|---|---|---:|---:|---|
| `LOAD_FROM_DISK` | Bool | - | `false` | Skip simulation and only load previously saved logs for plotting. |
| `BASE_SECTION` | String | - | `"straight_right"` | Maneuver name prefix used with `YEAR` to select the replay window. |
| `YEAR` | Int | year | `2025` | Selects data file, section times, and depower offset set. |
| `SETTLE` | Bool | - | `true` | Run pre-replay settling routine before replay loop. |
| `DEPOWER_OFFSET_2019` | Float | percent points | `7.0` | Constant depower bias used when `YEAR == 2019`. |
| `DEPOWER_OFFSET_2025` | Float | percent points | `-7.0` | Constant depower bias used when `YEAR == 2025`. |
| `STEERING_MULTIPLIER` | Float | scale | `1.0` | Multiplies CSV steering command before applying it. |
| `EXTRA_WING_DRAG_COEFF` | Float | Cd | `0.0` | Extra parasitic drag coefficient distributed over wing points. |
| `HEADING_KP` | Float | controller gain | `0.0` | Proportional heading tracking gain. |
| `HEADING_TI` | Float/Bool | s (if enabled) | `0.0` | Heading integral time in PID constructor (see note below). |
| `LATERAL_KP` | Float | controller gain | `0.0` | Proportional gain for lateral position correction. |
| `STEERING_OFFSET` | Float | percent points | `1.5` | Constant steering bias added in replay loop (`/100` before use). |
| `DISTANCE_BASED_STEERING` | Bool | - | `false` | Use path-distance indexed command lookup instead of step/time index. |
| `REDUCE_STEERING` | Bool | - | `true` | Enable steering tape base-length reduction in geometry config. |
| `STEERING_REDUCTION` | Float | m | `0.2` | Amount subtracted from steering tape base length when reduction is enabled. |
| `REDUCE_TIP` | Bool | - | `true` | Enable tip segment length reduction in geometry adjustments. |
| `TIP_REDUCTION` | Float | m | `0.2` | Amount removed from each configured tip segment. |
| `BODY_DAMPING` | Vector{Float} | damping coefficients | `[0.0, 0.0, 20.0]` | Body-frame damping pattern for points 1:38. |
| `AOA_OFFSET_A` | Float | deg/% | `-0.6831` | Slope of photogrammetry-based wing AoA correction. |
| `AOA_OFFSET_B` | Float | deg | `28.74` | Intercept of photogrammetry-based wing AoA correction. |
| `BODY_DAMPING_DELTA` | `Tuple{Vector{Int}, Vector{Float}}` | (point indices, damping) | `([37, 38], [0.0, 20.0, 20.0])` | Body-frame damping override for the listed points. |
| `SAVE_FIGS` | Bool | - | `true` | Save PDF outputs for trajectory/panels/body-frame/twist plots. |
| `FIGURES_DIR` | String/Path | path | `joinpath(@__DIR__, "..", "output")` | Destination directory for exported figures. |
| `WIND_SOURCE_SPEED` | Symbol | - | `:ekf` | Source for wind magnitude (`:ekf` or `:lidar`). |
| `WIND_SOURCE_DIR` | Symbol | - | `:ekf` | Source for wind direction vector (`:ekf` or `:lidar`). |

## Detailed behavior by category

### 1) Data and maneuver selection

- `YEAR` chooses the HDF5 source file and section schedule.
- `BASE_SECTION` + `YEAR` builds `SECTION`, which maps to a UTC start/end pair.
- `DEPOWER_OFFSET_2019` and `DEPOWER_OFFSET_2025` are converted to normalized depower offset by dividing by 100 and passed into `V3GeomAdjustConfig.depower_offset`.

### 2) Command shaping and feedback

- The replay computes steering in layers. Using normalized steering ($-1$ to $1$):

$$
u_{csv} = \mathrm{clamp}(u_{raw}, -1, 1)
$$

$$
e_{\psi} = \mathrm{wrapTo\,\pi}(\psi_{data} - \psi_{sim})
$$

$$
u_{h} = \mathrm{PID}_{heading}(e_{\psi}; K=\texttt{HEADING\_KP},\; T_i=\texttt{HEADING\_TI})
$$

$$
e_{lat} = (\mathbf{p}_{sim}-\mathbf{p}_{data}) \cdot \hat{\mathbf{y}}_{body}
$$

$$
u_{lat} = \mathrm{PID}_{lateral}(e_{lat}; K=\texttt{LATERAL\_KP})
$$

$$
u_{cmd} = u_{csv}\,\texttt{STEERING\_MULTIPLIER} + u_h + u_{lat} + \frac{\texttt{STEERING\_OFFSET}}{100}
$$

- The command actually sent to the tape-length mapping is $u_{cmd}$ above.
- The model then converts $u_{cmd}$ to left/right steering tape lengths via `set_steering!`.

#### STEERING_OFFSET (detailed)

- `STEERING_OFFSET` is a constant bias in percent points, converted to normalized input by dividing by 100.
- Example: `STEERING_OFFSET = 1.5` means a constant $+0.015$ steering command is added at every step.
- It is added after heading and lateral corrections, so it acts like a fixed trim on top of feedback.
- Positive value biases the command in the positive steering direction; negative value biases it in the opposite direction.
- Because this offset is always added, it shifts both:
	- the command sent to geometry (`set_steering!`), and
	- the logged `set_steering` signal used in replay plots and steering-difference metrics.

#### Practical interpretation

- Use `STEERING_OFFSET` to remove persistent bias (for example, if feedback terms average near zero but tracking shows a consistent left/right mismatch).
- Tune it in small steps (for example, $0.2\%$ to $0.5\%$ increments) because it is applied continuously across the whole replay.

### 3) Geometry and tape adjustments

- `REDUCE_STEERING` and `STEERING_REDUCTION` affect steering tape base lengths in `set_steering!`.
- `REDUCE_TIP` and `TIP_REDUCTION` shorten configured tip segments in `apply_geom_adjustments!`.
- Depower offset from `DEPOWER_OFFSET_*` enters `set_depower!` via geometry config.

### 4) Damping and initialization

- `SETTLE = true` runs `settle_wing(...)` before replay.
- `BODY_DAMPING` applies to points 1:38.
- `BODY_DAMPING_DELTA` overrides the listed point indices.
- In settle config, these damping vectors are multiplied by 2.0 during initialization.

### 5) Aerodynamics and wind model choices

- `EXTRA_WING_DRAG_COEFF` is distributed across wing points (`distribute_wing_drag!`).
- `AOA_OFFSET_A` and `AOA_OFFSET_B` define a linear correction for wing AoA:

	`wing_aoa = kite_aoa + deg2rad(AOA_OFFSET_A * depower_pct + AOA_OFFSET_B)`

- `WIND_SOURCE_SPEED` and `WIND_SOURCE_DIR` can be mixed:
	- Speed from one source, direction from another.
	- If both are equal, the raw vector from that source is used.

### 6) Replay mode and outputs

- `DISTANCE_BASED_STEERING = false` means normal time-step-indexed replay.
- If set `true`, commands are looked up by cumulative traveled distance, and loop termination also uses distance completion.
- `LOAD_FROM_DISK = true` bypasses simulation and only loads logs.
- `SAVE_FIGS` and `FIGURES_DIR` control figure export.

## Important notes

- `HEADING_TI` in helper constructors is typically intended as a positive time constant or `false` to disable integral action. If you keep `HEADING_KP = 0.0`, the controller output remains zero regardless.
- `STEERING_OFFSET` is expressed in percent points in this script and converted using `/100` before being added.
- `generate_drag_adjusted_polars(1.0)` regenerates adjusted polars with unchanged Cd scaling (effectively a no-op rewrite with current factor).

## Suggested calibration workflow

For clean tuning, isolate effects in this order:

1. Open-loop replay: set `HEADING_KP = 0`, `LATERAL_KP = 0`, `STEERING_OFFSET = 0`.
2. Tune steering scaling and bias: `STEERING_MULTIPLIER`, then `STEERING_OFFSET`.
3. Enable heading correction: increase `HEADING_KP` gradually.
4. Add lateral correction: increase `LATERAL_KP` gradually.
5. Adjust geometry effects: `STEERING_REDUCTION`, `TIP_REDUCTION`, and depower offsets.
6. Finally tune damping and extra drag.

