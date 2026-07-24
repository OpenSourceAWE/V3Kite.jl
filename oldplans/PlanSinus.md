# Add example simple_sinus.jl, using the high level interface

Create `examples/simple_sinus.jl`: sinusoidal heading tracking of the V3 kite
built on the high-level `init()`/`step!()` interface, in the style of
`examples/simple_parking.jl`, plus a companion `examples/simple_sinus_plots.jl`.

## Framing

`simple_sinus.jl` is to the heading-tracking loop of `examples/v3kite.jl` what
`simple_parking.jl` is to `parking.jl`: same maneuver, rebuilt on
`init`/`step!`. The controller comes from `v3kite.jl`: a plain
`create_heading_pid` (`src/sim_helpers.jl`) driving the single steering
channel.

## Design decisions

1. **Controller**: `create_heading_pid` with the v3kite.jl gains as starting
   point (`K = 1.0`, `Ti = false`, `Td = 0.0`, `umin/umax = ∓0.15`). The PID
   output is used directly as the `rel_steering` keyword of `step!`.
   - Units: `set_steering!(sys, steering, gc)` takes a *fractional* steering
     value (it converts ×100 to percentage internally), so the v3kite.jl PID
     output and `step!`'s `rel_steering` are the same scale. The
     "tape delta in m" comment in v3kite.jl is misleading — the conversion to
     tape meters happens inside `set_steering!`
     (`steering_percentage_to_lengths`, `src/calibration.jl`).
   - Difference vs v3kite.jl: `step!` routes `rel_steering` through the KCU
     (`set_depower_steering` → `on_timer` → `get_steering`), which adds
     rate-limited actuator dynamics (`set.v_steering`) and depower-dependent
     scaling that the direct `set_steering!` path in v3kite.jl does not have.
     Expect a mild gain retune; per CLAUDE.md change gains by ≤10% per
     iteration.
2. **Winch**: position mode, `set_length = l0` with
   `l0 = s.sys_state.l_tether[1]` after `init`, exactly as in
   simple_parking.jl (no net reel-out).
3. **Setpoint**: `target = deg2rad(MAX_HEADING) * sin(2π/HEADING_PERIOD * t)`
   with `t = s.sys_state.time + s.dt`, engaged from the first step (v3kite.jl
   does the same; no settling ramp needed since `init` settles the wing).
4. **Log name**: save to `"tmp_sinus"` so the parking log `"tmp_run"` is not
   clobbered.
5. **Plotting**: separate `simple_sinus_plots.jl` following the local
   convention (simple_parking_plots.jl). The plots script recomputes the
   setpoint from `sl.time` and its own `MAX_HEADING`/`HEADING_PERIOD`
   constants (kept in sync with the sim script, like `DEPOWER_SETPOINT` in
   simple_parking_plots.jl) — no spare var needed. `var_15`/`var_16` (L/D
   wing / L/D eff) are already filled by `step!`.

## Files

### examples/simple_sinus.jl

- Docstring: what it does, relation to v3kite.jl and simple_parking.jl,
  verification hint.
- USER PARAMETERS block:
  - `PROJECT = "system_cabauw.yaml"` — system project to use (see
    `data/system_*.yaml`), passed to `init` as `system_yaml = PROJECT`,
  - `V_WIND = 10.0`,
    `TETHER_LENGTH = 150.0`, `DEPOWER_SETPOINT = 0.25` (the validated
    simple_parking.jl operating point),
  - `SIM_TIME = 60.0`, `MAX_HEADING = 40.0` deg, `HEADING_PERIOD = 30.0` s
    (from v3kite.jl; reduce MAX_HEADING if tracking at the cabauw operating
    point is poor),
  - PID gains `HEADING_P/I/D` and `MAX_STEERING = 0.15`.
- `init(...)`, build PID, loop over `s.steps` in try/catch (pattern of
  simple_parking.jl): compute target, `u = pid(target, s.sys_state.heading,
  0.0)`, `step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering = u,
  set_length = l0)`.
- After the loop: `save_log(s.logger, "tmp_sinus")`, then print the heading
  tracking RMS error over the settled part (skip the first
  `HEADING_PERIOD`).

### examples/simple_sinus_plots.jl

- Clone of simple_parking_plots.jl adapted to: elevation, azimuth, heading
  vs recomputed setpoint, `set_steering` (logged by `step!` as
  `sys_state.set_steering`), AoA, L/D (`var_15`/`var_16`). Load
  `"tmp_sinus"`.
- Also gets a `PROJECT = "system_cabauw.yaml"` line (kept in sync with
  simple_sinus.jl, like the shared `MAX_HEADING`/`HEADING_PERIOD`/
  `DEPOWER_SETPOINT` constants), so any settings-dependent quantities are
  read from the same system file.

### examples/menu.jl (optional)

- menu.jl currently lists neither simple_parking.jl nor its plots script.
  Either add entries for all four simple_* scripts in one go, or skip the
  menu entirely for consistency with simple_parking.

## Verification

A 60 s run exceeds 90 s wall time on first run; per CLAUDE.md do not run it
as a verification test — finish with the message: For verification, run
`include("examples/simple_sinus.jl")`, then
`include("examples/simple_sinus_plots.jl")`, and check the printed heading
RMS error. Gains may be tuned afterwards (≤10% per iteration).
