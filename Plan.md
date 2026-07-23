# Implement high-level interface

A high-level interface using two functions, `init()` and `step!()`, shall be
implemented. It wraps the existing
settle/simulate/log boilerplate (see `examples/parking.jl`) so that an example
reduces to `init` + a loop of `step!` calls.

## Design decisions (resolved)

1. **`init`, not `init!`**: `init(v_wind_gnd, l_tether; ...)`
   builds and returns a ready `V3KITE` instance. No pre-constructed model is
   passed in, and there is no clash with the re-exported
   `SymbolicAWEModels.init!`.
2. **Dispatch on `V3KITE`**: `step!(s::V3KITE, ...)`. There is no `AKM` alias
   in V3Kite/KiteUtils, and the interface should not claim
   `AbstractKiteModel` for all kite models.
3. **Winch modes**: V3 has a single winch. `set_length !== nothing` selects
   position mode (a new controller, see below); otherwise `set_torque` is
   passed through as the winch set value. Exactly one mode is active per step.
4. **KCU actuator dynamics on**: `rel_depower`/`rel_steering` are routed
   through the `KCU` model (finite tape speed) instead of setting the
   geometry instantly:
   `set_depower_steering(kcu, ...)` → `on_timer(kcu, dt)` →
   `get_depower(kcu)`/`get_steering(kcu)` → `set_depower!`/`set_steering!`
   on the `sys_struct` with the stored `V3GeomAdjustConfig`.
5. **`init` runs `settle_wing`** (cached in `data/settled_*.bin`,
   `remake=false` by default), so simulations start from a settled
   equilibrium as in `examples/parking.jl`.

## New/extended fields of `V3KITE`

Mutable runtime state stored on the struct by `init`:

- `gc::V3GeomAdjustConfig` — geometry config used by the tape conversions
- `dt::Float64` — time step [s]
- `sys_state::Union{SysState, Nothing}` — current state, updated each step
- `logger::Union{Logger, Nothing}` — log of the run
- `steps::Int` — total number of simulation steps
- `t_prev::Float64`, `heading_prev::Float64` — for the `heading_rate`
  finite difference in `update_sys_state!`
- `last_step_time::Float64` — for realtime-factor progress prints
- position-controller state (integrator/limited setpoint, see below)

## Parameters of the `init()` function

Positional:

- `v_wind_gnd`   # ground wind speed at reference height (`set.h_ref`) [m/s]
- `l_tether`     # initial tether length [m]

Keyword:

- `elevation = nothing`        # initial elevation [deg]; fallback: settings
- `upwind_dir = -π/2`          # wind direction [rad]
- `depower_setpoint = 0.25`    # initial rel_depower in [0, 1] (NOT meters)
- `dt = nothing`               # fallback: `1/set.sample_freq`
- `sim_time = nothing`         # fallback: `set.sim_time`; sizes logger/steps
- `gc = V3GeomAdjustConfig()`  # geometry adjustments
- `remake = false`             # force re-settling (ignore cache)

Returns the `V3KITE` with logger, `sys_state`, `steps`, KCU and winch
controller initialized; winch un-braked.

## Parameters of the `step!()` function

```julia
step!(s::V3KITE; rel_depower = 0.0, rel_steering = 0.0,
      v_wind_gnd = nothing, upwind_dir = nothing,
      set_torque = nothing, set_length = nothing,
      speed_limit = Inf, acceleration_limit = Inf, prn = false)
```

- `rel_depower`  # 0.0 .. 1.0
- `rel_steering` # -1.0 .. 1.0
- `v_wind_gnd`, `upwind_dir` # if not `nothing`, update `set.wind_vec`
  before the step (verify SymbolicAWEModels reads it live; otherwise
  document as init-only and drop)
- `set_torque`   # winch torque [Nm], torque mode
- `set_length`   # tether length setpoint [m], position mode (overrides
                 # `set_torque`)
- `speed_limit`, `acceleration_limit` # saturation of the position
  controller [m/s], [m/s²]

Advances the simulation by `s.dt`, updates `s.sys_state` (including
`heading_rate`), logs it, and prints progress every 100 steps.

## Winch position controller (new code)

This is the main new component: nothing in V3Kite currently converts a
length setpoint into a winch torque.

- Structure: outer P on length error → speed setpoint, clamped to
  `±speed_limit` and rate-limited by `acceleration_limit`; inner PI on
  speed error → force, converted with `force_to_torque` (gravity/steady
  compensation from the measured winch force), output = winch set torque.
- Gains via `create_winch_pid` conventions in `sim_helpers.jl`; tune on
  `simple_parking.jl` (constant-length parking). Per CLAUDE.md: change
  existing PID parameters by not more than 10% per iteration.
- State (PID objects, previous speed setpoint) lives on the `V3KITE`.

## Example

Implement `examples/simple_parking.jl`: user parameters (SIM_TIME, V_WIND, TETHER_LENGTH,
DEPOWER_SETPOINT), `init`, a loop calling
`step!(s; rel_depower, set_length = l0)`, save the log ("tmp_run").
The existing `examples/parking.jl` stays as the manual (braked-winch)
reference and is not modified.

## Verification

- `simple_parking.jl` parks at constant tether length and elevation
  comparable to `parking.jl` results.
- For verification, run `include("examples/simple_parking.jl")` (do not run
  automatically; expected runtime > 90 s).
