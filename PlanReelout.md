# Port `reel_out_v3.jl` to the `init`/`step!` interface

## Goal

`examples/reel_out_v3.jl` already exists — it is a working port of KiteModels'
`reel_out_4p_torque_control.jl`, but it drives the model through the low-level
`init!(sam)` / `sim_step!(sam; set_values=[τ])` path, hand-building `Settings`,
`KCU`, `SystemStructure`, `SymbolicAWEModel` and `V3KITE` itself
(`examples/reel_out_v3.jl:40-79`, `:106`, `:133`).

Convert it to the high-level `V3Kite.init` / `step!` interface (torque mode),
so it matches `examples/simple_parking.jl` in structure, while preserving the
maneuver it simulates today.

**Do not copy or rename anything.** The copy/rename step of the original plan is
already done; re-copying `reel_out_4p_torque_control.jl` would discard the V3
work in the current file (VSM setup, `struc_geometry.yaml` loading, tether
position reconstruction for `plot2d`).

## Preserve

- Torque-controlled reel-out: hold torque, then add `dforce = +4.5` N·m after
  `t > 15 s` to start reeling out.
- `dt = 0.05/3` s, `STEPS = 600*3` (30 s of sim time).
- `V_WIND = 9.51` m/s, `TETHER_LENGTH = 150.0` m.
- Live 2D side/front view via `plot2d` every 5th step, and the final `plotx`
  summary of reel-out speed / force / elevation / heading / wind at kite.
- The recorded traces (`v_time`, `v_speed`, `v_force`, `v_elevation`,
  `v_heading`, `v_wind_speed`).

## Decisions (resolved — override if you disagree)

### 1. Parameter routing → new system-YAML pair

The script currently overrides eight settings by hand
(`examples/reel_out_v3.jl:44-54`), and most of them differ materially from
`data/settings.yaml`:

| setting | settings.yaml | script | consumed |
| --- | --- | --- | --- |
| `mass` | 11.0 | 6.2 | at structure build |
| `d_tether` | 13.5 | 4.0 | at structure build |
| `profile_law` | 0 | 3 | wind model |
| `alpha_zero` | — | 8.8 | aero |
| `drum_radius` | 0.110 | 0.1615 | winch |
| `gear_ratio` | 1.0 | 6.2 | winch |
| `f_coulomb` | 122.0 | 122.0 | winch |
| `c_vf` | 30.6 | 30.6 | winch |

`init` loads `Settings` internally and exposes only the `system_yaml` kwarg
(`src/interface.jl:434-451`), so post-hoc `s.set.*` mutation is not a general
option: `mass` and `d_tether` are consumed when the `SystemStructure` is built
(wing mass, segment diameter), i.e. **before** settling, and the winch fields
are copied into `sys.winches[1]` at construction and thereafter read live from
that struct — not from `set` — by the registered symbolic accessors. Writing
`s.set.drum_radius` after `init` would silently do nothing.

→ Add `data/system_reelout.yaml` (pointing at a new `data/settings_reelout.yaml`
plus the existing `wc_settings.yaml`) carrying all eight values, and call
`init(V_WIND, TETHER_LENGTH; system_yaml="system_reelout.yaml", ...)`. This also
keeps the settled-geometry cache correctly keyed, since the cache key encodes
the system-YAML name.

### 2. Plotting → keep the live animation

Keep the `plot2d` animation gated behind `PLOT` (default `true`) — README:47
advertises this example as "Single reel-out maneuver, 2D Makie plots", and it is
the example's identity. Additionally use the logger that `step!` fills for free
(`save_log(s.logger, "tmp_reel_out")`), matching `simple_parking.jl`.

Note the animation is the slow path; with settling in front of it the first run
will be noticeably slower than today.

### 3. Torque law → `force_to_torque`, and the file's open question is answered

`examples/reel_out_v3.jl:98-99` carries the comment *"This should be the same as
the above, but it isn't. Why?"* about `-r/n * force` vs
`force_to_torque(-force, sys)`.

Answer: `force_to_torque(force, sys) = -r/G * force + winch.friction`
(`src/sim_helpers.jl:56-60`). The two differ by exactly `winch.friction`, which
the manual expression omits. `step!`'s hold mode already uses
`force_to_torque(winch_force(s), s.sys)` (`src/interface.jl:565`).

→ Use `set_torque = force_to_torque(winch_force(s), s.sys) + dforce` and delete
the stale comment. Expect a small quantitative shift in the force/speed traces
versus today's run — the friction term is now included, which is the physically
correct behaviour.

## Implementation steps

1. Add `data/settings_reelout.yaml` (copy of `data/settings.yaml` with the eight
   overrides from the table) and `data/system_reelout.yaml` with
   `sim_settings: "settings_reelout.yaml"` and
   `wc_settings: "wc_settings.yaml"`.
2. Rewrite `examples/reel_out_v3.jl`:
   - Drop lines 40-79 and 132-134 (manual `Settings`/`KCU`/`vsm_set`/
     `load_sys_struct_from_yaml`/`SymbolicAWEModel`/`V3KITE`/`init!`/brake
     release) — `init` does all of it, including leaving the winch un-braked
     (`src/interface.jl:478`).
   - `s = init(V_WIND, TETHER_LENGTH; system_yaml = "system_reelout.yaml",
     depower_setpoint = DEPOWER_SETPOINT, dt = 0.05/3, sim_time = STEPS*dt)`.
     Passing `dt` explicitly is required — `init` otherwise defaults to
     `1/set.sample_freq` = 0.05 s.
   - Loop `for i in 1:s.steps`, calling
     `step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = ...)`.
     `rel_depower` **must** be passed every step: it defaults to `0.0`, which
     would drive the KCU away from the settled depower setting and introduce a
     geometry transient.
   - Replace `integrator.t` with `s.sys_state.time` — `step!` returns `nothing`
     and there is no integrator handle.
   - Wrap the loop in `try`/`catch` like `simple_parking.jl:45-53`, so a failed
     step still produces plots and a saved log.
   - Keep `lift_drag`, `winch_force`, `reel_out_speed`, `calc_elevation`,
     `calc_heading`, `v_wind_kite`, `states` calls unchanged — all already take
     a `V3KITE`.
3. Keep the SPDX header and the existing `Jelle Poland, Bart van de Lint`
   copyright line; add nothing else.

## Expected behavioural differences

These are consequences of the port, not regressions:

- `init` settles the wing before the run (disk-cached to `data/settled_*.bin`);
  the old script started from `init!(sam)`. The initial elevation and the first
  seconds of every trace will differ.
- First run pays the settling cost; subsequent runs hit the cache.
- Torque now includes the winch friction term (decision 3).
- `rel_depower` is now routed through the KCU actuator model (finite tape
  speed), which the old script bypassed entirely.

## Acceptance criteria

- `include("examples/reel_out_v3.jl")` runs the full 30 s without error.
- Reel-out speed is ≈0 for the first 15 s, then rises to a steady positive value
  once `dforce` is applied; tether force stays positive throughout.
- Elevation and heading stay bounded (no divergence / kite crash).
- `plotx` summary and the 2D animation both render.

## Out of scope

- `README.md:47` and `examples/menu.jl:22` already list the example and their
  descriptions stay accurate — no changes needed.
- No new tests; this is an example script.
- `examples/simple_parking.jl` and `examples/parking.jl` are not touched.
