# Changelog

## V3Kite (unreleased)

### Changed
- BREAKING: `step!` is torque-only. It takes `set_torque` [N·m], or applies the
  measured holding torque when omitted, and no longer accepts `set_length`,
  `speed_limit` or `acceleration_limit`. A run that held a length now builds its own
  torque; `examples/winch_adapter.jl` shows how.
- BREAKING: `init` no longer takes `wc`, and its `warmup_wfc` is now `warmup_torque`,
  a `(s, l_hold) -> torque` callback (`warmup!`'s `wfc` is likewise `winch_torque`).
  Omitting either holds the measured tether force where it used to hold the length.
- BREAKING: `WC_Settings`, `WinchPosController`, `WinchForceController` and
  `winch_force_torque!` are no longer exported, and `V3KITE` has no `winch_ctrl`
  field. The controllers live in
  [WinchControllers.jl](https://github.com/OpenSourceAWE/WinchControllers.jl) v0.6.0,
  so V3Kite no longer precompiles against them.
- The winch length loop the examples and tests run defaults `acceleration_limit` to
  the `winch: max_acc:` setting (4.0 m/s² for the V3) where `step!` defaulted it to
  `Inf`, so the commanded speed no longer slews faster than the real drum.
  `speed_limit` still defaults to `Inf`: it is one number where the settings give the
  signed pair `v_ro_max`/`v_ro_min`.
- `test/` no longer depends on WinchControllers.jl. `test/winch_hold_stub.jl` is a
  small length-holding torque controller (`LengthHoldController`/
  `length_hold_controller`/`hold_torque!`) built only from V3Kite's own public
  interface (`force_to_torque`, `unstretched_length`, `reel_out_speed`,
  `winch_force`), replacing `examples/winch_adapter.jl`'s WinchControllers-based
  adapter for `test-interface.jl`, `test-turbulence-injection.jl` and
  `test_parking_ripple.jl`. Its names are deliberately distinct from the
  adapter's (`WinchPosController`/`winch_pos_controller`/`winch_torque!`) so
  the two can never shadow each other if both get loaded into one session.
  `examples/` is unaffected and still uses WinchControllers.jl.

### Added
- `drum_params(s)`: the first winch's drum radius, gear ratio and current friction
  torque, the plant scalars WinchControllers.jl's torque controllers take.
- `examples/winch_adapter.jl`'s `winch_torque!` gained a `v_ff` keyword [m/s], the
  speed feed-forward of the position-mode winch (`v_sp = v_ff + winch_pos_kp *
  (set_length - l)`). A caller that builds `set_length` by integrating a speed
  setpoint should pass that speed, or the outer P loop rediscovers it from a
  length error at a lag of `1/winch_pos_kp` (2 s at the default `winch_pos_kp =
  0.5`) plus a standing error of `v_ro/winch_pos_kp`. The default `0.0`
  reproduces the previous behaviour exactly.

## V3Kite v1.1.1 13-08-2026

### Changed
- The `min_damping` default of `init` is now computed from `body_damping` as
  `0.8 .* body_damping` instead of the fixed `[0.0, 0.0, 20.0]`. With the default
  `body_damping = [0.0, 0.0, 40.0]` the floor becomes `[0.0, 0.0, 32.0]`, which is what the
  examples already pass explicitly and what the pre-2026-08-08 ramp ended at; raising
  `body_damping` now raises the flown damping with it, and an in-plane `body_damping` keeps
  its in-plane terms in flight. Callers that pass `min_damping` explicitly are unaffected;
  callers relying on the old default get a different settling cache key and re-settle.
- The `@compile_workload` in `src/precompile.jl` now mirrors `examples/simple_parking.jl`
  — `system_reelout.yaml` at `aero_mode = ContinuousAero()`, `dt = 0.05/3` and
  `damping_per_stiffness = 0.001` — instead of `system_cabauw.yaml` at the `AeroDirect()`
  default. The aero mode picks which model binary the cached SciML specializations are
  compiled against, so the old workload warmed a path none of the examples fly. Every
  keyword is passed explicitly now, including those equal to the current default, so the
  workload cannot drift from the example it mirrors. The first precompilation after this
  change has no cache for the new combination and rebuilds the model and settled geometry
  into `data/`, which takes minutes; `V3KITE_SKIP_PRECOMPILE_WORKLOAD=1` skips the workload
  instead.
- The examples write their logs to `examples/../output/` (created if missing) rather than
  the data path, and the matching `*_plots.jl` scripts read them from there:
  `simple_parking`, `simple_auto_parking`, `simple_sinus` and `steering_test_v3`.

### Added
- `damping_per_stiffness` [s], a new `init` keyword and `V3SettleConfig` field, sets the
  structural damping of the tether and bridle segments as a ratio of their stiffness
  (`unit_damping = ratio * unit_stiffness`), overriding the `damping_per_stiffness` of the
  material in `data/struc_geometry.yaml`; the wing frame keeps the damping given there.
  It is applied from the START of settling rather than to the settled model, so the run is
  damped throughout, with one floor: settling diverges below the new
  `MIN_SETTLE_DAMPING_PER_STIFFNESS` (0.0015), so a lower ratio settles at the floor and is
  then set on the settled structure, which has no transient left to destabilize. The
  floored value is what enters the settling cache key, so every flown ratio below the floor
  shares one `data/settled_*.bin`. Settling tolerates 0.0015 … 0.0028 for the V3; the
  examples fly 0.001. The default `nothing` leaves the segments as loaded — bridles at the
  material value, main tether undamped — and leaves existing cache file names unchanged. The
  underlying helpers `tether_bridle_segments` and `set_damping_per_stiffness!` (both in
  `src/model_setup.jl`, taking the `SystemStructure`) are exported too, and the examples
  expose the ratio as `DAMPING_PER_STIFFNESS`.

### Fixed
- `span_mean_aoa(sys)` returned `NaN` for every aero mode. It probed
  `hasproperty(wing, :vsm_solver)`, but the VSM engine hangs off `wing.aero` and reaches
  the wing through a `getproperty` forward that `propertynames` does not advertise, so the
  probe was always false. It now resolves the engine with
  `SymbolicAWEModels.vsm_engine(wing.aero)` and returns `NaN` only when the wing genuinely
  carries none. Covered by a new `span_mean_aoa` testset in `test/test-interface.jl`.

## V3Kite v1.1.0 10-08-2026

### Changed
- `AtmosphericModels` compat raised to `0.3.8`, which applies `use_turbulence` when the
  wind field is read instead of baking it into the stored field. One `windfield_*.npz`
  per ground wind speed now serves every turbulence level, so `set_default_turbulence` no
  longer warns that a value with two decimals shares a file, and switching level costs no
  regeneration. Existing `windfield_<grid>_1.0_<speed>.npz` files stay valid under the
  name with `_1.0` dropped; files generated at any other level are pre-scaled and must be
  deleted. 0.3.8 also moves new wind fields out of `data/` into a shared scratchspace
  (`AtmosphericModels.windfield_path()`); the ones already in `data/` are still found
  there, and can be moved into the scratchspace to be shared with the other repos.
- `SymbolicAWEModels` compat raised from `0.11` to `0.13` and `VortexStepMethod` from
  `3.3.4` to `4.0.0`, which is what brings the selectable aerodynamics model below.
- Settling is coarser and shorter: `V3SettleConfig` defaults move to
  `num_steps = 1600`, `decay_steps = 400` and a `min_damping` floor of
  `[0.0, 0.0, 20.0]` (was `8000`/`2000`/zero), and `init` settles at `dt = 0.05` in
  40 steps of 1 substep instead of `dt = 0.001` in 400 steps of 5. `body_damping` is
  therefore the value settling *starts* from — it decays linearly to `min_damping`
  over `decay_steps`, and that floor is what the returned model runs with.

### Added
- Selectable aerodynamics: `AeroDirect` (the default, VSM load held frozen over a
  step) and `ContinuousAero` (the load integrated) are exported and reach the model
  through `init(...; aero_mode)` and `V3SimConfig.aero_mode`. A `nothing` on the
  config resolves by wing type — `AeroDirect()` for `PARTICLE_DYNAMICS`, the
  upstream default for `RIGID_DYNAMICS`.
- The aero mode enters the settled-geometry cache key (`_aero<tag>`, with the
  default `AeroDirect()` adding nothing, so files written before this stay in use),
  and a cached geometry whose wings disagree with the requested `aero_mode` is
  rejected and re-derived rather than silently overriding it. A settled
  `SystemStructure` carries its wing's aero object, which is what made a cache hit
  able to win over the request.
- `record_2d_trajectory` and `record_2d_panels` (V3KiteMakieExt), animating the 2D
  trajectory and the 2D time-series panels to a video/GIF.
- `settled_struct_path`, `load_settled_struct`, `set_body_frame_damping!` and
  `tether_point_idxs` are now exported.
- `examples/simple_parking.jl` logs and plots the tether compression force, and
  drives the run through a `REL_STEERING` constant.

### Removed
- The `WING` export, whose wing type was dropped upstream in `SymbolicAWEModels`.

## V3Kite v1.0.3 09-08-2026

### Added
- `init(...; use_turbulence)`, which overrides the `default_turbulence` of
  `data/gui.yaml`. A package driving V3Kite can then keep the preference in its
  own writable `data/` — `data_path` cannot serve that role, since it must be
  the directory holding the model geometry, which is read-only for a
  Pkg-installed V3Kite. `nothing` (the default) keeps reading `gui.yaml`, and
  the `"default"` keyword leaves the settings YAML in charge, exactly as the
  file's value does.
- Turbulent wind at the kite (Mann model, via
  `AtmosphericModels.calc_turbulent_wind`; `AtmosphericModels` compat bumped
  to `0.3.6`). `apply_turbulence!`, called from `step!`, samples the turbulent
  wind at the wing, divides out the height profile factor the DAE will apply
  again, and writes the result into `set.wind_vec` for the duration of the
  solver call; a `finally` restores the commanded mean, which is kept on the
  new `V3KITE.wind_vec_mean` field. The value is held constant over the step so
  the discontinuous nearest-grid-point wind lookup never enters the implicit
  solve, and the restore happens after `update_sys_state!`, so the logged
  `v_wind_gnd` is the instantaneous wind the kite saw.

  **Approximation:** `set.wind_vec` is a single ground vector, so the gust
  sampled at the kite acts *coherently* on every tether and bridle point as
  well — with `profile_law: 0` the whole tether feels the kite's gust. That
  overstates fluctuating tether drag. It is accepted deliberately: `wind_vec`
  is the only per-step wind hook that reaches the whole system, and the wing's
  `wind_disturb` parameter (the injection point of the first implementation)
  is dead on a `PARTICLE_DYNAMICS` wing, whose aero force is a per-point VSM
  solve that never reads it.
- `v_wind_gnds`/`rel_turbs` extended to `[3.483, 5.324, 8.163, 9.51]` /
  `[0.342, 0.465, 0.583, 0.626]` in `settings.yaml`, `settings_reelout.yaml`
  and `settings_cabauw.yaml`, closing the gap where `load_windfield` snapped
  an unlisted ground wind speed to a neighbour's turbulence intensity. The
  `0.626` point extends the log fit `rel_turb = 0.342 + 0.283*(ln v - 1.248)`
  fitted to the three Cabauw-calibrated points (turbulence intensity I₉₉ at
  99 m):

  | `v_wind_gnds` | `rel_turbs` | I₉₉ |
  | ---: | ---: | ---: |
  | 3.483 | 0.342 | 9.7 % |
  | 5.324 | 0.465 | 10.4 % |
  | 8.163 | 0.583 | 10.7 % |
  | 9.51 | 0.626 | 10.9 % predicted, 10.8 % measured |

  Wind fields pre-generated for 8.163 and 9.51 m/s (1.24 GB each); `grid:
  [100, 4050, 500, 70]` made explicit in all three settings files.
- `get_default_turbulence`/`set_default_turbulence`
  (`src/turbulence_config.jl`), persisting a per-checkout turbulence
  preference to `data/gui.yaml` (gitignored, created from the tracked
  `gui.yaml.default`). Accepts `"default"` to defer to
  `environment.use_turbulence` in the active settings YAML instead of
  shadowing it. The no-argument dialog is an interactive `RadioMenu`
  (`REPL.TerminalMenus`); a `set_default_turbulence` entry was added to
  `examples/menu2.jl`.

### Fixed
- `init` now re-points the shared `AtmosphericModel` at the live `Settings`
  and reloads the wind field, so turbulence (and `calc_wind_factor`/
  `calc_rho`) no longer silently used settings stale from a cached
  `settled_*.bin`.

### Notes
- Wind-field filenames keep only one decimal of `use_turbulence`
  (`calc_full_name`, `%.1f`), so e.g. `0.30` and `0.34` would silently share
  a field with up to ~15 % sigma error; `set_default_turbulence` now warns
  when a value doesn't round-trip through one decimal.

## V3Kite v1.0.2 04-08-2026

### Changed
- Require `KiteUtils` 0.11.11 (previously unbounded, resolving to 0.11.9):
  0.11.10 adds a `flap_angle` field to `SysState`/`Logger`, and 0.11.11 fixes
  the parsing of a bare `log_file` name in the settings, which threw a
  `BoundsError` — the form an external controller package uses when it keeps
  its logs in its own directory.

## V3Kite v1.0.1 03-08-2026

### Added
- High-level `init(v_wind_gnd, l_tether; ...)` / `step!` simulation interface
  (#37), replacing direct model-object manipulation: `init` settles the wing
  and returns a state carrying a `SysState`, `step!` advances it one timestep.
- Position- and force-mode winch control: `WinchPosController`/
  `WinchForceController`, `winch_force_torque!`, `wc_settings.yaml` /
  `WC_Settings` (`src/wc_settings.jl`), and a `winch_ff_scale` knob on the
  position controller's feed-forward.
- `warmup!`, plus `init`'s `warmup_time`/`warmup_wfc` keywords, to relax the
  settled state into an equilibrium of the run's own model and discard the
  transient before `t = 0`.
- `total_drag`, `span_mean_aoa` (span-averaged AoA for the whole wing),
  `create_heading_pid`/`create_winch_pid` (cascaded steering control, with an
  `N` derivative-filter argument), and a `vsm_interval` keyword on `step!` to
  hold the VSM aero load frozen between updates.
- Turn-rate-law identification (`src/turn_rate_id.jl`):
  `identify_turn_rate_law`, `fit_c1_c2`, `estimate_delay`, `shift_delay`,
  `turn_rate_gain`, `est_steering`, `format_turn_rate_report`, plus
  `test/test_turn_rate_id.jl`.
- AoA-ripple analysis (`src/ripple_metrics.jl`): `RippleSettings`,
  `ripple_metrics`, `aoa_ripple`, `format_ripple_report`, plus
  `data/ripple_settings.yaml` and `test/test_ripple_metrics.jl`.
- Examples built on the new interface: `parking.jl`, `simple_parking.jl` (+
  `_plots.jl`), `simple_sinus.jl` (+ `_plots.jl`), `simple_auto_parking.jl` (+
  `_plots.jl`), `steering_test_v3.jl` (+ `_plots.jl`), a torque-controlled
  `reel_out_v3.jl` (+ `_plots.jl`), `menu2.jl`, and Matlab REST clients
  `simple_parking_client.m` / `simple_sinus_client.m` backed by a new
  `examples/rest_server.jl` and `bin/run_server`.
- New system/settings projects for the above: `data/system_cabauw.yaml`,
  `data/system_reelout.yaml`, `data/settings_reelout.yaml`,
  `data/wc_settings.yaml`.
- Optional precompilation workload (`src/precompile.jl`, previously
  disabled): brings TTFX down from ~28 s to ~2 s at the cost of ~90 s longer
  precompile time; meant to stay off during package development and on during
  controller development/tuning.
- `tic()`/`toc()` timing helpers; each log's Arrow metadata now carries the
  simulation's timestamp, shown in the corresponding plot's figure name.
- `test/test-interface.jl` covering the new `init`/`step!` interface.

### Changed
- Default `body_damping` retuned repeatedly while identifying the turn-rate
  law, settling on `[0, 0, 40]` (#38) — cuts AoA ripple by roughly 10x and
  about doubles simulation speed versus the previously undamped model.
- `force_to_torque` gained an `ff_scale` keyword that scales only the load
  term of the winch feed-forward: `winch_position_torque!` previously scaled
  its whole result, which also scaled the friction-compensation term and
  leaked a direction- and speed-dependent force bias into the held winch
  force.
- Cache/data-path handling reworked so V3Kite behaves correctly as an
  installed dependency of an external controller package, not just as a dev
  checkout (#41, #43):
  - `init`/`step!` no longer move the caller's global KiteUtils data path —
    previously `init` called `set_data_path` and left it pointed at the
    model's own directory for the rest of the session. Every file `init`
    needs is now resolved against an explicit `data_path` argument, and an
    absolute `system_yaml` is honoured as given, ignoring `data_path`.
  - Settled geometry, the settling log, and the compiled model binary are
    now written to `cache_path` when one is given, instead of always landing
    in `data_path`. `default_cache_path` picks a `Scratch.jl` space keyed by
    the package UUID for an installed copy (so `Pkg.gc` can no longer
    silently discard an ~80 s model build once no environment references
    that version) and falls back to `data_path` for a dev checkout. The
    depot-installed check now requires a path-separator boundary, so a
    sibling directory such as `<depot>/packages_old/...` is no longer matched
    by prefix.
  - The settled-geometry cache key now includes the settling elevation
    (previously carried only through `init_row`'s position, so two runs at
    different elevations could silently share one `settled_*.bin`);
    `damping_tag` generalized to `num_tag` to reflect that it now tags more
    than damping.
  - `wc_settings` is resolved relative to `dirname(system_path)`, not
    `data_path`.
- `step!` gates the logged L/D ratio on a physical drag floor
  (`drag_floor`): `compute_drag`'s signed projection onto `v_a` passes
  through zero whenever the wing unloads, previously spiking the logged
  ratio; the affected samples are now `NaN` (`var_15`/`var_16`), and both
  Matlab clients map the resulting JSON `null` back to `NaN` instead of
  letting `[]` delete the sample.
- `init` hands the drum a holding torque when releasing the brake, removing
  a torque step at `t = 0`.
- Renamed private functions to drop their leading underscore.
- Updated default manifests, `bin/run_julia`, and `.JETLSConfig.toml.default`.

### Fixed
- `save_and_load_log` reading the log back from KiteUtils' default path
  after saving it somewhere else.
- Heading calculation bug in the reel-out example path.
