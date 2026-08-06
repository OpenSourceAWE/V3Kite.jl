# Changelog

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
