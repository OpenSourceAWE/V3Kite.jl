# Changelog

## Unreleased

### Added
- `test/test_settling.jl` settles the wing of `examples/simple_sinus.jl` from
  scratch, forcing both a model rebuild and a re-settle, so CI covers the path a
  machine with no cache takes and a settling divergence fails the suite rather
  than only the examples.

### Fixed
- `remake_model=true` rebuilds the model again. Settling built its own model from
  the project's `remake_model:` flag rather than the argument, and then reported
  the rebuild as done, so the argument reached neither the settling build nor the
  one after it: an `init(...; remake_model=true)` silently reused a `model_*.bin`
  written under an older dependency.

## V3Kite v1.3.0 02-09-2026

### Added
- `beam_joint_damping_scale` in the kite settings scales every Timoshenko joint's
  Rayleigh β without regenerating the geometry. The emitted β is ζ = 1 at each
  joint's transverse mode, which the chord modes outgrow under flight load until
  the implicit solver stalls; the beam projects fly at `45`. The scale is
  numerical, not material — ζ = 45 is far past anything an inflated tube has — so
  it stands in for damping the model is missing, such as the thin-airfoil
  flow-curvature moment the panels omit because a section's `va` is the mean of
  its edge velocities and so carries no pitch rate.
- `data/settle_settings_beam.yaml`, a settling schedule of the beam's own, so the
  ramps that quiet a beam do not change how the lattice settles. The shared
  default left every `beam_*_start_damping` at zero, which damps nothing on a beam
  wing — its wing nodes are `BODY_STATIC` points that the point damping cannot
  reach — so the run began by flying a ringing structure.
- `beam_angular_damping` in the kite settings and `beam_angular_start_damping`
  in the settle settings: per-axis spin damping on every leading- and
  trailing-edge body of a beam wing, `dω/dt -= c .* ω` [1/s] about the body's own
  axes. `[0, 20, 0]` resists rotation about `y`, the chord-flapping axis. It damps
  absolute spin, so unlike the joints' Rayleigh `damping` it brakes rigid wing
  rotation too. Zero by default, and it skips the wing bodies, whose angular
  damping carries the `y_damping` their constructor seeded.
- `beam_body_damping` / `beam_world_damping` in the kite settings and
  `beam_body_start_damping` / `beam_world_start_damping` in the settle settings:
  the point damping the two already had, now reaching the rigid bodies. A beam
  wing keeps its mass in bodies, and `body_sim_damping` never touched them —
  its wing nodes are `BODY_STATIC` points riding those bodies. `beam_body_damping`
  defaults to `null`, which follows `body_sim_damping`: both resolve on the wing's
  own axes, so the bodies damp what the points would have.
- `relax_bridle!` damps the bodies over the ramp as well as the points
  (`V3RelaxConfig.body_damping`), and reports progress every
  `report_every` steps. Without it a beam geometry never reached full stiffness:
  nothing damped the rigid-body motion of the 22 tube bodies, so the ramp
  diverged at `scale = 1.8e-4`. It now reaches `scale = 1.0` in 291 steps, and
  the beam project flies a full 60 s heading maneuver on the state it writes.
- `wing_twist_dist`, `differential_twist` and `aero_moment_z` for diagnosing
  steering: the per-station twist along the span, the antisymmetric part of it
  (the `+y` half mean less the `-y` half mean) and the VSM yaw moment to read it
  against. `wing_station_chords` finds each station's leading and trailing edge
  as the twist surface's extremes in chordwise CAD x, the rule SymbolicAWEModels
  itself maps aero sections onto.
- `V3Kite.SurfplanAdapter`, which turns a SurfplanAdapter export of the V3 into a
  Timoshenko-beam `struc_geometry.yaml`: the leading edge and the struts become
  `Body` chains joined by `TimoshenkoJoint`s, and every bridle branch end rides
  the nearest node body as a `BODY_STATIC` point. The bridle itself comes from
  the measured 2025 line system in `data/bridle_geometry_full_fem.yaml`, not from
  the export, so the KCU, the pulleys, the M-line and the three tapes are the
  flown ones. `V3BeamTopology` carries everything the emission depends on,
  `surfplan_to_struc` writes the file, `apply_comer_bending!` swaps the constant
  Breukels bending YAML can hold for the curvature-softening Comer-Levy law, and
  `apply_bridle_material!` sets the compression and damping knobs on a loaded
  structure so they can be swept without re-emitting. The emitted wing is a
  `PARTICLE_DYNAMICS` wing following `BODY_STATIC` points, which makes
  `data/struc_geometry_beam.yaml` a drop-in replacement for the particle
  `struc_geometry.yaml`.
- `relax_bridle!` and `V3RelaxConfig`. The measured bridle lengths and the
  measured node coordinates come from different upstream files and disagree, so
  several lines start above 100 % strain and no implicit solver can take a first
  step. Relaxation integrates the structure with every segment stiffness scaled
  down and hands it back as the residual falls, then holds at full stiffness
  while the settling damping decays away.
- `save_state_log`, `read_state_log` and `start_from_state!`, which write and
  restore the one-row state log the settled-state cache already used, and
  `relaxed_state_name`, which names the relaxed state of a geometry at a depower.
- `V3KiteConfig.init_state` starts a settling from a relaxed state instead of
  from the placed geometry. The state is restored before the flight state is
  applied, so the relaxed bridle shape rides along into the target pose; that
  makes a geometry whose bridle rest lengths disagree with its node positions
  settleable at all, and one that agrees settle faster.
- `V3KiteConfig.backend`, so a settling can run on the `KernelBackend`. Settling
  built its model without one, and the monolithic build is the dominant cost on a
  beam wing. `build_replay_sys_struct` takes one for the same reason.
- `flight_replay.jl` replays on the beam wing as well, picked from its menu:
  `system_beam_replay.yaml` brings the `KernelBackend` and none of the
  wing-lattice corrections. The beam does not settle yet:
  `update_sys_struct_from_data!` assigns the flight velocity point by point,
  which leaves the beam's rigid bodies at rest.
- The beam wing flies `AeroPressure` only. `data/kite_settings_beam.yaml` carries the
  surface-traction transfer the wing is built for, and the `ContinuousAero`
  variant is gone rather than kept as a second file. A project naming a
  pressure kite now needs the surface-resolved aero geometry, and
  `aero_geometry_path` says so at load time instead of letting the model build
  fail on a missing surface contour.
- `examples/v3beam_aero_geometry.jl`, which slices `V3_25.obj` into the
  surface-resolved aero geometry `AeroPressure` needs — airfoil contours, polars
  and the per-node `Cp`/`cf` tables — where the stock `aero_geometry.yaml`
  carries lift/drag/moment polars alone. It slices VortexStepMethod's copy of the
  mesh, which is already rotated into slicer convention and raised by 7.3 m and
  is what `aero_geometry.yaml` came from, because `obj_to_yaml` rotates but does
  not translate. The output is not in git, being derived and regenerable.
- `examples/v3beam_geometry.jl` emits the beam geometry and
  `examples/relax_bridle.jl` relaxes a geometry and logs the state. Both write
  files that are in git — `data/struc_geometry_beam.yaml` and
  `data/relaxed_*.arrow` — so no other example regenerates or relaxes anything,
  and a beam run costs nothing but the run. `relax_bridle.jl` is
  geometry-agnostic and gives the particle model a relaxed start too.
- `HeadingSettings` and `data/heading_settings.yaml`, the sibling of
  `WC_Settings` for the steering loop. The heading PID was retuned in every
  example that closed it — K between 1.0 and 1.2, `Td` 0 or 0.15, `max_steering`
  0.15 or 0.175 — with only `simple_sinus.jl` documenting its tuning, so the
  numbers now live in a file where the divergence is visible. The struct also
  carries the two things the examples did to the PID after building it, the
  `1/v_app` gain schedule and the `Td` ramp, so `heading_pid` plus
  `schedule_heading_pid!` covers all of them. The setpoint stays in the example,
  being the maneuver rather than the controller.
- `bin/delete_cache_files`, which removes V3Kite's cached `model_*.bin`,
  `settled_*.bin`/`.arrow` and settling-log files from the scratchspace
  `default_cache_path` now always uses. Cache names are version-stamped, so
  most upgrades are a plain cache miss; this is for the case a serialized
  type's layout changed without a version bump in the tag, and `deserialize`
  throws instead. `--dry-run` lists without deleting, `--yes` skips the
  confirmation prompt.

### Changed
- `examples/flight_replay.jl` runs the replay and saves the logs; the plotting
  moved to `examples/flight_replay_plots.jl`, which draws them from disk. This
  retires the `LOAD_FROM_DISK` toggle — running the plots alone is what it did.
- BREAKING: the replay's maneuver, feedback gains and figure options moved from
  constants at the top of `flight_replay.jl` into a `replay_settings:` key on the
  replay projects, loaded with `load_replay` into a `V3ReplayConfig`. The UTC
  windows are `maneuvers:` entries in `data/replay_settings.yaml` rather than an
  `if`/`elseif` chain in the script.
- The replay kite settings carry the same damping as the wings they replay:
  `kite_settings_psm_replay.yaml` gains the kernel backend and `body_sim_damping`
  of 20 about z, and `kite_settings_beam_replay.yaml` gains that plus
  `beam_body_damping`, `beam_world_damping`, `beam_angular_damping` and
  `beam_joint_damping_scale`. Both had drifted from their flying configuration
  without being corrections fitted to the flights, which is what those files are
  for; a beam replay was flying with no beam damping at all.
- BREAKING: the beam bodies in `data/struc_geometry_beam*.yaml` carry a `wing`
  column naming the wing each belongs to, which is what lets the body damping
  resolve a body's velocity against its parent wing. Regenerate with
  `examples/v3beam_geometry.jl`.
- The beam's four rigidities all come from `tube_linear_rigidities` at the tube
  pressure, so `EA`, `GA`, `EI0` and `GJ` share one Breukels provenance instead of
  taking `EA`/`GA`/`EI0` from the membrane laws. `EA` roughly doubles and `GA`
  drops by a fifth. `V3BeamTopology.pressure_bar` is now the V3 bridle file's
  measured `0.3` bar rather than 4.5 psi, which it was within 3% of anyway.
- BREAKING: the beam joints in `data/struc_geometry_beam*.yaml` carry a single
  `damping` column (Rayleigh β in seconds) instead of `damping_trans`/
  `damping_rot`. `V3BeamTopology.damping_ratio` is now converted to β via
  `β = 2ζ/ω` at the element's axial mode `ω = sqrt(EA/(L·m))`, following
  `ζ = βω/2`. The old dashpot on relative node velocity also braked rigid
  rotation of the whole wing — 15332 N·m·s/rad about yaw — which is what made
  the beam kite refuse to steer. Regenerate with `examples/v3beam_geometry.jl`.
- BREAKING: the `sim_settings:` files are named for the key that points at them
  and for the run they describe: `sim_settings_default.yaml` (was
  `settings.yaml`), `sim_settings_v3kite.yaml`,
  `sim_settings_reelout.yaml`, `sim_settings_cabauw.yaml`. `settings.yaml`
  against `settings_v3kite.yaml` said nothing about which held what; the new
  names line up with `kite_settings_*`, `settle_settings_*` and `wc_settings`.
- BREAKING: the two aero geometries are named for the solver they came from and
  sit side by side: `cfd_aero_geometry.yaml` (was `aero_geometry.yaml`) and
  `nf_aero_geometry.yaml`, the NeuralFoil sweep `v3beam_aero_geometry.jl` writes.
  Its per-node tables stay under `data/polars_neuralfoil/` while the geometry
  itself moves beside its sibling, using the `geometry_path` keyword
  `obj_to_yaml` gained. What a dataset can drive is still read off the file by
  `has_surface_tables`, not off its name.
- BREAKING: the kite and settling settings files are named for the project key
  that points at them — `kite_settings_psm.yaml`, `kite_settings_beam.yaml`,
  `settle_settings_default.yaml`, `settle_settings_replay.yaml` — and each now
  states every field of its struct explicitly, one line of comment per key,
  rather than leaving defaults implicit. The beam files omit the tip and
  trailing-edge reductions, which address lattice segments by index.
- A run now takes its kite and flight condition from a KiteUtils **project
  file** — `data/system_*.yaml`, a `system:` section of pointers — instead of
  from constants in the example script. Most of the mechanism already existed
  and was going unused: `structural_geometry:`, `aero_geometry:` and
  `vsm_settings:` are KiteUtils keys with KiteUtils accessors, and `Settings`
  already carried `sim_time`, `sample_freq`, `l_tether`, `elevation`, `depower`
  and `v_wind`. Two keys are new, `kite_settings:` and `settle_settings:`, and
  each points at a file loaded into a struct the way `wc_settings.yaml` and
  `ripple_settings.yaml` already were. Every key is choosable on its own, so
  examples share the files they agree on and differ only where they must.
- `V3KiteConfig` holds what KiteUtils has no concept of: the backend, the aero
  mode, the VSM interval, the damping, the geometry adjustments, the bridle
  material and how the model is brought up. `V3BridleConfig` splits the line and
  membrane material out of `V3BeamTopology`, so the generator that emits a
  bridle and the run that loads it share one definition.
- `V3SettleConfig` keeps only the settling schedule and embeds the kite it
  settles; its geometry paths, `system_yaml`, `aero_mode` and `backend` are gone,
  having duplicated `V3SimConfig`'s. It gets its own project key because the
  schedule is per-example while the kite is not.
- `build_v3_model(project)` brings up either kite from a project file, covering
  both the settling and the relaxed-state paths and applying the bridle material
  and bending law that the examples applied by hand.
- `create_v3_model` takes a project filename rather than a config struct. Pass
  `settings` to override the project's own, which is what rebuilding a structure
  to match a recorded log needs.
- `init`'s `aero_mode` and `gc` now default to `nothing`, taking both from the
  project's kite settings; pass either to override it.
- `examples/v3kite.jl` and `examples/flight_replay.jl` open a menu for the wing
  to run — `select_project` — and fly either kite through the same code.
  `examples/v3beam.jl` and `examples/v3beam_replay.jl` are gone, each having been
  its PSM counterpart with another project file's worth of constants. Without a
  terminal the menu takes its default instead of asking, so a scripted run is
  unaffected.
- `V3KiteConfig.wing_mass`, `wing_mass_le_frac` and `wing_drag_coeff` carry the
  lattice corrections `flight_replay.jl` applied by hand. Both redistribute over
  wing nodes, which a beam wing does not have mass or drag on, so the beam
  project turns them off by leaving them out.
- The settled state is cached as a one-row `Float64` log rather than a rewritten
  geometry YAML: `settled_struct_path` became `settled_state_path` and the
  structure is rebuilt from the source YAML with that state restored onto it. A
  `Float32` state does not reproduce `integrator.u` on a bridle this stiff.
- The source geometry enters the settled-state cache key, a state logged for one
  having the wrong number of points for another. Left out at the default
  `struc_geometry.yaml`, so existing cache files keep being found.
- `KiteUtils` 0.12, whose `SysState` carries a complete differential state — the
  point velocities, body turn rates and pulley lengths a single logged row needs
  to restart a simulation.
- Structural YAMLs carry a `variables` block instead of a `materials` table,
  which SymbolicAWEModels removed. `dyneema` is now a multi-variable filling
  `youngs_modulus`, `damping_per_stiffness` and `density` wherever it is
  written, so the segment and tether tables name those three columns instead
  of `material`. The hardcoded wing stiffnesses of `struc_geometry.yaml` moved
  to `wing_tube` and `wing_wire` variables, which give the same `unit_stiffness`
  (`youngs_modulus * area`, at the 1 mm diameter every wing row uses) and the
  same damping as the numbers they replace.
- Pulley rows carry an `efficiency`, SymbolicAWEModels having replaced the fixed
  pulley damping with a sheave friction that scales with line tension rather than
  with rope speed. Every V3 pulley is written at `0.95`, a sealed ball-bearing
  sheave, none of them having been measured; the artificial `damping` that sits
  next to it is a debugging aid and stays zero.
- Body- and world-frame damping are applied to every point. The tether-skipping
  `set_body_frame_damping!`, the `set_v3_body_damping!` two-region pattern and
  `V3SettleConfig.body_damping_overrides` are gone, along with the override
  suffix in the settled-geometry cache key; `V3SimConfig` gained
  `world_damping_pattern` next to `damping_pattern`.
- BREAKING: `default_cache_path(data_path)` now always resolves to the depot
  scratchspace keyed by V3Kite's own UUID, regardless of whether V3Kite itself
  is Pkg-installed or a development checkout, and regardless of what
  `data_path` names — a caller's own project directory outside V3Kite's tree
  included. A development checkout used to cache `settled_*.arrow`/
  `model_*.bin` straight into `data/`; it now lands in
  `$DEPOT/scratchspaces/<uuid>/v3kite_cache`, the same place
  `precompile.jl`'s workload writes, so every caller — dev'ed, installed, or a
  downstream package naming its own project file — agrees on where the cache
  lives. `data/system.yaml`, a stray top-level project file the old in-place
  cache location left behind, is removed along with the `isfile` test that
  pinned it.
- `bin/run_julia` activates `examples/` (`--project=examples`) rather than the
  root project, so a REPL started this way gets GLMakie and the other
  example-only dependencies without a manual `Pkg.activate`.
- `data/kite_settings_psm.yaml` changes its defaults: `backend: kernel` (was
  `monolith`), `vsm_interval: 5` (was `1`) and `remake_model: false` (was
  `true`) — matching what the examples already overrode by hand, so a run
  started from the file alone now flies the same configuration.
- `VortexStepMethod` compat capped at `~4.1.1` (was `4.1.0`, unbounded above);
  the resolved version is `4.1.2`.
- The V3-flying examples integrate at a 3x finer timestep (`dt = 0.05/3`, was
  `0.05`) in `reel_out_v3.jl` and `steering_test_v3.jl` while holding the VSM
  aero solve to every 5th step (`VSM_INTERVAL = 5`, was `1`) across all five
  V3 examples (`reel_out_v3`, `simple_auto_parking`, `simple_parking`,
  `simple_sinus`, `steering_test_v3`). `BODY_SIM_DAMPING`'s flown floor drops
  to `[0, 0, 32]` (was `[0, 0, 40]`), matching the `0.8 * body_damping` floor
  `init` has computed since v1.1.1. `simple_parking.jl` and
  `simple_auto_parking.jl` also time initialization with `Timers.tic()`/`toc()`.

### Fixed
- A run that aborts in its first seconds no longer dies in the AoA ripple metrics:
  the parking and sinus examples report through the new `print_ripple_report`,
  which warns that the log is too short instead of erroring out of the script
  before the failure is printed.
- Settling and the replay reinitialize the integrator with the solver `init!`
  built, instead of a bare `FBDF()`. That default differentiates forward, so on
  the monolith backend — which builds with `AutoFiniteDiff` — every `reinit!`
  compiled the whole right-hand side a second time at `ForwardDiff.Dual`.
  Settling the PSM wing on a warm model cache: 864 s -> 230 s.
- Settling applies the kite's bridle material and beam joint damping, through the
  same `apply_kite_material!` the flight model uses. It settled on the geometry
  YAML's stiffness and then flew `kite.bridle`'s, so the "settled" state was not
  an equilibrium of the model that started from it. Existing `settled_*.arrow`
  files are regenerated.
- `beam_joint_damping_scale` is applied once instead of twice. Settling ran it
  over the joints and then `build_v3_model` ran it again over the same structure,
  and since it multiplies in place the beam settled and flew at the square of the
  scale rather than the scale itself.
- `examples/flight_replay.jl` takes `remake_model` and `remake_settled_state` from
  the kite settings instead of forcing a re-settle on every run.
- `beam_body_damping` and `beam_world_damping` skip the wing bodies, as
  `beam_angular_damping` already did. On a `RIGID_DYNAMICS` kite the one dynamic
  body is the whole wing, and these settings damped its flight.
- `setup_settling_model` honours the project's `remake_model` instead of
  hardcoding `remake=false`. With `remake_model: true` a run settled on the
  serialized model and only rebuilt afterwards, so the settled state every
  later stage starts from was produced by the previous right-hand side.
  `settle_wing` then reuses that rebuild for the settled model rather than
  compiling the kernels a second time; the bin's structure hash rebuilds
  anyway if the settled structure is not the one settling built.
- `examples/relax_bridle.jl` rebuilt nothing: it called `init!(remake=false)`, so
  it read back the serialized model and ran the old right-hand side whenever the
  equations had changed. It now honours the project's `remake_model`, as
  `build_v3_model` does.
- The two replay projects flew different conditions: `system_psm_replay.yaml`
  read `settings.yaml` while `system_beam_replay.yaml` read
  `settings_v3kite.yaml`, so the pair differed in `sim_time` and depower as well
  as in wing model. Both now read `sim_settings_default.yaml`.
- The settling schedule no longer overrides the damping the kite file asks for.
  `min_damping` named the same quantity as `V3KiteConfig.body_damping` and won,
  so a kite settling to `[0, 0, 0]` flew at the schedule's `[0, 0, 20]`. The
  floor is now the kite's own, and both names say which phase they belong to:
  `body_start_damping`/`world_start_damping` in the settling schedule,
  `body_sim_damping`/`world_sim_damping` in the kite file.
  BREAKING: `settle_settings:` no longer accepts `min_damping`, and
  `body_damping`/`world_damping` are renamed in both files.
- `plot_twist_dist` paired wing nodes off two at a time, which holds only on the
  lattice. A beam station carries eleven chordwise control points that are wing
  nodes too, so the plot read one real station in six and made chords out of
  pairs of control points. It now reads the twist surfaces.
- `apply_geom_adjustments!` now skips the tip and trailing-edge reductions on a
  beam wing, as its docstring always said it would. Both address segments by
  index into the particle lattice, and the beam is the larger structure, so the
  in-range check they were guarded by passed and the corrections landed on canopy
  membranes — 0.2 m off a 1.31 m `spanwise_2_9` and its neighbours.
- `V3BeamTopology.frame_offset`'s default now `copy`s
  `V3_ADAPTER_FRAME_OFFSET` instead of aliasing it. A `Base.@kwdef` default is
  evaluated once and shared by every instance that takes it, so mutating one
  topology's `frame_offset` in place mutated the constant every later
  `V3BeamTopology()` silently inherited.

### Removed
- `V3SimConfig` and `run_v3_simulation`. Nothing called `run_v3_simulation`
  outside its own kwargs forwarder, and everything `V3SimConfig` carried has a
  home: the geometry paths are project-file keys, the flight condition is
  `Settings`, and the model options are `V3KiteConfig`.

## V3Kite v1.2.0 17-08-2026

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
  `[0.342, 0.465, 0.583, 0.626]` in `sim_settings_default.yaml`, `sim_settings_reelout.yaml`
  and `sim_settings_cabauw.yaml`, closing the gap where `load_windfield` snapped
  an unlisted ground wind speed to a neighbor's turbulence intensity. The
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
  `data/system_reelout.yaml`, `data/sim_settings_reelout.yaml`,
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
