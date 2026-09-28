### Changed
- V3Kite runs on VortexStepMethod v6, SymbolicAWEModels 0.19 and KiteUtils 0.13, and
  no longer on the versions before them.
- BREAKING: the `bodies` table of a beam geometry names a body's own mass
  `extra_mass`, as SymbolicAWEModels 0.19 reads it; the SurfplanAdapter writes it so,
  and a file of your own still saying `mass` errors when it loads.
- `read_state_log` and settling read a state log that declares no frame convention as
  KiteUtils 0.13's `KA`, which is what SymbolicAWEModels wrote into it. Read as `KS`,
  the default for such a log, the tracked relaxed states come back turned about 90 deg.
- `data/vsm_settings.yaml` names the apparent wind speed `condition.va` and drops the
  artificial damping keys, as VortexStepMethod v6 reads them.
- `examples/v3beam_aero_geometry.jl` slices the V3 mesh at `WINGTIP_DISTANCE = 0.05`
  m of span instead of 0.15 m of leading-edge arc: VortexStepMethod v6 spreads the
  sections evenly over the span of the quarter-chord line and measures the distance
  along it. The tip sections stay where 0.15 m of arc put them, on 0.87 m of chord,
  covering 8.311 m of the 8.333 m span; the interior sections move to even spanwise
  spacing, and every beam aero result moves with them. `data/nf_aero_geometry.yaml`
  is not in git, so regenerate it and rebuild the model once.

### Fixed
- `batch_run_circles.jl` and `batch_run_zenith_then_circles.jl` ramp the wind speed
  through `wind_vec`. The `v_wind` they assigned was discarded, because every
  `sim_settings_*.yaml` sets `use_wind_vec: true`, and KiteUtils 0.13 makes that
  assignment throw.
