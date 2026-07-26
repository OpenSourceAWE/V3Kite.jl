# Identify steering response

Port the script steering_test_4p.jl from the KiteModels repo to this kite model.
Use the same project and parameters as simple_parking.jl .

## Success criteria

- the script finishes without the kite crashing
- the elevation of the kite should stay above MIN_ELEVATION, for example 50 degrees
- the turn rate law parameters should be identified with a standard deviation of less than 35%

## Status: done (2026-07-26), all three criteria met

### What was implemented

- `examples/steering_test_v3.jl` — the port. Uses `system_reelout.yaml`, `V_WIND = 9.51`,
  `TETHER_LENGTH = 150`, `DEPOWER_SETPOINT = 0.25`, `body_damping = [20, 20, 40]`
  (same parameters as `simple_parking.jl`) and the `init`/`step!` interface in
  POSITION MODE (`set_length = l0`). A relay controller flips the steering
  whenever the heading leaves a ±10° band and steps the amplitude up every
  2 cycles, sweeping `u_s = 0.05 .. 0.25` in steps of 0.025.
- `examples/turn_rate_id.jl` — the identification: steering-delay estimation by
  cross-correlation, the gain statistic `G = ψ̇/(v_a·u_s)`, and a least-squares
  fit of `ψ̇ = c1·v_a·u_s + c2/v_a·sin(ψ)·cos(β)` with standard errors,
  residual RMS and `cond(A)`. Turn rate is recomputed with
  `V3Kite.calc_turn_rate` (frame-transport corrected), not taken from the log.
- `examples/steering_test_v3_plots.jl` — verification plots for the saved run.

Deviations from `steering_test_4p.jl` (KPS4) are forced by the V3 model and are
documented in the module docstring of `steering_test_v3.jl`: much smaller
amplitudes (0.05..0.25 instead of 0.1..0.5, because `V3_STEERING_GAIN = 1.4`),
a wider ±10° heading band, and no side-plate AoA plot.

### Measured result (run of 2026-07-26, log `data/tmp_steering.arrow`)

    outcome            : :sweep_done, 11895 steps, 198.25 s simulated
    min elevation      : 73.03°            (criterion: >= 50°)  PASS
    G                  : 0.0587 1/m ± 21.30 %  (3.365 °/m)  (criterion: < 35 %)  PASS
    c1                 : 0.0582 1/m ± 0.06 %
    c2                 : -2.0520      ± 0.84 %
    residual RMS       : 0.0058 rad/s (0.331 °/s)
    steering delay     : 0.00 s (correlation 0.996)
    cond(A)            : 532.9

All three success criteria pass: the sweep ran to `MAX_STEERING = 0.25` and
exited on `:sweep_done` (no crash), elevation never went below 73°, and the
gain scatter is 21.3 % against the 35 % limit.

### Open points / caveats

- `cond(A) = 533` is high: the two regressors are close to collinear over this
  run, so the *individual* split between `c1` and `c2` is less certain than the
  small standard errors suggest. Wider heading excursions (larger
  `HEADING_OFFSET`) would separate them better — at the cost of elevation loss.
- `c2` comes out **negative** (−2.05), i.e. the fitted gravity term turns the
  kite *towards* zenith rather than away from it, opposite in sign to the KPS4
  result. This is not yet explained; it may be the collinearity above, or a
  sign/frame convention difference in `calc_turn_rate`. Worth a follow-up before
  the coefficients are used in a controller.
- The measured steering delay is 0 samples. The KCU slew limit
  (`v_steering = 0.2` s⁻¹) shows up as tape-amplitude reduction rather than as a
  pure delay, and the fit uses the actual tape position `sl.steering`, not the
  command, so this is expected.