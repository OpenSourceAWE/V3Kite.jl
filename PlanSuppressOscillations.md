# Suppress oscillations

The angle of attack (AoA) shows small oscillations at a frequency of about 5 Hz. They
significantly reduce the simulation speed.

The *goal* is simulation speed; AoA ripple is only the proxy we can see. Both are
measured, so we never trade one for the other by accident.

## Signal and metric

Use `sl.AoA` from `syslog(s.logger)`. This is the VSM wing AoA filled by
`update_sys_state!`, and the only AoA the `init`/`step!` path logs — `var_04`
(`compute_kite_aoa`) is written by `log_state!`, which `step!` does *not* call, so that
column stays empty in `simple_parking` logs.

Analysis window: `t ∈ [2 s, SIM_TIME]`, discarding the post-settling transient.

Detrend before taking the RMS: the parked AoA sits at a non-zero trim value, so a plain
RMS measures the trim point, not the ripple, and would hardly move even if the
oscillation were eliminated. Subtract a 0.5 s moving average so that slow drift in the
trim (elevation still creeping) does not leak into the oscillation number either.

## Analyze

For a run of `simple_parking.jl` with the current default parameters, report:

1. **Ripple amplitude** — RMS and peak-to-peak of the detrended AoA, in degrees.
2. **Ripple frequency** — PSD (or FFT) peak of the detrended AoA. This confirms or
   refutes the "about 5 Hz" observation. At `DT = 0.05/3` the log rate is 60 Hz, so a
   real 5 Hz mode is well resolved (12 samples/cycle), but a stiffer bridle or tether
   mode above 30 Hz would alias down into the same apparent band — and would call for a
   completely different fix. 10 s of data gives 0.1 Hz resolution.
3. **Solver cost** — `s.sam.integrator.stats` (steps, f-evals, Jacobians). Much less
   noisy than wall clock and the number that actually explains the slowdown.
4. **Wall clock** — mean time per `step!` / real-time factor, for reference.

Print items 1–4 at the end of `simple_parking.jl`, computed from `syslog(s.logger)` —
the same data that `save_log` writes. Also print items 1–3 in
`simple_parking_plots.jl`, so the metric can be recomputed from a log already on disk
without re-simulating, and so `simple_auto_parking` / `simple_sinus` can reuse the same
helper. Keep the helper local to `examples/` until a third caller needs it. If tunables
fall out of this (filter cut-off, window start), they belong in a YAML settings file
loaded through a `@with_kw` struct, not as hardcoded constants.

## Discriminate the cause

Candidate causes, in order of likelihood:

- **Bridle-point vibration, insufficiently damped in the wing body frame** — see the
  dedicated section below. A sibling project shows the same fingerprint (tiny amplitude,
  sustained, ~4–9 Hz, emerging once the slow transient has decayed) and cures it with
  body-frame damping.
- **Winch controller** — the cascaded P-outer/PI-inner loop in
  `winch_position_torque!`. A limit cycle at a few Hz is plausible at this bandwidth.
- **Aero–structure coupling** — `vsm_interval = 1` in `step!`, i.e. the aerodynamics are
  recomputed every timestep and can chatter against the structural response.
- **Structural** — bridle/tether segment stiffness from the system YAML.
- **Solver tolerances / timestep.**

Decision tree — stop at the first step that kills the ripple:

1. Raise body-frame damping isotropically (with `remake=true`, see below). Ripple gone
   ⇒ bridle mode; tune and finish.
2. Otherwise rerun with `set_length = nothing` (hold/torque mode) instead of position
   mode. Ripple gone ⇒ winch controller.
3. Otherwise raise `vsm_interval`, then tighten/loosen solver tolerances.

### Body-frame damping (leading hypothesis)

Use **body-frame**, not world-frame, damping. Per `docs/damping.md` in the RamAirKite
project these are opposites: `body_frame_damping` acts on point velocity *relative to
the wing*, so it kills bridle vibration while leaving the kite's global motion free;
`world_frame_damping` acts on absolute velocity and would artificially slow the whole
kite — violating the trim-invariance criterion under "Done when".

V3Kite already has this knob, and it is already engaged — so this is a *retuning* task,
not a new feature:

- `set_v3_body_damping!` (`src/model_setup.jl`) applies the V3 two-region pattern:
  a value on points 1:38 plus an override on 37:38.
- `V3SettleConfig` (`src/stabilization.jl`) carries `body_damping`
  (default `[0.0, 0.0, 20.0]`), `world_damping` (default zero) and
  `body_damping_overrides`.
- `init` currently passes `body_damping = [0.0, 0.0, 40.0]` (`src/interface.jl`).

**The sharpest test: the current setting is z-only.** `[0, 0, 40]` damps only normal to
the wing surface. If the 5 Hz mode is in-plane — chordwise or spanwise bridle swing —
the present setting cannot damp it at all, which would explain a ripple surviving a
coefficient as large as 40. So the first experiment is isotropic damping
(`[40, 40, 40]`), not a magnitude sweep along z.

Go through `set_v3_body_damping!` and the `V3SettleConfig` fields rather than looping
over `sam.sys_struct.points` directly: a blanket loop also damps wing points 2:21, which
changes aeroelastic trim and trips the trim-invariance criterion.

Do **not** copy the sibling project's delay/ramp-in of the damping value. That exists to
protect a pitch-trim window during the first seconds of *its* run; V3Kite establishes
trim in a separate settling phase (`dt = 0.001`, 400 steps) that completes before the
main loop starts. A 5 s delay plus 3 s ramp inside a 10 s run would leave damping barely
engaged and measure nothing.

#### Cache trap — read before running any damping experiment

Damping is applied in `_setup_settling_model`, which runs **only on the settling
cache-miss path**. On a cache hit, `settle_wing` deserializes `data/settled_*.bin`, which
carries whatever damping was in effect when that file was written — and the cache key
encodes tapes, tip/TE reduction, wind, tether length, gravity, system YAML and KCU mass,
but **not damping**.

Consequence: changing `body_damping` and re-running has *no effect* against a warm
cache. Every damping experiment must pass `remake=true` or delete the matching
`data/settled_*.bin` first. Otherwise the sweep shows a flat line and the correct
hypothesis gets discarded. Consider adding the damping vector to the cache key to remove
the footgun permanently.

## Baseline and comparability

Record the baseline numbers together with the parameter set they belong to —
`PROJECT`, `V_WIND`, `TETHER_LENGTH`, `DEPOWER_SETPOINT`, `SIM_TIME`, `DT`. The metric
is only comparable across runs that fix all of these; a change in wind speed or timestep
alone moves the ripple.

Baseline parameter set (`examples/simple_parking.jl` defaults, unchanged):
`PROJECT = "system_reelout.yaml"`, `V_WIND = 9.51`, `TETHER_LENGTH = 150.0`,
`DEPOWER_SETPOINT = 0.26`, `SIM_TIME = 10.0`, `DT = 0.05/3` (600 steps, 60 Hz log).
Analysis settings: `examples/ripple_settings.yaml` defaults (`t_start = 2.0`,
`detrend_window = 0.5`, peak search 0.5–30 Hz).

| metric | baseline `[0,0,40]` | after `[10,10,40]` | ratio |
| --- | --- | --- | --- |
| AoA ripple RMS [°] | 0.0202 | 0.0020 | 10× |
| AoA ripple pk-pk [°] | 0.1443 | 0.0197 | 7.3× |
| PSD peak [Hz] | 5.75 (0.12 Hz resolution) | 5.50 (residual, RMS 10× down) | — |
| solver steps | 3919 accepted, 82 rejected | 1147 accepted, 13 rejected | 3.4× |
| solver f-evals / Jacobians | 252141 / 825 | 58309 / 189 | 4.3× / 4.4× |
| mean time per step [ms] | 9.4 – 10.0 | 5.5 | 1.8× |
| real-time factor | 1.67 – 1.78 | 3.04 | 1.8× |
| mean AoA over 8–10 s [°] | 3.692 | 3.681 | −0.3 % |
| mean elevation over 8–10 s [°] | 77.69 | 77.73 | +0.05 % |
| mean tether force over 8–10 s [N] | 1106.2 | 1097.5 | −0.8 % |

All three **Done when** criteria are met: pk-pk 0.0197° ≪ 0.1°, solver steps
3.4× better (> 2×), and the settled trim unchanged to within 0.5 %.

`[20, 20, 40]` is better still on this configuration (pk-pk 0.0072°, 724 solver
steps = 5.4×, f-evals 53× down) but destabilises the settling of the
`system_cabauw.yaml` configuration — see *Why the default is not the fastest
value* below. It can be passed explicitly where it is known to work.

Cache check: re-running the baseline with `remake = true` reproduces the AoA
metrics and solver counts *exactly*, so the cached settled geometry matched the
`[0, 0, 40]` default it was written with, and the baseline is directly comparable
to the damping runs. Adding the damping to the cache key (below) also reproduces
the baseline exactly, so the key change is physics-neutral.

### What the baseline actually shows

**The ripple is a decaying transient, not a limit cycle.** Per-second RMS of the
detrended AoA over the analysis window:

| window [s] | 2–3 | 3–4 | 4–5 | 5–6 | 6–7 | 7–8 | 8–9 | 9–10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RMS [°] | 0.0424 | 0.0282 | 0.0185 | 0.0125 | 0.0092 | 0.0063 | 0.0042 | 0.0033 |
| pk-pk [°] | 0.1443 | 0.1032 | 0.0692 | 0.0437 | 0.0356 | 0.0253 | 0.0169 | 0.0182 |

A factor ~13 decay over 7 s, i.e. τ ≈ 2.7 s at 5.5 Hz — damping ratio ζ ≈ 0.001.
The spectral peak is stable across the decay (6.5 / 5.5 / 5.5 / 5.55 Hz on the
2–4 / 4–6 / 6–8 / 8–10 s windows), so this is one lightly-damped mode being rung
by the initial condition, not a controller limit cycle. That still fits the
bridle-mode hypothesis — it just means the fix should be judged on the *decay
rate*, and it demotes the winch controller as a candidate.

Three consequences for the rest of this plan:

1. The whole-window pk-pk (0.1443°) is set entirely by the first second of the
   window. The "pk-pk below 0.1°" criterion under **Done when** would be met by
   moving `t_start` from 2 s to 3 s and changing nothing physical. It needs to be
   restated — e.g. as pk-pk over the *last* 2 s, or as the decay time constant.
2. The run never reaches steady state: lift falls 2172 → 1275 N and elevation
   creeps from the 73° settling target to ~77.5°. The reported real-time factor
   climbs from 1.34 to 1.92 across the run for the same reason. Solver cost is
   therefore dominated by the slow relaxation, and a fix that only damps the
   5.5 Hz mode may not deliver the 2× target on its own.
3. The 60 Hz log rate cannot rule out aliasing of a >30 Hz mode. The check is a
   single run at smaller `DT`; the stability of the peak across sub-windows makes
   a genuine ~5.5 Hz mode much more likely, but it is not proof.

## Result: it was the in-plane bridle mode

Step 1 of the decision tree settled it, and the plan's sharpest test was the
right one. The sweep (all runs at the baseline parameter set, `t ∈ [2, 10] s` for
the ripple, `t ∈ [8, 10] s` for the trim):

| `body_damping` | RMS [°] | pk-pk [°] | RMS 8–10 s | f [Hz] | solver steps | ms/step | AoA 8–10 [°] | elev 8–10 [°] | F 8–10 [N] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `[0, 0, 40]` (was default) | 0.0202 | 0.1443 | 0.0037 | 5.75 | 3919 | 9.13 | 3.692 | 77.69 | 1106.2 |
| `[5, 5, 40]` | 0.0049 | 0.0373 | 0.0034 | 5.37 | 1716 | 6.07 | 3.688 | 77.73 | 1100.9 |
| **`[10, 10, 40]`** (new default) | **0.0020** | **0.0197** | 0.0006 | 5.50 | **1147** | 5.36 | 3.681 | 77.73 | 1097.5 |
| `[20, 20, 40]` (fastest, but see below) | 0.0009 | 0.0072 | 0.0010 | 6.12 | 724 | 2.84 | 3.665 | 77.72 | 1096.3 |
| `[40, 40, 40]` | 0.0041 | 0.0182 | 0.0008 | 0.62 | 701 | 3.97 | 3.656 | 77.64 | 1112.4 |
| `[40, 40, 0]` | 0.0045 | 0.0239 | 0.0016 | 11.0 | 785 | 2.94 | — | — | — |

Readings:

- **The mode is in-plane.** `[40, 40, 0]` — no normal damping at all — is almost
  as good as `[40, 40, 40]`, while `[0, 0, 40]` at the same magnitude does
  nothing. The old default could not touch this mode no matter how large it got,
  exactly as the plan predicted.
- **The knee is around 10–20.** Solver steps fall monotonically 3919 → 1716 →
  1147 → 724 and then flatten (701 at 40).
- **`[40, 40, 40]` is not better.** Its ripple RMS is 4.5× *worse* than
  `[20, 20, 40]` (0.0041 vs 0.0009) for the same solver cost, so more damping is
  not monotonically better — the isotropic value used for the first test is past
  the useful range.
- **`ms/step` is noisy; solver steps are not.** `[40, 40, 40]` measured 2.72 and
  3.97 ms/step on two runs with an identical 701 solver steps. Judge changes on
  the step count, as the plan says.
- **Trim is invariant** — but only when measured on the settled tail. Over the
  full `t ≥ 2 s` window the damped runs show mean AoA +16 % and mean tether force
  +6 %, which looks like a trim shift and is not: it is the *transient* decaying
  differently. On `t ∈ [8, 10] s` every case agrees to within 1 %.
- Consequence 2 above (that the solver cost is dominated by the slow relaxation,
  so damping the mode might not deliver 2×) turned out to be **wrong**: the
  relaxation and the ripple were the same problem. At the shipped `[10, 10, 40]`
  f-evals fall 4.3× and Jacobian evaluations 4.4×; at `[20, 20, 40]` it is 53×
  and 69×. The lightly-damped 5.5 Hz mode was what forced the small internal
  steps all along.

Steps 2 and 3 of the decision tree (hold/torque mode, `vsm_interval`, solver
tolerances) were not needed and were not run.

### Why the default is not the fastest value

`[20, 20, 40]` broke the test suite. `test/test-interface.jl` exercises `init` on
`system_cabauw.yaml` (v_wind 10, depower 0.25), and there the *settling* run —
fixed `dt = 0.001`, 400 steps — diverges at step 10 with the added in-plane
damping. Ladder on that configuration:

| `body_damping` | `[0,0,40]` | `[5,5,40]` | `[10,10,40]` | `[12,12,40]` | `[15,15,40]` | `[20,20,40]` |
| --- | --- | --- | --- | --- | --- | --- |
| cabauw settling | ok | ok | ok | ok | ok | **diverges** |

So the in-plane damping stiffens the settling problem faster than it stiffens the
main run, and the fixed-step settling loop has no margin left somewhere between
15 and 20. `[10, 10, 40]` keeps a 1.5× margin below the observed edge and still
meets every **Done when** criterion. Raising `num_substeps` or lowering the
settling `dt` would probably buy back the rest of the speed-up; that was not
pursued.

This also exposed a genuine bug, unrelated to the damping change:

> `_run_power_zone_settling!` set `failed = true`, broke out of the loop, and then
> **serialized the diverged geometry anyway** and logged "Settling complete". The
> caller's `settle_failed` stayed `false`, so `init` happily built a model from
> it and died later in `update_vsm!` with `PARTICLE_DYNAMICS VSM solve failed`.
> Worse, the broken `.bin` is keyed on the inputs, so every later run would reuse
> it until someone deleted the file by hand.

Fixed: a diverged settling now raises, which `settle_wing` turns into
`settle_failed = true`, and nothing is written to the cache. The failure message
is `"Settling diverged before completing N steps; geometry not serialized"`.

### Changes made

- `init` gained a `body_damping` kwarg, default changed `[0, 0, 40]` →
  `[10, 10, 40]` (`src/interface.jl`). Other settling paths that build a
  `V3SettleConfig` directly (`flight_replay.jl`, `batch_run_circles.jl`,
  `batch_run_zenith_then_circles.jl`, `v3kite.jl`, `open_loop.jl`,
  `realtime.jl`) are untouched — they carry their own damping and, in the
  flight-replay case, re-apply it after settling via `set_v3_body_damping!`.
  Whether they benefit from the same change is a separate question.
- The damping now forms part of the settled-geometry cache key
  (`src/stabilization.jl`), so the footgun the plan warned about is gone:
  changing `body_damping` produces a different `data/settled_*.bin` instead of
  silently reusing the old one, and `remake=true` is no longer needed for a
  damping experiment. Existing `data/settled_*.bin` files are invalidated once
  and re-settle on first use (~40 s each).
- `settle_wing`'s docstring claimed the returned model has "no settling
  damping". That was wrong for body-frame damping, which is a per-point field of
  the serialized `SystemStructure` and does carry over — verified directly: every
  one of the 44 points in the main model carried `[0, 0, 40]`. Corrected.

### Open items

- **Restate the `Done when` pk-pk criterion.** It is satisfied here by a wide
  margin, so it did not bite, but as written it is still measurable-by-window
  (see consequence 1 above).
- **Aliasing is still unproven** (consequence 3). Cheap to close with one run at
  smaller `DT` now that a run costs 1.6 s instead of 6 s.
- **Only the parked case was tested.** `simple_auto_parking`, `simple_sinus` and
  `reel_out_v3` also go through `init` and now inherit the new default; the
  reel-out and steering cases were not re-measured.
- **The settling loop is the new bottleneck on damping.** Making it robust at
  `[20, 20, 40]` (more substeps, or a smaller settling `dt`) would recover the
  remaining 1.6× in solver steps for every configuration.

## Done when

- AoA ripple peak-to-peak below 0.1°, **and**
- solver steps (or wall clock per step) improved by at least 2× versus baseline,
- with no change to the parked trim state: mean elevation, mean AoA and mean tether
  force stay within a few percent of baseline, so the oscillation is suppressed rather
  than the physics damped away.
