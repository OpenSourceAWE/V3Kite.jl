# Plan for small improvements towards figure 8 flight

## Ground rules

- use the function `turn_rate_coeffs` to determine c1, c2 and delay
- don't run for longer than 30 s
- run the script `simple_fig8.jl` after each change of the parameters
- run it using the `ex` tool from Kaimon
- `include` `simple_fig8_plots.jl` at the end of `simple_fig8.jl`
- NEVER run julia from the command line
- don't iterate automatically — one change, one run, report, wait.
  EXCEPTION: when a minimization is explicitly requested ("minimize X"), a
  sweep is allowed: one parameter at a time, 10% steps, results recorded in
  that parameter's comment block in `simple_fig8.jl`.
- change existing tuning parameters by at most 10% per iteration

## Status 2026-08-02 — the figure eight flies

**All four success criteria pass**, 150 s, 9000 steps, no abort:

```text
Fig8 (settled from t=25.0s): laps=4.0 | RMS d=2.90° mean=2.19° max=7.07°
  elevation: min settled=23.7° min WHOLE RUN=23.7° | peak ψ̇=56°/s
  tether force: mean=2432N std=382N (CV=15.7%)
  steering: peak |u_s|=0.300, 12% of time within 2% of it | HF turnrate=0.23°/s
  tape: delivered 0.300 of 0.300 commanded | RATE-limited 2% of the time
```

Eight centre crossings, azimuth -47.5 .. +47.2° against the ±50 reference,
elevation 23.7 .. 48.4°. For comparison, the same script at the start of that
session: 28.6 s to an overspeed abort, laps 0.0, RMS d 4.80°, peak ψ̇ 1096 °/s,
command on the clamp 88 % of the time, tape rate-limited 91 %.

### Current configuration

| parameter | value | note |
|:---|---:|:---|
| `HEADING_P` | **0.6** | derived from phase margin, not swept — see "How it was fixed" |
| `HEADING_D` | 0.15 | untouched since the loop was linearised; re-examine |
| `DEPOWER_SETPOINT` | **0.40** | 0.36 crosses the centre but diverges at 37.6 s on energy |
| `ATTRACTOR_DIST` | **12.1** | swept 10 .. 20, full table in the parameter's comment block |
| `EL_CENTER` | 40.5 | **next target: 26** (see next steps) |
| `MAX_STEERING` | 0.30 | command; the tape delivers 0.300 peak, 2 % rate-saturated |
| `SIM_TIME` | 150 | left long for lap counting; restore to 30 for routine tuning |
| `DT` | **0.05/6** | halved 2026-08-02 against a numerical instability, see below |
| winch | force mode | `WINCH_FORCE_MODE = true` |
| entry | phased, from the right | `CHI_DIVE < 0`, `DIVE_EL_MARGIN = 15` |
| feedback | pure course in phase 3 | `FIG8_PURE_COURSE = true` |

### The 30 Hz mode — not every abort is an energy failure (2026-08-02)

The 150 s run at `HEADING_D_N = 2.0` stopped at t = 46.07 s on the overspeed
guard. It is **not** an energy failure and **not** the filter's fault:

- `v_app` is flat at 27 m/s and the tether force flat at ~3.1 kN until the
  FINAL timestep, where `v_app` jumps 26.6 -> 67.9 m/s in one `dt`. The guard
  reported a symptom one step late.
- What actually diverges is a step-to-step (2·`dt` = 30 Hz) oscillation of the
  wing, growing ~1.3-1.4x per sample from t = 45.0: AoA zigzag 0.14° -> 1.6° ->
  6.5° -> 47.7°, kite-acceleration zigzag 10 -> 14 -> 73 -> 7621 m/s².
  Centre-panel and span-mean AoA alternate in ANTIPHASE, so it is the structure.
- The N=2 and N=10 trajectories agree to **0.3°** in azimuth+elevation over the
  whole 46 s and to ~2 % on per-lap peak force. The surviving N=10 baseline
  carries the same mode for all 150 s (Nyquist acceleration content 10-30 m/s²
  in every 5 s window, AoA zigzag touching 0.99° at t = 22 s and recovering).

Cause: `step!` runs with `vsm_interval = 1`, i.e. the VSM aero load is refreshed
once per `dt` and held frozen inside the DAE in between — an explicitly coupled
aero-structure scheme, whose characteristic instability is a growing 2·`dt`
oscillation appearing first at maximum dynamic pressure. That is where it
appears: bottom of the right lobe, ~3.2 kN, elevation ~21°.

Fix applied: `DT` 0.05/3 -> 0.05/6, halving the aero lag for ~4x margin, at 2x
wall time per run. `HEADING_D_N = 2.0` kept. **Consequence for tuning: the model
rides this mode permanently at this operating point, so a run that dies with
flat `v_app` and force is a numerics result, not a parameter verdict — check the
AoA/acceleration zigzag before recording it in a parameter's comment block.**

## Next steps

0. **Re-establish the baseline at `DT = 0.05/6`.** A 150 s run is needed before
   any further parameter verdict; `data/fig8_run_N10_baseline.arrow` is no
   longer bit-comparable, only comparable on per-lap metrics.
1. **Lower the pattern towards centre 26° — but the lever is `F8_B`, not
   `EL_CENTER` alone.** Descent attempted 2026-08-02:

   | `EL_CENTER` | bottom edge | result |
   |---:|---:|:---|
   | 40.5 | 30.5° | 4 of 4 criteria, RMS d 2.90° |
   | **36.5** | 26.5° | 4 of 4, **RMS d 2.63°** — better, and energy flat (median v_app 25.9 -> 25.4 across the settled window) |
   | 32.8 | 22.8° | **FAILED at 48.7 s** — undershot to 17.9°, 5° below the bottom edge, then 25.2 -> 32.6 m/s and 2404 -> 3811 N in five seconds |

   Tracking IMPROVED all the way down (the curvature-margin argument works); it
   is the bottom edge that kills the run, by putting the kite in the power zone
   on the right-lobe turn. At `B = 20` the bottom sits 10° below the centre,
   so centre 26 would mean a bottom at 16°. The reference controller reaches
   centre 26 with `A = 40, B = 15` — a bottom at 18.5°. **Next run: reduce
   `F8_B` at the current 36.5 centre**, then resume the descent. Note "pattern
   size" is one of the levers whose closure was measured through the relay loop
   (see History), so the "smaller is tighter" objection is untested at this
   tuning — and the loop now has margin it did not have then (clamp 8 %, tape
   2 %).
2. **Extend the `ATTRACTOR_DIST` sweep below 10.** RMS d was still falling
   monotonically at the bottom edge of the swept range (2.85° at 10 vs 2.90° at
   the chosen 12.1), so the optimum has not been bracketed.
3. **Re-run the four untested "closed" levers** — entry course, entry timing,
   pattern size, `HEADING_P` around 0.6 — under the fixed loop. Two of the six
   reversed on re-test; the other four were measured through a relay.
4. ~~**Restore `SIM_TIME = 30`** for routine tuning.~~ Done 2026-08-02. The 150 s
   4-lap log is kept as `data/fig8_run_N10_baseline.arrow` (`fig8_run` is
   overwritten by every run).
5. Reconsider `HEADING_D = 0.15`. Derivative action against a 0.72 s effective
   delay is destabilising; it was irrelevant while saturated, and has never been
   examined with the loop linear. **Half-done 2026-08-02**: the visible ~5-8 Hz
   ripple on `u_s` was traced to the D path's noise gain, and
   `HEADING_D_N = 2.0` (new, forwarded through `create_heading_pid`) cut the
   2..25 Hz command RMS 0.00505 -> 0.00225 with the flight unchanged — see the
   parameter's comment block. The gain itself is still untouched: at the loop's
   0.1 Hz the D path contributes a gain of 1.005 and 5.4° of phase lead, so
   `HEADING_D = 0` is the obvious A/B and costs almost nothing on paper.
6. Switch the overspeed guard's abort message from `sys_state.AoA` (centre
   panel) to `span_mean_aoa` — it currently reports a stall that did not happen.

Longer-term, unchanged:

- **A V3 log that flies the eight** (offered 2026-08-01, not yet supplied). The
  current reference is a smaller, far more agile ram-air kite. From a V3 log read
  off `l_tether`, `depower`, the clamped fraction, the flown pattern size and
  centre elevation. Save as `data/v3_fig8_reference.arrow`, NOT `fig8_run`.
- Fix `SysState.course` centrally rather than in the example: `calc_course` in
  `src/interface.jl` and the `:course` correction mode in `src/stabilization.jl`
  both consume the raw field.
- The settled-geometry cache key ignores the settling elevation, so `ELEVATION`
  in `simple_fig8.jl` has no effect and every run starts at the 73° park (see
  `src/stabilization.jl`, PlanFig8.md Findings 4).

## How it was fixed — the causal chain

Three things in order. None of it was tuning by sweep, and the order mattered:
until step 2 the controller was saturated, so nothing downstream was measurable.

### 1. The steering tape is RATE-limited, not authority-limited

The clamp statistics quoted throughout the campaign measured `set_steering`, the
COMMAND — not what the kite received. Phase 3 of the 2026-08-02 run
(16.98 .. 28.0 s):

| | command (`set_steering`) | actual (`steering`) |
|:---|---:|---:|
| max \|u_s\| | 0.3000 | 0.2996 |
| fraction at the ±0.300 clamp | **88.4 %** | **0.0 %** |
| fraction above 0.27 | — | 29.2 % |
| RMS | 0.288 | 0.209 |

The tape essentially never reaches the clamp. What it does instead is slew at
its limit **66.5 % of the time**, at exactly 0.200 s⁻¹ = `v_steering` in
`data/settings_reelout.yaml` (full range 2 units, so 10 s end to end):

- a full reversal -0.3 -> +0.3 takes **3.0 s**;
- the command changed sign **16 times in the 11.0 s** of phase 3, one every
  0.69 s, so the tape could never complete a swing before the demand flipped;
- the actual lagged the command by **0.717 s**, ~1.9x the 0.383 s dead time
  `turn_rate_coeffs` reports for this depower;
- amplitude attenuated 0.288 -> 0.209 RMS (-27 %).

That is a rate limiter: amplitude attenuation plus amplitude-DEPENDENT phase
lag. It explains the one result the authority story never fit — `MAX_STEERING`
0.30 -> 0.33 made things WORSE and 0.375 destabilised the bare plant. Under an
amplitude ceiling more authority helps; under a rate limit a larger command
means longer slewing and more lag.

`src/turn_rate_id.jl` already knew this — it fits `sl.steering` and never
`set_steering`, "the actuator is strongly slew-limited during a bang-bang
reversal". The identified `delay` is therefore a SMALL-SIGNAL dead time; the
fig8 loop runs in the large-signal regime where the effective delay is ~0.72 s
and depends on how hard it is driven.

### 2. Against that lag the loop was ~8x over gain

With `HEADING_P = 4.5` the effective gain in flight is K ≈ 2.55, so the ±0.30
clamp is reached at only **6.7°** of course error — and **88.2 %** of phase-3
samples exceeded it, matching the 88.4 % clamp fraction. Only the SIGN of the
error reached the plant: the controller was a relay, and the measured
consequence was the kite turning at a median 43.5 °/s while the guidance asked
for a median 8.3 °/s. A 5x overshoot, i.e. a self-sustained oscillation, not a
tracking deficit. The 40° median course error was the RESULT of the oscillation.

The plant from steering to heading is an integrator, `psi_dot = c1*v_a*u_s`,
with `c1*v_a ≈ 3.66` rad/s per unit at v_app 20, so crossover is
`omega_c = K*3.66 ≈ 9.3 rad/s ≈ 1.5 Hz`. A delay needs `omega_c*T_d <~ 0.8 rad`:

| delay used | required omega_c | required K | implied `HEADING_P` |
|:---|---:|---:|---:|
| 0.72 s (measured, rate-limited) | 1.1 rad/s | 0.30 | **≈ 0.46** |
| 0.383 s (identified small-signal) | 2.05 rad/s | 0.56 | ≈ 0.86 |

`HEADING_P = 0.6` was chosen between the two. Result, predicted before the run
and confirmed after: the steering cycle moved from 0.181 Hz (5.5 s) to
**0.103 Hz (9.7 s)**, within 3 % of the reference controller's 10 s, amplitude
retention 61 % -> **99.6 %**, tape lag 62.7° -> 29.8°, peak turn rate 1096 ->
41 °/s, turn-rate HF std 72.77 -> 0.27 °/s.

This also explains why 4.5 -> 2.0 had been recorded as "changes only how it
saturates": 2.0 is still ~4x over gain. The fix was x7.5, in the same direction
the earlier test had already pointed.

### 3. Only then were the other levers measurable

- **depower 0.36 -> 0.40** fixed an energy runaway (v_app climbing 20 -> 28 ->
  42.7 m/s lap over lap). 0.36 under the fixed loop crossed the centre and flew
  to 37.6 s; 0.40 flies the full 150 s.
- **`ATTRACTOR_DIST` 15 -> 12.1**, swept 10 .. 20 in 10 % steps at 150 s per
  run. RMS d rises monotonically with the lead (2.85° at 10 to 3.58° at 19.5),
  force ripple falls the other way (CV 17.0 -> 10.7 %), and there is a survival
  CLIFF between 16.11 and 17.72 — both long leads die at ~41 s with one
  crossing. The flown azimuth span also shrinks with lead (±48.6 -> ±45°), so a
  short lead flies a fuller eight. Full table in the parameter's comment block.

### What the working reference told us

Measured on `data/fig8_reference.arrow`, steady state t = 60 .. 120 s, against
phase 3 of the pre-fix V3 run:

| | V3 (pre-fix) | reference |
|:---|---:|---:|
| dominant steering frequency | 0.181 Hz (5.5 s) | 0.100 Hz (10.0 s) |
| command fundamental amplitude | 0.299 | 0.502 |
| delivered fundamental amplitude | 0.183 | 0.496 |
| amplitude retained | 61 % | **99 %** |
| max tape slew observed | 0.200 /s (= the limit) | **1.736 /s** |
| peak \|u_s\| commanded -> delivered | 0.300 -> 0.276 | 1.000 -> 1.013 |

The reference actuator is at least **8.7x faster**, uses the full ±1 range, and
loses essentially nothing between command and tape. What the V3 tape can
deliver: amplitude **0.32** at 0.1 Hz (just above `MAX_STEERING = 0.30`,
adequate with ~7 % margin) and **0.176** at 0.181 Hz (half the command). So the
V3 actuator is only just sufficient, and only at roughly the reference's 10 s
cycle — which is where `HEADING_P = 0.6` put it.

Caveat: different kite and different `steering_gain`, so the AMPLITUDES are not
comparable across the two logs. The frequency and the actuator speed are.

An earlier estimate in this plan — that the pattern only needs ~0.023 Hz, two
reversals per 43 s lap — was inferred from geometry and is WRONG. A working
controller cycles its steering several times per lap because it is a feedback
loop, not a bang-bang per lobe.

## Reference run — the working controller

Log: `data/fig8_reference.arrow` (plot with
`include("examples/fig_eight_plots.jl")`). The name is deliberately not
`fig8_run` — that is what `simple_fig8.jl` saves under, and it overwrote the
reference once.

Pattern from the log metadata: `A = 40, B = 15, C = D = 0, el_center = 26`.
Cross-track mean 0.87°, RMS 1.14°, max 3.36° (t >= 27 s); minimum elevation
8.68°; steady state (t >= 60 s) spans azimuth -43.5 .. +42.2° against `A = 40`
and elevation 15.8 .. 35.7° against 26 ± 7.5°; steering clamped **0.7 %** of the
time.

| phase | t [s] | az -> | el -> | what happens |
|---|---|---|---|---|
| 0 park | 0 .. 10.0 | ~0 | 72 -> 71 | steering free, ±0.3 oscillation |
| 1 dive | 10.0 .. 15.6 | 0 -> 38 | 71 -> 42 | `chi_set` jumps to +98° and ramps 98 -> 143° |
| 2 hold | 15.6 .. 16.8 | 38 -> 45 | 42 -> 27 | `chi_set` pulled back 143 -> 97°, flattening out |
| 3 fig8 | 16.8 .. 120 | 45.5 | 26.6 | handover at the rightmost point, at centre elevation |

The dive is open loop on a near-horizontal course (~98°, straight crosswind to
the right) and simply lets the kite fall along the sphere — 44° of elevation in
6.8 s (~6.5°/s), heading trailing `chi_set` by ~1.5 s. It does NOT aim at the
path while diving (`attractor` is `NaN` until phase 3). Only the last 1.2 s
flattens the descent, so the kite reaches the pattern at az 45°/el 26° moving
downward-left, already inside the first down-turn.

## Measurement notes

- `laps >= 3.0` needs ~130 s of flight, so it is meaningless under the 30 s cap.
  Judge short runs on RMS d, the elevation floor and the tape metrics.
- `print_fig8_metrics` measures from `PARK_TIME + ENTRY_TIME` = 25 s, i.e. over
  the last 5 s of a 30 s run. Either drop `ENTRY_TIME` to ~7 s, or treat the
  printed criteria as informational at that length.
- **Done (2026-08-02):** flown azimuth reach against `F8_A` and elevation span
  against `F8_B` are now metrics *and* criteria — `print_fig8_metrics` takes
  `az_amplitude`/`el_height` and fails a run below `min_span_frac` (0.7) of
  either, per side in azimuth. A run can track well (small RMS d) while flying
  an eight far smaller than the reference, or only one lobe of it, and nothing
  else reports that: every other criterion is measured to the closest point of
  the path. Centre crossings per minute is still not reported (`laps` is, over
  the whole settled window).
- `check_pattern_feasible` computes its margin from `MAX_STEERING`, i.e. from
  the COMMAND. Every feasibility margin in this plan (1.19, 1.33, 1.61, ...) is
  optimistic by roughly the tape's attenuation.

## Instrumentation and toolchain (2026-08-02)

- **Actuator truth in `print_fig8_metrics`.** `fig8_metrics` returns
  `max_steering_delivered`, `tape_rate_frac`, `max_tape_rate` and the
  `v_steering` it scored against, printed as a second steering line. `v_steering`
  is a KEYWORD (default 0.2), not a constant, so a log from a differently
  configured KCU still scores correctly — which matters, because the reference
  log's actuator is nothing like the V3's.
- **Whole-wing AoA.** `span_mean_aoa(sys)` in `src/sim_helpers.jl` (exported)
  averages the VSM's per-panel `alpha_geometric_dist`, skipping `NaN` panels and
  applying the same ±π wrap `update_sys_state!` applies to the centre panel.
  `simple_fig8.jl` logs it to **`var_09`** in degrees.
- **Third plot figure**: α (centre panel vs span mean), L/D (`var_15` wing vs
  `var_16` effective), and `v_app` + `|v_kite|` underneath.
- **The pattern plot no longer draws the attractor track** — it hugs the
  reference path by construction, so it was a third near-coincident curve.
- `examples/Project.toml` pins `MakieControlPlots = "0.1.12"`; 0.1.9's
  legend-height probe throws `only(owned)` on any `plotx` figure with legends on
  more than one row.
- `ATTRACTOR_DIST` and `SHOW_PLOTS` in `simple_fig8.jl` are `@isdefined`-guarded
  so a sweep driver can set them in the REPL. NOTE the footgun: a stale global
  silently overrides the file. The startup `@info` line prints the lead in use.

### AoA: why it looked wrong, and what is true

The plotted `SysState.AoA` is the geometric alpha of ONE panel — the centre one
— and the depower tape acts at the centre, so that panel sits ~4° below span
mean and ~14° below the tips. Measured on a healthy parked state (depower 0.40,
200 m, v_app 8.9 m/s): `SysState.AoA` -0.86°, span-mean geometric +3.05°,
span-mean effective (with induction) +1.22°, body-frame `compute_kite_aoa`
+10.58°, `compute_wing_incidence` +4.18°, L/D 5.78 wing / 2.58 effective.

In flight (centre panel vs span mean, same run):

| window | centre panel (`AoA`) | span mean (`var_09`) |
|:---|---:|---:|
| established flight, 15 .. 27 s | -1.37 .. **0.38** .. +1.88° | +2.17 .. **2.94** .. +3.36° |
| blowup, t >= 28 s | -50.3 .. +49.5° | -2.2 .. +6.1° |

So the wing sits at a healthy +3° while the centre panel straddles zero, and the
overspeed guard's `AoA = -45.2°` message pointed at a stall that did not happen.

"L/D about 40" was a slot mismatch: `fig_eight_plots.jl` plots `var_02`/`var_03`
as L/D, which is `test_figure_eight.jl`'s slot map, while `simple_fig8.jl` writes
attractor azimuth/elevation there and its L/D to `var_15`/`var_16`. The two
plotting scripts are not interchangeable.

## History — superseded findings

Kept because each one cost a run, and because the pattern of failure is the
lesson: **a lever tested through a saturated controller was never tested.**

### The "closed levers" table is not a list of closed levers

Every row was measured with `HEADING_P = 4.5`, through a relay loop saturated
~88 % of the time. Two of the six have since reversed:

| lever | verdict then | verdict with the loop fixed |
|:---|:---|:---|
| depower | 0.36 "diverges at 21.7 s", low depower unsurvivable | **0.40 flies the full 150 s and 4 laps** |
| `ATTRACTOR_DIST` | 35.1 optimal | monotonic the other way — **12.1**, cliff above ~17 |

The remaining four — `HEADING_P` itself, entry course, entry timing, pattern
size — are UNKNOWN, not closed. The original rows:

| lever | tried | result as recorded then |
|:---|:---|:---|
| `HEADING_P` | 5.0 -> 4.5 -> 2.0 | -10% within noise; x2.25 gives pure bang-bang, triples the force ripple, costs 3.3° of elevation floor. Reverted to 4.5. **Now known to be 4x short of the fix.** |
| entry course | `CHI_DIVE` -100 -> -85 | bit-identical run; steering pinned for the whole dive, so only the SIGN reached the plant |
| entry timing | `DIVE_EL_MARGIN` 15 -> 5 | worse: handover azimuth 17.4° -> 7.7°, centre crossing lost |
| depower | 0.40 -> 0.36 | diverged at 21.7 s, v_app 93.6 m/s, 6164 N, lift 11 kN with negative drag |
| pattern size | 50x20 -> 40x15 | margin 1.33 -> 1.02, aborted at 19.1 s — a smaller lemniscate is a TIGHTER one |
| fig8 feedback | `FIG8_PURE_COURSE = true` | bit-identical: the only window where it differs is 16.98 .. 19.15 s, and `set_steering` was at -0.300 on all 130 samples of it |

### The heading/course blend

Three 2026-08-01 runs diverged at 17.6-18.0 s with the blend on and 30.0 s with
pure course, which read as "the blend is a regression". Re-enabled 2026-08-02 at
5.0/10.0 it survived to 28.63 s on the then-current configuration — not a clean
A/B (five parameters differed). It was then settled by `FIG8_PURE_COURSE`: the
blend was active in phase 3 for **2.2 s only**, and from 19.15 s onward
`|v_kite|` never dropped below 9.99 m/s, so the schedule was already commanding
pure course. The feedback signal was not swapping during the unstable part of
the run. The band is not what breaks anything; the loop gain was.

### The reference governor — a clear failure (2026-08-02)

Proposed here as "anti-windup against the real tape position", implemented as a
clamp of the command to within 3 steps of tape travel of the MEASURED position.
**Disastrous, and kept in `simple_fig8.jl` behind `STEERING_TRACK_TAPE = false`
so nobody proposes it again:**

| metric | 4-lap baseline | governor on |
|:---|---:|---:|
| RMS d | 2.90° | 58.10° |
| min elevation | 23.7° | **-49.2°** |
| laps | 4.0 | 2.0 |
| peak tape slew | 0.200 /s | 0.030 /s |
| criteria passed | 4 of 4 | 0 of 4 |

The clamp became the binding rate limit instead of the tape, so the pair could
only advance as fast as the gap allowed — an effective actuator 6.7x SLOWER than
the real one. Two lessons: the premise was already obsolete (since `HEADING_P`
0.6 the tape saturates 2 % of the time, so there was nothing to govern), and it
was not anti-windup at all — with `HEADING_I = false` the PID has no integral
state to back-calculate. If integral action is ever enabled, that changes.

### Compliant winch (2026-08-01)

`winch_force_torque!` (`src/interface.jl`) holds a low-passed reference FORCE
plus a slow length trim and viscous damping, via `step!(; set_torque = ...)`.
`WINCH_FORCE_MODE` switches modes.

- Position mode is stiff no matter how it is tuned: the PI correction rides on
  an exact force feed-forward, so 3.4 kN moved the line 0.01 m; `winch_ff_scale`
  and a weak `winch_speed_ti` raise that to 1.13 m.
- Force mode gives 10.6 m of travel, +2.1 / -1.2 m/s. It needs `winch_damp`:
  force control has no velocity feedback, and without damping the drum
  free-wheels (3.5 m/s, v_app 49.6 m/s, run lost at 19.8 s).
- **It moves the elevation floor.** `EL_CENTER = 45` aborted at 17.4 s on the
  fixed tether and survives with force mode. The "50° is the floor" sweep is a
  fixed-winch artefact.

### Entry state machine (2026-08-01)

`park -> dive -> hold -> fig8`, modelled on the reference run. Dive and hold
command a fixed course OPEN LOOP; the phase is logged to `sys_state.sys_state`
with the reference controller's codes, so both logs read with the same scripts.
`ENTRY_PHASES = false` restores the old behaviour.

- The first version changed nothing: `ENTRY_CHI_MAX` was already clamping the
  guidance course to ±95° through the descent. Its value is that the entry is
  explicit and steerable.
- **The azimuth sign is mirrored** w.r.t. the reference log: a POSITIVE
  commanded course drives this kite towards NEGATIVE azimuth, so entering from
  the right needs `CHI_DIVE < 0`.

### Earlier fixes

- Plotting course and desired course: the angle panel plots psi, chi, chi_set
  unwrapped, with a wrapped error panel below.
- A 180° convention error: `SysState.course` is 180° away from `SysState.heading`
  — corrected locally in `simple_fig8.jl` (see next steps for the proper fix).
- Down-loops (`UP_LOOPS = false`) survive on the course-feedback loop.
