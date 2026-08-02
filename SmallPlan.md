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

## Done

- plot course and desired course — angle panel plots psi, chi, chi_set
  unwrapped, with a separate wrapped error panel below it
- the PID feedback signal: heading at low speed, course at high, blended.
  Scheduled on `|vel_kite|` (NOT `v_app`: a parked V3 already sees v_app ~ the
  ambient 9.5 m/s, so apparent wind speed cannot tell flying from hanging
  still). Pure course at/above 10 m/s. **BUT the blend at 5/10 m/s is a
  regression — see the open item below.**
- fixed a 180° convention error: `SysState.course` is 180° away from
  `SysState.heading` — corrected locally in `simple_fig8.jl` for now
- ~~`ATTRACTOR_DIST` swept 16.2 .. 38.6 to minimize the RMS course error;
  35.1 is the minimum in both windows~~ **SUPERSEDED 2026-08-02**: re-swept
  10 .. 20 under the fixed loop, RMS d rises monotonically with the lead and
  everything above ~17 dies at ~41 s. Now **12.1** (table in the parameter's
  comment block)
- down-loops (`UP_LOOPS = false`) survive on the course-feedback loop, and at
  `ATTRACTOR_DIST = 35.1` are the first configuration on record that crosses
  the pattern centre instead of circling one lobe
- plotted the working controller's log (`data/fig8_reference.arrow`, 120 s, the
  4-phase state machine — not a `simple_fig8.jl` run) with
  `include("examples/fig_eight_plots.jl")`; see "Reference run" below
- **`HEADING_P` 4.5 -> 0.6** (2026-08-02), derived from phase margin against the
  measured 0.72 s tape lag, not swept. This is the change that made the pattern
  flyable — see the top of this file
- honest actuator metrics in `print_fig8_metrics` (`tape_rate_frac`,
  `max_steering_delivered`), whole-wing AoA in `var_09` via `span_mean_aoa`,
  and a third plot figure for AoA / L/D / speeds
- the pattern plot no longer draws the attractor track: it hugs the reference
  path by construction, so it was a third near-coincident curve hiding the two
  that matter

## SOLVED 2026-08-02 — the figure eight flies

**All four success criteria passed** for the first time, at `HEADING_P = 0.6`,
`DEPOWER_SETPOINT = 0.40`, `ATTRACTOR_DIST = 12.1`, force-mode winch, phased
entry, `EL_CENTER` 40.5, 150 s:

```text
Fig8 (settled from t=25.0s): laps=4.0 | RMS d=2.90° mean=2.19° max=7.07°
  elevation: min settled=23.7° min WHOLE RUN=23.7° | peak ψ̇=56°/s
  tether force: mean=2432N std=382N (CV=15.7%)
  steering: peak |u_s|=0.300, 12% of time within 2% of it | HF turnrate=0.23°/s
  tape: delivered 0.300 of 0.300 commanded | RATE-limited 2% of the time
```

Eight centre crossings, azimuth -47.5 .. +47.2° against the ±50 reference,
elevation 23.7 .. 48.4°, 9000 steps, no abort. Where the session started, same
script: 28.6 s to an overspeed abort, laps 0.0, RMS d 4.80°, peak ψ̇ 1096 °/s,
command on the clamp 88 % of the time, tape rate-limited 91 %.

The order of causes mattered, and none of it was tuning by sweep:

1. the steering tape is RATE-limited, not authority-limited (see "Closed
   levers"), which put **62.7° of phase lag** in the loop at its own 0.181 Hz;
2. against that lag the loop was **~8x over gain** — `HEADING_P` 4.5 clamps at
   6.7° of course error, and 88 % of samples exceeded it, so the controller was
   a relay. `HEADING_P` 4.5 -> **0.6** was DERIVED from phase margin, not swept;
3. only then did the other levers become measurable at all. Depower 0.36 -> 0.40
   fixed the energy runaway; `ATTRACTOR_DIST` 15 -> 12.1 took RMS d under 3°.

Both of those last two were recorded as CLOSED in this plan, on evidence
gathered through the relay loop.

## Open — measurement

- ~~`laps` has been 0.0 in EVERY run recorded so far~~ — **fixed 2026-08-02**:
  4.0 laps at 150 s. `laps >= 3.0` still needs ~130 s of flight, so it stays
  meaningless under the 30 s cap; judge short runs on RMS d, the elevation floor
  and the tape metrics instead.
- `print_fig8_metrics` measures from `PARK_TIME + ENTRY_TIME` = 25 s, i.e.
  over the last 5 s of a 30 s run. Either drop `ENTRY_TIME` to ~7 s, or treat
  the printed criteria as informational at this run length and quote a
  t >= 15 s window instead (what the attractor sweep used).
- better criterion to add: centre crossings per minute, plus flown azimuth
  span against `F8_A`. At `ATTRACTOR_DIST = 35.1` the kite crosses the centre
  twice in 18 s but spans only -23..+33° of the +-50° reference — it flies an
  eight roughly 40% too small, which no current metric reports.

## Reference run — the working controller

Log: `data/fig8_reference.arrow` (plot it with
`include("examples/fig_eight_plots.jl")`). The name is deliberately not
`fig8_run` — that is what `simple_fig8.jl` saves its own runs under, and it
overwrote the reference once.
Pattern from the log metadata: `A = 40, B = 15, C = D = 0, el_center = 26`.
Result: cross-track mean 0.87°, RMS 1.14°, max 3.36° (t >= 27 s); minimum
elevation 8.68°; steady state (t >= 60 s) spans azimuth -43.5 .. +42.2° against
`A = 40` and elevation 15.8 .. 35.7° against 26 +- 7.5°; steering clamped only
**0.7 %** of the time.

Initial transition, phase by phase (`sl.sys_state`):

| phase | t [s] | az -> | el -> | what happens |
|---|---|---|---|---|
| 0 park | 0 .. 10.0 | ~0 | 72 -> 71 | steering free, +-0.3 oscillation |
| 1 dive | 10.0 .. 15.6 | 0 -> 38 | 71 -> 42 | `chi_set` jumps to +98° and ramps 98 -> 143° |
| 2 hold | 15.6 .. 16.8 | 38 -> 45 | 42 -> 27 | `chi_set` pulled back 143 -> 97°, flattening out |
| 3 fig8 | 16.8 .. 120 | 45.5 | 26.6 | handover at the rightmost point, at centre elevation |

The dive is open loop on a near-horizontal course (~98°, straight crosswind to
the right) and simply lets the kite fall along the sphere — 44° of elevation in
6.8 s (~6.5°/s), heading trailing `chi_set` by ~1.5 s. It does NOT aim at the
path while diving (`attractor` is `NaN` until phase 3). Only the last 1.2 s
flattens the descent, so the kite reaches the pattern at az 45°/el 26° moving
downward-left, already inside the first down-turn.

## Open — the heading/course blend (2026-08-01, revisited 2026-08-02)

The `ATTRACTOR_DIST` table's 30 s survival at attr 35.1 stopped reproducing:
three runs (stock winch, soft winch gains, and `winch_ff_scale = 0.9`) all
diverged at 17.6-18.0 s with `min_el ~28°` instead of 40.2°. Isolated by
disabling the blend (`V_KITE_HEADING/V_KITE_COURSE = 0.0/0.001`, i.e.
`w_course == 1` throughout = pure course feedback):

| feedback | t_end | RMS d (>=25 s) | min el whole run | force CV |
|:---|---:|---:|---:|---:|
| blend 5/10 m/s | 18.0 s | — (diverged) | 27.6° | — |
| pure course | **30.0 s** | 4.18° | 40.3° | 8.1% |
| sweep table row (attr 35.1) | 30.0 s | 4.21° | 40.2° | 8.6% |

Pure course reproduces the table row to within noise, so the blend — added in
the same commit as the table and never run successfully before (it carried an
undefined `norm`, fixed 2026-08-01) — is what destabilizes the run. The
reasoning behind the blend still stands (course is undefined at zero velocity);
what is wrong is the band. The kite passes through 5-10 m/s *while flying*, so
the feedback signal swaps mid-manoeuvre. Options: move the band well below the
flying speed range, or gate on phase instead of speed.

### Update 2026-08-02: the blend is no longer what breaks first

`V_KITE_HEADING/V_KITE_COURSE` restored to **5.0/10.0** (blend RE-ENABLED) on
the current configuration — depower 0.36, `ATTRACTOR_DIST` 15, force-mode winch,
phased entry, `EL_CENTER` 40.5:

| | blend, 2026-08-01 | blend, this run |
|:---|---:|---:|
| t_end | 18.0 s | **28.63 s** (overspeed guard, v_app 47.1 m/s) |
| min elevation, whole run | 27.6° | 28.0° |
| peak `ψ̇` | — | 1096 °/s |
| turn-rate HF std | — | 72.77 °/s (vs **1.12** on the pure-course run) |
| force | — | mean 3533 N, CV 11.4% |
| steering on the clamp | — | 94 % |
| RMS d (>= 25 s) | — | 4.80° — window is 25 .. 28.6 s, so it mostly measures the blowup |

The transition phase is visibly good (the reason the blend was restored) and the
flight is well behaved to ~28 s; then it goes violently unstable inside about a
second — peak turn rate 1096 °/s and an HF std 65x the pure-course run are the
signature, not a gradual loss of tracking.

**This is not a clean A/B.** The 2026-08-01 rows were taken at depower 0.40,
attr 35.1, position-mode winch and no entry phases; five things differ. What it
does establish is that the blend is no longer the first failure — the run now
survives the entry that used to kill it and fails at the end instead.

### Update 2026-08-02b: the band cannot be causing the late instability

`FIG8_PURE_COURSE` (new, default `true`) gates the feedback signal on PHASE
instead of speed: phase 3 feeds back course at any speed, the entry phases keep
the `V_KITE_*` schedule. This is the "gate on phase instead of speed" option
proposed above. The run is in the closed-levers table — bit-identical, because
the steering is clamped through the only window where the two differ — but it
settles the question that motivated it:

- the blend was active in phase 3 for **2.2 s only**, from the 16.98 s handover
  to 19.15 s;
- from 19.15 s to the 28.63 s abort, `|v_kite|` never drops below 9.99 m/s, so
  the speed schedule was already commanding (essentially) pure course;
- so the feedback signal was NOT swapping during the part of the run that goes
  unstable. Whatever drives the late divergence, it is not the blend.

The remaining suspects for the 28.6 s blowup are energy and authority, not the
feedback angle: 94 % clamp saturation, force climbing to 5591 N in the last
second, and `v_app` reaching 47 m/s.

## Done — compliant winch (2026-08-01)

`winch_force_torque!` (`src/interface.jl`) implements PlanFig8.md option 1: the
drum holds a low-passed reference FORCE plus a slow length trim and viscous
damping, commanded through `step!(; set_torque = ...)`. `WINCH_FORCE_MODE` in
`simple_fig8.jl` switches between it and the old position mode.

- position mode is stiff no matter how it is tuned: the PI correction rides on
  an EXACT force feed-forward, so 3.4 kN moved the line 0.01 m. `winch_ff_scale`
  (new) and a weak `winch_speed_ti` raise that to 1.13 m — still nothing.
- force mode gives 10.6 m of travel, +2.1 / -1.2 m/s, paying out on the dive and
  hauling in on the climb. It needs `winch_damp`: force control has no velocity
  feedback, and without damping the drum free-wheels (3.5 m/s, `v_app` 49.6 m/s,
  run lost at 19.8 s).
- **it moves the elevation floor.** `EL_CENTER = 45` aborted at 17.4 s on the
  fixed tether (an ENERGY failure: overspeed at 3494 N) and now survives the
  full 30 s, with the steering off its clamp for the first time (92% vs 100%).
  40.5 survives too. The "50° is the floor" sweep is an artefact of the
  fixed-length winch and should be re-run.
- what it does NOT do: relieve the turn-authority limit. Every closed lever
  below still saturates the steering, and a low depower still diverges.

## Done — entry state machine (2026-08-01)

`park -> dive -> hold -> fig8` in `simple_fig8.jl`, modelled on the reference
run above. The dive and hold command a fixed course OPEN LOOP (the guidance
still runs, but its course is discarded), and the phase is logged to
`sys_state.sys_state` with the reference controller's codes, so both logs read
with the same plotting scripts. `ENTRY_PHASES = false` restores the old
behaviour.

Two findings from building it:

- **the first version changed nothing.** `ENTRY_CHI_MAX` was already clamping
  the guidance course to ±95° through the descent, so an explicit +100° dive
  command reproduced the old trajectory bit for bit. The state machine's value
  is that the entry is explicit and steerable, not that it improves anything by
  itself.
- **the azimuth sign is mirrored** w.r.t. the reference log: a POSITIVE
  commanded course drives this kite towards NEGATIVE azimuth. Entering from the
  right (the reference's geometry) needs `CHI_DIVE < 0`. Doing so widened the
  settled pattern from `1.3..42.6°` to `-4.7..+49.2°` and produced the only
  centre crossing seen under force mode.

Current best configuration: entry from the right, `DIVE_EL_MARGIN = 15`,
`EL_CENTER = 40.5`, `ATTRACTOR_DIST = 15`, force-mode winch, blend disabled.
30 s, RMS d 5.99°, span -4.7..+49.2° against ±50°, 1 centre crossing.

## Closed levers — the steering tape is RATE-limited (2026-08-01, re-read 08-02)

> **Correction 2026-08-02.** This section previously read "all
> authority-limited", i.e. the kite is at its turn limit because the command
> sits on the ±0.300 clamp. That is wrong, and the clamp statistics quoted
> throughout this plan measure `set_steering`, the COMMAND — not what the kite
> received. Measured on the 2026-08-02 run, phase 3 (16.98 .. 28.0 s):
>
> | | command (`set_steering`) | actual (`steering`) |
> |:---|---:|---:|
> | max \|u_s\| | 0.3000 | 0.2996 |
> | fraction at the ±0.300 clamp | **88.4 %** | **0.0 %** |
> | fraction above 0.27 | — | 29.2 % |
> | RMS | 0.288 | 0.209 |
>
> The tape essentially never reaches the clamp. What it IS doing is slewing at
> its limit **66.5 % of the time**, at exactly 0.200 s⁻¹ = `v_steering` in
> `data/settings_reelout.yaml`. Consequences, all measured on the same run:
>
> - a full reversal -0.3 -> +0.3 takes **3.0 s**;
> - the command changes sign **16 times in the 11.0 s** of phase 3, one every
>   0.69 s, so the tape can never complete a swing before the demand flips — it
>   tracks a triangle wave against a bang-bang command;
> - the actual lags the command by **0.717 s**, ~1.9x the 0.383 s dead time
>   `turn_rate_coeffs` reports for this depower;
> - amplitude is attenuated 0.288 -> 0.209 RMS (-27 %).
>
> That is a rate limiter: amplitude attenuation plus amplitude-DEPENDENT phase
> lag. It retro-explains the one result the authority story never fit —
> `MAX_STEERING` 0.30 -> 0.33 made things WORSE and 0.375 destabilised the bare
> plant. Under an amplitude ceiling more authority helps; under a rate limit a
> larger command means longer slewing and more lag, so it hurts. Classic
> rate-limiter-induced oscillation.
>
> `src/turn_rate_id.jl` already knew this — it fits `sl.steering` and never
> `set_steering`, "the actuator is strongly slew-limited during a bang-bang
> reversal". So the identified `delay` is a SMALL-SIGNAL dead time; the fig8
> loop runs in the large-signal regime where the effective delay is ~0.72 s and
> depends on how hard it is driven.
>
> **What this changes about the levers below:** they did not close because the
> kite is out of turning authority. They closed because a command that spends
> most of its time saturated hides them — and it saturates because the loop is
> fighting the tape rate. The lever to attack is the demanded steering RATE, not
> its amplitude: lower `HEADING_P`, a longer attractor lead, or a rate-aware
> command shaper. `MAX_STEERING` should probably come DOWN, not up.
>
> Open question before designing around it: does `v_steering = 0.2` s⁻¹ match
> the real V3 KCU? If it does, this is physics and the controller must respect
> it; if it is a placeholder, the whole campaign has been tuning against an
> actuator that does not exist.
>
> **Correction 2, 2026-08-02 (later the same day): this table is NOT a list of
> closed levers.** Every row below was measured with `HEADING_P = 4.5`, i.e.
> through a relay loop that saturated ~88 % of the time. Two of the six have
> since been re-tested at `HEADING_P = 0.6` and BOTH reversed:
>
> | lever | verdict here | verdict with the loop fixed |
> |:---|:---|:---|
> | depower | 0.36 "diverges at 21.7 s", low depower unsurvivable | 0.36 crosses the centre and flies to 37.6 s; **0.40 flies the full 150 s and 4 laps** |
> | `ATTRACTOR_DIST` | 35.1 optimal (2026-08-01 sweep) | monotonic the OTHER way — **12.1**, with a survival cliff above ~17 |
>
> A lever tested through a saturated controller was never tested. The remaining
> four rows — `HEADING_P` itself, entry course, entry timing, pattern size —
> should be treated as UNKNOWN until re-run, not as closed. The `HEADING_P` row
> is actively misleading: "x2.25 gives pure bang-bang" was the loop still being
> ~4x over gain, and the fix was x7.5 in the same direction.

Six levers were tried and closed, and every one failed the same way: the
steering COMMAND sits at exactly ±0.300 through the whole entry and most of the
pattern, so the controller is effectively open loop — see the corrections above
for why the command gets there, and why that invalidates the table.

| lever | tried | result |
|:---|:---|:---|
| `HEADING_P` | 5.0 -> 4.5 -> 2.0 | -10% within noise; x2.25 gives pure bang-bang (steering HF std 0.0000), triples the force ripple, costs 3.3° of elevation floor for 0.4° of RMS. Reverted to 4.5. |
| entry course | `CHI_DIVE` -100 -> -85 | **bit-identical run.** Steering pinned at 0.300 for the whole dive, so only the SIGN of the command ever reaches the plant. |
| entry timing | `DIVE_EL_MARGIN` 15 -> 5 | worse: handover azimuth goes 17.4° -> 7.7° and the centre crossing is lost. The kite flies down-LEFT while commanded horizontally right — it never reaches the commanded course at all. |
| depower | 0.40 -> 0.36 | **diverges at 21.7 s**, v_app 93.6 m/s, 6164 N, lift 11 kN with negative drag. The winch paid out MORE than at 0.40 (13.4 m vs 10.6 m) and it made no difference: reel-out at ~2 m/s does not make a low depower survivable at 200 m. PlanFig8.md's standing note on this is not supported. |
| pattern size | 50x20 -> 40x15 | **wrong direction, and `check_pattern_feasible` said so before the run**: a smaller lemniscate is a TIGHTER one. Margin 1.33 -> 1.02, aborted at 19.1 s. The reference flies 40x15 because its ram-air kite turns several times harder. |
| fig8 feedback signal | `FIG8_PURE_COURSE = true` (2026-08-02) | **bit-identical run**, second instance of the `CHI_DIVE` effect. Correctly applied (`var_08` == 1 for every phase-3 sample), but the only window where it differs from the speed schedule is 16.98 .. 19.15 s — the first 2.2 s of phase 3, where the old blend fell to `w = 0` (pure heading) — and `set_steering` is at **-0.300 on all 130 samples** of it. Saturated output, same plant input, same run. After 19.15 s `\|v_kite\|` never drops below 9.99 m/s, so the schedule would have asked for course anyway. |

The geometry behind all of it: `rho = 1/(L*c1*u_s)`, so the margin improves with
a LONGER tether or a BIGGER pattern, not a smaller one. Note what the correction
above does to this argument: `check_pattern_feasible` computes the margin from
`MAX_STEERING`, i.e. from the COMMAND, and the tape delivers an RMS of 0.209
instead. The feasibility margins quoted throughout this plan (1.19, 1.61, ...)
are therefore optimistic by roughly the same 27 % — a run whose printed margin
is comfortably above 1 can still be curvature-limited in practice. The geometry
conclusion (longer tether, bigger pattern) is unchanged; only its safety factor
is smaller than advertised.

## Done — AoA/LD instrumentation (2026-08-02)

Closes the follow-up under "Open — next steps" item 0. The AoA that was plotted
described one panel; now the whole wing is logged and both are shown side by
side.

- `span_mean_aoa(sys)` in `src/sim_helpers.jl` (exported), next to
  `compute_kite_aoa`/`compute_wing_incidence`. Mean of the VSM's per-panel
  `alpha_geometric_dist`, skipping `NaN` panels and applying the same ±π wrap
  `update_sys_state!` applies to the centre panel — the VSM returns some panels
  as `pi + atan(...)`, which would otherwise drag the mean. `NaN` for a wing
  with no VSM solver rather than an error.
- `simple_fig8.jl` logs it to **`var_09`** in degrees (slot table updated).
- `simple_fig8_plots.jl` gains a THIRD figure: α (centre panel vs span mean),
  L/D (`var_15` wing vs `var_16` effective), and `v_app` underneath. Its own
  figure, not two more rows on the six-row time-series stack.
- Span-mean was chosen over `compute_wing_incidence` because the incidence is a
  mid-chord geometric angle from struts 3/4 — it shares the centre panel's blind
  spot in a turn, which is exactly the case this instrumentation is for.
- **VERIFIED 2026-08-02** on the blend run above: `var_09` filled on all 1717
  flying rows, no `NaN`s, three figures displayed. (Any log written before this
  change plots the span-mean curve as a flat zero — re-simulate, don't replot.)

It earns its place immediately. Centre panel vs span mean, same run:

| window | centre panel (`AoA`) | span mean (`var_09`) |
|:---|---:|---:|
| established flight, 15 .. 27 s | -1.37 .. **0.38** .. +1.88° | +2.17 .. **2.94** .. +3.36° |
| blowup, t >= 28 s | -50.3 .. +49.5° | -2.2 .. +6.1° |

Median L/D in flight: 5.98 wing, 3.13 effective — consistent with item 0 above.

Two consequences:

- "the AoA is mostly negative" is now measured, not inferred: the centre panel
  straddles zero while the whole wing sits at a healthy **+3°**.
- **the overspeed guard's abort message is misleading.** It prints
  `sys_state.AoA`, so the 2026-08-02 abort read `AoA = -45.2°` while the span
  mean never left -2 .. +6°. It points at a stall that did not happen; the real
  event is v_app 47 m/s at 1096 °/s of turn rate. Worth switching that message
  to the span mean (not done — one change per run).

## Done — plotting toolchain (2026-08-02)

`examples/Project.toml` now pins `MakieControlPlots = "0.1.12"`. 0.1.9's
legend-height probe throws `only(owned)` (`ArgumentError: Collection has
multiple elements`) when displaying a `plotx` figure with a legend on more than
one row — i.e. every `*_plots.jl` script here. That is why the first 90 s run of
this campaign produced the pattern figure but no time series.

## Open — next steps

**Status: the pattern flies (see the top of this file). These are refinements,
not blockers.**

1. **Re-run the four untested "closed" levers** — entry course, entry timing,
   pattern size, and `HEADING_P` around 0.6 — under the fixed loop. Two of the
   six reversed on re-test; there is no reason to trust the other four.
2. **Extend the `ATTRACTOR_DIST` sweep below 10.** RMS d was still falling
   monotonically at the bottom edge of the 10 .. 20 range (2.85° at 10 vs 2.90°
   at the chosen 12.1), so the optimum is outside what has been measured.
3. **Restore `SIM_TIME = 30`** for routine tuning. It is left at 150 s from the
   lap-counting runs; the comment block in `simple_fig8.jl` says so.
4. `MAX_STEERING` DOWN rather than up, and a rate-aware command shaper, are the
   remaining ideas aimed at the tape rate itself. Neither is needed now that the
   tape sits at 2 % rate saturation — keep them for when the pattern is pushed
   harder (lower `EL_CENTER`, bigger `F8_A`).
5. Switch the overspeed guard's abort message from `sys_state.AoA` (centre
   panel) to `span_mean_aoa` — see the AoA section below for why it currently
   reports a stall that did not happen.

Still worth confirming, though no longer urgent: **`v_steering = 0.2` s⁻¹
against the real V3 KCU.** The controller now respects it with margin, but if
the real tape is faster, the 0.6 gain is more conservative than it needs to be.

~~then attack the demanded steering RATE~~ **DONE 2026-08-02** — `HEADING_P`
4.5 -> 0.6 put the steering cycle at 0.103 Hz (9.7 s), within 3 % of the
reference controller's 10 s target, with 99.6 % amplitude retention and the tape
lag halved to 29.8°.
- ~~add the honest saturation metric to `print_fig8_metrics`~~ **DONE
  2026-08-02.** `fig8_metrics` now returns `max_steering_delivered`,
  `tape_rate_frac`, `max_tape_rate` and the `v_steering` it scored against, and
  `print_fig8_metrics` prints a second steering line:

      tape: delivered peak |u_s|=0.276 of 0.300 commanded | RATE-limited 91% of
      the time (peak 0.200/s of 0.200/s)

  `v_steering` is a KEYWORD (default 0.2), not a constant, so a log from a
  differently configured KCU still scores correctly — which matters immediately,
  because the reference log's actuator is nothing like the V3's.

### The working reference proves the loop is ~1.8x too fast (2026-08-02)

Measured on `data/fig8_reference.arrow`, steady state t = 60 .. 120 s, against
phase 3 of the 2026-08-02 V3 run. Both are figure-eight flight; the reference is
the controller that works.

| | V3 `fig8_run` | reference |
|:---|---:|---:|
| dominant steering frequency | 0.181 Hz (**5.5 s**) | 0.100 Hz (**10.0 s**) |
| command fundamental amplitude | 0.299 | 0.502 |
| delivered fundamental amplitude | 0.183 | 0.496 |
| amplitude retained | 61 % | **99 %** |
| max tape slew observed | 0.200 /s (= the limit) | **1.736 /s** |
| peak \|u_s\| commanded -> delivered | 0.300 -> 0.276 | 1.000 -> 1.013 |

The reference actuator is at least **8.7x faster** than `v_steering = 0.2`, uses
the full ±1 range, and loses essentially nothing between command and tape. Its
controller is not fighting its actuator at all.

**Correction to an earlier estimate in this plan.** The claim that the pattern
only needs ~0.023 Hz (two reversals per 43 s lap), so the actuator has ample
margin, was inferred from geometry and is WRONG. A real working controller
cycles its steering at 0.1 Hz, several times per lap, because it is a feedback
loop and not a bang-bang per lobe. What the V3 tape can deliver:

- at **0.1 Hz** (the reference's rate): amplitude **0.32**, just above
  `MAX_STEERING = 0.30` — adequate, with ~7 % margin and nothing to spare;
- at **0.181 Hz** (what our loop demands): amplitude **0.176**, barely half the
  command.

So the V3 actuator is not hopeless, but it is only just sufficient, and only if
the loop is slowed to roughly the reference's 10 s cycle. Caveat: different kite
and different `steering_gain`, so the AMPLITUDES are not comparable across the
two logs; the frequency and the actuator speed are.

0. ~~Investigate the AoA and the L/D. The AoA is most of the time negative, the
   L/D is about 40, very unrealistic.~~ **DONE 2026-08-01 — both are reporting
   artefacts, the aerodynamics are fine.** Measured on a healthy parked state
   (depower 0.40, 200 m, v_app 8.9 m/s):

   | quantity | value |
   |:---|---:|
   | `SysState.AoA` (what is plotted) | **-0.86°** |
   | span-mean geometric alpha | +3.05° |
   | span-mean EFFECTIVE alpha (with induction) | +1.22° |
   | body-frame AoA (`compute_kite_aoa`) | +10.58° |
   | incidence vs mid-chord (`compute_wing_incidence`) | +4.18° |
   | L/D wing (`var_15`) | 5.78 |
   | L/D effective (`var_16`) | 2.58 |

   - **L/D "about 40" is a slot mismatch.** `fig_eight_plots.jl` plots
     `var_02`/`var_03` as L/D_wing/L/D_eff — that is `test_figure_eight.jl`'s
     slot map. `simple_fig8.jl` writes ATTRACTOR AZIMUTH and ELEVATION there
     (medians 44.9° and 48.4° in the current log — hence "about 40"); its L/D
     goes to `var_15`/`var_16`, where it reads 6.17 / 3.18 in flight. Realistic
     for a V3. The two plotting scripts are not interchangeable.
   - **Negative AoA is a definition artefact.** For a `VSMWing`,
     `update_sys_state!` sets `AoA` to the geometric alpha of ONE panel — the
     centre one (`alpha_geometric_dist[mid]`, wrapped to ±180°). The depower
     tape acts at the centre, so that panel sits ~4° below span-mean and ~14°
     below the tips (tips +13.3°, mid -0.9°). The EFFECTIVE alpha is positive
     across the whole span (0.3 .. 2.9°) — the wing is not flying at negative
     incidence.
   - ~~Follow-ups worth doing: give `simple_fig8_plots.jl` its own AoA/L/D panels
     reading `var_15`/`var_16`, and log a span-mean or `compute_wing_incidence`
     AoA to a spare slot so the plotted angle means something for the whole
     wing.~~ **DONE 2026-08-02, see "Done — AoA/LD instrumentation" below.**
   - Caveat: the log inspected was a depower-0.36 run that diverged at 21.7 s,
     so its in-flight extremes (`L/D` up to 131, AoA down to -20°) are the
     divergence, not normal operation.

1. **A V3 log that flies the eight** (offered 2026-08-01, not yet supplied).
   The current reference is a smaller, far more agile ram-air kite, so it cannot
   answer whether the crossover is achievable on this plant at all. From a V3
   log, read off `l_tether`, `depower`, the fraction of time `set_steering` is
   clamped, the flown pattern size and centre elevation, and — if it covers the
   entry — how the kite gets established on the LEFT lobe. Save it as
   `data/v3_fig8_reference.arrow` (NOT `fig8_run`). If it is a measured flight
   rather than a simulation, a gap would point at `c1` and the turn-rate
   identification rather than at the controller.
2. **300 m tether** — the last untried lever that attacks the authority limit
   itself rather than working around it (`rho = 1/(L*c1*u_s)`; 150 -> 200 m was
   made for exactly this reason).
3. re-run the `EL_CENTER` sweep under force mode — 45 and 40.5 both survive now,
   and the old "50° is the floor" result is a fixed-winch artefact. 40.5 gives
   RMS d 5.27° but ZERO centre crossings; the pattern drifts right as the centre
   drops.
4. fix `SysState.course` centrally instead of in the example: `calc_course` in
   `src/interface.jl` and the `:course` correction mode in `src/stabilization.jl`
   both consume the raw field
5. the settled-geometry cache key ignores the settling elevation, so
   `ELEVATION` in `simple_fig8.jl` has no effect and every run starts at the 73°
   park (see `src/stabilization.jl`, PlanFig8.md Findings 4)

- Cut HEADING_P to ~0.6. **DONE** One number, derived above, and the only change that takes the loop out of relay mode. Everything else is cosmetic until this is done.
- Then reconsider HEADING_D = 0.15. Derivative action on a signal delayed 0.72 s is destabilising; it is currently irrelevant (saturated), but it will matter once the loop is linear.
- MAX_STEERING down to ~0.20. Raises the tape's follow-able frequency from 0.106 to 0.16 Hz and shrinks the relay amplitude — the opposite of the closed "raise the authority" option.
- Anti-windup against the real tape position. sys_state.steering is available every step, so the PID state can be back-calculated from what the actuator actually did instead of what was commanded. This is the principled fix for rate-limiter-induced oscillation, and worth doing after 1.
