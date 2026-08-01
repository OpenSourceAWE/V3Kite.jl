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
- `ATTRACTOR_DIST` swept 16.2 .. 38.6 to minimize the RMS course error;
  35.1 is the minimum in both windows
- down-loops (`UP_LOOPS = false`) survive on the course-feedback loop, and at
  `ATTRACTOR_DIST = 35.1` are the first configuration on record that crosses
  the pattern centre instead of circling one lobe
- plotted the working controller's log (`data/fig8_reference.arrow`, 120 s, the
  4-phase state machine — not a `simple_fig8.jl` run) with
  `include("examples/fig_eight_plots.jl")`; see "Reference run" below

## Open — measurement

The campaign currently has no working measure of "flies a figure eight":

- `laps` has been 0.0 in EVERY run recorded so far, including the 200 s ones,
  and `laps >= 3.0` needs ~130 s of flight — unreachable under the 30 s cap.
  So every run prints "Success criteria FAILED" regardless of quality.
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

## Open — the heading/course blend is a regression (2026-08-01)

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
flying speed range, or gate on phase instead of speed. Currently the example is
left with the blend DISABLED.

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

## Closed levers — all authority-limited (2026-08-01)

Five levers have now each been tried and closed, and every one failed the same
way: the steering command sits at exactly ±0.300 through the whole entry and
most of the pattern, so the plant is at its turn limit and the controller is
effectively open loop.

| lever | tried | result |
|:---|:---|:---|
| `HEADING_P` | 5.0 -> 4.5 -> 2.0 | -10% within noise; x2.25 gives pure bang-bang (steering HF std 0.0000), triples the force ripple, costs 3.3° of elevation floor for 0.4° of RMS. Reverted to 4.5. |
| entry course | `CHI_DIVE` -100 -> -85 | **bit-identical run.** Steering pinned at 0.300 for the whole dive, so only the SIGN of the command ever reaches the plant. |
| entry timing | `DIVE_EL_MARGIN` 15 -> 5 | worse: handover azimuth goes 17.4° -> 7.7° and the centre crossing is lost. The kite flies down-LEFT while commanded horizontally right — it never reaches the commanded course at all. |
| depower | 0.40 -> 0.36 | **diverges at 21.7 s**, v_app 93.6 m/s, 6164 N, lift 11 kN with negative drag. The winch paid out MORE than at 0.40 (13.4 m vs 10.6 m) and it made no difference: reel-out at ~2 m/s does not make a low depower survivable at 200 m. PlanFig8.md's standing note on this is not supported. |
| pattern size | 50x20 -> 40x15 | **wrong direction, and `check_pattern_feasible` said so before the run**: a smaller lemniscate is a TIGHTER one. Margin 1.33 -> 1.02, aborted at 19.1 s. The reference flies 40x15 because its ram-air kite turns several times harder. |

The geometry behind all of it: `rho = 1/(L*c1*u_s)`, so the margin improves with
a LONGER tether or a BIGGER pattern, not a smaller one.

## Open — next steps

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
   - Follow-ups worth doing: give `simple_fig8_plots.jl` its own AoA/L/D panels
     reading `var_15`/`var_16`, and log a span-mean or `compute_wing_incidence`
     AoA to a spare slot so the plotted angle means something for the whole wing.
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
