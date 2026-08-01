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
  The "50° is the floor" sweep is an artefact of the fixed-length winch and
  should be re-run.

## Open — the steering gain is not the lever

`HEADING_P` was cut -10% (5.0 -> 4.5: nothing changed, RMS d 4.97 -> 4.93°) and
then by a factor 2.25 (4.5 -> 2.0). The command never left its ±0.30 limit; at
2.0 it became pure bang-bang (steering HF std exactly 0.0000), which cost a
tripling of the force ripple and 3.3° of elevation floor for 0.4° of RMS. The
loop is authority-limited, so scaling what it asks for changes only HOW it
saturates. Reverted to 4.5. Relieve the saturation instead — which is what the
lower centre above does.

## Open — next steps

Derived from the reference run, in priority order:

- `EL_CENTER = 50` is far too high — the working eight is centred at **26°** and
  dips to 8.7°. The low centre is what buys apparent wind and hence turn-rate
  authority, and is the direct answer to the clamping item below. Walk it down
  in 10 % steps.
- add a park/dive/hold entry phase instead of doing the job inside the guidance
  loop with `ENTRY_CHI_MAX`: fixed ~+98° course for ~6 s, flatten, then hand
  over at a *specific* point on the pattern (rightmost, centre elevation,
  already turning down). A 30 s run splits as ~5 s park + ~7 s dive + ~18 s
  pattern, so it fits under the cap.
- pattern shape: reference is `A = 40, B = 15` (aspect 2.7) vs the current
  `F8_A = 50, F8_B = 20`.

- steering is clamped 77-97% of the time in EVERY configuration tried — the
  binding constraint. The listed levers are reel-out (restores c1 = 0.3159 and
  0.03 s dead time by making a low depower survivable) or a 300 m tether.
- fix `SysState.course` centrally instead of in the example: `calc_course` in
  `src/interface.jl` and the `:course` correction mode in `src/stabilization.jl`
  both consume the raw field
- the settled-geometry cache key ignores the settling elevation, so
  `ELEVATION` in `simple_fig8.jl` has no effect and every run starts at the 73°
  park (see `src/stabilization.jl`, PlanFig8.md Findings 4)
