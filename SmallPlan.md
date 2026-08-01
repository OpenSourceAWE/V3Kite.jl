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
  still). Pure course at/above 10 m/s.
- fixed a 180° convention error: `SysState.course` is 180° away from
  `SysState.heading` — corrected locally in `simple_fig8.jl` for now
- `ATTRACTOR_DIST` swept 16.2 .. 38.6 to minimize the RMS course error;
  35.1 is the minimum in both windows
- down-loops (`UP_LOOPS = false`) survive on the course-feedback loop, and at
  `ATTRACTOR_DIST = 35.1` are the first configuration on record that crosses
  the pattern centre instead of circling one lobe

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

## Open — next steps

- get log file from working controller and plot it. Needs: which controller,
  where the log is, and what to compare against (course tracking? pattern
  geometry? steering activity?)
- steering is clamped 77-97% of the time in EVERY configuration tried — the
  binding constraint. The listed levers are reel-out (restores c1 = 0.3159 and
  0.03 s dead time by making a low depower survivable) or a 300 m tether.
- fix `SysState.course` centrally instead of in the example: `calc_course` in
  `src/interface.jl` and the `:course` correction mode in `src/stabilization.jl`
  both consume the raw field
- the settled-geometry cache key ignores the settling elevation, so
  `ELEVATION` in `simple_fig8.jl` has no effect and every run starts at the 73°
  park (see `src/stabilization.jl`, PlanFig8.md Findings 4)
