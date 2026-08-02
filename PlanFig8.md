# Add a figure-eight path-following controller to V3Kite.jl

## Goal

Fly the V3 kite on a repeatable lemniscate pattern in the (azimuth, elevation)
plane, at constant tether length, using the L0 attractor-point guidance already
implemented and tuned for another kite.

## Scope

**In scope:** constant tether length (winch in position mode), fixed depower,
single `rel_steering` channel, guidance + heading PID, metrics, plots, unit
tests.

**Out of scope for this plan:** reel-out / power production (pumping cycles),
depower feedback loops, RIGID_DYNAMICS variant. Add these only after the
constant-length pattern is stable.

## What gets built, and how much is inherited

| V3Kite file | Origin and notes |
|:---|:---|
| `src/fig8_controller.jl` | **Adopted nearly verbatim** from the earlier guidance implementation. Pure spherical geometry, wing-agnostic; its only dependency is `KiteUtils.wrap2pi`. Keep `up_loops` and `set_path_center!` — they were the two highest-leverage knobs in the earlier tuning. |
| `test/test_fig8_controller.jl` | **Adopted:** the 5 unit tests that shipped alongside that guidance code, plus the V3-specific turn-radius feasibility tests. `include` from [test/runtests.jl](test/runtests.jl) next to `test_turn_rate_id.jl`. Pure geometry, fast. |
| `src/fig8_metrics.jl` | **Adapted** to the V3 log layout. `sl.winch_force[.,1]` carries over unchanged (SysState's `winch_force` is `MVector{4}` for every model; V3 uses slot 1). `sl.set_steering` was *meters* there and is dimensionless `rel_steering` here — the HF-content threshold has to be re-scaled. `sys_state==3` and `var_01` are conventions this code must set deliberately. |
| — (`descent_diagnostics`) | **Not carried over.** It diagnoses the other kite's roll spiral through a dive we are not flying. |
| `examples/simple_fig8.jl` | **Written fresh**, ~150 lines on `init`/`step!`. The earlier run script was 1389 lines across two wing branches, welded to MixerControl / `mc_params` / 4-winch mixing / front+back channel assist — a source of tuning history, not of code. |
| `examples/simple_fig8_plots.jl` | **Written fresh** against V3Kite's log layout. |

### Deliberately left out

Every item below is a fix for a failure mode of that other, different airframe.
Start without them; add back only if the V3 demonstrably shows the same problem.

- `CHI_RATE` course slew limiter (P1)
- `K_R` heading-rate damping, `K_ROLL` roll-rate damping (P2/P3)
- the AoA-feedback depower loop and its `TT_AOA` back-calculation anti-windup
- `LR_TO_BACK_RATIO` / back-channel assist (the V3 has one steering channel)
- the `park → dive → hold → engage` state machine (see STEP 2)

## Correcting the original premise

The first draft of this plan assumed the V3 needs no transition logic "because
of the short delay of the steering system". Two corrections:

1. **The V3 steering delay is not negligible.** `v_steering: 0.2` [1/s] in
   [data/settings.yaml:34](data/settings.yaml#L34) is a KCU tape rate limit:
   reaching the `MAX_STEERING = 0.175` used in the examples takes ~0.9 s. That
   is the same order as the rate limits the earlier campaign had to tune around.
   The difference is that the KCU *is* the rate limiter — we don't add our own.
2. **The earlier transition machine is not about the actuator.** It exists
   because that kite, diving from a 75° park to a 35° pattern center, departs into
   a roll spiral. That is an airframe problem and does not disappear with faster
   steering. What lets us skip it here is the *entry strategy* in STEP 2, not
   the steering bandwidth.

## Findings that changed the design (2026-07-26, during implementation)

Two measured results invalidated parts of the plan as originally written. Both
are now encoded in `src/fig8_controller.jl` and `examples/simple_fig8.jl`.

**1. A figure-eight near zenith is geometrically impossible for this kite.**
The identified turn-rate law fixes the smallest angular turn radius the kite can
fly: `ρ = 1/(L·c1·u_s)` — the apparent wind speed cancels, so it depends only on
tether length and steering authority. The tightest curvature of a lemniscate is
*not* the lobe tip (`B²/(A·cos el)`) but the lobe's upper shoulder, and it
collapses as the pattern is raised, because the azimuth axis is compressed by
`cos(elevation)`. At `c1 = 0.3159`, `u_s = 0.175`, `L = 150` m (kite minimum
radius 6.9°):

| pattern | tightest path radius | margin |
|:---|---:|---:|
| A=45 B=20, centre 73° (the planned capture elevation) | 0.4° | **0.06** |
| A=45 B=20, centre 60° | 3.1° | 0.45 |
| A=45 B=20, centre 35° | 8.5° | 1.23 |
| A=50 B=25, centre 30° | 10.5° | **1.52** ← chosen |

So **STEP 2's "capture high at the park elevation, then walk down" is void** —
that is the single least flyable place to put the pattern. The pattern must be
low and wide, and the kite descends onto it under guidance instead. The walk
mechanism (STEP 4) survives, but as a way to move between two *low* centres, not
as an entry strategy.

**2. `body_damping` is a flight parameter, not a solver setting.** It changes
the steering coefficient `c1` — and hence the achievable turn radius — by 5.6×:

| `body_damping` | `c1` [1/m] | min turn radius at `u_s`=0.175, L=150 m |
|:---|---:|---:|
| `[0, 0, 40]` (`init`'s default) | 0.3159 | 6.9° |
| `[10, 10, 40]` | 0.0982 | 22.2° |
| `[20, 20, 40]` | 0.0567 | 38.5° |

At `[10,10,40]` the chosen pattern would need `u_s = 0.5` to reach a comparable
margin, which extrapolates the turn-rate law far beyond the `|u_s| ≤ 0.175` it
was identified over. The example therefore uses `[0, 0, 40]` and looks `c1` up
from `V3_TURN_RATE_COEFFS` keyed on the damping in use; `turn_rate_coeffs`
throws rather than guessing for an unlisted damping.

**3. The entry from the park needs energy management after all — the first run
diverged.** `include("examples/simple_fig8.jl")` aborted at t=7.35 s with a
solver `dt_epsilon` error. The guidance was working (cross-track error 35° →
1.7°, the kite flew onto the path); the *descent* was the problem:

| t [s] | elevation | v_app [m/s] | AoA |
|---:|---:|---:|---:|
| 0.0 | 73.0° | 15.7 | 4.6° |
| 5.98 | 53.6° | 28.7 | 3.8° |
| 6.98 | 38.9° | 40.9 | −5.8° |
| 7.33 | 32.8° | 51.3 | −92.9° (diverged) |

A 40° dive converted potential energy into a 3.3× overspeed; the wing unloaded
to negative AoA and the model broke. A second defect appeared in the same run:
with the attractor nearly straight below, `chi_set` hunted across the ±180°
branch cut (+154.8° → −155.2° → −153.3°) and the steering chattered between its
clamps.

So this plan's claim that the V3 needs no transition logic was **wrong**. It is
wrong for a different reason than the earlier one — an energy problem, not that
kite's roll spiral — but the descent does need managing. The fix added to
`simple_fig8.jl` limits the commanded *course* rather than its rate: while
`d > ENTRY_D_GATE`, never command a course steeper than `ENTRY_CHI_MAX = 105°`
(90° = constant elevation), so the approach is a shallow glide drag can bleed.
Choosing whichever of `±ENTRY_CHI_MAX` needs the smaller heading change also
removes the branch-cut chatter; the choice is latched with 30° of hysteresis
because at the parked heading (~0°) the two candidates are equidistant and a
bare nearest-candidate rule flips on noise (verified by replaying the diverged
log: 1 sign flip with hysteresis, chatter without). The limiter self-disables on
the path, where the pattern legitimately needs steep courses.

An overspeed guard (`V_APP_ABORT`) now stops the run with the actual cause
instead of an opaque solver abort.

**4. The settled-geometry cache key does not include the settling elevation.**
Found while trying to start the kite on the pattern instead of at the park:
setting `init(...; elevation = 33.0)` silently had no effect — the run still
started at 73.0°, because `settle_wing` loaded
`settled_dp1.45_…_vapp9.51_lt150_g98_syssystem_reelout_kcu84_bd0-0-40.bin`,
whose name encodes depower, steering, tip/TE, wind, tether length, gravity,
system and body damping — **but not elevation**. `init` passes the elevation
only through the `position` kwarg of `settle_wing`, which is not part of the
config the cache name is built from.

This is a latent trap for any example that varies the settling elevation: two
different elevations share one cache entry, and `remake = true` would overwrite
the shared file with geometry the other examples do not expect. Worth fixing at
the source (add the elevation to the cache key) rather than working around it —
not done here, since it changes cache names for every existing example.

*Fixed since:* `settle_wing` now appends `_el<deg>` (from the `init_row`
position) to the cache name, so every existing settled file re-settles once.

**5. BLOCKER: the model cannot sustain a continuous turn in this configuration.**
Established open-loop, with no controller in the loop at all — constant
`rel_steering` applied from the park at `system_reelout.yaml`, 150 m tether,
9.51 m/s wind, depower 0.25, `body_damping = [0,0,40]`:

| `u_s` | survived | peak v_app | peak \|AoA\| |
|---:|---:|---:|---:|
| 0.10 | 13.4 s | 51.6 m/s | 38.9° |
| 0.175 | 10.9 s | 39.0 m/s | 36.9° |
| 0.30 | 10.0 s | 31.8 m/s | 37.5° |

Every case ends in a solver divergence after the kite spirals up in speed and
the AoA spikes to ~37-39°. `simple_sinus.jl` survives at this configuration only
because it *oscillates* the heading and never holds a turn.

**A figure-of-eight is made of sustained turns, so no amount of guidance or PID
tuning can succeed until this is resolved.** All nine closed-loop runs died the
same way (10-16 s, AoA spike at 25-32 m/s) — that was the plant, not the
controller. This invalidates the plan's implicit assumption that the model would
fly the pattern once the controller was right.

**RESOLVED by depowering.** Same open-loop test at `u_s = 0.10`:

| depower | survived | peak v_app | peak \|AoA\| |
|---:|---:|---:|---:|
| 0.25 | 13.4 s | 51.6 m/s | 38.9° |
| 0.40 | 29.2 s | 36.3 m/s | 44.0° |
| **0.55** | **40 s ✓** | **21.6 m/s** | **3.2°** |

Depowering caps the crosswind speed and keeps AoA in a sane band. Note that
*less* steering is worse (`u_s = 0.05` at depower 0.25 diverged at 17.1 s with
v_app 66 m/s) — a wider turn spends longer accelerating. `body_damping =
[10,10,40]` did not help (21.8 s).

**6. Current state: the model flies, the controller does not yet track.** At
depower 0.55 the full 200 s run completes with no divergence — but the kite
settles into a stable *circle* at azimuth 29-45°, elevation 37-47°, never
crossing the pattern centre and sitting below the commanded band (45-65°), with
the steering bang-banging between ±0.30 (36% at +clamp, 61% at −clamp). So the
plant is now healthy and the remaining problem is genuinely control/tuning:

- steering saturated 97% of the settled window — the clamp is binding, and now
  that the plant survives, raising `MAX_STEERING` is a legitimate lever again;
- `c1` was identified at depower 0.25, but the run flies at 0.55.

**7. Re-identification at depower 0.55 explains the circle — the pattern is
2× too tight.** `steering_test_v3.jl` re-run at depower 0.55 with
`body_damping = [0,0,40]`, sweep extended to `u_s = 0.300` (13481 samples, all
three PASS criteria met):

| | depower 0.25 | depower 0.55 |
|:---|---:|---:|
| `c1` [1/m] | 0.3159 ± 0.09% | **0.1071 ± 0.07%** |
| `c2` [-] | −0.3837 | +0.3444 |
| steering dead time | 0.03 s | **0.55 s** |

**Depower is a control-authority parameter, not just a trim.** 0.25 → 0.55 costs
a factor 2.95 of steering authority and raises the dead time 16×. Consequences
for the run in Finding 6:

| pattern A=50 B=20 centre 55°, L=150 m, u_s=0.30 | kite turn radius | margin |
|:---|---:|---:|
| with the assumed `c1` = 0.3159 | 4.0° | 1.41 ← what the run printed |
| with the **actual** `c1` = 0.1071 | 11.9° | **0.48** ← reality |

So the kite flew a circle because it physically could not turn tightly enough —
the pattern was 2× beyond its turn radius. This was not a tuning failure, and no
gain would have fixed it. `V3_TURN_RATE_COEFFS` is now keyed on
`(body_damping, depower)` and `turn_rate_coeffs` throws for an unidentified
combination, so this class of error cannot recur silently.

**The V3 is caught in a bind at 150 m tether:** depower 0.25 is agile
(`c1` = 0.3159) but cannot survive a sustained turn; depower 0.55 survives but is
too sluggish to fly the pattern. Escaping it needs a longer tether and/or more
steering — `ρ = 1/(L·c1·u_s)` improves with both:

| tether | `u_s` | kite radius | margin |
|---:|---:|---:|---:|
| 150 m | 0.30 | 11.9° | 0.48 |
| 150 m | 0.50 | 7.1° | 0.80 |
| 250 m | 0.30 | 7.1° | 0.80 |
| 300 m | 0.30 | 5.9° | 0.96 |
| **300 m** | **0.50** | **3.6°** | **1.59 ✓** |

A 300 m tether at `u_s = 0.50` is the first configuration with real margin, and
is also closer to the reel-out examples' operating point.

**8. Depower 0.40 + 200 m tether (user-requested): survives, still circles.**
`c1` re-identified at that configuration: **0.1513 ± 0.06%**, `c2` = 0.1951,
dead time 0.42 s (G scatter 9.8%, all PASS) — neatly between the 0.25 and 0.55
rows, confirming the monotonic trend.

A real guidance bug was found and fixed on the way. At the lemniscate
self-intersection the closest point Q jumped to the *far branch*, flipping the
commanded course by ~188° (90.6° → −97.5°) and slamming the steering across;
that alone diverged a run at t = 13.3 s. `calc_attractor` now constrains Q to a
window of arc around its previous value (`search_window`), with automatic global
re-acquisition beyond `reacquire_dist`. This is a genuine improvement to the
guidance, not a workaround, and it took the run from 13.3 s to the full 200 s.

Current result at depower 0.40 / 200 m / centre 50° (margin 1.19): the full 200 s
completes, the orbit is stable and repeatable, tether force ~2000 N (CV 13%) —
**but the kite still flies a closed loop around the LEFT LOBE only** (azimuth
−47…−24°, never crossing the centre). Genuine laps: **0**. Steering is saturated
97% of the time at `u_s = 0.30`, so authority is still the binding constraint:
the kite can hold a ~10° circle but cannot make the lobe-to-lobe crossover.

A third lap-counting bug surfaced here: `print_fig8_metrics` accepted
`az_center` but did not *forward* it, so passing it from the example silently did
nothing and the run reported 18 phantom laps. All three lap-counting bugs now
have regression tests.

**9. `ATTRACTOR_DIST` is not the lever — swept in 10% steps, conclusively.**
Depower 0.40 / 200 m / centre 50°:

| lead | survived | laps | RMS d | min el | azimuth | saturation |
|---:|---:|---:|---:|---:|:---|---:|
| 7° | diverged | — | — | — | — | — |
| 10° | 13.6 s | — | — | — | — | — |
| 14.6° | 200 s | 0 | 6.34° | 29.3° | −48…−27° | 97% |
| 16.2° | 200 s | 0 | 5.94° | 29.8° | −47…−26° | 97% |
| 18.0° | 200 s | 0 | 5.76° | 29.9° | −47…−24° | 97% |
| 19.8° | 200 s | 0 | 5.50° | 30.5° | −46…−23° | 97% |

Every surviving value circles the **left lobe** and never crosses the centre.
The lead changes only tracking quality, and monotonically in the *opposite*
direction to the hypothesis that a shorter lead would force the crossover:
longer lead gives lower RMS and a higher floor. Below ~14° the commanded course
rotates faster than the kite can follow given the **0.42 s steering dead time**
at this depower (10° died at 13.6 s after the course swung −154° → −45° in
2.5 s, with the heading lagging 85°).

**The 97% steering saturation in every case is the real constraint.** The
crossover is authority-limited, not guidance-limited, and no value of the lead
addresses that. `ATTRACTOR_DIST` is now 19.8 (best of the swept values); further
tuning of it is not worthwhile until the saturation is relieved.

**10. Raising `MAX_STEERING` does not work either — the authority ceiling is
below what the crossover needs.** Tested after re-identifying `c1` at the higher
amplitude (depower 0.40 / 200 m):

| `MAX_STEERING` | outcome |
|---:|:---|
| 0.30 | survives 200 s, 97% saturated, circles one lobe |
| 0.33 | **diverged at t = 30.9 s** — peak turn rate 949°/s, turn-rate HF 45.6°/s (vs 0.45°/s at 0.30): the loop goes violently unstable, not merely saturated |
| 0.375 | the **plant** diverges, in plain bang-bang oscillation with no controller at all |

`c1` is linear across the whole usable range (0.1495 up to `u_s` = 0.374 vs
0.1513 up to 0.300, both ≤0.15% standard error), so this is a genuine dynamic
limit rather than a modelling artefact or a bad feasibility estimate.

The consequence is important: **97% saturation at `u_s` = 0.30 is not a tuning
oversight — it is the plant's limit at this depower.** Options 3 (more
authority) and the `ATTRACTOR_DIST` sweep are both now closed, and neither
reached the lobe crossover.

**11. The pattern currently tracked** (A=50, B=20, centre 50°, tether 200 m):

| | angular | physical |
|:---|:---|:---|
| width | azimuth −50°…+50° = 100° (±`F8_A`) | 224 m |
| height | elevation 40°…60° = 20° (`F8_B`) | 70 m |
| path length | 161° of arc | 564 m per lap |
| tightest curvature | 7.5° | 26 m radius |

Aspect ratio 3.2:1 — a very flat pattern. The azimuth arc is compressed by
`cos(50°) = 0.64`. In the surviving runs the kite covers only ~22° of the 100°
width, so the crossover it cannot make is a ~50° azimuth traverse.

**12. Lowering `EL_CENTER` makes things monotonically worse, despite improving
the margin.** Swept in 10% steps (A=50, B=20, `u_s` = 0.30):

| `EL_CENTER` | elevation span | margin | survived |
|---:|:---|---:|---:|
| **50.0°** | 40–60° | 1.19 | **200 s ✓** |
| 45.0° | 35–55° | 1.30 | 18.4 s |
| 40.5° | 30–50° | 1.33 | 13.9 s |
| 36.5° | 26–46° | 1.35 | 13.7 s |

Lowering the centre eases the `cos(elevation)` compression and so improves the
curvature margin, but pushes the pattern deeper into the power zone — and the
energy problem binds first. **50° is the lowest survivable centre at this
depower/tether**, not a conservative choice. Lowering it requires solving the
energy problem first.

**13. Up-loops confirmed by measurement, not inheritance.** `UP_LOOPS = false`
diverges at 17.8 s against 200 s for `true`, all else equal. Up-loops shed energy
through the turn where down-loops convert height into speed, which matters
because the failure mode here is overspeed. This had been held fixed on the
earlier campaign's evidence for the whole session; it is now tested on the V3.

**14. Reducing `winch_pos_kp` does NOT produce reel-out — wrong knob.**

| `winch_pos_kp` | survived | tether length p-p | reel-out speed range |
|---:|---:|---:|---:|
| 0.5 (default) | 200 s | 0.01 m | 0.01 m/s |
| 0.05 | 200 s | 0.01 m | 0.01 m/s |
| 0.02 | 200 s | 0.01 m | 0.01 m/s |
| 0.01 | 200 s | 0.01 m | 0.01 m/s |

A 50× gain reduction changes the tether motion by nothing. The cascade in
`winch_position_torque!` uses `kp_pos` only to turn length error into a *speed
setpoint*; an inner speed PI (`winch_speed_k = 30`, `Ti = 2 s`) then drives the
drum to it. Lowering `kp_pos` drives `v_sp → 0`, so the inner loop enforces zero
speed **harder**. The length is pinned by the speed loop, and the outer gain
cannot release it.

Real reel-out needs the drum speed to respond to force. Two options that fit the
existing architecture:

1. **Torque mode** — `step!(s; set_torque = τ₀)` instead of `set_length`. The
   drum pays out when tether force exceeds τ₀ and hauls in when it drops. Mean
   length is held by a slow outer loop trimming τ₀ at a bandwidth well below the
   lap rate. Closest to a real pumping winch and reuses machinery already in
   `step!`.
2. **Force term in the speed setpoint** — keep the cascade but use
   `v_sp = kp_pos·(l₀ − l) + k_f·(F − F_ref)`.

Both are controller changes, not parameter tweaks. **Undecided — pick before
implementing.**

## Standing conclusion (2026-07-26)

The controller works: guidance, conventions, feasibility machinery and metrics
are implemented and unit-tested (50 tests), and the kite reliably descends onto
the pattern and holds a stable, repeatable orbit for the full 200 s. What it has
never done is the **lobe-to-lobe crossover** — it circles one lobe.

Every cheap lever has now been swept and closed:

| lever | result |
|:---|:---|
| `ATTRACTOR_DIST` 14.6–19.8 | all survive, all 0 laps, all 97% saturated |
| `MAX_STEERING` 0.33 | diverges (plant ceiling 0.375, measured open-loop) |
| `EL_CENTER` 45 / 40.5 / 36.5 | all diverge; 50° is the floor |
| `UP_LOOPS = false` | diverges at 17.8 s |
| `winch_pos_kp` 0.05–0.01 | no effect at all (wrong knob) |

The binding constraint is consistent across all of them: at depower 0.40 the
kite is at 97% steering saturation and cannot turn hard enough to cross the
centre, while any change that would relieve the geometry (lower centre, more
authority) hits an energy or stability limit first.

**Next step: reel-out**, via option 1 or 2 above — it is the only remaining lever
that changes the operating point rather than trading inside it, because bleeding
energy through the winch makes a *lower* depower survivable, which restores
`c1` = 0.3159 (3× the authority) and drops the dead time 0.42 s → 0.03 s.

Two lap-counter bugs were found and fixed along the way, both of which reported
success that had not happened: counting bare azimuth sign changes scored 42.5
laps for a kite stuck in a limit cycle, and counting against the *flown mean*
azimuth scored 14 laps for the circle above. Laps are now counted against the
pattern's own centre azimuth (`az_center`), which scores that run 0.

## Steps

### STEP 0 — Pin down the sign and frame conventions ✅ DONE

Answered from existing logs (`data/tmp_steering.arrow`, `data/tmp_sinus.arrow`)
without running a new simulation. Results are recorded in the convention block
at the top of [src/fig8_controller.jl](src/fig8_controller.jl):

- **Positive `rel_steering` → positive heading rate** (r = +0.998 between tape
  position and frame-transport-corrected turn rate). The PID output is fed to
  `rel_steering` **unnegated**; the other kite's negation does not transfer.
- **The guidance course convention matches `SysState.heading`** — same zero,
  same sign, circular-mean offset +13.3° with a 7.6° spread over samples where
  the kite actually flies. No `neg_azimuth`, no π correction anywhere. The +13°
  is the real course-minus-heading drift angle; it appears as a small steady
  cross-track bias that the guidance itself corrects.
- Turn-rate law identified: see the damping table above.

Original rationale, kept because it is why this step came first: a sign error
looks exactly like "the controller is unstable, needs tuning", and can burn a day.

Known asymmetries:

- [`calc_heading`](src/interface.jl#L295) applies a `+π` correction at
  [interface.jl:307](src/interface.jl#L307) (SymbolicAWEModels' body x-axis
  points opposite to the Xsens convention `KiteUtils.calc_heading` assumes).
- [`calc_azimuth`](src/interface.jl#L219) is positive **anti-clockwise seen from
  above**; `calc_azimuth_east` is positive clockwise. `calc_heading`/
  `calc_course` both take a `neg_azimuth` kwarg.
- The guidance assumes bearing `0` = towards zenith, positive towards
  **increasing** azimuth (`navigate_fig8`, and `_update_course!`'s
  `atan(d_az, d_el)`).
- [simple_auto_parking.jl:86](examples/simple_auto_parking.jl#L86) feeds the
  heading-PID output to `rel_steering` **unnegated**; the other kite's branch
  negates it.

**Deliverable:** a short open-loop run (extend
[examples/open_loop.jl](examples/open_loop.jl) or reuse
[examples/steering_test_v3.jl](examples/steering_test_v3.jl)) that answers, in
writing, at the top of the new controller file:

- sign of `d(sys_state.heading)/dt` for a positive `rel_steering` step;
- sign of `d(sys_state.azimuth)/dt` when the kite flies at `heading = +90°`;
- whether `sys_state.azimuth` as filled by `update_sys_state!` matches
  `calc_azimuth(s)` or its negation.

Only then wire up the loop.

### STEP 1 — Port the guidance module

`src/fig8_controller.jl`, `include`d from [src/V3Kite.jl](src/V3Kite.jl) after
`sim_helpers.jl`. Export `FigureEightSettings`, `FigureEightController`,
`figure_eight_path`, `calc_attractor`, `navigate_fig8`, `set_path_center!`.

Put it in `src/` (not `examples/`, as the original did) so
[test/runtests.jl](test/runtests.jl) can unit-test it, matching how
[src/turn_rate_id.jl](src/turn_rate_id.jl) is tested in
[test/test_turn_rate_id.jl](test/test_turn_rate_id.jl).

`FigureEightSettings.dt` drives the course low-pass and the speed estimate —
pass `s.dt` (examples use `0.05/3`), **not** the original default of 0.02.

Note: `calc_attractor` is O(n) over 361 path points, twice per step. Negligible
next to the DAE solve — do not "optimize" it.

**Verify:** `include("test/test_fig8_controller.jl")` — pure geometry, seconds.

### STEP 2 — Minimal figure-eight example: capture high, no state machine

`examples/simple_fig8.jl`, built on `init`/`step!` in the style of
[simple_auto_parking.jl](examples/simple_auto_parking.jl). The `simple_` prefix
makes [menu2.jl](examples/menu2.jl) pick it up automatically.

The entry strategy that replaces the dive/hold machine (**revised** — the
original "capture at the park elevation" is geometrically impossible, see
Findings 1):

> Engage the fig8 guidance from `t = 0` with the path centre at its **operating
> elevation** (~30°), and let the L0 attractor fly the kite down onto the
> pattern from the ~73° park. The cross-track error starts large (~40°), which
> is exactly the case the L0 attractor is well defined for — that is the
> property that removes the need for a transition machine, not a small initial
> error. `ENTRY_TIME` excludes the descent from the tracking statistics; the
> elevation floor is still checked over the whole run.

Loop skeleton:

```
s = init(V_WIND, TETHER_LENGTH; depower_setpoint, sim_time, dt, system_yaml)
l0 = s.sys_state.l_tether[1]
fec = FigureEightController(FigureEightSettings(; dt=s.dt, A, B, C, D,
                            az_center=0.0, el_center=EL_CENTER,
                            attractor_distance=ATTRACTOR_DIST, up_loops=UP_LOOPS))
heading_pid = create_heading_pid(; K, Ti, Td, dt=s.dt,
                                 umin=-MAX_STEERING, umax=MAX_STEERING)

for _ in 1:s.steps
    chi_set, az_attr, el_attr, dmin =
        navigate_fig8(fec, s.sys_state.azimuth, s.sys_state.elevation)
    s.sys_state.bearing = chi_set                       # tracked course
    s.sys_state.attractor .= (deg2rad(az_attr), deg2rad(el_attr))
    s.sys_state.var_01 = dmin                           # cross-track error [deg]
    set_K!(heading_pid, HEADING_P * V_APP_REF / max(s.sys_state.v_app, V_APP_MIN),
           chi_set, s.sys_state.heading)                # bumpless gain schedule
    rel_steering = heading_pid(chi_set, s.sys_state.heading, 0.0)   # sign per STEP 0
    step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering, set_length = l0)
end
```

Details that matter:

- **Wrapped error.** `DiscretePID` computes a plain `setpoint − measurement`
  difference with no ±π wrapping. Feed it a wrapped error (pass `0.0` as
  setpoint and `wrap_to_pi(heading − chi_set)` as measurement, as the earlier
  run script does), or the loop will command a full turn the long
  way round every time the error crosses ±180°.
- **`bearing`, not an ad-hoc field.** Setpoint tracking goes through
  `sys_state.bearing` so the existing plot/metric tooling picks it up.
- **Free log slots.** `step!` already writes `var_14` (rel_depower command),
  `var_15` (L/D_wing), `var_16` (L/D_eff) — see
  [interface.jl:558](src/interface.jl#L558). `var_01…var_13` are free. Assign
  them once, up front, and document the mapping in the module docstring:
  `var_01` = cross-track error [deg], `var_02` = attractor azimuth,
  `var_03` = attractor elevation, `var_04` = pattern-centre elevation [deg].
- **Log name.** Use a dedicated `"fig8_run"`, not the shared `"tmp_run"` — the
  earlier note about scripts clobbering each other's logs applies here too.
- **Gain starting point.** [simple_sinus.jl](examples/simple_sinus.jl) tracks a
  ±40° heading sinusoid with `HEADING_P = 5.0`, `Ti = false`, `Td = 0.15`,
  `V_APP_REF = 13.1`, `MAX_STEERING = 0.175` at `V_WIND = 9.51`. Start there.

**Verify:** `include("examples/simple_fig8.jl")`. Settling is cached after the
first run, so re-runs are fast. Do not report success without running it.

### STEP 3 — Metrics and plots

- `src/fig8_metrics.jl`, exported `fig8_metrics` / `print_fig8_metrics`, called
  unconditionally at the end of `simple_fig8.jl` (no GLMakie dependency, so
  headless sweeps still print the numbers).
- `examples/simple_fig8_plots.jl` following the existing
  `*_plots.jl` convention: pattern in the (azimuth, elevation) plane with the
  reference path and attractor overlaid, plus time series of cross-track error,
  heading vs. bearing, `rel_steering` command vs. KCU actual, elevation, and
  tether force.

### STEP 4 — Walk the pattern centre ✅ IMPLEMENTED (disabled by default)

`WALK_RATE` / `EL_FINAL` / `WALK_START` in `simple_fig8.jl` ramp the centre with
`set_path_center!` at a bounded rate, logging `el_center_cur` to `var_04`. A
single large step in the centre demands a heading change big enough to fight the
airframe's own dynamics — that is what the walk avoids.

Defaults to `WALK_RATE = 0` (disabled): per Findings 1 the run now *starts* at a
low, flyable centre, so there is no high-capture to walk down from. Use it to
move between two low centres — e.g. 30° → 25° for more tether force — and
re-check the feasibility margin at the destination, since lowering the centre
changes the path curvature.

### STEP 5 — Tune

Order of knobs, most to least effective (from the earlier campaign, but
re-verify on this plant — nothing there transfers automatically):

0. **The feasibility margin printed at startup.** If it is near or below 1, the
   tracking error is curvature-limited and no gain will fix it — enlarge the
   pattern, lower its centre, lower `BODY_DAMPING`, or raise `MAX_STEERING`
   first. `print_fig8_metrics` also reports how much of the run the steering
   command spent within 2% of its peak; a high value means the clamp is binding.
1. `MAX_STEERING` — currently 0.175, the top of the range `c1` was identified
   over. Raising it buys turn authority directly, but re-run
   `steering_test_v3.jl` at the higher amplitude first to confirm the turn-rate
   law is still linear there.
2. `ATTRACTOR_DIST` — interacts strongly with `HEADING_P`; tune the pair jointly.
3. `HEADING_P`, then `HEADING_D`, then `HEADING_I`.
4. `UP_LOOPS` — a discrete choice, not a gain. Try both; up-loops let the kite
   climb through the turns and were what unblocked the earlier campaign.
5. `F8_B` (pattern height), `EL_CENTER`.

**Per [CLAUDE.md](CLAUDE.md): change an existing parameter by at most 10% per
iteration.** Record every sweep result as a comment next to the parameter, as
the earlier run script did — that history is what makes the next
retune cheap.

Before sweeping blind, consider [src/turn_rate_id.jl](src/turn_rate_id.jl): it
fits the V3's turn-rate law `ψ̇ = c1·v_a·u_s + c2/v_a·sin(ψ)·cos(β)` from a log.
That gives a physically-grounded heading gain and 1/v_app schedule from
measurement instead of a search.

### STEP 6 — Low-wind envelope (5.0 m/s ground wind)

Do this after STEP 5, but do not treat it as optional polish: it is a stated
requirement, and it is the step most likely to change the *pattern geometry*
rather than just the gains.

What changes going from 9.51 → 5.0 m/s ground wind:

- **Loop gain: handled.** `v_app` drops roughly proportionally (~13 → ~7 m/s),
  and the turn-rate law `ψ̇ = c1·v_a·u_s` drops with it — but the kite also flies
  proportionally slower, so the *required* turn rate per lap drops by the same
  factor. To first order these cancel and the existing `K ∝ 1/v_app` schedule
  covers the rest. Watch the `V_APP_MIN` clamp (5.0 in
  [simple_auto_parking.jl](examples/simple_auto_parking.jl)) — at `v_app ≈ 7`
  we are close to it, so the schedule is near the end of its range.
- **Lift: not handled, and this is the real problem.** Lift scales with `v_a²`,
  so it falls ~4× while the kite's weight and the 8.4 kg KCU do not. The
  climbing half of each lobe is energy-limited — exactly the "can descend, can't
  climb" failure that capped the other kite at half a lap. Expect to need some
  combination of: smaller `F8_A`/`F8_B`, higher `EL_CENTER`, less depower
  (`DEPOWER_SETPOINT` closer to fully powered), and `UP_LOOPS = true`.
- **Elevation floor gets tighter, not looser.** Less lift means the 10°
  worst-case floor is more likely to be breached at 5 m/s than at 9.51.

**Deliverable:** either one parameter set that meets the success criteria at both
wind speeds, or an explicit wind-scheduled set (pattern size and centre
elevation as a function of `v_wind`) with the schedule documented in the example
header. The second outcome is the more likely one — say so in the results rather
than forcing a single compromise set that flies badly at both ends.

**Practical note:** `init` caches settling per configuration in
`data/settled_*.bin`. The first run at 5.0 m/s re-settles and is slow; re-runs
are fast.

## Success criteria

Set the targets before tuning, so there is a stopping rule. Reported by
`print_fig8_metrics`. **Must hold at both 9.51 and 5.0 m/s** (STEP 6) — record
the two sets of numbers separately, never a single averaged verdict.

| Metric | Target | Note |
|:---|:---|:---|
| laps completed in `SIM_TIME` | ≥ 3 | pattern is actually repeatable |
| RMS cross-track error (settled) | < 3° | the earlier ram-air branch reached ~1.0-1.8° |
| max cross-track error (settled) | < 8° | no per-lap excursion |
| **worst-case elevation, whole run** | **> 10°** | `min_elevation_all`, not `min_elevation_settled` — the floor is a safety limit, so a breach during the entry transient counts as a breach |
| **azimuth reach, each side** | **≥ 0.7·`A`** | `az_reach_pos`/`az_reach_neg`, the mean per-lobe extreme. Tested per side, never as one span: everything above is measured to the *closest point* of the path, so a small eight — or one lobe flown in half the wind window — passes all of it while never leaving the path |
| **elevation span (settled)** | **≥ 0.7·`B`** | same hole in the vertical: `el_span`, the other half of "flying the pattern it was asked for" |
| tether-force CV | report | baseline for the later reel-out work |
| HF steering content | report | ringing/chatter watchdog; units differ from the original, so establish a V3 baseline rather than importing a threshold |
| solver survives to `SIM_TIME` | yes | no early `next_step!` failure |

"Settled" = engagement + 10 s, as in `fig8_metrics`.

## Answered questions

- **Config: `system_reelout.yaml`** (the one the other `simple_*` examples use).
  Loaded values, verified: `d_tether` 4.0 mm vs 13.5 mm in `system.yaml`,
  `elevation` 73° vs 70°, `l_tether` 150 m vs 0.
  *Correction to the stated rationale:* the KCU is not lighter here — it is the
  only one with a mass at all. `settings_reelout.yaml` sets `kcu_mass: 8.4`,
  while `settings.yaml` omits it and `Settings` therefore loads
  `kcu_mass = 0.0` (KiteUtils' struct default; there is no fallback to
  KiteUtils' own `data/settings.yaml`). So `system_reelout.yaml` is the
  *physically complete* config — realistic 4 mm tether, real KCU mass — which is
  the right reason to choose it. The thick-tether half of the rationale is
  correct; the heavy-KCU half is inverted.
- **`V_WIND` = 9.51 m/s** for tuning. **Must also fly at 5.0 m/s** ground wind —
  see STEP 6, which this requirement makes a mandatory part of the plan rather
  than a nice-to-have.
- **Elevation floor ≈ 10°.** Note what this means physically: at 150 m tether,
  10° elevation puts the kite at ~26 m height, and the tether sags below that.
  If there is a real ground-clearance limit, it binds on the tether, not the
  kite.

