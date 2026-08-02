# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Figure-of-eight path following of the V3 kite via the `init`/`step!` interface
(see PlanFig8.md).

Single phase, no state machine: the L0 attractor guidance
(`src/fig8_controller.jl`) is engaged from `t = 0` with the reference path fixed
at its operating elevation, and a heading PID (as in `simple_sinus.jl` /
`simple_auto_parking.jl`) tracks the commanded course. The kite starts parked at
~73° and the guidance flies it down onto the pattern — the L0 attractor is well
defined at any distance from the path, which is exactly why no dive/hold
transition logic is needed.

# Why the pattern is large and low, and why BODY_DAMPING is a flight parameter

Neither is a free choice. The V3's identified turn-rate law fixes the smallest
angular turn radius the kite can fly:

    rho = 1 / (L * c1 * u_s)        [rad]   -- apparent wind speed cancels

and `c1` depends strongly on the `body_damping` passed to `init` (see
`V3_TURN_RATE_COEFFS`). The in-plane damping is added to suppress the parked AoA
ripple and cut solver cost, but it also damps the steering response — across the
identified range it changes `c1`, and hence the achievable turn radius, by 5.6x:

    body_damping        c1        min turn radius at u_s = 0.175, L = 150 m
    [ 0,  0, 40]    0.3159                    6.9°       <- init's default, used here
    [10, 10, 40]    0.0982                   22.2°
    [20, 20, 40]    0.0567                   38.5°

The tightest curvature of a lemniscate in (azimuth, elevation) is not at the
lobe tip but on the lobe's upper shoulder, and it collapses as the pattern is
raised, because the azimuth axis is compressed by `cos(elevation)`. At
`c1 = 0.3159` and `u_s = 0.175`:

    A=45 B=20, centre 73° -> tightest path radius  0.4°   (margin 0.06)
    A=45 B=20, centre 60° ->                       3.1°   (margin 0.45)
    A=45 B=20, centre 35° ->                       8.5°   (margin 1.23)
    A=50 B=25, centre 30° ->                      10.5°   (margin 1.52)  <- default

Two consequences worth stating plainly:

1. **A figure-eight near zenith is geometrically impossible for this kite** at
   any PID tuning — the "capture high, then walk the pattern down" entry used on
   a different airframe cannot work here. The pattern must be flown low and
   wide, and the kite descends onto it under guidance instead.
2. **Raising `body_damping` above `[0, 0, 40]` can make the pattern unflyable**
   on its own. At `[10, 10, 40]` the same pattern needs `u_s = 0.5` to reach a
   comparable margin, which extrapolates the linear turn-rate law well beyond
   the `|u_s| <= 0.175` range it was identified over.

`check_pattern_feasible` prints the margin at startup. Below ~1 the tracking
error is curvature-limited and no PID tuning will fix it — enlarge the pattern,
lower its centre, lower the damping, or raise `MAX_STEERING`.

Logs the run to "fig8_run" (a dedicated name, not the shared "tmp_run"). For
verification run `include("examples/simple_fig8.jl")` — a 90 s simulation is
several thousand `step!` calls and takes minutes of wall time. The script
`include`s `simple_fig8_plots.jl` itself at the end, so the pattern and
time-series figures come up without a second call.

Log slot mapping (`step!` already fills `var_14`/`var_15`/`var_16`):

| slot     | quantity                                  |
|:---------|:------------------------------------------|
| `var_01` | cross-track error d [deg]                 |
| `var_02` | attractor azimuth [deg]                   |
| `var_03` | attractor elevation [deg]                 |
| `var_04` | pattern-centre elevation [deg]            |
| `var_05` | raw guidance course chi_set [rad]         |
| `var_06` | regulated error (feedback - chi_cmd) [deg] |
| `var_07` | 1 while the entry descent limiter is active |
| `var_08` | course/heading blend weight (0 = heading, 1 = course) |
| `var_09` | span-mean geometric AoA [deg]             |

`bearing` carries `chi_cmd`, the course the loop actually tracks, so
`course - bearing` is the path-following error; the unmodified guidance course
is kept in `var_05`. The feedback angle the PID regulates is heading at low
kite speed and course at high (`V_KITE_HEADING`/`V_KITE_COURSE`, scheduled on
`|vel_kite|`), so `var_06` equals `heading - bearing` at low speed and
`course - bearing` at high. In FIG8 mode that schedule is bypassed and the
course is fed back at any speed (`FIG8_PURE_COURSE`), so `var_08` is 1
throughout phase 3 and the schedule governs the entry only.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers; tic()
using V3Kite
using DiscretePIDs: set_K!
using LinearAlgebra: norm
using Printf

@info "simple_fig8.jl: figure-of-eight path following of the V3 kite."

# ==================== USER PARAMETERS ==================== #

PROJECT          = "system_reelout.yaml"  # System project (see data/system_*.yaml)
SIM_TIME         = 150.0     # Total simulation time [s]; ~43 s per lap at
                            # v_app 13 m/s, plus the descent from the park.
                            # 30 -> 150 (2026-08-02): a DELIBERATE one-off
                            # exception to SmallPlan.md's 30 s cap, taken after
                            # HEADING_P 4.5 -> 0.6 made the loop stable (RMS d
                            # 1.02°, tape 0% rate-limited). The question the cap
                            # cannot answer: at 30 s the kite is on the left lobe
                            # only (azimuth -56.7..-6.5°) with laps = 0, and that
                            # is equally consistent with "it has not got there
                            # yet" and "it has settled into a stable one-lobe
                            # orbit" — which need opposite fixes. `laps >= 3.0`
                            # needs ~130 s of flight, so 150 s gives it a fair
                            # chance; the first 120 s also line up with
                            # data/fig8_reference.arrow for a like-for-like read.
                            # 150 -> 30 again (2026-08-02) once that question was
                            # answered (4 laps, 4 of 4 criteria). NOTE the metrics
                            # window opens at PARK_TIME + ENTRY_TIME = 25 s, so a
                            # 30 s run scores only its last 5 s and `laps` is
                            # meaningless — judge short runs on RMS d, the
                            # elevation floor and the tape metrics, and go back to
                            # 150 for anything that has to be counted in laps.
DT               = 0.05/6   # Simulation timestep [s].
                            #
                            # 0.05/3 -> 0.05/6 (2026-08-02): NOT a tuning
                            # parameter — this is a NUMERICAL stability fix, and
                            # it costs 2x wall time per run.
                            #
                            # The 150 s run at HEADING_D_N = 2.0 stopped at
                            # t = 46.07 s on the overspeed guard, but the guard
                            # reported a symptom one step late: v_app is FLAT at
                            # 27 m/s and the tether force flat at ~3.1 kN until
                            # the final timestep, where v_app jumps 26.6 -> 67.9
                            # m/s in one dt. That is not the energy runaway that
                            # killed depower 0.36 and EL_CENTER 32.8, which crept
                            # over seconds.
                            #
                            # The real failure is a step-to-step (2*dt = 30 Hz)
                            # oscillation of the WING, growing ~1.3-1.4x per
                            # sample from t = 45.0:
                            #
                            #   t [s]   AoA zigzag   |acc| zigzag   v_app  force
                            #   45.0      0.14°          10          26.8   3142
                            #   45.2      1.6°           14          26.9   3165
                            #   45.9      6.5°           73          27.3   3108
                            #   46.0     47.7°          253          27.4   3323
                            #   46.05    39.3°         7621          26.6   3850
                            #
                            # Centre-panel AoA and span-mean AoA alternate in
                            # ANTIPHASE each step and the kite acceleration
                            # carries the same Nyquist content, so it is the
                            # structure shaking, not a logging artefact.
                            #
                            # WHERE IT COMES FROM: `step!` runs with
                            # vsm_interval = 1 (VSM_INTERVAL below), i.e. the VSM
                            # aero load is refreshed once per dt and held FROZEN
                            # inside the DAE in between. That is an explicitly
                            # coupled aero-structure scheme, whose characteristic
                            # instability is exactly a growing 2*dt oscillation
                            # appearing first at maximum dynamic pressure — which
                            # is where and when this one appears (bottom of the
                            # right lobe, ~3.2 kN, elevation ~21°). Halving dt
                            # halves the aero lag and gives ~4x margin on the
                            # mode. abs_tol/rel_tol = 0.01 in data/settings.yaml
                            # leave nothing to absorb it either, but tightening
                            # them does not touch the coupling lag.
                            #
                            # HEADING_D_N is NOT the cause and was not reverted:
                            # the N=2 and N=10 trajectories agree to 0.3° in
                            # azimuth+elevation over the whole 46 s and to ~2% on
                            # per-lap peak force, and the surviving N=10 baseline
                            # carries the SAME marginal mode for all 150 s (its
                            # Nyquist acceleration content is 10-30 m/s² in every
                            # 5 s window, its AoA zigzag touches 0.99° at t = 22 s
                            # and recovers). The model rides this mode
                            # permanently at this operating point; a
                            # roundoff-level difference decides whether it trips.
                            #
                            # Raising the in-plane BODY_DAMPING would also damp
                            # it, at the cost of the turn authority the pattern
                            # needs (see the docstring), so that route stays shut.
                            #
                            # NOTE this re-bases the trajectory:
                            # data/fig8_run_N10_baseline.arrow is no longer
                            # bit-comparable, only comparable on per-lap metrics.
VSM_INTERVAL     = 2        # Steps between VSM aero updates; the load is held
                            # frozen inside the DAE in between (0 disables the
                            # update entirely). 1 is the TIGHTEST coupling
                            # available, so this can only be raised — trading
                            # aero lag for wall time. Exposed here to sweep the
                            # explicit-coupling mode described under DT above,
                            # not as a lever that can stabilize it.
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
WINCH_FORCE_MODE = true     # Winch mode. `false` = POSITION mode: `set_length`
                            # holds the tether length, and the drum only yields
                            # as far as `winch_ff_scale` lets it (1.13 m over a
                            # 30 s run at ff = 0.7). `true` = FORCE mode
                            # (PlanFig8.md option 1): the drum holds a low-passed
                            # reference force instead, so it pays out on every
                            # dive and hauls in on every climb, with the mean
                            # length kept by a slow trim. Gains live in
                            # data/wc_settings.yaml (winch_force_tau,
                            # winch_len_kp); see `winch_force_torque!`.
TETHER_LENGTH    = 200.0    # Tether length [m], held constant (position mode).
                            # 150 -> 200 (2026-07-26): the minimum angular turn
                            # radius is rho = 1/(L*c1*u_s), so a LONGER tether
                            # lets the kite turn tighter in angular terms — the
                            # single most effective lever on pattern
                            # feasibility after c1 itself.
DEPOWER_SETPOINT = 0.40     # Depower setting held during the run [-].
                            #
                            # 0.36 -> 0.40 (2026-08-02): targeting ENERGY, which
                            # is now the binding constraint. With HEADING_P 0.6
                            # the loop is stable (RMS d 1.65°, tape 1% rate-
                            # limited) and the kite CROSSED THE CENTRE for the
                            # first time under a non-saturated loop — azimuth
                            # -56.7..+45.0°, laps 0.5 — then diverged at t =
                            # 37.6 s at the right-lobe tip, v_app 27 -> 42.7 m/s
                            # in one second. v_app had crept from ~20 m/s in the
                            # first lobe to ~28 by the crossing: the kite gains
                            # energy lap over lap and the right-lobe turn is
                            # where it cashes out.
                            #
                            # This RE-TESTS the 2026-08-01 verdict below, which
                            # was taken under the relay loop that no longer
                            # exists — there the run never reached a second lobe
                            # at all, so "0.36 diverges" was measured on a
                            # trajectory that had nothing to do with this one.
                            #
                            # Side effect to watch: depower moves the turn-rate
                            # law. c1 0.1830 -> 0.1513 (-17%) and the dead time
                            # 0.383 -> 0.420 s, so LOOP GAIN DROPS ~17% and the
                            # delay grows slightly. Both push towards more
                            # stability, so HEADING_P 0.6 stays valid; if the
                            # tracking goes sluggish, that is the reason.
                            #
                            # 0.36 -> 0.40 (2026-08-01): REVERTED, see below.
                            #
                            # 0.40 -> 0.36 (2026-08-01, -10%): the entry is
                            # turn-rate limited (steering pinned at 0.300 for the
                            # whole dive, see CHI_DIVE), and depower is the only
                            # lever left that changes the authority itself —
                            # c1 = 0.1513 at 0.40 vs 0.3159 at 0.25, with the
                            # dead time falling 0.42 s -> 0.03 s. PlanFig8.md's
                            # standing note is that reel-out is what makes a low
                            # depower survivable, and the winch now runs in force
                            # mode, so the failure mode that closed this lever
                            # (overspeed in a sustained turn) is the one the
                            # compliant winch relieves.
                            # RESULT: NO. Diverged at t = 21.7 s on the overspeed
                            # guard, and violently: v_app 93.6 m/s (the 0.40 run
                            # peaks at 26.6), tether force 6164 N vs 2434 N, lift
                            # 11 kN with the drag going NEGATIVE. The winch did
                            # its part — 13.4 m paid out, more than the 10.6 m at
                            # 0.40 — and it made no difference, so reel-out at
                            # this rate does NOT make a low depower survivable
                            # here. The entry was unchanged too (steering still
                            # pinned at 0.300 through the dive), so the extra
                            # authority never even reached the trajectory before
                            # the energy problem ended the run. PlanFig8.md's
                            # "reel-out unlocks low depower" note is not
                            # supported at 200 m with this pattern.
                            #
                            # Previously: 0.40 is the
                            # middle of the two settings that each fail alone at
                            # 150 m: 0.25 is agile (c1 = 0.3159) but cannot
                            # survive a sustained turn (open-loop divergence at
                            # 13 s, v_app 51 m/s); 0.55 survives but is far too
                            # sluggish (c1 = 0.1071, dead time 0.55 s) and flew a
                            # circle instead of the pattern. 0.40 survived 29 s
                            # open-loop at 150 m — better than 0.25, short of
                            # 0.55 — and is paired here with the longer tether.
# NOTE: this parameter currently has NO EFFECT (2026-07-26). It was added to
# settle the kite on the pattern instead of at the 73° park, but the settled
# geometry cache key encodes depower, steering, tip/TE, wind, tether length,
# gravity, system and body damping — NOT the settling elevation. `settle_wing`
# therefore reuses the existing 73° geometry and the run still starts there
# (verified: logged elevation at t=0 is 73.0°, not 33.0°). Forcing `remake=true`
# would overwrite a cache file shared with simple_sinus.jl / simple_parking.jl
# with geometry they do not expect, so that is not a safe workaround from here.
# Fixing it properly means adding the elevation to the cache key in
# stabilization.jl — see PlanFig8.md, Findings 4. Left in place, and honest
# about doing nothing, rather than deleted, because starting on the pattern is
# the right way to develop the pattern controller once the key is fixed.
ELEVATION        = 73.0     # [deg] settling elevation = the natural parked
                            # equilibrium, so the kite starts where it wants to be
# Parking phase: hold zero steering so the transients left by init/settling
# decay before the controller starts demanding maneuvers. Without it the
# guidance engaged at t=0 and drove the steering straight to its clamp while the
# model was still relaxing. The guidance still runs during the park (its course
# estimate is low-passed and needs warming up), but its output is not applied.
PARK_TIME        = 5.0      # [s]

# ---- Entry state machine: park -> dive -> hold -> fig8 -------------------- #
# Modelled on the working controller's log (SmallPlan.md, "Reference run"). That
# controller does NOT let the path guidance fly the descent: from the park it
# commands a near-horizontal course open loop and lets the kite fall along the
# sphere (no attractor at all — the logged attractor is NaN until handover),
# flattens out for the last second, and hands over at the pattern's RIGHTMOST
# point, at the centre elevation, already moving downwards into the first turn.
#
# WHY here: with the guidance flying the entry, every configuration tried so far
# ends up orbiting the right-hand lobe — at EL_CENTER 40.5 the settled pattern
# spans azimuth +1.3..+42.6° with ZERO centre crossings against a ±50° reference
# centred on zero. The eight is symmetric; the way it is entered is not. The kite
# arrives from the park on the right and never gets established on the left lobe.
#
# Reference timings (its park is 10 s, ours 5 s): dive 5.6 s covering 71 -> 42°
# of elevation (~5.2°/s), hold 1.2 s covering the last 42 -> 27° (the kite is
# fastest here, so this is the steepest part), handover at the centre elevation.
ENTRY_PHASES     = true     # false = old behaviour, guidance engages at PARK_TIME
CHI_DIVE         = -85.0    # [deg] course commanded during the dive. |chi| > 90
                            # is descending, |chi| < 90 climbing, 90 exactly
                            # horizontal.
                            #
                            # -100 -> -85 (2026-08-01): the entry is TURN-RATE
                            # limited (see DIVE_EL_MARGIN below) — the kite never
                            # reaches the commanded course, so a descending
                            # command just adds to the fall it is already taking
                            # while it turns out of the park. -85° asks it to
                            # CLIMB slightly, spending the turn on azimuth travel
                            # instead of altitude, which is what the reference's
                            # ramp (98° -> 143°, shallow first) achieves.
                            # RESULT: bit-identical to -100. Same phase times
                            # (5.05 / 10.45 / 11.65 s), same handover (az 17.4°,
                            # el 46.1°), same settled span (-4.7..49.2°), same
                            # RMS d 5.99°. The steering command sits at exactly
                            # +0.300 for the WHOLE dive, so the kite is turning
                            # as hard as the plant allows and the numeric value
                            # of the command is irrelevant — only its SIGN
                            # reaches the plant. Within this authority the entry
                            # cannot be shaped by CHI_DIVE, CHI_HOLD or
                            # DIVE_EL_MARGIN at all; the only entry choice that
                            # exists is which side to enter from. Getting a
                            # shapeable entry needs more turn authority, i.e. a
                            # lower depower (c1 0.1513 -> 0.3159), which the
                            # force-mode winch may now make survivable.
                            #
                            # SIGN, measured 2026-08-01: a POSITIVE commanded
                            # course drives the kite towards NEGATIVE azimuth
                            # (+100° took it from az 0.1° to -18.6°). The
                            # reference enters at the pattern's rightmost point,
                            # i.e. positive azimuth, so the command is negative
                            # here. This is the opposite of the reference log's
                            # +98°, whose azimuth sign convention is mirrored.
CHI_HOLD         = -90.0    # [deg] course commanded during the hold: exactly
                            # horizontal, i.e. stop descending and let the kite
                            # arrive at the pattern flat rather than diving into it
DIVE_EL_MARGIN   = 15.0     # [deg] above EL_CENTER at which the dive ends and the
                            # hold begins (reference: 42° vs a 26° centre = 16°)
                            # 15 -> 5 (2026-08-01): at 15 the handover landed at
                            # el 46.1°, 5.6° above the centre and well short of
                            # the pattern's right edge. The reference hands over
                            # AT the centre elevation; the hold itself then costs
                            # ~1° here, so 5 aims the handover at ~centre + 4.
                            # RESULT: WORSE. Handover elevation is right (39.1°
                            # vs a 40.5° centre) but the azimuth went the wrong
                            # way: 17.4° -> 7.7°, and the settled pattern lost
                            # its centre crossing (1 -> 0, span 0.5..45.4°,
                            # RMS d 5.99 -> 6.97°, min el 21.3 -> 20.2°).
                            # WHY: during the hold the kite tracks az 16.8 ->
                            # 7.7 while dropping 45.2 -> 39.1°, i.e. it flies
                            # down-LEFT, not horizontally right as commanded.
                            # It never reaches the commanded course at all — at
                            # chi = -100° the great-circle geometry gives ~5.7°
                            # of azimuth per degree of elevation lost, which
                            # would be ~160° of azimuth over this descent; the
                            # kite manages 17°. The entry is TURN-RATE limited,
                            # so lengthening the dive does not carry it further
                            # right, it only drops it lower in the same place.
HOLD_TIME        = 1.2      # [s] duration of the hold, from the reference log
#
# FIRST RESULTS (2026-08-01, EL_CENTER 40.5, force mode, attr 15):
#   - the phases fire as intended: park -> dive at 5.05 s, dive -> hold at
#     10.45 s (el 55.2), hold -> fig8 at 11.65 s (az 17.4°, el 46.1°).
#   - with CHI_DIVE = +100 the settled metrics came out IDENTICAL to the run
#     without any state machine (RMS d 5.27°, span 1.3..42.6°, 0 crossings).
#     That is not a bug: ENTRY_CHI_MAX was already clamping the guidance course
#     to ±95° during the descent, so the "new" entry commanded almost exactly
#     what the limiter had been commanding. The state machine's value is that
#     the entry is now explicit and steerable, not that it changed the flight.
#   - flipping to CHI_DIVE = -100 (entering from the RIGHT, as the reference
#     does) is what actually moves the pattern: azimuth span 1.3..42.6° ->
#     -4.7..+49.2° and 0 -> 1 centre crossing. Cost: RMS d 5.27 -> 5.99°, max
#     7.88 -> 9.17°, steering back to 100% clamped (bang-bang, HF std 0.0000),
#     elevation floor 23.2 -> 21.3°.
# STILL NOT THE REFERENCE GEOMETRY: it hands over at az 17.4° / el 46.1°, i.e.
# 5.6° ABOVE the pattern centre and nowhere near the rightmost point of a ±50°
# eight. The reference hands over AT the centre elevation, at the far edge,
# already turning down. Lowering DIVE_EL_MARGIN (and/or steepening CHI_DIVE) to
# carry the dive further right and lower is the obvious next step.
# In-plane body damping is a FLIGHT parameter here, not just a solver setting:
# it sets c1 and hence the achievable turn radius (see the docstring). init's
# default [0,0,40] is the most agile and the only one that flies this pattern
# inside the identified steering range. Raising it costs turn authority; it buys
# a smaller parked AoA ripple and ~3.4x fewer solver steps (see `init`).
BODY_DAMPING     = [0.0, 0.0, 40.0]

# Pattern geometry [deg]. Sized by the turn-radius argument in the docstring;
# check the feasibility margin printed at startup before changing these.
F8_A             = 50.0     # Width of the eight (azimuth spans +-A)
F8_B             = 20.0     # Height of the eight (elevation spans +-B/2)
                            # 40/15 -> 50/20 (2026-08-01): REVERTED. The premise
                            # was wrong and check_pattern_feasible said so before
                            # the run even started: a SMALLER lemniscate is a
                            # TIGHTER one. Tightest path radius 8.4° -> 6.4°
                            # against a kite minimum of 6.3°, i.e. margin
                            # 1.33 -> 1.02, right at the limit. The run aborted
                            # at t = 19.1 s, 7 s after handover, with the kite
                            # never getting past azimuth 19.2°. The reference
                            # controller flies 40/15 because its ram-air kite has
                            # several times this kite's turn authority, not
                            # because a small pattern is easier. For the V3 the
                            # margin improves with a BIGGER pattern or a longer
                            # tether (rho = 1/(L*c1*u_s)).
                            #
                            # Tried 40/15 (2026-08-01): the working
                            # controller's pattern (SmallPlan.md, "Reference
                            # run"). Three separate levers — steering gain, entry
                            # course, depower — have each failed to produce the
                            # lobe crossover, all because the kite already turns
                            # at its physical maximum (steering pinned at 0.300
                            # through the whole entry) and still gains only ~17°
                            # of azimuth. check_pattern_feasible has reported a
                            # margin of just 1.19-1.30 throughout. A smaller
                            # pattern raises that margin instead of asking the
                            # kite for authority it does not have.
F8_C             = 0.0      # Size of the right part
F8_D             = 0.0      # Asymmetry factor
EL_CENTER        = 36.5     # Pattern-centre elevation; spans 26.5-46.5° at B=20.
                            #
                            # 32.8 -> 36.5 (2026-08-02): REVERTED, rung 2 failed.
                            #
                            # 36.5 -> 32.8 (2026-08-02, -10.1%): rung 2 of the
                            # descent. FAILED at t = 48.7 s, and the mechanism is
                            # worth having: coming round the RIGHT lobe the kite
                            # undershot to elevation 17.9°, five degrees BELOW
                            # the pattern's own bottom edge (22.8° at this
                            # centre). That put it deep in the power zone and
                            # speed and force ran away together — 25.2 -> 32.6
                            # m/s and 2404 -> 3811 N in the last five seconds.
                            # Same lobe and same signature as the depower-0.36
                            # failure at 37.6 s. Tracking was fine until it was
                            # not (RMS d 3.55° includes the blowup).
                            #
                            # SO THE CENTRE IS NOT THE BINDING PARAMETER — the
                            # pattern's BOTTOM EDGE is, and at B = 20 the bottom
                            # sits 10° below the centre. The reference controller
                            # reaches centre 26° with A = 40, B = 15, i.e. a
                            # bottom at 18.5°; our B = 20 at centre 26 would put
                            # the bottom at 16°. Reaching 26 almost certainly
                            # needs B (and probably A) to come down with it, and
                            # "pattern size" is one of the levers whose closure
                            # was measured through the relay loop, so it is
                            # untested at this tuning.
                            #
                            # RUNG 1 RESULT (40.5 -> 36.5): 4 of 4 criteria, and
                            # BETTER — RMS d 2.90 -> 2.63°, max d 7.07 -> 6.22°,
                            # clamp 12 -> 8%. The curvature-margin argument
                            # working as predicted. Energy moved the wrong way
                            # but only slightly: mean force 2432 -> 2655 N (+9%),
                            # peak v_app 27.0 -> 27.9 m/s, elevation floor 23.7
                            # -> 20.5°. Crucially it does NOT accumulate — median
                            # v_app is 25.9 m/s over the first half of the
                            # settled window and 25.4 over the second, i.e. flat
                            # to falling. Azimuth span -51.6..+47.4°, so the
                            # pattern now slightly OVER-flies the left lobe.
                            #
                            # 40.5 -> 36.5 (2026-08-02, -9.9%): step 1 of the
                            # descent towards 26°, the reference controller's
                            # centre — a known-flyable operating point for this
                            # pattern family, so the target is not arbitrary.
                            # Ladder: 40.5, 36.5, 32.8, 29.5, 26.6, 26.0.
                            #
                            # WHAT TO WATCH, and it is not RMS d. Two forces pull
                            # opposite ways: a lower centre IMPROVES the curvature
                            # margin (less cos(elevation) compression of the
                            # azimuth axis) but pushes the pattern deeper into the
                            # power zone. Every failure at low centre so far has
                            # been an ENERGY failure — v_app and tether force run
                            # away, the tracking looks fine until it does not
                            # (depower 0.36 held RMS d 1.65° right up to a 27 ->
                            # 42.7 m/s blowup at 37.6 s). So: peak v_app, mean and
                            # peak force, and the elevation floor first; RMS d
                            # second. Run at SIM_TIME 150, not 30 — a 30 s run
                            # cannot see the failure this sweep is looking for.
                            #
                            # The 2026-08-01 entries below were all taken at
                            # HEADING_P 4.5, i.e. through the relay loop, so their
                            # RMS/clamp numbers do not transfer; their ENERGY
                            # observations (mean force climbing as the centre
                            # drops) do, and are the reason for the caution above.
                            #
                            # 45 -> 40.5 (2026-08-01, -10%): continuing down now
                            # that the floor is known to be movable. 40.5 is the
                            # second value of the old fixed-tether sweep below,
                            # where it also diverged. Reference controller: 26°.
                            # RESULT: survives 30 s as well, so the old floor is
                            # gone for good. RMS d 6.79 -> 5.27°, max 10.02 ->
                            # 7.88°, force CV 15.2 -> 9.2%, but the mean force
                            # keeps climbing (2456 -> 2794 N) and the elevation
                            # floor is down to 23.2°. Steering 96% clamped (45°
                            # gave 92%), and the pattern drifted FURTHER off
                            # centre: azimuth 1.3..42.6°, ZERO centre crossings
                            # (45° had 2). Lower centres are survivable now but
                            # do not by themselves produce the crossover — the
                            # pattern is offset to the right, which points at the
                            # entry/asymmetry, not at the centre elevation.
                            #
                            # 50 -> 45 (2026-08-01, -10%): RE-TEST. The failure
                            # recorded below was on the FIXED-LENGTH tether; the
                            # winch now runs in force mode (10.6 m of travel,
                            # +2.1/-1.2 m/s), and that failure was an ENERGY
                            # failure — overspeed in the power zone at 3494 N —
                            # which is exactly what a paying-out winch relieves.
                            # A tether that yields under load is the one change
                            # since then that could plausibly move this floor.
                            # RESULT: IT WORKS. 45° now SURVIVES the full 30 s
                            # where it aborted at 17.4 s on the fixed tether, and
                            # the steering finally comes off the clamp (92% vs
                            # 100%). Cost: RMS d 4.93 -> 6.79°, max 7.77 ->
                            # 10.02°, force 1675 -> 2456 N (CV 15.2%). The
                            # elevation floor recorded below is therefore an
                            # artefact of the FIXED-LENGTH winch, not a property
                            # of the kite — the whole "50° is the floor" sweep is
                            # worth re-running in force mode.
                            #
                            # Tried 45.0 on the course-feedback loop
                            # (2026-08-01, -10%) — 45° was the lowest centre
                            # that FAILED the sweep below (18.4 s) back when the
                            # loop regulated heading, so it was a deliberate
                            # re-test of that limit. It STILL FAILS, and in the
                            # same way: solver abort at t = 17.4 s, at elevation
                            # 33° with v_app 27.5 m/s, tether force 3494 N and
                            # AoA 45°. The better feedback signal does NOT buy a
                            # lower centre, because the binding limit is energy,
                            # not tracking: cross-track error was 0.4-6.8° right
                            # up to the abort, i.e. the guidance was working and
                            # the kite simply overspeeds in the power zone. The
                            # curvature margin even IMPROVES (1.19 -> 1.30), as
                            # the sweep predicted. Reverted to 50.0.
                            # This is the LOWEST SURVIVABLE centre, not a
                            # conservative pick. Swept in 10% steps
                            # (2026-07-26, A=50 B=20, u_s=0.30):
                            #   centre  el span   margin   survived
                            #    50.0°   40-60°    1.19      200 s   <- kept
                            #    45.0°   35-55°    1.30     18.4 s
                            #    40.5°   30-50°    1.33     13.9 s
                            #    36.5°   26-46°    1.35     13.7 s
                            # Lowering the centre eases the cos(elevation)
                            # compression and so IMPROVES the curvature margin,
                            # but pushes the pattern deeper into the power zone
                            # and the energy limit binds first. Geometry and
                            # energy pull in opposite directions here; the
                            # energy side wins until reel-out exists.
                            # Pattern at this centre: 224 m wide x 70 m tall,
                            # 564 m of path per lap, tightest radius 26 m.
@isdefined(ATTRACTOR_DIST) || (
ATTRACTOR_DIST   = 12.1)    # Arc distance Q -> attractor [deg]. Guarded with
                            # `@isdefined` so a sweep driver can set it in the
                            # REPL before `include`ing this file and have its
                            # value survive (same pattern as F8_* in
                            # simple_fig8_plots.jl); the number here is the
                            # default for a plain run. NOTE the footgun: a stale
                            # ATTRACTOR_DIST left in the REPL by a sweep silently
                            # wins over this value. The startup @info line prints
                            # the lead actually used — read it.
                            #
                            # 15.0 -> 12.1 (2026-08-02). SWEPT 10 .. 20 in 10%
                            # steps, 150 s per run, depower 0.40, HEADING_P 0.6,
                            # metrics from t = 25 s:
                            #
                            #   attr   t_end  laps  RMS d  max d  az span      x  CV
                            #   10.0    150    3.5   2.85   7.25  -48.7..48.5  7  17.0%
                            #   11.0    150    3.5   2.87   7.18  -48.1..47.9  8  16.4%
                            #   12.1    150    4.0   2.90   7.07  -47.5..47.2  8  15.7%  <- kept
                            #   13.31   150    4.0   2.97   6.92  -46.8..46.5  8  15.5%
                            #   14.64   150    4.0   3.13   6.94  -46.0..45.6  8  14.9%
                            #   16.11   150    4.0   3.36   6.89  -45.2..44.8  8  14.1%
                            #   17.72    41.1  0.0   3.38   5.88  -21.9..43.9  1  11.0%
                            #   19.49    40.4  0.0   3.58   6.42  -19.9..43.0  1  10.7%
                            #
                            # Three results. (1) RMS d rises MONOTONICALLY with
                            # the lead, 2.85 -> 3.58°, so the minimum is at the
                            # bottom edge of the swept range and may lie below
                            # 10. (2) There is a survival CLIFF between 16.11 and
                            # 17.72: both long leads die at ~41 s with one
                            # crossing and a one-sided span. (3) Force ripple
                            # trades the other way, CV 17.0 -> 10.7%, i.e. the
                            # gentlest force is at leads that no longer fly the
                            # pattern. The flown azimuth span also SHRINKS with
                            # lead (±48.6 -> ±45°), so a short lead flies a
                            # fuller eight, not merely a better-tracked one.
                            #
                            # 12.1 is kept over the 2.85° minimum at 10: it is
                            # the shortest lead giving the full 4 laps and 8
                            # crossings, passes RMS d < 3.0° with margin, has
                            # 1.3 points less force ripple than 10, and sits
                            # clear of the cliff. Buying 0.05° of RMS with half a
                            # lap and more force ripple is not a good trade.
                            #
                            # This INVERTS the 2026-08-01 sweep below, which
                            # found 35.1 optimal. That sweep ran under the relay
                            # loop (HEADING_P 4.5, steering clamped ~90% of the
                            # time), where the plan itself recorded that guidance
                            # changes made no difference — it was measuring a
                            # saturated controller, not the guidance.
                            #
                            # 35.1 -> 15.0 (2026-08-01): a 35° lead is ~22% of a
                            # lap, and the sweep below noted that part of its
                            # advantage was the kite CUTTING CORNERS on a
                            # smoother, higher trajectory rather than tracking
                            # the pattern. With the winch in force mode the
                            # cross-track error is 12.5° RMS and the flown eight
                            # is far too small, so the lead is shortened to make
                            # the guidance follow the path instead of the chord.
                            # 10.0 is the other candidate but has a recorded
                            # divergence at 13.6 s (see the 200 s sweep below).
                            # RESULT (force mode, 30 s): RMS d 12.46 -> 4.97°,
                            # max d 22.0 -> 7.9°, mean force 2621 -> 1673 N, and
                            # the elevation floor holds at 35.5°. The shorter
                            # lead recovers all of the tracking the moving
                            # tether cost. Two things it does NOT fix: steering
                            # is now clamped 100% of the time (was 93%), and the
                            # settled pattern is one lobe, azimuth -3.9..+47.8°
                            # against the +-50° reference.
                            #
                            # 19.8 -> 35.1 (2026-08-01): swept in 10% steps to
                            # MINIMIZE the RMS COURSE error (course - chi_cmd,
                            # both in the guidance convention), on the
                            # course-feedback loop with DOWN-loops, 30 s runs,
                            # depower 0.40 / 200 m / centre 50°:
                            #   attr  t_end  RMSchi>=15s  >=25s  RMSd>=25s  min_el
                            #   16.2   30.0     60.1      79.2     5.24     30.6
                            #   18.0   30.0     63.7      87.6     6.72     28.2
                            #   19.8   30.0     60.4      82.4     8.37     27.9
                            #   21.8   30.0     59.8      73.3     8.30     28.5
                            #   24.0   30.0     58.4      62.2     7.42     29.9
                            #   26.4   30.0     55.8      47.7     6.26     31.3
                            #   29.0   30.0     57.5      45.0     5.81     31.4
                            #   31.9   27.1     77.6      42.7     9.43     28.4  DIVERGED
                            #   35.1   30.0     41.9      29.2     4.21     40.2  <- min
                            #   38.6   30.0     45.7      36.7     6.51     40.2
                            # 35.1 is the minimum in BOTH windows and is also
                            # the best on cross-track error, elevation floor
                            # (40.2° vs ~29°) and force ripple (CV 8.6% vs 27%).
                            # Three caveats before trusting it:
                            # 1. The landscape is NOT smooth — 31.9 diverges
                            #    between two surviving neighbours, so these are
                            #    single 30 s runs of a marginally stable loop,
                            #    not a converged optimum.
                            # 2. RMS course error is ~42° even at the minimum.
                            #    Steering is clamped 77% of the time, so this
                            #    picks the least-bad point inside a saturated
                            #    regime; the course oscillates +-60° about the
                            #    command (measured, not a wrap artefact — only
                            #    12% of samples exceed 90°).
                            # 3. A 35° lead is ~22% of the ~162° of arc in one
                            #    lap. Part of the gain is the kite cutting
                            #    corners on a smoother, higher trajectory rather
                            #    than tracking the pattern more faithfully.
                            #
                            # EARLIER sweep, 200 s runs on the HEADING loop with
                            # UP-loops (kept because it maps the low end, where
                            # the 2026-08-01 sweep did not go):
                            #   attr  survived  laps  RMS d  min el  saturation
                            #   10.0     13.6s     -      -       -       -
                            #   14.6      200s     0   6.34°   29.3°     97%
                            #   16.2      200s     0   5.94°   29.8°     97%
                            #   18.0      200s     0   5.76°   29.9°     97%
                            #   19.8      200s     0   5.50°   30.5°     97%
                            #   attr  survived  laps  RMS d  min el  saturation
                            #   10.0     13.6s     -      -       -       -
                            #   14.6      200s     0   6.34°   29.3°     97%
                            #   16.2      200s     0   5.94°   29.8°     97%
                            #   18.0      200s     0   5.76°   29.9°     97%
                            #   19.8      200s     0   5.50°   30.5°     97%
                            # The lead is NOT the lever for the lobe crossover:
                            # every surviving value circles the left lobe
                            # (azimuth ~ -47..-25°) and never crosses the centre.
                            # It only trades tracking quality, and monotonically
                            # the OTHER way than expected — longer lead gives
                            # lower RMS and a higher floor. Below ~14° the
                            # command rotates faster than the kite can follow
                            # given the 0.42 s steering dead time at this
                            # depower, and the run diverges (10° died at 13.6 s
                            # after the course swung -154° -> -45° in 2.5 s).
                            # The heavy steering saturation in EVERY case is the
                            # real constraint: the crossover is authority-limited.
                            # Interacts strongly with HEADING_P; tune jointly.
UP_LOOPS         = false    # Fly DOWN-loops: at large |azimuth| the kite passes
                            # the azimuth extreme moving downwards. The flag
                            # reverses the traversal direction of the reference
                            # path (`_build_path` in src/fig8_controller.jl);
                            # the path shape itself is unchanged, so the
                            # curvature feasibility margin is unaffected.
                            # MEASURED on the V3 (2026-07-26) on the HEADING
                            # loop: true -> survives 200 s; false -> diverges at
                            # 17.8 s, all else equal. Up-loops shed energy
                            # through the turn where down-loops convert height
                            # into speed, and the failure mode of this
                            # configuration is overspeed.
                            # RE-MEASURED on the course-feedback loop
                            # (2026-08-01): down-loops now SURVIVE the full 30 s
                            # at this centre, so the 17.8 s divergence above was
                            # partly a feedback-signal problem, not purely an
                            # energy one. At the then-current lead of 19.8 they
                            # tracked worse (RMS d 8.37° vs 3.86° for up-loops);
                            # after retuning ATTRACTOR_DIST to 35.1 they are the
                            # first configuration recorded here that actually
                            # CROSSES the pattern centre (2 crossings in 18 s,
                            # azimuth -23..+33°) instead of circling one lobe.

# Optional walk of the pattern centre (PlanFig8.md STEP 4). 0 disables it; the
# run then flies the whole time at EL_CENTER. Use it to move the pattern to a
# lower, more force-optimal centre after a stable capture.
WALK_RATE        = 0.0      # [deg/s] rate to walk the centre towards EL_FINAL
EL_FINAL         = 25.0     # [deg] final pattern-centre elevation
WALK_START       = 60.0     # [s] time after which the walk begins

# Heading PID. Output is rel_steering (dimensionless, -1..1), fed UNNEGATED:
# positive rel_steering produces a positive heading rate on this plant
# (measured, r = +0.998 — see src/fig8_controller.jl).
HEADING_P        = 0.6      # Gain at v_app == V_APP_REF (simple_sinus.jl value
                            # was 5.0)
                            #
                            # 4.5 -> 0.6 (2026-08-02, factor 7.5, deliberately
                            # OUTSIDE the 10%-per-iteration rule — see below for
                            # why 10% steps provably cannot reach it). Derived,
                            # not guessed:
                            #
                            #   plant:  psi_dot = c1*v_a*u_s, an INTEGRATOR with
                            #           gain c1*v_a = 0.183*20 = 3.66 rad/s per
                            #           unit u_s at flight speed
                            #   loop:   omega_c = K * 3.66, and at the current
                            #           K = 4.5*13.1/20 = 2.55 that is 9.3 rad/s
                            #           = 1.5 Hz
                            #   delay:  0.72 s MEASURED (the steering tape's
                            #           rate-limit lag, ~2x the 0.383 s
                            #           small-signal dead time from
                            #           turn_rate_coeffs)
                            #   margin: a delay needs omega_c*T_d <~ 0.8 rad, so
                            #           omega_c <= 1.1 rad/s, K <= 0.30, i.e.
                            #           HEADING_P ~ 0.46 at v_app 20. Against the
                            #           optimistic 0.383 s figure it is 0.86.
                            #           0.6 sits between the two.
                            #
                            # The loop was running ~8x over gain, hence a relay:
                            # K = 2.55 clamps at only 6.7° of course error, and
                            # 88.2% of phase-3 samples exceeded that. Measured
                            # consequence: the kite turned at a median 43.5 deg/s
                            # while the guidance asked for a median 8.3 deg/s —
                            # a 5x overshoot, i.e. a self-sustained oscillation,
                            # NOT a tracking deficit. The 40° median course error
                            # is the RESULT of that oscillation.
                            #
                            # This also explains both entries below. 4.5 -> 2.0
                            # is still ~4x over gain, so the loop stayed a relay
                            # and the run "changed only how it saturates" — that
                            # was the correct observation with the wrong cause
                            # attached to it ("authority-limited"; it is RATE-
                            # limited, see SmallPlan.md).
                            #
                            # 2.0 -> 4.5 (2026-08-01): REVERTED. The large cut
                            # below bought 0.4° of RMS and cost a tripling of the
                            # force ripple and 3.3° of elevation floor.
                            #
                            # 4.5 -> 2.0 (2026-08-01): the 10% step below did
                            # nothing, so this was the deliberate large cut
                            # (factor 2.25, outside the 10%-per-iteration rule)
                            # meant to pull the steering command off its clamp.
                            # RESULT: it does NOT come off the clamp — still 100%
                            # saturated, but now as a pure BANG-BANG command
                            # (steering HF std 0.0000, i.e. the command only ever
                            # sits at +0.30 or -0.30 and never in between). RMS d
                            # 4.93 -> 4.49°, max d 7.77 -> 8.34°, but the force
                            # ripple triples (CV 10.4 -> 21.6%) and the elevation
                            # floor drops 35.1 -> 31.8°. Pattern still one lobe
                            # (-1.8..+46.9°). The gain is not the binding
                            # constraint: the loop is authority-limited, so
                            # scaling what it asks for only changes how it
                            # saturates, not whether it does.
                            #
                            # 5.0 -> 4.5 (2026-08-01, -10%): at ATTRACTOR_DIST
                            # 15 the steering command is clamped 100% of the
                            # time, so the loop is running open — a lower gain
                            # should bring it back inside its authority. Held to
                            # the 10%-per-iteration rule; repeat if the effect is
                            # within noise.
                            # RESULT: within noise, as expected while saturated.
                            # RMS d 4.97 -> 4.93°, max 7.88 -> 7.77°, min el 35.5
                            # -> 35.1°, force 1673 -> 1675 N, still clamped 100%
                            # of the time, pattern still one lobe (-3.8..+47.8°).
                            # A gain the plant never applies cannot change the
                            # flight: while |u_s| sits on its limit the loop is
                            # open, so this knob only starts to act once the
                            # command comes off the clamp. Either cut it much
                            # harder (x2-x3, i.e. outside the 10% rule) or treat
                            # the saturation itself as the thing to fix.
HEADING_I        = false    # No integral action: a steady heading bias shows up
                            # as a steady cross-track error, which the guidance
                            # itself already corrects by pulling the attractor
                            # back onto the path. Try a finite Ti only if a
                            # persistent one-sided cross-track offset remains.
HEADING_D        = 0.15     # Derivative time [s], damps the initial transient
HEADING_D_N      = 2.0      # Derivative filter: maximum gain of the D path,
                            # which is K*Td*s/(1 + s*Td/N). Flat at K below
                            # N/(2*pi*Td) Hz, rising to N*K above it.
                            # 10 (the DiscretePIDs default) -> 2 (2026-08-02),
                            # after u_s was seen carrying a visible high-frequency
                            # ripple. MEASURED on the 150 s 4-lap log, settled
                            # window t = 30..150 s: the fed-back course and
                            # heading carry only BROADBAND noise (~0.003°, no
                            # peak), while the command has 0.00505 RMS in
                            # 2..25 Hz, peaking at 7.95 Hz. That peak is not a
                            # plant mode — multiplying the measured error-noise
                            # spectrum by the PID's own magnitude reproduces both
                            # the shape and the 7.95 Hz peak, which sits where the
                            # rising D gain meets the falling noise floor. With
                            # N = 10 the corner is at 10.6 Hz and the HF gain
                            # 10*K; at N = 2 it is 2.1 Hz and 2*K.
                            # At the loop's own 0.1 Hz the D path contributes a
                            # gain of 1.005 and 5.4° of phase lead either way, so
                            # this is a filter change, not a gain change.
                            # The delivered tape barely saw the ripple (HF RMS
                            # 0.00027, 19x smaller than the command), so the cost
                            # was never flight quality — it was that command-side
                            # slew (RMS 0.89 /s against the 0.2 /s tape limit) is
                            # mostly noise, which makes every rate metric measured
                            # on set_steering unreadable.
                            # RESULT, verified two ways. (a) Offline replay of the
                            # PID over the BASELINE log's own error signal, so the
                            # only difference is the filter: HF (2..25 Hz) command
                            # RMS 0.00505 -> 0.00225 (-56%), band ratios 0.77 /
                            # 0.54 / 0.43 / 0.39 over 2-4 / 4-8 / 8-15 / 15-25 Hz
                            # against 0.80 / 0.52 / 0.36 / 0.31 predicted, while
                            # the 0.05..0.5 Hz content — everything the loop
                            # actually uses — moved 0.0549 -> 0.0551 (+0.4%).
                            # (b) A 30 s run against the baseline over the same
                            # 25..30 s window: RMS d 3.14 -> 3.10°, min elevation
                            # 22.2 -> 22.3°, azimuth span identical to 0.2°, and
                            # peak TAPE slew 0.171 -> 0.115 /s (-33%) — same
                            # flight, less actuator work.
                            # Do NOT judge this on a 17..30 s window: there the
                            # command's own transients (clamped 8% of the time)
                            # dominate the HF tails through the P path and hide
                            # the effect (band ratios only 0.92 / 0.79 / 0.71).
V_APP_REF        = 13.1     # Reference apparent wind speed for the schedule [m/s]
V_APP_MIN        = 5.0      # Lower clamp on v_app, limits the gain boost [m/s]

# WHAT THE LOOP REGULATES: heading at low KITE speed, course at high.
# The guidance commands a COURSE, so course is the signal that actually closes
# the path-following loop and is what must be regulated once the kite is flying
# fast. When the kite is slow it is the wrong feedback: the flight-path
# direction is undefined at zero velocity and noisy just above it, while heading
# stays clean and still has the right sign of steering authority. Regulating
# course throughout also asks the loop to chase the drift angle, which the kite
# cannot change directly.
#
# Scheduled on |vel_kite|, NOT on v_app. A parked V3 already sees v_app ~ the
# ambient wind, so apparent wind speed cannot tell "flying" from "hanging
# still" — measured on this configuration:
#
#     signal          park          flying (t >= 15 s)
#     v_app           9.1 m/s       21.1 m/s, never below 10
#     |vel_kite|      4.2 m/s       15.5 m/s (8.3 .. 22.3)
#
# A 10 m/s threshold on v_app therefore puts the PARK at blend weight 0.82 —
# nearly full course feedback on a kite that is barely moving, which is the one
# case the rule exists to prevent. On |vel_kite| the park is unambiguously
# heading, and the weight modulates in flight when the kite slows through a
# turn, which is when the course estimate is worst.
#
# Below V_KITE_HEADING the error is formed from heading alone, above
# V_KITE_COURSE from course alone, and linearly blended in between. The band
# exists to avoid a hard switch: heading and course differ by the drift angle
# (~13-15° on the V3), so a step change of feedback signal at one speed would
# step the steering command by HEADING_P * drift. Widen it if that shows.
V_KITE_HEADING   = 5.0      # [m/s] at/below: pure heading feedback
V_KITE_COURSE    = 10.0    # [m/s] at/above: pure course feedback ("high" per
                            # the flight note in SmallPlan.md; the lower edge is
                            # a choice — 5 m/s is just above the 4.2 m/s the
                            # kite drifts at during the park)
                            #
                            # 5.0/10.0 -> 0.0/0.001 (2026-08-01): DISABLES the
                            # blend (w_course == 1 at any speed), i.e. pure
                            # course feedback. This reproduces the conditions
                            # the ATTRACTOR_DIST table above was swept under, to
                            # test whether the blend is why that table's 30 s
                            # survival at attr 35.1 no longer reproduces (three
                            # runs today, incl. two winch variants, all diverged
                            # at 17.6-18.0 s with min_el ~28° instead of 40.2°).
                            # Restored to 5.0/10.0 (2026-08-02): with the phased
                            # entry the transition looks good again, and the band
                            # now applies to the ENTRY ONLY — see below.
FIG8_PURE_COURSE = true     # In FIG8 mode (phase 3), feed back COURSE alone and
                            # ignore the V_KITE_* schedule; the entry phases keep
                            # it. This is SmallPlan.md's "gate on phase instead
                            # of speed" option. Rationale: path following is a
                            # course problem, and on the pattern the kite is fast
                            # enough that the schedule asks for course anyway —
                            # it only dips into the band during the slow part of
                            # a turn, swapping the feedback signal mid-turn for
                            # no benefit. `false` restores the pure speed
                            # schedule in every phase.
STEERING_TRACK_TAPE = false # Clamp the steering command to within STEERING_LEAD
                            # steps of tape travel of the MEASURED tape position
                            # (`sys_state.steering`). SmallPlan.md proposed this
                            # as "anti-windup against the real tape position".
                            #
                            # TRIED 2026-08-02 AT LEAD 3, AND IT IS A DISASTER —
                            # kept only so nobody proposes it again:
                            #
                            #   metric        4-lap baseline   governor on
                            #   RMS d              2.90°          58.10°
                            #   min elevation     23.7°          -49.2° (!)
                            #   laps               4.0            2.0
                            #   peak tape slew    0.200/s        0.030/s
                            #   criteria passed    4 of 4         0 of 4
                            #
                            # The mechanism: pinning the command to within 0.010
                            # of the measured position makes THE CLAMP the
                            # binding rate limit instead of the tape, and the
                            # pair can then only advance as fast as the gap
                            # allows — an effective actuator 6.7x SLOWER than the
                            # real one. The kite lost the pattern and went below
                            # the horizon.
                            #
                            # The premise was wrong anyway: since HEADING_P 4.5
                            # -> 0.6 the tape saturates only 2% of the time, so
                            # there was nothing to govern. Fixing the loop gain
                            # already solved what this was meant to solve.
STEERING_LEAD    = 3        # [steps] how far ahead of the tape the command may
                            # sit when STEERING_TRACK_TAPE is on. 3 * 0.2 * dt =
                            # 0.010 of travel. A LARGER lead is less harmful, not
                            # more — but the whole idea is closed, see above.
MAX_STEERING     = 0.30     # Steering command limit [-]. OPTION 3 (raise the
                            # authority to relieve the 97% clamp saturation) is
                            # CLOSED — it does not work on this plant:
                            #   0.30 -> survives the full 200 s
                            #   0.33 -> DIVERGED at t = 30.9 s, peak turn rate
                            #           949 deg/s, turn-rate HF 45.6 deg/s (vs
                            #           0.45 at 0.30) — the loop goes violently
                            #           unstable, not merely saturated
                            #   0.375 -> the PLANT itself diverges, in plain
                            #           bang-bang oscillation with no controller
                            #           (identification sweep, 2026-07-26)
                            # So the usable authority ceiling is BELOW what the
                            # lobe crossover needs, and 97% saturation at 0.30 is
                            # not a tuning oversight but the plant's limit at
                            # this depower. c1 is linear over the range
                            # (0.1495 to u_s 0.374 vs 0.1513 to 0.300), so this
                            # is a real dynamic limit, not a modelling artefact.
                            # The remaining levers change the operating point:
                            # reel-out (restores c1 = 0.3159 and 0.03 s dead time
                            # by making a low depower survivable) or a 300 m
                            # tether. See PlanFig8.md.

# Entry descent limiter. WHY (2026-07-26, first run): without it the kite dove
# straight from the 73° park to the pattern, converting 40° of potential energy
# into a 3.3x overspeed — v_app 15.7 -> 51 m/s in 7 s, AoA driven negative as the
# wing unloaded, and the solver aborted at t=7.35 s. The guidance was working
# (cross-track error 35° -> 1.7°); the descent was simply flown far too steeply.
#
# The fix limits the COMMANDED course, not its rate: while the kite is far off
# the path, never command a course steeper than ENTRY_CHI_MAX (90° = constant
# elevation, >90° = descending), so the approach is a shallow glide that drag
# can bleed instead of a plunge. It also cures a second defect seen in the same
# run: with the attractor nearly straight below, chi_set hunted across the ±180°
# branch cut (+154.8° -> -155.2° -> -153.3°) and the steering chattered between
# its clamps. Picking whichever of ±ENTRY_CHI_MAX needs the smaller heading
# change makes that choice continuous, with no latch or state machine.
#
# The limiter is gated on the cross-track error and self-disables: the pattern
# itself legitimately requires steep courses (chi_set = -118° on the path at the
# lobe crossing), so once |d| < ENTRY_D_GATE the raw guidance course passes
# through untouched. Set ENTRY_CHI_MAX = 180 to disable the limiter entirely.
ENTRY_CHI_MAX    = 95.0     # [deg] steepest commanded course while off-path.
                            # 105 -> 95 (2026-07-26): at 105° the descent from
                            # the 73° park still reached 45.6 m/s by elevation
                            # 40° (AoA -21.9°). 95° is only 5° below the local
                            # horizontal, so the kite spirals down slowly enough
                            # for drag to bleed the energy it gains.
ENTRY_D_GATE     = 12.0     # [deg] cross-track error below which it is bypassed
ENTRY_CUT_MARGIN = deg2rad(30.0)  # how close to ±180° chi_set must be before
                            # its sign is treated as degenerate and the latched
                            # tangent sign is used instead

# Abort guard: the first run's failure showed up as an opaque solver
# `dt_epsilon` abort. Catching the overspeed that causes it reports the actual
# problem instead.
V_APP_ABORT      = 45.0     # [m/s] stop the run above this apparent wind speed

# Metrics window: park plus the time allowed to settle onto the pattern before
# the tracking statistics start.
ENTRY_TIME       = 20.0     # [s] after PARK_TIME
MIN_ELEVATION    = 10.0     # [deg] floor criterion, evaluated over the WHOLE run

# ======================== INIT =========================== #

s = init(V_WIND, TETHER_LENGTH; body_damping = BODY_DAMPING,
    elevation = ELEVATION,
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
    system_yaml = PROJECT)

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]

fec = FigureEightController(FigureEightSettings(;
    dt = s.dt, A = F8_A, B = F8_B, C = F8_C, D = F8_D,
    az_center = 0.0, el_center = EL_CENTER,
    attractor_distance = ATTRACTOR_DIST, up_loops = UP_LOOPS))

# Turn-rate law of the plant actually being flown. `turn_rate_coeffs` is the
# single source for all three coefficients of
#
#     psi_dot = c1 * v_app * u_s + c2 / v_app * sin(psi) * cos(beta)
#
# at this BODY_DAMPING and DEPOWER_SETPOINT — never hardcode them, both
# arguments move them a lot (see the function's docstring). They are printed
# every run because every parameter decision below is argued against them:
#   c1     -> steering authority, hence the curvature feasibility margin
#   c2     -> the gravity/turn term the heading loop has to fight
#   delay  -> steering dead time, the limit on how fast the commanded course
#             may rotate (the lever behind ATTRACTOR_DIST and HEADING_D)
# turn_rate_coeffs interpolates for a DEPOWER_SETPOINT between identified grid
# points (see PlanC1C2.md); a run using interpolated values says so rather than
# reporting the margin as if it came from an identified one.
coeffs = turn_rate_coeffs(BODY_DAMPING, DEPOWER_SETPOINT)
c1, c2, delay = coeffs.c1, coeffs.c2, coeffs.delay
@info @sprintf("Turn-rate law at body_damping=%s, depower=%.2f%s: \
                c1 = %.4f 1/m, c2 = %.4f m/s^2, delay = %.3f s",
               BODY_DAMPING, DEPOWER_SETPOINT,
               coeffs.interpolated ? " (INTERPOLATED, not identified)" : "",
               c1, c2, delay)

# Curvature feasibility: a pattern tighter than the kite's minimum turn radius
# cannot be tracked at any PID tuning (see the docstring). c1 must match the
# body damping actually in use — that is what makes this check meaningful.
feas = check_pattern_feasible(fec, TETHER_LENGTH, MAX_STEERING; c1)
feas.feasible ||
    @warn "Pattern is tighter than the kite's minimum turn radius — expect \
           curvature-limited tracking, not a tuning problem."

# Dead-time context for ATTRACTOR_DIST. The attractor sits ATTRACTOR_DIST of arc
# ahead of the kite, so the commanded course turns over roughly the time the
# kite needs to cover that arc; `delay` is how long the plant takes to react at
# all. Reported, not enforced: the recorded failure at ATTRACTOR_DIST = 10°
# (lead 2.7 s against a 0.42 s dead time) is one data point, not a threshold.
lead_time = deg2rad(ATTRACTOR_DIST) * TETHER_LENGTH / V_APP_REF
@info @sprintf("Attractor lead %.1f° ≈ %.1f s of flight at v_app %.1f m/s, \
                vs %.2f s steering dead time (ratio %.1f).",
               ATTRACTOR_DIST, lead_time, V_APP_REF, delay, lead_time / delay)

heading_pid = create_heading_pid(;
    K = HEADING_P, Ti = HEADING_I, Td = HEADING_D, N = HEADING_D_N, dt = s.dt,
    umin = -MAX_STEERING, umax = MAX_STEERING)

el_center_cur = EL_CENTER
entry_sign = 0              # latched sign of the entry descent limiter (0 = unset)

# Entry state machine (see the parameter block). Codes match the reference
# controller's log so both can be read with the same plotting scripts:
#   0 = park, 1 = dive, 2 = hold, 3 = figure-eight guidance engaged.
phase = 0
hold_start = NaN            # [s] time the hold began

toc("Start simulation loop...")

# ==================== SIMULATION LOOP ==================== #

try
    for _ in 1:s.steps
        t = s.sys_state.time

        # Optional walk of the pattern centre (STEP 4). Runs before the
        # guidance so it sees the updated path this step.
        if WALK_RATE > 0 && t >= WALK_START && el_center_cur > EL_FINAL
            global el_center_cur = max(EL_FINAL, el_center_cur - WALK_RATE * s.dt)
            set_path_center!(fec, 0.0, el_center_cur)
        end

        # L0 attractor guidance -> commanded course [rad].
        chi_set, az_attr, el_attr, dmin =
            navigate_fig8(fec, Float64(s.sys_state.azimuth),
                          Float64(s.sys_state.elevation))

        # ---- Entry state machine (park -> dive -> hold -> fig8) ------------ #
        # Advances on elevation and time, never backwards. The guidance above
        # keeps running through all phases so `dmin`/the attractor stay logged,
        # but in phases 1-2 its course is discarded in favour of the open-loop
        # entry command below — the point of the phases is that the descent is
        # NOT flown by the path controller.
        local el_deg = rad2deg(Float64(s.sys_state.elevation))
        if !ENTRY_PHASES
            global phase = t < PARK_TIME ? 0 : 3
        elseif phase == 0 && t >= PARK_TIME
            global phase = 1
        elseif phase == 1 && el_deg <= el_center_cur + DIVE_EL_MARGIN
            global phase = 2
            global hold_start = t
        elseif phase == 2 && t - hold_start >= HOLD_TIME
            global phase = 3
        end

        # Entry descent limiter (see the parameter block). Active only while the
        # kite is far off the path; on the path the raw guidance course passes
        # through, because the pattern needs steep courses of its own.
        heading = Float64(s.sys_state.heading)
        chi_cmd = chi_set
        if dmin > ENTRY_D_GATE && abs(chi_set) > deg2rad(ENTRY_CHI_MAX)
            # chi_set is the HOMING law (great-circle course to the attractor);
            # only its steepness is limited, never its homing intent. An earlier
            # version commanded the path tangent instead — pure feed-forward
            # with no reference to the path's POSITION — and the kite flew off
            # to azimuth -65° and sat in a limit cycle for 180 s with the
            # cross-track error frozen at 24°. A course limiter must clamp the
            # homing course, not replace it.
            #
            # Sign: chi_set's own, except near the ±180° branch cut, where the
            # kite sits almost directly above the pattern, every attractor point
            # is "straight down", and the sign is numerical noise. There the
            # latched tangent-at-Q sign is used instead — it is ~±108° at the
            # park, nowhere near the cut, and encodes which way round the path
            # is traversed, so the kite arrives moving in the right direction.
            # Latching arbitrarily and then reversing mid-descent is what broke
            # the two earlier runs.
            tang = path_tangent(fec)
            entry_sign == 0 && (global entry_sign = tang >= 0 ? 1 : -1)
            sgn = abs(chi_set) < pi - ENTRY_CUT_MARGIN ?
                  (chi_set >= 0 ? 1 : -1) : entry_sign
            chi_cmd = sgn * deg2rad(ENTRY_CHI_MAX)
        end

        # Open-loop entry command. Overrides the guidance (and its limiter) for
        # the dive and the hold; positive = towards positive azimuth, matching
        # the reference controller, which enters the pattern from the right.
        if phase == 1
            chi_cmd = deg2rad(CHI_DIVE)
        elseif phase == 2
            chi_cmd = deg2rad(CHI_HOLD)
        end

        # Feedback angle: heading at low kite speed, course at high (see the
        # parameter block). Blended on the WRAPPED difference so the transition
        # stays continuous across the ±180° cut, and so the two endpoints are
        # exactly `heading` and `course`.
        #
        # In FIG8 mode the speed schedule is bypassed and the course is fed back
        # unconditionally (FIG8_PURE_COURSE). Path following is a course problem:
        # what must lie on the path is where the kite GOES, and the ~13-15° drift
        # angle means heading feedback tracks the path with a standing offset. On
        # the pattern the kite is also fast enough that the schedule asks for
        # course anyway — it just dips into the band during the slow part of a
        # turn and swaps the feedback signal mid-manoeuvre, which is the failure
        # mode recorded in SmallPlan.md (|v_kite| crossed the 10 m/s edge twice
        # in the 15-27 s window of the 2026-08-02 run, 9.8% of it inside the
        # band). The entry phases keep the schedule: during park and dive the
        # kite really can be too slow for a meaningful course.
        v_kite = norm(s.sys_state.vel_kite)
        w_course = if FIG8_PURE_COURSE && phase == 3
            1.0
        else
            clamp((v_kite - V_KITE_HEADING) /
                  (V_KITE_COURSE - V_KITE_HEADING), 0.0, 1.0)
        end
        # +π: `SysState.course` is SymbolicAWEModels' raw tangent-frame course,
        # whose zero points AWAY from zenith, while `SysState.heading` and the
        # guidance both use 0 = towards zenith. The flip is not symmetric
        # between the two fields — the V3's body x-axis is reversed w.r.t. the
        # sensor convention, which cancels the frame flip for the heading but
        # cannot for a velocity direction. MEASURED on this model over the
        # samples where the kite flies (>2°/s): with the correction,
        # course - heading is the +13..15° drift angle documented in
        # src/fig8_controller.jl; without it, -165°. Feeding the raw field back
        # is positive feedback and diverged the run at t = 19.7 s.
        course = wrap_to_pi(Float64(s.sys_state.course) + pi)
        fb = heading + w_course * wrap_to_pi(course - heading)

        # Heading PID on the WRAPPED course error. DiscretePID computes a plain
        # (r - y) difference with no ±π wrapping, so the error is formed here
        # and passed as the measurement against a zero reference; otherwise the
        # loop commands a full turn the long way round at every wrap crossing.
        err = wrap_to_pi(fb - chi_cmd)
        # Gain scheduling: turn rate ~ u_s * v_app, so K ~ 1/v_app. Still on
        # APPARENT wind speed — that is the plant gain; only the choice of
        # feedback angle above schedules on kite speed.
        v_app = max(Float64(s.sys_state.v_app), V_APP_MIN)
        set_K!(heading_pid, HEADING_P * V_APP_REF / v_app, 0.0, err)
        # Park: hold zero steering while the settling transients decay. The PID
        # is still stepped (with zero error) so its derivative state is current
        # and engagement is bumpless.
        rel_steering = if phase == 0
            heading_pid(0.0, 0.0, 0.0)
            0.0
        else
            heading_pid(0.0, err, 0.0)
        end

        # Reference governor against the REAL tape position (SmallPlan.md).
        #
        # The KCU tape slews at set.v_steering [1/s] and no faster, so a command
        # far from where the tape currently is cannot be followed — it just puts
        # the actuator on its rate limit, which costs amplitude and adds
        # amplitude-dependent phase lag (measured: 62.7° at the loop's own
        # frequency, before HEADING_P was fixed). Clamping the command to within
        # STEERING_LEAD steps of travel of the MEASURED position keeps every
        # command achievable, and the tape is never chasing a distant target.
        #
        # NOTE this is a governor, not classic anti-windup: with HEADING_I =
        # false the PID has no integral state, so there is nothing to
        # back-calculate. If integral action is ever enabled, the tracking term
        # has to be added here as well, or the integrator will wind up against
        # this clamp exactly as it would against umin/umax.
        if STEERING_TRACK_TAPE && phase != 0
            reach = STEERING_LEAD * s.set.v_steering * s.dt
            u_act = Float64(s.sys_state.steering)
            rel_steering = clamp(rel_steering, u_act - reach, u_act + reach)
        end

        # Winch: force mode pays out under load and trims the mean length back
        # slowly; position mode holds the length outright (see WINCH_FORCE_MODE).
        if WINCH_FORCE_MODE
            step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering,
                  set_torque = winch_force_torque!(s, l0),
                  vsm_interval = VSM_INTERVAL)
        else
            step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering, set_length = l0,
                  vsm_interval = VSM_INTERVAL)
        end

        # Overspeed guard: report the cause instead of letting it surface as an
        # opaque solver dt_epsilon abort a few steps later.
        if Float64(s.sys_state.v_app) > V_APP_ABORT
            @error @sprintf("Overspeed at t=%.2fs: v_app=%.1f m/s > %.1f (elevation %.1f°, AoA %.1f°). \
                             Stopping before the solver diverges.",
                            s.sys_state.time, s.sys_state.v_app, V_APP_ABORT,
                            rad2deg(s.sys_state.elevation), rad2deg(s.sys_state.AoA))
            break
        end

        # Logged after step! (which overwrites parts of sys_state). bearing is
        # the commanded course, so course - bearing is the path-following error
        # and heading - bearing is what the loop sees while w_course = 0.
        s.sys_state.sys_state = Int16(phase)   # 0 park, 1 dive, 2 hold, 3 fig8
        s.sys_state.bearing = chi_cmd          # the course actually tracked
        s.sys_state.attractor .= (deg2rad(az_attr), deg2rad(el_attr))
        s.sys_state.var_01 = dmin              # cross-track error [deg]
        s.sys_state.var_02 = az_attr           # attractor azimuth [deg]
        s.sys_state.var_03 = el_attr           # attractor elevation [deg]
        s.sys_state.var_04 = el_center_cur     # pattern-centre elevation [deg]
        s.sys_state.var_05 = chi_set           # RAW guidance course [rad]
        s.sys_state.var_06 = rad2deg(err)      # REGULATED error [deg]
        s.sys_state.var_07 = chi_cmd == chi_set ? 0.0 : 1.0  # entry limiter active
        s.sys_state.var_08 = w_course          # course/heading blend weight [-]
        # Whole-wing AoA. `sys_state.AoA` is the CENTRE PANEL only, which is
        # representative while the wing is loaded symmetrically but not in a
        # turn, where steering twists the two halves in opposite directions —
        # and this pattern is flown with the steering on its clamp nearly all
        # the time. The plots show both so the gap between them is visible.
        s.sys_state.var_09 = rad2deg(span_mean_aoa(s.sys))
    end
catch exc
    # `exc`, not `e`: a stray global `e` in the REPL turns the catch binding
    # into a soft-scope local and emits a confusing warning on every run.
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(exc, catch_backtrace())
end

@info "Save the log"
save_log(s.logger, "fig8_run"; colmeta = timestamp_colmeta())

# ==================== RESULTS ==================== #

syslog = load_log("fig8_run")
sl = syslog.syslog
print_fig8_metrics(sl; t_start = PARK_TIME, settle_time = ENTRY_TIME,
                   min_elevation = MIN_ELEVATION, az_center = 0.0)

# Plots come up with the run — the plotting script reuses the F8_* constants
# defined above, so the reference overlay always matches the pattern flown.
# `SHOW_PLOTS = false` in the REPL suppresses them, which is what makes a sweep
# bearable: three GLMakie windows per run adds up fast.
@isdefined(SHOW_PLOTS) || (SHOW_PLOTS = true)
SHOW_PLOTS && include(joinpath(@__DIR__, "simple_fig8_plots.jl"))

nothing
