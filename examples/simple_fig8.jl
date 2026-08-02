# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Figure-of-eight path following of the V3 kite via the `init`/`step!` interface
(see PlanFig8.md).

The kite starts parked at ~73° and reaches the pattern through a four-phase
entry (park -> dive -> hold -> fig8, see ENTRY_PHASES). Once engaged, the L0
attractor guidance (`src/fig8_controller.jl`) commands a course and a PID (as in
`simple_sinus.jl` / `simple_auto_parking.jl`) tracks it with the steering tape.

# Why the pattern is large and low

Neither is a free choice. The V3's identified turn-rate law fixes the smallest
angular turn radius the kite can fly:

    rho = 1 / (L * c1 * u_s)        [rad]   -- apparent wind speed cancels

`c1` depends strongly on the `body_damping` passed to `init` (see
`V3_TURN_RATE_COEFFS`): across the identified range it changes the achievable
turn radius by 5.6x, from 6.9° at `[0, 0, 40]` to 38.5° at `[20, 20, 40]`. In
plane body damping is therefore a FLIGHT parameter here, not a solver setting.

The tightest curvature of a lemniscate in (azimuth, elevation) is not at the
lobe tip but on the lobe's upper shoulder, and it collapses as the pattern is
raised, because the azimuth axis is compressed by `cos(elevation)`. At
`c1 = 0.3159` and `u_s = 0.175`:

    A=45 B=20, centre 73° -> tightest path radius  0.4°   (margin 0.06)
    A=45 B=20, centre 60° ->                       3.1°   (margin 0.45)
    A=45 B=20, centre 35° ->                       8.5°   (margin 1.23)
    A=50 B=25, centre 30° ->                      10.5°   (margin 1.52)

So a figure-eight near zenith is geometrically impossible for this kite at any
PID tuning: the pattern must be flown low and wide, and the kite descends onto
it. `check_pattern_feasible` prints the margin at startup. Below ~1 the tracking
error is curvature-limited and no PID tuning will fix it — enlarge the pattern,
lower its centre, lower the damping, or raise `MAX_STEERING`.

Logs the run to "fig8_run" (a dedicated name, not the shared "tmp_run"). For
verification run `include("examples/simple_fig8.jl")` — even a 30 s simulation
is several thousand `step!` calls and takes minutes of wall time. The script
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
| `var_07` | entry descent limiter weight (0 = raw guidance, 1 = fully limited) |
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

The dated record of how these parameters were arrived at — sweeps, reverted
attempts and the failures behind each closed lever — is in
`docs/fig8_tuning_log.md`. Add new findings there, not here.
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
SIM_TIME         = 150.0    # Total simulation time [s]; ~43 s per lap at v_app
                            # 13 m/s, plus the descent from the park. The metrics
                            # window opens at PARK_TIME + ENTRY_TIME = 25 s, so a
                            # 30 s run scores only its last 5 s and `laps` is
                            # meaningless — judge short runs on RMS d, the
                            # elevation floor and the tape metrics, and use 150
                            # for anything that has to be counted in laps.
DT               = 0.05/6   # Simulation timestep [s]. NOT a tuning parameter:
                            # 0.05/3 was numerically unstable here. `step!` holds
                            # the VSM aero load frozen inside the DAE between
                            # updates, and that explicit coupling develops a
                            # growing 2*dt (30 Hz) structural oscillation at
                            # maximum dynamic pressure — measured at the bottom of
                            # the right lobe, ~3.2 kN, which ended a run in one
                            # timestep. Halving dt halves the aero lag and gives
                            # ~4x margin on the mode, at 2x wall time.
VSM_INTERVAL     = 1        # Steps between VSM aero updates; the load is held
                            # frozen inside the DAE in between (0 disables the
                            # update entirely). 1 is the TIGHTEST coupling
                            # available, so this can only be raised — trading aero
                            # lag for wall time. Exposed to sweep the coupling
                            # mode described under DT, not as a lever that can
                            # stabilize it.
V_WIND           = 5.0     # Ground wind speed at reference height [m/s]
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
                            # The minimum angular turn radius is
                            # rho = 1/(L*c1*u_s), so a LONGER tether lets the kite
                            # turn tighter in angular terms — the most effective
                            # lever on pattern feasibility after c1 itself.
DEPOWER_SETPOINT = 0.26     # Depower setting held during the run [-]. Sets the
                            # operating point of the turn-rate law: 0.25 is agile
                            # (c1 = 0.3159) but cannot survive a sustained turn,
                            # 0.55 survives but is far too sluggish
                            # (c1 = 0.1071, dead time 0.55 s) and flies a circle
                            # instead of the pattern. 0.40 gives c1 = 0.1513 and a
                            # 0.42 s dead time. Lowering it to 0.36 has been tried
                            # twice, most recently under the current loop, and
                            # both times the run diverged on energy — reel-out
                            # does not make it survivable at 200 m.
# NOTE: ELEVATION currently has NO EFFECT on where the run starts. `settle_wing`'s
# cache key does not include the settling elevation, so the existing 73° geometry
# is reused (verified: logged elevation at t = 0 is 73.0°). Forcing `remake=true`
# would overwrite a cache shared with simple_sinus.jl / simple_parking.jl. Fixing
# it means adding the elevation to the cache key in stabilization.jl — see
# PlanFig8.md, Findings 4. Left in place because starting on the pattern is the
# right way to develop the pattern controller once the key is fixed.
ELEVATION        = 73.0     # [deg] settling elevation = the natural parked
                            # equilibrium, so the kite starts where it wants to be
# Parking phase: hold zero steering so the transients left by init/settling
# decay before the controller starts demanding maneuvers. Without it the
# guidance engaged at t=0 and drove the steering straight to its clamp while the
# model was still relaxing. The guidance still runs during the park (its course
# estimate is low-passed and needs warming up), but its output is not applied.
PARK_TIME        = 5.0      # [s]
# Warm-up, run INSIDE `init` and discarded (see `warmup!`). The park above lets
# the settling transients decay; this lets them decay BEFORE t = 0, so they are
# not in the log at all. They are not the run's data: `settle_wing` returns an
# equilibrium of the settling model (dt = 0.001, damped, winch braked) and the
# first second of the run is that state relaxing into an equilibrium of the
# model actually being integrated — the brake released, the drum taking up the
# load at its own torque, the aero applied at the run's dt. It showed up most
# sharply in the logged L/D, which is a ratio of two forces that both dip while
# the wing is unloaded. Costs WARMUP_TIME / DT full steps of wall time (240 at
# 2 s), and must be long enough to cover the decay — the transient under
# investigation peaked at t = 0.66 s. 0.0 disables it.
WARMUP_TIME      = 2.0      # [s]

# ---- Entry state machine: park -> dive -> hold -> fig8 -------------------- #
# Modelled on the working controller's log (SmallPlan.md, "Reference run"). That
# controller does NOT let the path guidance fly the descent: from the park it
# commands a near-horizontal course open loop and lets the kite fall along the
# sphere (no attractor at all — the logged attractor is NaN until handover),
# flattens out for the last second, and hands over at the pattern's RIGHTMOST
# point, at the centre elevation, already moving downwards into the first turn.
#
# WHY here: with the guidance flying the entry, every configuration tried ended
# up orbiting the right-hand lobe, with zero centre crossings against a ±50°
# reference centred on zero. The eight is symmetric; the way it is entered is
# not. The kite arrives from the park on the right and never gets established on
# the left lobe.
#
# Reference timings (its park is 10 s, ours 5 s): dive 5.6 s covering 71 -> 42°
# of elevation (~5.2°/s), hold 1.2 s covering the last 42 -> 27° (the kite is
# fastest here, so this is the steepest part), handover at the centre elevation.
#
# NOTE the entry is TURN-RATE LIMITED at this depower: the steering command sits
# at its clamp for the whole dive, so only the SIGN of CHI_DIVE reaches the
# plant and the entry cannot be shaped by the three constants below — the only
# entry choice that exists is which side to enter from. See the tuning log.
ENTRY_PHASES     = true     # false = old behaviour, guidance engages at PARK_TIME
CHI_DIVE         = -85.0    # [deg] course commanded during the dive. |chi| > 90
                            # is descending, |chi| < 90 climbing, 90 exactly
                            # horizontal. SIGN, measured: a POSITIVE commanded
                            # course drives the kite towards NEGATIVE azimuth, and
                            # the reference enters at the pattern's rightmost
                            # point, so the command is negative here.
CHI_HOLD         = -90.0    # [deg] course commanded during the hold: exactly
                            # horizontal, i.e. stop descending and let the kite
                            # arrive at the pattern flat rather than diving into it
DIVE_EL_MARGIN   = 7.0     # [deg] above EL_CENTER at which the dive ends and the
                            # hold begins (reference: 42° vs a 26° centre = 16°)
HOLD_TIME        = 0.8      # [s] duration of the hold, from the reference log

# In-plane body damping is a FLIGHT parameter here, not just a solver setting:
# it sets c1 and hence the achievable turn radius (see the docstring). init's
# default [0,0,40] is the most agile and the only one that flies this pattern
# inside the identified steering range. Raising it costs turn authority; it buys
# a smaller parked AoA ripple and ~3.4x fewer solver steps (see `init`).
BODY_DAMPING     = [0.0, 0.0, 40.0]

# Pattern geometry [deg]. Sized by the turn-radius argument in the docstring;
# check the feasibility margin printed at startup before changing these. Note a
# SMALLER lemniscate is a TIGHTER one: the reference controller's 40/15 drops
# the margin to 1.02 and does not fly here.
F8_A             = 50.0     # Width of the eight (azimuth spans +-A)
F8_B             = 20.0     # Height of the eight (elevation spans +-B/2)
F8_C             = 0.0      # Size of the right part
F8_D             = 0.0      # Asymmetry factor
EL_CENTER        = 26.0     # Pattern-centre elevation; spans 16-36° at B=20.
                            # The reference controller's centre, and the lowest
                            # one flown here. Two forces pull opposite ways: a
                            # lower centre IMPROVES the curvature margin (less
                            # cos(elevation) compression of the azimuth axis) but
                            # pushes the pattern deeper into the power zone, and
                            # every failure at low centre has been an ENERGY
                            # failure (v_app and force run away while the tracking
                            # still looks fine).
                            #
                            # WHAT BINDS NOW is the pattern's BOTTOM EDGE, not the
                            # centre: at B = 20 the bottom sits at 16° and the
                            # kite undershoots it by ~5° on EVERY lap, leaving
                            # 0.9° of margin on MIN_ELEVATION and producing the
                            # largest cross-track errors. Next lever is B (and
                            # probably A) coming down with the centre — the
                            # reference flies A = 40, B = 15 at this centre.

ATTRACTOR_DIST   = 10        # Arc distance Q -> attractor [deg]. Guarded with
                            # `@isdefined` so a sweep driver can set it in the
                            # REPL before `include`ing this file and have its
                            # value survive (same pattern as F8_* in
                            # simple_fig8_plots.jl); the number here is the
                            # default for a plain run. NOTE the footgun: a stale
                            # ATTRACTOR_DIST left in the REPL by a sweep silently
                            # wins over this value. The startup @info line prints
                            # the lead actually used — read it.
                            #
                            # Swept 10..20: RMS d rises monotonically with the
                            # lead (2.85 -> 3.58°) while force ripple falls the
                            # other way, and there is a survival cliff between
                            # 16.11 and 17.72 (both die at ~41 s). 12.1 is the
                            # shortest lead giving the full 4 laps and 8 centre
                            # crossings; a shorter lead flies a fuller eight, not
                            # merely a better-tracked one.
UP_LOOPS         = false    # Fly DOWN-loops: at large |azimuth| the kite passes
                            # the azimuth extreme moving downwards. The flag
                            # reverses the traversal direction of the reference
                            # path (`_build_path` in src/fig8_controller.jl);
                            # the path shape itself is unchanged, so the
                            # curvature feasibility margin is unaffected.
                            # Down-loops convert height into speed where up-loops
                            # shed energy through the turn, so they were unflyable
                            # on the old heading loop; on course feedback they are
                            # the only configuration that crosses the centre
                            # instead of circling one lobe.

# Optional walk of the pattern centre (PlanFig8.md STEP 4). 0 disables it; the
# run then flies the whole time at EL_CENTER. Use it to move the pattern to a
# lower, more force-optimal centre after a stable capture.
WALK_RATE        = 0.0      # [deg/s] rate to walk the centre towards EL_FINAL
EL_FINAL         = 25.0     # [deg] final pattern-centre elevation
WALK_START       = 60.0     # [s] time after which the walk begins

# Heading PID. Output is rel_steering (dimensionless, -1..1), fed UNNEGATED:
# positive rel_steering produces a positive heading rate on this plant
# (measured, r = +0.998 — see src/fig8_controller.jl).
HEADING_P        = 0.6      # Gain at v_app == V_APP_REF. DERIVED, not tuned: the
                            # plant psi_dot = c1*v_a*u_s is an INTEGRATOR of gain
                            # c1*v_a = 3.66 rad/s per unit u_s at flight speed, so
                            # the crossover is omega_c = K*3.66. Against the
                            # 0.72 s measured tape lag a delay needs
                            # omega_c*T_d <~ 0.8 rad, giving HEADING_P ~ 0.46; the
                            # optimistic 0.383 s small-signal dead time gives
                            # 0.86. 0.6 sits between the two.
                            #
                            # The earlier 4.5 was ~8x over gain, i.e. a relay:
                            # it clamped at 6.7° of course error, exceeded by 88%
                            # of phase-3 samples, and the kite turned at a median
                            # 43.5 deg/s against the 8.3 deg/s the guidance asked
                            # for. Everything measured at that gain describes a
                            # self-oscillating loop, not tracking.
HEADING_I        = false    # No integral action: a steady heading bias shows up
                            # as a steady cross-track error, which the guidance
                            # itself already corrects by pulling the attractor
                            # back onto the path. Try a finite Ti only if a
                            # persistent one-sided cross-track offset remains.
HEADING_D        = 0.15     # Derivative time [s], damps the initial transient
HEADING_D_N      = 2.0      # Derivative filter: maximum gain of the D path,
                            # which is K*Td*s/(1 + s*Td/N). Flat at K below
                            # N/(2*pi*Td) Hz, rising to N*K above it. 2 rather
                            # than the DiscretePIDs default of 10: the fed-back
                            # angles carry broadband noise, and at N = 10 the
                            # rising D gain amplified it into a 7.95 Hz ripple on
                            # the command. At the loop's own 0.1 Hz the D path
                            # contributes a gain of 1.005 and 5.4° of phase lead
                            # either way, so this is a filter change, not a gain
                            # change: same flight, 33% less peak tape slew.
V_APP_REF        = 13.1     # Reference apparent wind speed for the schedule [m/s]
V_APP_MIN        = 10.0      # Lower clamp on v_app, limits the gain boost [m/s]
ENTRY_GAIN       = 0.25     # Factor on HEADING_P during the ENTRY phases (dive
                            # and hold, phases 1-2); phase 3 flies at the full
                            # gain. The entry is turn-rate limited — the steering
                            # command sits on its clamp for the whole dive — so
                            # detuning the loop there costs nothing in tracking
                            # and takes the command off the clamp, which is the
                            # only way to shape the descent from the loop side.
ENTRY_DEPOWER    = 0.34     # Depower held during the ENTRY phases (dive and
                            # hold, phases 1-2); the park and phase 3 fly at
                            # DEPOWER_SETPOINT. A higher depower than the
                            # pattern's is the second lever on the descent: it
                            # lowers c1 (less turn authority, which the entry
                            # does not need — it is clamp-limited anyway) and
                            # unloads the wing, bleeding some of the energy the
                            # dive from the 73° park converts out of height.
                            # The park is excluded on purpose: `init` settles at
                            # DEPOWER_SETPOINT, and changing the tape during the
                            # park would inject exactly the transient the park
                            # exists to let decay. Both transitions are rate
                            # limited by the KCU tape speed inside `step!`, so
                            # the change is a ramp, not a step.

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
V_KITE_COURSE    = 10.0     # [m/s] at/above: pure course feedback ("high" per
                            # the flight note in SmallPlan.md; the lower edge is
                            # a choice — 5 m/s is just above the 4.2 m/s the
                            # kite drifts at during the park). With
                            # FIG8_PURE_COURSE on, this band governs the ENTRY
                            # only.
FIG8_PURE_COURSE = false    # In FIG8 mode (phase 3), feed back COURSE alone and
                            # ignore the V_KITE_* schedule; the entry phases keep
                            # it. This is SmallPlan.md's "gate on phase instead
                            # of speed" option. Rationale: path following is a
                            # course problem, and on the pattern the kite is fast
                            # enough that the schedule asks for course anyway —
                            # it only dips into the band during the slow part of
                            # a turn, swapping the feedback signal mid-turn for
                            # no benefit. `false` restores the pure speed
                            # schedule in every phase.
MAX_STEERING     = 0.32     # Steering command limit [-]. Raising it to relieve
                            # the clamp saturation is CLOSED: at 0.33 the loop
                            # goes violently unstable (diverged at t = 30.9 s,
                            # peak turn rate 949 deg/s) and at 0.375 the PLANT
                            # itself diverges in bang-bang oscillation with no
                            # controller at all. c1 is linear over the range, so
                            # this is a real dynamic limit, not a modelling
                            # artefact — the usable authority ceiling is a
                            # property of the plant at this depower. The remaining
                            # levers change the operating point: reel-out or a
                            # 300 m tether. See PlanFig8.md.

# Entry descent limiter. WHY: without it the kite dove straight from the 73° park
# to the pattern, converting 40° of potential energy into a 3.3x overspeed —
# v_app 15.7 -> 51 m/s in 7 s, AoA driven negative as the wing unloaded, and the
# solver aborted at t=7.35 s. The guidance was working (cross-track error 35° ->
# 1.7°); the descent was simply flown far too steeply.
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
#
# The handover is BLENDED over ENTRY_D_BLEND, not switched. As a hard gate it
# stepped the commanded course by the full clamp violation in one timestep, and
# the PID's D path turned that step into a spike that reversed the sign of the
# command — see the tuning log, ENTRY_D_BLEND.
ENTRY_CHI_MAX    = 95.0     # [deg] steepest commanded course while off-path. At
                            # 105° the descent from the 73° park still reached
                            # 45.6 m/s by elevation 40°; 95° is only 5° below the
                            # local horizontal, so the kite spirals down slowly
                            # enough for drag to bleed the energy it gains.
ENTRY_D_GATE     = 12.0     # [deg] cross-track error below which it is bypassed
ENTRY_D_BLEND    = 4.0      # [deg] width of the band ABOVE ENTRY_D_GATE over
                            # which the limited and raw courses are blended:
                            # fully limited at d >= GATE + BLEND, fully raw at
                            # d <= GATE. 0 restores the old hard switch. Sized
                            # against the rate d closes at (~3.4 deg/s here), so
                            # 4° is ~1.2 s of traversal — 16x slower than the
                            # 0.075 s derivative filter, hence tracked rather
                            # than differentiated (D contribution ~0.05 instead
                            # of the +0.73 the step produced). It also makes
                            # CHATTER on the gate harmless: d is not monotonic,
                            # and a hard switch re-fires the full step on every
                            # recrossing.
ENTRY_CUT_MARGIN = deg2rad(30.0)  # how close to ±180° chi_set must be before
                            # its sign is treated as degenerate and the latched
                            # tangent sign is used instead

# Abort guard: the first run's failure showed up as an opaque solver
# `dt_epsilon` abort. Catching the overspeed that causes it reports the actual
# problem instead.
V_APP_ABORT      = 45.0     # [m/s] stop the run above this apparent wind speed

# Metrics window: park plus the time allowed to settle onto the pattern before
# the tracking statistics start.
ENTRY_TIME       = 48.0     # [s] after PARK_TIME
MIN_ELEVATION    = 10.0     # [deg] floor criterion, evaluated over the WHOLE run

# ======================== INIT =========================== #

s = init(V_WIND, TETHER_LENGTH; body_damping = BODY_DAMPING,
    elevation = ELEVATION,
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
    system_yaml = PROJECT,
    # The warm-up must relax against the winch the loop below will command,
    # or it hands the run the very discontinuity it exists to remove.
    warmup_time = WARMUP_TIME, warmup_force_mode = WINCH_FORCE_MODE)

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
# every run because every parameter decision above is argued against them:
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
        # Limiter weight: 1 = fully limited, 0 = raw guidance, linear in between
        # over ENTRY_D_BLEND above the gate. With ENTRY_D_BLEND = 0 this is the
        # old hard switch (the `>` keeps d == GATE on the raw side either way).
        w_lim = ENTRY_D_BLEND > 0 ?
                clamp((dmin - ENTRY_D_GATE) / ENTRY_D_BLEND, 0.0, 1.0) :
                (dmin > ENTRY_D_GATE ? 1.0 : 0.0)
        if w_lim > 0 && abs(chi_set) > deg2rad(ENTRY_CHI_MAX)
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
            chi_lim = sgn * deg2rad(ENTRY_CHI_MAX)
            # Blend on the WRAPPED difference, exactly as the heading/course
            # feedback blend below: chi_lim and chi_set can straddle the ±180°
            # cut (the limiter exists partly because chi_set hunts across it),
            # and a plain convex combination would then sweep the command the
            # long way round through zero. This form is continuous there and
            # still returns chi_lim exactly at w_lim = 1 and chi_set exactly
            # at w_lim = 0, so the blend adds no offset in either limit.
            chi_cmd = wrap_to_pi(chi_set + w_lim * wrap_to_pi(chi_lim - chi_set))
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
        # The entry phases run at ENTRY_GAIN * HEADING_P; the pattern itself
        # (phase 3) at the full gain.
        v_app = max(Float64(s.sys_state.v_app), V_APP_MIN)
        K_phase = phase == 3 ? HEADING_P : ENTRY_GAIN * HEADING_P
        set_K!(heading_pid, K_phase * V_APP_REF / v_app, 0.0, err)
        # Park: hold zero steering while the settling transients decay. The PID
        # is still stepped (with zero error) so its derivative state is current
        # and engagement is bumpless.
        rel_steering = if phase == 0
            heading_pid(0.0, 0.0, 0.0)
            0.0
        else
            heading_pid(0.0, err, 0.0)
        end

        # Depower: ENTRY_DEPOWER during the dive and the hold, DEPOWER_SETPOINT
        # during the park and on the pattern (see the parameter block).
        rel_depower = (phase == 1 || phase == 2) ? ENTRY_DEPOWER : DEPOWER_SETPOINT

        # Winch: force mode pays out under load and trims the mean length back
        # slowly; position mode holds the length outright (see WINCH_FORCE_MODE).
        if WINCH_FORCE_MODE
            step!(s; rel_depower, rel_steering,
                  set_torque = winch_force_torque!(s, l0),
                  vsm_interval = VSM_INTERVAL)
        else
            step!(s; rel_depower, rel_steering, set_length = l0,
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
        # Entry limiter weight, not a flag: 0 = raw guidance course, 1 = fully
        # limited, fractional inside the ENTRY_D_BLEND band. Logged as the
        # weight so the handover is visible as the ramp it now is — a plot that
        # still shows a step here means ENTRY_D_BLEND is too narrow for the rate
        # d is closing at.
        s.sys_state.var_07 = abs(chi_set) > deg2rad(ENTRY_CHI_MAX) ? w_lim : 0.0
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
