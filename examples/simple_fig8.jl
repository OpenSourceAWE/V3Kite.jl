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

`bearing` carries `chi_cmd`, the course the loop actually tracks, so
`course - bearing` is the path-following error; the unmodified guidance course
is kept in `var_05`. The feedback angle the PID regulates is heading at low
apparent wind speed and course at high (`V_APP_HEADING`/`V_APP_COURSE`), so
`var_06` equals `heading - bearing` at low speed and `course - bearing` at
high.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers; tic()
using V3Kite
using DiscretePIDs: set_K!
using Printf

@info "simple_fig8.jl: figure-of-eight path following of the V3 kite."

# ==================== USER PARAMETERS ==================== #

PROJECT          = "system_reelout.yaml"  # System project (see data/system_*.yaml)
SIM_TIME         = 30.0     # Total simulation time [s]; ~43 s per lap at
                            # v_app 13 m/s, plus the descent from the park.
                            # Capped at 30 s for the tuning campaign, one run
                            # per parameter change. CAUTION: the metrics window
                            # opens at PARK_TIME + ENTRY_TIME = 25 s, so the
                            # tracking statistics are computed over the last 5 s
                            # only and `laps` cannot reach its 3.0 criterion at
                            # any tuning. At this length the run answers "does
                            # it survive the entry, and where does it arrive",
                            # not "how well does it track" — for the latter
                            # lengthen the run or shorten ENTRY_TIME.
DT               = 0.05/3   # Simulation timestep [s]
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 200.0    # Tether length [m], held constant (position mode).
                            # 150 -> 200 (2026-07-26): the minimum angular turn
                            # radius is rho = 1/(L*c1*u_s), so a LONGER tether
                            # lets the kite turn tighter in angular terms — the
                            # single most effective lever on pattern
                            # feasibility after c1 itself.
DEPOWER_SETPOINT = 0.40     # Depower setting held during the run [-]. The
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
F8_C             = 0.0      # Size of the right part
F8_D             = 0.0      # Asymmetry factor
EL_CENTER        = 50.0     # Pattern-centre elevation; spans 40-60° at B=20.
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
ATTRACTOR_DIST   = 19.8     # Arc distance Q -> attractor [deg]. Swept in 10%
                            # steps at depower 0.40 / 200 m / centre 50°:
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
                            # 19.8 chosen as the best of the swept values.
                            # The 97% steering saturation in EVERY case is the
                            # real constraint: the crossover is authority-limited.
                            # Interacts strongly with HEADING_P; tune jointly.
UP_LOOPS         = true     # Fly upwards during the turns at large |azimuth|.
                            # MEASURED on the V3 (2026-07-26), not inherited:
                            # true -> survives 200 s; false -> diverges at 17.8 s,
                            # all else equal. Up-loops shed energy through the
                            # turn where down-loops convert height into speed,
                            # which decides it here because the failure mode is
                            # overspeed.

# Optional walk of the pattern centre (PlanFig8.md STEP 4). 0 disables it; the
# run then flies the whole time at EL_CENTER. Use it to move the pattern to a
# lower, more force-optimal centre after a stable capture.
WALK_RATE        = 0.0      # [deg/s] rate to walk the centre towards EL_FINAL
EL_FINAL         = 25.0     # [deg] final pattern-centre elevation
WALK_START       = 60.0     # [s] time after which the walk begins

# Heading PID. Output is rel_steering (dimensionless, -1..1), fed UNNEGATED:
# positive rel_steering produces a positive heading rate on this plant
# (measured, r = +0.998 — see src/fig8_controller.jl).
HEADING_P        = 5.0      # Gain at v_app == V_APP_REF (simple_sinus.jl value)
HEADING_I        = false    # No integral action: a steady heading bias shows up
                            # as a steady cross-track error, which the guidance
                            # itself already corrects by pulling the attractor
                            # back onto the path. Try a finite Ti only if a
                            # persistent one-sided cross-track offset remains.
HEADING_D        = 0.15     # Derivative time [s], damps the initial transient
V_APP_REF        = 13.1     # Reference apparent wind speed for the schedule [m/s]
V_APP_MIN        = 5.0      # Lower clamp on v_app, limits the gain boost [m/s]

# WHAT THE LOOP REGULATES: heading at low apparent wind speed, course at high.
# The guidance commands a COURSE, so course is the signal that actually closes
# the path-following loop and is what must be regulated once the kite is flying
# fast. At low v_app it is the wrong feedback: the flight-path direction is the
# small difference of two nearly equal velocities there, so it is noisy and
# ill-defined (undefined at v_app = 0, e.g. during the park), while heading
# stays clean and still has the right sign of steering authority. Regulating
# course throughout also asks the loop to chase the drift angle, which the kite
# cannot change directly.
#
# Below V_APP_HEADING the error is formed from heading alone, above V_APP_COURSE
# from course alone, and linearly blended in between. The band exists to avoid a
# hard switch: heading and course differ by the drift angle (~13-15° on the V3),
# so a step change of feedback signal at one speed would step the steering
# command by HEADING_P * drift. Widen the band if that transient still shows.
V_APP_HEADING    = 5.0      # [m/s] at/below: pure heading feedback
V_APP_COURSE     = 10.0     # [m/s] at/above: pure course feedback ("high" per
                            # the flight note; the lower edge is a choice, set
                            # to V_APP_MIN so the blend spans the whole range
                            # over which the gain schedule is already clamped)
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
    K = HEADING_P, Ti = HEADING_I, Td = HEADING_D, dt = s.dt,
    umin = -MAX_STEERING, umax = MAX_STEERING)

el_center_cur = EL_CENTER
entry_sign = 0              # latched sign of the entry descent limiter (0 = unset)

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

        # Feedback angle: heading at low apparent wind speed, course at high
        # (see the parameter block). Blended on the WRAPPED difference so the
        # transition stays continuous across the ±180° cut, and so the two
        # endpoints are exactly `heading` and `course`.
        v_app_raw = Float64(s.sys_state.v_app)
        w_course = clamp((v_app_raw - V_APP_HEADING) /
                         (V_APP_COURSE - V_APP_HEADING), 0.0, 1.0)
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
        # Gain scheduling: turn rate ~ u_s * v_app, so K ~ 1/v_app.
        v_app = max(v_app_raw, V_APP_MIN)
        set_K!(heading_pid, HEADING_P * V_APP_REF / v_app, 0.0, err)
        # Park: hold zero steering while the settling transients decay. The PID
        # is still stepped (with zero error) so its derivative state is current
        # and engagement is bumpless.
        rel_steering = if t < PARK_TIME
            heading_pid(0.0, 0.0, 0.0)
            0.0
        else
            heading_pid(0.0, err, 0.0)
        end

        # Position mode: `set_length` holds the tether length constant.
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering, set_length = l0)

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
    end
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
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
include(joinpath(@__DIR__, "simple_fig8_plots.jl"))

nothing
