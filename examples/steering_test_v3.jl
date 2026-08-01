# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Identify the steering response of the V3 kite: port of `steering_test_4p.jl`
from KiteModels.jl to the V3 model and the high-level `init`/`step!` interface.

The maneuver is the bang-bang heading oscillation of the original script. The
kite is settled and parked at a constant tether length (`step!` in POSITION
MODE via `set_length`, as in `simple_parking.jl`/`simple_sinus.jl`), then a
relay controller flips the steering between `−u_s` and `+u_s` whenever the
heading leaves a `±HEADING_OFFSET` band. Each time the heading crosses the
upper edge going up, the amplitude `u_s` is stepped up by `STEERING_STEP`, so
one run sweeps `START_STEERING .. MAX_STEERING` and the log covers a range of
steering amplitudes at a roughly constant flight state. The run ends when the
sweep is done (the intended exit), when `SIM_TIME` runs out, or when the
elevation drops below `MIN_ELEVATION`.

Afterwards the turn-rate law is fitted (see `src/turn_rate_id.jl`) and the three
success criteria are reported: completion, minimum elevation, and the scatter
of the identified turn-rate gain.

Differences from the KPS4 original, all forced by the V3 model:

- Amplitudes are much smaller. `V3_STEERING_GAIN = 1.4` m of differential tape
  at `|u_s| = 1`, and the range the other V3 examples stay inside is
  `|u_s| ≲ 0.175` (the ±40° sinus of `simple_sinus.jl` peaks at 0.078), so the
  0.1 → 0.5 sweep of the original would leave the calibrated geometry.
- The KCU actuator is slew-limited (`v_steering = 0.2` s⁻¹, `steering_gain =
  3.0`), so a reversal of amplitude `u_s` takes `2·u_s/0.2` seconds — 2.5 s at
  `u_s = 0.25`. The heading therefore overshoots the `±HEADING_OFFSET`
  band by a wide margin, and the tape traces a triangle wave that never quite
  reaches the command. Both are harmless here and in fact helpful: the
  overshoot is what excites the `c2` (gravity) term of the turn-rate law, and
  the identification fits the *actual* tape position `sl.steering`, not the
  command.
- The turn rate is not logged by this script. `step!` already fills
  `heading_rate`, and `var_15`/`var_16` (used by the original for the turn rate
  and the side-plate AoA) are taken by L/D_wing and L/D_eff. The analysis
  recomputes the frame-transport-corrected turn rate with
  `V3Kite.calc_turn_rate`, which is the quantity the turn-rate law is written
  in. The V3 wing has no side plates, so there is no `alpha_3`/`alpha_4`
  equivalent of the original's AoA plot — `sl.AoA` is plotted instead.

Logs the run to "tmp_steering". For verification, run
`include("examples/steering_test_v3.jl")`, check the printed report, then
`include("examples/steering_test_v3_plots.jl")`. A full sweep is several
thousand `step!` calls and takes minutes of wall time.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers; tic()
using V3Kite
import KiteUtils   # for KiteUtils.syslog; V3Kite does not re-export it
using Printf

@info "steering_test_v3.jl: identifying the V3 steering response."

# ==================== USER PARAMETERS ==================== #

PROJECT          = "system_reelout.yaml"  # System project (see data/system_*.yaml)
SIM_TIME         = 200.0    # Simulation time limit [s]; sized so the sweep ends first
                            # (~15 s per amplitude level, plus T_START)
DT               = 0.05/3   # Simulation timestep [s]
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 150.0    # Initial tether length [m]
DEPOWER_SETPOINT = 0.25     # Depower setting held during the run [-]

# Relay controller. The KPS4 original uses a ±5° band; it is widened here
# because `sin(ψ)` has to leave zero for the gravity term `c2` of the turn-rate
# law to be identifiable. With the slew-limited V3 tape the heading overshoots
# the band substantially anyway, which sets the oscillation amplitude.
T_START          = 10.0     # Start of the excitation [s]; before that u_s = 0
HEADING_OFFSET   = 10.0      # Heading band edge [deg]; steering flips outside ±this
START_STEERING   = 0.05     # First steering amplitude [-]
STEERING_STEP    = 0.025    # Amplitude increment per completed cycle group [-]
MAX_STEERING     = 0.175    # Last amplitude [-]; the run ends after this one
# Upward band crossings (≈ oscillation cycles) spent at each amplitude before it
# is stepped up. 1 reproduces the KPS4 original exactly; 2 doubles the samples
# per amplitude, which is what the gain-scatter criterion needs.
CYCLES_PER_LEVEL = 2

# Success criteria (see the plan): elevation floor and the maximum relative
# standard deviation of the identified turn-rate gain.
MIN_ELEVATION    = 50.0     # [deg]
MAX_REL_STD      = 0.35     # [-]

# G mask: samples with |u_s| below this are dropped from the gain statistic,
# where the ratio psi_dot/(v_a*u_s) is near-singular. Half the smallest swept
# amplitude, so every amplitude level contributes.
MIN_STEERING_FIT = START_STEERING / 2

# ======================== INIT =========================== #

# `body_damping` matches simple_parking.jl (the `init` default is [0, 0, 40]); it
# is part of the settling cache key, so the first run with it re-settles the wing.
# The in-plane (x, y) terms damp the bridle oscillation but also suppress the
# steering deformation: c1, the turn-rate gain, falls by 3x per damping level.
# with [20.0, 20.0, 40.0]
#  c1                 :   0.0567 1/m ± 0.0000  (0.06 %)
#  c2                 :  -2.0841 [-] ± 0.0167  (0.80 %)
# with [10.0, 10.0, 40.0]
#  c1                 :   0.0982 1/m ± 0.0001  (0.07 %)
#  c2                 :  -0.4850 [-] ± 0.0279  (5.75 %)
# with [0.0, 0.0, 40.0]
#  c1                 :   0.3159 1/m ± 0.0003  (0.09 %)
#  c2                 :  -0.3837 [-] ± 0.0573  (14.93 %)
#
# DEPOWER also changes the steering response, and by as much as the damping does
# (2026-07-26, body_damping = [0.0, 0.0, 40.0], sweep extended to u_s = 0.300):
#  depower 0.25, 150 m: c1 = 0.3159 1/m, c2 = -0.3837, delay 0.03 s
#  depower 0.40, 200 m: c1 = 0.1513 1/m (0.06 %), c2 = 0.1951, delay 0.42 s
#    extended to u_s = 0.50 at the same setting: c1 = 0.1495 (0.15 %), so the
#    law stays LINEAR over the whole usable range — but the run DIVERGED the
#    moment the amplitude reached u_s = 0.375, in plain bang-bang oscillation.
#    The plant therefore caps out around u_s ~ 0.35 here, and G scatter rises
#    9.8 % -> 19.9 % (residual RMS 1.10 -> 4.53 deg/s) approaching it.
#  depower 0.55, 150 m: c1 = 0.1071 1/m (0.07 %), c2 = 0.3444, delay 0.55 s
# i.e. depowering costs steering authority AND adds dead time (0.03 -> 0.55 s).
# Both matter for figure-eight path following: see PlanFig8.md and
# examples/simple_fig8.jl.
s = init(V_WIND, TETHER_LENGTH; body_damping = [10.0, 10.0, 40.0],
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
    system_yaml = PROJECT)

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]

toc("Start simulation loop...")

# ==================== SIMULATION LOOP ==================== #

steering = START_STEERING       # current sweep amplitude
rel_steering = 0.0              # commanded steering
heading = 0.0                   # band-controller heading state [rad]
cycles = 0                      # upward crossings at the current amplitude
min_elevation = Inf             # lowest elevation seen [deg]
steps_done = 0
outcome = :time_limit           # :sweep_done | :time_limit | :low_elevation | :error

t_loop = @elapsed try
    for _ in 1:s.steps
        t = s.sys_state.time + s.dt

        # Kick the oscillation off with a single step to -u_s, exactly as the
        # KPS4 original does at its t = 10 s.
        if T_START <= t < T_START + s.dt
            global rel_steering = -steering
        end

        # Relay: flip the steering whenever the heading leaves the band, and
        # step the amplitude up on every upward crossing of the top edge. The
        # `last_heading` guard makes that one increment per crossing, not one
        # per timestep.
        global heading
        last_heading = heading
        if t > T_START + s.dt
            heading = wrap_to_pi(s.sys_state.heading)
            if rad2deg(heading) < -HEADING_OFFSET
                global rel_steering = steering
            elseif rad2deg(heading) > HEADING_OFFSET
                global rel_steering = -steering
                if rad2deg(last_heading) <= HEADING_OFFSET
                    global cycles += 1
                    if cycles >= CYCLES_PER_LEVEL
                        if steering >= MAX_STEERING - 1e-9
                            global outcome = :sweep_done
                            break
                        end
                        global cycles = 0
                        global steering = min(steering + STEERING_STEP, MAX_STEERING)
                        @info @sprintf("t = %6.2f s: steering amplitude -> %.3f",
                                       t, steering)
                    end
                end
            end
        end

        # Position mode: `set_length` holds the mean tether length.
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering, set_length = l0)
        global steps_done += 1

        # Elevation floor: the parked kite loses elevation as it turns, and a
        # sweep that dives is not a usable identification data set.
        el = rad2deg(s.sys_state.elevation)
        global min_elevation = min(min_elevation, el)
        if el < MIN_ELEVATION
            @warn @sprintf("elevation %.2f° below MIN_ELEVATION = %.1f° at t = %.2f s, stopping",
                           el, MIN_ELEVATION, t)
            global outcome = :low_elevation
            break
        end
    end
catch e
    global outcome = :error
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
end

@info "Save the log"
save_log(s.logger, "tmp_steering"; colmeta=timestamp_colmeta())

# ================== IDENTIFICATION ======================= #

sl = KiteUtils.syslog(s.logger)
r = identify_turn_rate_law(sl; dt = DT, t_start = T_START,
                           min_steering = MIN_STEERING_FIT)

print("\n", format_turn_rate_report(r; max_rel_std = MAX_REL_STD))

completed = outcome in (:sweep_done, :time_limit)
@printf("  %s completion   : %s (%d steps, %.1f s simulated in %.1f s wall)\n",
        completed ? "PASS" : "FAIL", outcome, steps_done,
        s.sys_state.time, t_loop)
@printf("  %s elevation    : min %.2f° %s MIN_ELEVATION = %.1f°\n",
        min_elevation >= MIN_ELEVATION ? "PASS" : "FAIL", min_elevation,
        min_elevation >= MIN_ELEVATION ? ">=" : "<", MIN_ELEVATION)
@printf("  max amplitude reached: %.3f of MAX_STEERING = %.3f\n",
        steering, MAX_STEERING)

nothing
