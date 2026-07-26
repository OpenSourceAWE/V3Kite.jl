# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Example script demonstrating sinusoidal heading tracking of the V3 kite using
the high-level `init` + `step!` interface.

Does the same as the heading-PID loop in `examples/v3kite.jl` (track a
sinusoidal heading setpoint on a settled, depowered wing), but built on
`init()`/`step!()` in the style of `examples/simple_parking.jl`: the wing is
settled at a fixed depower setting (DEPOWER_SETPOINT) and parked at a
constant tether length (`step!` runs in POSITION MODE via `set_length`),
while a `create_heading_pid` controller drives `rel_steering` each step.

Logs the run to "tmp_sinus" (kept separate from simple_parking.jl's
"tmp_run"). For verification, run `include("examples/simple_sinus.jl")`, then
`include("examples/simple_sinus_plots.jl")`, and check the printed heading
tracking RMS error.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers; tic()
using V3Kite
using DiscretePIDs: set_Td!
using Statistics
using Printf

@info "simple_sinus.jl: sinusoidal heading tracking of the V3 kite via the init/step! interface."

# ==================== USER PARAMETERS ==================== #

PROJECT =        "system_reelout.yaml"  # System project to use (see data/system_*.yaml)
SIM_TIME         = 120.0     # Total simulation time [s]
DT               = 0.05/3   # Simulation timestep [s]
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 150.0    # Initial tether length [m]
DEPOWER_SETPOINT = 0.26     # Depower setting held during the run [-]
MAX_HEADING      = 40.0     # Heading setpoint amplitude [deg]
HEADING_PERIOD   = 30.0     # Heading setpoint period [s]

# Heading PID gains (output is rel_steering, dimensionless, -1..1)
# The tracking error is almost pure phase lag, so it scales with 1/loop gain:
# K = 1.1 gave 0.32 s lag (RMS 2.6°), K = 5.0 gives 0.05 s (RMS 0.49°). The
# v3kite.jl baseline of K = 1.0 is enough there only because it flies at
# 15.4 m/s — at the 9.5 m/s used here the turn rate per unit steering, and
# hence the plant gain, is ~1.6x lower.
HEADING_P = 5.0
HEADING_I = false
# Sustained derivative action degrades tracking (K = 4.5 with a constant
# Td = 0.1: RMS 1.05° instead of 0.55°), but it is what damps the initial u_s
# swing, so it is ramped to zero over the first heading period: p2p u_s during
# t < 10 s drops 0.343 → 0.215 and the command no longer saturates. Since Td
# has reached zero by the time the error is measured (t ≥ HEADING_PERIOD), the
# settled RMS is unaffected — identical to 3 digits for Td = 0…0.5.
HEADING_D = 0.15
MAX_STEERING = 0.175        # settled |u_s| peaks at 0.078, so this is not binding

# ======================== INIT =========================== #

s = init(V_WIND, TETHER_LENGTH;
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT, system_yaml = PROJECT)

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]

heading_pid = create_heading_pid(;
    K = HEADING_P, Ti = HEADING_I, Td = HEADING_D, dt = s.dt,
    umin = -MAX_STEERING, umax = MAX_STEERING)

max_heading_rad = deg2rad(MAX_HEADING)
angular_freq = 2pi / HEADING_PERIOD
toc("Start simulation loop...")

# ==================== SIMULATION LOOP ==================== #

try
    for _ in 1:s.steps
        t = s.sys_state.time + s.dt
        target = max_heading_rad * sin(angular_freq * t)
        s.sys_state.bearing = target
        # Ramp HEADING_D down to zero at t = HEADING_PERIOD.
        set_Td!(heading_pid, HEADING_D * (1 - ramp_factor(t, 0.0, HEADING_PERIOD)))
        rel_steering = heading_pid(target, s.sys_state.heading, 0.0)
        # Position mode: `set_length` holds the mean tether length.
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering, set_length = l0)
        # The current system state is available via `s.sys_state`.
    end
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
end

@info "Save the log"
save_log(s.logger, "tmp_sinus"; colmeta=timestamp_colmeta())

# Tracking error over the settled part (skip the first period).
syslog = load_log("tmp_sinus")
sl = syslog.syslog
settled = findall(t -> t >= HEADING_PERIOD, sl.time)
if !isempty(settled)
    track_err = rad2deg.(sl.heading[settled] .- sl.bearing[settled])
    @printf("Heading tracking RMS error (t ≥ %.0f s): %.2f°\n",
            sl.time[settled[1]], sqrt(mean(track_err .^ 2)))
end

nothing
