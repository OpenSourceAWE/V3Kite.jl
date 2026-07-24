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

using V3Kite
using DiscretePIDs: set_Td!
using Statistics
using Printf

@info "simple_sinus.jl: sinusoidal heading tracking of the V3 kite via the init/step! interface."

# ==================== USER PARAMETERS ==================== #

PROJECT =        "system_cabauw.yaml"  # System project to use (see data/system_*.yaml)
SIM_TIME         = 120.0     # Total simulation time [s]
V_WIND           = 10.0     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 150.0    # Initial tether length [m]
DEPOWER_SETPOINT = 0.25     # Depower setting held during the run [-]
MAX_HEADING      = 40.0     # Heading setpoint amplitude [deg]
HEADING_PERIOD   = 30.0     # Heading setpoint period [s]

# Heading PID gains (output is rel_steering, dimensionless, -1..1)
HEADING_P = 1.1             # +10% from the v3kite.jl baseline (was 1.0; RMS 1.1° there)
HEADING_I = false
HEADING_D = 0.15            # damps the initial u_s overshoot/ringing (was 0.0)
HEADING_D_RAMP_TIME = 30.0  # ramp HEADING_D down to 0 over this many seconds
MAX_STEERING = 0.175        # +10% (was 0.15)

# ======================== INIT =========================== #

s = init(V_WIND, TETHER_LENGTH;
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, system_yaml = PROJECT)

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]

heading_pid = create_heading_pid(;
    K = HEADING_P, Ti = HEADING_I, Td = HEADING_D, dt = s.dt,
    umin = -MAX_STEERING, umax = MAX_STEERING)

max_heading_rad = deg2rad(MAX_HEADING)
angular_freq = 2pi / HEADING_PERIOD

# ==================== SIMULATION LOOP ==================== #

heading_setpoint = Float64[]      # sinusoidal setpoint [rad]
sizehint!(heading_setpoint, s.steps)

try
    for _ in 1:s.steps
        t = s.sys_state.time + s.dt
        target = max_heading_rad * sin(angular_freq * t)
        push!(heading_setpoint, target)
        set_Td!(heading_pid, HEADING_D * (1 - ramp_factor(t, 0.0, HEADING_D_RAMP_TIME)))
        rel_steering = heading_pid(target, s.sys_state.heading, 0.0)
        # Position mode: `set_length` holds the mean tether length.
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering, set_length = l0)
        # The current system state is available via `s.sys_state`.
    end
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
end

@info "Save the log"
save_log(s.logger, "tmp_sinus")

# Tracking error over the settled part (skip the first period).
syslog = load_log("tmp_sinus")
sl = syslog.syslog
n_logged = min(length(heading_setpoint), length(sl.time) - 1)
time_vec = sl.time[2:n_logged+1]
settled = findall(t -> t >= HEADING_PERIOD, time_vec)
if !isempty(settled)
    track_err = rad2deg.(sl.heading[settled .+ 1] .- heading_setpoint[settled])
    @printf("Heading tracking RMS error (t ≥ %.0f s): %.2f°\n",
            time_vec[settled[1]], sqrt(mean(track_err .^ 2)))
end

nothing
