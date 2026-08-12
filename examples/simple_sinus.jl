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

Logs the run to "tmp_sinus" in the `output` folder (`examples/../output`,
created if missing; kept separate from simple_parking.jl's "tmp_parking").
For verification, run `include("examples/simple_sinus.jl")`, then
`include("examples/simple_sinus_plots.jl")`, and check the printed heading
tracking RMS error.

At the end it also prints the AoA ripple metrics (see `src/ripple_metrics.jl` and
PlanSuppressOscillations.md) together with the solver cost and the wall clock, as
`examples/simple_parking.jl` does, so a change to e.g. the body-frame damping can
be judged on both the oscillation and the simulation speed. Unlike the parking
examples the AoA here also swings with the maneuver itself; the metrics are taken
over the same settled window as the tracking error (t ≥ HEADING_PERIOD) and on
the detrended signal, so the slow 1/HEADING_PERIOD component is removed and what
is left is the fast ripple. The numbers are only comparable across runs that fix
PROJECT, V_WIND, TETHER_LENGTH, DEPOWER_SETPOINT, the heading gains and setpoint,
SIM_TIME and DT.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers; tic()
using V3Kite
using V3Kite: init, step!
using DiscretePIDs: set_Td!
using Statistics
using Printf

@info "simple_sinus.jl: sinusoidal heading tracking of the V3 kite via the init/step! interface."

# ==================== USER PARAMETERS ==================== #

PROJECT =        "system_reelout.yaml"  # System project to use (see data/system_*.yaml)
SIM_TIME         = 120.0     # Total simulation time [s]
DT               = 0.05/3     # Simulation timestep [s]
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 150.0    # Initial tether length [m]
DEPOWER_SETPOINT = 0.25     # Depower setting held during the run [-]
MAX_HEADING      = 40.0     # Heading setpoint amplitude [deg]
HEADING_PERIOD   = 30.0     # Heading setpoint period [s]

# Heading PID gains (output is rel_steering, dimensionless, -1..1)
# The tracking error is almost pure phase lag, so it scales with 1/loop gain,
# but only below the stability boundary. Sweep at the BODY_DAMPING set below,
# [0, 0, 40] (RMS over t ≥ HEADING_PERIOD, u_s peak-to-peak in
# the same window): K = 1.2 keeps a
# ~1.6x margin to the boundary at K ≈ 1.9; K = 1.6 tracks better (0.52°) but
# sits right at the edge.
HEADING_P = 1.2
HEADING_I = false
# Sustained derivative action damps the fast mode — it keeps even K = 2.0 stable
# (RMS 1.18°) — but it floors the settled RMS at ~1.2° regardless of K (K = 1.6
# with a constant Td: 1.29° instead of 0.72° at K = 1.2). It is what damps the
# initial u_s swing, so it is ramped to zero over the first heading period:
# |u_s| during t < HEADING_PERIOD peaks at 0.050 rather than saturating. Since
# Td has reached zero by the time the error is measured (t ≥ HEADING_PERIOD),
# the settled RMS is unaffected.
HEADING_D = 0.15
MAX_STEERING = 0.175        # settled |u_s| peaks at 0.028, so this is not binding
AERO_MODE = AeroDirect()    # ContinuousAero() or AeroDirect()
VSM_INTERVAL = 1   # steps between VSM aero solves
# `BODY_DAMPING` only shapes the settling transient, decaying to the `min_damping`
# floor of `init` (0.8 x this by default), which is the damping the run actually
# FLIES with — and which the heading gains above were tuned at. Both are part of
# the settling cache key, so changing this re-settles instead of reusing the
# cached geometry.
BODY_DAMPING = [0.0, 0.0, 40.0]   # Damping settling starts from, per axis [1/s]
# Structural damping of the tether and bridle lines, given as the ratio of the
# damping to the stiffness of a segment: unit_damping = ratio * unit_stiffness [s].
# It overrides the `damping_per_stiffness` of the `dyneema` material in
# `data/struc_geometry.yaml`; the wing frame keeps the damping hardcoded there.
# `init` applies it from the START of settling, floored at 0.0015 for the
# settling run only (below that settling diverges) and set unfloored on the
# settled structure. 0.002 is the material value the bridles already carry; the
# tether carries none by default. See simple_parking.jl for the details.
DAMPING_PER_STIFFNESS = 0.001  # Damping per stiffness of tether and bridles [s]

# ======================== INIT =========================== #

# `init` leaves the data path alone, so `save_log`/`load_log` below need it set here.
set_data_path(v3_data_path())
s = init(V_WIND, TETHER_LENGTH; body_damping = BODY_DAMPING,
    damping_per_stiffness = DAMPING_PER_STIFFNESS,
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
    system_yaml = PROJECT, aero_mode = AERO_MODE)

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]

heading_pid = create_heading_pid(;
    K = HEADING_P, Ti = HEADING_I, Td = HEADING_D, dt = s.dt,
    umin = -MAX_STEERING, umax = MAX_STEERING)

max_heading_rad = deg2rad(MAX_HEADING)
angular_freq = 2pi / HEADING_PERIOD
toc("Start simulation loop...")

# ==================== SIMULATION LOOP ==================== #

steps_done = 0
t_loop = @elapsed try
    for _ in 1:s.steps
        t = s.sys_state.time + s.dt
        target = max_heading_rad * sin(angular_freq * t)
        s.sys_state.bearing = target
        # Ramp HEADING_D down to zero at t = HEADING_PERIOD.
        set_Td!(heading_pid, HEADING_D * (1 - ramp_factor(t, 0.0, HEADING_PERIOD)))
        rel_steering = heading_pid(target, s.sys_state.heading, 0.0)
        # Position mode: `set_length` holds the mean tether length.
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering, set_length = l0,
              vsm_interval = VSM_INTERVAL)
        global steps_done += 1
        # The current system state is available via `s.sys_state`.
    end
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
end

@info "Save the log"
OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
mkpath(OUTPUT_DIR)
save_log(s.logger, "tmp_sinus"; path=OUTPUT_DIR, colmeta=timestamp_colmeta())

# Tracking error over the settled part (skip the first period).
syslog = load_log("tmp_sinus"; path=OUTPUT_DIR)
sl = syslog.syslog
settled = findall(t -> t >= HEADING_PERIOD, sl.time)
if !isempty(settled)
    track_err = rad2deg.(sl.heading[settled] .- sl.bearing[settled])
    @printf("Heading tracking RMS error (t ≥ %.0f s): %.2f°\n",
            sl.time[settled[1]], sqrt(mean(track_err .^ 2)))
end

# ==================== RIPPLE METRICS ===================== #

# `sl` is the syslog table of the run just saved, so this measures the same run
# `aoa_ripple` would see via `KiteUtils.syslog(s.logger)` in simple_parking.jl.
# Measure over the settled window only: before t = HEADING_PERIOD the Td ramp is
# still active and the AoA still carries the settling transient.
rs = RippleSettings("ripple_settings.yaml")
rs.t_start = HEADING_PERIOD
ripple = aoa_ripple(sl; rs)
print("\n", format_ripple_report(ripple; sl, stats = s.sam.integrator.stats,
                                 t_loop, n_steps = steps_done))

nothing
