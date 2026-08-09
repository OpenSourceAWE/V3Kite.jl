# Copyright (c) 2025 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Example script demonstrating a parking maneuver of the V3 kite using the
high-level `init` + `step!` interface.

The wing is settled at a fixed depower setting (DEPOWER_SETPOINT) and then
parked at a constant tether length: `step!` runs in POSITION MODE, with
`set_length` holding the initial tether length via the cascaded winch
controller, so the kite parks without any net reel-out.

The manual, braked-winch reference `examples/parking.jl` is not modified; this
script is its `init`/`step!` counterpart. Logs the run to "tmp_parking".

At the end it prints the AoA ripple metrics (see `src/ripple_metrics.jl` and
PlanSuppressOscillations.md) together with the solver cost and the wall clock, so
a change to e.g. the body-frame damping can be judged on both the oscillation and
the simulation speed. The numbers are only comparable across runs that fix
PROJECT, V_WIND, TETHER_LENGTH, DEPOWER_SETPOINT, REL_STEERING, SIM_TIME and DT.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using V3Kite: init, step!
import KiteUtils   # for KiteUtils.syslog; V3Kite does not re-export it, and the
                   # plots scripts bind the bare name `syslog` to a SysLog value
using LinearAlgebra: norm

@info "simple_parking.jl: parking the V3 kite via the init/step! interface."

# ==================== USER PARAMETERS ==================== #

PROJECT =        "system_reelout.yaml"  # System project to use (see data/system_*.yaml)
SIM_TIME         = 10.0     # Total simulation time [s]
DT               = 0.05/3     # Simulation timestep [s]
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 150.0    # Initial tether length [m]
DEPOWER_SETPOINT = 0.25     # Depower setting held during parking [-]
REL_STEERING     = 0.0040   # Fixed steering trim, tuned so |heading(end)| < 10 degrees
AERO_MODE        = ContinuousAero()
VSM_INTERVAL     = 1   # steps between VSM aero solves

# ======================== INIT =========================== #

# `init` leaves the data path alone, so `save_log` below needs it set here.
set_data_path(v3_data_path())
s = init(V_WIND, TETHER_LENGTH; body_damping = [0.0, 0.0, 40.0],
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
    system_yaml = PROJECT, aero_mode = AERO_MODE)

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]

# ==================== SIMULATION LOOP ==================== #

steps_done = 0
t_loop = @elapsed try
    for _ in 1:s.steps
        # Position mode: hold the mean tether length at its initial value.
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering = REL_STEERING,
              set_length = l0, vsm_interval = VSM_INTERVAL)
        global steps_done += 1
        # The current system state is available via `s.sys_state`.
    end
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
end

@info "Save the log"
save_log(s.logger, "tmp_parking"; colmeta=timestamp_colmeta())

@info "Wind speed at kite height: $(round(norm(v_wind_kite(s)), digits=2)) m/s"

# ==================== RIPPLE METRICS ===================== #

sl = KiteUtils.syslog(s.logger)
ripple = aoa_ripple(sl)
print("\n", format_ripple_report(ripple; sl, stats = s.sam.integrator.stats,
                                 t_loop, n_steps = steps_done))

nothing
