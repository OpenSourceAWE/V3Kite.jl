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
script is its `init`/`step!` counterpart. Logs the run to "tmp_run".
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using LinearAlgebra: norm

@info "simple_parking.jl: parking the V3 kite via the init/step! interface."

# ==================== USER PARAMETERS ==================== #

PROJECT =        "system_reelout.yaml"  # System project to use (see data/system_*.yaml)
SIM_TIME         = 10.0     # Total simulation time [s]
DT               = 0.05/3   # Simulation timestep [s]
V_WIND           = 9.51     # Ground wind speed at reference height [m/s]
TETHER_LENGTH    = 150.0    # Initial tether length [m]
DEPOWER_SETPOINT = 0.25     # Depower setting held during parking [-]

# ======================== INIT =========================== #

s = init(V_WIND, TETHER_LENGTH;
    depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT, system_yaml = PROJECT)

# Constant-length setpoint: the tether length just after settling.
l0 = s.sys_state.l_tether[1]

# ==================== SIMULATION LOOP ==================== #

try
    for _ in 1:s.steps
        # Position mode: hold the mean tether length at its initial value.
        step!(s; rel_depower = DEPOWER_SETPOINT, set_length = l0)
        # The current system state is available via `s.sys_state`.
    end
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
end

@info "Save the log"
save_log(s.logger, "tmp_run")

@info "Wind speed at kite height: $(round(norm(v_wind_kite(s)), digits=2)) m/s"

nothing
