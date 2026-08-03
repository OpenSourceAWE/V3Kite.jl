# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite Simulation Example

Power-zone settling, then heading PID with a sinusoidal setpoint,
winch PID for constant tether length, and 3D visualization.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using SymbolicAWEModels
using LinearAlgebra

# =============================================================================
# Configuration
# =============================================================================

SIM_TIME = 60.0
FPS = 20
AERO_MODE = AeroDirect()
MAX_HEADING = 40.0    # degrees
PERIOD = 30.0         # seconds
V_WIND = 15.4
TETHER_LENGTH = 250.0
ELEVATION = 70.0      # degrees
DEPOWER = 0.30        # fraction [0, 1]

# PID gains
HEADING_P = 1.0
HEADING_I = false
HEADING_D = 0.0

# =============================================================================
# Settling
# =============================================================================

@info "V3 Kite Simulation Example"
@info "Calibration:" steering_l0=V3_STEERING_L0_BASE depower_l0=V3_DEPOWER_L0_BASE

el_rad = deg2rad(ELEVATION)
position = [cos(el_rad) * TETHER_LENGTH, 0.0,
            sin(el_rad) * TETHER_LENGTH]
wind_vec = [V_WIND, 0.0, 0.0]

settle_config = V3SettleConfig(
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    dt = 0.05,
    num_steps = 40,
    num_substeps = 1,
    body_damping = [0.0, 0.0, 40.0],
    start_depower = DEPOWER * 100.0 + 10.0,
    course_correction_mode = :heading,
    course_correction_gain = 0.05,
    geom = V3GeomAdjustConfig(),
    aero_mode = AERO_MODE,
)
gc = settle_config.geom

@info "Settling V3 model..."
sam, settle_log, settle_failed = settle_wing(settle_config;
    position, velocity=[0.0, 0.0, 0.0], heading=0.0,
    steering=0.0, depower=DEPOWER, wind_vec, remake=false)
settle_failed && error("Settling failed")
sys = sam.sys_struct
sys.winches[1].brake = true

n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps
logger, sys_state = create_logger(sam, n_steps)

# Heading PID (outputs steering tape delta in m)
nominal_steering = V3Kite.get_steering(sys, gc)
max_heading_rad = deg2rad(MAX_HEADING)
angular_freq = 2pi / PERIOD
max_steering = 0.15

heading_pid = create_heading_pid(;
    K = HEADING_P,
    Ti = HEADING_I,
    Td = HEADING_D,
    dt, umin=-abs(max_steering),
    umax=abs(max_steering))

# =============================================================================
# Simulation loop
# =============================================================================

@info "Starting simulation" n_steps dt
sim_start = time()

for step in 1:n_steps
    t = step * dt

    target_rad = max_heading_rad * sin(angular_freq * t)
    current = sam.sys_struct.wings[1].heading
    steer_ctrl = heading_pid(target_rad, current, 0.0)
    sys_state.bearing = target_rad

    set_steering!(sys, nominal_steering + steer_ctrl, gc)

    if !sim_step!(sam; dt, vsm_interval=10)
        @error "Simulation failed" step
        break
    end
    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps)
        elapsed = time() - sim_start
        @info "Step $step/$n_steps" times_realtime=round(t/elapsed, digits=2)
    end
end

report_performance(SIM_TIME, time() - sim_start)

save_log(logger, "v3kite_example")
syslog = load_log("v3kite_example")

# =============================================================================
# Visualization
# =============================================================================

@info "Creating visualization..."
fig = Makie.plot(sam.sys_struct, syslog;
    plot_tether=true,
    setpoints=Dict(:heading => syslog.syslog.bearing))
display(GLMakie.Screen(), fig)

scene = SymbolicAWEModels.replay(syslog, sam.sys_struct)
display(GLMakie.Screen(), scene)

@info "Example complete!"
nothing
