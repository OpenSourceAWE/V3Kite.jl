# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite Simulation Example

Heading PID control with sinusoidal setpoint, winch PID
control for constant tether length, and 3D visualization.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using VortexStepMethod
using GLMakie
using LinearAlgebra
using DiscretePIDs

# =============================================================================
# Configuration
# =============================================================================

SIM_TIME = 60.0
FPS = 60
MAX_HEADING = 40.0    # degrees
PERIOD = 30.0         # seconds
V_WIND = 15.4
TETHER_LENGTH = 250.0
DEPOWER = 0.30        # fraction [0, 1]
WORLD_DAMPING = 40.0  # initial world damping
DAMPING_DECAY_STEPS = 200

# PID gains
HEADING_P = 5.0
HEADING_I = false
HEADING_D = 0.0
WINCH_P = 1000.0
WINCH_I = 100.0
WINCH_D = 50.0

# =============================================================================
# Setup
# =============================================================================

@info "V3 Kite Simulation Example"
@info "Calibration:" steering_l0=V3_STEERING_L0_BASE depower_l0=V3_DEPOWER_L0_BASE

gc = V3GeomAdjustConfig()

data_path = v3_data_path()
set_data_path(data_path)
set = Settings("system.yaml")
set.g_earth = 9.81
set.wind_vec = [V_WIND, 0.0, 0.0]
set.l_tether = TETHER_LENGTH
set.profile_law = 0

source_struc = joinpath(data_path, "struc_geometry.yaml")
source_aero = joinpath(data_path, "aero_geometry.yaml")
vsm_path = joinpath(data_path, "vsm_settings.yaml")
vsm_set = VortexStepMethod.VSMSettings(vsm_path; data_prefix=false)
vsm_set.wings[1].geometry_file = source_aero

@info "Creating V3 model..."
sys = load_sys_struct_from_yaml(source_struc;
    system_name=V3_MODEL_NAME, set,
    wing_type=REFINE, vsm_set)
sam = SymbolicAWEModel(set, sys)

@info "Initializing model..."
init!(sam; remake=false, remake_vsm=true)
sys.winches[1].brake = false
set_depower!(sys, DEPOWER, 0.0, gc)

n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps
logger, sys_state = create_logger(sam, n_steps)

# Heading PID (outputs steering tape delta in m)
nominal_steering = get_steering(sys, gc)
max_heading_rad = deg2rad(MAX_HEADING)
angular_freq = 2pi / PERIOD
max_steering = 0.15

heading_pid = create_heading_pid(;
    K = HEADING_P,
    Ti = HEADING_I,
    Td = HEADING_D,
    dt, umin=-abs(max_steering),
    umax=abs(max_steering))

# Winch PID
nominal_tether_length = sys.tethers[1].len
init_winch_torque!(sys)
winch_pid = create_winch_pid(;
    K = WINCH_P,
    Ti = WINCH_I > 0 ? WINCH_P / WINCH_I : false,
    Td = WINCH_D > 0 ? WINCH_D / WINCH_P : false,
    dt)

heading_setpoint = [0.0]

# =============================================================================
# Simulation loop
# =============================================================================

@info "Starting simulation" n_steps dt
sim_start = time()

for step in 1:n_steps
    t = step * dt

    damping = max(WORLD_DAMPING *
        (1.0 - step / DAMPING_DECAY_STEPS), 0.0)
    SymbolicAWEModels.set_world_frame_damping(sys, damping)

    # PID heading control with sine wave setpoint
    target_rad = max_heading_rad * sin(angular_freq * t)
    current = sam.sys_struct.wings[1].heading
    steer_ctrl = heading_pid(target_rad, current, 0.0)
    push!(heading_setpoint, target_rad)

    set_steering!(sys, nominal_steering + steer_ctrl, gc)

    # Winch PID
    tl = sys.tethers[1].len
    wf = winch_pid(nominal_tether_length, tl, 0.0)
    wt = force_to_torque(wf, sys)
    sys.winches[1].set_value = -wt

    if !sim_step!(sam;
            set_values=[-wt], dt, vsm_interval=1)
        @error "Simulation failed" step
        break
    end
    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps)
        elapsed = time() - sim_start
        @info "Step $step/$n_steps" times_realtime=round(t/elapsed, digits=2) damping=round(damping, digits=2)
    end
end

report_performance(SIM_TIME, time() - sim_start)

save_log(logger, "v3kite_example")
syslog = load_log("v3kite_example")

# =============================================================================
# Visualization
# =============================================================================

@info "Creating visualization..."
fig = plot(sam.sys_struct, syslog;
    plot_tether=true,
    setpoints=Dict(:heading => heading_setpoint))
display(fig)

scene = replay(syslog, sam.sys_struct)
display(scene)

@info "Example complete!"
nothing
