# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Zenith Hold with Azimuth PID

Holds the kite at a target azimuth (zenith) using an azimuth PID
controller with a winch PID to maintain constant tether length.
Uses world frame damping that decays linearly over `decay_time`.

Usage:
    julia --project=examples examples/zenith_hold.jl
"""

using V3Kite
using SymbolicAWEModels
using GLMakie
using LinearAlgebra
using DiscretePIDs
using Dates

# =============================================================================
# Configuration
# =============================================================================

SIM_TIME = 2.0
FPS = 60
V_WIND = 15.0
TETHER_LENGTH = 250.0
UP = 0.3                   # Depower fraction (0-1) #2025 flight definition
MAX_US = 0.05              # Max steering fraction for azimuth PID
TARGET_AZIMUTH = 0.0       # Target azimuth [rad]

# Damping
INITIAL_DAMPING = 50.0     # Initial world frame damping [N*s/m]
DECAY_TIME = 5.0           # Time for damping to decay [s]

# Ramp
RAMP_START = 0.1           # Power tape ramp start [s]
RAMP_TIME = 2.0            # Power tape ramp duration [s]

# PID gains
HEADING_P = 0.0
HEADING_I = 0.1
HEADING_D = 0.0
WINCH_P = 1000.0
WINCH_I = 100.0
WINCH_D = 50.0

# =============================================================================
# Model setup
# =============================================================================

config = V3SimConfig(
    struc_yaml_path="struc_geometry_stable.yaml",
    aero_yaml_path="aero_geometry_stable.yaml",
    sim_time=SIM_TIME,
    fps=FPS,
    v_wind=V_WIND,
    tether_length=TETHER_LENGTH,
    wing_type=REFINE,
    brake=false,
)

@info "Creating V3 model..."
sam, sys = create_v3_model(config)

# Initialize damping
SymbolicAWEModels.set_world_frame_damping(sys, INITIAL_DAMPING, 1:38)

# Initialize model
@info "Initializing model..."
init!(sam; remake=config.remake_cache, ignore_l0=false, remake_vsm=true)

# Compute power tape target
nominal_l0_88 = sys.segments[V3_DEPOWER_IDX].l0
power_tape_change = 0.2 + 5 * u_dp - nominal_l0_88

# Logger
n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps
logger, sys_state = create_logger(sam, n_steps)

# Store nominal segment lengths
nominal_l0_left = sys.segments[V3_STEERING_LEFT_IDX].l0
nominal_l0_right = sys.segments[V3_STEERING_RIGHT_IDX].l0

# Azimuth PID
max_steering = MAX_US * V3_STEERING_GAIN
azimuth_pid = create_heading_pid(;
    K=HEADING_P > 0 ? HEADING_P : 1.0,
    Ti=HEADING_I > 0 ? 1.0 / HEADING_I : false,
    Td=HEADING_D > 0 ? HEADING_D : false,
    dt, umin=-abs(max_steering), umax=abs(max_steering))

# Winch PID
nominal_tether_length = sys.winches[1].tether_len
init_winch_torque!(sys)
winch_pid = create_winch_pid(;
    K=WINCH_P,
    Ti=WINCH_I > 0 ? WINCH_P / WINCH_I : false,
    Td=WINCH_D > 0 ? WINCH_D / WINCH_P : false,
    dt)

azimuth_setpoint = Float64[TARGET_AZIMUTH]

# =============================================================================
# Simulation loop
# =============================================================================

@info "Starting simulation" n_steps dt target_azimuth = TARGET_AZIMUTH
sim_start_time = time()

for step in 1:n_steps
    t = step * dt

    # Damping decay
    if t <= DECAY_TIME
        damping = INITIAL_DAMPING * (1.0 - t / DECAY_TIME)
        SymbolicAWEModels.set_world_frame_damping(sys, damping, 1:38)
    else
        SymbolicAWEModels.set_world_frame_damping(sys, 0.0, 1:38)
    end

    # Azimuth PID control
    current_azimuth = sys.wings[1].azimuth
    steering_control = azimuth_pid(
        TARGET_AZIMUTH, current_azimuth, 0.0)
    push!(azimuth_setpoint, TARGET_AZIMUTH)

    # Power tape ramp
    rf = ramp_factor(t, RAMP_START, RAMP_START + RAMP_TIME)
    sys.segments[V3_DEPOWER_IDX].l0 =
        nominal_l0_88 + power_tape_change * rf

    # Apply steering
    sys.segments[V3_STEERING_LEFT_IDX].l0 =
        nominal_l0_left + steering_control
    sys.segments[V3_STEERING_RIGHT_IDX].l0 =
        nominal_l0_right - steering_control

    # Winch PID
    tether_len = sys.winches[1].tether_len
    winch_force = winch_pid(
        nominal_tether_length, tether_len, 0.0)
    winch_torque = force_to_torque(winch_force, sys)
    sys.winches[1].set_value = -winch_torque

    # Step
    if !sim_step!(sam;
        set_values=[-winch_torque], dt, vsm_interval=1)
        @error "Simulation failed" step
        break
    end

    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps)
        elapsed = time() - sim_start_time
        @info "Step $step/$n_steps (t=$(round(t, digits=2))s)" times_realtime = round(t / elapsed, digits=2)
    end
end

report_performance(SIM_TIME, time() - sim_start_time)

# =============================================================================
# Save and plot
# =============================================================================

log_name = "zenith_hold"
save_log(logger, log_name)
syslog = load_log(log_name)

fig = plot(sam.sys_struct, syslog;
    plot_turn_rates=true, plot_reelout=false, plot_gk=true,
    plot_aoa=true, plot_heading=false, plot_elevation=true,
    plot_azimuth=true, plot_winch_force=false,
    plot_set_values=false)
display(fig)

scene = replay(syslog, sam.sys_struct)
display(scene)

nothing
