# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Zenith Hold Followed by Circular Flight

Two-phase simulation:
1. Zenith hold with azimuth PID and winch PID
2. Circular flight with ramped steering (brake engaged)

Uses CORRECT geometry files.

Usage:
    julia --project=examples examples/zenith_circles.jl
"""

using V3Kite
using GLMakie
using LinearAlgebra
using DiscretePIDs
using Dates

# =============================================================================
# Configuration
# =============================================================================

# General
V_WIND = 8.6
V_WIND_BASE = 8.6
UP = 0.18                  # Depower fraction (0-1), old convention
TETHER_LENGTH = 268.0
ELEVATION = 65.0
G_EARTH = 9.81

# Zenith phase
SIM_TIME_ZENITH = 300.0
FPS_ZENITH = 60
MAX_US_ZENITH = 0.02       # Max steering for azimuth PID
TARGET_AZIMUTH = 0.0       # rad
RAMP_START = 0.1
RAMP_TIME_UP = 2.0

# Circular phase
SIM_TIME_CIRCLES = 0.0     # Set > 0 to enable circular phase
FPS_CIRCLES = 60
US_CIRCLES = 0.2           # Steering for circular phase
RAMP_TIME_US = 5.0

# Damping
INITIAL_DAMPING = 100.0
DAMPING_PATTERN = [0.0, 0.0, 1.0]
DECAY_TIME = 2.0
MIN_DAMPING = 1.0

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
    struc_yaml_path = "CORRECT_struc_geometry.yaml",
    aero_yaml_path = "CORRECT_aero_geometry.yaml",
    vsm_settings_path = "CORRECT_vsm_settings.yaml",
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    wing_type = REFINE,
    elevation = ELEVATION,
)

@info "Creating V3 model..."
sam, sys = create_v3_model(config)

# Set gravity
sam.set.g_earth = G_EARTH

# Damping
SymbolicAWEModels.set_body_frame_damping(
    sys, DAMPING_PATTERN * INITIAL_DAMPING)

# Initialize
@info "Initializing model..."
init!(sam; remake=config.remake_cache,
    ignore_l0=false, remake_vsm=true)

# Compute power tape target (old 0-1 convention)
nominal_l0_left = sys.segments[V3_STEERING_LEFT_IDX].l0
nominal_l0_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
nominal_l0_88 = sys.segments[V3_DEPOWER_IDX].l0
power_tape_change = ((200 + 5000 * UP) / 1000) - nominal_l0_88

# Logger for both phases
n_steps_zenith = max(1, Int(round(FPS_ZENITH * SIM_TIME_ZENITH)))
dt_zenith = SIM_TIME_ZENITH / n_steps_zenith
n_steps_circles = max(1, Int(round(FPS_CIRCLES * SIM_TIME_CIRCLES)))
dt_circles = SIM_TIME_CIRCLES > 0 ?
    SIM_TIME_CIRCLES / n_steps_circles : dt_zenith
total_steps = n_steps_zenith + n_steps_circles
logger, sys_state = create_logger(sam, total_steps)

# Azimuth PID
max_steering = MAX_US_ZENITH * V3_STEERING_GAIN
azimuth_pid = create_heading_pid(;
    K = HEADING_P > 0 ? HEADING_P : 1.0,
    Ti = HEADING_I > 0 ? 1.0 / HEADING_I : false,
    Td = HEADING_D > 0 ? HEADING_D : false,
    dt=dt_zenith, umin=-abs(max_steering),
    umax=abs(max_steering))

# Winch PID
nominal_tether_length = sys.winches[1].tether_len
init_winch_torque!(sys)
winch_pid = create_winch_pid(;
    K = WINCH_P,
    Ti = WINCH_I > 0 ? WINCH_P / WINCH_I : false,
    Td = WINCH_D > 0 ? WINCH_D / WINCH_P : false,
    dt=dt_zenith)

azimuth_setpoint = Float64[TARGET_AZIMUTH]

# =============================================================================
# Phase 1: Zenith hold
# =============================================================================

@info "Phase 1: Zenith hold" n_steps_zenith dt_zenith
sim_start_time = time()

for step in 1:n_steps_zenith
    t = step * dt_zenith

    # Damping decay
    damping = max(INITIAL_DAMPING * (1.0 - t / DECAY_TIME),
        MIN_DAMPING)
    SymbolicAWEModels.set_body_frame_damping(
        sys, DAMPING_PATTERN * damping)

    # Azimuth PID
    azimuth = sys.wings[1].azimuth
    steering_ctrl = azimuth_pid(TARGET_AZIMUTH, azimuth, 0.0)
    push!(azimuth_setpoint, TARGET_AZIMUTH)

    # Power ramp
    rf = ramp_factor(t, RAMP_START, RAMP_START + RAMP_TIME_UP)
    sys.segments[V3_DEPOWER_IDX].l0 =
        nominal_l0_88 + power_tape_change * rf

    # Steering
    sys.segments[V3_STEERING_LEFT_IDX].l0 =
        nominal_l0_left + steering_ctrl
    sys.segments[V3_STEERING_RIGHT_IDX].l0 =
        nominal_l0_right - steering_ctrl

    # Winch PID
    tether_len = sys.winches[1].tether_len
    wf = winch_pid(nominal_tether_length, tether_len, 0.0)
    wt = force_to_torque(wf, sys)
    sys.winches[1].set_value = -wt

    if !sim_step!(sam;
            set_values=[-wt], dt=dt_zenith, vsm_interval=1)
        @error "Zenith phase failed" step
        break
    end

    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps_zenith)
        elapsed = time() - sim_start_time
        @info "Zenith $step/$n_steps_zenith (t=$(round(t, digits=2))s)" times_realtime=round(t / elapsed, digits=2)
    end
end

# =============================================================================
# Phase 2: Circular flight (only if SIM_TIME_CIRCLES > 0)
# =============================================================================

if SIM_TIME_CIRCLES > 0
    @info "Phase 2: Circular flight" n_steps_circles dt_circles

    # Reset damping for circular phase
    SymbolicAWEModels.set_body_frame_damping(
        sys, DAMPING_PATTERN * INITIAL_DAMPING)

    # Lock tether
    sys.winches[1].brake = true
    sys.winches[1].set_value = 0.0

    # Circular phase targets
    steering_tape = V3_STEERING_GAIN * US_CIRCLES
    power_target = nominal_l0_88 + power_tape_change
    steer_target_left = nominal_l0_left + steering_tape
    steer_target_right = nominal_l0_right - steering_tape
    steer_start_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    steer_start_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
    power_start = sys.segments[V3_DEPOWER_IDX].l0

    for step in 1:n_steps_circles
        t_stage = step * dt_circles
        t_total = SIM_TIME_ZENITH + t_stage

        # Damping decay
        damping = max(INITIAL_DAMPING *
            (1.0 - t_stage / DECAY_TIME), MIN_DAMPING)
        SymbolicAWEModels.set_body_frame_damping(
            sys, DAMPING_PATTERN * damping)

        rf = ramp_factor(t_stage, 0.0, RAMP_TIME_US)

        sys.segments[V3_DEPOWER_IDX].l0 =
            power_start + (power_target - power_start) * rf
        sys.segments[V3_STEERING_LEFT_IDX].l0 =
            steer_start_left +
            (steer_target_left - steer_start_left) * rf
        sys.segments[V3_STEERING_RIGHT_IDX].l0 =
            steer_start_right +
            (steer_target_right - steer_start_right) * rf

        if !sim_step!(sam;
                set_values=[0.0], dt=dt_circles,
                vsm_interval=1)
            @error "Circular phase failed" step
            break
        end

        log_state!(logger, sys_state, sam, t_total)

        if should_report(step, n_steps_circles)
            elapsed = time() - sim_start_time
            @info "Circle $step/$n_steps_circles (t=$(round(t_total, digits=2))s)" times_realtime=round(t_total / elapsed, digits=2)
        end
    end
end

total_sim = SIM_TIME_ZENITH + SIM_TIME_CIRCLES
report_performance(total_sim, time() - sim_start_time)

# =============================================================================
# Save and plot
# =============================================================================

lt_tag = Int(round(TETHER_LENGTH))
log_name = "zenith_circles_lt_$(lt_tag)"
save_log(logger, log_name)
syslog = load_log(log_name)

fig = plot(sam.sys_struct, syslog;
    plot_turn_rates=true, plot_yaw_rate=true,
    plot_yaw_rate_paper=true, plot_reelout=false,
    plot_gk=true, gk_ylims=(0.0, 10.0),
    plot_aoa=true, plot_heading=false,
    plot_elevation=true, plot_azimuth=true,
    plot_winch_force=false, plot_set_values=false,
    plot_us=true)

scene = replay(syslog, sam.sys_struct)

display(fig)
display(scene)

nothing
