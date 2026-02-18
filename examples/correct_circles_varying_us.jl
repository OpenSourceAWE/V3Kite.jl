# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Multi-Phase Circular Flight with Varying Steering

Zenith hold followed by multiple circular flight phases with different
steering inputs. Each phase has its own duration, FPS, ramp time, and
steering command. Uses CORRECT geometry.

Usage:
    julia --project=examples examples/correct_circles_varying_us.jl
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
V_WIND = 15.0
V_WIND_BASE = 15.0
UP = 0.3                   # Depower fraction (0-1)

# Body damping (decays from init to min)
INITIAL_BODY_DAMPING = [0.0, 0.0, 0.0]
MIN_BODY_DAMPING = [0.0, 30.0, 60.0]

# Zenith phase
SIM_TIME_ZENITH = 50.0
FPS_ZENITH = 60
MAX_US_ZENITH = 0.02
TARGET_AZIMUTH = 0.0       # rad
RAMP_START = 0.1
RAMP_TIME_UP = 10.0

# World damping
INITIAL_DAMPING = 100.0
DECAY_TIME = 2.0

# Circular phases (vectors for multi-phase)
SIM_TIME_CIRCLES = [150.0, 150.0, 150.0, 150.0, 150.0]
FPS_CIRCLES = [60, 60, 60, 60, 60]
RAMP_TIME_US = [10.0, 10.0, 10.0, 10.0, 10.0]
US_CIRCLES = [0.1, 0.2, 0.25, 0.2, 0.1]

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
    wing_type = REFINE,
)

@info "Creating V3 model..."
sam, sys = create_v3_model(config)

# Damping
SymbolicAWEModels.set_world_frame_damping(
    sys, INITIAL_DAMPING, 1:38)
SymbolicAWEModels.set_body_frame_damping(
    sys, INITIAL_BODY_DAMPING, 1:38)

# Initialize
@info "Initializing model..."
init!(sam; remake=config.remake_cache,
    ignore_l0=false, remake_vsm=true)

# Compute power tape target (old 0-1 convention)
nominal_l0_left = sys.segments[V3_STEERING_LEFT_IDX].l0
nominal_l0_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
nominal_l0_88 = sys.segments[V3_DEPOWER_IDX].l0
power_tape_change = ((200 + 5000 * UP) / 1000) - nominal_l0_88

# Validate multi-phase vectors
n_phases = length(SIM_TIME_CIRCLES)
@assert all(length(v) == n_phases for v in
    (FPS_CIRCLES, RAMP_TIME_US, US_CIRCLES)) "Phase vectors must have same length"

# Logger
n_steps_zenith = max(1, Int(round(FPS_ZENITH * SIM_TIME_ZENITH)))
dt_zenith = SIM_TIME_ZENITH / n_steps_zenith
n_steps_per_phase = [max(1, Int(round(FPS_CIRCLES[i] *
    SIM_TIME_CIRCLES[i]))) for i in 1:n_phases]
total_steps = n_steps_zenith + sum(n_steps_per_phase)
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
winch = sys.winches[1]

# =============================================================================
# Phase 1: Zenith hold
# =============================================================================

@info "Phase 1: Zenith hold" n_steps_zenith dt_zenith
sim_start_time = time()

for step in 1:n_steps_zenith
    t = step * dt_zenith

    # Damping decay (world + body)
    if t <= DECAY_TIME
        frac = 1.0 - t / DECAY_TIME
        SymbolicAWEModels.set_world_frame_damping(
            sys, INITIAL_DAMPING * frac, 1:38)
        bd = (INITIAL_BODY_DAMPING .- MIN_BODY_DAMPING) .*
            frac .+ MIN_BODY_DAMPING
        SymbolicAWEModels.set_body_frame_damping(
            sys, bd, 1:38)
    else
        SymbolicAWEModels.set_world_frame_damping(
            sys, 0.0, 1:38)
        SymbolicAWEModels.set_body_frame_damping(
            sys, MIN_BODY_DAMPING, 1:38)
    end

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
# Phase 2+: Circular flight phases
# =============================================================================

@info "Switching to circular flight" n_phases
SymbolicAWEModels.set_world_frame_damping(
    sys, INITIAL_DAMPING, 1:38)
SymbolicAWEModels.set_body_frame_damping(
    sys, INITIAL_BODY_DAMPING, 1:38)
winch.brake = true
winch.set_value = 0.0

power_target = nominal_l0_88 + power_tape_change
power_start = sys.segments[V3_DEPOWER_IDX].l0
steer_start_left = sys.segments[V3_STEERING_LEFT_IDX].l0
steer_start_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
current_time = SIM_TIME_ZENITH

for phase in 1:n_phases
    n_steps_c = n_steps_per_phase[phase]
    dt_c = SIM_TIME_CIRCLES[phase] / n_steps_c
    us_phase = US_CIRCLES[phase]
    ramp_us = RAMP_TIME_US[phase]

    steer_tape = V3_STEERING_GAIN * us_phase
    target_left = nominal_l0_left + steer_tape
    target_right = nominal_l0_right - steer_tape

    @info "Circular phase $phase/$n_phases" steps=n_steps_c us=us_phase

    for step in 1:n_steps_c
        t_stage = step * dt_c
        t_total = current_time + t_stage

        # Damping decay (reset each phase)
        if t_stage <= DECAY_TIME
            frac = 1.0 - t_stage / DECAY_TIME
            SymbolicAWEModels.set_world_frame_damping(
                sys, INITIAL_DAMPING * frac, 1:38)
            bd = (INITIAL_BODY_DAMPING .- MIN_BODY_DAMPING) .*
                frac .+ MIN_BODY_DAMPING
            SymbolicAWEModels.set_body_frame_damping(
                sys, bd, 1:38)
        else
            SymbolicAWEModels.set_world_frame_damping(
                sys, 0.0, 1:38)
            SymbolicAWEModels.set_body_frame_damping(
                sys, MIN_BODY_DAMPING, 1:38)
        end

        rf = ramp_factor(t_stage, 0.0, ramp_us)

        sys.segments[V3_DEPOWER_IDX].l0 =
            power_start + (power_target - power_start) * rf
        sys.segments[V3_STEERING_LEFT_IDX].l0 =
            steer_start_left +
            (target_left - steer_start_left) * rf
        sys.segments[V3_STEERING_RIGHT_IDX].l0 =
            steer_start_right +
            (target_right - steer_start_right) * rf

        if !sim_step!(sam;
                set_values=[0.0], dt=dt_c, vsm_interval=1)
            @error "Circular phase $phase failed" step
            break
        end

        log_state!(logger, sys_state, sam, t_total)

        if should_report(step, n_steps_c)
            elapsed = time() - sim_start_time
            @info "Phase $phase: $step/$n_steps_c (t=$(round(t_total, digits=2))s)" times_realtime=round(t_total / elapsed, digits=2)
        end
    end

    # Update start points for next phase
    power_start = sys.segments[V3_DEPOWER_IDX].l0
    steer_start_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    steer_start_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
    current_time += SIM_TIME_CIRCLES[phase]
end

total_sim = SIM_TIME_ZENITH + sum(SIM_TIME_CIRCLES)
report_performance(total_sim, time() - sim_start_time)

# =============================================================================
# Save and plot
# =============================================================================

log_name = "correct_circles_varying_us"
save_log(logger, log_name)
syslog = load_log(log_name)

fig = plot(sam.sys_struct, syslog;
    plot_turn_rates=true, plot_reelout=false,
    plot_gk=true, plot_aoa=true,
    plot_heading=false, plot_elevation=true,
    plot_azimuth=true, plot_winch_force=false,
    plot_set_values=false, plot_us=true,
    plot_aero_moment=true)

scene = replay(syslog, sam.sys_struct)

display(fig)
display(scene)

nothing
