# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Open-Loop Circular Flight

Single-phase simulation with ramped steering and depower using settled
geometry. Winch brake engaged (constant tether length).

Usage:
    julia --project=examples examples/open_loop.jl
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using CairoMakie
GLMakie.activate!()
using LinearAlgebra
using Statistics
using Dates

# =============================================================================
# Configuration
# =============================================================================

TETHER_LENGTH = 262
ELEVATION = 20.0            # degrees

# Geometry config
gc = V3GeomAdjustConfig()
GEOM_SUFFIX = build_geom_suffix(V3_DEPOWER_L0_BASE,
    V3_STEERING_L0_BASE, V3_STEERING_L0_BASE,
    gc.tip_reduction, gc.te_frac)
struc_candidate = "struc_geometry_$(GEOM_SUFFIX).yaml"
aero_candidate = "aero_geometry_$(GEOM_SUFFIX).yaml"
struc_yaml = isfile(joinpath(v3_data_path(), struc_candidate)) ?
    struc_candidate : "struc_geometry.yaml"
aero_yaml = isfile(joinpath(v3_data_path(), aero_candidate)) ?
    aero_candidate : "aero_geometry.yaml"

# Control
US = 0.1                   # Steering percentage [-100, 100]
UP = 0.42                   # Depower percentage [0, 100] (old fraction)
V_WIND = 7.6
DAMPING_PATTERN = [0.0, 0.0, 20.0]
STARTUP_DAMPING_PATTERN = [0.0, 30.0, 60.0]
STARTUP_DECAY_TIME = 2.0

# Ramp timing
RAMP_START_UP = 0.1
RAMP_END_UP = 1.5
RAMP_START_US = 3.0
RAMP_END_US = 5.0
STARTUP_HOLD_TIME = 0.5

SIM_TIME = 60.0
FPS = 240

damping_profile = function (t)
    if STARTUP_DECAY_TIME <= 0
        return DAMPING_PATTERN
    end
    mix = clamp(t / STARTUP_DECAY_TIME, 0.0, 1.0)
    return STARTUP_DAMPING_PATTERN .+
           (DAMPING_PATTERN .- STARTUP_DAMPING_PATTERN) .* mix
end

# =============================================================================
# Model setup (settle first, then run open-loop)
# =============================================================================

settle_config = V3SettleConfig(
    source_struc_path = struc_yaml,
    source_aero_path = aero_yaml,
    vsm_settings_path = "vsm_settings.yaml",
    v_wind = V_WIND,
    elevation = ELEVATION,
    tether_length = TETHER_LENGTH,
    body_damping = STARTUP_DAMPING_PATTERN,
    geom = gc,
)

@info "Creating V3 model..."
sam, _ = settle_wing(settle_config; remake=false)
sys = sam.sys_struct
sys.winches[1].brake = true

# Logger
n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps
logger, sys_state = create_logger(sam, n_steps)

# Start from the initialized trim and blend to the requested controls
nominal_steering = get_steering(sys, gc)
nominal_depower = get_depower(sys, gc)

# Stretch stats storage
max_stretch_samples = Float64[]
mean_stretch_samples = Float64[]
max_idx_samples = Int[]

# Tape logging
tape_times = Float64[]
tape_steering_pct = Float64[]
tape_depower_pct = Float64[]

# =============================================================================
# Simulation loop
# =============================================================================

@info "Starting simulation" n_steps dt
sim_start_time = time()

for step in 1:n_steps
    t = step * dt
    t_ctrl = max(0.0, t - STARTUP_HOLD_TIME)

    SymbolicAWEModels.set_body_frame_damping(
        sys, damping_profile(t))

    # Ramp steering
    rf_us = ramp_factor(t_ctrl, RAMP_START_US, RAMP_END_US)
    steering_cmd = nominal_steering +
                   (US - nominal_steering) * rf_us
    set_steering!(sys, steering_cmd, gc)

    # Ramp depower from initialized trim to requested input
    rf_up = ramp_factor(t_ctrl, RAMP_START_UP, RAMP_END_UP)
    depower_cmd = nominal_depower +
                  (UP - nominal_depower) * rf_up
    set_depower!(sys, depower_cmd, 0.0, gc)

    push!(tape_times, t)
    push!(tape_steering_pct, steering_cmd * 100)
    push!(tape_depower_pct, depower_cmd * 100)

    # Step
    if !sim_step!(sam; set_values=[0.0], dt, vsm_interval=1)
        @error "Simulation failed" step
        break
    end

    log_state!(logger, sys_state, sam, t)

    # Stretch stats after t > 1.0
    if t > 1.0
        ms, ms_mean, ms_idx = SymbolicAWEModels.segment_stretch_stats(
            sam.sys_struct)
        push!(max_stretch_samples, ms)
        push!(mean_stretch_samples, ms_mean)
        push!(max_idx_samples, ms_idx)
    end

    if should_report(step, n_steps)
        elapsed = time() - sim_start_time
        @info "Step $step/$n_steps (t=$(round(t, digits=2))s)" times_realtime=round(t / elapsed, digits=2)
    end
end

report_performance(SIM_TIME, time() - sim_start_time)

# Report stretch stats
if !isempty(max_stretch_samples)
    @info "Segment stretch (t > 1.0)" max_pct=round(maximum(max_stretch_samples) * 100, digits=4) mean_pct=round(mean(mean_stretch_samples) * 100, digits=4) worst_seg=max_idx_samples[argmax(max_stretch_samples)]
end

# =============================================================================
# Save and plot
# =============================================================================

lt_tag = Int(round(TETHER_LENGTH))
log_name = "open_loop_lt_$(lt_tag)"
save_log(logger, log_name)
syslog = load_log(log_name)

fig = plot(sam.sys_struct, syslog)

scene = replay(syslog, sam.sys_struct, show_panes=false)

display(fig)
display(scene)

nothing
