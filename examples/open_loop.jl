# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Open-Loop Circular Flight

Single-phase simulation with ramped steering and depower using settled
geometry. Winch brake engaged (constant tether length).

Usage:
    julia --project=examples examples/open_loop.jl
"""

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

# Geometry (must match settle output)
TE_FRAC = 0.95
TIP_REDUCTION = 0.4
GEOM_SUFFIX = build_geom_suffix(V3_DEPOWER_L0, TIP_REDUCTION, TE_FRAC)

# Control
US = 0.05                   # Steering percentage [-100, 100]
UP = 0.42                   # Depower percentage [0, 100] (old fraction)
V_WIND = 7.6
DAMPING_PATTERN = [0.0, 0.0, 20.0]

# Ramp timing
RAMP_START_UP = 0.1
RAMP_END_UP = 1.5
RAMP_START_US = 3.0
RAMP_END_US = 5.0

SIM_TIME = 60.0
FPS = 120

# =============================================================================
# Model setup
# =============================================================================

config = V3SimConfig(
    struc_yaml_path = "struc_geometry_$(GEOM_SUFFIX).yaml",
    aero_yaml_path = "aero_geometry_$(GEOM_SUFFIX).yaml",
    vsm_settings_path = "CORRECT_vsm_settings.yaml",
    sim_time = SIM_TIME,
    fps = FPS,
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    up = UP * 100,        # Convert old 0-1 fraction to percentage
    us = US * 100,        # Convert old 0-1 fraction to percentage
    ramp_start_time_up = RAMP_START_UP,
    ramp_end_time_up = RAMP_END_UP,
    ramp_start_time_us = RAMP_START_US,
    ramp_end_time_us = RAMP_END_US,
    wing_type = REFINE,
    brake = true,
    damping_pattern = DAMPING_PATTERN,
    elevation = ELEVATION,
)

@info "Creating V3 model..."
sam, sys = create_v3_model(config)

@info "Initializing model..."
init!(sam; remake=config.remake_cache, ignore_l0=false, remake_vsm=true)
sys.winches[1].brake = true

# Logger
n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps
logger, sys_state = create_logger(sam, n_steps)

# Store nominal lengths for ramping
nominal_l0_left = sys.segments[V3_STEERING_LEFT_IDX].l0
nominal_l0_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
nominal_l0_depower = sys.segments[V3_DEPOWER_IDX].l0

# Target tape lengths from percentages
L_left_target, L_right_target = steering_percentage_to_lengths(
    US * 100)
L_depower_target = depower_percentage_to_length(UP * 100)

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

    # Ramp steering
    rf_us = ramp_factor(t, RAMP_START_US, RAMP_END_US)
    sys.segments[V3_STEERING_LEFT_IDX].l0 =
        nominal_l0_left + rf_us * (L_left_target - nominal_l0_left)
    sys.segments[V3_STEERING_RIGHT_IDX].l0 =
        nominal_l0_right + rf_us * (L_right_target - nominal_l0_right)

    # Instant depower
    sys.segments[V3_DEPOWER_IDX].l0 = L_depower_target

    push!(tape_times, t)
    push!(tape_steering_pct, rf_us * US * 100)
    push!(tape_depower_pct, UP * 100)

    # Step
    if !sim_step!(sam; set_values=[0.0], dt, vsm_interval=1)
        @error "Simulation failed" step
        break
    end

    log_state!(logger, sys_state, sam, t)

    # Stretch stats after t > 1.0
    if t > 1.0
        ms, ms_mean, ms_idx = segment_stretch_stats(
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

fig = plot(sam.sys_struct, syslog;
    plot_turn_rates=false, plot_reelout=false,
    plot_twist=false, plot_yaw_rate_paper=true,
    plot_v_app=true, plot_kite_vel=true,
    plot_gk=true, plot_aoa=true,
    plot_heading=false, plot_elevation=true,
    plot_azimuth=true, plot_winch_force=false,
    plot_set_values=false,
    yaw_rate_paper_ylims=(0.0, 50.0),
    ylims=Dict(:aoa => (0.0, 15.0), :gk => (0.0, 15.0)),
    plot_tether_actual=true, plot_us=true)

scene = replay(syslog, sam.sys_struct, show_panes=false)

display(fig)
display(scene)

nothing
