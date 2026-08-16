# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Open-Loop Circular Flight

Power-zone settling followed by ramped open-loop steering and
depower. Winch brake engaged (constant tether length).
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using MakieControlPlots
using CairoMakie
GLMakie.activate!()
using SymbolicAWEModels
using LinearAlgebra
using Statistics
using Dates

# =============================================================================
# Configuration
# =============================================================================

TETHER_LENGTH = 262.0
ELEVATION = 70.0       # degrees
AZIMUTH = 0.0          # degrees

V_WIND = 7.6
US = 0.1               # Steering fraction [-1, 1]
UP = 0.25             # Depower fraction [0, 1]

# Ramp timing
RAMP_START_US = 3.0
RAMP_END_US = 5.0

SIM_TIME = 60.0
FPS = 20
VSM_INTERVAL = 1   # steps between VSM aero solves
PROJECT = "system.yaml"   # project file: geometry, settings and kite
AERO_MODE = ContinuousAero()

# =============================================================================
# Settling setup (matches flight_replay)
# =============================================================================

el_rad = deg2rad(ELEVATION)
az_rad = deg2rad(AZIMUTH)
position = [
    cos(el_rad) * cos(az_rad) * TETHER_LENGTH,
    cos(el_rad) * sin(az_rad) * TETHER_LENGTH,
    sin(el_rad) * TETHER_LENGTH,
]
velocity = [0.0, 0.0, 0.0]
heading = 0.0
wind_vec = [V_WIND, 0.0, 0.0]

kite = load_kite(PROJECT)
kite.aero_mode = AERO_MODE
settle_config = V3SettleConfig(
    project = PROJECT,
    kite = kite,
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    dt = 0.05,
    num_steps = 80,
    num_substeps = 1,
    decay_steps = 30,
    body_start_damping = [0.0, 0.0, 40.0],
    start_depower = UP * 100.0 + 10.0,
    course_correction_mode = :heading,
    course_correction_gain = 0.05,
)
gc = settle_config.kite.geom

@info "Settling V3 model..."
sam, settle_log, settle_failed = settle_wing(settle_config;
    position, velocity, heading,
    steering = 0.0, depower = UP, wind_vec)
settle_failed && error("Settling failed")
sys = sam.sys_struct
sys.winches[1].brake = true

# Logger
n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps
logger, sys_state = create_logger(sam, n_steps)

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

    rf_us = ramp_factor(t, RAMP_START_US, RAMP_END_US)
    set_steering!(sys, rf_us * US, gc)
    set_depower!(sys, UP, 0.0, gc)

    push!(tape_times, t)
    push!(tape_steering_pct, rf_us * US * 100)
    push!(tape_depower_pct, UP * 100)

    if !sim_step!(sam; set_values=[0.0], dt, vsm_interval = VSM_INTERVAL)
        @error "Simulation failed" step
        break
    end

    log_state!(logger, sys_state, sam, t)

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

fig = Makie.plot(sam.sys_struct, syslog)
scene = SymbolicAWEModels.replay(syslog, sam.sys_struct, show_panes=false)

display(fig)
display(scene)

nothing
