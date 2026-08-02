# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Package for simulation and validation of the TU Delft V3 ram-air kite.
Provides calibration functions, model setup utilities, CSV replay capabilities,
and simulation functions built on top of SymbolicAWEModels.jl.
"""
module V3Kite

using SymbolicAWEModels
using VortexStepMethod
using KiteUtils
using KitePodModels
using AtmosphericModels
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using UnPack
using Rotations
using Dates
using DiscretePIDs
using HDF5
using Serialization
using Parameters
using StaticArrays
using YAML

import KiteUtils: calc_elevation, calc_heading

# Re-export commonly used types from SymbolicAWEModels
export SymbolicAWEModel, SystemStructure, Logger, SysState
export load_sys_struct_from_yaml, set_data_path, get_data_path
export init!, next_step!, update_sys_state!, log!, save_log, load_log, replay
export PARTICLE_DYNAMICS, RIGID_DYNAMICS, WING
export Settings
export SymbolicAWEModels
export record, replay
export V3KITE

# Include submodules (model_setup before calibration: calibration uses V3GeomAdjustConfig)
include("model_setup.jl")
include("calibration.jl")
include("coordinate_utils.jl")
include("turn_rate_id.jl")
include("ripple_metrics.jl")
include("flight_data.jl")
include("photogrammetry.jl")
include("sim_helpers.jl")
include("turn_rate_table.jl")
include("fig8_controller.jl")
include("fig8_metrics.jl")
include("simulation.jl")
include("stabilization.jl")
include("wc_settings.jl")
include("fc_settings.jl")
include("interface.jl")

"""
    __init__()

Runs once per Julia session that loads V3Kite (including under a precompiled
sysimage, unlike top-level `include`d code). Reads `data/turn_rate_coeffs.yaml`
into [`turn_rate_coeffs`](@ref)'s table — see
[`reload_turn_rate_table!`](@ref) to refresh it mid-session, e.g. after
`examples/build_turn_rate_table.jl` appends rows.
"""
function __init__()
    reload_turn_rate_table!()
    return nothing
end

# Calibration exports
# Base values (official KCU measurements)
export V3_STEERING_L0_BASE, V3_DEPOWER_L0_BASE
# Gains
export V3_STEERING_GAIN, V3_DEPOWER_GAIN
# Segment indices
export V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX
export V3_DEPOWER_IDX
# Conversion functions
export steering_percentage_to_lengths
export depower_percentage_to_length
export steering_length_to_percentage, depower_length_to_percentage
export build_geom_suffix
# Normalized control functions (KiteModels.jl compatible)
export get_steering, set_steering!
export get_depower, set_depower!

# Model setup exports
export V3GeomAdjustConfig, apply_geom_adjustments!
export distribute_wing_drag!, distribute_wing_mass!
export set_v3_body_damping!
export generate_drag_adjusted_polars
export segment_stretch_stats

# Coordinate utilities exports
export wrap_to_pi, euler_to_quaternion
export calc_heading, calc_csv_heading
export calc_R_b_w, calc_turn_rate
export COURSE_RATE_WINDOW_SEC

# Turn-rate-law identification exports
export identify_turn_rate_law, format_turn_rate_report
export estimate_delay, shift_delay, turn_rate_gain, fit_c1_c2, est_steering

# AoA-ripple analysis exports
export RippleSettings, ripple_metrics, aoa_ripple, format_ripple_report

# Flight data exports
export parse_time_to_seconds, unix_to_utc_seconds
export utc_to_video_frame
export load_flight_data, find_indices_by_utc, limit_by_utc
export add_distance_column, get_row_at_distance,
    find_closest_trajectory_index
export interpolate_flight_data
export update_sys_struct_from_data!
export compute_wind_vec, interpolate_lidar_wind

# Simulation helper exports
export create_logger, ramp_factor, timestamp_colmeta, log_created_at
export init_winch_torque!, force_to_torque, winch_force_torque!
export sim_step!, log_state!, should_report
export compute_drag, compute_lift, compute_lift_drag, compute_tether_drag
export compute_drag_coeff, compute_lift_coeff, drag_floor
export compute_tether_drag_coeff, compute_bridle_drag_coeff
export compute_kcu_drag_coeff
export compute_kite_aoa, compute_bridle_euler, span_mean_aoa
export compute_wing_incidence
export compute_bridle_pitch_angle
export chord_ref_mid
export mean_te_segment_force
export save_and_load_log
export create_heading_pid, create_winch_pid
export report_performance
export build_replay_name
export find_frame_syslog_idxs, build_replay_sys_struct

# Figure-of-eight guidance exports
export FigureEightSettings, FigureEightController
export figure_eight_path, calc_attractor, navigate_fig8, set_path_center!
export path_tangent
export min_turn_radius, path_min_radius, path_radius_profile, check_pattern_feasible
export V3_TURN_RATE_COEFFS, turn_rate_coeffs, V3_TURN_RATE_C1, V3_TURN_RATE_C2
export reload_turn_rate_table!
export fig8_metrics, print_fig8_metrics

# Simulation exports
export V3SimConfig, create_v3_model, run_v3_simulation, v3_data_path
export V3_MODEL_NAME, V3_RIGID_DYNAMICS_MODEL_NAME

# Winch-controller settings export
export WC_Settings

# Figure-eight flight-controller settings export
export FC_Settings

# Interface exports
export init, step!, warmup!
export lift_drag, total_drag, unstretched_length, v_wind_kite, pos_kite, tether_length, calc_height, cl_cd, winch_force, reel_out_speed, states, spring_forces
export calc_elevation, calc_azimuth, calc_azimuth_north, calc_azimuth_east, upwind_dir, calc_heading, calc_course
export kite_ref_frame, calc_orient_quat, orient_euler

# Stabilization exports
export V3SettleConfig, settle_wing

# Photogrammetry exports
export load_extra_points

# Extension exports (provided by V3KiteMakieExt when GLMakie is loaded)
export plot_body_frame_local, plot_twist_dist
export plot_photogrammetry
export plot_yaw_rate_vs_steering, plot_turn_rate_vs_time
export plot_wind_compare
export plot_replay, plot_sphere_trajectory
export plot_2d_trajectory, plot_2d_panels

"""
    plot_body_frame_local(sys_structs; kwargs...)

Plot wing points in 2D body frame. Requires GLMakie to be loaded.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_body_frame_local end

"""
    plot_twist_dist(sys_structs; kwargs...)

Plot twist distribution along the span. Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_twist_dist end

"""
    plot_photogrammetry(points, groups; dir, kwargs...)

Plot photogrammetry points in 2D. Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_photogrammetry end

"""
    plot_yaw_rate_vs_steering(syslogs; source, kwargs...)

Scatter plot of |turn rate| vs |u_s * v_a|. Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_yaw_rate_vs_steering end

"""
    plot_turn_rate_vs_time(syslogs; sources, labels, dt, kwargs...)

Time-series plot of frame-transport-corrected turn rate, with
one or more `sources` (`:heading` and/or `:course`) per log.
Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_turn_rate_vs_time end

"""
    plot_wind_compare(syslog; kwargs...)

Plot EKF vs lidar wind components (x, y, z) over time.
Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_wind_compare end

"""
    plot_replay(syss, logs; tape_lengths, suffixes, size)

Custom time-series plot for V3 kite replay data. Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_replay end

"""
    plot_sphere_trajectory(logs; radius, colors, labels, kwargs...)

Plot kite trajectories on a unit sphere with body-frame axes
at final positions. Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_sphere_trajectory end

"""
    plot_2d_trajectory(logs; kwargs...)

Plot kite y vs z position colored by a gradient quantity
(`:vel` or `:steering`). Requires GLMakie/CairoMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_2d_trajectory end

"""
    plot_2d_panels(logs; tapes, labels, kwargs...)

Time-series panel plot (steering, winch force, v_app, etc.)
split from `plot_2d_trajectory`. Requires GLMakie/CairoMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_2d_panels end

include("precompile.jl") # precompilation workload (see PlanPrecompile.md)

end # module
