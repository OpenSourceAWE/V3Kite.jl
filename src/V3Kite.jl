# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
    V3Kite

Package for simulation and validation of the TU Delft V3 ram-air kite.
Provides calibration functions, model setup utilities, CSV replay capabilities,
and simulation functions built on top of SymbolicAWEModels.jl.
"""
module V3Kite

using SymbolicAWEModels
using VortexStepMethod
using KiteUtils
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using UnPack
using Rotations
using Dates
using DiscretePIDs
using HDF5

# Re-export commonly used types from SymbolicAWEModels
export SymbolicAWEModel, SystemStructure, Logger, SysState
export load_sys_struct_from_yaml, set_data_path, get_data_path
export init!, next_step!, update_sys_state!, log!, save_log, load_log, replay
export REFINE, QUATERNION, WING
export Settings
export SymbolicAWEModels
export record, replay

# Include submodules (model_setup before calibration: calibration uses V3GeomAdjustConfig)
include("model_setup.jl")
include("calibration.jl")
include("coordinate_utils.jl")
include("flight_data.jl")
include("photogrammetry.jl")
include("sim_helpers.jl")
include("simulation.jl")
include("stabilization.jl")

# Calibration exports
# Base values (official KCU measurements)
export V3_STEERING_L0_BASE, V3_DEPOWER_L0_BASE
# Gains
export V3_STEERING_GAIN, V3_DEPOWER_GAIN
# Segment indices
export V3_TETHER_POINT_IDXS
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
export adjust_tether_length!, adjust_elevation!
export segment_stretch_stats

# Coordinate utilities exports
export wrap_to_pi, euler_to_quaternion
export calc_heading, calc_csv_heading
export calc_R_b_w

# Flight data exports
export parse_time_to_seconds, unix_to_utc_seconds
export utc_to_video_frame
export load_flight_data, find_indices_by_utc, limit_by_utc
export add_distance_column, get_row_at_distance
export interpolate_flight_data
export update_sys_struct_from_data!

# Simulation helper exports
export create_logger, ramp_factor
export init_winch_torque!, force_to_torque
export sim_step!, log_state!, should_report
export save_and_load_log
export create_heading_pid, create_winch_pid
export report_performance
export build_replay_name

# Simulation exports
export V3SimConfig, create_v3_model, run_v3_simulation, v3_data_path
export V3_MODEL_NAME, V3_QUAT_MODEL_NAME

# Stabilization exports
export V3SettleConfig, settle_wing

# Photogrammetry exports
export load_extra_points

# Extension exports (provided by V3KiteMakieExt when GLMakie is loaded)
export plot_body_frame_local
export plot_yaw_rate_vs_steering

"""
    plot_body_frame_local(sys_structs; kwargs...)

Plot wing points in 2D body frame. Requires GLMakie to be loaded.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_body_frame_local end

"""
    plot_yaw_rate_vs_steering(syslogs, tapes; kwargs...)

Scatter plot of |yaw rate| vs |u_s * v_a|. Requires GLMakie.

This function is provided by the V3KiteMakieExt extension.
Load GLMakie before using: `using GLMakie`
"""
function plot_yaw_rate_vs_steering end

# include("precompile.jl") # disabled: precompilation workload

end # module
