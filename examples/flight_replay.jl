# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Flight Data Replay Example

Reads flight test data from HDF5 and replays steering inputs
through the SymbolicAWEModel simulator. The kite is initialized
to steady state, then recorded steering commands are applied.

Usage:
    julia --project=examples examples/flight_replay.jl
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using SymbolicAWEModels: reposition!, rotate_around_z,
    rotate_around_y, calc_steady_torque, FBDF
using GLMakie
using CairoMakie
GLMakie.activate!()
using Statistics
using Rotations
using UnPack
using LinearAlgebra
using OrdinaryDiffEqBDF
using KiteUtils: calc_elevation, azimuth_east
using Dates
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

SECTION = "power_depower"
H5_PATH = joinpath(v3_data_path(),
    "flight_data", "ekf_awe_2025-10-09.h5")
WARN_STEP = false

# Video frame mapping
VIDEO_FRAME_REF = 7182
UTC_REF_SECONDS = 15*3600 + 36*60 + 31.0
VIDEO_FPS = 29.97

# Defaults for optional per-section settings
EXTRA_POINTS_CSV = nothing
EXTRA_POINTS_FRAME = nothing
STOP_EARLY = false

# Maneuver selection
if SECTION == "straight_right"
    START_UTC = "15:36:29.0"
    END_UTC = "15:36:41.0"
elseif SECTION == "straight_left"
    START_UTC = "15:36:49.0"
    END_UTC = "15:36:52.0"
elseif SECTION == "power_depower"
    START_UTC = "15:42:11.0"
    END_UTC = "15:42:22.0"
else
    error("Unknown section: $SECTION")
end

STEERING_MULTIPLIER = 1.0
MIN_DAMPING = [0.0, 0.0, 20.0]

# =============================================================================
# Replay helper functions
# =============================================================================

"""Feed-forward torque from tether force."""
function calc_feedforward_torque(tether_force_n, winch)
    return -winch.drum_radius / winch.gear_ratio *
        tether_force_n + winch.friction
end

# Distance tracker state
const SIM_PREV_POS = Ref{Vector{Float64}}(zeros(3))
const SIM_CUMULATIVE_DIST = Ref{Float64}(0.0)

function reset_distance_tracker!()
    SIM_PREV_POS[] = zeros(3)
    SIM_CUMULATIVE_DIST[] = 0.0
end

function update_sim_distance!(wing_pos)
    if SIM_PREV_POS[] == zeros(3)
        SIM_PREV_POS[] = copy(wing_pos)
        return 0.0
    end
    dist = norm(wing_pos - SIM_PREV_POS[])
    SIM_CUMULATIVE_DIST[] += dist
    SIM_PREV_POS[] = copy(wing_pos)
    return SIM_CUMULATIVE_DIST[]
end

function update_vel_from_csv!(sys, row, brake,
        gc::V3GeomAdjustConfig)
    @unpack wings, points, winches, segments = sys
    wing = wings[1]

    update_sim_distance!(wing.pos_w)
    sys.set.v_wind = row.wind_speed
    sys.set.upwind_dir = row.wind_dir + -90

    # CSV steering (KCU convention: positive = left turn)
    steering = clamp(row.steering, -100.0, 100.0)
    set_steering!(sys,
        -steering * STEERING_MULTIPLIER / 100.0, gc;
        min_l0=0.01)

    # Winch feed-forward
    winch = winches[1]
    ff_torque = calc_feedforward_torque(
        row.tether_force, winch)
    winch.brake = brake
    winch.set_value = ff_torque

    # Depower from CSV
    set_depower!(sys, row.depower / 100.0, gc)

    eff_steering = steering * STEERING_MULTIPLIER
    return winch.set_value, eff_steering, row.depower
end

# =============================================================================
# Main replay function
# =============================================================================

function run_physics_replay(h5_path;
        start_utc=START_UTC, end_utc=END_UTC,
        n_substeps=20)

    data = load_flight_data(h5_path)
    limited_data, _ = limit_by_utc(
        data, start_utc, end_utc)
    limited_data = add_distance_column(limited_data)

    @info "Interpolating flight data" n_substeps
    csv_data = interpolate_flight_data(
        limited_data, n_substeps)

    function get_row(data, step)
        return (
            time = data.time[step],
            video_frame = round(Int, data.video_frame[step]),
            roll = data.ekf_kite_roll[step],
            pitch = data.ekf_kite_pitch[step],
            yaw = data.ekf_kite_yaw[step],
            x = data.ekf_kite_position_x[step],
            y = data.ekf_kite_position_y[step],
            z = data.ekf_kite_position_z[step],
            vx = data.ekf_kite_velocity_x[step],
            vy = data.ekf_kite_velocity_y[step],
            vz = data.ekf_kite_velocity_z[step],
            tether_len = data.ekf_tether_length[step],
            tether_vel = data.tether_reelout_speed[step],
            tether_force = data.ground_tether_force[step],
            steering = data.kcu_actual_steering[step],
            depower = data.kcu_actual_depower[step],
            distance = data.distance[step],
            cumulative_distance =
                data.cumulative_distance[step],
            wind_speed =
                data.ekf_wind_speed_horizontal[step],
            wind_dir = data.ekf_wind_direction[step],
            angle_of_attack = deg2rad(
                data.ekf_wing_angle_of_attack[step]),
            v_app =
                data.ekf_kite_apparent_windspeed[step],
        )
    end

    # Settle wing with first CSV conditions
    row1 = get_row(csv_data, 1)
    tether_len = Float64(row1.tether_len)
    settle_config = V3SettleConfig(
        depower_pct=row1.depower,
        tether_length=tether_len,
        geom=V3GeomAdjustConfig(
            reduce_tip=true, reduce_te=true,
            reduce_depower=false,
            tether_length=tether_len))
    sam, syslog = settle_wing(settle_config;
        v_app=row1.v_app,
        tether_length=tether_len,
        remake=false)
    # wait(display(replay(syslog, sam.sys_struct)))
    sys_struct = sam.sys_struct
    set = sam.set
    set.l_tether = tether_len

    n_steps = length(csv_data.time)
    sys_state = SysState(sam)
    logger = Logger(sam, n_steps)

    # CSV reference model
    csv_sam = SymbolicAWEModel(set, deepcopy(sam.sys_struct))
    init!(csv_sam)
    csv_state = SysState(csv_sam)
    csv_logger = Logger(csv_sam, n_steps)

    # Tape storage
    csv_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])
    phys_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])

    extra_pts = nothing
    extra_groups = nothing

    @info "Replaying CSV data..."
    replay_start = time()
    sys = sam.sys_struct
    SymbolicAWEModels.set_body_frame_damping(
        sys, MIN_DAMPING, 1:38)

    dt = csv_data.time[2] - csv_data.time[1]
    first_csv_tether = csv_data.ekf_tether_length[1]
    tether_delta = set.l_tether - first_csv_tether

    reset_distance_tracker!()

    try
        for step in 1:n_steps-1
            row = get_row(csv_data, step)

            # Update CSV reference
            update_sys_struct_from_data!(
                csv_sam.sys_struct, row)
            SymbolicAWEModels.reinit!(
                csv_sam, csv_sam.prob, FBDF())
            update_sys_state!(csv_state, csv_sam)
            csv_state.winch_force[1] = row.tether_force
            csv_state.AoA = row.angle_of_attack
            csv_state.v_app = row.v_app
            csv_state.time = row.time
            csv_state.l_tether[1] = row.tether_len
            csv_state.v_reelout[1] = row.tether_vel
            csv_state.v_wind_gnd[1] = row.wind_speed
            log!(csv_logger, csv_state)

            push!(csv_tape.time, row.time)
            push!(csv_tape.steering, row.steering)
            push!(csv_tape.depower, row.depower)

            if step == 1
                update_sys_struct_from_data!(
                    sam.sys_struct, row)
                SymbolicAWEModels.reinit!(
                    sam, sam.prob, FBDF())
            end

            set_value, eff_steer, eff_dep =
                update_vel_from_csv!(
                    sam.sys_struct, row, true,
                    settle_config.geom)

            sam.sys_struct.winches[1].tether_len =
                row.tether_len + tether_delta
            sam.sys_struct.winches[1].tether_vel =
                row.tether_vel
            SymbolicAWEModels.reinit!(
                sam, sam.prob, FBDF())

            next_step!(sam; dt, set_values=[set_value])

            # Extra points comparison
            if !isnothing(EXTRA_POINTS_CSV) &&
               row.video_frame == EXTRA_POINTS_FRAME
                extra_pts, extra_groups =
                    load_extra_points(
                        EXTRA_POINTS_CSV, sam.sys_struct)
                if STOP_EARLY
                    break
                end
            end

            log_state!(logger, sys_state, sam, row.time)

            push!(phys_tape.time, row.time)
            push!(phys_tape.steering, eff_steer)
            push!(phys_tape.depower, eff_dep)

            if should_report(step, n_steps)
                elapsed = time() - replay_start
                @info "Step $step/$n_steps (t=$(round(row.time, digits=2))s, frame=$(row.video_frame))" times_realtime=round(row.time / elapsed, digits=2)
            end
        end
    catch err
        if err isa ErrorException &&
                contains(err.msg, "Unstable")
            @warn "Solver unstable, stopping replay" msg=err.msg
        else
            rethrow(err)
        end
    end

    @info "Replay done" elapsed=round(time() - replay_start, digits=2)

    save_log(logger, "csv_replay")
    save_log(csv_logger, "csv_reference")
    syslog = load_log("csv_replay")
    csvlog = load_log("csv_reference")

    return sam, syslog, csv_sam, csvlog, csv_data,
        phys_tape, csv_tape, extra_pts, extra_groups
end

# =============================================================================
# Main execution
# =============================================================================

sam, syslog, csv_sam, csvlog, csv_data, phys_tape, csv_tape,
    extra_pts, extra_groups = run_physics_replay(H5_PATH)

fig = plot([sam.sys_struct, csv_sam.sys_struct],
    [syslog, csvlog];
    plot_tether=true, plot_aero_force=false,
    plot_kite_vel=true, plot_wind=false,
    plot_reelout=false, plot_v_app=true,
    plot_turn_rates=true,
    tape_lengths=[phys_tape, csv_tape],
    suffixes=["phys", "csv"])

sphere = plot_sphere_trajectory([syslog, csvlog])

scene = replay([syslog, csvlog], [sam.sys_struct, csv_sam.sys_struct])
