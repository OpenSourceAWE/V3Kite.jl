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

using V3Kite
using VortexStepMethod
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

using Dates
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

SECTION = "straight_right_2"
H5_PATH = joinpath(v3_data_path(),
    "flight_data", "ekf_awe_2025-10-09.h5")
# H5_PATH = joinpath(v3_data_path(),
#     "flight_data", "ekf_awe_2019-10-08.h5")
WARN_STEP = false

# Video frame mapping
VIDEO_FRAME_REF = 7182
UTC_REF_SECONDS = 15*3600 + 36*60 + 31.0
VIDEO_FPS = 29.97

EXTRA_POINTS_CSV = nothing
EXTRA_POINTS_FRAME = nothing
STOP_EARLY = false
SETTLE_ONLY = false
SETTLE = true
DEPOWER_OFFSET_2019 = 7.0
DEPOWER_OFFSET_2025 = -4.0

# Maneuver selection
if SECTION == "straight_right"
    START_UTC = "15:36:29.0"
    END_UTC = "15:36:41.0"
elseif SECTION == "straight_left"
    START_UTC = "15:36:49.0"
    END_UTC = "15:36:52.0"
elseif SECTION == "power_depower"
    START_UTC = "15:42:11.0"
    END_UTC = "15:42:21.0"
elseif SECTION == "straight_right_2"
    START_UTC = "15:37:39.0"
    END_UTC = "15:37:49.0"
elseif SECTION == "straight_right_2019"
    START_UTC = "11:10:00.0"
    END_UTC = "11:10:10.0"
else
    error("Unknown section: $SECTION")
end

STEERING_MULTIPLIER = 1.0
HEADING_KP = 0.3
BODY_DAMPING = [0.0, 0.0, 20.0]

# =============================================================================
# Replay helper functions
# =============================================================================

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

function update_vel_from_csv!(sys, row,
        gc::V3GeomAdjustConfig;
        heading_correction=0.0)
    sys.set.v_wind = row.wind_speed
    sys.set.upwind_dir = row.wind_dir + -90

    # CSV steering (positive = right turn)
    steering = clamp(row.steering, -100.0, 100.0)
    set_steering!(sys,
        steering * STEERING_MULTIPLIER / 100.0 +
            heading_correction, gc;
        min_l0=0.01)

    # Speed-controlled winch
    winch = sys.winches[1]
    winch.speed_controlled = true
    winch.tether_vel = row.tether_vel
    winch.tether_len = row.tether_len

    # Depower from CSV
    set_depower!(sys, row.depower / 100.0, gc)

    eff_steering = steering * STEERING_MULTIPLIER +
        heading_correction * 100.0
    return eff_steering, row.depower
end

# =============================================================================
# Main replay function
# =============================================================================

function run_physics_replay(h5_path;
        start_utc=START_UTC, end_utc=END_UTC,
        n_substeps=20)

    data = load_flight_data(h5_path)
    @show unix2datetime(data.unix_time[1])
    limited_data, _ = limit_by_utc(
        data, start_utc, end_utc)
    limited_data = add_distance_column(limited_data)

    @info "Interpolating flight data" n_substeps
    data = interpolate_flight_data(
        limited_data, n_substeps)

    is_2019 = occursin("2019", basename(h5_path))

    function make_row(raw)
        dp = raw.kcu_actual_depower
        if is_2019
            # u_dp_2025 = 0.2564 - 0.0768 * u_p_2019
            dp = (0.2564 - 0.0768 * dp / 100.0) * 100.0
            dp += DEPOWER_OFFSET_2019
        else
            dp += DEPOWER_OFFSET_2025
        end
        return (
            time = raw.time,
            video_frame = round(Int, raw.video_frame),
            roll = raw.ekf_kite_roll,
            pitch = raw.ekf_kite_pitch,
            yaw = raw.ekf_kite_yaw,
            x = raw.ekf_kite_position_x,
            y = raw.ekf_kite_position_y,
            z = raw.ekf_kite_position_z,
            vx = raw.ekf_kite_velocity_x,
            vy = raw.ekf_kite_velocity_y,
            vz = raw.ekf_kite_velocity_z,
            tether_len = raw.ekf_tether_length,
            tether_vel = raw.tether_reelout_speed,
            tether_force = raw.ground_tether_force,
            steering = raw.kcu_actual_steering,
            depower = dp,
            distance = raw.distance,
            cumulative_distance = raw.cumulative_distance,
            wind_speed = raw.ekf_wind_speed_horizontal,
            wind_dir = raw.ekf_wind_direction,
            angle_of_attack = deg2rad(
                raw.ekf_wing_angle_of_attack),
            v_app = raw.ekf_kite_apparent_windspeed,
        )
    end

    function get_row(data, step)
        ks = keys(data)
        raw = NamedTuple{ks}(
            Tuple(data[k][step] for k in ks))
        return make_row(raw)
    end

    # Settle wing with first CSV conditions
    row1 = get_row(data, 1)
    tether_len = Float64(row1.tether_len)
    settle_config = V3SettleConfig(
        world_damping=0.0,
        body_damping=BODY_DAMPING,
        min_damping=0.0,
        v_wind=row1.v_app,
        tether_length=tether_len,
        dt=0.001,
        num_steps=100,
        num_substeps=5,
        start_depower=row1.depower+10.0,
        geom=V3GeomAdjustConfig(
            reduce_tip=true, reduce_te=true,
            reduce_depower=false),
        fix_sphere_idxs=[])
    if SETTLE
        sam, settle_log = settle_wing(settle_config;
            init_row=row1, power_zone=true, remake=false)
    else
        data_path = v3_data_path()
        set_data_path(data_path)
        set = Settings("system.yaml")
        set.v_wind = row1.v_app
        set.l_tether = tether_len
        set.profile_law = 0

        gc = settle_config.geom
        source_struc = joinpath(
            data_path, settle_config.source_struc_path)
        source_aero = joinpath(
            data_path, settle_config.source_aero_path)
        vsm_path = joinpath(
            data_path, settle_config.vsm_settings_path)
        vsm_set = VortexStepMethod.VSMSettings(
            vsm_path; data_prefix=false)
        vsm_set.wings[1].n_panels =
            settle_config.n_panels
        vsm_set.wings[1].geometry_file = source_aero

        sys = load_sys_struct_from_yaml(source_struc;
            system_name=V3_MODEL_NAME, set,
            wing_type=SymbolicAWEModels.REFINE, vsm_set)
        sam = SymbolicAWEModel(set, sys)
        apply_geom_adjustments!(sys, gc)
        SymbolicAWEModels.init!(sam;
            remake=false, ignore_l0=false,
            remake_vsm=true)
        settle_log = nothing
    end
    if SETTLE_ONLY
        return sam, nothing, nothing, nothing, data,
            nothing, nothing, nothing, nothing,
            settle_config, settle_log
    end
    sys_struct = sam.sys_struct
    set = sam.set
    set.l_tether = tether_len

    n_steps = length(data.time)
    sys_state = SysState(sam)
    logger = Logger(sam, n_steps)

    # CSV reference model
    data_sam = SymbolicAWEModel(set, deepcopy(sam.sys_struct))
    init!(data_sam)
    data_state = SysState(data_sam)
    data_logger = Logger(data_sam, n_steps)

    # Tape storage
    data_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])
    sim_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])

    extra_pts = nothing
    extra_groups = nothing

    @info "Replaying CSV data..."
    replay_start = time()
    sys = sam.sys_struct
    SymbolicAWEModels.set_body_frame_damping(
        sys, BODY_DAMPING, 1:38)

    dt = data.time[2] - data.time[1]
    reset_distance_tracker!()
    heading_pid = create_heading_pid(;
        K=HEADING_KP, dt)

    max_dist = data.cumulative_distance[end]

    # Log full CSV reference independently of sim
    for step in 1:n_steps-1
        row = get_row(data, step)
        update_sys_struct_from_data!(
            data_sam.sys_struct, row)
        SymbolicAWEModels.reinit!(
            data_sam, data_sam.prob, FBDF())
        update_sys_state!(data_state, data_sam)
        data_state.winch_force[1] = row.tether_force
        data_state.AoA = row.angle_of_attack
        data_state.v_app = row.v_app
        data_state.time = row.cumulative_distance
        data_state.l_tether[1] = row.tether_len
        data_state.v_reelout[1] = row.tether_vel
        data_state.v_wind_gnd[1] = row.wind_speed
        log!(data_logger, data_state)

        push!(data_tape.time,
            row.cumulative_distance)
        push!(data_tape.steering, row.steering)
        push!(data_tape.depower, row.depower)
    end

    try
        for step in 1:n_steps-1
            row = get_row(data, step)

            if step == 1
                update_sys_struct_from_data!(
                    sam.sys_struct, row)
                SymbolicAWEModels.reinit!(
                    sam, sam.prob, FBDF())
            end

            # Update sim distance before input lookup
            update_sim_distance!(
                sam.sys_struct.wings[1].pos_w)
            if SIM_CUMULATIVE_DIST[] >= max_dist
                break
            end

            # Distance-based input lookup for physics
            phys_row = make_row(
                get_row_at_distance(
                    data, SIM_CUMULATIVE_DIST[]))

            data_heading = calc_csv_heading(
                phys_row.roll, phys_row.pitch,
                phys_row.yaw)
            sim_heading =
                sam.sys_struct.wings[1].heading
            heading_error = wrap_to_pi(
                data_heading - sim_heading)
            heading_correction = heading_pid(
                heading_error, 0.0, 0.0)

            eff_steer, eff_dep =
                update_vel_from_csv!(
                    sam.sys_struct, phys_row,
                    settle_config.geom;
                    heading_correction)

            SymbolicAWEModels.reinit!(
                sam, sam.prob, FBDF())

            next_step!(sam; dt)

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

            log_state!(logger, sys_state, sam,
                SIM_CUMULATIVE_DIST[])

            push!(sim_tape.time,
                SIM_CUMULATIVE_DIST[])
            push!(sim_tape.steering, eff_steer)
            push!(sim_tape.depower, eff_dep)

            if should_report(step, n_steps)
                elapsed = time() - replay_start
                dist = round(
                    SIM_CUMULATIVE_DIST[], digits=1)
                @info "Step $step/$n_steps" *
                    " (d=$(dist)m," *
                    " frame=$(row.video_frame))"
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

    base_name = build_replay_name(h5_path, start_utc,
        end_utc, row1.depower, row1.steering,
        settle_config.geom)
    save_log(logger, base_name * "_sim")
    save_log(data_logger, base_name * "_data")
    syslog = load_log(base_name * "_sim")
    datalog = load_log(base_name * "_data")

    return sam, syslog, data_sam, datalog, data,
        sim_tape, data_tape, extra_pts, extra_groups,
        settle_config, settle_log, dt
end

# =============================================================================
# Main execution
# =============================================================================

sam, syslog, data_sam, datalog, data, sim_tape, data_tape,
    extra_pts, extra_groups,
    settle_config, settle_log, dt = run_physics_replay(H5_PATH)
geom_config = settle_config.geom

if !SETTLE_ONLY
    fig = plot([sam.sys_struct, data_sam.sys_struct],
        [syslog, datalog];
        plot_tether=true, plot_aero_force=true,
        plot_kite_vel=true, plot_wind=false,
        plot_reelout=false, plot_v_app=true,
        tape_lengths=[sim_tape, data_tape],
        suffixes=["sim", "data"])

    sphere = plot_sphere_trajectory([syslog, datalog])

    yaw_fig = plot_yaw_rate_vs_steering(
        [syslog, datalog],
        [sim_tape, data_tape];
        labels=["sim", "data"], dt)

    scene = replay([syslog, datalog],
        [sam.sys_struct, data_sam.sys_struct])
end
