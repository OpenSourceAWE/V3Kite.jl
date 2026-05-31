# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Flight Data Replay Example

Reads flight test data from HDF5 and replays steering inputs
through the SymbolicAWEModel simulator. The kite is initialized
to steady state, then recorded steering commands are applied.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using VortexStepMethod
using SymbolicAWEModels: FBDF, update_from_sysstate!
using GLMakie
using CairoMakie
GLMakie.activate!()
using Statistics
using Rotations
using UnPack
using LinearAlgebra
using OrdinaryDiffEqBDF
using KiteUtils

using Dates

# =============================================================================
# Configuration
# =============================================================================

generate_drag_adjusted_polars(1.0)

LOAD_FROM_DISK = false   # toggle to skip sim and just plot
BASE_SECTION = "straight_right"
YEAR = 2025
SETTLE = true
REMAKE_SETTLE = true
IS_VISUALIZE_SETTLE = false #TODO: new flag
IS_WITH_ONE_SHOT_ADJUST_POS_VEL_ORIENT = false #TODO: new flag
DEPOWER_OFFSET_2019 = 7.0
DEPOWER_OFFSET_2025 = -16.0
STEERING_MULTIPLIER = 1.0
EXTRA_WING_DRAG_COEFF = 0.0
HEADING_KP = 0.0
HEADING_TI = 0.0
LATERAL_KP = 0.0
STEERING_OFFSET = 1.5 #TODO: steering offset, used to be 1.5%, now set to 0.
DISTANCE_BASED_STEERING = false
REDUCE_STEERING = false #TODO: set to false
STEERING_REDUCTION = 0.2
REDUCE_TIP = false #tip segment adjustment #TODO: adjusted to false
TIP_REDUCTION = 0.2 # amount removed in meters
BODY_DAMPING = [0.0, 0.0, 20.0]
# Photogrammetry linear AoA offset model:
AOA_OFFSET_A = -0.6831
AOA_OFFSET_B = 28.74
POINT_37_38_DAMPING = [0.0, 20.0, 20.0] # part of wing so is identical between old&new yaml
SAVE_FIGS = true
FIGURES_DIR = joinpath(@__DIR__, "..", "output")
REPLAY_LOG_DIR = joinpath(dirname(@__DIR__), "processed_data")
WIND_SOURCE_SPEED = :ekf   # :ekf or :lidar
WIND_SOURCE_DIR = :ekf   # :ekf or :lidar (also vert)
# input yaml files
DEFAULT_STRUC_GEOM_YAML = "python_yamls/struc_geometry_julia_generated.yaml"
DEFAULT_AERO_GEOM_YAML = "aero_geometry.yaml"
DEFAULT_VSM_SETTINGS_PATH = "vsm_settings.yaml"

# Maneuver selection
if YEAR == 2025
    h5_path = joinpath(v3_data_path(),
        "flight_data", "ekf_awe_2025-10-09.h5")
else
    h5_path = joinpath(v3_data_path(),
        "flight_data", "ekf_awe_2019-10-08.h5")
end
depower_offset_pct = YEAR == 2019 ?
                     DEPOWER_OFFSET_2019 : DEPOWER_OFFSET_2025
SECTION = "$(BASE_SECTION)_$(YEAR)"
if SECTION == "straight_right_2025"
    start_utc = "15:36:29.0"
    end_utc = "15:36:38.0"
elseif SECTION == "straight_left_2025"
    start_utc = "15:36:47.0"
    end_utc = "15:36:57.0"
elseif SECTION == "power_depower_2025"
    start_utc = "15:42:10.0"
    end_utc = "15:42:21.0"
elseif SECTION == "straight_right_2019"
    start_utc = "11:10:03.0"
    end_utc = "11:10:13.0"
elseif SECTION == "straight_left_2019"
    start_utc = "11:10:13.0"
    end_utc = "11:10:23.0"
else
    error("Unknown section: $SECTION")
end

# Auto-pick frame CSVs for 2025 flights
if YEAR == 2025
    _start_f = utc_to_video_frame(
        parse_time_to_seconds(start_utc))
    _end_f = utc_to_video_frame(
        parse_time_to_seconds(end_utc))
    frame_csvs = Tuple{String,Int}[]
    for f in readdir(v3_data_path(); join=true)
        m = match(r"frame_(\d+)\.csv$", f)
        isnothing(m) && continue
        frame = parse(Int, m.captures[1])
        if _start_f <= frame <= _end_f
            push!(frame_csvs, (f, frame))
        end
    end
    isempty(frame_csvs) &&
        @warn "No frame CSV in range" _start_f _end_f
else
    frame_csvs = Tuple{String,Int}[]
end

# =============================================================================
# Replay helper functions
# =============================================================================

function update_vel_from_csv!(sys, row,
    gc::V3GeomAdjustConfig;
    heading_correction=0.0)
    sys.set.wind_vec = KiteUtils.MVec3(row.wind_vec)

    # CSV steering (positive = right turn)
    steering = clamp(row.steering, -1.0, 1.0)
    set_steering!(sys,
        steering * STEERING_MULTIPLIER +
        heading_correction, gc;
        min_l0=0.01)

    # Speed-controlled winch
    winch = sys.winches[1]
    winch.speed_controlled = true
    winch.vel = row.tether_vel
    sys.tethers[1].len = row.tether_len
    sys.tethers[1].stretched_len = row.tether_len

    # Depower from CSV (returns adjusted dp)
    eff_depower = set_depower!(
        sys, row.depower, row.steering, gc)

    eff_steering = steering * STEERING_MULTIPLIER +
                   heading_correction
    return eff_steering, eff_depower
end

# =============================================================================
# Main replay function
# =============================================================================

function run_physics_replay(h5_path;
    start_utc=start_utc, end_utc=end_utc,
    n_substeps=20)

    full_data = load_flight_data(h5_path)
    limited_data, _ = limit_by_utc(
        full_data, start_utc, end_utc)
    limited_data = add_distance_column(limited_data)
    total_data_dist =
        limited_data.cumulative_distance[end]

    # Extended data for distance-based steering lookup
    # (from start_utc to end of recording, so we can
    # look ahead beyond end_utc)
    dist_lookup_data, _ = limit_by_utc(
        full_data, start_utc)
    dist_lookup_data = add_distance_column(
        dist_lookup_data)

    @info "Interpolating flight data" n_substeps
    data = interpolate_flight_data(
        limited_data, n_substeps)

    is_2019 = occursin("2019", basename(h5_path))

    function make_row(raw)
        dp = raw.kcu_actual_depower
        if is_2019
            dp = 0.2564 - 0.0768 * dp / 100.0
        else
            dp = dp / 100.0
        end
        quat2R =
            SymbolicAWEModels.quaternion_to_rotation_matrix
        R_b_w = quat2R(euler_to_quaternion(
            raw.ekf_kite_roll, raw.ekf_kite_pitch,
            raw.ekf_kite_yaw))
        alt = raw.ekf_kite_position_z
        wind_vec_ekf = compute_wind_vec(raw, alt;
            speed_source=:ekf, dir_source=:ekf)
        wind_vec_lidar = compute_wind_vec(raw, alt;
            speed_source=:lidar, dir_source=:lidar)
        wind_vec = compute_wind_vec(raw, alt;
            speed_source=WIND_SOURCE_SPEED,
            dir_source=WIND_SOURCE_DIR)
        wh = sqrt(wind_vec[1]^2 + wind_vec[2]^2)
        wv = wind_vec[3]
        wdir = atan(wind_vec[2], wind_vec[1])
        kite_vel = [raw.ekf_kite_velocity_x,
            raw.ekf_kite_velocity_y,
            raw.ekf_kite_velocity_z]
        kite_aoa = compute_kite_aoa(
            R_b_w, kite_vel, wind_vec)
        wing_aoa = kite_aoa + deg2rad(
            AOA_OFFSET_A * dp * 100 + AOA_OFFSET_B)
        kite_pos = [raw.ekf_kite_position_x,
            raw.ekf_kite_position_y,
            raw.ekf_kite_position_z]
        return (
            time=raw.time,
            video_frame=round(Int, raw.video_frame),
            roll=raw.ekf_kite_roll,
            pitch=raw.ekf_kite_pitch,
            yaw=raw.ekf_kite_yaw, # TODO: try using kin
            heading=calc_csv_heading(
                raw.ekf_kite_roll, raw.ekf_kite_pitch,
                raw.ekf_kite_yaw, kite_pos),
            x=kite_pos[1],
            y=kite_pos[2],
            z=kite_pos[3],
            vx=raw.ekf_kite_velocity_x,
            vy=raw.ekf_kite_velocity_y,
            vz=raw.ekf_kite_velocity_z,
            tether_len=raw.ekf_tether_length,
            tether_vel=raw.tether_reelout_speed,
            tether_force=raw.ground_tether_force,
            steering=raw.kcu_actual_steering / 100.0,
            depower=dp,
            distance=raw.distance,
            cumulative_distance=raw.cumulative_distance,
            wind_speed=wh,
            upwind_dir=wrap_to_pi(
                -wdir - π / 2),
            wind_speed_vertical=wv,
            R_b_w=R_b_w,
            v_app=raw.ekf_kite_apparent_windspeed,
            drag_coeff=raw.ekf_wing_drag_coefficient,
            lift_coeff=raw.ekf_wing_lift_coefficient,
            tether_drag_coeff=
            raw.ekf_tether_drag_coefficient,
            bridle_drag_coeff=
            raw.ekf_bridles_drag_coefficient,
            kcu_drag_coeff=
            raw.ekf_kcu_drag_coefficient,
            wind_elevation=atan(wv, wh),
            wind_vec=wind_vec,
            wind_vec_ekf=wind_vec_ekf,
            wind_vec_lidar=wind_vec_lidar,
            kite_aoa=kite_aoa,
            wing_aoa=wing_aoa,
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
        source_struc_path=DEFAULT_STRUC_GEOM_YAML,
        source_aero_path=DEFAULT_AERO_GEOM_YAML,
        vsm_settings_path=DEFAULT_VSM_SETTINGS_PATH,
        world_damping=100.0,
        decay_steps=400,
        body_damping=BODY_DAMPING * 2.0,
        body_damping_overrides=[
            (37:38, POINT_37_38_DAMPING * 2.0)],
        min_damping=0.0,
        v_wind=row1.v_app,
        tether_length=tether_len,
        dt=0.001,
        num_steps=800,
        num_substeps=5,
        start_depower=nothing,
        course_correction_gain=0.02,
        course_correction_mode=:heading,
        geom=V3GeomAdjustConfig(
            reduce_tip=REDUCE_TIP, reduce_te=true,
            reduce_depower=false,
            reduce_steering=REDUCE_STEERING,
            steering_reduction=STEERING_REDUCTION,
            tip_reduction=TIP_REDUCTION,
            depower_offset=depower_offset_pct / 100.0),
        fix_sphere_idxs=[])
    settle_log = nothing
    sam = nothing
    data_sam = nothing
    logger = nothing
    data_logger = nothing
    replay_start = time()
    dt = data.time[2] - data.time[1]

    try

        data_path = v3_data_path()
        source_struc = joinpath(
            data_path, settle_config.source_struc_path)
        source_aero = joinpath(
            data_path, settle_config.source_aero_path)
        vsm_path = joinpath(
            data_path, settle_config.vsm_settings_path)
        vsm_set = VortexStepMethod.VSMSettings(
            vsm_path; data_prefix=false)
        vsm_set.wings[1].geometry_file = source_aero

        if SETTLE
            sam, settle_log, settle_failed =
                settle_wing(settle_config, row1;
                    remake=REMAKE_SETTLE)
            if IS_VISUALIZE_SETTLE
                if isnothing(settle_log)
                    @warn "No settle log to visualize. Set REMAKE_SETTLE=true if cached settled geometry was reused."
                elseif !isnothing(sam)
                    settle_fig = plot(sam.sys_struct,
                        settle_log; plot_tether=true)
                    display(settle_fig)

                    settle_scene = replay(settle_log,
                        sam.sys_struct)
                    display(settle_scene)
                end
            end
            if settle_failed
                @warn "Settling failed — skipping sim"
                base_name = build_replay_name(h5_path,
                    start_utc, end_utc,
                    row1.depower, row1.steering,
                    settle_config.geom)
                return sam, nothing, nothing, nothing, data,
                settle_config, settle_log, dt, base_name
            end
        else
            set_data_path(data_path)
            set = Settings("system.yaml")
            set.g_earth = 9.81
            set.v_wind = row1.v_app
            set.l_tether = tether_len
            set.profile_law = 0

            gc = settle_config.geom
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
        set = sam.set
        set.l_tether = tether_len

        # -------------------------------------------------------------
        # Optional one-shot post-settle alignment:
        # Re-impose row1 pose/velocity after settling so replay starts
        # from measured kinematics while keeping settled geometry.
        # -------------------------------------------------------------
        if IS_WITH_ONE_SHOT_ADJUST_POS_VEL_ORIENT
            update_sys_struct_from_data!(
                sam.sys_struct, row1;
                config=settle_config.geom)
            SymbolicAWEModels.reinit!(
                sam, sam.prob, FBDF())
            @info "Applied one-shot post-settle row1 alignment" enabled = IS_WITH_ONE_SHOT_ADJUST_POS_VEL_ORIENT
        end

        n_data_steps = length(data.time)
        max_sim_steps = DISTANCE_BASED_STEERING ?
                        n_data_steps * 3 : n_data_steps
        sys_state = SysState(sam)
        logger = Logger(sam, max_sim_steps)

        # CSV reference model
        data_struct = load_sys_struct_from_yaml(source_struc;
            system_name=V3_MODEL_NAME, set,
            wing_type=SymbolicAWEModels.REFINE, vsm_set)
        data_sam = SymbolicAWEModel(set, data_struct)
        data_sam.sys_struct.tethers[1].init_unstretched_len = tether_len
        data_sam.sys_struct.tethers[1].init_stretched_len = tether_len
        init!(data_sam)
        data_state = SysState(data_sam)
        data_logger = Logger(data_sam, n_data_steps)

        @info "Replaying CSV data..."
        replay_start = time()
        last_report_time = replay_start
        last_report_sim = 0.0
        sys = sam.sys_struct
        set_v3_body_damping!(sys, BODY_DAMPING, POINT_37_38_DAMPING)
        distribute_wing_mass!(sys, 11.0; dist=0.5)
        distribute_wing_drag!(sys,
            sys.wings[1].vsm_aero.projected_area,
            EXTRA_WING_DRAG_COEFF)

        heading_pid = create_heading_pid(;
            K=HEADING_KP, Ti=HEADING_TI, dt)
        lateral_pid = create_heading_pid(;
            K=LATERAL_KP, dt)

        # Log full CSV reference independently of sim
        for step in 1:n_data_steps-1
            row = get_row(data, step)
            update_sys_struct_from_data!(
                data_sam.sys_struct, row)
            SymbolicAWEModels.reinit!(
                data_sam, data_sam.prob, FBDF())
            update_sys_state!(data_state, data_sam)
            data_state.winch_force[1] = row.tether_force
            data_state.v_app = row.v_app
            data_state.time = row.time
            data_state.l_tether[1] = row.tether_len
            data_state.v_reelout[1] = row.tether_vel
            data_state.var_01 = row.drag_coeff
            data_state.var_02 = row.lift_coeff
            data_state.var_09 = row.tether_drag_coeff
            data_state.var_10 = row.bridle_drag_coeff
            data_state.var_11 = row.kcu_drag_coeff
            data_state.var_05 = wrap_to_pi(row.yaw)
            data_state.var_06 = wrap_to_pi(row.pitch)
            data_state.var_07 = wrap_to_pi(row.roll)
            data_R_b_w = row.R_b_w
            data_state.var_08 = compute_bridle_pitch_angle(
                data_sam.sys_struct, data_R_b_w)
            data_state.v_wind_gnd .= row.wind_vec_ekf
            data_state.v_wind_200m .= row.wind_vec_lidar
            data_state.AoA = row.kite_aoa
            data_state.var_04 = row.kite_aoa
            data_state.var_12 = row.wing_aoa
            data_state.set_steering = row.steering
            data_state.depower = row.depower
            data_state.var_14 = row.video_frame
            log!(data_logger, data_state)
        end

        sim_cum_dist = 0.0
        prev_sim_pos = copy(sam.sys_struct.wings[1].pos_w)

        step = 0
        sim_time = 0.0
        while true
            step += 1

            # Termination conditions
            if DISTANCE_BASED_STEERING
                sim_cum_dist >= total_data_dist && break
            else
                step > n_data_steps - 1 && break
            end

            # Get data row (clamp to data range)
            data_step = min(step, n_data_steps)
            row = get_row(data, data_step)

            if step == 1
                sim_time = row.time
                prev_sim_pos = copy(
                    sam.sys_struct.wings[1].pos_w)
            end

            if DISTANCE_BASED_STEERING
                dist_raw = get_row_at_distance(
                    dist_lookup_data, sim_cum_dist)
                phys_row = make_row(dist_raw)
            else
                phys_row = row
            end

            data_pos_enu = [phys_row.x, phys_row.y,
                phys_row.z]
            data_heading = calc_csv_heading(
                phys_row.roll, phys_row.pitch,
                phys_row.yaw, data_pos_enu)
            sim_heading =
                sam.sys_struct.wings[1].heading
            heading_error = wrap_to_pi(
                data_heading - sim_heading)
            heading_correction = heading_pid(
                heading_error, 0.0, 0.0)

            # Lateral position feedback
            data_pos = [phys_row.x, phys_row.y,
                phys_row.z]
            sim_pos = sam.sys_struct.wings[1].pos_w
            body_y_world =
                sam.sys_struct.wings[1].R_b_to_w[:, 2]
            lateral_error = dot(
                sim_pos - data_pos, body_y_world)
            lateral_correction = lateral_pid(
                lateral_error, 0.0, 0.0)

            eff_steer, eff_dep =
                update_vel_from_csv!(
                    sam.sys_struct, phys_row,
                    settle_config.geom;
                    heading_correction=
                    heading_correction +
                    lateral_correction +
                    STEERING_OFFSET / 100)

            SymbolicAWEModels.reinit!(
                sam, sam.prob, FBDF())

            next_step!(sam; dt)
            if !isapprox(sam.set.wind_vec,
                row.wind_vec; atol=1e-6)
                @warn "wind_vec mismatch" step row.wind_vec sam.set.wind_vec
                error("wind_vec mismatch at step $step")
            end

            cur_sim_pos = sam.sys_struct.wings[1].pos_w
            sim_cum_dist += norm(
                cur_sim_pos - prev_sim_pos)
            prev_sim_pos = copy(cur_sim_pos)
            sim_time += dt

            if step % n_substeps == 0
                sys = sam.sys_struct
                for i in (4, 5)
                    f = sys.points[i].aero_force_b[2]
                    if f < 0.0
                        @warn "Aero y-force negative" point = i force = round(f, digits=2)
                    end
                end
                for i in (18, 19)
                    f = sys.points[i].aero_force_b[2]
                    if f > 0.0
                        @warn "Aero y-force positive" point = i force = round(f, digits=2)
                    end
                end
            end

            log_state!(logger, sys_state, sam, sim_time;
                set_steering=eff_steer, depower=eff_dep,
                video_frame=row.video_frame,
                wind_vec_ekf=row.wind_vec_ekf,
                wind_vec_lidar=row.wind_vec_lidar)

            if DISTANCE_BASED_STEERING
                pct = sim_cum_dist / total_data_dist
                report = pct >= 1.0 ||
                         floor(Int, pct * 10) >
                         floor(Int,
                    (pct - dt / total_data_dist)
                    *
                    10)
            else
                report = should_report(
                    step, n_data_steps)
            end
            if report
                sim_t = round(sim_time, digits=2)
                wall_t = round(
                    time() - replay_start, digits=1)
                now_t = time()
                dt_wall = now_t - last_report_time
                dt_sim = sim_time - last_report_sim
                rt = dt_wall > 0 ?
                     round(dt_sim / dt_wall,
                    digits=2) : 0.0
                last_report_time = now_t
                last_report_sim = sim_time
                d = round(norm(sim_pos - data_pos),
                    digits=2)
                dist_pct = DISTANCE_BASED_STEERING ?
                           round(sim_cum_dist /
                                 total_data_dist * 100,
                    digits=1) : 0.0
                msg = "Step $step" *
                      " (t=$(sim_t)s," *
                      " wall=$(wall_t)s," *
                      " $(rt)x realtime"
                if DISTANCE_BASED_STEERING
                    msg *= ", dist=$(dist_pct)%"
                end
                msg *= ", pos_err=$(d)m" *
                       ", frame=$(row.video_frame))"
                @info msg
            end
        end

    catch err
        is_interrupt = err isa InterruptException ||
                       any(e isa InterruptException
                           for (e, _) in current_exceptions())
        if is_interrupt
            @warn "Interrupted, stopping sim"
        elseif err isa ErrorException
            @warn "Sim aborted, keeping partial log" msg = err.msg
        else
            rethrow(err)
        end
    end

    elapsed = round(time() - replay_start, digits=2)
    @info "Replay done" elapsed

    base_name = build_replay_name(h5_path, start_utc,
        end_utc, row1.depower, row1.steering,
        settle_config.geom)
    mkpath(REPLAY_LOG_DIR)
    sim_log_name = base_name * "_sim"
    data_log_name = base_name * "_data"
    if !isnothing(logger) && logger.index > 1
        save_log(logger, sim_log_name; path=REPLAY_LOG_DIR)
        syslog = load_log(sim_log_name; path=REPLAY_LOG_DIR)
    else
        syslog = nothing
    end
    if !isnothing(data_logger) && data_logger.index > 1
        save_log(data_logger, data_log_name; path=REPLAY_LOG_DIR)
        datalog = load_log(data_log_name; path=REPLAY_LOG_DIR)
    else
        datalog = nothing
    end

    return sam, syslog, data_sam, datalog, data,
    settle_config, settle_log, dt, base_name
end

# =============================================================================
# Plot helpers
# =============================================================================

"""
    syslog_to_tape(syslog, geom) -> NamedTuple

Build a tape payload from a syslog with both command-space
signals (`steering`, `depower`) and corresponding tape lengths
(`steering_left_m`, `steering_right_m`, `depower_m`) derived
with the current geometry reductions.
"""
function syslog_to_tape(syslog,
    geom::V3GeomAdjustConfig)
    sl = hasproperty(syslog, :syslog) ? syslog.syslog :
         syslog
    steering = collect(sl.set_steering)
    depower = collect(sl.depower)
    steering_reduction = geom.reduce_steering ?
                         geom.steering_reduction : 0.0
    depower_reduction = geom.reduce_depower ?
                        geom.depower_reduction : 0.0
    l0_steering = V3_STEERING_L0_BASE -
                  steering_reduction
    l0_depower = V3_DEPOWER_L0_BASE -
                 depower_reduction
    steering_left_m = similar(steering)
    steering_right_m = similar(steering)
    depower_m = similar(depower)
    @inbounds for i in eachindex(steering)
        L_left, L_right = steering_percentage_to_lengths(
            steering[i] * 100.0;
            l0_base=l0_steering)
        steering_left_m[i] = L_left
        steering_right_m[i] = L_right
        depower_m[i] = depower_percentage_to_length(
            depower[i] * 100.0;
            l0_base=l0_depower)
    end
    return (time=collect(sl.time),
        steering=steering,
        depower=depower,
        steering_left_m=steering_left_m,
        steering_right_m=steering_right_m,
        depower_m=depower_m)
end

function create_replay_plots(;
    sys_struct, data_sys_struct,
    syslog, datalog,
    frame_csvs,
    geom::V3GeomAdjustConfig,
    section, distance_based_steering,
    depower_offset_pct,
    figures_dir, save_figs)
    dt = syslog.syslog.time[2] - syslog.syslog.time[1]
    frame_syslog_idxs = find_frame_syslog_idxs(
        syslog, frame_csvs)

    sim_tape = syslog_to_tape(syslog, geom)
    data_tape = syslog_to_tape(datalog, geom)
    _logs = [syslog, datalog]
    _tapes = [sim_tape, data_tape]
    _labels = ["simulation", "data"]

    fig = plot_replay(
        [sys_struct, data_sys_struct],
        _logs;
        tape_lengths=_tapes,
        suffixes=_labels,
        size=(1200, 800), labelsize=18)

    trajectory_kwargs = (;
        gradient=:vel, tapes=_tapes, labels=_labels,
        size=(560, 420), labelsize=20,
        frame_indexes=frame_syslog_idxs)
    panels_kwargs = (;
        tapes=_tapes, labels=_labels,
        show_aoa=false, labelsize=20,
        twin_time_axes=distance_based_steering,
        frame_indexes=frame_syslog_idxs,
        show_heading=false,
        show_course=true,
        show_drag_coeff=false,
        show_lift_coeff=false,
        show_lift_drag_ratio=false)
    yaw_heading_kwargs = (;
        source=:heading,
        min_steering=0.05,
        labels=_labels,
        figsize=(600, 400), labelsize=18, dt)
    yaw_course_kwargs = (;
        source=:course,
        min_steering=0.05,
        labels=_labels,
        figsize=(600, 400), labelsize=18, dt)

    # GLMakie display
    trajectory = plot_2d_trajectory(_logs; trajectory_kwargs...)
    panels = plot_2d_panels(_logs; panels_kwargs...)
    yaw_fig_heading = plot_yaw_rate_vs_steering(
        _logs; yaw_heading_kwargs...)
    yaw_fig_course = plot_yaw_rate_vs_steering(
        _logs; yaw_course_kwargs...)

    # CairoMakie PDF saves
    sr = geom.reduce_steering ? geom.steering_reduction : 0.0
    tr = geom.reduce_tip ? geom.tip_reduction : 0.0
    dist_suffix = distance_based_steering ?
                  "_dist" : ""
    config_suffix = "_dpoff_$(depower_offset_pct)" *
                    "_sr_$(sr)_tr_$(tr)"
    suffix = "_$(section)" * config_suffix
    CairoMakie.activate!()
    traj_2d = plot_2d_trajectory(_logs; trajectory_kwargs...)
    panels_2d = plot_2d_panels(_logs; panels_kwargs...)
    yaw_heading_2d = plot_yaw_rate_vs_steering(
        _logs; yaw_heading_kwargs...)
    yaw_course_2d = plot_yaw_rate_vs_steering(
        _logs; yaw_course_kwargs...)
    if save_figs
        mkpath(figures_dir)
        function save_with_dist(name, figure)
            fname = "$(name)$(suffix).pdf"
            @info "Saving $fname"
            save(joinpath(figures_dir, fname), figure)
            fig_path = joinpath(figures_dir, replace(
                fname, ".pdf" => "$(dist_suffix).pdf"))
            @info "Saving $fig_path"
            save(fig_path, figure)
        end
        save_with_dist("trajectory_2d", traj_2d)
        save_with_dist("panels_2d", panels_2d)
        save_with_dist("yaw_rate_heading", yaw_heading_2d)
        save_with_dist("yaw_rate_course", yaw_course_2d)
    end
    GLMakie.activate!()

    # 2D body frame plots for PDF export
    body = Dict{Int,Dict{Symbol,Any}}()
    twist = Dict{Int,Any}()
    CairoMakie.activate!()
    frame_annotations = ["right turn", "straight flight"]
    for (fi, (csv, target_frame)) in
        enumerate(frame_csvs)
        idx = findfirst(
            x -> x[1] == target_frame,
            frame_syslog_idxs)
        isnothing(idx) && continue
        _, syslog_idx = frame_syslog_idxs[idx]
        update_from_sysstate!(sys_struct,
            syslog.syslog[syslog_idx])
        pts, groups = load_extra_points(
            csv, sys_struct)
        ann = get(frame_annotations, fi, "")
        frame_figs = Dict{Symbol,Any}()
        for dir in (:front, :side, :top)
            show_leg = dir == :side &&
                       target_frame == 7362
            no_adjust = !geom.reduce_steering &&
                        !geom.reduce_tip
            if dir == :front && no_adjust
                show_leg = true
            end
            leg_pos = dir == :front ? :top : :right
            dir_ann = if dir == :front
                geom.reduce_tip ?
                "bridle reduced $(tr)m" :
                "bridle unreduced"
            else
                ann
            end
            bf = plot_body_frame_local(sys_struct;
                extra_points=pts,
                extra_groups=groups, dir,
                title=false, legend=show_leg,
                legend_position=leg_pos,
                show_incidence=false,
                show_kcu=false,
                show_camera=false,
                annotation=dir_ann)
            fname = "body_frame_$(dir)" *
                    "_$(section)" *
                    "_frame_$(target_frame)" *
                    "$(config_suffix).pdf"
            if save_figs
                @info "Saving $fname"
                save(joinpath(figures_dir, fname), bf)
                fig_fname = replace(fname,
                    ".pdf" => "$(dist_suffix).pdf")
                save(joinpath(figures_dir, fig_fname), bf)
            end
            frame_figs[dir] = bf
        end
        # Twist distribution
        twist_fig = plot_twist_dist(sys_struct;
            extra_points=pts,
            extra_groups=groups,
            figsize=(560 * 0.8, 210 * 0.8),
            labelsize=24,
            title=false, legend=false,
            limits=(-3, 14),
            annotation=ann)
        twist_fname = "twist_dist" *
                      "_$(section)" *
                      "_frame_$(target_frame)" *
                      "$(config_suffix).pdf"
        if save_figs
            @info "Saving $twist_fname"
            save(joinpath(figures_dir, twist_fname), twist_fig)
            fig_twist = replace(twist_fname,
                ".pdf" => "$(dist_suffix).pdf")
            save(joinpath(figures_dir, fig_twist),
                twist_fig)
        end
        twist[target_frame] = twist_fig
        frame_figs[:twist] = twist_fig

        body[target_frame] = frame_figs
        @info "Saved 2D body frame + twist" target_frame
    end
    GLMakie.activate!()

    # Average gk for |steering| > 5%
    mean_gk = Dict{Tuple{String,Symbol},Float64}()
    for (label, lg) in [("sim", syslog),
        ("data", datalog)]
        sl = lg.syslog
        for source in (:heading, :course)
            rate = calc_turn_rate(lg; source, dt)
            us = sl.set_steering[2:end]
            va = sl.v_app[2:end]
            mask = abs.(us) .> 0.05
            gk_vals = rate[mask] ./ (va[mask] .* us[mask])
            mean_gk[(label, source)] = mean(gk_vals)
        end
    end
    for source in (:heading, :course)
        sim_gk = mean_gk[("sim", source)]
        data_gk = mean_gk[("data", source)]
        pct = (sim_gk - data_gk) / data_gk * 100
        @info "Mean gk" source sim = round(sim_gk; digits=3) data = round(data_gk; digits=3) pct = round(pct; digits=1)
    end

    n_steer = min(length(sim_tape.steering),
        length(data_tape.steering))
    steer_diff = sim_tape.steering[1:n_steer] .-
                 data_tape.steering[1:n_steer]
    @info "Mean steering input diff (sim - data)" mean = round(
        mean(steer_diff); digits=4) mean_abs = round(
        mean(abs.(steer_diff)); digits=4)

    hdot_fig = plot_turn_rate_vs_time(_logs;
        labels=["sim", "data"], dt)
    display(hdot_fig)

    wind_fig = plot_wind_compare(datalog)
    display(wind_fig)

    return (; fig, trajectory, panels, traj_2d, panels_2d,
        yaw_fig_heading, yaw_fig_course,
        body, twist, hdot_fig, wind_fig)
end

# =============================================================================
# Main execution
# =============================================================================

geom_config = V3GeomAdjustConfig(
    reduce_tip=REDUCE_TIP, reduce_te=true,
    reduce_depower=false,
    reduce_steering=REDUCE_STEERING,
    steering_reduction=STEERING_REDUCTION,
    tip_reduction=TIP_REDUCTION,
    depower_offset=depower_offset_pct / 100.0)

if LOAD_FROM_DISK
    full_data = load_flight_data(h5_path)
    limited_data, _ = limit_by_utc(
        full_data, start_utc, end_utc)
    is_2019 = occursin("2019", basename(h5_path))
    raw_dp = limited_data.kcu_actual_depower[1]
    row1_depower = is_2019 ?
                   (0.2564 - 0.0768 * raw_dp / 100.0) :
                   raw_dp / 100.0
    row1_steering = limited_data.kcu_actual_steering[1] /
                    100.0
    base_name = build_replay_name(h5_path, start_utc,
        end_utc, row1_depower, row1_steering, geom_config)
    sim_log_name = base_name * "_sim"
    data_log_name = base_name * "_data"
    syslog = load_log(sim_log_name; path=REPLAY_LOG_DIR)
    datalog = load_log(data_log_name; path=REPLAY_LOG_DIR)

    data_path = v3_data_path()
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.g_earth = 9.81
    set.profile_law = 0
    source_struc = joinpath(data_path,
        DEFAULT_STRUC_GEOM_YAML)
    source_aero = joinpath(data_path,
        DEFAULT_AERO_GEOM_YAML)
    vsm_path = joinpath(data_path, DEFAULT_VSM_SETTINGS_PATH)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = source_aero
    sam, _ = build_replay_sys_struct(
        set, geom_config, source_struc, vsm_set)
    data_sam, _ = build_replay_sys_struct(
        set, geom_config, source_struc, vsm_set)
else
    sam, syslog, data_sam, datalog, data,
    settle_config, settle_log, dt,
    base_name = run_physics_replay(h5_path)
end

if !isnothing(syslog)
    plots = create_replay_plots(;
        sys_struct=sam.sys_struct,
        data_sys_struct=data_sam.sys_struct,
        syslog, datalog,
        frame_csvs, geom=geom_config,
        section=SECTION,
        distance_based_steering=DISTANCE_BASED_STEERING,
        depower_offset_pct=depower_offset_pct,
        figures_dir=FIGURES_DIR, save_figs=SAVE_FIGS)
end

sim_log_name = base_name * "_sim"
data_log_name = base_name * "_data"
@info "Replay log_name (sim)" log_name = joinpath(
    "processed_data", sim_log_name)
@info "Replay log_name (data)" log_name = joinpath(
    "processed_data", data_log_name)
