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
using VortexStepMethod
using SymbolicAWEModels: reposition!, rotate_around_x,
    rotate_around_z, rotate_around_y,
    calc_steady_torque, FBDF
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
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

generate_drag_adjusted_polars(1.0)

SECTION = "straight_right"
YEAR = 2025
SETTLE = true
DEPOWER_OFFSET_2019 = 7.0
DEPOWER_OFFSET_2025 = -9.0
STEERING_MULTIPLIER = 1.0
HEADING_KP = 0.1
HEADING_TI = 0.0
LATERAL_KP = 0.0
STEERING_OFFSET = 0.0
DISTANCE_BASED_STEERING = false
REDUCE_STEERING = true
STEERING_REDUCTION = 0.2
REDUCE_TIP = true
TIP_REDUCTION = 0.2
BODY_DAMPING = [0.0, 0.0, 20.0]
# Photogrammetry linear AoA offset model:
AOA_OFFSET_A = -0.6831
AOA_OFFSET_B = 28.74
POINT_37_38_DAMPING = [0.0, 20.0, 20.0]
SAVE_FIGS = true
FIGURES_DIR = joinpath(@__DIR__, "..", "..",
    "T26-BART", "figures")

# Maneuver selection
if YEAR == 2025
    h5_path = joinpath(v3_data_path(),
        "flight_data", "ekf_awe_2025-10-09.h5")
else
    h5_path = joinpath(v3_data_path(),
        "flight_data", "ekf_awe_2019-10-08.h5")
end
SECTION = "$(SECTION)_$(YEAR)"
if SECTION == "straight_2025"
    start_utc = "15:36:27.0"
    end_utc = "15:36:33.1"
elseif SECTION == "right_2025"
    start_utc = "15:36:34.0"
    end_utc = "15:36:38.0"
elseif SECTION == "straight_right_2025"
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
    frame_csvs = Tuple{String, Int}[]
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
    frame_csvs = Tuple{String, Int}[]
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
            # u_dp_2025 = 0.2564 - 0.0768 * u_p_2019
            dp = 0.2564 - 0.0768 * dp / 100.0
            dp += DEPOWER_OFFSET_2019 / 100.0
        else
            dp = dp / 100.0
        end
        quat2R =
            SymbolicAWEModels.quaternion_to_rotation_matrix
        R_b_w = quat2R(euler_to_quaternion(
            raw.ekf_kite_roll, raw.ekf_kite_pitch,
            raw.ekf_kite_yaw))
        wdir = raw.ekf_wind_direction
        wh = raw.ekf_wind_speed_horizontal
        wv = raw.ekf_wind_speed_vertical
        wind_vec = [wh * cos(wdir),
                    wh * sin(wdir), wv]
        kite_vel = [raw.ekf_kite_velocity_x,
                    raw.ekf_kite_velocity_y,
                    raw.ekf_kite_velocity_z]
        kite_aoa = compute_kite_aoa(
            R_b_w, kite_vel, wind_vec)
        wing_aoa = kite_aoa + deg2rad(
            AOA_OFFSET_A * dp * 100 + AOA_OFFSET_B)
        return (
            time = raw.time,
            video_frame = round(Int, raw.video_frame),
            roll = raw.ekf_kite_roll,
            pitch = raw.ekf_kite_pitch,
            yaw = raw.ekf_kite_yaw, # TODO: try using kin
            x = raw.ekf_kite_position_x,
            y = raw.ekf_kite_position_y,
            z = raw.ekf_kite_position_z,
            vx = raw.ekf_kite_velocity_x,
            vy = raw.ekf_kite_velocity_y,
            vz = raw.ekf_kite_velocity_z,
            tether_len = raw.ekf_tether_length,
            tether_vel = raw.tether_reelout_speed,
            tether_force = raw.ground_tether_force,
            steering = raw.kcu_actual_steering / 100.0,
            depower = dp,
            distance = raw.distance,
            cumulative_distance = raw.cumulative_distance,
            wind_speed = raw.ekf_wind_speed_horizontal,
            upwind_dir = wrap_to_pi(
                -wdir - π/2),
            wind_speed_vertical =
                raw.ekf_wind_speed_vertical,
            R_b_w = R_b_w,
            v_app = raw.ekf_kite_apparent_windspeed,
            drag_coeff = raw.ekf_wing_drag_coefficient,
            lift_coeff = raw.ekf_wing_lift_coefficient,
            tether_drag_coeff =
                raw.ekf_tether_drag_coefficient,
            bridle_drag_coeff =
                raw.ekf_bridles_drag_coefficient,
            kcu_drag_coeff =
                raw.ekf_kcu_drag_coefficient,
            wind_elevation = atan(wv, wh),
            wind_vec = wind_vec,
            kite_aoa = kite_aoa,
            wing_aoa = wing_aoa,
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
        num_steps=1000,
        num_substeps=5,
        start_depower=row1.depower * 100.0 + 10.0,
        geom=V3GeomAdjustConfig(
            reduce_tip=REDUCE_TIP, reduce_te=true,
            reduce_depower=false,
            reduce_steering=REDUCE_STEERING,
            steering_reduction=STEERING_REDUCTION,
            tip_reduction=TIP_REDUCTION,
            depower_offset=DEPOWER_OFFSET_2025 / 100.0),
        fix_sphere_idxs=[])
    settle_log = nothing
    sam = nothing
    data_sam = nothing
    logger = nothing
    data_logger = nothing
    sim_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])
    data_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])
    frame_syslog_idxs = Tuple{Int, Int}[]
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
        sam, settle_log = settle_wing(settle_config;
            init_row=row1, power_zone=true, remake=false)
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
    sys_struct = sam.sys_struct
    set = sam.set
    set.l_tether = tether_len

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
    SymbolicAWEModels.set_body_frame_damping(
        sys, BODY_DAMPING, 1:38)
    SymbolicAWEModels.set_body_frame_damping(
        sys, POINT_37_38_DAMPING, 37:38)
    distribute_wing_mass!(sys, 11.0; dist=0.5)

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
        data_state.v_wind_gnd .= row.wind_vec
        data_state.AoA = row.kite_aoa
        data_state.var_04 = row.kite_aoa
        data_state.var_12 = row.wing_aoa
        log!(data_logger, data_state)

        push!(data_tape.time, row.time)
        push!(data_tape.steering, row.steering)
        push!(data_tape.depower, row.depower)
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
                    STEERING_OFFSET/100)

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
                    @warn "Aero y-force negative" point=i force=round(f, digits=2)
                end
            end
            for i in (18, 19)
                f = sys.points[i].aero_force_b[2]
                if f > 0.0
                    @warn "Aero y-force positive" point=i force=round(f, digits=2)
                end
            end
        end

        # Record syslog index for matched frames
        if step <= n_data_steps - 1
            for (_, frame) in frame_csvs
                if row.video_frame == frame &&
                        !any(x -> x[1] == frame,
                            frame_syslog_idxs)
                    push!(frame_syslog_idxs,
                        (frame, logger.index))
                end
            end
        end

        log_state!(logger, sys_state, sam,
            sim_time)

        push!(sim_tape.time, sim_time)
        push!(sim_tape.steering, eff_steer)
        push!(sim_tape.depower, eff_dep)

        if DISTANCE_BASED_STEERING
            pct = sim_cum_dist / total_data_dist
            report = pct >= 1.0 ||
                floor(Int, pct * 10) >
                floor(Int,
                    (pct - dt / total_data_dist)
                    * 10)
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
        if err isa ErrorException &&
                contains(err.msg, "Unstable")
            @warn "Solver unstable" msg=err.msg
        elseif is_interrupt
            @warn "Interrupted, stopping sim"
        else
            rethrow(err)
        end
    end

    elapsed = round(time() - replay_start, digits=2)
    @info "Replay done" elapsed

    base_name = build_replay_name(h5_path, start_utc,
        end_utc, row1.depower, row1.steering,
        settle_config.geom)
    if !isnothing(logger) && logger.index > 1
        save_log(logger, base_name * "_sim")
        syslog = load_log(base_name * "_sim")
    else
        syslog = nothing
    end
    if !isnothing(data_logger) && data_logger.index > 1
        save_log(data_logger, base_name * "_data")
        datalog = load_log(base_name * "_data")
    else
        datalog = nothing
    end

    return sam, syslog, data_sam, datalog, data,
        sim_tape, data_tape, frame_syslog_idxs,
        settle_config, settle_log, dt, base_name
end

# =============================================================================
# Main execution
# =============================================================================

sam, syslog, data_sam, datalog, data, sim_tape, data_tape,
    frame_syslog_idxs,
    settle_config, settle_log, dt,
    base_name = run_physics_replay(h5_path)
geom_config = settle_config.geom

function create_plots()
    global fig, trajectory, panels, traj_2d, panels_2d
    global yaw_fig, body, twist, hdot_fig
    fig = plot_replay(
        [sam.sys_struct, data_sam.sys_struct],
        [syslog, datalog];
        tape_lengths=[sim_tape, data_tape],
        suffixes=["simulation", "data"],
        size=(1200, 800), labelsize=18)

    _logs = [syslog, datalog]
    _tapes = [sim_tape, data_tape]
    _labels = ["simulation", "data"]

    # GLMakie display
    trajectory = plot_2d_trajectory(_logs;
        gradient=:vel, tapes=_tapes, labels=_labels,
        size=(560, 420), labelsize=20,
        frame_indexes=frame_syslog_idxs,
)
    panels = plot_2d_panels(_logs;
        tapes=_tapes, labels=_labels,
        show_aoa=true, labelsize=20,
        twin_time_axes=DISTANCE_BASED_STEERING,
        frame_indexes=frame_syslog_idxs,
        show_heading=true,
        show_drag_coeff=false,
        show_lift_coeff=false,
        show_lift_drag_ratio=true)

    # CairoMakie PDF saves
    sr = REDUCE_STEERING ? STEERING_REDUCTION : 0.0
    tr = REDUCE_TIP ? TIP_REDUCTION : 0.0
    dist_suffix = DISTANCE_BASED_STEERING ?
        "_dist" : ""
    config_suffix = "_dpoff_$(DEPOWER_OFFSET_2025)" *
        "_sr_$(sr)_tr_$(tr)" *
        "_ecd_$(WING_EXTRA_DRAG_COEFF)"
    suffix = "_$(SECTION)" * config_suffix
    CairoMakie.activate!()
    traj_2d = plot_2d_trajectory(_logs;
        gradient=:vel, tapes=_tapes, labels=_labels,
        size=(560, 420), labelsize=20,
        frame_indexes=frame_syslog_idxs,
)
    panels_2d = plot_2d_panels(_logs;
        tapes=_tapes, labels=_labels,
        show_aoa=true, labelsize=20,
        twin_time_axes=DISTANCE_BASED_STEERING,
        frame_indexes=frame_syslog_idxs,
        show_heading=true,
        show_drag_coeff=false,
        show_lift_coeff=false,
        show_lift_drag_ratio=true)
    if SAVE_FIGS
        mkpath(FIGURES_DIR)
        traj_fname = "trajectory_2d$(suffix).pdf"
        panels_fname = "panels_2d$(suffix).pdf"
        @info "Saving $traj_fname"
        save(traj_fname, traj_2d)
        fig_traj = joinpath(FIGURES_DIR, replace(
            traj_fname,
            ".pdf" => "$(dist_suffix).pdf"))
        @info "Saving $fig_traj"
        save(fig_traj, traj_2d)
        @info "Saving $panels_fname"
        save(panels_fname, panels_2d)
        fig_panels = joinpath(FIGURES_DIR, replace(
            panels_fname,
            ".pdf" => "$(dist_suffix).pdf"))
        @info "Saving $fig_panels"
        save(fig_panels, panels_2d)
    end
    GLMakie.activate!()

    yaw_fig = plot_yaw_rate_vs_steering(
        [syslog, datalog],
        [sim_tape, data_tape];
        min_steering=0.05,
        labels=["simulation", "data"],
        figsize=(600, 400), labelsize=18, dt)

    # 2D body frame plots for PDF export
    body = Dict{Int, Dict{Symbol, Any}}()
    twist = Dict{Int, Any}()
    CairoMakie.activate!()
    frame_annotations = ["right turn", "straight flight"]
    for (fi, (csv, target_frame)) in
            enumerate(frame_csvs)
        idx = findfirst(
            x -> x[1] == target_frame,
            frame_syslog_idxs)
        isnothing(idx) && continue
        _, syslog_idx = frame_syslog_idxs[idx]
        update_from_sysstate!(
            sam.sys_struct,
            syslog.syslog[syslog_idx])
        pts, groups = load_extra_points(
            csv, sam.sys_struct)
        ann = get(frame_annotations, fi, "")
        frame_figs = Dict{Symbol, Any}()
        sr = REDUCE_STEERING ? STEERING_REDUCTION : 0.0
        tr = REDUCE_TIP ? TIP_REDUCTION : 0.0
        for dir in (:front, :side, :top)
            show_leg = dir == :side &&
                target_frame == 7362
            no_adjust = !REDUCE_STEERING &&
                !REDUCE_TIP
            if dir == :front && no_adjust
                show_leg = true
            end
            leg_pos = dir == :front ? :top : :right
            is_side = dir == :side
            dir_ann = if dir == :front
                REDUCE_TIP ?
                    "bridle reduced $(tr)m" :
                    "bridle unreduced"
            else
                ann
            end
            bf = plot_body_frame_local(
                sam.sys_struct;
                extra_points=pts,
                extra_groups=groups, dir,
                title=false, legend=show_leg,
                legend_position=leg_pos,
                show_incidence=false,
                show_kcu=false,
                show_camera=false,
                annotation=dir_ann)
            fname = "body_frame_$(dir)" *
                "_$(SECTION)" *
                "_frame_$(target_frame)" *
                "$(config_suffix).pdf"
            if SAVE_FIGS
                @info "Saving $fname"
                save(fname, bf)
                fig_fname = replace(fname,
                    ".pdf" => "$(dist_suffix).pdf")
                save(joinpath(FIGURES_DIR, fig_fname), bf)
            end
            frame_figs[dir] = bf
        end
        # Twist distribution
        twist_fig = plot_twist_dist(
            sam.sys_struct;
            extra_points=pts,
            extra_groups=groups,
            figsize=(560*0.8, 210*0.8),
            labelsize=24,
            title=false, legend=false,
            limits=(-3, 14),
            annotation=ann)
        twist_fname = "twist_dist" *
            "_$(SECTION)" *
            "_frame_$(target_frame)" *
            "$(config_suffix).pdf"
        if SAVE_FIGS
            @info "Saving $twist_fname"
            save(twist_fname, twist_fig)
            fig_twist = replace(twist_fname,
                ".pdf" => "$(dist_suffix).pdf")
            save(joinpath(FIGURES_DIR, fig_twist),
                twist_fig)
        end
        twist[target_frame] = twist_fig
        frame_figs[:twist] = twist_fig

        body[target_frame] = frame_figs
        @info "Saved 2D body frame + twist" target_frame
    end
    GLMakie.activate!()

    # Average gk for |steering| > 5%
    mean_gk = Dict{String,Float64}()
    hdot_series = Dict{String,Vector{Float64}}()
    hdot_time = Dict{String,Vector{Float64}}()
    for (label, lg, tape) in [("sim", syslog, sim_tape),
                               ("data", datalog, data_tape)]
        sl = lg.syslog
        hw = copy(sl.heading)
        for j in 2:length(hw)
            while hw[j] - hw[j-1] > pi; hw[j] -= 2pi; end
            while hw[j] - hw[j-1] < -pi; hw[j] += 2pi; end
        end
        hdot = diff(hw) ./ dt
        # Subtract frame transport: ψ̇_m = ψ̇_T − φ̇·sin(β)
        az = copy(sl.azimuth)
        for j in 2:length(az)
            while az[j] - az[j-1] > pi; az[j] -= 2pi; end
            while az[j] - az[j-1] < -pi; az[j] += 2pi; end
        end
        az_dot = diff(az) ./ dt
        hdot .= hdot .- az_dot .* sin.(sl.elevation[2:end])
        hdot_series[label] = hdot
        hdot_time[label] = tape.time[2:end]
        us = tape.steering[2:end]
        va = sl.v_app[2:end]
        mask = abs.(us) .> 0.05
        gk_vals = hdot[mask] ./ (va[mask] .* us[mask])
        mean_gk[label] = mean(gk_vals)
        @info "Mean gk ($label)" gk=round(
            mean_gk[label]; digits=3)
    end
    pct = (mean_gk["sim"] - mean_gk["data"]) /
        mean_gk["data"] * 100
    @info "gk difference" pct=round(pct; digits=1)

    # Validation plot: heading rate vs time
    hdot_fig = Figure(size=(900, 400))
    ax = Axis(hdot_fig[1, 1];
        xlabel="time [s]",
        ylabel="heading rate [rad/s]",
        title="Body yaw rate (frame-transport corrected)")
    lines!(ax, hdot_time["sim"], hdot_series["sim"];
        label="sim", color=:blue)
    lines!(ax, hdot_time["data"], hdot_series["data"];
        label="data", color=:red)
    axislegend(ax)
    display(hdot_fig)

    panels
end
if !isnothing(syslog)
    create_plots()
end
