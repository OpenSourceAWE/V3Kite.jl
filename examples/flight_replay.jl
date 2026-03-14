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

using Dates
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

generate_drag_adjusted_polars(1.0)

SECTION = "straight_right"
YEAR = 2025
PLOT_FRAME = true
SETTLE_ONLY = false
SETTLE = true
CONST_WIND = false
DEPOWER_OFFSET_2019 = 7.0
DEPOWER_OFFSET_2025 = -10.0
WING_SYMMETRIC_FORCE = [0.0, 0.0, 0.0]
STEERING_MULTIPLIER = 1.0
HEADING_KP = 0.0
HEADING_TI = 0.0
LATERAL_KP = 0.0
STEERING_OFFSET = 0.0
DISTANCE_BASED_STEERING = false
BODY_DAMPING = [0.0, 0.0, 20.0]
# Photogrammetry chord-bridle AoA offset model:
#   offset = -atand((L_dp(u) - L_ref) / c)
CHORD_C = 2.266      # chord length at strut (m)
CHORD_L_REF = 2.34  # reference depower tape length (m)
POINT_37_38_DAMPING = [0.0, 100.0, 0.0]
FIGURES_DIR = joinpath(@__DIR__, "..", "..",
    "Torque2026", "figures")

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
    start_utc = "15:36:32.0"
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
if PLOT_FRAME && YEAR == 2025
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
    if !CONST_WIND
        sys.set.v_wind = hypot(
            row.wind_speed, row.wind_speed_vertical)
        sys.set.upwind_dir = rad2deg(row.upwind_dir)
        sys.wind_elevation = atan(
            row.wind_speed_vertical, row.wind_speed)
    end

    # CSV steering (positive = right turn)
    steering = clamp(row.steering, -1.0, 1.0)
    set_steering!(sys,
        steering * STEERING_MULTIPLIER +
            heading_correction, gc;
        min_l0=0.01)

    # Speed-controlled winch
    winch = sys.winches[1]
    winch.speed_controlled = true
    winch.tether_vel = row.tether_vel
    winch.tether_len = row.tether_len

    # Depower from CSV
    set_depower!(sys, row.depower, gc)

    eff_steering = steering * STEERING_MULTIPLIER +
        heading_correction
    return eff_steering, row.depower
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
            dp += DEPOWER_OFFSET_2025 / 100.0
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
            steering = raw.kcu_actual_steering / 100.0,
            depower = dp,
            distance = raw.distance,
            cumulative_distance = raw.cumulative_distance,
            wind_speed = raw.ekf_wind_speed_horizontal,
            upwind_dir = wrap_to_pi(-raw.ekf_wind_direction - π/2),
            wind_speed_vertical =
                raw.ekf_wind_speed_vertical,
            angle_of_attack = deg2rad(
                raw.ekf_wing_angle_of_attack_bridle -
                atand((depower_percentage_to_length(
                    raw.kcu_actual_depower) - CHORD_L_REF)
                    / CHORD_C)),
            v_app = raw.ekf_kite_apparent_windspeed,
            drag_coeff = raw.ekf_wing_drag_coefficient,
            lift_coeff = raw.ekf_wing_lift_coefficient,
            bridle_aoa = deg2rad(
                raw.ekf_wing_angle_of_attack_bridle),
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
        start_depower=row1.depower * 100.0 + 10.0,
        geom=V3GeomAdjustConfig(
            reduce_tip=true, reduce_te=true,
            reduce_depower=false, reduce_steering=true, steering_reduction=0.2,
            tip_reduction=0.2),
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
            nothing, nothing, Tuple{Int, Int}[],
            settle_config, settle_log, nothing, ""
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

    frame_syslog_idxs = Tuple{Int, Int}[]

    @info "Replaying CSV data..."
    replay_start = time()
    sys = sam.sys_struct
    SymbolicAWEModels.set_body_frame_damping(
        sys, BODY_DAMPING, 1:38)
    SymbolicAWEModels.set_body_frame_damping(
        sys, POINT_37_38_DAMPING, 37:38)
    distribute_wing_mass!(sys, 11.0; dist=0.5)

    dt = data.time[2] - data.time[1]
    heading_pid = create_heading_pid(;
        K=HEADING_KP, Ti=HEADING_TI, dt)
    lateral_pid = create_heading_pid(;
        K=LATERAL_KP, dt)

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
        data_state.time = row.time
        data_state.l_tether[1] = row.tether_len
        data_state.v_reelout[1] = row.tether_vel
        data_state.var_01 = row.drag_coeff
        data_state.var_02 = row.lift_coeff
        data_state.var_04 = row.bridle_aoa
        wind_elev = atan(
            row.wind_speed_vertical, row.wind_speed)
        wind_vec = rotate_around_z(
            rotate_around_x(
                [0.0, -1.0, 0.0], wind_elev),
            -row.upwind_dir)
        v_wind_total = hypot(
            row.wind_speed, row.wind_speed_vertical)
        data_state.v_wind_gnd .= v_wind_total .* wind_vec
        log!(data_logger, data_state)

        push!(data_tape.time, row.time)
        push!(data_tape.steering, row.steering)
        push!(data_tape.depower, row.depower)
    end

    sim_cum_dist = 0.0
    prev_sim_pos = copy(sam.sys_struct.wings[1].pos_w)

    try
        for step in 1:n_steps-1
            row = get_row(data, step)

            if step == 1
                update_sys_struct_from_data!(
                    sam.sys_struct, row)
                SymbolicAWEModels.reinit!(
                    sam, sam.prob, FBDF())
                prev_sim_pos = copy(
                    sam.sys_struct.wings[1].pos_w)
            end

            phys_row = row
            if DISTANCE_BASED_STEERING
                dist_raw = get_row_at_distance(
                    dist_lookup_data, sim_cum_dist)
                dist_row = make_row(dist_raw)
                phys_row = merge(phys_row,
                    (steering=dist_row.steering,
                     depower=dist_row.depower))
            end

            data_heading = calc_csv_heading(
                phys_row.roll, phys_row.pitch,
                phys_row.yaw)
            sim_heading =
                sam.sys_struct.wings[1].heading
            heading_error = wrap_to_pi(
                data_heading - sim_heading)
            heading_correction = heading_pid(
                heading_error, 0.0, 0.0)

            # Lateral position feedback
            data_pos = [phys_row.x, phys_row.y, phys_row.z]
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
                    heading_correction=heading_correction +
                        lateral_correction + STEERING_OFFSET/100)

            R = sam.sys_struct.wings[1].R_b_to_w
            f_w = R * WING_SYMMETRIC_FORCE
            for i in 2:11
                sam.sys_struct.points[i].disturb .= f_w
            end
            for i in 12:21
                sam.sys_struct.points[i].disturb .= -f_w
            end

            SymbolicAWEModels.reinit!(
                sam, sam.prob, FBDF())

            next_step!(sam; dt)

            cur_sim_pos = sam.sys_struct.wings[1].pos_w
            sim_cum_dist += norm(
                cur_sim_pos - prev_sim_pos)
            prev_sim_pos = copy(cur_sim_pos)

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
            for (_, frame) in frame_csvs
                if row.video_frame == frame &&
                        !any(x -> x[1] == frame,
                            frame_syslog_idxs)
                    push!(frame_syslog_idxs,
                        (frame, logger.index))
                end
            end

            log_state!(logger, sys_state, sam, row.time)

            push!(sim_tape.time, row.time)
            push!(sim_tape.steering, eff_steer)
            push!(sim_tape.depower, eff_dep)

            if should_report(step, n_steps)
                sim_t = round(row.time, digits=2)
                wall_t = round(
                    time() - replay_start, digits=1)
                rt = round(row.time / wall_t, digits=2)
                d = round(norm(sim_pos - data_pos),
                    digits=2)
                @info "Step $step/$n_steps" *
                    " (t=$(sim_t)s," *
                    " wall=$(wall_t)s," *
                    " $(rt)x realtime," *
                    " frame=$(row.video_frame)," *
                    " dist=$(d)m)"
            end
        end
    catch err
        is_interrupt = err isa InterruptException ||
            any(e isa InterruptException
                for (e, _) in current_exceptions())
        if err isa ErrorException &&
                contains(err.msg, "Unstable")
            @warn "Solver unstable, stopping replay" msg=err.msg
        elseif is_interrupt
            @warn "Interrupted, stopping sim"
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

if !SETTLE_ONLY
    fig = plot_replay(
        [sam.sys_struct, data_sam.sys_struct],
        [syslog, datalog];
        tape_lengths=[sim_tape, data_tape],
        suffixes=["sim", "data"])

    trajectory = plot_2d_trajectory([syslog, datalog];
        gradient=:steering,
        tapes=[sim_tape, data_tape],
        labels=["sim", "data"],
        show_te_force=false,
        show_aoa=true,
    )
    CairoMakie.activate!()
    traj_2d = plot_2d_trajectory([syslog, datalog];
        gradient=:steering,
        tapes=[sim_tape, data_tape],
        labels=["sim", "data"],
        show_aoa=true)
    save("trajectory_2d_$(SECTION).pdf", traj_2d)
    mkpath(FIGURES_DIR)
    save(joinpath(FIGURES_DIR,
        "trajectory_2d_$(SECTION).pdf"), traj_2d)
    GLMakie.activate!()

    yaw_fig = plot_yaw_rate_vs_steering(
        [syslog, datalog],
        [sim_tape, data_tape];
        min_steering=0.05,
        labels=["sim", "data"], dt)

    # 2D body frame plots for PDF export
    body = Dict{Int, Dict{Symbol, Any}}()
    CairoMakie.activate!()
    for (csv, target_frame) in frame_csvs
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
        frame_figs = Dict{Symbol, Any}()
        for dir in (:front, :side, :top)
            bf = plot_body_frame_local(
                sam.sys_struct;
                extra_points=pts,
                extra_groups=groups, dir)
            fname = "body_frame_$(dir)" *
                "_$(SECTION)" *
                "_frame_$(target_frame).pdf"
            save(fname, bf)
            frame_figs[dir] = bf
        end
        body[target_frame] = frame_figs
        @info "Saved 2D body frame" target_frame
    end
    GLMakie.activate!()

    trajectory
end
