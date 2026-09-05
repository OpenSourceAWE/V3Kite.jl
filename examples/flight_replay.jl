# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Flight Data Replay

Reads flight test data from HDF5 and replays the recorded steering inputs
through the SymbolicAWEModel simulator. The kite is settled onto the first data
row, then the commands are applied. A second model is stepped through the
recorded state alone, giving the reference the simulation is judged against.

Both runs are saved as logs. Plotting them is `flight_replay_plots.jl`:

    include("examples/flight_replay.jl")
    include("examples/flight_replay_plots.jl")

Which maneuver, which corrections and which wing come from the project file's
`replay_settings:` and `kite_settings:` keys, so nothing here needs editing to
replay a different section.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using HDF5
using VortexStepMethod
using V3Kite: reinit_integrator!
using LinearAlgebra
using LazyArtifacts
using OrdinaryDiffEqBDF
using KiteUtils

generate_drag_adjusted_polars(1.0)

PROJECT = select_project(
    ["particle lattice" => "system_psm_replay.yaml",
     "Timoshenko-beam wing" => "system_beam_replay.yaml"];
    prompt = "Which wing model should the flight data be replayed on?")

set_data_path(v3_data_path())
kite_set = load_kite(PROJECT)
replay = load_replay(PROJECT)
SECTION, start_utc, end_utc = replay_maneuver(replay)

datadir = joinpath(artifact"flight_data", "flight_data")
h5_path = joinpath(datadir, replay.year == 2025 ?
    "ekf_awe_2025-10-09.h5" : "ekf_awe_2019-10-08.h5")
replay.year == 2019 && (kite_set.geom.depower_offset = replay.depower_offset_2019)

@info "Flight replay" PROJECT SECTION start_utc end_utc

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
        steering * replay.steering_multiplier +
            heading_correction, gc;
        min_l0=0.01)

    # Speed-controlled winch
    winch = sys.winches[1]
    winch.brake = true
    winch.vel = row.tether_vel
    sys.tethers[1].len = row.tether_len
    sys.tethers[1].stretched_len = row.tether_len

    # Depower from CSV (returns adjusted dp)
    eff_depower = set_depower!(
        sys, row.depower, row.steering, gc)

    eff_steering = steering * replay.steering_multiplier +
        heading_correction
    return eff_steering, eff_depower
end

# =============================================================================
# Main replay function
# =============================================================================

"""
    has_samples(logger)

True when `logger` exists and holds at least one logged state, which is what
decides whether an aborted replay leaves a log worth saving.
"""
has_samples(logger) = !isnothing(logger) && logger.index > 1

function run_physics_replay(h5_path;
        start_utc=start_utc, end_utc=end_utc,
        n_substeps=replay.n_substeps, vsm_interval=kite_set.vsm_interval)

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
            speed_source=replay.wind_source_speed,
            dir_source=replay.wind_source_dir)
        wh = sqrt(wind_vec[1]^2 + wind_vec[2]^2)
        wv = wind_vec[3]
        wdir = atan(wind_vec[2], wind_vec[1])
        kite_vel = [raw.ekf_kite_velocity_x,
                    raw.ekf_kite_velocity_y,
                    raw.ekf_kite_velocity_z]
        kite_aoa = compute_kite_aoa(
            R_b_w, kite_vel, wind_vec)
        wing_aoa = kite_aoa + deg2rad(
            replay.aoa_offset_slope * dp * 100 +
            replay.aoa_offset_intercept)
        kite_pos = [raw.ekf_kite_position_x,
                    raw.ekf_kite_position_y,
                    raw.ekf_kite_position_z]
        return (
            time = raw.time,
            video_frame = round(Int, raw.video_frame),
            roll = raw.ekf_kite_roll,
            pitch = raw.ekf_kite_pitch,
            yaw = raw.ekf_kite_yaw, # TODO: try using kin
            heading = calc_csv_heading(
                raw.ekf_kite_roll, raw.ekf_kite_pitch,
                raw.ekf_kite_yaw, kite_pos),
            x = kite_pos[1],
            y = kite_pos[2],
            z = kite_pos[3],
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
            wind_speed = wh,
            upwind_dir = wrap_to_pi(
                -wdir - π/2),
            wind_speed_vertical = wv,
            R_b_w = R_b_w,
            v_app = raw.ekf_kite_apparent_windspeed,
            omega_b = [raw.ekf_roll_rate, raw.ekf_pitch_rate,
                       raw.ekf_yaw_rate],
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
            wind_vec_ekf = wind_vec_ekf,
            wind_vec_lidar = wind_vec_lidar,
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
    settle_config = load_settle(PROJECT; kite_set)
    settle_config.v_wind = row1.v_app
    settle_config.tether_length = tether_len
    settle_config.start_depower = row1.depower * 100.0 + 10.0
    settle_log = nothing
    sam = nothing
    data_sam = nothing
    logger = nothing
    data_logger = nothing
    replay_start = time()
    dt = data.time[2] - data.time[1]

    try

    data_path = v3_data_path()
    source_struc = struc_geometry_path(PROJECT; data_path)
    source_aero = aero_geometry_path(PROJECT; data_path,
        aero_mode = resolve_aero_mode(kite_set))
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_settings_path(PROJECT; data_path); data_prefix=false)
    vsm_set.wings[1].geometry_file = source_aero

    if replay.settle
        sam, settle_log, settle_failed =
            settle_wing(settle_config, row1;
                remake_model=kite_set.remake_model,
                remake_settled_state=kite_set.remake_settled_state)
        if settle_failed
            @warn "Settling failed — skipping sim"
            return sam, nothing, nothing, nothing, data,
                settle_config, settle_log, dt
        end
    else
        set_data_path(data_path)
        set = Settings(PROJECT)
        set.g_earth = 9.81
        set.v_wind = row1.v_app
        set.l_tether = tether_len
        set.profile_law = 0

        gc = kite_set.geom
        sys = load_sys_struct_from_yaml(source_struc;
            system_name=V3_MODEL_NAME, set,
            dynamics_type=kite_set.wing_type, vsm_set,
            aero_mode=resolve_aero_mode(kite_set))
        sam = SymbolicAWEModel(set, sys; backend = kite_set.backend)
        apply_geom_adjustments!(sys, gc)
        V3Kite.with_model_cache(V3Kite.default_cache_path()) do
            SymbolicAWEModels.init!(sam;
                remake=false, ignore_l0=false,
                remake_vsm=true)
        end
        settle_log = nothing
    end
    set = sam.set
    set.l_tether = tether_len

    n_data_steps = length(data.time)
    max_sim_steps = replay.distance_based_steering ?
        n_data_steps * 3 : n_data_steps
    sys_state = SysState(sam)
    logger = Logger(sam, max_sim_steps)

    # CSV reference: same settled geometry as the sim so equal elev/azim give equal y/z.
    data_struct = load_settled_struct(
        settle_config, row1; set)
    data_sam = SymbolicAWEModel(set, data_struct; backend = kite_set.backend)
    data_sam.sys_struct.tethers[1].init_stretched_len = tether_len
    V3Kite.with_model_cache(V3Kite.default_cache_path()) do
        init!(data_sam; remake=false, remake_vsm=true,
            reinit_sys=false)
    end
    data_state = SysState(data_sam)
    data_logger = Logger(data_sam, n_data_steps)

    @info "Replaying CSV data..."
    replay_start = time()
    last_report_time = replay_start
    last_report_sim = 0.0
    sys = sam.sys_struct
    if kite_set.wing_mass > 0
        distribute_wing_mass!(sys, kite_set.wing_mass;
            dist=kite_set.wing_mass_le_frac)
    end
    if kite_set.wing_drag_coeff > 0
        distribute_wing_drag!(sys,
            sys.wings[1].vsm_aero.projected_area,
            kite_set.wing_drag_coeff)
    end

    heading_pid = create_heading_pid(;
        K=replay.heading_K, Ti=replay.heading_Ti, dt)
    lateral_pid = create_heading_pid(;
        K=replay.lateral_K, dt)

    # Log full CSV reference independently of sim. It is posed, never stepped:
    # reinit! is what syncs the imposed state through the symbolic getters.
    for step in 1:n_data_steps-1
        row = get_row(data, step)
        update_sys_struct_from_data!(
            data_sam.sys_struct, row)
        reinit_integrator!(data_sam; prn=false)
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
        data_state.steering = row.steering
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
        if replay.distance_based_steering
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

        if replay.distance_based_steering
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
                settle_config.kite_set.geom;
                heading_correction=
                    heading_correction +
                    lateral_correction +
                    replay.steering_offset/100)

        # Log pre-step state so sim point i aligns with data point i.
        log_state!(logger, sys_state, sam, sim_time;
            set_steering=eff_steer, depower=eff_dep,
            video_frame=row.video_frame,
            wind_vec_ekf=row.wind_vec_ekf,
            wind_vec_lidar=row.wind_vec_lidar)

        reinit_integrator!(sam; prn=false)

        next_step!(sam; dt, vsm_interval)
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

        if replay.distance_based_steering
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
            dist_pct = replay.distance_based_steering ?
                round(sim_cum_dist /
                    total_data_dist * 100,
                    digits=1) : 0.0
            msg = "Step $step" *
                " (t=$(sim_t)s," *
                " wall=$(wall_t)s," *
                " $(rt)x realtime"
            if replay.distance_based_steering
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
        elseif err isa ErrorException || err isa AssertionError
            @warn "Replay aborted, keeping whatever was logged" msg=err.msg
        else
            rethrow(err)
        end
    end

    elapsed = round(time() - replay_start, digits=2)
    @info "Replay done" elapsed

    sim_name, data_name = replay_log_names(h5_path, start_utc, end_utc,
                                           settle_config.kite_set.geom)
    has_samples(logger) && save_log(logger, sim_name)
    has_samples(data_logger) && save_log(data_logger, data_name)

    return sam, logger, data_sam, data_logger, data,
        settle_config, settle_log, dt
end

# =============================================================================
# Main execution
# =============================================================================

sam, logger, data_sam, data_logger, data,
    settle_config, settle_log, dt = run_physics_replay(h5_path)

sim_name, data_name = replay_log_names(h5_path, start_utc, end_utc, kite_set.geom)
if has_samples(logger) && has_samples(data_logger)
    @info "Logs saved" sim_name data_name
    @info "Plot them with: include(\"flight_replay_plots.jl\")"
else
    has_samples(logger) && @info "Sim log saved" sim_name
    has_samples(data_logger) && @info "Data log saved" data_name
    @warn "flight_replay_plots.jl needs both logs, so it cannot run on this " *
        "aborted replay"
end

nothing
