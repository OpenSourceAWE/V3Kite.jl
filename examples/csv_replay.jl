# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
CSV Replay Example

Reads flight test data from CSV and replays steering inputs through the
SymbolicAWEModel simulator. The kite is initialized to steady state,
then CSV steering commands are applied during simulation.

Uses in-package functions for steering/depower conversion, CSV parsing,
heading calculation, and coordinate utilities.

Usage:
    julia --project=examples examples/csv_replay.jl
"""

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX,
    V3_DEPOWER_IDX, V3_STEERING_GAIN
using SymbolicAWEModels: reposition!, rotate_around_z,
    rotate_around_y, calc_steady_torque, FBDF
using GLMakie
using CairoMakie
GLMakie.activate!()
using CSV, DataFrames, DiscretePIDs
using Statistics
using Rotations
using UnPack
using LinearAlgebra
using OrdinaryDiffEqBDF
using KiteUtils: calc_elevation, azimuth_east
using Dates
using VortexStepMethod
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

SECTION = "straight_right"
CSV_PATH = joinpath(v3_data_path(), "v3_2025-10-09-ekf.csv")
WARN_STEP = false

# Geometry
DEPOWER_PERCENTAGE = 39.37
TETHER_LENGTH = 248
TE_FRAC = 0.95
TIP_REDUCTION = 0.4
GEOM_SUFFIX = build_geom_suffix(V3_DEPOWER_L0, TIP_REDUCTION, TE_FRAC)

STRUC_YAML_PATH = joinpath(
    v3_data_path(), "struc_geometry_$(GEOM_SUFFIX).yaml")
AERO_YAML_PATH = joinpath(
    v3_data_path(), "aero_geometry_$(GEOM_SUFFIX).yaml")

# Video frame mapping
VIDEO_FRAME_REF = 7182
UTC_REF_SECONDS = 15*3600 + 36*60 + 31.0
VIDEO_FPS = 29.97

# Maneuver selection
if SECTION == "straight_right"
    START_UTC = "15:36:31.0"
    END_UTC = "15:36:41.1"
    EXTRA_POINTS_CSV = nothing
    EXTRA_POINTS_FRAME = nothing
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
RESTABLE = false
STOP_EARLY = false
MIN_DAMPING = [0.0, 60, 120]

# Tracking forces
CSV_SPRING_K = 0.1
CSV_DAMPING_K = 0.01

# PID parameters
HEADING_KP = 0.5
HEADING_TAU_I = false

# =============================================================================
# CSV replay helper functions
# =============================================================================

"""Feed-forward torque from tether force."""
function calc_feedforward_torque(tether_force_n, winch)
    return -winch.drum_radius / winch.gear_ratio *
        tether_force_n + winch.friction
end

# Heading spike filter state
const PREV_CSV_HEADING = Ref{Float64}(NaN)
const MAX_HEADING_CHANGE = deg2rad(20.0)

function filter_csv_heading(heading)
    if isnan(PREV_CSV_HEADING[])
        PREV_CSV_HEADING[] = heading
        return heading
    end
    delta = wrap_to_pi(heading - PREV_CSV_HEADING[])
    if abs(delta) > MAX_HEADING_CHANGE
        return PREV_CSV_HEADING[]
    end
    PREV_CSV_HEADING[] = heading
    return heading
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

"""Apply spring+damping forces to points 1:38."""
function apply_csv_tracking_forces!(sim_sys, csv_sys;
        k_spring=CSV_SPRING_K, k_damping=CSV_DAMPING_K)
    for i in 1:38
        sim_pt = sim_sys.points[i]
        csv_pt = csv_sys.points[i]
        pos_err = csv_pt.pos_w - sim_pt.pos_w
        vel_err = csv_pt.vel_w - sim_pt.vel_w
        sim_pt.disturb .= k_spring * pos_err +
            k_damping * vel_err
    end
end

function update_vel_from_csv!(sys, row, brake, heading_pid)
    @unpack wings, points, winches, segments = sys
    wing = wings[1]

    raw_csv_heading = calc_csv_heading(
        row.roll, row.pitch, row.yaw, sys)
    csv_heading = filter_csv_heading(raw_csv_heading)
    wing.R_b_w = calc_R_b_w(sys)
    curr_heading = calc_heading(sys, wing.R_b_w)
    delta_heading = -wrap_to_pi(csv_heading - curr_heading)

    update_sim_distance!(wing.pos_w)
    sys.set.v_wind = row.wind_speed
    sys.set.upwind_dir = row.wind_dir + -90

    # PID steering + CSV steering
    steering_ctrl = DiscretePIDs.calculate_control!(
        heading_pid, 0.0, delta_heading, 0.0)
    steering = clamp(row.steering, -100.0, 100.0)
    L_left, L_right = csv_steering_percentage_to_lengths(
        steering * STEERING_MULTIPLIER)
    segments[V3_STEERING_LEFT_IDX].l0 =
        L_left + V3_STEERING_GAIN * steering_ctrl
    segments[V3_STEERING_RIGHT_IDX].l0 =
        L_right - V3_STEERING_GAIN * steering_ctrl

    # Winch feed-forward
    winch = winches[1]
    ff_torque = calc_feedforward_torque(
        row.tether_force, winch)
    winch.brake = brake
    winch.set_value = ff_torque

    # Depower from CSV
    L_depower = depower_percentage_to_length(row.depower)
    segments[V3_DEPOWER_IDX].l0 = L_depower

    eff_steering = steering * STEERING_MULTIPLIER +
        steering_ctrl * 100
    return winch.set_value, eff_steering, row.depower
end

# =============================================================================
# Main replay function
# =============================================================================

function run_physics_replay(csv_path;
        start_utc=START_UTC, end_utc=END_UTC,
        n_substeps=5)

    df = load_flight_data(csv_path)
    limited_data, _ = limit_by_utc(df, start_utc, end_utc)
    limited_data = add_distance_column(limited_data)

    @info "Interpolating CSV data" n_substeps
    csv_data = interpolate_csv_data(limited_data, n_substeps)

    @info "Loading V3 kite" STRUC_YAML_PATH AERO_YAML_PATH
    set_data_path(v3_data_path())
    set = Settings("system.yaml")
    set.g_earth = 9.81
    set.l_tether = TETHER_LENGTH
    vsm_path = joinpath(v3_data_path(),
        "vsm_settings_reduced_for_coupling.yaml")
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = AERO_YAML_PATH
    sys_struct = load_sys_struct_from_yaml(STRUC_YAML_PATH;
        system_name="v3", set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)
    sam = SymbolicAWEModel(set, sys_struct)
    init!(sam)

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

    PREV_CSV_HEADING[] = NaN
    reset_distance_tracker!()

    heading_pid = DiscretePID(;
        K=HEADING_KP, Ti=HEADING_TAU_I, Td=false,
        Ts=dt, umin=-1.0, umax=1.0)

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
            cumulative_distance = data.cumulative_distance[step],
            wind_speed = data.ekf_wind_speed_horizontal[step],
            wind_dir = data.ekf_wind_direction[step],
            angle_of_attack = deg2rad(
                data.ekf_wing_angle_of_attack[step]),
            v_app = data.ekf_kite_apparent_windspeed[step],
        )
    end

    try
        for step in 1:n_steps-1
            row = get_row(csv_data, step)

            # Update CSV reference
            update_sys_struct_from_csv!(
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
                update_sys_struct_from_csv!(
                    sam.sys_struct, row)
                SymbolicAWEModels.reinit!(
                    sam, sam.prob, FBDF())
            end

            set_value, eff_steer, eff_dep =
                update_vel_from_csv!(
                    sam.sys_struct, row, true, heading_pid)

            sam.sys_struct.winches[1].tether_len =
                row.tether_len + tether_delta
            sam.sys_struct.winches[1].tether_vel =
                row.tether_vel
            SymbolicAWEModels.reinit!(
                sam, sam.prob, FBDF())

            # Tracking forces
            total_mass = sam.sys_struct.total_mass
            apply_csv_tracking_forces!(
                sam.sys_struct, csv_sam.sys_struct;
                k_spring=CSV_SPRING_K * total_mass,
                k_damping=CSV_DAMPING_K * total_mass)

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
                @info "Step $step/$n_steps (t=$(round(row.time, digits=2))s, frame=$(row.video_frame))"
            end
        end
    catch err
        if err isa AssertionError
            @warn "Simulation stopped (AssertionError)"
        else
            rethrow(err)
        end
    end

    @info "Replay done" elapsed=round(time() - replay_start, digits=2)

    save_log(logger, "csv_replay")
    save_log(csv_logger, "csv_reference")
    syslog = load_log("csv_replay")
    csvlog = load_log("csv_reference")

    if RESTABLE
        SymbolicAWEModels.update_yaml_from_sys_struct!(
            sam.sys_struct, STRUC_YAML_PATH, STRUC_YAML_PATH,
            AERO_YAML_PATH, AERO_YAML_PATH)
    end

    return sam, syslog, csv_sam, csvlog, csv_data,
        phys_tape, csv_tape, extra_pts, extra_groups
end

# =============================================================================
# Main execution
# =============================================================================

sam, syslog, csv_sam, csvlog, csv_data, phys_tape, csv_tape,
    extra_pts, extra_groups = run_physics_replay(CSV_PATH)

fig = plot([sam.sys_struct, csv_sam.sys_struct],
    [syslog, csvlog];
    plot_tether=true, plot_aero_force=false,
    plot_kite_vel=true, plot_wind=false,
    plot_reelout=false, plot_v_app=true,
    plot_turn_rates=true,
    tape_lengths=[phys_tape, csv_tape],
    suffixes=["phys", "csv"])

sphere = plot_sphere_trajectory([syslog, csvlog])

nothing
