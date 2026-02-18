# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
CSV Replay with Equilibrium Turn Rate Analysis

Uses ProtoLogger CSV format (old flight data) with NonlinearSolve to
find the turn rate that zeroes VSM yaw moment at each timestep.

Note: This script uses different calibration constants from the
standard V3Kite values (STEERING_GAIN=1.2 instead of 1.4, and
DEPOWER_OFFSET=-15.0) to match the old ProtoLogger CSV format.

Usage:
    julia --project=examples examples/csv_replay_equilibrium.jl
"""

using V3Kite
using SymbolicAWEModels: reposition!, rotate_around_z,
    rotate_around_y, calc_steady_torque, FBDF
using GLMakie
using CSV, DataFrames
using Statistics
using Rotations
using UnPack
using LinearAlgebra
using OrdinaryDiffEqBDF
using KiteUtils: calc_elevation, azimuth_east
using NonlinearSolve, ADTypes

# =============================================================================
# Configuration
# =============================================================================

CSV_PATH = joinpath(v3_data_path(),
    "2025-10-09_16-58-33_ProtoLogger_lidar.csv")
START_FRAME = 22068 + 120
END_FRAME = START_FRAME + 50
WING_CD = 0.0

# Script-local calibration (differs from V3Kite defaults!)
STEERING_L0 = 1.6
STEERING_GAIN_LOCAL = 1.2  # Different from V3_STEERING_GAIN=1.4
STEERING_MULTIPLIER = 1.0
DEPOWER_L0_LOCAL = 0.2
DEPOWER_GAIN_LOCAL = 5.0
DEPOWER_OFFSET = -15.0

INITIAL_DAMPING = [0.0, 300.0, 600.0]
DECAY_TIME = 1.0
MIN_DAMPING = [0.0, 30, 60]

# =============================================================================
# Script-local calibration functions (ProtoLogger format)
# =============================================================================

function local_steering_to_lengths(percentage)
    u_s = percentage / 100.0
    L_left = STEERING_L0 +
        STEERING_GAIN_LOCAL * STEERING_MULTIPLIER * u_s
    L_right = STEERING_L0 -
        STEERING_GAIN_LOCAL * STEERING_MULTIPLIER * u_s
    return L_left, L_right
end

function local_depower_to_length(percentage)
    u_p = percentage / 100.0
    return DEPOWER_L0_LOCAL + DEPOWER_GAIN_LOCAL * u_p
end

function local_steering_to_pct(L_right)
    u_s = (STEERING_L0 - L_right) /
        (STEERING_GAIN_LOCAL * STEERING_MULTIPLIER)
    return u_s * 100.0
end

function local_depower_to_pct(L_depower)
    u_p = (L_depower - DEPOWER_L0_LOCAL) / DEPOWER_GAIN_LOCAL
    return u_p * 100.0 - DEPOWER_OFFSET
end

# =============================================================================
# Helper functions
# =============================================================================

function calc_feedforward_torque(tether_force_n, winch)
    return -winch.drum_radius / winch.gear_ratio *
        tether_force_n + winch.friction
end

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

function update_vel_from_csv!(sys, row, brake)
    @unpack wings, points, winches, segments = sys
    wing = wings[1]
    sys.set.v_wind = row.wind_at_kite

    steering = clamp(row.steering, -100.0, 100.0)
    L_left, L_right = local_steering_to_lengths(steering)
    segments[V3_STEERING_LEFT_IDX].l0 = L_left
    segments[V3_STEERING_RIGHT_IDX].l0 = L_right

    winch = winches[1]
    ff_torque = calc_feedforward_torque(
        row.tether_force, winch)
    winch.brake = brake
    winch.set_value = ff_torque

    L_depower = local_depower_to_length(
        row.depower + DEPOWER_OFFSET)
    segments[V3_DEPOWER_IDX].l0 = L_depower

    return winch.set_value
end

function local_update_sys_from_csv!(sys, row)
    @unpack wings, points, winches, segments, transforms = sys
    wing = wings[1]
    transform = transforms[1]

    quat = euler_to_quaternion(row.roll, row.pitch, row.yaw)
    csv_heading = calc_heading(sys,
        SymbolicAWEModels.quaternion_to_rotation_matrix(quat)) + pi
    wing.R_b_w = calc_R_b_w(sys)

    csv_pos = [row.x, row.y, row.z]
    for (n, pidx) in enumerate(39:44)
        points[pidx].pos_b .= [
            0.0, 0.0, -n * row.tether_len / 6 * 1.01]
    end
    transform.elevation = calc_elevation(csv_pos)
    transform.azimuth = azimuth_east(csv_pos)
    transform.heading = csv_heading
    SymbolicAWEModels.reinit!([transform], sys)

    csv_vel = [row.vx, row.vy, row.vz]
    wing.vel_w .= csv_vel
    for point in points
        frac = point.pos_w ⋅ normalize(wing.pos_w) /
            norm(wing.pos_w)
        point.vel_w .= frac * csv_vel
    end

    winches[1].brake = true
    L_left, L_right = local_steering_to_lengths(row.steering)
    L_depower = local_depower_to_length(
        row.depower + DEPOWER_OFFSET)
    segments[V3_STEERING_LEFT_IDX].l0 = L_left
    segments[V3_STEERING_RIGHT_IDX].l0 = L_right
    segments[V3_DEPOWER_IDX].l0 = L_depower
end

function find_equilibrium_turn_rate(wing)
    body_aero = wing.vsm_aero
    solver = wing.vsm_solver
    va_b = wing.va_b
    v_a = norm(va_b)
    v_a < 1e-6 && return NaN

    alpha = atan(va_b[3], va_b[1])
    beta = asin(clamp(va_b[2] / v_a, -1.0, 1.0))
    gamma_prev = copy(solver.sol.gamma_distribution)

    function residual!(F, turn_rate, p)
        omega = [0.0, 0.0, turn_rate[1]]
        va_vec = [cos(alpha) * cos(beta), sin(beta),
            sin(alpha)] * v_a
        VortexStepMethod.set_va!(body_aero, va_vec, omega)
        VortexStepMethod.solve!(
            solver, body_aero, gamma_prev; log=false)
        F[1] = solver.sol.moment[3]
        return nothing
    end

    prob = NonlinearProblem(residual!, [wing.turn_rate[3]])
    sol = solve(prob,
        NewtonRaphson(; autodiff=AutoFiniteDiff());
        abstol=1e-8, reltol=1e-8)
    return sol.retcode == ReturnCode.Success ? sol.u[1] : NaN
end

# =============================================================================
# Replay function (ProtoLogger format)
# =============================================================================

function load_proto_data(csv_path)
    @info "Loading ProtoLogger CSV" csv_path
    df = CSV.read(csv_path, DataFrame;
        delim=' ', silencewarnings=true,
        normalizenames=true, types=Float64, strict=false)
    t0 = df.time[START_FRAME]
    df.time .= df.time .- t0
    col_names = Tuple(Symbol(name) for name in names(df))
    return NamedTuple{col_names}(Tuple(eachcol(df)))
end

function limit_frames(data; start_frame=1, end_frame=nothing)
    n = length(data.time)
    s = max(1, min(start_frame, n))
    e = isnothing(end_frame) ? n : max(s, min(end_frame, n))
    return NamedTuple{keys(data)}(
        Tuple(f[s:e] for f in data))
end

function proto_add_distance(data)
    n = length(data.time)
    dists = zeros(n)
    cum = zeros(n)
    for i in 2:n
        dx = data.kite_pos_east[i] - data.kite_pos_east[i-1]
        dy = data.kite_pos_north[i] - data.kite_pos_north[i-1]
        dz = data.kite_height[i] - data.kite_height[i-1]
        dists[i] = sqrt(dx^2 + dy^2 + dz^2)
        cum[i] = cum[i-1] + dists[i]
    end
    return merge(data, (distance=dists,
        cumulative_distance=cum))
end

function run_physics_replay(csv_path;
        start_frame=START_FRAME, end_frame=END_FRAME,
        n_substeps=5)

    raw = load_proto_data(csv_path)
    limited = limit_frames(raw; start_frame, end_frame)
    limited = proto_add_distance(limited)
    csv_data = interpolate_csv_data(limited, n_substeps)

    set_data_path(v3_data_path())
    set = Settings("system.yaml")
    set.g_earth = 9.81
    set.l_tether = 212.68
    vsm_path = joinpath(v3_data_path(),
        "vsm_settings_reduced_for_coupling.yaml")
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = joinpath(
        v3_data_path(), "aero_geometry_stable.yaml")
    struc = joinpath(v3_data_path(),
        "struc_geometry_stable.yaml")
    sys_struct = load_sys_struct_from_yaml(struc;
        system_name=V3_MODEL_NAME, set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)

    # Set wing CD
    n_wp = count(p -> p.type == WING, sys_struct.points)
    wa = sys_struct.wings[1].vsm_aero.projected_area / n_wp
    for p in sys_struct.points
        if p.type == WING
            p.area = wa
            p.drag_coeff = WING_CD
        end
    end

    sam = SymbolicAWEModel(set, sys_struct)
    init!(sam)

    csv_sam = SymbolicAWEModel(set, deepcopy(sam.sys_struct))
    init!(csv_sam)

    n_steps = length(csv_data.time)
    sys_state = SysState(sam)
    logger = Logger(sam, n_steps)
    csv_state = SysState(csv_sam)
    csv_logger = Logger(csv_sam, n_steps)

    csv_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])
    phys_tape = (time=Float64[], steering=Float64[],
        depower=Float64[])

    @info "Replaying..." n_steps
    replay_start = time()
    sys = sam.sys_struct
    SymbolicAWEModels.set_body_frame_damping(
        sys, INITIAL_DAMPING, 1:38)
    dt = csv_data.time[2] - csv_data.time[1]
    tether_delta = set.l_tether -
        csv_data.ground_tether_length[1]

    PREV_CSV_HEADING[] = NaN
    reset_distance_tracker!()

    function get_row(data, step)
        return (
            time = data.time[step],
            roll = data.kite_0_roll[step],
            pitch = data.kite_0_pitch[step],
            yaw = data.kite_0_yaw[step],
            x = data.kite_pos_east[step],
            y = data.kite_pos_north[step],
            z = data.kite_height[step],
            vx = data.kite_est_vx[step],
            vy = data.kite_est_vy[step],
            vz = data.kite_est_vz[step],
            tether_len = data.ground_tether_length[step],
            tether_vel = data.ground_tether_reelout_speed[step],
            tether_force = data.ground_tether_force[step] * 9.81,
            steering = data.kite_actual_steering[step],
            depower = data.kite_actual_depower[step],
            distance = data.distance[step],
            cumulative_distance = data.cumulative_distance[step],
            wind_at_kite = coalesce(
                data.lidar_wind_velocity_at_kite_mps[step], 10.0),
            angle_of_attack = deg2rad(coalesce(
                data.airspeed_angle_of_attack[step], 0.0)),
        )
    end

    try
        for step in 1:n_steps-1
            row = get_row(csv_data, step)

            local_update_sys_from_csv!(csv_sam.sys_struct, row)
            SymbolicAWEModels.reinit!(
                csv_sam, csv_sam.prob, FBDF())
            update_sys_state!(csv_state, csv_sam)
            csv_state.winch_force[1] = row.tether_force
            csv_state.AoA = row.angle_of_attack
            csv_state.time = row.time
            csv_state.l_tether[1] = row.tether_len
            csv_state.v_reelout[1] = row.tether_vel
            csv_state.v_wind_gnd[1] = row.wind_at_kite
            log!(csv_logger, csv_state)

            push!(csv_tape.time, row.time)
            push!(csv_tape.steering, row.steering)
            push!(csv_tape.depower, row.depower)

            if step == 1
                local_update_sys_from_csv!(
                    sam.sys_struct, row)
                SymbolicAWEModels.reinit!(
                    sam, sam.prob, FBDF())
            end

            t = row.time
            if t <= DECAY_TIME
                bd = (INITIAL_DAMPING - MIN_DAMPING) *
                    (1.0 - t / DECAY_TIME) + MIN_DAMPING
                SymbolicAWEModels.set_body_frame_damping(
                    sam.sys_struct, bd, 1:38)
            end

            set_value = update_vel_from_csv!(
                sam.sys_struct, row, true)
            sam.sys_struct.winches[1].tether_len =
                row.tether_len + tether_delta
            sam.sys_struct.winches[1].tether_vel =
                row.tether_vel
            SymbolicAWEModels.reinit!(
                sam, sam.prob, FBDF())

            next_step!(sam; dt, set_values=[set_value])

            update_sys_state!(sys_state, sam)
            sys_state.time = t
            log!(logger, sys_state)

            push!(phys_tape.time, t)
            push!(phys_tape.steering, local_steering_to_pct(
                sam.sys_struct.segments[V3_STEERING_RIGHT_IDX].l0))
            push!(phys_tape.depower, local_depower_to_pct(
                sam.sys_struct.segments[V3_DEPOWER_IDX].l0))

            if should_report(step, n_steps)
                @info "Step $step/$n_steps (t=$(round(t, digits=2))s)"
            end
        end
    catch err
        err isa AssertionError || rethrow(err)
    end

    @info "Replay done" elapsed=round(time() - replay_start, digits=2)
    save_log(logger, "csv_replay_eq")
    save_log(csv_logger, "csv_reference_eq")
    syslog = load_log("csv_replay_eq")
    csvlog = load_log("csv_reference_eq")

    return sam, syslog, csv_sam, csvlog, csv_data, raw,
        phys_tape, csv_tape
end

# =============================================================================
# Main execution
# =============================================================================

sam, syslog, csv_sam, csvlog, csv_data, raw,
    phys_tape, csv_tape = run_physics_replay(CSV_PATH)

# Equilibrium analysis
@info "Finding equilibrium turn rate..."
wing = sam.sys_struct.wings[1]

dt_log = syslog.syslog.time[end] - syslog.syslog.time[end-1]
coupled_tr = (syslog.syslog.heading[end] -
    syslog.syslog.heading[end-1]) / dt_log
csv_tr = (csvlog.syslog.heading[end] -
    csvlog.syslog.heading[end-1]) / dt_log
eq_tr = find_equilibrium_turn_rate(wing)

println("\n", "="^60)
println("TURN RATE COMPARISON")
println("="^60)
println("Coupled:     $(round(rad2deg(coupled_tr), digits=4)) deg/s")
println("CSV:         $(round(rad2deg(csv_tr), digits=4)) deg/s")
println("Equilibrium: $(round(rad2deg(eq_tr), digits=4)) deg/s")
println("Current M_z: $(round(wing.vsm_solver.sol.moment[3], digits=4)) Nm")
println("="^60)

fig = plot([sam.sys_struct, csv_sam.sys_struct],
    [syslog, csvlog];
    plot_tether=true, plot_aero_force=false,
    plot_kite_vel=true, plot_wind=true,
    plot_reelout=false, plot_v_app=false,
    plot_turn_rates=true,
    tape_lengths=[phys_tape, csv_tape],
    suffixes=["phys", "csv"])
display(fig)

sphere = plot_sphere_trajectory([syslog, csvlog])

nothing
