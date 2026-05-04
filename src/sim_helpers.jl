# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Simulation helper functions to reduce boilerplate in V3 kite examples.
"""

"""
    create_logger(sam, n_steps) -> (logger, sys_state)

Create a Logger and SysState, log the initial state at t=0.

# Returns
- `(logger, sys_state)`: Logger with n_steps+1 capacity and
  initialized SysState
"""
function create_logger(sam, n_steps)
    logger = Logger(sam, n_steps + 1)
    sys_state = SysState(sam)
    sys_state.time = 0.0
    log!(logger, sys_state)
    return logger, sys_state
end

"""
    ramp_factor(t, t_start, t_end) -> Float64

Linear ramp from 0 to 1 between `t_start` and `t_end`.
Returns 0.0 for `t <= t_start`, 1.0 for `t >= t_end`.
"""
function ramp_factor(t, t_start, t_end)
    t_end <= t_start && return 1.0
    return clamp((t - t_start) / (t_end - t_start), 0.0, 1.0)
end

"""
    init_winch_torque!(sys) -> initial_torque

Initialize winch torque from current tether force.
Sets `winch.set_value` and returns the computed torque.
"""
function init_winch_torque!(sys)
    winch = sys.winches[1]
    initial_force = norm(winch.force)
    torque = force_to_torque(initial_force, sys)
    winch.set_value = torque
    return torque
end

"""
    force_to_torque(force, sys) -> torque

Convert tether force [N] to winch torque [N·m].
Uses `τ = -r/G * F + friction` from the first winch.
"""
function force_to_torque(force, sys)
    winch = sys.winches[1]
    return -winch.drum_radius / winch.gear_ratio * force +
        winch.friction
end

"""
    sim_step!(sam; kwargs...) -> Bool

Call `next_step!` with AssertionError handling.
Returns `true` on success, `false` if an AssertionError occurs.
All keyword arguments are forwarded to `next_step!`.
"""
function sim_step!(sam; kwargs...)
    try
        next_step!(sam; kwargs...)
        return true
    catch err
        err isa AssertionError && return false
        rethrow(err)
    end
end

"""
    compute_drag_coeff(sam) -> Float64

Wing drag coefficient: VSM aero drag plus parasitic drag
from WING-type points. Together with tether, bridle, and
KCU drag coefficients, the four sum to the total CD.
"""
function compute_drag_coeff(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    va_b = wing.va_b
    v_app = norm(va_b)
    v_app < 1e-6 && return 0.0
    va_hat_b, q_ref = _drag_coeff_ref(wing)

    # VSM drag component (body frame → scalar)
    vsm_drag = dot(wing.aero_force_b, va_hat_b)

    # Parasitic drag from wing points (world frame)
    parasitic_w = zeros(3)
    for p in sys.points
        p.type == WING || continue
        parasitic_w .+= p.drag_force
    end
    R_b_w = calc_R_b_w(sys)
    va_hat_w = R_b_w * va_hat_b
    parasitic_drag = dot(parasitic_w, va_hat_w)

    return (vsm_drag + parasitic_drag) / q_ref
end

"""
    compute_lift_coeff(sam) -> Float64

Compute the lift coefficient from the first wing's aero force
component perpendicular to the apparent wind direction,
normalized by `q_inf * A_proj`. Uses `rho = 1.225 kg/m³`.
"""
function compute_lift_coeff(sam)
    wing = sam.sys_struct.wings[1]
    va_b = wing.va_b
    v_app = norm(va_b)
    v_app < 1e-6 && return 0.0
    va_hat = va_b / v_app
    lift_vec = wing.aero_force_b - dot(wing.aero_force_b, va_hat) * va_hat
    A_proj = calculate_projected_area(wing.vsm_wing)
    q_inf = 0.5 * _RHO_SL * v_app^2
    return norm(lift_vec) / (q_inf * A_proj)
end

const _RHO_SL = 1.225

"""
    _tether_point_idxs(sys) -> Set{Int}

Collect all point indices that belong to tether segments.
"""
function _tether_point_idxs(sys)
    pts = Set{Int}()
    for tether in sys.tethers
        for seg_idx in tether.segment_idxs
            seg = sys.segments[seg_idx]
            push!(pts, seg.point_idxs[1])
            push!(pts, seg.point_idxs[2])
        end
    end
    return pts
end

"""
    _drag_coeff_ref(wing) -> (va_hat_b, q_ref)

Common reference quantities for drag coefficient functions:
apparent-wind unit vector (body frame) and dynamic pressure
times projected area.
"""
function _drag_coeff_ref(wing)
    va_b = wing.va_b
    v_app = norm(va_b)
    va_hat_b = va_b / v_app
    A_proj = calculate_projected_area(wing.vsm_wing)
    q_ref = 0.5 * _RHO_SL * v_app^2 * A_proj
    return va_hat_b, q_ref
end

"""
    _point_drag_cd(sys, wing, idxs) -> Float64

Sum `point.drag_force` for points whose index is in `idxs`,
project onto kite apparent wind, normalize by `q * A_proj`.
"""
function _point_drag_cd(sys, wing, idxs)
    va_hat_b, q_ref = _drag_coeff_ref(wing)
    drag_w = zeros(3)
    for p in sys.points
        p.idx in idxs || continue
        drag_w .+= p.drag_force
    end
    R_b_w = calc_R_b_w(sys)
    va_hat_w = R_b_w * va_hat_b
    return dot(drag_w, va_hat_w) / q_ref
end

"""
    compute_tether_drag_coeff(sam) -> Float64

Tether drag coefficient from tether-point drag forces,
projected onto the kite apparent wind and normalized by
`q * A_proj`.
"""
function compute_tether_drag_coeff(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    norm(wing.va_b) < 1e-6 && return 0.0
    return _point_drag_cd(sys, wing,
        _tether_point_idxs(sys))
end

"""
    compute_bridle_drag_coeff(sam) -> Float64

Bridle drag coefficient: drag from non-tether, non-wing
points (excluding KCU / point 1), normalized by
`q * A_proj`.
"""
function compute_bridle_drag_coeff(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    norm(wing.va_b) < 1e-6 && return 0.0
    tether_pts = _tether_point_idxs(sys)
    bridle_idxs = Set{Int}()
    for p in sys.points
        p.idx in tether_pts && continue
        p.type == WING && continue
        p.idx == 1 && continue  # KCU
        push!(bridle_idxs, p.idx)
    end
    return _point_drag_cd(sys, wing, bridle_idxs)
end

"""
    compute_kcu_drag_coeff(sam) -> Float64

KCU drag coefficient from point 1's drag force,
normalized by `q * A_proj`.
"""
function compute_kcu_drag_coeff(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    norm(wing.va_b) < 1e-6 && return 0.0
    return _point_drag_cd(sys, wing, Set([1]))
end

"""
    mean_te_segment_force(sam) -> Float64

Mean force of trailing-edge-connected segments (indices
3, 5, 7, …, 21).
"""
function mean_te_segment_force(sam)
    return mean(sam.sys_struct.segments[i].force
                for i in 3:2:21)
end

"""
    chord_ref_mid(le_3, te_3, le_4, te_4; frac=0.3)

Chord reference point averaged over both strut sides.
Returns midpoint of `((1-frac)·LE + frac·TE)` for each
side. Default `frac=0.3` gives the 30%-chord point.
"""
function chord_ref_mid(le_3, te_3, le_4, te_4;
        frac=0.3)
    cr_3 = (1 - frac) .* le_3 .+ frac .* te_3
    cr_4 = (1 - frac) .* le_4 .+ frac .* te_4
    return (cr_3 .+ cr_4) ./ 2
end

"""
    chord_ref_mid(sys; frac=0.3)

Chord reference point from sys_struct points 10-13.
"""
function chord_ref_mid(sys; frac=0.3)
    return chord_ref_mid(
        sys.points[10].pos_w,
        sys.points[11].pos_w,
        sys.points[12].pos_w,
        sys.points[13].pos_w; frac)
end

"""
    compute_kite_aoa(R_b_w, kite_vel, wind_vec)

Kite angle of attack from body-frame apparent wind.
`R_b_w` rotates body→world, `kite_vel` and `wind_vec`
are in world frame.
"""
function compute_kite_aoa(R_b_w, kite_vel, wind_vec)
    v_app_b = R_b_w' * (kite_vel - wind_vec)
    return atan(v_app_b[3], v_app_b[1])
end

"""
    compute_kite_aoa(sys) -> Float64

Convenience wrapper extracting R_b_w and apparent wind
from the system structure.
"""
function compute_kite_aoa(sys)
    R_b_w = calc_R_b_w(sys)
    v_app_b = sys.wings[1].va_b
    return atan(v_app_b[3], v_app_b[1])
end

"""
    compute_wing_incidence(sys) -> Float64

AoA of the apparent wind relative to the mid-chord
direction. Uses the average chord (LE→TE of struts 3 & 4)
as x-axis with the body spanwise y-axis to form the
reference frame.
"""
function compute_wing_incidence(sys)
    le_3 = sys.points[10].pos_w
    te_3 = sys.points[11].pos_w
    le_4 = sys.points[12].pos_w
    te_4 = sys.points[13].pos_w
    R_b_w = calc_R_b_w(sys)
    y_body = R_b_w[:, 2]
    chord_w = normalize(
        normalize(te_3 - le_3) +
        normalize(te_4 - le_4))
    z_chord = normalize(cross(chord_w, y_body))
    wing = sys.wings[1]
    v_app_w = R_b_w * wing.va_b
    return atan(
        dot(v_app_w, z_chord),
        dot(v_app_w, chord_w))
end

"""
    compute_bridle_euler(sys) -> (yaw, pitch, roll)

Compute NED Euler angles (ZYX convention) of the bridle
reference frame. This is the inverse of `euler_to_quaternion`
which goes NED Euler → RotZYX → R_ned_to_enu → ENU rotation.
Here we go backwards: bridle R_br_w → R_enu_to_ned → extract
ZYX Euler angles.

The bridle z-axis points downward (wing → KCU) to match
the NED body-frame z-down convention used by the EKF.
"""
function compute_bridle_euler(sys)
    mid_w = chord_ref_mid(sys)
    z_br = normalize(sys.points[1].pos_w - mid_w)
    R_b_w = calc_R_b_w(sys)
    y_br = R_b_w[:, 2]
    x_br = normalize(cross(y_br, z_br))
    R_br_w = hcat(x_br, y_br, z_br)
    R_ned = [0 1 0; 1 0 0; 0 0 -1] * R_br_w
    pitch = asin(clamp(-R_ned[3, 1], -1, 1))
    yaw = atan(R_ned[2, 1], R_ned[1, 1])
    roll = atan(R_ned[3, 2], R_ned[3, 3])
    return (wrap_to_pi(yaw), pitch, wrap_to_pi(roll))
end

"""
    compute_bridle_pitch_angle(sys) -> Float64
    compute_bridle_pitch_angle(sys, R_b_w) -> Float64

Angle between tether direction (ground → KCU) and bridle
direction (KCU → 30%-chord midpoint), projected into
the body xz plane. Positive when the bridle pitches forward
(toward body +x) relative to the tether.

The two-argument form allows passing an explicit `R_b_w`
(e.g. built from EKF Euler angles) instead of deriving it
from the sys_struct geometry.
"""
function compute_bridle_pitch_angle(sys)
    return compute_bridle_pitch_angle(sys, calc_R_b_w(sys))
end

function compute_bridle_pitch_angle(sys, R_b_w)
    kcu_w = sys.points[1].pos_w
    qc_w = chord_ref_mid(sys)
    v_tether_w = normalize(kcu_w)
    v_bridle_w = normalize(qc_w - kcu_w)
    R_w_b = R_b_w'
    vt_b = R_w_b * v_tether_w
    vb_b = R_w_b * v_bridle_w
    return atan(
        vt_b[1] * vb_b[3] - vt_b[3] * vb_b[1],
        vt_b[1] * vb_b[1] + vt_b[3] * vb_b[3])
end

"""
    compute_cop_x(sam) -> Float64

Center-of-pressure x-coordinate in body frame, computed as
the force-magnitude-weighted average of `pos_b[1]` for wing
points 2:21.
"""
function compute_cop_x(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    R_w_b = calc_R_b_w(sys)'
    total_force = 0.0
    weighted_x = 0.0
    for i in 2:21
        p = sys.points[i]
        f = norm(p.aero_force_b)
        pos_b = R_w_b * (p.pos_w - wing.pos_w)
        weighted_x += pos_b[1] * f
        total_force += f
    end
    total_force < 1e-6 && return 0.0
    return weighted_x / total_force
end

"""
    log_state!(logger, sys_state, sam, t)

Update sys_state from the model, set time, and log.

Computed variables:
- `var_01`: total non-tether CD (VSM + parasitic)
- `var_02`: wing lift coefficient
- `var_03`: mean TE segment force
- `var_04`: kite AoA
- `var_05`-`var_07`: bridle NED Euler (yaw, pitch, roll)
- `var_08`: bridle pitch angle
- `var_09`-`var_11`: tether/bridle/KCU drag coefficients
  (from point drag forces)
- `var_12`: geometric wing incidence (photogrammetry-style)
- `var_13`: center-of-pressure x-coordinate (body frame)
"""
function log_state!(logger, sys_state, sam, t;
        set_steering=nothing, depower=nothing,
        video_frame=nothing,
        wind_vec_ekf=nothing,
        wind_vec_lidar=nothing)
    update_sys_state!(sys_state, sam)
    sys_state.var_01 = compute_drag_coeff(sam)
    sys_state.var_02 = compute_lift_coeff(sam)
    sys_state.var_03 = mean_te_segment_force(sam)
    sys_state.var_04 = compute_kite_aoa(sam.sys_struct)
    yaw, pitch, roll = compute_bridle_euler(sam.sys_struct)
    sys_state.var_05 = yaw
    sys_state.var_06 = pitch
    sys_state.var_07 = roll
    sys_state.var_08 = compute_bridle_pitch_angle(
        sam.sys_struct)
    sys_state.var_09 = compute_tether_drag_coeff(sam)
    sys_state.var_10 = compute_bridle_drag_coeff(sam)
    sys_state.var_11 = compute_kcu_drag_coeff(sam)
    sys_state.var_12 = compute_wing_incidence(
        sam.sys_struct)
    sys_state.var_13 = compute_cop_x(sam)
    if set_steering !== nothing
        sys_state.set_steering = set_steering
    end
    if depower !== nothing
        sys_state.depower = depower
    end
    if video_frame !== nothing
        sys_state.var_14 = video_frame
    end
    if wind_vec_ekf !== nothing
        sys_state.v_wind_gnd .= wind_vec_ekf
    end
    if wind_vec_lidar !== nothing
        sys_state.v_wind_200m .= wind_vec_lidar
    end
    sys_state.time = t
    log!(logger, sys_state)
    return nothing
end

"""
    should_report(step, n_steps; interval=10) -> Bool

Return `true` at every `interval`% of progress and at the last step.
"""
function should_report(step, n_steps; interval=10)
    step == n_steps && return true
    return step % max(1, div(n_steps, interval)) == 0
end

"""
    save_and_load_log(logger, name; path=nothing) -> syslog

Save a Logger and immediately load the resulting SysLog.
"""
function save_and_load_log(logger, name; path=nothing)
    if isnothing(path)
        save_log(logger, name)
    else
        save_log(logger, name; path)
    end
    return load_log(name)
end

"""
    create_heading_pid(; K, Ti, Td, dt, umin, umax) -> DiscretePID

Create a heading PID controller with standard V3 kite conventions.
Gains `Ti` and `Td` accept `false` to disable integral/derivative.
"""
function create_heading_pid(; K=1.0, Ti=false, Td=false,
                             dt, umin=-1.0, umax=1.0)
    return DiscretePID(; K, Ti, Td, Ts=dt, umin, umax)
end

"""
    create_winch_pid(; K, Ti, Td, dt, max_force=50000.0) -> DiscretePID

Create a winch PID controller that outputs force [N].
"""
function create_winch_pid(; K=1000.0, Ti=false, Td=false,
                           dt, max_force=50000.0)
    return DiscretePID(; K, Ti, Td, Ts=dt,
                       umin=-max_force, umax=max_force)
end

"""
    report_performance(sim_time, wall_time; label="")

Log simulation performance (wall time and realtime factor).
"""
function report_performance(sim_time, wall_time; label="")
    times_rt = sim_time / wall_time
    msg = isempty(label) ? "Simulation completed" :
        "Simulation completed: $label"
    @info msg wall_time=round(wall_time, digits=2) times_realtime=round(times_rt, digits=2)
    return nothing
end

"""
    find_frame_syslog_idxs(syslog, frame_csvs) ->
        Vector{Tuple{Int, Int}}

Match each `(csv_path, video_frame)` entry in `frame_csvs`
against `sl.var_14` (which is expected to hold the per-step
video frame number). Returns a vector of
`(video_frame, syslog_idx)` for the first match of each
target frame, in the order of `frame_csvs`.
"""
function find_frame_syslog_idxs(syslog, frame_csvs)
    sl = hasproperty(syslog, :syslog) ? syslog.syslog :
        syslog
    out = Tuple{Int, Int}[]
    for (_, target) in frame_csvs
        idx = findfirst(==(Float64(target)), sl.var_14)
        isnothing(idx) && continue
        push!(out, (target, idx))
    end
    return out
end

"""
    build_replay_sys_struct(set, geom, source_struc,
        vsm_set) -> (sam, sys_struct)

Build a fresh `SymbolicAWEModel` and `SystemStructure`
without running settling — load the YAML, apply geometry
adjustments, and call `init!`. Mirrors the `SETTLE=false`
branch of `flight_replay.jl` so plotting can be done after
deserializing a syslog.
"""
function build_replay_sys_struct(set,
        geom::V3GeomAdjustConfig, source_struc, vsm_set)
    sys = load_sys_struct_from_yaml(source_struc;
        system_name=V3_MODEL_NAME, set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)
    sam = SymbolicAWEModel(set, sys)
    apply_geom_adjustments!(sys, geom)
    SymbolicAWEModels.init!(sam;
        remake=false, ignore_l0=false, remake_vsm=true)
    return sam, sys
end

"""
    build_replay_name(h5_path, start_utc, end_utc,
        depower_pct, steering_pct,
        gc::V3GeomAdjustConfig) -> String

Build a descriptive log name encoding flight year, time
window, and geometry configuration.
"""
function build_replay_name(h5_path, start_utc, end_utc,
        depower_pct, steering_pct,
        gc::V3GeomAdjustConfig)
    year = match(r"(\d{4})", basename(h5_path))[1]
    sanitize(s) = replace(s, r"[:\.]" => "")
    start_san = sanitize(start_utc)
    end_san = sanitize(end_utc)

    dp_reduction = gc.reduce_depower ?
        gc.depower_reduction : 0.0
    st_reduction = gc.reduce_steering ?
        gc.steering_reduction : 0.0
    dp_tape = depower_percentage_to_length(depower_pct;
        l0_base=V3_DEPOWER_L0_BASE - dp_reduction)
    L_left, L_right = steering_percentage_to_lengths(
        steering_pct;
        l0_base=V3_STEERING_L0_BASE - st_reduction)
    suffix = build_geom_suffix(dp_tape, L_left, L_right,
        gc.tip_reduction, gc.te_frac)
    suffix = replace(suffix, "." => "")
    return "$(year)_$(start_san)_$(end_san)_$(suffix)"
end
