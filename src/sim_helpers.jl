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

Compute the drag coefficient from the first wing's aero force
projected onto the apparent wind direction, normalized by
`q_inf * A_proj`. Uses `rho = 1.225 kg/m³`.
"""
function compute_drag_coeff(sam)
    wing = sam.sys_struct.wings[1]
    va_b = wing.va_b
    v_app = norm(va_b)
    v_app < 1e-6 && return 0.0
    drag_force = dot(wing.aero_force_b, va_b / v_app)
    A_proj = calculate_projected_area(wing.vsm_wing)
    q_inf = 0.5 * 1.225 * v_app^2
    return drag_force / (q_inf * A_proj)
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
    q_inf = 0.5 * 1.225 * v_app^2
    return norm(lift_vec) / (q_inf * A_proj)
end

const _RHO_SL = 1.225
const _CF_SKIN = 0.01  # EKF-AWE hardcoded skin friction

"""
    _segment_ekf_drag(sys, seg; cd, cf) -> Float64

EKF-AWE drag force magnitude for one segment. Uses the
EKF formula `cd_t = cd * sin(θ)³ + π * cf * cos(θ)³`
with ground-level wind (consistent with `compute_drag_coeff`
using `ρ = 1.225`).
"""
function _segment_ekf_drag(sys, seg; cd, cf=_CF_SKIN)
    p1, p2 = seg.point_idxs
    vel_mid = 0.5 .* (sys.points[p1].vel_w .+
                       sys.points[p2].vel_w)
    va = sys.wind_vec_gnd .- vel_mid
    va_norm = norm(va)
    va_norm < 1e-6 && return 0.0
    e_seg = sys.points[p2].pos_w .- sys.points[p1].pos_w
    e_len = norm(e_seg)
    e_len < 1e-6 && return 0.0
    e_hat = e_seg ./ e_len
    cos_th = clamp(abs(dot(va, e_hat)) / va_norm, 0.0, 1.0)
    sin_th = sqrt(1.0 - cos_th^2)
    cd_t = cd * sin_th^3 + π * cf * cos_th^3
    area = seg.len * seg.diameter
    return 0.5 * _RHO_SL * va_norm^2 * area * cd_t
end

"""
    _classify_segments(sys) -> (tether_idxs, bridle_idxs)

Classify segments into tether vs bridle. Wing structural
segments (both endpoints WING) are excluded from both.
"""
function _classify_segments(sys)
    tether_set = Set{Int}()
    for tether in sys.tethers
        union!(tether_set, tether.segment_idxs)
    end
    tether_idxs = Int[]
    bridle_idxs = Int[]
    for seg in sys.segments
        p1_type = sys.points[seg.point_idxs[1]].type
        p2_type = sys.points[seg.point_idxs[2]].type
        p1_type == WING && p2_type == WING && continue
        if seg.idx in tether_set
            push!(tether_idxs, seg.idx)
        else
            push!(bridle_idxs, seg.idx)
        end
    end
    return tether_idxs, bridle_idxs
end

"""
    compute_tether_drag_coeff(sam) -> Float64

Tether drag coefficient using EKF-AWE formula, normalized
by the same `q_inf * A_proj` reference as `compute_drag_coeff`.
"""
function compute_tether_drag_coeff(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    v_app = norm(wing.va_b)
    v_app < 1e-6 && return 0.0
    A_proj = calculate_projected_area(wing.vsm_wing)
    q_ref = 0.5 * _RHO_SL * v_app^2 * A_proj
    cd = sam.set.cd_tether
    tether_idxs, _ = _classify_segments(sys)
    D = sum(
        _segment_ekf_drag(sys, sys.segments[i]; cd)
        for i in tether_idxs; init=0.0)
    return D / q_ref
end

"""
    compute_bridle_drag_coeff(sam) -> Float64

Bridle drag coefficient using EKF-AWE formula, normalized
by the same `q_inf * A_proj` reference as `compute_drag_coeff`.
"""
function compute_bridle_drag_coeff(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    v_app = norm(wing.va_b)
    v_app < 1e-6 && return 0.0
    A_proj = calculate_projected_area(wing.vsm_wing)
    q_ref = 0.5 * _RHO_SL * v_app^2 * A_proj
    cd = sam.set.cd_tether
    _, bridle_idxs = _classify_segments(sys)
    D = sum(
        _segment_ekf_drag(sys, sys.segments[i]; cd)
        for i in bridle_idxs; init=0.0)
    return D / q_ref
end

"""
    compute_kcu_drag_coeff(sam) -> Float64

KCU drag coefficient. Nondimensionalizes the sim's own
KCU drag (point 1 with area and drag_coeff from YAML)
by `q_inf * A_proj` using kite apparent wind speed.
"""
function compute_kcu_drag_coeff(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    v_app_kite = norm(wing.va_b)
    v_app_kite < 1e-6 && return 0.0
    kcu = sys.points[1]
    va_kcu = sys.wind_vec_gnd .- kcu.vel_w
    D_kcu = 0.5 * _RHO_SL * norm(va_kcu)^2 *
            kcu.area * kcu.drag_coeff
    A_proj = calculate_projected_area(wing.vsm_wing)
    q_ref = 0.5 * _RHO_SL * v_app_kite^2 * A_proj
    return D_kcu / q_ref
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
    quarter_chord_mid(le_3, te_3, le_4, te_4)

Quarter-chord reference point averaged over both strut sides.
Returns midpoint of (0.75·LE + 0.25·TE) for each side.
"""
function quarter_chord_mid(le_3, te_3, le_4, te_4)
    qc_3 = 0.75 .* le_3 .+ 0.25 .* te_3
    qc_4 = 0.75 .* le_4 .+ 0.25 .* te_4
    return (qc_3 .+ qc_4) ./ 2
end

"""
    quarter_chord_mid(sys)

Quarter-chord ref from sys_struct points 10-13.
"""
function quarter_chord_mid(sys)
    return quarter_chord_mid(
        sys.points[10].pos_w,
        sys.points[11].pos_w,
        sys.points[12].pos_w,
        sys.points[13].pos_w)
end

"""
    compute_bridle_aoa(sys) -> Float64

Compute the angle of attack in the bridle reference frame.
The bridle z-axis points from KCU (point 1) to the
quarter-chord midpoint, x-axis is perpendicular in the
body y–bridle z plane.
"""
function compute_bridle_aoa(sys)
    mid_w = quarter_chord_mid(sys)
    z_br = normalize(mid_w - sys.points[1].pos_w)
    R_b_w = calc_R_b_w(sys)
    y_br = R_b_w[:, 2]
    x_br = normalize(cross(y_br, z_br))
    wing = sys.wings[1]
    v_app_w = R_b_w * wing.va_b
    return atan(dot(v_app_w, z_br), dot(v_app_w, x_br))
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
    mid_w = quarter_chord_mid(sys)
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
direction (KCU → quarter-chord midpoint), projected into
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
    qc_w = quarter_chord_mid(sys)
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
    log_state!(logger, sys_state, sam, t)

Update sys_state from the model, set time, and log.

Computed variables:
- `var_01`/`var_02`: wing drag/lift coefficients
- `var_03`: mean TE segment force
- `var_04`: bridle AoA
- `var_05`-`var_07`: bridle NED Euler (yaw, pitch, roll)
- `var_08`: bridle pitch angle
- `var_09`-`var_11`: tether/bridle/KCU drag coefficients
  (EKF-AWE formula for apples-to-apples comparison)
- `var_12`: geometric wing incidence (photogrammetry-style)
"""
function log_state!(logger, sys_state, sam, t)
    update_sys_state!(sys_state, sam)
    sys_state.var_01 = compute_drag_coeff(sam)
    sys_state.var_02 = compute_lift_coeff(sam)
    sys_state.var_03 = mean_te_segment_force(sam)
    sys_state.var_04 = compute_bridle_aoa(sam.sys_struct)
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
