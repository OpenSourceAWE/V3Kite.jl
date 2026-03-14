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
    log_state!(logger, sys_state, sam, t)

Update sys_state from the model, set time, compute drag/lift
coefficients into `var_01`/`var_02`, mean TE segment force
into `var_03`, and log.
"""
function log_state!(logger, sys_state, sam, t)
    update_sys_state!(sys_state, sam)
    sys_state.var_01 = compute_drag_coeff(sam)
    sys_state.var_02 = compute_lift_coeff(sam)
    sys_state.var_03 = mean_te_segment_force(sam)
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
