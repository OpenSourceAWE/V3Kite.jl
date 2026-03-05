#!/usr/bin/env julia
# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Minimal smoke test for `initialize_state`.

Usage:
    julia --project=. examples/test_initialize_state.jl
"""

using V3Kite
using LinearAlgebra: norm
using Statistics: mean
using Dates

# ============================================================================
# Shared control endpoints used by initialize_state and post-init hold phase.
# ============================================================================
const START_UDP_CMD = 0.4    # Start control command [-].
const TARGET_UDP_CMD = 0.25  # Final/hold control command [-].

# ============================================================================
# Post-init hold-check phase settings.
# ============================================================================
const HOLD_TIME_S = 5.0   # Total hold duration [s].
const HOLD_FPS = 360       # Hold integration/reporting rate [Hz].
const HOLD_WINDOW_S = 1.0  # Window length for "recent mean" metrics [s].

# Hold validation mode:
# - true  => enforce proximity to cfg.elevation.
# - false => only enforce that elevation settles to a stable band.
const HOLD_REQUIRE_TARGET_MATCH = false

# Target-match tolerance (used only when HOLD_REQUIRE_TARGET_MATCH=true).
const HOLD_TARGET_ELEV_TOL_DEG = 1.0  # |mean-last - target| [deg].

# Stability-band tolerances (used when HOLD_REQUIRE_TARGET_MATCH=false).
const HOLD_SETTLE_END_TO_MEAN_TOL_DEG = 0.3  # |end - mean-last| [deg].
const HOLD_SETTLE_DRIFT_TOL_DEG = 0.5         # |mean-last - mean-prev| [deg].
const HOLD_USE_HEADING_CONTROLLER = true      # Apply heading PID during hold.

# Hold-phase-only damping profile settings.
const START_RAMP_TIME_S = 1.0  # Time before damping starts decaying [s].
const STARTUP_DECAY_TIME_S = 2.0  # Duration of startup->nominal damping blend [s].
const STARTUP_DAMPING_PATTERN = [100.0, 500.0, 1000.0]  # Strong startup body damping.
const NOMINAL_DAMPING_PATTERN = [0.0, 0.0, 20.0]        # Nominal body damping.

"""Return body-frame damping pattern at time `t` during the post-init hold."""
function damping_profile(t)
    if t < START_RAMP_TIME_S
        return STARTUP_DAMPING_PATTERN
    end
    if STARTUP_DECAY_TIME_S <= 0
        return NOMINAL_DAMPING_PATTERN
    end
    mix = clamp((t - START_RAMP_TIME_S) / STARTUP_DECAY_TIME_S, 0.0, 1.0)
    return STARTUP_DAMPING_PATTERN .+
           (NOMINAL_DAMPING_PATTERN .- STARTUP_DAMPING_PATTERN) .* mix
end

"""Mean of values in the final `window_sec` seconds."""
function mean_last_window(values, times; window_sec)
    t_end = times[end]                      # End time of full history.
    mask = times .>= (t_end - window_sec)   # Keep samples from final window.
    data = values[mask]                     # Final-window samples.
    return isempty(data) ? values[end] : mean(data)
end

"""Mean of values in the window just before the final `window_sec`."""
function mean_prev_window(values, times; window_sec)
    t_end = times[end]                         # End time of full history.
    t_hi = t_end - window_sec                  # Upper bound of previous window.
    t_lo = t_hi - window_sec                   # Lower bound of previous window.
    mask = (times .>= t_lo) .& (times .< t_hi) # Previous-window selector.
    data = values[mask]                        # Previous-window samples.
    return isempty(data) ? values[1] : mean(data)
end

"""Map control command `udp` to depower tape length [m] (legacy batch relation)."""
function udp_to_l0(udp)
    # Batch relation: target_l0 = (200 + 5000 * udp) / 1000 [m]
    return (200.0 + 5000.0 * udp) / 1000.0
end

"""Map depower tape length [m] back to control command `udp`."""
function l0_to_udp(target_l0)
    return (1000.0 * target_l0 - 200.0) / 5000.0
end

"""Apply control command `udp` directly to depower tape length [m]."""
function set_udp!(sys, udp, gc)
    sys.segments[V3_DEPOWER_IDX].l0 = udp_to_l0(udp) - gc.reduce_depower_tape_by
    return nothing
end

"""Build a load_and_plot-compatible run name from initialization settings."""
function build_init_log_name(cfg, timestamp)
    up_t = Int(round(cfg.target_udp * 100))      # UDP tag in percent-like integer.
    us_t = Int(round(cfg.target_steering * 100)) # Steering tag in percent-like integer.
    vw_t = Int(round(cfg.target_wind))           # Wind tag [m/s].
    lt_t = Int(round(cfg.tether_length))         # Tether length tag [m].
    return "initializing_state__up_$(up_t)_us_$(us_t)_vw_$(vw_t)_lt_$(lt_t)_date_$(timestamp)"
end


"""
    V3InitConfig

Configuration for robust coupled-state initialization.
This is additive functionality and does not alter `settle_wing`.
"""
Base.@kwdef mutable struct V3InitConfig
    # Geometry paths
    source_struc_path::String = "struc_geometry.yaml"
    source_aero_path::String = "aero_geometry.yaml"
    vsm_settings_path::String = "vsm_settings.yaml"
    n_panels::Int = 36

    # Target operating point
    tether_length::Float64 = 240.0
    elevation::Float64 = 70.0
    azimuth::Float64 = 0.0
    heading::Float64 = 0.0
    upwind_dir::Float64 = -90.0
    g_earth::Float64 = 9.81
    target_wind::Float64 = 8.0
    target_udp::Float64 = 0.36
    target_steering::Float64 = 0.0

    # Continuation start point
    start_wind::Float64 = 0.0
    start_udp::Float64 = 0.36
    start_steering::Float64 = 0.0

    # Continuation and steady-state solve settings
    n_stages::Int = 6
    settle_time::Float64 = 1.0
    settle_dt::Float64 = 0.02
    vsm_interval::Int = 1
    use_heading_controller::Bool = false
    use_heading_controller_b::Bool = false
    use_heading_controller_c::Bool = false
    heading_ctrl_p::Float64 = 1.0
    heading_ctrl_i::Float64 = 0.1
    heading_ctrl_d::Float64 = 0.0
    heading_ctrl_max_steering::Float64 = 0.1
    runtime_a::Union{Nothing,Float64} = nothing
    fps_a::Union{Nothing,Float64} = nothing

    # Final convergence checks and optional refinement
    max_extra_stages::Int = 3
    extra_settle_time_factor::Float64 = 1.5
    convergence_tol_elevation_deg::Float64 = 0.25
    convergence_tol_azimuth_deg::Float64 = 0.25
    convergence_tol_heading_deg::Float64 = 0.5
    convergence_tol_kite_speed::Float64 = 0.4
    enforce_convergence::Bool = false
    runtime_b::Union{Nothing,Float64} = nothing
    fps_b::Union{Nothing,Float64} = nothing

    # Damping schedule (legacy/global defaults)
    world_damping_start::Float64 = 500.0
    world_damping_end::Float64 = 50.0
    body_damping::Float64 = 300.0

    # Optional phase-specific damping overrides
    # Phase A: continuation stage loop
    phase_a_world_damping_start::Union{Nothing,Float64} = nothing
    phase_a_world_damping_end::Union{Nothing,Float64} = nothing
    phase_a_body_damping_start::Union{Nothing,Float64,AbstractVector{<:Real}} = nothing
    phase_a_body_damping_end::Union{Nothing,Float64,AbstractVector{<:Real}} = nothing
    # Phase B: target convergence refinement
    phase_b_world_damping::Union{Nothing,Float64} = nothing
    phase_b_body_damping::Union{Nothing,Float64,AbstractVector{<:Real}} = nothing
    # Phase C: dynamic warmup
    phase_c_world_damping_start::Union{Nothing,Float64} = nothing
    phase_c_world_damping_end::Union{Nothing,Float64} = nothing
    phase_c_body_damping_start::Union{Nothing,Float64,AbstractVector{<:Real}} = nothing
    phase_c_body_damping_end::Union{Nothing,Float64,AbstractVector{<:Real}} = nothing

    # Optional dynamic warmup
    warmup_time::Float64 = 0.0
    warmup_fps::Int = 120
    runtime_c::Union{Nothing,Float64} = nothing
    fps_c::Union{Nothing,Float64} = nothing

    # Debugging/output
    save_partial_log_on_failure::Bool = false

    # Optional model choices
    use_quasi_static::Bool = false
    use_settled_geometry::Bool = false
    settle_wind::Union{Nothing,Float64} = nothing
    fallback_to_raw_geometry::Bool = true
    settle_remake::Bool = false
    settle_num_steps::Int = 2000
    settle_step_dt::Float64 = 0.01

    # Geometry modifications
    geom::V3GeomAdjustConfig = V3GeomAdjustConfig(
        reduce_tip=true, reduce_te=true,
        tether_length=240.0)
end

@inline _lerp(a, b, f) = a + (b - a) * f
@inline _udp_to_l0(udp) = (200.0 + 5000.0 * udp) / 1000.0
@inline _l0_to_udp(l0) = (1000.0 * l0 - 200.0) / 5000.0
@inline _udp_to_depower(udp, gc=V3GeomAdjustConfig()) = clamp(
    depower_length_to_percentage(
        _udp_to_l0(udp);
        delta=-gc.reduce_depower_tape_by) / 100.0,
    0.0, 1.0)
@inline _depower_to_udp(depower, gc=V3GeomAdjustConfig()) = (
    1000.0 * depower_percentage_to_length(
        depower * 100.0;
        delta=-gc.reduce_depower_tape_by) - 200.0
) / 5000.0
@inline function _set_udp!(sys, udp, gc)
    sys.segments[V3_DEPOWER_IDX].l0 = _udp_to_l0(udp) - gc.reduce_depower_tape_by
    return nothing
end

@inline _coerce_world_damping(v::Real) = float(v)
@inline _coerce_body_damping(v::Real) = float(v)
@inline _coerce_body_damping(v::AbstractVector{<:Real}) = Float64.(collect(v))
@inline _coerce_body_damping(v::Vector{Float64}) = v

@inline _resolve_or_default(v, default) = isnothing(v) ? default : v

@inline _lerp_body_damping(a::Real, b::Real, f) = _lerp(a, b, f)
@inline _lerp_body_damping(a::AbstractVector, b::AbstractVector, f) = _lerp(a, b, f)
@inline _lerp_body_damping(a::Real, b::AbstractVector, f) = _lerp(fill(float(a), length(b)), b, f)
@inline _lerp_body_damping(a::AbstractVector, b::Real, f) = _lerp(a, fill(float(b), length(a)), f)

@inline _damping_log_value(v::Real) = round(v, digits=2)
@inline _damping_log_value(v::AbstractVector{<:Real}) = round.(v; digits=2)

function _phase_damping_schedule(config::V3InitConfig)
    phase_a_world_start = _coerce_world_damping(_resolve_or_default(
        config.phase_a_world_damping_start, config.world_damping_start))
    phase_a_world_end = _coerce_world_damping(_resolve_or_default(
        config.phase_a_world_damping_end, config.world_damping_end))
    phase_a_body_start = _coerce_body_damping(_resolve_or_default(
        config.phase_a_body_damping_start, config.body_damping))
    phase_a_body_end = _coerce_body_damping(_resolve_or_default(
        config.phase_a_body_damping_end, phase_a_body_start))

    phase_b_world = _coerce_world_damping(_resolve_or_default(
        config.phase_b_world_damping, phase_a_world_end))
    phase_b_body = _coerce_body_damping(_resolve_or_default(
        config.phase_b_body_damping, phase_a_body_end))

    phase_c_world_start = _coerce_world_damping(_resolve_or_default(
        config.phase_c_world_damping_start, phase_b_world))
    phase_c_world_end = _coerce_world_damping(_resolve_or_default(
        config.phase_c_world_damping_end, 0.0))
    phase_c_body_start = _coerce_body_damping(_resolve_or_default(
        config.phase_c_body_damping_start, phase_b_body))
    phase_c_body_end = _coerce_body_damping(_resolve_or_default(
        config.phase_c_body_damping_end, phase_c_body_start))

    return (
        phase_a_world_start=phase_a_world_start,
        phase_a_world_end=phase_a_world_end,
        phase_a_body_start=phase_a_body_start,
        phase_a_body_end=phase_a_body_end,
        phase_b_world=phase_b_world,
        phase_b_body=phase_b_body,
        phase_c_world_start=phase_c_world_start,
        phase_c_world_end=phase_c_world_end,
        phase_c_body_start=phase_c_body_start,
        phase_c_body_end=phase_c_body_end,
    )
end

function _phase_timing_schedule(config::V3InitConfig)
    runtime_a = float(_resolve_or_default(config.runtime_a, config.settle_time))
    fps_a = float(_resolve_or_default(config.fps_a, 1.0 / config.settle_dt))
    n_steps_a = max(1, Int(round(runtime_a * fps_a)))
    dt_a = runtime_a / n_steps_a

    runtime_b = float(_resolve_or_default(
        config.runtime_b, config.settle_time * config.extra_settle_time_factor))
    fps_b = float(_resolve_or_default(config.fps_b, 1.0 / config.settle_dt))
    n_steps_b = max(1, Int(round(runtime_b * fps_b)))
    dt_b = runtime_b / n_steps_b

    runtime_c = float(_resolve_or_default(config.runtime_c, config.warmup_time))
    fps_c = float(_resolve_or_default(config.fps_c, float(config.warmup_fps)))
    n_steps_c = (runtime_c > 0 && fps_c > 0) ?
                max(1, Int(round(runtime_c * fps_c))) : 0
    dt_c = n_steps_c > 0 ? runtime_c / n_steps_c : 0.0

    return (
        runtime_a=runtime_a,
        fps_a=fps_a,
        n_steps_a=n_steps_a,
        dt_a=dt_a,
        runtime_b=runtime_b,
        fps_b=fps_b,
        n_steps_b=n_steps_b,
        dt_b=dt_b,
        runtime_c=runtime_c,
        fps_c=fps_c,
        n_steps_c=n_steps_c,
        dt_c=dt_c,
    )
end

function _format_phase_timeline(phase_a_end, phase_b_end, phase_c_end, t_final)
    a_hi = isnan(phase_a_end) ? t_final : phase_a_end
    b_hi = isnan(phase_b_end) ? max(a_hi, t_final) : phase_b_end
    c_hi = isnan(phase_c_end) ? max(b_hi, t_final) : phase_c_end
    return string(
        round(0.0, digits=2), "-", round(a_hi, digits=2), "s A | ",
        round(a_hi, digits=2), "-", round(b_hi, digits=2), "s B | ",
        round(b_hi, digits=2), "-", round(c_hi, digits=2), "s C")
end

@inline function _init_log_basename(config::V3InitConfig, ts; failed=false)
    up_t = Int(round(config.target_udp * 100))
    us_t = Int(round(config.target_steering * 100))
    vw_t = Int(round(config.target_wind))
    lt_t = Int(round(config.tether_length))
    base = "initializing_state__up_$(up_t)_us_$(us_t)_vw_$(vw_t)_lt_$(lt_t)_date_$(ts)"
    return failed ? base * "_failed" : base
end

function _save_initialize_partial_log!(logger, config::V3InitConfig;
                                       data_path, show_progress=true)
    save_root = joinpath(dirname(data_path), "processed_data")
    ts = Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
    save_subdir = "initializing_state_" * ts
    save_dir = joinpath(save_root, save_subdir)
    isdir(save_dir) || mkpath(save_dir)
    log_basename = _init_log_basename(config, ts; failed=true)
    save_log(logger, log_basename; path=save_dir)
    saved_log_name = joinpath(save_subdir, log_basename)
    if show_progress
        @info "Saved partial initialization log" log_name=saved_log_name path=joinpath(save_dir, log_basename * ".arrow")
    end
    return saved_log_name
end

function _make_heading_pid(config::V3InitConfig; dt_override=nothing)
    Ti = config.heading_ctrl_i > 0 ? 1.0 / config.heading_ctrl_i : false
    Td = config.heading_ctrl_d > 0 ? config.heading_ctrl_d : false
    umax = abs(config.heading_ctrl_max_steering)
    pid_dt = isnothing(dt_override) ? config.settle_dt : float(dt_override)
    return create_heading_pid(;
        K=config.heading_ctrl_p,
        Ti,
        Td,
        dt=pid_dt,
        umin=-umax,
        umax=umax)
end

function _find_steady_state_heading_controlled!(
    sam, sys, config::V3InitConfig, heading_pid, gc;
    base_steering, runtime, fps,
    logger=nothing, sys_state=nothing, t_clock=nothing)
    n_steps = max(1, Int(round(runtime * fps)))
    dt = runtime / n_steps
    heading_target = deg2rad(config.heading)
    for step in 1:n_steps
        steer_corr = heading_pid(heading_target, sys.wings[1].heading, 0.0)
        set_steering!(sys, clamp(base_steering + steer_corr, -1.0, 1.0), gc)
        if !sim_step!(sam; dt, vsm_interval=config.vsm_interval)
            throw(ErrorException(
                "heading-controlled settle unstable at " *
                "step $step/$n_steps"))
        end
        if !isnothing(logger) && !isnothing(sys_state) && !isnothing(t_clock)
            t_clock[] += dt
            log_state!(logger, sys_state, sam, t_clock[])
        end
    end
    return nothing
end

function _geom_with_tether(config::V3InitConfig)
    gc = config.geom
    return V3GeomAdjustConfig(
        reduce_tip=gc.reduce_tip,
        tip_reduction=gc.tip_reduction,
        tip_segments=gc.tip_segments,
        reduce_te=gc.reduce_te,
        te_frac=gc.te_frac,
        te_segments=gc.te_segments,
        reduce_depower_tape_by=gc.reduce_depower_tape_by,
        reduce_steering_tapes_by=gc.reduce_steering_tapes_by,
        tether_length=config.tether_length)
end

function _validate_world_damping(name, value)
    value >= 0 || throw(ArgumentError("$name must be >= 0"))
    return nothing
end

function _validate_body_damping(name, value)
    if value isa Real
        value >= 0 || throw(ArgumentError("$name must be >= 0"))
        return nothing
    end
    if value isa AbstractVector{<:Real}
        length(value) == 3 || throw(ArgumentError(
            "$name must have length 3 when provided as a vector"))
        all(v -> v >= 0, value) || throw(ArgumentError(
            "$name entries must be >= 0"))
        return nothing
    end
    throw(ArgumentError(
        "$name must be a nonnegative scalar or length-3 vector"))
end

function _validate_init_config(config::V3InitConfig)
    config.n_stages > 0 || throw(ArgumentError("n_stages must be > 0"))
    if config.runtime_a === nothing || config.runtime_b === nothing
        config.settle_time > 0 ||
            throw(ArgumentError("settle_time must be > 0 when used as fallback"))
    end
    if config.fps_a === nothing || config.fps_b === nothing
        config.settle_dt > 0 ||
            throw(ArgumentError("settle_dt must be > 0 when used as fallback"))
    end
    config.heading_ctrl_p >= 0 ||
        throw(ArgumentError("heading_ctrl_p must be >= 0"))
    config.heading_ctrl_i >= 0 ||
        throw(ArgumentError("heading_ctrl_i must be >= 0"))
    config.heading_ctrl_d >= 0 ||
        throw(ArgumentError("heading_ctrl_d must be >= 0"))
    0.0 <= config.heading_ctrl_max_steering <= 1.0 ||
        throw(ArgumentError("heading_ctrl_max_steering must be in [0, 1]"))
    config.start_wind >= 0 ||
        throw(ArgumentError("start_wind must be >= 0"))
    config.target_wind >= 0 ||
        throw(ArgumentError("target_wind must be >= 0"))
    if config.runtime_a !== nothing && config.runtime_a <= 0
        throw(ArgumentError("runtime_a must be > 0"))
    end
    if config.fps_a !== nothing && config.fps_a <= 0
        throw(ArgumentError("fps_a must be > 0"))
    end
    if config.runtime_b !== nothing && config.runtime_b <= 0
        throw(ArgumentError("runtime_b must be > 0"))
    end
    if config.fps_b !== nothing && config.fps_b <= 0
        throw(ArgumentError("fps_b must be > 0"))
    end
    if config.runtime_c !== nothing && config.runtime_c < 0
        throw(ArgumentError("runtime_c must be >= 0"))
    end
    if config.fps_c !== nothing && config.fps_c <= 0
        throw(ArgumentError("fps_c must be > 0"))
    end
    if config.settle_wind !== nothing && config.settle_wind < 0
        throw(ArgumentError("settle_wind must be >= 0"))
    end
    0.0 <= config.start_udp <= 1.0 ||
        throw(ArgumentError("start_udp must be in [0, 1]"))
    0.0 <= config.target_udp <= 1.0 ||
        throw(ArgumentError("target_udp must be in [0, 1]"))
    -1.0 <= config.start_steering <= 1.0 ||
        throw(ArgumentError("start_steering must be in [-1, 1]"))
    -1.0 <= config.target_steering <= 1.0 ||
        throw(ArgumentError("target_steering must be in [-1, 1]"))
    config.geom.reduce_depower_tape_by >= 0 ||
        throw(ArgumentError("geom.reduce_depower_tape_by must be >= 0"))
    config.geom.reduce_steering_tapes_by >= 0 ||
        throw(ArgumentError("geom.reduce_steering_tapes_by must be >= 0"))
    config.max_extra_stages >= 0 ||
        throw(ArgumentError("max_extra_stages must be >= 0"))
    if config.runtime_b === nothing
        config.extra_settle_time_factor > 0 || throw(ArgumentError(
            "extra_settle_time_factor must be > 0 when used as fallback"))
    end
    config.convergence_tol_elevation_deg >= 0 ||
        throw(ArgumentError("convergence_tol_elevation_deg must be >= 0"))
    config.convergence_tol_azimuth_deg >= 0 ||
        throw(ArgumentError("convergence_tol_azimuth_deg must be >= 0"))
    config.convergence_tol_heading_deg >= 0 ||
        throw(ArgumentError("convergence_tol_heading_deg must be >= 0"))
    config.convergence_tol_kite_speed >= 0 ||
        throw(ArgumentError("convergence_tol_kite_speed must be >= 0"))
    _validate_world_damping("world_damping_start", config.world_damping_start)
    _validate_world_damping("world_damping_end", config.world_damping_end)
    _validate_body_damping("body_damping", config.body_damping)
    if config.phase_a_world_damping_start !== nothing
        _validate_world_damping(
            "phase_a_world_damping_start",
            config.phase_a_world_damping_start)
    end
    if config.phase_a_world_damping_end !== nothing
        _validate_world_damping(
            "phase_a_world_damping_end",
            config.phase_a_world_damping_end)
    end
    if config.phase_a_body_damping_start !== nothing
        _validate_body_damping(
            "phase_a_body_damping_start",
            config.phase_a_body_damping_start)
    end
    if config.phase_a_body_damping_end !== nothing
        _validate_body_damping(
            "phase_a_body_damping_end",
            config.phase_a_body_damping_end)
    end
    if config.phase_b_world_damping !== nothing
        _validate_world_damping(
            "phase_b_world_damping",
            config.phase_b_world_damping)
    end
    if config.phase_b_body_damping !== nothing
        _validate_body_damping(
            "phase_b_body_damping",
            config.phase_b_body_damping)
    end
    if config.phase_c_world_damping_start !== nothing
        _validate_world_damping(
            "phase_c_world_damping_start",
            config.phase_c_world_damping_start)
    end
    if config.phase_c_world_damping_end !== nothing
        _validate_world_damping(
            "phase_c_world_damping_end",
            config.phase_c_world_damping_end)
    end
    if config.phase_c_body_damping_start !== nothing
        _validate_body_damping(
            "phase_c_body_damping_start",
            config.phase_c_body_damping_start)
    end
    if config.phase_c_body_damping_end !== nothing
        _validate_body_damping(
            "phase_c_body_damping_end",
            config.phase_c_body_damping_end)
    end
    damping = _phase_damping_schedule(config)
    _validate_body_damping("phase_a_body_damping_start", damping.phase_a_body_start)
    _validate_body_damping("phase_a_body_damping_end", damping.phase_a_body_end)
    _validate_body_damping("phase_b_body_damping", damping.phase_b_body)
    _validate_body_damping("phase_c_body_damping_start", damping.phase_c_body_start)
    _validate_body_damping("phase_c_body_damping_end", damping.phase_c_body_end)
    timing = _phase_timing_schedule(config)
    timing.runtime_a > 0 || throw(ArgumentError("phase A runtime must be > 0"))
    timing.fps_a > 0 || throw(ArgumentError("phase A fps must be > 0"))
    timing.runtime_b > 0 || throw(ArgumentError("phase B runtime must be > 0"))
    timing.fps_b > 0 || throw(ArgumentError("phase B fps must be > 0"))
    timing.runtime_c >= 0 || throw(ArgumentError("phase C runtime must be >= 0"))
    if timing.runtime_c > 0
        timing.fps_c > 0 || throw(ArgumentError("phase C fps must be > 0"))
    end
    return nothing
end

function _init_convergence_metrics(sam, config::V3InitConfig)
    sys = sam.sys_struct
    wing = sys.wings[1]
    st = SysState(sam)
    update_sys_state!(st, sam)

    elevation_err_deg = rad2deg(wing.elevation) - config.elevation
    azimuth_err_deg = rad2deg(wrap_to_pi(
        wing.azimuth - deg2rad(config.azimuth)))
    heading_err_deg = rad2deg(wrap_to_pi(
        wing.heading - deg2rad(config.heading)))

    return (
        elevation_err_deg=elevation_err_deg,
        azimuth_err_deg=azimuth_err_deg,
        heading_err_deg=heading_err_deg,
        kite_speed=norm(st.vel_kite),
        aoa_deg=rad2deg(st.AoA),
    )
end

function _init_is_converged(metrics, config::V3InitConfig)
    return abs(metrics.elevation_err_deg) <= config.convergence_tol_elevation_deg &&
           abs(metrics.azimuth_err_deg) <= config.convergence_tol_azimuth_deg &&
           abs(metrics.heading_err_deg) <= config.convergence_tol_heading_deg &&
           metrics.kite_speed <= config.convergence_tol_kite_speed
end

function _stage_state_metrics(sam)
    sys = sam.sys_struct
    wing = sys.wings[1]
    st = SysState(sam)
    update_sys_state!(st, sam)
    tether_force_N = isempty(sys.winches) ? NaN : norm(sys.winches[1].force)
    return (
        elevation_deg=rad2deg(wing.elevation),
        azimuth_deg=rad2deg(wing.azimuth),
        heading_deg=rad2deg(wing.heading),
        kite_speed=norm(st.vel_kite),
        aoa_deg=rad2deg(st.AoA),
        tether_force_N=tether_force_N,
    )
end

function _enforce_target_state!(sys, config::V3InitConfig, gc;
                                phase_b_world_damping,
                                phase_b_body_damping)
    sys.set.v_wind = config.target_wind
    sys.set.upwind_dir = config.upwind_dir
    SymbolicAWEModels.set_world_frame_damping(
        sys, phase_b_world_damping)
    SymbolicAWEModels.set_body_frame_damping(
        sys, phase_b_body_damping)
    _set_udp!(sys, config.target_udp, gc)
    set_steering!(sys, config.target_steering, gc)
    for winch in sys.winches
        winch.brake = true
        winch.tether_len = config.tether_length
        winch.tether_vel = 0.0
        winch.set_value = 0.0
    end
    return nothing
end

function _build_init_model(config::V3InitConfig; data_path, show_progress)
    gc = _geom_with_tether(config)

    if config.use_settled_geometry
        settle_v_app = isnothing(config.settle_wind) ?
            max(config.start_wind, config.target_wind) :
            float(config.settle_wind)
        settle_cfg = V3SettleConfig(
            source_struc_path=config.source_struc_path,
            source_aero_path=config.source_aero_path,
            vsm_settings_path=config.vsm_settings_path,
            num_steps=config.settle_num_steps,
            dt=config.settle_step_dt,
            v_wind=settle_v_app,
            elevation=config.elevation,
            tether_length=config.tether_length,
            steering_pct=100.0 * config.start_steering,
            udp_cmd=config.start_udp,
            geom=gc,
            n_panels=config.n_panels)
        try
            sam, _ = settle_wing(
                settle_cfg;
                v_app=settle_v_app,
                tether_length=config.tether_length,
                elevation=config.elevation,
                data_path=data_path,
                show_progress=show_progress,
                remake=config.settle_remake)
            return sam, sam.sys_struct, gc
        catch err
            if !config.fallback_to_raw_geometry
                rethrow(err)
            end
            @warn "settle_wing failed; falling back to raw initialization model" settle_v_app err
        end
    end

    sim_cfg = V3SimConfig(
        struc_yaml_path=config.source_struc_path,
        aero_yaml_path=config.source_aero_path,
        vsm_settings_path=config.vsm_settings_path,
        v_wind=config.start_wind,
        upwind_dir=config.upwind_dir,
        tether_length=config.tether_length,
        elevation=config.elevation,
        damping_pattern=[0.0, 0.0, 20.0],
        wing_type=REFINE,
        n_panels=config.n_panels)
    sam, sys = create_v3_model(sim_cfg; data_path=data_path)
    apply_geom_adjustments!(sys, gc)
    init!(sam; remake=false, ignore_l0=false, remake_vsm=true)
    return sam, sys, gc
end

"""
    initialize_state(config::V3InitConfig; data_path=nothing,
                     show_progress=true) -> sam

Robustly initialize a coupled tether + kite state using continuation
and `find_steady_state!` at each stage.
"""
function initialize_state(config::V3InitConfig;
                          data_path=nothing,
                          show_progress=true)
    _validate_init_config(config)
    if isnothing(data_path)
        data_path = v3_data_path()
    end

    sam, sys, gc = _build_init_model(
        config; data_path, show_progress=false)

    # Place transform explicitly at requested orientation.
    if !isempty(sys.transforms)
        tr = sys.transforms[1]
        tr.elevation = deg2rad(config.elevation)
        tr.azimuth = deg2rad(config.azimuth)
        tr.heading = deg2rad(config.heading)
        SymbolicAWEModels.reinit!([tr], sys)
    end

    damping = _phase_damping_schedule(config)
    timing = _phase_timing_schedule(config)

    sam.set.g_earth = config.g_earth
    sys.set.upwind_dir = config.upwind_dir
    SymbolicAWEModels.set_world_frame_damping(
        sys, damping.phase_a_world_start)
    SymbolicAWEModels.set_body_frame_damping(
        sys, damping.phase_a_body_start)

    for winch in sys.winches
        winch.brake = true
        winch.tether_len = config.tether_length
        winch.tether_vel = 0.0
        winch.set_value = 0.0
    end

    _set_udp!(sys, config.start_udp, gc)
    set_steering!(sys, config.start_steering, gc)
    SymbolicAWEModels.reinit!(
        sam, sam.prob, SymbolicAWEModels.FBDF())

    if config.use_quasi_static
        sam.set.quasi_static = true
        init!(sam; remake=false,
            ignore_l0=false, remake_vsm=true)
    end

    heading_pid_a = config.use_heading_controller ?
                    _make_heading_pid(config; dt_override=timing.dt_a) : nothing
    heading_pid_b = config.use_heading_controller_b ?
                    _make_heading_pid(config; dt_override=timing.dt_b) : nothing
    heading_pid_c = (config.use_heading_controller_c && timing.n_steps_c > 0) ?
                    _make_heading_pid(config; dt_override=timing.dt_c) : nothing
    n_stages = max(1, config.n_stages)
    est_stage_logs = config.use_heading_controller ?
                     n_stages * (timing.n_steps_a + 2) :
                     2 * n_stages
    est_total_logs = est_stage_logs + config.max_extra_stages + timing.n_steps_c + 16
    init_logger = nothing
    init_sys_state = nothing
    t_clock = Ref(0.0)
    phase_a_end = NaN
    phase_b_end = NaN
    phase_c_end = NaN
    if config.save_partial_log_on_failure
        init_logger, init_sys_state = create_logger(sam, est_total_logs)
    end

    try
        for stage in 1:n_stages
            frac = stage / n_stages
            stage_wind = _lerp(
                config.start_wind, config.target_wind, frac)
            stage_udp = _lerp(
                config.start_udp, config.target_udp, frac)
            stage_steering = _lerp(
                config.start_steering, config.target_steering, frac)
            stage_world_damping = _lerp(
                damping.phase_a_world_start,
                damping.phase_a_world_end, frac)
            stage_body_damping = _lerp_body_damping(
                damping.phase_a_body_start,
                damping.phase_a_body_end, frac)

            sys.set.v_wind = stage_wind
            sys.set.upwind_dir = config.upwind_dir
            SymbolicAWEModels.set_world_frame_damping(
                sys, stage_world_damping)
            SymbolicAWEModels.set_body_frame_damping(
                sys, stage_body_damping)
            _set_udp!(sys, stage_udp, gc)
            set_steering!(sys, stage_steering, gc)
            for winch in sys.winches
                winch.brake = true
                winch.tether_len = config.tether_length
                winch.tether_vel = 0.0
                winch.set_value = 0.0
            end

            if !isnothing(init_logger)
                log_state!(init_logger, init_sys_state, sam, t_clock[])
            end

            stage_state = _stage_state_metrics(sam)
            if show_progress
                @info "Initialize stage" stage n_stages wind=round(stage_wind, digits=3) vwind=round(sys.set.v_wind, digits=3) udp=round(stage_udp, digits=3) steering=round(stage_steering, digits=3) world_damping=round(stage_world_damping, digits=2) body_damping=_damping_log_value(stage_body_damping) runtime=round(timing.runtime_a, digits=3) fps=round(timing.fps_a, digits=2) use_heading_controller=config.use_heading_controller elevation_deg=round(stage_state.elevation_deg, digits=3) azimuth_deg=round(stage_state.azimuth_deg, digits=3) heading_deg=round(stage_state.heading_deg, digits=3) kite_speed=round(stage_state.kite_speed, digits=5) aoa_deg=round(stage_state.aoa_deg, digits=3) tether_force_N=round(stage_state.tether_force_N, digits=2)
            end

            try
                if config.use_heading_controller
                    _find_steady_state_heading_controlled!(
                        sam, sys, config, heading_pid_a, gc;
                        base_steering=stage_steering,
                        runtime=timing.runtime_a,
                        fps=timing.fps_a,
                        logger=init_logger,
                        sys_state=init_sys_state,
                        t_clock=t_clock)
                else
                    SymbolicAWEModels.find_steady_state!(
                        sam;
                        t=timing.runtime_a,
                        dt=timing.dt_a,
                        vsm_interval=config.vsm_interval)
                    if !isnothing(init_logger)
                        t_clock[] += timing.runtime_a
                        log_state!(init_logger, init_sys_state, sam, t_clock[])
                    end
                end
                if show_progress
                    stage_state = _stage_state_metrics(sam)
                    @info "Initialize stage result" stage n_stages elevation_deg=round(stage_state.elevation_deg, digits=3) azimuth_deg=round(stage_state.azimuth_deg, digits=3) heading_deg=round(stage_state.heading_deg, digits=3) kite_speed=round(stage_state.kite_speed, digits=5) aoa_deg=round(stage_state.aoa_deg, digits=3) tether_force_N=round(stage_state.tether_force_N, digits=2)
                end
            catch err
                stage_state = _stage_state_metrics(sam)
                throw(ErrorException(
                    "initialize_state failed at stage " *
                    "$stage/$n_stages (wind=$(round(stage_wind, digits=3)), " *
                    "udp=$(round(stage_udp, digits=3)), " *
                    "steering=$(round(stage_steering, digits=3)), " *
                    "elevation=$(round(stage_state.elevation_deg, digits=3)) deg, " *
                    "azimuth=$(round(stage_state.azimuth_deg, digits=3)) deg, " *
                    "heading=$(round(stage_state.heading_deg, digits=3)) deg, " *
                    "kite_speed=$(round(stage_state.kite_speed, digits=5)) m/s, " *
                    "AoA=$(round(stage_state.aoa_deg, digits=3)) deg, " *
                    "tether_force=$(round(stage_state.tether_force_N, digits=2)) N): " *
                    sprint(showerror, err)))
            end
        end
        phase_a_end = t_clock[]

        if config.use_quasi_static
            sam.set.quasi_static = false
            init!(sam; remake=false,
                ignore_l0=false, remake_vsm=true)
        end

        # Check convergence at the requested target state and, if needed,
        # run extra steady-state passes at final settings.
        _enforce_target_state!(
            sys, config, gc;
            phase_b_world_damping=damping.phase_b_world,
            phase_b_body_damping=damping.phase_b_body)
        if !isnothing(init_logger)
            log_state!(init_logger, init_sys_state, sam, t_clock[])
        end
        converged = false
        metrics = _init_convergence_metrics(sam, config)
        for pass in 0:config.max_extra_stages
            metrics = _init_convergence_metrics(sam, config)
            converged = _init_is_converged(metrics, config)
            if show_progress || !converged
                @info "Initialization convergence" pass converged runtime=round(timing.runtime_b, digits=3) fps=round(timing.fps_b, digits=2) world_damping=round(damping.phase_b_world, digits=2) body_damping=_damping_log_value(damping.phase_b_body) elevation_err_deg=round(metrics.elevation_err_deg, digits=4) azimuth_err_deg=round(metrics.azimuth_err_deg, digits=4) heading_err_deg=round(metrics.heading_err_deg, digits=4) kite_speed=round(metrics.kite_speed, digits=5) aoa_deg=round(metrics.aoa_deg, digits=4)
            end
            if converged || pass == config.max_extra_stages
                break
            end
            try
                if config.use_heading_controller_b
                    _find_steady_state_heading_controlled!(
                        sam, sys, config, heading_pid_b, gc;
                        base_steering=config.target_steering,
                        runtime=timing.runtime_b,
                        fps=timing.fps_b,
                        logger=init_logger,
                        sys_state=init_sys_state,
                        t_clock=t_clock)
                else
                    SymbolicAWEModels.find_steady_state!(
                        sam;
                        t=timing.runtime_b,
                        dt=timing.dt_b,
                        vsm_interval=config.vsm_interval)
                    if !isnothing(init_logger)
                        t_clock[] += timing.runtime_b
                        log_state!(init_logger, init_sys_state, sam, t_clock[])
                    end
                end
            catch err
                throw(ErrorException(
                    "initialize_state failed in extra stage " *
                    "$(pass + 1)/$(config.max_extra_stages) " *
                    "(t=$(round(timing.runtime_b, digits=4))): " *
                    sprint(showerror, err)))
            end
        end
        phase_b_end = t_clock[]

        if !converged
            msg = "initialize_state convergence criteria not met " *
                  "(elev_err=$(round(metrics.elevation_err_deg, digits=4)) deg, " *
                  "az_err=$(round(metrics.azimuth_err_deg, digits=4)) deg, " *
                  "heading_err=$(round(metrics.heading_err_deg, digits=4)) deg, " *
                  "kite_speed=$(round(metrics.kite_speed, digits=5)) m/s)"
            if config.enforce_convergence
                throw(ErrorException(msg))
            elseif show_progress
                @warn msg
            end
        end

        if timing.n_steps_c > 0
            SymbolicAWEModels.set_world_frame_damping(
                sys, max(0.0, damping.phase_c_world_start))
            SymbolicAWEModels.set_body_frame_damping(
                sys, damping.phase_c_body_start)
            if !isnothing(init_logger)
                log_state!(init_logger, init_sys_state, sam, t_clock[])
            end
            for step in 1:timing.n_steps_c
                frac = step / timing.n_steps_c
                warm_world_damping = _lerp(
                    damping.phase_c_world_start,
                    damping.phase_c_world_end, frac)
                warm_body_damping = _lerp_body_damping(
                    damping.phase_c_body_start,
                    damping.phase_c_body_end, frac)
                SymbolicAWEModels.set_world_frame_damping(
                    sys, max(0.0, warm_world_damping))
                SymbolicAWEModels.set_body_frame_damping(
                    sys, warm_body_damping)
                if config.use_heading_controller_c
                    steer_corr = heading_pid_c(
                        deg2rad(config.heading),
                        sys.wings[1].heading, 0.0)
                    set_steering!(
                        sys,
                        clamp(config.target_steering + steer_corr, -1.0, 1.0),
                        gc)
                else
                    set_steering!(sys, config.target_steering, gc)
                end
                if !sim_step!(sam;
                    set_values=[0.0],
                    dt=timing.dt_c,
                    vsm_interval=config.vsm_interval)
                    throw(ErrorException(
                        "initialize_state warmup unstable at " *
                        "step $step/$(timing.n_steps_c)"))
                end
                if !isnothing(init_logger)
                    t_clock[] += timing.dt_c
                    log_state!(init_logger, init_sys_state, sam, t_clock[])
                end
            end
            phase_c_end = t_clock[]
        else
            phase_c_end = phase_b_end
        end

        if show_progress
            @info "Initialization phase timeline [s]" timeline=_format_phase_timeline(
                phase_a_end, phase_b_end, phase_c_end, t_clock[])
        end
    catch err
        if config.save_partial_log_on_failure && !isnothing(init_logger)
            try
                _save_initialize_partial_log!(init_logger, config;
                    data_path=data_path,
                    show_progress=true)
            catch save_err
                @warn "Failed to save partial initialization log" save_err
            end
        end
        if show_progress
            @warn "Initialization phase timeline [s]" timeline=_format_phase_timeline(
                phase_a_end, phase_b_end, phase_c_end, t_clock[])
        end
        rethrow(err)
    end

    return sam
end

function _normalize_init_kwargs(kwargs)
    args = Dict{Symbol,Any}(pairs(kwargs))
    gc = haskey(args, :geom) && args[:geom] isa V3GeomAdjustConfig ?
         args[:geom] : V3GeomAdjustConfig()
    used_legacy = false

    if haskey(args, :start_depower)
        haskey(args, :start_udp) && throw(ArgumentError(
            "Provide only one of start_udp or start_depower"))
        args[:start_udp] = _depower_to_udp(float(args[:start_depower]), gc)
        delete!(args, :start_depower)
        used_legacy = true
    end

    if haskey(args, :target_depower)
        haskey(args, :target_udp) && throw(ArgumentError(
            "Provide only one of target_udp or target_depower"))
        args[:target_udp] = _depower_to_udp(float(args[:target_depower]), gc)
        delete!(args, :target_depower)
        used_legacy = true
    end

    if used_legacy
        @warn "start_depower/target_depower are deprecated; use start_udp/target_udp"
    end

    return (; (k => v for (k, v) in args)...)
end

function initialize_state(; data_path=nothing,
                          show_progress=true,
                          kwargs...)
    config_kwargs = _normalize_init_kwargs(kwargs)
    config = V3InitConfig(; config_kwargs...)
    return initialize_state(config;
        data_path=data_path,
        show_progress=show_progress)
end

# ============================================================================
# initialize_state phase inputs.
# ============================================================================
cfg = V3InitConfig(
    # ------------------------------------------------------------------------
    # Common model/geometry inputs.
    # ------------------------------------------------------------------------
    source_struc_path="struc_geometry.yaml",  # Structural YAML input.
    source_aero_path="aero_geometry.yaml",    # Aero geometry YAML input.
    vsm_settings_path="vsm_settings.yaml",    # VSM settings YAML input.
    geom=V3GeomAdjustConfig(
        reduce_tip=true,       # Apply leading-edge tip reduction.
        reduce_te=true,        # Apply trailing-edge shortening.
        reduce_depower_tape_by=0.0,    # Depower tape neutral reduction [m].
        reduce_steering_tapes_by=0.0,  # Steering tape neutral reduction [m].
        tether_length=269.0),  # Geometry tether length used in adjustment.

    # ------------------------------------------------------------------------
    # Target operating point (reference for continuation + convergence checks).
    # ------------------------------------------------------------------------
    tether_length=269.0,                      # Target tether length [m].
    elevation=50.0,                           # Initial requested elevation [deg].
    azimuth=0.0,                              # Initial requested azimuth [deg].
    heading=0.0,                              # Initial requested heading [deg].
    upwind_dir=-90.0,                         # Wind direction convention [deg].
    g_earth=0.0,                              # Gravity [m/s^2].
    target_wind=8.4,                          # Final operating wind speed [m/s].
    target_udp=TARGET_UDP_CMD,                # Final control command [-].
    target_steering=0.0,                      # Final steering command [-].

    # ------------------------------------------------------------------------
    # Phase A: continuation to the target operating point.
    # This phase is the "path-following" phase: each stage interpolates from
    # start_* to target_* controls, while solving toward equilibrium each time.
    # Unique role: robustly approach the final point in gradual steps.
    # ------------------------------------------------------------------------
    start_wind=7.0,                           # Continuation start wind [m/s].
    start_udp=START_UDP_CMD,                  # Continuation start control command [-].
    start_steering=0.0,                       # Continuation start steering [-].
    n_stages=10,                             # Number of continuation stages.
    runtime_a=1.0,                            # Solve runtime per stage [s].
    fps_a=180.0,                             # Solve frequency for Phase A [Hz].
    vsm_interval=1,                           # VSM update interval in steps.
    use_heading_controller=true,              # Enable heading PID during stage settling.
    use_heading_controller_b=true,            # Enable heading PID during Phase B refinement.
    use_heading_controller_c=true,            # Enable heading PID during Phase C warmup.
    heading_ctrl_p=1.0,                       # Match batch zenith effective K (heading_p default 0.0 falls back to 1.0).
    heading_ctrl_i=0.1,                       # Match batch zenith default heading_i.
    heading_ctrl_d=0.0,                       # PID derivative gain.
    heading_ctrl_max_steering=0.2,           # Match batch configuration max_us_zenith=0.02.
    phase_a_world_damping_start=10.0,         # World damping at stage 1.
    phase_a_world_damping_end=0.0,            # World damping at final stage.
    phase_a_body_damping_start=[10.0, 10.0, 100.0],  # Body damping at stage 1.
    phase_a_body_damping_end=[0.0, 0.0, 20.0],    # Body damping at final stage.

    # ------------------------------------------------------------------------
    # Phase B: fixed-target convergence refinement.
    # This phase no longer ramps controls: it holds the final target point and
    # performs additional settle passes until tolerances are met or passes end.
    # Unique role: improve final accuracy at the exact target settings.
    # ------------------------------------------------------------------------
    max_extra_stages=0,                       # Extra settle passes after stage loop.
    runtime_b=1.0,                             # Runtime per extra convergence pass [s].
    fps_b=360.0,                              # Solve frequency for Phase B [Hz].
    phase_b_world_damping=0.0,                 # Fixed world damping during Phase B.
    phase_b_body_damping=[0.0, 0.0, 20.0],  # Fixed body damping during Phase B.
    convergence_tol_elevation_deg=1,        # Init convergence tol: elevation [deg].
    convergence_tol_azimuth_deg=0.1,          # Init convergence tol: azimuth [deg].
    convergence_tol_heading_deg=0.1,          # Init convergence tol: heading [deg].
    convergence_tol_kite_speed=0.5,           # Init convergence tol: kite speed [m/s].
    enforce_convergence=false,                 # Throw if init convergence is not met.

    # ------------------------------------------------------------------------
    # Phase C: dynamic warmup under forward simulation.
    # This phase switches from steady-state solving to time stepping (`sim_step!`)
    # and checks whether the initialized state stays stable in real dynamics.
    # Unique role: bridge solver equilibrium to runtime simulation robustness.
    # ------------------------------------------------------------------------
    runtime_c=10.0,                            # Total warmup runtime [s].
    fps_c=360.0,                               # Warmup integration frequency [Hz].
    phase_c_world_damping_start=0.0,           # Warmup world damping at start.
    phase_c_world_damping_end=0.0,             # Warmup world damping at end.
    phase_c_body_damping_start=[0.0, 0.0, 20.0],  # Warmup body damping at start.
    phase_c_body_damping_end=[0.0, 0.0, 20.0],    # Warmup body damping at end.

    # ------------------------------------------------------------------------
    # Failure logging + optional pre-initialize geometry settle.
    # ------------------------------------------------------------------------
    save_partial_log_on_failure=true,         # Save partial init log to processed_data if initialization fails.
    use_quasi_static=false,                   # Optional quasi-static mode during init.
    use_settled_geometry=false,                # Pre-settle geometry before initialization.
    settle_wind=7.0,                          # Wind used in geometry settling pre-pass [m/s].
    fallback_to_raw_geometry=false,           # Fallback if settling pass fails.
    settle_remake=false,                      # Recompute settled files even if cached.
    settle_num_steps=1200,                    # Geometry-settling simulation steps.
    settle_step_dt=0.01,                      # Geometry-settling step size [s].
)

@info "Initializing coupled state..."
@info "UDP control mapping" start_udp = START_UDP_CMD target_udp = TARGET_UDP_CMD start_l0_m = udp_to_l0(START_UDP_CMD) target_l0_m = udp_to_l0(TARGET_UDP_CMD)
# Use closure + invokelatest to avoid world-age issues for keyword dispatch
# in long-running REPL sessions.
sam = Base.invokelatest(() ->
    initialize_state(cfg; show_progress=true))  # Initialized SymbolicAWEModel.

sys = sam.sys_struct                    # Underlying mutable system structure.
wing = sys.wings[1]                     # Primary wing object (single-wing setup).
init_state = SysState(sam)              # Snapshot container for derived state.
update_sys_state!(init_state, sam)
elevation_deg = rad2deg(wing.elevation)  # Current wing elevation [deg].
azimuth_deg = rad2deg(wing.azimuth)      # Current wing azimuth [deg].
heading_deg = rad2deg(wing.heading)      # Current wing heading [deg].
elevation_err_deg = elevation_deg - cfg.elevation  # Elevation relative to requested init angle [deg].
azimuth_err_deg = rad2deg(wrap_to_pi(wing.azimuth - deg2rad(cfg.azimuth)))  # Wrapped azimuth error [deg].
heading_err_deg = rad2deg(wrap_to_pi(wing.heading - deg2rad(cfg.heading)))  # Wrapped heading error [deg].
kite_speed = norm(init_state.vel_kite)    # Kite speed magnitude [m/s].
udp_l0_m = sys.segments[V3_DEPOWER_IDX].l0  # Current tape length [m].
udp_value = l0_to_udp(udp_l0_m)             # User-facing control command [-].

println("Initialization summary")
println("  wind [m/s]        : ", round(sys.set.v_wind, digits=4))
println("  upwind_dir [deg]  : ", round(sys.set.upwind_dir, digits=4))
println("  elevation [deg]   : ", round(elevation_deg, digits=4))
println("  azimuth [deg]     : ", round(azimuth_deg, digits=4))
println("  heading [deg]     : ", round(heading_deg, digits=4))
println("  elev err [deg]    : ", round(elevation_err_deg, digits=4))
println("  az err [deg]      : ", round(azimuth_err_deg, digits=4))
println("  heading err [deg] : ", round(heading_err_deg, digits=4))
println("  tether_len [m]    : ", round(sys.winches[1].tether_len, digits=4))
println("  udp [-]           : ", round(udp_value, digits=4))
println("  udp_l0 [m]        : ", round(udp_l0_m, digits=4))
println("  steering [-]      : ", round(get_steering(sys), digits=4))
println("  apparent wind [m/s]: ", round(norm(wing.va_b), digits=4))
println("  kite speed [m/s]  : ", round(kite_speed, digits=5))
println("  AoA [deg]         : ", round(rad2deg(init_state.AoA), digits=4))

@info "Running post-init zenith hold check..." hold_time = HOLD_TIME_S fps = HOLD_FPS
n_hold = max(1, Int(round(HOLD_TIME_S * HOLD_FPS)))  # Total hold integration steps.
dt_hold = HOLD_TIME_S / n_hold                        # Hold step size [s].
hold_t = Float64[0.0]                                 # Recorded times [s].
hold_elev = Float64[rad2deg(sys.wings[1].elevation)] # Recorded elevation trace [deg].
logger, hold_state = create_logger(sam, n_hold + 1)  # Persisted state history for post-analysis.
log_state!(logger, hold_state, sam, 0.0)             # Log initial post-init sample.
hold_heading_pid = if HOLD_USE_HEADING_CONTROLLER
    Ti = cfg.heading_ctrl_i > 0 ? 1.0 / cfg.heading_ctrl_i : false
    Td = cfg.heading_ctrl_d > 0 ? cfg.heading_ctrl_d : false
    umax = abs(cfg.heading_ctrl_max_steering)
    create_heading_pid(;
        K=cfg.heading_ctrl_p,
        Ti,
        Td,
        dt=dt_hold,
        umin=-umax,
        umax=umax)
else
    nothing
end
hold_heading_target = deg2rad(cfg.heading)
hold_unstable_msg = nothing
for step in 1:n_hold
    t = step * dt_hold  # Simulation time [s].
    SymbolicAWEModels.set_body_frame_damping(sys, damping_profile(t))
    set_udp!(sys, cfg.target_udp, cfg.geom)           # Hold target UDP command.
    if hold_heading_pid === nothing
        set_steering!(sys, cfg.target_steering, cfg.geom) # Hold steering fixed at target.
    else
        steer_corr = hold_heading_pid(
            hold_heading_target, sys.wings[1].heading, 0.0)
        set_steering!(
            sys,
            clamp(cfg.target_steering + steer_corr, -1.0, 1.0),
            cfg.geom)
    end
    for winch in sys.winches
        winch.brake = true                     # Lock winch (no reel-out command).
        winch.tether_len = cfg.tether_length   # Enforce fixed tether length [m].
        winch.tether_vel = 0.0                 # Zero reel speed [m/s].
        winch.set_value = 0.0                  # Zero direct winch actuation.
    end
    if !sim_step!(sam;
        set_values=[0.0],
        dt=dt_hold,
        vsm_interval=cfg.vsm_interval)
        hold_unstable_msg = "Post-init hold unstable at step $step/$n_hold"
        break
    end
    log_state!(logger, hold_state, sam, t)            # Append logged sample for replay/plotting.
    push!(hold_t, t)                                # Save time sample [s].
    push!(hold_elev, rad2deg(sys.wings[1].elevation))  # Save elevation sample [deg].
end

# Post-hold metrics.
hold_elev_end = hold_elev[end]  # Final elevation sample [deg].
hold_elev_mean_last = mean_last_window(hold_elev, hold_t;
    window_sec=HOLD_WINDOW_S)  # Mean elevation in last window [deg].
hold_elev_mean_prev = mean_prev_window(hold_elev, hold_t;
    window_sec=HOLD_WINDOW_S)  # Mean elevation in previous window [deg].
hold_elev_shift_from_init = hold_elev_mean_last - elevation_deg  # How much hold state moved from post-init state [deg].
hold_elev_target_err = hold_elev_mean_last - cfg.elevation       # Distance from requested init elevation [deg].
hold_elev_end_to_mean = hold_elev_end - hold_elev_mean_last      # Endpoint consistency with recent mean [deg].
hold_elev_drift = hold_elev_mean_last - hold_elev_mean_prev      # Trend between last and previous windows [deg].

println("Post-init hold summary")
println("  hold_time [s]       : ", HOLD_TIME_S)
println("  hold_time simulated [s]: ", round(hold_t[end], digits=4))
println("  elevation end [deg] : ", round(hold_elev_end, digits=4))
println("  elevation mean-last [deg]: ", round(hold_elev_mean_last, digits=4))
println("  elevation mean-prev [deg]: ", round(hold_elev_mean_prev, digits=4))
println("  elev shift from init [deg] : ", round(hold_elev_shift_from_init, digits=4))
println("  elev err to target [deg]   : ", round(hold_elev_target_err, digits=4))
println("  settle end-mean [deg]      : ", round(hold_elev_end_to_mean, digits=4))
println("  settle drift [deg]         : ", round(hold_elev_drift, digits=4))

# Persist run in a timestamped folder under processed_data, with load_and_plot tags.
ts = Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
save_root = joinpath(dirname(@__DIR__), "processed_data")
save_subdir = "initializing_state_" * ts
save_dir = joinpath(save_root, save_subdir)
isdir(save_dir) || mkpath(save_dir)
log_basename = build_init_log_name(cfg, ts)
save_log(logger, log_basename; path=save_dir)
saved_log_name = joinpath(save_subdir, log_basename)
@info "Saved initialization run log" log_name = saved_log_name path = joinpath(save_dir, log_basename * ".arrow")

if hold_unstable_msg !== nothing
    error(hold_unstable_msg)
end

if HOLD_REQUIRE_TARGET_MATCH
    if abs(hold_elev_target_err) > HOLD_TARGET_ELEV_TOL_DEG
        error(
            "Post-init hold did not converge near target elevation. " *
            "mean-last=$(round(hold_elev_mean_last, digits=4)) deg, " *
            "target=$(round(cfg.elevation, digits=4)) deg, " *
            "error=$(round(hold_elev_target_err, digits=4)) deg, " *
            "tol=$(round(HOLD_TARGET_ELEV_TOL_DEG, digits=4)) deg.")
    end
else
    # Stability-only mode: accept any equilibrium elevation if it is settled.
    if abs(hold_elev_end_to_mean) > HOLD_SETTLE_END_TO_MEAN_TOL_DEG ||
       abs(hold_elev_drift) > HOLD_SETTLE_DRIFT_TOL_DEG
        error(
            "Post-init hold did not settle to a stable elevation band. " *
            "end_to_mean=$(round(hold_elev_end_to_mean, digits=4)) deg " *
            "(tol=$(round(HOLD_SETTLE_END_TO_MEAN_TOL_DEG, digits=4)) deg), " *
            "drift=$(round(hold_elev_drift, digits=4)) deg " *
            "(tol=$(round(HOLD_SETTLE_DRIFT_TOL_DEG, digits=4)) deg).")
    end
end
