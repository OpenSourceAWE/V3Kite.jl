# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Run for Circular Flight

Runs multiple parameter combinations for the v3 kite
circular-flight simulation. Each run saves a permanent log
with parameter tags in the filename. Initial equilibrium is
established by `settle_wing`.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX, V3_STEERING_GAIN
using SymbolicAWEModels
using KiteUtils: wind_vec_from_angles
using GLMakie
using LinearAlgebra
using Dates

# =============================================================================
# Configuration
# =============================================================================
# these are 2019 and 2025 averages, except for elevation & udp that is.
defaults = (
    elevation=35, g_earth=0.0,
    us=0.05, udp=0.235, vw=8.0, lt=270,
    d_tether=13.5, kcu_mass=22.65,
)
sweeps = nothing
# combine_all = (
#     us=[0.05, 0.1, 0.15],
#     udp=[0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42],
# )
# #TODO: when you are done you will still need to do:
# udp = [0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43]
# us = 0.075
#combine_all = (
#    us=[0.05, 0.1, 0.125, 0.15],
#    udp=[0.27, 0.31, 0.35, 0.39, 0.43],
#)
combine_all = (
    us=[0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18],
    udp=[0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43],
)
IS_REMAKE_SETTLE = true
IS_VISUALIZE_SETTLE = false
IS_DEBUG_ON_FAILURE = false
IS_REPLAY_ON_FAILURE = false

batch_tag = "circles_batch_" *
            Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
PROJECT_DIR = dirname(@__DIR__)
DEFAULT_STRUC_YAML_PATH = joinpath(PROJECT_DIR, "data/python_yamls/struc_geometry_julia_generated.yaml")
DEFAULT_AERO_YAML_PATH = joinpath(PROJECT_DIR, "data/aero_geometry.yaml")
DEFAULT_VSM_SETTINGS_PATH = joinpath(PROJECT_DIR, "data/vsm_settings.yaml")


# =============================================================================
# Circular flight simulation function
# =============================================================================

function _unwrap_step(prev_uw, raw)
    delta = raw - mod(prev_uw, 2pi)
    if delta > pi
        delta -= 2pi
    elseif delta < -pi
        delta += 2pi
    end
    return prev_uw + delta
end

function course_rate_converged(course_uw, azimuth_uw, elevation,
    time_buf, n; window_sec, rel_tol, dt)
    t_end = time_buf[n]
    i_lo = n
    while i_lo > 1 && time_buf[i_lo-1] >= t_end - window_sec
        i_lo -= 1
    end
    (n - i_lo) < 2 && return false
    sum_abs = 0.0
    minv = Inf
    maxv = -Inf
    cnt = 0
    @inbounds for k in (i_lo+1):n
        r = (course_uw[k] - course_uw[k-1]) / dt -
            ((azimuth_uw[k] - azimuth_uw[k-1]) / dt) *
            sin(elevation[k])
        a = abs(r)
        sum_abs += a
        minv = min(minv, a)
        maxv = max(maxv, a)
        cnt += 1
    end
    cnt == 0 && return false
    mean_abs = sum_abs / cnt
    mean_abs <= 0 && return false
    return (maxv - minv) / mean_abs < rel_tol
end

function _round_vec(v; digits=4)
    return Tuple(round.(Float64.(collect(v)); digits=digits))
end

function _log_refine_vsm_diagnostics(sys, step, t, err)
    wing = sys.wings[1]
    err_msg = isnothing(err) ? "sim_step! returned false" :
              sprint(showerror, err)
    wing_points = [
        p for p in sys.points
        if p.type == SymbolicAWEModels.WING &&
        p.wing_idx == wing.idx
    ]
    va_stats = [
        (idx=p.idx, va_norm=norm(p.va_b),
            va_b=_round_vec(p.va_b))
        for p in wing_points
    ]
    low_va = filter(x -> !isfinite(x.va_norm) ||
            x.va_norm < 0.5, va_stats)
    sort!(low_va, by=x -> x.va_norm)
    nonfinite_points = [
        p.idx for p in wing_points
        if any(x -> !isfinite(x), p.pos_w) ||
        any(x -> !isfinite(x), p.vel_w) ||
        any(x -> !isfinite(x), p.va_b)
    ]
    stretch_stats = [
        (idx=s.idx, stretch=s.l0 != 0 ? (s.len - s.l0) / s.l0 : NaN,
            len=s.len, l0=s.l0)
        for s in sys.segments
        if isfinite(s.len) && isfinite(s.l0)
    ]
    max_stretch = isempty(stretch_stats) ? NaN :
                  maximum(abs(x.stretch) for x in stretch_stats)
    min_va = isempty(va_stats) ? NaN :
             minimum(x.va_norm for x in va_stats)
    max_va = isempty(va_stats) ? NaN :
             maximum(x.va_norm for x in va_stats)

    @error "REFINE VSM failure diagnostics" step t err = err_msg wing_va_norm = norm(wing.va_b) wing_va_b = _round_vec(wing.va_b) elevation = rad2deg(wing.elevation) heading = rad2deg(wing.heading) course = rad2deg(wing.course) min_wing_point_va = min_va max_wing_point_va = max_va low_va_points = length(low_va) nonfinite_points max_segment_stretch = max_stretch left_l0 = sys.segments[V3_STEERING_LEFT_IDX].l0 right_l0 = sys.segments[V3_STEERING_RIGHT_IDX].l0
    for item in first(low_va, min(length(low_va), 8))
        @warn "Low/non-finite wing-point apparent velocity" point = item.idx va_norm = item.va_norm va_b = item.va_b
    end
end

function _try_log_failure_state!(logger, sys_state, sam, t)
    try
        log_state!(logger, sys_state, sam, t)
        return true
    catch err
        @warn "Could not append failure state to log" err
        return false
    end
end

function _save_circle_failure_artifacts!(logger, sys_state, sam,
    step, t, err; save_dir, run_tag, debug_on_failure,
    replay_on_failure)

    _log_refine_vsm_diagnostics(sam.sys_struct, step, t, err)
    logged_failure_state =
        _try_log_failure_state!(logger, sys_state, sam, t)
    debug_on_failure || return nothing

    tag = isempty(run_tag) ? "run" : run_tag
    fail_name = "partial_circles_failure_$(tag)_" *
                Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
    save_log(logger, fail_name; path=save_dir)
    fail_path = joinpath(save_dir, fail_name * ".arrow")
    @info "Saved partial circular failure log" path = fail_path logged_failure_state

    global _circle_failure_log, _circle_failure_step,
    _circle_failure_error, _circle_failure_scene
    _circle_failure_log = load_log(fail_name; path=save_dir)
    _circle_failure_step = step
    _circle_failure_error = err

    if replay_on_failure
        _circle_failure_scene = replay(_circle_failure_log,
            sam.sys_struct; autoplay=false)
        display(_circle_failure_scene)
    end
    return nothing
end

"""
    run_circles(; kwargs...) -> (syslog, sam)

Run a v3 kite circular-flight simulation. The starting
state is produced by `settle_wing`; winch brake is engaged
and steering is ramped from the settled trim toward the
circular-flight target.
"""
function run_circles(;
    sim_time_circles=0.0, fps_circles=1,
    body_damping=[0.0, 0.0, 20.0],
    body_damping_delta=([37, 38], [0.0, 20.0, 20.0]),
    udp=0.4,
    ramp_time_us=25.0,
    us=0.1,
    v_wind=15.4, v_wind_base=15.0,
    tether_length=150.0, d_tether=nothing,
    elevation=nothing,
    g_earth=9.81,
    kcu_mass=nothing,
    stop_window_sec=COURSE_RATE_WINDOW_SEC,
    stop_rel_tol=0.03,
    stop_check_every=nothing,
    early_stop=true,
    remake_settle=IS_REMAKE_SETTLE,
    visualize_settle=IS_VISUALIZE_SETTLE,
    debug_on_failure=IS_DEBUG_ON_FAILURE,
    replay_on_failure=IS_REPLAY_ON_FAILURE,
    save_subdir="", run_tag="")

    global sam, _settle_log

    elev_deg = isnothing(elevation) ? 70.0 : float(elevation)
    elev_rad = deg2rad(elev_deg)
    position = [tether_length * cos(elev_rad), 0.0,
        tether_length * sin(elev_rad)]
    wind_vec = wind_vec_from_angles(
        v_wind, deg2rad(-90.0), 0.0)

    settle_config = V3SettleConfig(
        source_struc_path=DEFAULT_STRUC_YAML_PATH,
        source_aero_path=DEFAULT_AERO_YAML_PATH,
        vsm_settings_path=DEFAULT_VSM_SETTINGS_PATH,
        v_wind=v_wind,
        tether_length=tether_length,
        d_tether=d_tether,
        g_earth=g_earth,
        kcu_mass=kcu_mass,
        body_damping=body_damping .* 2.0,
        body_damping_delta=(body_damping_delta[1],
            body_damping_delta[2] .* 2.0),
        geom=V3GeomAdjustConfig(tether_length=tether_length),
        num_steps=1500, num_substeps=5, dt=0.001,
        decay_steps=1200,
        start_depower=23.5,
        course_correction_gain=0.02,
        course_correction_mode=:heading,
        initial_damping=100.0,
    )
    sam, _settle_log, settle_failed = settle_wing(
        settle_config;
        position=position,
        velocity=[0.0, 0.0, 0.0],
        heading=0.0,
        steering=0.0, depower=udp,
        wind_vec=wind_vec,
        remake=remake_settle)
    if visualize_settle
        if isnothing(_settle_log)
            @warn "No settle log to visualize. Set remake_settle=true if cached settled geometry was reused."
        else
            fig = plot(something(sam).sys_struct,
                _settle_log; plot_tether=true)
            display(fig)

            scene = replay(_settle_log,
                something(sam).sys_struct)
            display(scene)
        end
    end
    settle_failed && error(
        "settle_wing failed for elevation=$elev_deg, " *
        "v_wind=$v_wind, lt=$tether_length")
    sys = something(sam).sys_struct

    set_v3_body_damping!(sys, body_damping,
        body_damping_delta)

    @assert !isnothing(sys.vsm_set) "sys.vsm_set is missing"
    for ws in sys.vsm_set.wings
        ws.use_prior_polar = true
    end
    for wing in sys.wings
        wing.vsm_wing.use_prior_polar = true
    end

    n_c = (sim_time_circles > 0 && fps_circles > 0) ?
          max(1, Int(round(fps_circles * sim_time_circles))) : 0
    dt_c = n_c > 0 ? sim_time_circles / n_c : 0.0
    n_c > 0 || throw(ArgumentError(
        "Circular phase disabled. Set positive sim_time_circles and fps_circles."))
    save_root = "processed_data"
    save_dir = isempty(save_subdir) ? save_root :
               joinpath(save_root, save_subdir)
    isdir(save_dir) || mkpath(save_dir)
    logger, sys_state = create_logger(something(sam), n_c)

    course_uw = Vector{Float64}(undef, n_c)
    azimuth_uw = Vector{Float64}(undef, n_c)
    elevation_v = Vector{Float64}(undef, n_c)
    time_buf = Vector{Float64}(undef, n_c)
    n_logged = 0
    check_every = something(stop_check_every,
        max(1, fps_circles))

    nom_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    nom_right = sys.segments[V3_STEERING_RIGHT_IDX].l0

    sim_start = time()

    @info "Circular phase" n_c dt_c

    sys.winches[1].brake = true
    sys.winches[1].set_value = 0.0

    steer_change = V3_STEERING_GAIN * us
    vw_change = v_wind - v_wind_base
    steer_target_left = nom_left + steer_change
    steer_target_right = nom_right - steer_change
    steer_start_left =
        sys.segments[V3_STEERING_LEFT_IDX].l0
    steer_start_right =
        sys.segments[V3_STEERING_RIGHT_IDX].l0

    for step in 1:n_c
        t = step * dt_c

        rf = ramp_factor(t, 0.0, ramp_time_us)

        sys.segments[V3_STEERING_LEFT_IDX].l0 =
            steer_start_left +
            (steer_target_left - steer_start_left) * rf
        sys.segments[V3_STEERING_RIGHT_IDX].l0 =
            steer_start_right +
            (steer_target_right - steer_start_right) * rf

        sys.set.v_wind = v_wind_base + vw_change * rf

        try
            if !sim_step!(something(sam);
                set_values=[0.0], dt=dt_c, vsm_interval=1)
                @error "Circular phase failed" step
                _save_circle_failure_artifacts!(
                    logger, sys_state, something(sam),
                    step, t, nothing;
                    save_dir, run_tag, debug_on_failure,
                    replay_on_failure)
                break
            end
        catch err
            _save_circle_failure_artifacts!(
                logger, sys_state, something(sam),
                step, t, err;
                save_dir, run_tag, debug_on_failure,
                replay_on_failure)
            rethrow(err)
        end
        log_state!(logger, sys_state, something(sam), t)

        n_logged += 1
        c_raw = sys_state.course
        az_raw = sys_state.azimuth
        if n_logged == 1
            course_uw[1] = c_raw
            azimuth_uw[1] = az_raw
        else
            course_uw[n_logged] = _unwrap_step(
                course_uw[n_logged-1], c_raw)
            azimuth_uw[n_logged] = _unwrap_step(
                azimuth_uw[n_logged-1], az_raw)
        end
        elevation_v[n_logged] = sys_state.elevation
        time_buf[n_logged] = t

        if early_stop && n_logged > 1 &&
           t >= stop_window_sec &&
           (step % check_every == 0)
            if course_rate_converged(course_uw, azimuth_uw,
                elevation_v, time_buf, n_logged;
                window_sec=stop_window_sec,
                rel_tol=stop_rel_tol, dt=dt_c)
                @info "Course rate converged - stopping early" t = round(t, digits=2) step n_c
                break
            end
        end
    end

    report_performance(sim_time_circles, time() - sim_start)

    lt_tag = Int(round(tether_length))
    tmp_name = "tmp_run_refine_lt_$(lt_tag)"
    @info "Saving temporary log" name = tmp_name
    save_log(logger, tmp_name)
    syslog = load_log(tmp_name)

    ts = Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
    udp_t = Int(round(udp * 100))
    us_t = Int(round(us * 100))
    vw_t = Int(round(v_wind))
    dt_t = isnothing(d_tether) ? "yaml" :
           Int(round(d_tether * 10))
    el_t = elevation !== nothing ?
           Int(round(elevation)) : "yaml"
    g_t = g_earth !== nothing ?
          Int(round(g_earth * 10)) : "yaml"
    ln = "circles__udp_$(udp_t)_us_$(us_t)" *
         "_vw_$(vw_t)_lt_$(lt_tag)" *
         "_dt_$(dt_t)" *
         "_el_$(el_t)_g_$(g_t)"
    if !isempty(run_tag)
        ln *= "_" * run_tag
    end
    ln *= "_date_" * ts
    @info "Saving run log" name = ln path = save_dir
    save_log(logger, ln; path=save_dir)

    return syslog, something(sam)
end

# =============================================================================
# Batch sweep 1: 2019 kite parameters
# =============================================================================
defaults = (
    elevation=35, g_earth=0.0,
    us=0.05, udp=0.235, vw=8.4, lt=269,
    d_tether=10, kcu_mass=22.0,)
sweeps = nothing
# combine_all = (
#     us=[0.05, 0.1, 0.15],
#     udp=[0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42],
# )
# #TODO: when you are done you will still need to do:
# udp = [0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43]
# us = 0.075
#combine_all = (
#    us=[0.05, 0.1, 0.125, 0.15],
#    udp=[0.27, 0.31, 0.35, 0.39, 0.43],
#)
combine_all = (
    us=[0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18],
    udp=[0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43],
)

function generate_run_combos(defaults::NamedTuple,
    sweeps=nothing, combine_all=nothing)
    sw = sweeps === nothing ? NamedTuple() : sweeps
    ca = combine_all === nothing ? NamedTuple() : combine_all
    variants = NamedTuple[defaults]
    for param in keys(sw)
        for v in sw[param]
            cand = merge(defaults,
                NamedTuple{(param,)}((v,)))
            isequal(cand, defaults) && continue
            any(isequal(cand), variants) && continue
            push!(variants, cand)
        end
    end
    isempty(ca) && return variants
    ca_keys = keys(ca)
    ca_iter = Iterators.product(
        (ca[k] for k in ca_keys)...)
    combos = NamedTuple[]
    for cav in ca_iter
        ca_nt = NamedTuple{ca_keys}(cav)
        for var in variants
            push!(combos, merge(var, ca_nt))
        end
    end
    return combos
end

batch_dir = joinpath("processed_data", batch_tag)
isdir(batch_dir) || mkpath(batch_dir)
@info "Batch output directory" batch_dir

sim_time_circles = 200
ramp_time_us = 2

fps_circles = 200
body_damping = [0.0, 0.0, 20.0]
body_damping_delta = ([37, 38], [0.0, 20.0, 20.0])
remake_settle = IS_REMAKE_SETTLE
visualize_settle = IS_VISUALIZE_SETTLE
debug_on_failure = true
replay_on_failure = false

combos = generate_run_combos(defaults, sweeps, combine_all)
@info "Batch combos generated" n = length(combos)

const failed_runs = NamedTuple[]

for (run_id, p) in enumerate(combos)
    run_tag = "run_" * lpad(string(run_id), 3, '0')
    @info "Starting run" run_id elevation = p.elevation g_earth = p.g_earth us = p.us udp = p.udp vw = p.vw lt = p.lt d_tether = p.d_tether kcu_mass = p.kcu_mass
    try
        run_circles(;
            v_wind=p.vw, v_wind_base=p.vw,
            udp=p.udp, tether_length=p.lt,
            d_tether=p.d_tether,
            elevation=p.elevation, g_earth=p.g_earth,
            kcu_mass=p.kcu_mass,
            body_damping, body_damping_delta,
            sim_time_circles, fps_circles,
            ramp_time_us, us=p.us,
            remake_settle,
            visualize_settle,
            debug_on_failure,
            replay_on_failure,
            save_subdir=batch_tag,
            run_tag)
        @info "Completed" run_id
    catch err
        @error "Failed" run_id err
        push!(failed_runs, merge((run_id=run_id,), p,
            (error=err,)))
    end
    GC.gc()
end

if !isempty(failed_runs)
    fp = joinpath(batch_dir, "failed_runs.txt")
    open(fp, "w") do io
        for fr in failed_runs
            println(io, "Run $(fr.run_id): " *
                        "el=$(fr.elevation), g=$(fr.g_earth), " *
                        "us=$(fr.us), udp=$(fr.udp), " *
                        "vw=$(fr.vw), lt=$(fr.lt), " *
                        "d_tether=$(fr.d_tether), " *
                        "kcu_mass=$(fr.kcu_mass)")
            println(io, "  Error: $(fr.error)")
        end
    end
    @info "Wrote failure list" path = fp
end

@info "Batch 1 completed" total = length(combos) failed = length(failed_runs)


# =============================================================================
# Batch sweep 1: 2025 kite parameters
# =============================================================================
defaults = (
    elevation=35, g_earth=0.0,
    us=0.05, udp=0.235, vw=7.6, lt=271,
    d_tether=13.5, kcu_mass=23.3,
)
sweeps = nothing
# combine_all = (
#     us=[0.05, 0.1, 0.15],
#     udp=[0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42],
# )
# #TODO: when you are done you will still need to do:
# udp = [0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43]
# us = 0.075
#combine_all = (
#    us=[0.05, 0.1, 0.125, 0.15],
#    udp=[0.27, 0.31, 0.35, 0.39, 0.43],
#)
combine_all = (
    us=[0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18],
    udp=[0.19, 0.23, 0.27, 0.31, 0.35, 0.39, 0.43],
)
function generate_run_combos(defaults::NamedTuple,
    sweeps=nothing, combine_all=nothing)
    sw = sweeps === nothing ? NamedTuple() : sweeps
    ca = combine_all === nothing ? NamedTuple() : combine_all
    variants = NamedTuple[defaults]
    for param in keys(sw)
        for v in sw[param]
            cand = merge(defaults,
                NamedTuple{(param,)}((v,)))
            isequal(cand, defaults) && continue
            any(isequal(cand), variants) && continue
            push!(variants, cand)
        end
    end
    isempty(ca) && return variants
    ca_keys = keys(ca)
    ca_iter = Iterators.product(
        (ca[k] for k in ca_keys)...)
    combos = NamedTuple[]
    for cav in ca_iter
        ca_nt = NamedTuple{ca_keys}(cav)
        for var in variants
            push!(combos, merge(var, ca_nt))
        end
    end
    return combos
end

batch_dir = joinpath("processed_data", batch_tag)
isdir(batch_dir) || mkpath(batch_dir)
@info "Batch output directory" batch_dir

sim_time_circles = 200
ramp_time_us = 2

fps_circles = 200
body_damping = [0.0, 0.0, 20.0]
point_37_38_damping = [0.0, 20.0, 20.0]
remake_settle = IS_REMAKE_SETTLE
visualize_settle = IS_VISUALIZE_SETTLE
debug_on_failure = true
replay_on_failure = false

combos = generate_run_combos(defaults, sweeps, combine_all)
@info "Batch combos generated" n = length(combos)

const failed_runs = NamedTuple[]

for (run_id, p) in enumerate(combos)
    run_tag = "run_" * lpad(string(run_id), 3, '0')
    @info "Starting run" run_id elevation = p.elevation g_earth = p.g_earth us = p.us udp = p.udp vw = p.vw lt = p.lt d_tether = p.d_tether kcu_mass = p.kcu_mass
    try
        run_circles(;
            v_wind=p.vw, v_wind_base=p.vw,
            udp=p.udp, tether_length=p.lt,
            d_tether=p.d_tether,
            elevation=p.elevation, g_earth=p.g_earth,
            kcu_mass=p.kcu_mass,
            body_damping, point_37_38_damping,
            sim_time_circles, fps_circles,
            ramp_time_us, us=p.us,
            remake_settle,
            visualize_settle,
            debug_on_failure,
            replay_on_failure,
            save_subdir=batch_tag,
            run_tag)
        @info "Completed" run_id
    catch err
        @error "Failed" run_id err
        push!(failed_runs, merge((run_id=run_id,), p,
            (error=err,)))
    end
    GC.gc()
end

if !isempty(failed_runs)
    fp = joinpath(batch_dir, "failed_runs.txt")
    open(fp, "w") do io
        for fr in failed_runs
            println(io, "Run $(fr.run_id): " *
                        "el=$(fr.elevation), g=$(fr.g_earth), " *
                        "us=$(fr.us), udp=$(fr.udp), " *
                        "vw=$(fr.vw), lt=$(fr.lt), " *
                        "d_tether=$(fr.d_tether), " *
                        "kcu_mass=$(fr.kcu_mass)")
            println(io, "  Error: $(fr.error)")
        end
    end
    @info "Wrote failure list" path = fp
end

@info "Batch 1 completed" total = length(combos) failed = length(failed_runs)
