# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Run for Zenith Hold + Circular Flight

Runs multiple parameter combinations for the two-phase
v3 kite simulation (zenith azimuth hold, then circular
flight). Each run saves a permanent log with parameter
tags in the filename.

Usage:
    julia --project=examples examples/batch_run_zenith_then_circles.jl
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX,
    V3_DEPOWER_IDX, V3_STEERING_GAIN
using SymbolicAWEModels
using KiteUtils: wind_vec_from_angles
using GLMakie
using LinearAlgebra
using DiscretePIDs
using Dates

# =============================================================================
# Two-phase simulation function
# =============================================================================

"""
    run_zenith_circles(; kwargs...) -> (syslog, sam, azimuth_setpoint)

Run a two-phase v3 kite simulation: zenith azimuth hold
followed by circular flight.
"""
function run_zenith_circles(;
        sim_time_zenith = 10.0, fps_zenith = 1,
        sim_time_circles = 0.0, fps_circles = 1,
        body_sim_damping = [0.0, 0.0, 20.0],
        up = 0.4,
        ramp_time_us = 25.0,
        max_us_zenith = 0.1, us = 0.1,
        v_wind = 15.4, v_wind_base = 15.0,
        heading_p = 0.0, heading_i = 0.1, heading_d = 0.0,
        winch_p = 1000.0, winch_i = 100.0, winch_d = 50.0,
        target_azimuth = 0.0,
        tether_length = 150.0, elevation = nothing,
        g_earth = 9.81,
        kcu_mass = nothing,
        save_subdir = "", run_tag = ""
    )

    global sam, _settle_log

    elev_deg = isnothing(elevation) ? 70.0 : float(elevation)
    elev_rad = deg2rad(elev_deg)
    position = [
        tether_length * cos(elev_rad), 0.0,
        tether_length * sin(elev_rad),
    ]
    wind_vec = wind_vec_from_angles(
        v_wind, deg2rad(-90.0), 0.0
    )

    kite_set = load_kite(PROJECT)
    kite_set.aero_mode = AERO_MODE
    kite_set.geom = V3GeomAdjustConfig(
        reduce_te = true, tether_length = tether_length)
    kite_set.body_sim_damping = body_sim_damping
    settle_config = V3SettleConfig(
        project = PROJECT,
        kite_set = kite_set,
        v_wind = v_wind,
        tether_length = tether_length,
        g_earth = g_earth,
        kcu_mass = kcu_mass,
        body_start_damping = [0.0, 0.0, 40.0],
        decay_steps = 30,
        num_steps = 40, num_substeps = 1, dt = 0.05,
        start_depower = 40.0,
        course_correction_gain = 0.0,
        course_correction_mode = :heading,
        world_start_damping = 0.0,
    )
    sam, _settle_log, settle_failed = settle_wing(
        settle_config;
        position = position,
        velocity = [0.0, 0.0, 0.0],
        heading = 0.0,
        steering = 0.0, depower = up,
        wind_vec = wind_vec
    )
    settle_failed && error(
        "settle_wing failed for elevation=$elev_deg, " *
            "v_wind=$v_wind, lt=$tether_length"
    )
    sys = sam.sys_struct

    @assert !isnothing(sys.vsm_set) "sys.vsm_set is missing"
    for ws in sys.vsm_set.wings
        ws.use_prior_polar = true
    end
    for wing in sys.wings
        wing.vsm_wing.use_prior_polar = true
    end

    # Logger for both phases; zenith phase is truly optional
    n_z = (sim_time_zenith > 0 && fps_zenith > 0) ?
        max(1, Int(round(fps_zenith * sim_time_zenith))) : 0
    dt_z = n_z > 0 ? sim_time_zenith / n_z : 0.0
    n_c = (sim_time_circles > 0 && fps_circles > 0) ?
        max(1, Int(round(fps_circles * sim_time_circles))) : 0
    dt_c = n_c > 0 ? sim_time_circles / n_c : 0.0
    (n_z + n_c) > 0 || throw(
        ArgumentError(
            "Both phases are disabled. Set positive sim_time/fps for zenith or circles."
        )
    )
    logger, sys_state = create_logger(sam, n_z + n_c)

    # Nominal segment lengths
    nom_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    nom_right = sys.segments[V3_STEERING_RIGHT_IDX].l0

    azimuth_setpoint = Float64[target_azimuth]

    # Phase 1: Zenith hold
    sim_start = time()

    if n_z > 0
        @info "Zenith phase" n_z dt_z

        # Azimuth PID
        max_steering = max_us_zenith * V3_STEERING_GAIN
        azimuth_pid = create_heading_pid(;
            K = heading_p > 0 ? heading_p : 1.0,
            Ti = heading_i > 0 ? 1.0 / heading_i : false,
            Td = heading_d > 0 ? heading_d : false,
            dt = dt_z, umin = -abs(max_steering),
            umax = abs(max_steering)
        )

        # Winch PID
        nominal_tether_length = sys.tethers[1].len
        init_winch_torque!(sys)
        winch_pid = create_winch_pid(;
            K = winch_p,
            Ti = winch_i > 0 ? winch_p / winch_i : false,
            Td = winch_d > 0 ? winch_d / winch_p : false,
            dt = dt_z
        )

        for step in 1:n_z
            t = step * dt_z

            # Azimuth PID
            azimuth = sys.wings[1].azimuth
            steer_ctrl = azimuth_pid(
                target_azimuth, azimuth, 0.0
            )
            push!(azimuth_setpoint, target_azimuth)

            # Steering
            sys.segments[V3_STEERING_LEFT_IDX].l0 =
                nom_left + steer_ctrl
            sys.segments[V3_STEERING_RIGHT_IDX].l0 =
                nom_right - steer_ctrl

            # Winch PID
            tl = sys.tethers[1].len
            wf = winch_pid(nominal_tether_length, tl, 0.0)
            wt = force_to_torque(wf, sys)
            sys.winches[1].set_value = -wt

            if !sim_step!(
                    sam;
                    set_values = [-wt], dt = dt_z, vsm_interval = VSM_INTERVAL
                )
                @error "Zenith phase failed" step
                break
            end
            log_state!(logger, sys_state, sam, t)
        end
    else
        @info "Zenith phase skipped" sim_time_zenith fps_zenith
    end

    # Phase 2: Circular flight
    if n_c > 0
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
            t = sim_time_zenith + step * dt_c
            pt = step * dt_c

            rf = ramp_factor(pt, 0.0, ramp_time_us)

            sys.segments[V3_STEERING_LEFT_IDX].l0 =
                steer_start_left +
                (steer_target_left - steer_start_left) * rf
            sys.segments[V3_STEERING_RIGHT_IDX].l0 =
                steer_start_right +
                (steer_target_right - steer_start_right) * rf

            sys.set.v_wind = v_wind_base + vw_change * rf

            if !sim_step!(
                    sam;
                    set_values = [0.0], dt = dt_c, vsm_interval = VSM_INTERVAL
                )
                @error "Circular phase failed" step
                break
            end
            log_state!(logger, sys_state, sam, t)
        end
    else
        @info "Circular phase skipped" sim_time_circles fps_circles
    end

    # Performance
    total_sim = sim_time_zenith + sim_time_circles
    report_performance(total_sim, time() - sim_start)

    # Save
    lt_tag = Int(round(tether_length))
    save_log(logger, "tmp_run_particle_dynamics_lt_$(lt_tag)")
    syslog = load_log("tmp_run_particle_dynamics_lt_$(lt_tag)")

    save_root = "processed_data"
    save_dir = isempty(save_subdir) ? save_root :
        joinpath(save_root, save_subdir)
    isdir(save_dir) || mkpath(save_dir)
    ts = Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
    up_t = Int(round(up * 100))
    us_t = Int(round(us * 100))
    vw_t = Int(round(v_wind))
    el_t = elevation !== nothing ?
        Int(round(elevation)) : "yaml"
    g_t = g_earth !== nothing ?
        Int(round(g_earth * 10)) : "yaml"
    run_prefix = n_c > 0 ?
        "hold_at_zenith_then_circles" : "zenith_circle"
    ln = "$(run_prefix)__up_$(up_t)_us_$(us_t)" *
        "_vw_$(vw_t)_lt_$(lt_tag)" *
        "_el_$(el_t)_g_$(g_t)"
    if !isempty(run_tag)
        ln *= "_" * run_tag
    end
    ln *= "_date_" * ts
    save_log(logger, ln; path = save_dir)

    return syslog, sam, azimuth_setpoint
end

# =============================================================================
# Batch sweep 1: 2019 kite parameters
# =============================================================================

# elevation_vals = [20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85]
elevation_vals = [70]
g_earth_vals = [0.0]
us_vals = [0.1] #, 0.1, 0.15, 0.2]
up_vals = [0.18]
# vw_vals = [8.6, 19.8]
vw_vals = [8.6]
lt_vals = [268]
kcu_mass_2019 = 22.0
kcu_mass_2025 = 23.3
kcu_mass_vals = [kcu_mass_2019]
max_us_zenith = 0.02 # maximum allowed steering to keep it on zenith
batch_tag = "zenith_2019_batch_" *
    Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
batch_dir = joinpath("processed_data", batch_tag)
isdir(batch_dir) || mkpath(batch_dir)

sim_time_zenith = 2
sim_time_circles = 200
ramp_time_us = 2

fps_zenith = 20
fps_circles = 20
VSM_INTERVAL = 1   # steps between VSM aero solves
body_sim_damping = [0.0, 0.0, 20.0]
PROJECT = "system_psm.yaml"   # project file: geometry, settings and kite
AERO_MODE = ContinuousAero()

const failed_runs = NamedTuple[]

for (run_id, (elev, g, us, up, vw, lt, kcu_mass_val)) in enumerate(
        Iterators.product(
            elevation_vals, g_earth_vals,
            us_vals, up_vals, vw_vals, lt_vals, kcu_mass_vals
        )
    )
    run_tag = "run_" * lpad(string(run_id), 3, '0')
    @info "Starting run" run_id elevation = elev g_earth = g us up vw lt kcu_mass = kcu_mass_val
    try
        run_zenith_circles(;
            v_wind = vw, v_wind_base = vw,
            up = up, tether_length = lt,
            elevation = elev, g_earth = g,
            kcu_mass = kcu_mass_val,
            sim_time_zenith, fps_zenith,
            body_sim_damping,
            max_us_zenith, target_azimuth = 0.0,
            sim_time_circles, fps_circles,
            ramp_time_us, us = us,
            save_subdir = batch_tag,
            run_tag
        )
        @info "Completed" run_id
    catch err
        @error "Failed" run_id err
        push!(
            failed_runs, (
                run_id = run_id,
                elevation = elev, g_earth = g,
                us = us, up = up, vw = vw, lt = lt,
                kcu_mass = kcu_mass_val, error = err,
            )
        )
    end
    GC.gc()
end

if !isempty(failed_runs)
    fp = joinpath(batch_dir, "failed_runs.txt")
    open(fp, "w") do io
        for fr in failed_runs
            println(
                io, "Run $(fr.run_id): " *
                    "el=$(fr.elevation), g=$(fr.g_earth), " *
                    "us=$(fr.us), up=$(fr.up), " *
                    "vw=$(fr.vw), lt=$(fr.lt), " *
                    "kcu_mass=$(fr.kcu_mass)"
            )
            println(io, "  Error: $(fr.error)")
        end
    end
    @info "Wrote failure list" path = fp
end

n_total = length(
    collect(
        Iterators.product(
            elevation_vals, g_earth_vals,
            us_vals, up_vals, vw_vals, lt_vals, kcu_mass_vals
        )
    )
)
@info "Batch 1 completed" total = n_total failed = length(failed_runs)

#TODO: check updates from above to complete the below when you start using it
# # =============================================================================
# # Batch sweep 2: 2025 kite parameters
# # =============================================================================

# elevation_vals = [
#     20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85]
# g_earth_vals = [0.0]
# us_vals = [0.0]
# up_vals = [0.42]
# vw_vals = [7.8, 19.7]
# lt_vals = [262]

# batch_tag = "zenith_2025_batch_" *
#             Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")

# sim_time_zenith = 200.0
# failed_runs = NamedTuple[]

# for (run_id, (elev, g, us, up, vw, lt)) in enumerate(
#     Iterators.product(elevation_vals, g_earth_vals,
#         us_vals, up_vals, vw_vals, lt_vals))
#     run_tag = "run_" * lpad(string(run_id), 3, '0')
#     @info "Starting run" run_id elevation = elev g_earth = g us up vw lt
#     try
#         run_zenith_circles(;
#             v_wind=vw, v_wind_base=vw,
#             up=up, tether_length=lt,
#             elevation=elev, g_earth=g,
#             sim_time_zenith, fps_zenith,
#             start_ramp_time, ramp_time_up,
#             startup_damping_pattern, damping_pattern,
#             startup_decay_time,
#             max_us_zenith, target_azimuth=0.0,
#             sim_time_circles, fps_circles,
#             ramp_time_us, us=us,
#             save_subdir=batch_tag, run_tag)
#         @info "Completed" run_id
#     catch err
#         @error "Failed" run_id err
#         push!(failed_runs, (run_id=run_id,
#             elevation=elev, g_earth=g,
#             us=us, up=up, vw=vw, lt=lt, error=err))
#     end
#     GC.gc()
# end

# if !isempty(failed_runs)
#     fp = joinpath("processed_data",
#         batch_tag, "failed_runs.txt")
#     open(fp, "w") do io
#         for fr in failed_runs
#             println(io, "Run $(fr.run_id): " *
#                         "el=$(fr.elevation), g=$(fr.g_earth), " *
#                         "us=$(fr.us), up=$(fr.up), " *
#                         "vw=$(fr.vw), lt=$(fr.lt)")
#             println(io, "  Error: $(fr.error)")
#         end
#     end
#     @info "Wrote failure list" path = fp
# end

# n_total = length(collect(Iterators.product(
#     elevation_vals, g_earth_vals,
#     us_vals, up_vals, vw_vals, lt_vals)))
# @info "Batch 2 completed" total = n_total failed = length(failed_runs)
