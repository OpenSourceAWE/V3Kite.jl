# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Run for Circular Flight

Runs multiple parameter combinations for the v3 kite
circular-flight simulation. Each run saves a permanent log
with parameter tags in the filename. Initial equilibrium is
established by `settle_wing`.

Usage:
    julia --project=examples examples/batch_run_circles.jl
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
using Dates

# =============================================================================
# Circular flight simulation function
# =============================================================================

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
    point_37_38_damping=[0.0, 20.0, 20.0],
    up=0.4,
    ramp_time_us=25.0,
    us=0.1,
    v_wind=15.4, v_wind_base=15.0,
    tether_length=150.0, elevation=nothing,
    g_earth=9.81,
    kcu_mass=nothing,
    save_subdir="", run_tag="")

    global sam, _settle_log

    elev_deg = isnothing(elevation) ? 70.0 : float(elevation)
    elev_rad = deg2rad(elev_deg)
    position = [tether_length * cos(elev_rad), 0.0,
                tether_length * sin(elev_rad)]
    wind_vec = wind_vec_from_angles(
        v_wind, deg2rad(-90.0), 0.0)

    settle_config = V3SettleConfig(
        source_struc_path="struc_geometry.yaml",
        source_aero_path="aero_geometry.yaml",
        vsm_settings_path="vsm_settings.yaml",
        v_wind=v_wind,
        tether_length=tether_length,
        g_earth=g_earth,
        kcu_mass=kcu_mass,
        body_damping=body_damping .* 2.0,
        body_damping_overrides=[
            (37:38, point_37_38_damping .* 2.0)],
        geom=V3GeomAdjustConfig(
            reduce_te=true, tether_length=tether_length),
        num_steps=400, num_substeps=5, dt=0.001,
        start_depower=40.0,
        course_correction_gain=0.0,
        course_correction_mode=:heading,
        world_damping=0.0, min_damping=0.0,
    )
    sam, _settle_log, settle_failed = settle_wing(
        settle_config;
        position=position,
        velocity=[0.0, 0.0, 0.0],
        heading=0.0,
        steering=0.0, depower=up,
        wind_vec=wind_vec,
        remake=false)
    settle_failed && error(
        "settle_wing failed for elevation=$elev_deg, " *
        "v_wind=$v_wind, lt=$tether_length")
    sys = sam.sys_struct

    set_v3_body_damping!(sys, body_damping,
                         point_37_38_damping)

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
    logger, sys_state = create_logger(sam, n_c)

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

        if !sim_step!(sam;
            set_values=[0.0], dt=dt_c, vsm_interval=1)
            @error "Circular phase failed" step
            break
        end
        log_state!(logger, sys_state, sam, t)

        if should_report(step, n_c)
            @info "Circular step $step/$n_c" t=round(t, digits=2) v_wind=round(sys.set.v_wind, digits=2) rf=round(rf, digits=3)
        end
    end

    report_performance(sim_time_circles, time() - sim_start)

    lt_tag = Int(round(tether_length))
    tmp_name = "tmp_run_refine_lt_$(lt_tag)"
    @info "Saving temporary log" name=tmp_name
    save_log(logger, tmp_name)
    syslog = load_log(tmp_name)

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
    ln = "circles__up_$(up_t)_us_$(us_t)" *
         "_vw_$(vw_t)_lt_$(lt_tag)" *
         "_el_$(el_t)_g_$(g_t)"
    if !isempty(run_tag)
        ln *= "_" * run_tag
    end
    ln *= "_date_" * ts
    @info "Saving run log" name=ln path=save_dir
    save_log(logger, ln; path=save_dir)

    return syslog, sam
end

# =============================================================================
# Batch sweep 1: 2019 kite parameters
# =============================================================================

# elevation_vals = [20, 25, 30, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85]
elevation_vals = [70]
g_earth_vals = [0.0, 9.81]
us_vals = [0.1, 0.15, 0.2]
up_vals = [0.2, 0.3, 0.4]
# vw_vals = [8.6, 19.8]
vw_vals = [8.6]
lt_vals = [268]
kcu_mass_2019 = 22.0
kcu_mass_2025 = 23.3
kcu_mass_vals = [kcu_mass_2019]
batch_tag = "circles_2019_batch_" *
            Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
batch_dir = joinpath("processed_data", batch_tag)
isdir(batch_dir) || mkpath(batch_dir)
@info "Batch output directory" batch_dir

sim_time_circles = 200
ramp_time_us = 2

fps_circles = 200
body_damping = [0.0, 0.0, 20.0]
point_37_38_damping = [0.0, 20.0, 20.0]

failed_runs = NamedTuple[]

for (run_id, (elev, g, us, up, vw, lt, kcu_mass_val)) in enumerate(
    Iterators.product(elevation_vals, g_earth_vals,
        us_vals, up_vals, vw_vals, lt_vals, kcu_mass_vals))
    run_tag = "run_" * lpad(string(run_id), 3, '0')
    @info "Starting run" run_id elevation = elev g_earth = g us up vw lt kcu_mass = kcu_mass_val
    try
        run_circles(;
            v_wind=vw, v_wind_base=vw,
            up=up, tether_length=lt,
            elevation=elev, g_earth=g,
            kcu_mass=kcu_mass_val,
            body_damping, point_37_38_damping,
            sim_time_circles, fps_circles,
            ramp_time_us, us=us,
            save_subdir=batch_tag,
            run_tag)
        @info "Completed" run_id
    catch err
        @error "Failed" run_id err
        push!(failed_runs, (run_id=run_id,
            elevation=elev, g_earth=g,
            us=us, up=up, vw=vw, lt=lt,
            kcu_mass=kcu_mass_val, error=err))
    end
    GC.gc()
end

if !isempty(failed_runs)
    fp = joinpath(batch_dir, "failed_runs.txt")
    open(fp, "w") do io
        for fr in failed_runs
            println(io, "Run $(fr.run_id): " *
                        "el=$(fr.elevation), g=$(fr.g_earth), " *
                        "us=$(fr.us), up=$(fr.up), " *
                        "vw=$(fr.vw), lt=$(fr.lt), " *
                        "kcu_mass=$(fr.kcu_mass)")
            println(io, "  Error: $(fr.error)")
        end
    end
    @info "Wrote failure list" path = fp
end

n_total = length(collect(Iterators.product(
    elevation_vals, g_earth_vals,
    us_vals, up_vals, vw_vals, lt_vals, kcu_mass_vals)))
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

# batch_tag = "circles_2025_batch_" *
#             Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")

# failed_runs = NamedTuple[]

# for (run_id, (elev, g, us, up, vw, lt)) in enumerate(
#     Iterators.product(elevation_vals, g_earth_vals,
#         us_vals, up_vals, vw_vals, lt_vals))
#     run_tag = "run_" * lpad(string(run_id), 3, '0')
#     @info "Starting run" run_id elevation = elev g_earth = g us up vw lt
#     try
#         run_circles(;
#             v_wind=vw, v_wind_base=vw,
#             up=up, tether_length=lt,
#             elevation=elev, g_earth=g,
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
