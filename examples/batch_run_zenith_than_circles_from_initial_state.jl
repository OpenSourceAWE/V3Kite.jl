# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Run from User Geometry, Directly into Circles

This script starts from user-provided structural and aerodynamic geometry files,
without applying geometric modifications in code (no tip/TE reduction, no tether
point repositioning, no elevation reset).

Only circular-flight simulation is run (no zenith phase).

Usage:
    julia --project=. examples/batch_run_zenith_than_circles_from_initial_state.jl
"""

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX, V3_DEPOWER_IDX, V3_STEERING_GAIN
using SymbolicAWEModels
using VortexStepMethod
using Dates
using Serialization

function resolve_input_path(path::String, data_root::String)
    if isabspath(path)
        return path
    end
    if isfile(path)
        return abspath(path)
    end
    return joinpath(data_root, path)
end

function resolve_snapshot_path(path::String, data_root::String)
    if isabspath(path)
        return path
    end
    if isfile(path)
        return abspath(path)
    end
    return joinpath(data_root, path)
end

function load_from_snapshot!(sam, sys; snapshot_path::String)
    data_root = v3_data_path()
    snapshot_full = resolve_snapshot_path(snapshot_path, data_root)
    isfile(snapshot_full) || error("Snapshot file not found: $snapshot_full")

    payload = open(snapshot_full, "r") do io
        deserialize(io)
    end
    haskey(payload, :format_version) || error("Invalid snapshot format (missing format_version)")

    point_by_idx = Dict(Int(p.idx) => p for p in sys.points)
    for ps in payload.points
        idx = Int(ps.idx)
        haskey(point_by_idx, idx) || continue
        p = point_by_idx[idx]
        p.pos_w .= ps.pos_w
        p.vel_w .= ps.vel_w
    end

    winch_by_idx = Dict(Int(w.idx) => w for w in sys.winches)
    for ws in payload.winches
        idx = Int(ws.idx)
        haskey(winch_by_idx, idx) || continue
        w = winch_by_idx[idx]
        w.tether_len = ws.tether_len
        w.tether_vel = ws.tether_vel
        w.set_value = ws.set_value
        w.brake = ws.brake
    end

    group_by_idx = Dict(Int(g.idx) => g for g in sys.groups)
    for gs in payload.groups
        idx = Int(gs.idx)
        haskey(group_by_idx, idx) || continue
        g = group_by_idx[idx]
        g.twist = gs.twist
        g.twist_ω = gs.twist_rate
    end

    wing_by_idx = Dict(Int(w.idx) => w for w in sys.wings)
    for ws in payload.wings
        idx = Int(ws.idx)
        haskey(wing_by_idx, idx) || continue
        w = wing_by_idx[idx]
        w.pos_w .= ws.pos_w
        w.vel_w .= ws.vel_w
        w.Q_b_to_w .= ws.Q_b_to_w
        w.ω_b .= ws.omega_b
        w.turn_rate .= ws.turn_rate
        w.elevation = ws.elevation
        w.azimuth = ws.azimuth
        w.heading = ws.heading
    end

    # Reinitialize integrator from the updated sys_struct state so startup uses the snapshot.
    SymbolicAWEModels.reinit!(sam, sam.prob, SymbolicAWEModels.FBDF())
    @info "Loaded state from snapshot" snapshot = snapshot_full
    return snapshot_full
end

function make_model_from_user_geometry(;
    struc_yaml_path::String,
    aero_yaml_path::String,
    vsm_settings_path::String,
    v_wind::Float64,
    upwind_dir::Float64,
    damping_pattern::Vector{Float64},
    n_panels::Int,
    g_earth::Union{Nothing,Float64})

    data_root = v3_data_path()
    set_data_path(data_root)

    struc_full = resolve_input_path(struc_yaml_path, data_root)
    aero_full = resolve_input_path(aero_yaml_path, data_root)
    vsm_full = resolve_input_path(vsm_settings_path, data_root)

    set = Settings("system.yaml")
    set.v_wind = v_wind
    set.upwind_dir = upwind_dir
    if g_earth !== nothing
        set.g_earth = float(g_earth)
    end

    vsm_set = VSMSettings(vsm_full; data_prefix=false)
    vsm_set.wings[1].geometry_file = aero_full
    vsm_set.wings[1].n_panels = n_panels

    sys = load_sys_struct_from_yaml(struc_full;
        system_name=V3_MODEL_NAME, set,
        wing_type=REFINE, vsm_set)

    SymbolicAWEModels.set_body_frame_damping(
        sys, damping_pattern)

    sam = SymbolicAWEModel(set, sys)
    init!(sam; remake=false, ignore_l0=false, remake_vsm=true)

    for ws in sys.vsm_set.wings
        ws.use_prior_polar = true
    end
    for wing in sys.wings
        wing.vsm_wing.use_prior_polar = true
    end

    return sam, sys, struc_full, aero_full, vsm_full
end

function run_circles_from_initial_state(;
    struc_yaml_path="struc_geometry.yaml",
    aero_yaml_path="aero_geometry.yaml",
    vsm_settings_path="vsm_settings.yaml",
    sim_time_circles=120.0,
    fps_circles=180,
    damping_pattern=[0.0, 0.0, 20.0],
    startup_damping_pattern=nothing,
    startup_decay_time=2.0,
    us=0.1,
    start_ramp_time=0.0,
    ramp_time_us=25.0,
    v_wind=8.4,
    v_wind_base=8.4,
    upwind_dir=-90.0,
    g_earth=nothing,
    kcu_mass=nothing,
    snapshot_path::Union{Nothing,String}=nothing,
    n_panels=36,
    save_subdir="",
    run_tag="")

    n_c = (sim_time_circles > 0 && fps_circles > 0) ?
          max(1, Int(round(fps_circles * sim_time_circles))) : 0
    n_c > 0 || throw(ArgumentError(
        "Circle phase is disabled. Set positive sim_time_circles and fps_circles."))
    dt_c = sim_time_circles / n_c

    startup_pattern = isnothing(startup_damping_pattern) ?
                      damping_pattern : startup_damping_pattern
    damping_profile = function (t)
        if startup_decay_time <= 0
            return damping_pattern
        end
        mix = clamp(t / startup_decay_time, 0.0, 1.0)
        return startup_pattern .+
               (damping_pattern .- startup_pattern) .* mix
    end

    sam, sys, struc_full, aero_full, _ = make_model_from_user_geometry(;
        struc_yaml_path,
        aero_yaml_path,
        vsm_settings_path,
        v_wind=v_wind_base,
        upwind_dir,
        damping_pattern=damping_profile(0.0),
        n_panels,
        g_earth)

    if kcu_mass !== nothing
        sys.points[1].extra_mass = float(kcu_mass)
    end
    if !isnothing(snapshot_path)
        load_from_snapshot!(sam, sys;
            snapshot_path=snapshot_path)
    end

    logger, sys_state = create_logger(sam, n_c)

    nom_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    nom_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
    depower_l0 = sys.segments[V3_DEPOWER_IDX].l0
    up_from_geometry = (1000.0 * depower_l0 - 200.0) / 5000.0
    steer_target_left = nom_left + V3_STEERING_GAIN * us
    steer_target_right = nom_right - V3_STEERING_GAIN * us
    steer_start_left = nom_left
    steer_start_right = nom_right
    vw_change = v_wind - v_wind_base

    sys.winches[1].brake = true
    sys.winches[1].set_value = 0.0

    lt_tag = Int(round(sys.winches[1].tether_len))
    save_root = "processed_data"
    save_dir = isempty(save_subdir) ? save_root :
               joinpath(save_root, save_subdir)
    isdir(save_dir) || mkpath(save_dir)
    ts = Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
    up_t = Int(round(up_from_geometry * 100))
    us_t = Int(round(us * 100))
    vw_t = Int(round(v_wind))
    g_t = g_earth !== nothing ? Int(round(g_earth * 10)) : "yaml"
    ln = "circles_from_initial_state__up_$(up_t)_us_$(us_t)" *
         "_vw_$(vw_t)_lt_$(lt_tag)_g_$(g_t)"
    if !isempty(run_tag)
        ln *= "_" * run_tag
    end
    ln *= "_date_" * ts

    sim_start = time()
    try
        @info "Circle phase" n_c dt_c
        @info "Geometry input files" struc = struc_full aero = aero_full
        for step in 1:n_c
            t = step * dt_c

            SymbolicAWEModels.set_body_frame_damping(
                sys, damping_profile(t))

            us_rf = ramp_factor(
                t,
                start_ramp_time,
                start_ramp_time + ramp_time_us)

            sys.segments[V3_STEERING_LEFT_IDX].l0 =
                steer_start_left +
                (steer_target_left - steer_start_left) * us_rf
            sys.segments[V3_STEERING_RIGHT_IDX].l0 =
                steer_start_right +
                (steer_target_right - steer_start_right) * us_rf
            sys.set.v_wind = v_wind_base + vw_change * us_rf

            if !sim_step!(sam; set_values=[0.0], dt=dt_c, vsm_interval=1)
                @error "Circular phase failed" step
                break
            end
            log_state!(logger, sys_state, sam, t)
        end

        report_performance(sim_time_circles, time() - sim_start)

        tmp_name = "tmp_circle_from_initial_state_" * ts
        save_log(logger, tmp_name)
        syslog = load_log(tmp_name)
        save_log(logger, ln; path=save_dir)
        return syslog, sam
    catch err
        failed_ln = ln * "_failed"
        @error "Run failed; saving partial log" run_tag err failed_log = failed_ln
        try
            save_log(logger, failed_ln; path=save_dir)
            @info "Saved partial failure log" path = joinpath(save_dir, failed_ln * ".arrow")
        catch save_err
            @error "Failed to save partial failure log" save_err
        end
        rethrow(err)
    end
end

# # # =============================================================================
# # # 2025
# # # =============================================================================
# # # =============================================================================
# # # Batch sweep (circle phase only)
# # # =============================================================================
# struc_yaml_input = "struc_geometry_initial_state_lt_271_vw_76_udp_042.yaml"
# aero_yaml_input = "aero_geometry_initial_state_lt_271_vw_76_udp_042.yaml"
# vsm_settings_input = "vsm_settings.yaml"
# snapshot_input = "initial_state_snapshot_lt_271_vw_76_udp_042.jls"

# g_earth_vals = [9.81]
# us_vals = [0.2] #[0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
# vw_vals = [7.6]
# kcu_mass_vals = [nothing] # keep KCU mass from YAML by default

# batch_tag = "circles_from_initial_state_2025_" *
#             Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
# batch_dir = joinpath("processed_data", batch_tag)
# isdir(batch_dir) || mkpath(batch_dir)

# sim_time_circles = 30
# fps_circles = 180
# start_ramp_time = 2.0
# ramp_time_us = 5.0
# startup_decay_time = 2.0
# startup_damping_pattern = [0.0, 0.0, 20.0]
# damping_pattern = [0.0, 0.0, 20.0]

# failed_runs = NamedTuple[]

# for (run_id, (g, us, vw, kcu_mass_val)) in enumerate(
#     Iterators.product(g_earth_vals, us_vals, vw_vals, kcu_mass_vals))
#     run_tag = "run_" * lpad(string(run_id), 3, '0')
#     @info "Starting run" run_id g_earth = g us vw kcu_mass = kcu_mass_val struc_yaml = struc_yaml_input aero_yaml = aero_yaml_input
#     try
#         run_circles_from_initial_state(;
#             struc_yaml_path=struc_yaml_input,
#             aero_yaml_path=aero_yaml_input,
#             vsm_settings_path=vsm_settings_input,
#             v_wind=vw,
#             v_wind_base=vw,
#             us=us,
#             g_earth=g,
#             kcu_mass=kcu_mass_val,
#             snapshot_path=snapshot_input,
#             sim_time_circles,
#             fps_circles,
#             start_ramp_time,
#             ramp_time_us,
#             startup_damping_pattern,
#             damping_pattern,
#             startup_decay_time,
#             save_subdir=batch_tag,
#             run_tag)
#         @info "Completed" run_id
#     catch err
#         @error "Failed" run_id err
#         push!(failed_runs, (run_id=run_id,
#             g_earth=g,
#             us=us,
#             vw=vw,
#             kcu_mass=kcu_mass_val,
#             struc_yaml=struc_yaml_input,
#             aero_yaml=aero_yaml_input,
#             error=err))
#     end
#     GC.gc()
# end

# if !isempty(failed_runs)
#     fp = joinpath(batch_dir, "failed_runs.txt")
#     open(fp, "w") do io
#         for fr in failed_runs
#             println(io, "Run $(fr.run_id): " *
#                         "g=$(fr.g_earth), " *
#                         "us=$(fr.us), " *
#                         "vw=$(fr.vw), " *
#                         "kcu_mass=$(fr.kcu_mass), " *
#                         "struc=$(fr.struc_yaml), " *
#                         "aero=$(fr.aero_yaml)")
#             println(io, "  Error: $(fr.error)")
#         end
#     end
#     @info "Wrote failure list" path = fp
# end

# n_total = length(collect(Iterators.product(
#     g_earth_vals, us_vals, vw_vals, kcu_mass_vals)))
# @info "Batch completed" total = n_total failed = length(failed_runs)

# =============================================================================
# 2019
# =============================================================================
# =============================================================================
# Batch sweep (circle phase only)
# =============================================================================
# struc_yaml_input = "struc_geometry_initial_state_lt_269_vw_84_udp_019.yaml"
# aero_yaml_input = "aero_geometry_initial_state_lt_269_vw_84_udp_019.yaml"
# vsm_settings_input = "vsm_settings.yaml"


# =============================================================================
#### 0.25
# =============================================================================
udp = 0.25
udp_str = lpad(string(Int(round(udp * 100))), 3, '0')
struc_yaml_input = "struc_geometry_initial_state_lt_269_vw_84_udp_$udp_str.yaml"
aero_yaml_input = "aero_geometry_initial_state_lt_269_vw_84_udp_$udp_str.yaml"
vsm_settings_input = "vsm_settings.yaml"
snapshot_input = "initial_state_snapshot_lt_271_vw_76_udp_$udp_str.jls"
g_earth_vals = [0.0]
us_vals = [0.2] #[0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
vw_vals = [8.4]
kcu_mass_vals = [nothing]

batch_tag = "2019_circles_udp_$udp_str" *
            Dates.format(Dates.now(), "__yyyy_mm_dd_HH_MM_SS")
batch_dir = joinpath("processed_data", batch_tag)
isdir(batch_dir) || mkpath(batch_dir)

sim_time_circles = 20
fps_circles = 420
start_ramp_time = 0.0
ramp_time_us = 3.0
startup_decay_time = sim_time_circles * 0.75
startup_damping_pattern = [0.0, 20.0, 20.0]
damping_pattern = [0.0, 6.0, 20.0]

failed_runs = NamedTuple[]

for (run_id, (g, us, vw, kcu_mass_val)) in enumerate(
    Iterators.product(g_earth_vals, us_vals, vw_vals, kcu_mass_vals))
    run_tag = "run_" * lpad(string(run_id), 3, '0')
    @info "Starting run" run_id g_earth = g us vw kcu_mass = kcu_mass_val struc_yaml = struc_yaml_input aero_yaml = aero_yaml_input
    try
        run_circles_from_initial_state(;
            struc_yaml_path=struc_yaml_input,
            aero_yaml_path=aero_yaml_input,
            vsm_settings_path=vsm_settings_input,
            v_wind=vw,
            v_wind_base=vw,
            us=us,
            g_earth=g,
            kcu_mass=kcu_mass_val,
            sim_time_circles,
            fps_circles,
            start_ramp_time,
            ramp_time_us,
            startup_damping_pattern,
            damping_pattern,
            startup_decay_time,
            save_subdir=batch_tag,
            run_tag)
        @info "Completed" run_id
    catch err
        @error "Failed" run_id err
        push!(failed_runs, (run_id=run_id,
            g_earth=g,
            us=us,
            vw=vw,
            kcu_mass=kcu_mass_val,
            struc_yaml=struc_yaml_input,
            aero_yaml=aero_yaml_input,
            error=err))
    end
    GC.gc()
end

if !isempty(failed_runs)
    fp = joinpath(batch_dir, "failed_runs.txt")
    open(fp, "w") do io
        for fr in failed_runs
            println(io, "Run $(fr.run_id): " *
                        "g=$(fr.g_earth), " *
                        "us=$(fr.us), " *
                        "vw=$(fr.vw), " *
                        "kcu_mass=$(fr.kcu_mass), " *
                        "struc=$(fr.struc_yaml), " *
                        "aero=$(fr.aero_yaml)")
            println(io, "  Error: $(fr.error)")
        end
    end
    @info "Wrote failure list" path = fp
end

n_total = length(collect(Iterators.product(
    g_earth_vals, us_vals, vw_vals, kcu_mass_vals)))
@info "Batch completed" total = n_total failed = length(failed_runs)


# # =============================================================================
# #### 0.30
# # =============================================================================
# udp = 0.30
# udp_str = lpad(string(Int(round(udp * 100))), 3, '0')
# struc_yaml_input = "struc_geometry_initial_state_lt_269_vw_84_udp_$udp_str.yaml"
# aero_yaml_input = "aero_geometry_initial_state_lt_269_vw_84_udp_$udp_str.yaml"
# vsm_settings_input = "vsm_settings.yaml"
# snapshot_input = "initial_state_snapshot_lt_271_vw_76_udp_$udp_str.jls"
# g_earth_vals = [0.0]
# us_vals = [0.2] #[0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
# vw_vals = [8.4]
# kcu_mass_vals = [nothing]

# batch_tag = "circles_from_initial_state_2019_$udp_str" *
#             Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
# batch_dir = joinpath("processed_data", batch_tag)
# isdir(batch_dir) || mkpath(batch_dir)

# sim_time_circles = 30
# fps_circles = 240
# start_ramp_time = 0.0
# ramp_time_us = 3.0
# startup_decay_time = sim_time_circles * 0.75
# startup_damping_pattern = [0.0, 20.0, 20.0]
# damping_pattern = [0.0, 6.0, 20.0]

# failed_runs = NamedTuple[]

# for (run_id, (g, us, vw, kcu_mass_val)) in enumerate(
#     Iterators.product(g_earth_vals, us_vals, vw_vals, kcu_mass_vals))
#     run_tag = "run_" * lpad(string(run_id), 3, '0')
#     @info "Starting run" run_id g_earth = g us vw kcu_mass = kcu_mass_val struc_yaml = struc_yaml_input aero_yaml = aero_yaml_input
#     try
#         run_circles_from_initial_state(;
#             struc_yaml_path=struc_yaml_input,
#             aero_yaml_path=aero_yaml_input,
#             vsm_settings_path=vsm_settings_input,
#             v_wind=vw,
#             v_wind_base=vw,
#             us=us,
#             g_earth=g,
#             kcu_mass=kcu_mass_val,
#             sim_time_circles,
#             fps_circles,
#             start_ramp_time,
#             ramp_time_us,
#             startup_damping_pattern,
#             damping_pattern,
#             startup_decay_time,
#             save_subdir=batch_tag,
#             run_tag)
#         @info "Completed" run_id
#     catch err
#         @error "Failed" run_id err
#         push!(failed_runs, (run_id=run_id,
#             g_earth=g,
#             us=us,
#             vw=vw,
#             kcu_mass=kcu_mass_val,
#             struc_yaml=struc_yaml_input,
#             aero_yaml=aero_yaml_input,
#             error=err))
#     end
#     GC.gc()
# end

# if !isempty(failed_runs)
#     fp = joinpath(batch_dir, "failed_runs.txt")
#     open(fp, "w") do io
#         for fr in failed_runs
#             println(io, "Run $(fr.run_id): " *
#                         "g=$(fr.g_earth), " *
#                         "us=$(fr.us), " *
#                         "vw=$(fr.vw), " *
#                         "kcu_mass=$(fr.kcu_mass), " *
#                         "struc=$(fr.struc_yaml), " *
#                         "aero=$(fr.aero_yaml)")
#             println(io, "  Error: $(fr.error)")
#         end
#     end
#     @info "Wrote failure list" path = fp
# end

# n_total = length(collect(Iterators.product(
#     g_earth_vals, us_vals, vw_vals, kcu_mass_vals)))
# @info "Batch completed" total = n_total failed = length(failed_runs)

# # =============================================================================
# #### 0.35
# # =============================================================================
# udp = 0.35
# udp_str = lpad(string(Int(round(udp * 100))), 3, '0')
# struc_yaml_input = "struc_geometry_initial_state_lt_269_vw_84_udp_$udp_str.yaml"
# aero_yaml_input = "aero_geometry_initial_state_lt_269_vw_84_udp_$udp_str.yaml"
# vsm_settings_input = "vsm_settings.yaml"
# snapshot_input = "initial_state_snapshot_lt_271_vw_76_udp_$udp_str.jls"
# g_earth_vals = [0.0]
# us_vals = [0.1] #[0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
# vw_vals = [8.4]
# kcu_mass_vals = [nothing]

# batch_tag = "circles_from_initial_state_2019_$udp_str" *
#             Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
# batch_dir = joinpath("processed_data", batch_tag)
# isdir(batch_dir) || mkpath(batch_dir)

# sim_time_circles = 100
# fps_circles = 240
# start_ramp_time = 0.0
# ramp_time_us = 20.0
# startup_decay_time = 100.0
# startup_damping_pattern = [0.0, 20.0, 20.0]
# damping_pattern = [0.0, 5.0, 20.0]

# failed_runs = NamedTuple[]

# for (run_id, (g, us, vw, kcu_mass_val)) in enumerate(
#     Iterators.product(g_earth_vals, us_vals, vw_vals, kcu_mass_vals))
#     run_tag = "run_" * lpad(string(run_id), 3, '0')
#     @info "Starting run" run_id g_earth = g us vw kcu_mass = kcu_mass_val struc_yaml = struc_yaml_input aero_yaml = aero_yaml_input
#     try
#         run_circles_from_initial_state(;
#             struc_yaml_path=struc_yaml_input,
#             aero_yaml_path=aero_yaml_input,
#             vsm_settings_path=vsm_settings_input,
#             v_wind=vw,
#             v_wind_base=vw,
#             us=us,
#             g_earth=g,
#             kcu_mass=kcu_mass_val,
#             sim_time_circles,
#             fps_circles,
#             start_ramp_time,
#             ramp_time_us,
#             startup_damping_pattern,
#             damping_pattern,
#             startup_decay_time,
#             save_subdir=batch_tag,
#             run_tag)
#         @info "Completed" run_id
#     catch err
#         @error "Failed" run_id err
#         push!(failed_runs, (run_id=run_id,
#             g_earth=g,
#             us=us,
#             vw=vw,
#             kcu_mass=kcu_mass_val,
#             struc_yaml=struc_yaml_input,
#             aero_yaml=aero_yaml_input,
#             error=err))
#     end
#     GC.gc()
# end

# if !isempty(failed_runs)
#     fp = joinpath(batch_dir, "failed_runs.txt")
#     open(fp, "w") do io
#         for fr in failed_runs
#             println(io, "Run $(fr.run_id): " *
#                         "g=$(fr.g_earth), " *
#                         "us=$(fr.us), " *
#                         "vw=$(fr.vw), " *
#                         "kcu_mass=$(fr.kcu_mass), " *
#                         "struc=$(fr.struc_yaml), " *
#                         "aero=$(fr.aero_yaml)")
#             println(io, "  Error: $(fr.error)")
#         end
#     end
#     @info "Wrote failure list" path = fp
# end

# n_total = length(collect(Iterators.product(
#     g_earth_vals, us_vals, vw_vals, kcu_mass_vals)))
# @info "Batch completed" total = n_total failed = length(failed_runs)
