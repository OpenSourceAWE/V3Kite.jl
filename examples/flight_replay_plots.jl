# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for flight_replay.jl results.

Loads the simulation and reference logs flight_replay.jl saved and draws the
comparison: trajectory, panels, yaw rate against steering, the body-frame and
twist views at the photogrammetry frames, and animations of all of it.

`save_figs` in the project's `replay_settings:` governs everything written to
disk — the PDFs and the animations alike — and `figures_dir` says where. With it
off the figures are still built and displayed, which is the fast path when only
the numbers matter: recording the animations is the slow part of this script.

Run from the REPL after (or instead of, if the logs already exist) running
flight_replay.jl:

    include("flight_replay_plots.jl")

Answer the wing prompt the same way both times: which logs exist and which
reference structure they are drawn on both follow from the project.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using VortexStepMethod
using SymbolicAWEModels: update_from_sysstate!
using GLMakie
using GLMakie: save
using CairoMakie
using MakieControlPlots
GLMakie.activate!(; px_per_unit=2.0)
using Statistics
using LinearAlgebra
using LazyArtifacts
using KiteUtils

generate_drag_adjusted_polars(1.0)

PROJECT = select_project(
    ["particle lattice" => "system_psm_replay.yaml",
     "Timoshenko-beam wing" => "system_beam_replay.yaml"];
    prompt = "Which wing model were the flight data replayed on?")

set_data_path(v3_data_path())
kite_set = load_kite(PROJECT)
replay = load_replay(PROJECT)
SECTION, start_utc, end_utc = replay_maneuver(replay)

datadir = joinpath(artifact"flight_data", "flight_data")
h5_path = joinpath(datadir, replay.year == 2025 ?
    "ekf_awe_2025-10-09.h5" : "ekf_awe_2019-10-08.h5")
replay.year == 2019 && (kite_set.geom.depower_offset = replay.depower_offset_2019)
depower_offset_pct = kite_set.geom.depower_offset * 100.0

FIGURES_DIR = isabspath(replay.figures_dir) ? replay.figures_dir :
    joinpath(@__DIR__, "..", replay.figures_dir)

# Photogrammetry frames of this window, matched against the data directory.
if replay.year == 2025
    start_frame = utc_to_video_frame(parse_time_to_seconds(start_utc))
    end_frame = utc_to_video_frame(parse_time_to_seconds(end_utc))
    frame_csvs = Tuple{String, Int}[]
    for f in readdir(v3_data_path(); join=true)
        m = match(r"frame_(\d+)\.csv$", f)
        isnothing(m) && continue
        frame = parse(Int, m.captures[1])
        start_frame <= frame <= end_frame && push!(frame_csvs, (f, frame))
    end
    isempty(frame_csvs) && @warn "No frame CSV in range" start_frame end_frame
else
    frame_csvs = Tuple{String, Int}[]
end

# =============================================================================
# Plot helpers
# =============================================================================

"""
    syslog_to_tape(syslog) -> NamedTuple

Build a `(time, steering, depower)` tape from a syslog,
reading the applied inputs from `sl.steering` and `sl.depower`.
Used for plot functions that still take a `tapes` argument.
"""
function syslog_to_tape(syslog)
    sl = hasproperty(syslog, :syslog) ? syslog.syslog :
        syslog
    return (time=collect(sl.time),
        steering=collect(sl.steering),
        depower=collect(sl.depower))
end

function create_replay_plots(;
        sys_struct, data_sys_struct,
        syslog, datalog,
        frame_csvs,
        geom::V3GeomAdjustConfig,
        section, distance_based_steering,
        depower_offset_pct,
        figures_dir, save_figs)
    dt = syslog.syslog.time[2] - syslog.syslog.time[1]
    frame_syslog_idxs = find_frame_syslog_idxs(
        syslog, frame_csvs)

    sim_tape  = syslog_to_tape(syslog)
    data_tape = syslog_to_tape(datalog)
    logs   = [syslog, datalog]
    tapes  = [sim_tape, data_tape]
    labels = ["simulation", "data"]

    fig = plot_replay(
        [sys_struct, data_sys_struct],
        logs;
        tape_lengths=tapes,
        suffixes=labels,
        size=(1200, 800), labelsize=18)

    trajectory_kwargs = (;
        gradient=:vel, tapes=tapes, labels=labels,
        size=(560, 420), labelsize=20,
        frame_indexes=frame_syslog_idxs)
    panels_kwargs = (;
        tapes=tapes, labels=labels,
        show_aoa=false, labelsize=20,
        twin_time_axes=distance_based_steering,
        frame_indexes=frame_syslog_idxs,
        show_heading=false,
        show_course=true,
        show_tether_len=true,
        show_drag_coeff=false,
        show_lift_coeff=false,
        show_lift_drag_ratio=false)
    yaw_heading_kwargs = (;
        source=:heading,
        min_steering=0.05,
        labels=labels,
        strides=[replay.n_substeps, replay.n_substeps],
        figsize=(600, 400), labelsize=18, dt)
    yaw_course_kwargs = (;
        source=:course,
        min_steering=0.05,
        labels=labels,
        strides=[replay.n_substeps, replay.n_substeps],
        figsize=(600, 400), labelsize=18, dt)

    # GLMakie display
    trajectory = plot_2d_trajectory(logs; trajectory_kwargs...)
    panels = plot_2d_panels(logs; panels_kwargs...)
    yaw_fig_heading = plot_yaw_rate_vs_steering(
        logs; yaw_heading_kwargs...)
    yaw_fig_course = plot_yaw_rate_vs_steering(
        logs; yaw_course_kwargs...)

    # CairoMakie PDF saves
    sr = geom.reduce_steering ? geom.steering_reduction : 0.0
    tr = geom.reduce_tip ? geom.tip_reduction : 0.0
    dist_suffix = distance_based_steering ?
        "_dist" : ""
    config_suffix = "_dpoff_$(depower_offset_pct)" *
        "_sr_$(sr)_tr_$(tr)"
    suffix = "_$(section)" * config_suffix
    CairoMakie.activate!(; px_per_unit=2.0)
    traj_2d = plot_2d_trajectory(logs; trajectory_kwargs...)
    panels_2d = plot_2d_panels(logs; panels_kwargs...)
    yaw_heading_2d = plot_yaw_rate_vs_steering(
        logs; yaw_heading_kwargs...)
    yaw_course_2d = plot_yaw_rate_vs_steering(
        logs; yaw_course_kwargs...)
    if save_figs
        mkpath(figures_dir)
        function save_with_dist(name, figure)
            fname = "$(name)$(suffix).pdf"
            @info "Saving $fname"
            save(joinpath(figures_dir, fname), figure)
            fig_path = joinpath(figures_dir, replace(
                fname, ".pdf" => "$(dist_suffix).pdf"))
            @info "Saving $fig_path"
            save(fig_path, figure)
        end
        save_with_dist("trajectory_2d", traj_2d)
        save_with_dist("panels_2d", panels_2d)
        save_with_dist("yaw_rate_heading", yaw_heading_2d)
        save_with_dist("yaw_rate_course", yaw_course_2d)
    end
    GLMakie.activate!(; px_per_unit=2.0)

    # 2D body frame plots for PDF export
    body = Dict{Int, Dict{Symbol, Any}}()
    twist = Dict{Int, Any}()
    CairoMakie.activate!(; px_per_unit=2.0)
    frame_annotations = ["right turn", "straight flight"]
    for (fi, (csv, target_frame)) in
            enumerate(frame_csvs)
        idx = findfirst(
            x -> x[1] == target_frame,
            frame_syslog_idxs)
        isnothing(idx) && continue
        _, syslog_idx = frame_syslog_idxs[idx]
        update_from_sysstate!(sys_struct,
            syslog.syslog[syslog_idx])
        pts, groups = load_extra_points(
            csv, sys_struct)
        ann = get(frame_annotations, fi, "")
        frame_figs = Dict{Symbol, Any}()
        for dir in (:front, :side, :top)
            show_leg = dir == :side &&
                target_frame == 7362
            no_adjust = !geom.reduce_steering &&
                !geom.reduce_tip
            if dir == :front && no_adjust
                show_leg = true
            end
            leg_pos = dir == :front ? :top : :right
            dir_ann = if dir == :front
                geom.reduce_tip ?
                    "bridle reduced $(tr)m" :
                    "bridle unreduced"
            else
                ann
            end
            bf = plot_body_frame_local(sys_struct;
                extra_points=pts,
                extra_groups=groups, dir,
                title=false, legend=show_leg,
                legend_position=leg_pos,
                show_incidence=false,
                show_kcu=false,
                show_camera=false,
                annotation=dir_ann)
            fname = "body_frame_$(dir)" *
                "_$(section)" *
                "_frame_$(target_frame)" *
                "$(config_suffix).pdf"
            if save_figs
                @info "Saving $fname"
                save(joinpath(figures_dir, fname), bf)
                fig_fname = replace(fname,
                    ".pdf" => "$(dist_suffix).pdf")
                save(joinpath(figures_dir, fig_fname), bf)
            end
            frame_figs[dir] = bf
        end
        # Twist distribution
        twist_fig = plot_twist_dist(sys_struct;
            extra_points=pts,
            extra_groups=groups,
            figsize=(560*0.8, 210*0.8),
            labelsize=24,
            title=false, legend=false,
            limits=(-3, 14),
            annotation=ann)
        twist_fname = "twist_dist" *
            "_$(section)" *
            "_frame_$(target_frame)" *
            "$(config_suffix).pdf"
        if save_figs
            @info "Saving $twist_fname"
            save(joinpath(figures_dir, twist_fname), twist_fig)
            fig_twist = replace(twist_fname,
                ".pdf" => "$(dist_suffix).pdf")
            save(joinpath(figures_dir, fig_twist),
                twist_fig)
        end
        twist[target_frame] = twist_fig
        frame_figs[:twist] = twist_fig

        body[target_frame] = frame_figs
        @info "Saved 2D body frame + twist" target_frame
    end
    GLMakie.activate!(; px_per_unit=2.0)

    mean_gk = Dict{Tuple{String,Symbol},Float64}()
    for (label, lg) in [("sim", syslog),
                         ("data", datalog)]
        sl = lg.syslog
        for source in (:heading, :course)
            rate = calc_turn_rate(lg; source, dt)
            us = sl.steering[2:end]
            va = sl.v_app[2:end]
            mask = abs.(us) .> 0.05
            x = abs.(us[mask] .* va[mask])[1:replay.n_substeps:end]
            y = abs.(rate[mask])[1:replay.n_substeps:end]
            mean_gk[(label, source)] = dot(x, y) / dot(x, x)
        end
    end
    for source in (:heading, :course)
        sim_gk  = mean_gk[("sim",  source)]
        data_gk = mean_gk[("data", source)]
        pct = (sim_gk - data_gk) / data_gk * 100
        @info "gk (least-squares, plot-matching)" source sim=round(sim_gk; digits=3) data=round(data_gk; digits=3) pct=round(pct; digits=1)
    end

    n_steer = min(length(sim_tape.steering),
        length(data_tape.steering))
    steer_diff = sim_tape.steering[1:n_steer] .-
        data_tape.steering[1:n_steer]
    @info "Mean steering input diff (sim - data)" mean=round(
            mean(steer_diff); digits=4) mean_abs=round(
            mean(abs.(steer_diff)); digits=4)

    hdot_fig = plot_turn_rate_vs_time(logs;
        labels=["sim", "data"], dt)
    display(hdot_fig)

    wind_fig = plot_wind_compare(datalog)
    display(wind_fig)

    return (; fig, trajectory, panels, traj_2d, panels_2d,
        yaw_fig_heading, yaw_fig_course,
        body, twist, hdot_fig, wind_fig)
end

# =============================================================================
# Main execution
# =============================================================================

sim_name, data_name = replay_log_names(h5_path, start_utc, end_utc, kite_set.geom)
@info "Loading replay logs" sim_name data_name
for name in (sim_name, data_name)
    isfile(joinpath(v3_data_path(), name * ".arrow")) || error(
        "No log $name.arrow. A replay that aborts before logging a state " *
        "writes no log, so re-run flight_replay.jl and check that it " *
        "reports both logs as saved.")
end
syslog  = load_log(sim_name)
datalog = load_log(data_name)

data_path = v3_data_path()
set = Settings(PROJECT)
set.g_earth = 9.81
set.profile_law = 0
source_struc = struc_geometry_path(PROJECT; data_path)
source_aero = aero_geometry_path(PROJECT; data_path,
    aero_mode = resolve_aero_mode(kite_set))
vsm_set = VortexStepMethod.VSMSettings(
    vsm_settings_path(PROJECT; data_path); data_prefix=false)
vsm_set.wings[1].geometry_file = source_aero
sam, _ = build_replay_sys_struct(set, kite_set, source_struc, vsm_set)
data_sam, _ = build_replay_sys_struct(set, kite_set, source_struc, vsm_set)

plots = create_replay_plots(;
    sys_struct=sam.sys_struct,
    data_sys_struct=data_sam.sys_struct,
    syslog, datalog,
    frame_csvs, geom=kite_set.geom,
    section=SECTION,
    distance_based_steering=replay.distance_based_steering,
    depower_offset_pct=depower_offset_pct,
    figures_dir=FIGURES_DIR, save_figs=replay.save_figs)

if replay.save_figs
    sim_tape = syslog_to_tape(syslog)
    data_tape = syslog_to_tape(datalog)
    frame_syslog_idxs = find_frame_syslog_idxs(syslog, frame_csvs)
    mkpath(FIGURES_DIR)
    gif(name) = joinpath(FIGURES_DIR, "$(name)_$(SECTION).gif")
    # Realtime: gif fps = log sample rate (1/dt).
    sl_dt = syslog.syslog.time[2] - syslog.syslog.time[1]
    fps = max(1, round(Int, 1 / sl_dt))
    GLMakie.activate!(; px_per_unit=2.0)
    record_2d_trajectory([syslog, datalog], gif("trajectory_anim");
        gradient=:vel, tapes=[sim_tape, data_tape],
        labels=["simulation", "data"], framerate=fps,
        frame_indexes=frame_syslog_idxs)
    record_2d_panels([syslog, datalog], gif("panels_anim");
        tapes=[sim_tape, data_tape],
        labels=["simulation", "data"],
        show_aoa=false, show_course=true,
        show_heading=false, show_drag_coeff=false,
        show_lift_coeff=false, show_lift_drag_ratio=false,
        framerate=fps, frame_indexes=frame_syslog_idxs)
    SymbolicAWEModels.record(syslog, sam.sys_struct,
        gif("replay_3d_world"); body_frame=false, size=(800, 800),
        framerate=fps)
    SymbolicAWEModels.record(syslog, sam.sys_struct,
        gif("replay_3d_body"); body_frame=true, pan_vertical=5.0,
        size=(800, 800), framerate=fps)
end

SymbolicAWEModels.replay(syslog, sam.sys_struct)

nothing
