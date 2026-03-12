# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Load and Plot V3 Kite Log Data

Post-processing script that loads saved simulation logs,
reconstructs the system model, and produces time series plots,
3D replay, wing node body-frame plots, and line stretch analysis.

Usage:
    julia --project=examples examples/load_and_plot.jl
"""

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX,
    V3_DEPOWER_IDX
using GLMakie
using CairoMakie
using LinearAlgebra
using Statistics

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = joinpath(
    dirname(@__DIR__),
    "processed_data")

# Replay styling (post-processed on the Makie scene).
# `REPLAY_SEGMENT_LINEWIDTH` targets the main tether/segment plot.
# `REPLAY_OTHER_LINEWIDTH` applies to other line plots that expose linewidth.
REPLAY_SEGMENT_LINEWIDTH = nothing
REPLAY_OTHER_LINEWIDTH = nothing
# Segment IDs to emphasize in replay (from `data/struc_geometry.yaml`):
# 1:9 leading edge tubes, 10:19 struts (chords).
REPLAY_THICK_SEGMENT_IDS = collect(1:19)
REPLAY_THICK_SEGMENT_LINEWIDTH = 5.0
REPLAY_THICK_SEGMENT_COLOR = :black

# =============================================================================
# Helper functions
# =============================================================================

function resolve_log_file(log_name, base_dir)
    candidates = String[]

    function add_candidate!(path)
        push!(candidates, path)
        if !endswith(path, ".arrow")
            push!(candidates, path * ".arrow")
        end
    end

    add_candidate!(log_name)
    add_candidate!(joinpath(base_dir, log_name))

    # Handle names that use the legacy "v3_kite/" prefix.
    if startswith(log_name, "v3_kite/")
        rel = log_name[(length("v3_kite/")+1):end]
        add_candidate!(joinpath(base_dir, rel))
    end

    # Also try relative to processed_data root.
    add_candidate!(joinpath(dirname(base_dir), log_name))

    for c in unique(candidates)
        if isfile(c)
            return splitext(basename(c))[1], dirname(c), c
        end
    end

    error("Could not find log file. Tried:\n" *
          join(unique(candidates), "\n"))
end

function _read_plot_attr(attr)
    try
        return attr[]
    catch
        return attr
    end
end

function _set_linewidth!(plot_obj, width::Real)
    hasproperty(plot_obj, :linewidth) || return false
    lw_attr = getproperty(plot_obj, :linewidth)
    try
        lw_attr[] = float(width)
    catch
        return false
    end
    return true
end

function _walk_scene_plots!(f::Function, obj)
    f(obj)
    hasproperty(obj, :plots) || return
    for child in getproperty(obj, :plots)
        _walk_scene_plots!(f, child)
    end
end

# Backward-compatible argument order.
_walk_scene_plots!(obj, f::Function) = _walk_scene_plots!(f, obj)

function _normalize_segment_ids(segment_ids)
    ids = Int[]
    for sid in segment_ids
        sid_int = Int(sid)
        sid_int > 0 || continue
        push!(ids, sid_int)
    end
    sort!(ids)
    unique!(ids)
    return ids
end

function _subset_segment_points(points, segment_ids)
    out = Point3f[]
    for seg_id in segment_ids
        i1 = 2 * seg_id - 1
        i2 = i1 + 1
        if i2 <= length(points)
            push!(out, points[i1])
            push!(out, points[i2])
        end
    end
    return out
end

function _find_plot_by_label(scene, target_label::AbstractString)
    found = Ref{Any}(nothing)
    _walk_scene_plots!(scene) do plot_obj
        found[] === nothing || return
        hasproperty(plot_obj, :label) || return
        label = _read_plot_attr(getproperty(plot_obj, :label))
        if label == target_label
            found[] = plot_obj
        end
    end
    return found[]
end

function _overlay_thick_segments!(scene, segment_ids;
    linewidth::Union{Nothing,Real}=nothing, color=:black)
    linewidth === nothing && return nothing
    ids = _normalize_segment_ids(segment_ids)
    isempty(ids) && return nothing

    segment_plot = _find_plot_by_label(scene, "Segments")
    segment_plot === nothing && return nothing

    segment_points_attr = try
        segment_plot[1]
    catch
        return nothing
    end

    overlay_points = try
        lift(points -> _subset_segment_points(points, ids),
            segment_points_attr)
    catch
        _subset_segment_points(_read_plot_attr(segment_points_attr),
            ids)
    end

    return linesegments!(scene, overlay_points;
        color=color, linewidth=float(linewidth), transparency=true)
end

function style_replay_scene!(scene;
    segment_linewidth::Union{Nothing,Real}=nothing,
    other_linewidth::Union{Nothing,Real}=nothing,
    thick_segment_ids=Int[],
    thick_segment_linewidth::Union{Nothing,Real}=nothing,
    thick_segment_color=:black)
    if segment_linewidth === nothing &&
       other_linewidth === nothing &&
       thick_segment_linewidth === nothing
        return scene
    end

    _walk_scene_plots!(scene) do plot_obj
        hasproperty(plot_obj, :linewidth) || return

        label = hasproperty(plot_obj, :label) ?
                _read_plot_attr(getproperty(plot_obj, :label)) : nothing

        if segment_linewidth !== nothing && label == "Segments"
            _set_linewidth!(plot_obj, segment_linewidth)
        elseif other_linewidth !== nothing
            _set_linewidth!(plot_obj, other_linewidth)
        end
    end

    _overlay_thick_segments!(scene, thick_segment_ids;
        linewidth=thick_segment_linewidth,
        color=thick_segment_color)

    return scene
end

function udp_tag_from_log_name(log_name::AbstractString, fallback_udp::Real)
    m = match(r"_udp_([0-9]{3})", log_name)
    if m !== nothing
        return String(m.captures[1])
    end
    return lpad(string(Int(round(fallback_udp * 100))), 3, '0')
end

function find_initial_state_geometry(; lt::Int, udp_tag::AbstractString,
    v_wind::Real, data_root::String=v3_data_path())
    udp_tag_s = String(udp_tag)
    pat = Regex("^struc_geometry_initial_state_lt_$(lt)_vw_([0-9]+)_udp_$(udp_tag_s)\\.yaml" * "\$")
    target_vw = Int(round(v_wind * 10))
    candidates = Tuple{String,Int}[]
    for name in readdir(data_root)
        m = match(pat, name)
        m === nothing && continue
        push!(candidates, (name, parse(Int, m.captures[1])))
    end
    isempty(candidates) && return nothing

    sort!(candidates; by=x -> (abs(x[2] - target_vw), x[2], x[1]))
    struc_name = first(candidates)[1]
    aero_name = replace(struc_name,
        "struc_geometry_" => "aero_geometry_")
    isfile(joinpath(data_root, aero_name)) || return nothing
    return (struc_yaml_path=struc_name,
        aero_yaml_path=aero_name)
end

function effective_v_wind_from_log(lg, fallback::Real)
    sl = hasproperty(lg, :syslog) ? lg.syslog : lg
    if !isempty(sl) && hasproperty(sl[1], :v_wind_gnd)
        vw = getproperty(sl[1], :v_wind_gnd)
        if !isempty(vw)
            v = abs(float(vw[1]))
            if isfinite(v) && v > 0
                return v
            end
        end
    end
    return float(fallback)
end

function load_log_and_system(; log_name)
    m = match(
        r"_(?:u_dp|up)_([0-9]+(?:\\.[0-9]+)?)_us_([0-9._-]+)_vw_([0-9]+)_lt_([0-9]+)",
        log_name)
    m === nothing && error(
        "Could not parse u_dp/up/us/vw/lt from: $log_name")
    u_dp = parse(Float64, m.captures[1]) / 100.0
    us_token = first(split(m.captures[2], "_"))
    us = parse(Float64, us_token) / 100.0
    v_wind = parse(Int, m.captures[3])
    lt = parse(Int, m.captures[4])
    log_file, log_dir, log_path = resolve_log_file(
        log_name, DATA_DIR)
    @info "Resolved log file" log_path
    lg = load_log(log_file; path=log_dir)
    v_wind_ref = effective_v_wind_from_log(lg, v_wind)
    @info "Parsed tags" u_dp us v_wind lt v_wind_ref

    struc_yaml_path = "struc_geometry.yaml"
    aero_yaml_path = "aero_geometry.yaml"
    geom_adjust_cfg = V3GeomAdjustConfig(
        reduce_tip=true, reduce_te=true)

    if occursin("circles_from_initial_state", log_name)
        udp_tag = udp_tag_from_log_name(log_name, u_dp)
        geom = find_initial_state_geometry(;
            lt, udp_tag, v_wind=v_wind_ref)
        if geom !== nothing
            struc_yaml_path = geom.struc_yaml_path
            aero_yaml_path = geom.aero_yaml_path
            geom_adjust_cfg = nothing
            @info "Using initial-state geometry" struc_yaml_path aero_yaml_path
        else
            @warn "Initial-state geometry not found; falling back to base geometry + model_setup adjustments" lt udp_tag
        end
    end

    config = V3SimConfig(
        struc_yaml_path=struc_yaml_path,
        aero_yaml_path=aero_yaml_path,
        vsm_settings_path="vsm_settings.yaml",
        v_wind=v_wind_ref,
        tether_length=Float64(lt),
        wing_type=REFINE,
    )
    sam, sys = create_v3_model(config)
    if geom_adjust_cfg !== nothing
        apply_geom_adjustments!(sys, geom_adjust_cfg)
    end

    return lg, sam, u_dp, us, v_wind_ref, lt
end

# =============================================================================
# Wing body-frame plotting
# =============================================================================

function print_and_plot_wing(lg, sam; is_print=false)
    lg_last = lg.syslog[end]
    wing = sam.sys_struct.wings[1]
    origin_idx = wing.origin_idx
    origin_w = [
        lg_last.X[origin_idx],
        lg_last.Y[origin_idx],
        lg_last.Z[origin_idx],
    ]
    R_b_w = V3Kite.SymbolicAWEModels.quaternion_to_rotation_matrix(
        lg_last.orient)
    wing_point_idxs = [
        p.idx for p in sam.sys_struct.points
        if p.type == WING]

    if is_print
        println("\n# Wing node positions (world frame):")
        for idx in wing_point_idxs
            pos_w = [lg_last.X[idx], lg_last.Y[idx],
                lg_last.Z[idx]]
            println("- [$idx, [$(Float64(pos_w[1])), " *
                    "$(Float64(pos_w[2])), " *
                    "$(Float64(pos_w[3]))], " *
                    "WING, 1, 1, 0.0, 10.0, 0.0]")
        end

        println("\n# Wing node positions (body frame):")
        for idx in wing_point_idxs
            pos_w = [lg_last.X[idx], lg_last.Y[idx],
                lg_last.Z[idx]]
            pos_b = R_b_w' * (pos_w .- origin_w)
            println("- [$idx, [$(Float64(pos_b[1])), " *
                    "$(Float64(pos_b[2])), " *
                    "$(Float64(pos_b[3]))], " *
                    "WING, 1, 1, 0.0, 10.0, 0.0]")
        end

        bridle_pairs = [
            (22, 25), (23, 24), (26, 27), (28, 31),
            (29, 30), (32, 33), (34, 36), (37, 38)]
        bridle_center = [35]

        println("\n# Bridle node positions (body frame):")
        for (idx_pos, idx_neg) in bridle_pairs
            pos_w_pos = [lg_last.X[idx_pos],
                lg_last.Y[idx_pos], lg_last.Z[idx_pos]]
            pos_b_pos = R_b_w' * (pos_w_pos .- origin_w)
            pos_w_neg = [lg_last.X[idx_neg],
                lg_last.Y[idx_neg], lg_last.Z[idx_neg]]
            pos_b_neg = R_b_w' * (pos_w_neg .- origin_w)
            y_c = (pos_b_pos[2] + pos_b_neg[2]) / 2.0
            y_off = (pos_b_pos[2] - pos_b_neg[2]) / 2.0
            println("- [$idx_pos, " *
                    "[$(Float64(pos_b_pos[1])), " *
                    "$(Float64(y_c + y_off)), " *
                    "$(Float64(pos_b_pos[3]))], " *
                    "DYNAMIC, 1, 1, 0.0, 30.000, 0.0, " *
                    "0.0, 0.0]")
            println("- [$idx_neg, " *
                    "[$(Float64(pos_b_neg[1])), " *
                    "$(Float64(y_c - y_off)), " *
                    "$(Float64(pos_b_neg[3]))], " *
                    "DYNAMIC, 1, 1, 0.0, 30.000, 0.0, " *
                    "0.0, 0.0]")
        end
        for idx in bridle_center
            pos_w = [lg_last.X[idx], lg_last.Y[idx],
                lg_last.Z[idx]]
            pos_b = R_b_w' * (pos_w .- origin_w)
            println("- [$idx, " *
                    "[$(Float64(pos_b[1])), " *
                    "$(Float64(pos_b[2])), " *
                    "$(Float64(pos_b[3]))], " *
                    "DYNAMIC, 1, 1, 0.1, 30.000, 0.0, " *
                    "0.0, 0.0]")
        end
    end

    # 2D scatter plots of wing nodes in body frame
    xs_b, ys_b, zs_b = Float64[], Float64[], Float64[]
    for idx in wing_point_idxs
        pos_w = [lg_last.X[idx], lg_last.Y[idx],
            lg_last.Z[idx]]
        pos_b = R_b_w' * (pos_w .- origin_w)
        push!(xs_b, pos_b[1])
        push!(ys_b, pos_b[2])
        push!(zs_b, pos_b[3])
    end

    fig2 = Figure(resolution=(900, 300))
    ax_xy = Axis(fig2[1, 1], title="Top (x,y)",
        xlabel="y_b", ylabel="x_b")
    ax_xz = Axis(fig2[1, 2], title="Side (x,z)",
        xlabel="x_b", ylabel="z_b")
    ax_yz = Axis(fig2[1, 3], title="Rear (y,z)",
        xlabel="y_b", ylabel="z_b")
    ax_xy.aspect = DataAspect()
    ax_xz.aspect = DataAspect()
    ax_yz.aspect = DataAspect()

    xs_b_rot = ys_b
    ys_b_rot = [-x for x in xs_b]

    scatter!(ax_xy, xs_b_rot, ys_b_rot)
    scatter!(ax_xz, xs_b, zs_b)
    scatter!(ax_yz, ys_b, zs_b)

    spans = [
        maximum(xs_b_rot) - minimum(xs_b_rot),
        maximum(ys_b_rot) - minimum(ys_b_rot),
        maximum(xs_b) - minimum(xs_b),
        maximum(zs_b) - minimum(zs_b),
        maximum(ys_b) - minimum(ys_b),
    ]
    global_span = maximum(spans)
    global_span = global_span > 0 ? global_span : 1.0
    global_span *= 1.05

    function set_square_limits!(ax, xs, ys, span)
        cx, cy = mean(xs), mean(ys)
        half = span / 2
        xlims!(ax, cx - half, cx + half)
        ylims!(ax, cy - half * 0.5, cy + half * 0.5)
    end

    set_square_limits!(ax_xy, xs_b_rot, ys_b_rot,
        global_span)
    set_square_limits!(ax_xz, xs_b, zs_b, global_span)
    set_square_limits!(ax_yz, ys_b, zs_b, global_span)

    # Draw wing structural segments (indices 1:19)
    wing_seg_idxs = 1:19
    lines_xy, lines_xz, lines_yz =
        Point2f[], Point2f[], Point2f[]
    for seg_idx in wing_seg_idxs
        seg = sam.sys_struct.segments[seg_idx]
        p1, p2 = seg.point_idxs
        pos1_w = [lg_last.X[p1], lg_last.Y[p1],
            lg_last.Z[p1]]
        pos2_w = [lg_last.X[p2], lg_last.Y[p2],
            lg_last.Z[p2]]
        pos1_b = R_b_w' * (pos1_w .- origin_w)
        pos2_b = R_b_w' * (pos2_w .- origin_w)
        x1, y1, z1 = pos1_b
        x2, y2, z2 = pos2_b
        push!(lines_xy,
            Point2f(y1, -x1), Point2f(y2, -x2),
            Point2f(NaN, NaN))
        push!(lines_xz,
            Point2f(x1, z1), Point2f(x2, z2),
            Point2f(NaN, NaN))
        push!(lines_yz,
            Point2f(y1, z1), Point2f(y2, z2),
            Point2f(NaN, NaN))
    end

    lines!(ax_xy, lines_xy, color=:gray)
    lines!(ax_xz, lines_xz, color=:gray)
    lines!(ax_yz, lines_yz, color=:gray)

    return fig2
end

# =============================================================================
# Time series plotting
# =============================================================================

"""Reconstruct steering command [%] from left-right tape lengths."""
function steering_tape_trace(lg, sam)
    sl = hasproperty(lg, :syslog) ? lg.syslog : lg
    if isempty(sl)
        return (time=Float64[], steering=Float64[])
    end

    seg_left = sam.sys_struct.segments[V3_STEERING_LEFT_IDX]
    seg_right = sam.sys_struct.segments[V3_STEERING_RIGHT_IDX]
    li_l, lj_l = seg_left.point_idxs
    li_r, lj_r = seg_right.point_idxs

    n = length(sl)
    time = Vector{Float64}(undef, n)
    steering = Vector{Float64}(undef, n)

    @inbounds for k in 1:n
        state = sl[k]
        X, Y, Z = state.X, state.Y, state.Z

        l_left = sqrt(
            (X[lj_l] - X[li_l])^2 +
            (Y[lj_l] - Y[li_l])^2 +
            (Z[lj_l] - Z[li_l])^2)
        l_right = sqrt(
            (X[lj_r] - X[li_r])^2 +
            (Y[lj_r] - Y[li_r])^2 +
            (Z[lj_r] - Z[li_r])^2)

        steering[k] = steering_length_to_percentage(
            l_left, l_right)
        time[k] = Float64(state.time)
    end
    return (time=time, steering=steering)
end

function unwrap_heading(heading)
    hw = copy(heading)
    for j in 2:length(hw)
        while hw[j] - hw[j-1] > pi
            hw[j] -= 2pi
        end
        while hw[j] - hw[j-1] < -pi
            hw[j] += 2pi
        end
    end
    return hw
end

function heading_rate_from_states(sl)
    n = length(sl)
    n < 2 && return Float64[], Float64[]
    heading = Vector{Float64}(undef, n)
    time = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        state = sl[k]
        heading[k] = Float64(state.heading)
        time[k] = Float64(state.time)
    end
    hw = unwrap_heading(heading)
    rates = diff(hw) ./ diff(time)
    return rates, time[1:end-1]
end

function steering_command_from_states(sl, sys)
    seg_left = sys.segments[V3_STEERING_LEFT_IDX]
    seg_right = sys.segments[V3_STEERING_RIGHT_IDX]
    li_l, lj_l = seg_left.point_idxs
    li_r, lj_r = seg_right.point_idxs
    n = length(sl)
    us_cmd = zeros(Float64, n)
    @inbounds for k in 1:n
        state = sl[k]
        X, Y, Z = state.X, state.Y, state.Z
        l_left = sqrt(
            (X[lj_l] - X[li_l])^2 +
            (Y[lj_l] - Y[li_l])^2 +
            (Z[lj_l] - Z[li_l])^2)
        l_right = sqrt(
            (X[lj_r] - X[li_r])^2 +
            (Y[lj_r] - Y[li_r])^2 +
            (Z[lj_r] - Z[li_r])^2)
        us_cmd[k] = steering_length_to_percentage(
            l_left, l_right) / 100.0
    end
    return us_cmd
end

function gk_series(lg, sam)
    sl = hasproperty(lg, :syslog) ? lg.syslog : lg
    n = length(sl)
    n < 2 && return Float64[], Float64[]

    hr, _ = heading_rate_from_states(sl)
    us_cmd = steering_command_from_states(sl, sam.sys_struct)

    v_app = Vector{Float64}(undef, n)
    time = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        state = sl[k]
        v_app[k] = Float64(state.v_app)
        time[k] = Float64(state.time)
    end

    va = v_app[2:end]
    us_seg = us_cmd[2:end]
    gk = similar(hr)
    @inbounds for k in eachindex(gk)
        gk[k] = abs(us_seg[k]) > 1e-8 ?
                hr[k] / (va[k] * us_seg[k]) : NaN
    end
    return gk, time[2:end]
end

function print_gk_window_summary(lg, sam;
    udp::Float64, window_sec::Float64=1.0)
    gk, gk_time = gk_series(lg, sam)
    @info "udp = $(round(udp, digits=3))"
    isempty(gk) && return

    rel_t = gk_time .- gk_time[1]
    t_end = rel_t[end]
    n_bins = max(1, Int(ceil(t_end / window_sec)))
    window_3_10 = (rel_t .>= 3.0) .& (rel_t .<= 10.0)
    vals_3_10 = gk[window_3_10]
    vals_3_10 = vals_3_10[isfinite.(vals_3_10)]
    gk_3_10_mean = isempty(vals_3_10) ? NaN : mean(vals_3_10)
    gk_3_10_s = isfinite(gk_3_10_mean) ?
                string(round(gk_3_10_mean, digits=4)) : "NaN"

    for b in 0:(n_bins-1)
        t0 = b * window_sec
        t1 = (b + 1) * window_sec
        if b == n_bins - 1
            mask = (rel_t .>= t0) .& (rel_t .<= t1)
        else
            mask = (rel_t .>= t0) .& (rel_t .< t1)
        end
        vals = gk[mask]
        vals = vals[isfinite.(vals)]
        gk_mean = isempty(vals) ? NaN : mean(vals)
        gk_s = isfinite(gk_mean) ?
               string(round(gk_mean, digits=4)) : "NaN"
        println("| t = $(Int(round(t0)))-$(Int(round(t1)))s " *
                "gk = $gk_s")
    end
    println("| t = 3-10s gk(avg) = $gk_3_10_s")
end

function plot_time_series(lg, sam)
    tape_lengths = [steering_tape_trace(lg, sam)]
    # return plot(sam.sys_struct, lg;
    #     tape_lengths=tape_lengths,
    #     plot_turn_rates=false, plot_reelout=false,
    #     plot_twist=false,
    #     plot_yaw_rate_paper=false,
    #     yaw_rate_paper_ylims=(-90.0, 90.0),
    #     yaw_rate_paper_compare=false,
    #     plot_v_app=true, plot_kite_vel=false,
    #     plot_gk=false,
    #     plot_aoa=true,
    #     ylims=Dict(:aoa => (0.0, 15.0), :gk => (0.0, 15.0)),
    #     plot_heading=false, plot_elevation=true,
    #     plot_azimuth=true, plot_winch_force=false,
    #     plot_set_values=false, plot_us=true,
    #     plot_tether_actual=false,
    #     plot_turn_radius=true,
    #     turn_radius_ylims=(0.0, 40.0),
    #     plot_cs=true, cs_ylims=(0.0, 0.02))
    return plot(sam.sys_struct, lg;
        tape_lengths=tape_lengths,
        plot_turn_rates=false, plot_reelout=false,
        plot_twist=false,
        plot_yaw_rate_paper=false,
        yaw_rate_paper_ylims=(-90.0, 90.0),
        yaw_rate_paper_compare=false,
        plot_v_app=true, plot_kite_vel=false,
        plot_gk=false,
        plot_aoa=false,
        ylims=Dict(:aoa => (0.0, 15.0), :gk => (0.0, 15.0)),
        plot_heading=false, plot_elevation=true,
        plot_azimuth=true, plot_winch_force=true,
        plot_set_values=false, plot_us=true,
        plot_tether_actual=false,
        plot_turn_radius=false,
        turn_radius_ylims=(0.0, 40.0),
        plot_cs=false, cs_ylims=(0.0, 0.02),
        plot_aero_force=false)
end


# lot(::Vector{<:SystemStructure}, ::Vector{<:KiteUtils.SysLog}; plot_default, plot_reelout, plot_aero_force, plot_twist, plot_us, plot_gk, gk_ylims, plot_yaw_rate, plot_yaw_rate_paper, yaw_rate_paper_ylims, yaw_rate_paper_compare, plot_gk_paper, plot_cs, cs_ylims, turn_radius_ylims, plot_v_app, plot_kite_vel, plot_aoa, plot_sideslip, plot_heading, plot_course, plot_kiteutils_course, plot_aero_moment, plot_turn_rates, plot_turn_radius, plot_elevation, plot_azimuth, plot_wind, plot_tether_moment, plot_tether_actual, plot_winch_force, plot_set_values, plot_distance, plot_cone_angle, plot_old_heading, plot_tether, setpoints, ylims, tape_lengths, suffixes, size, show_legend, ticklabelsize, compact_labels, legend_position, legendsize)

# =============================================================================
# Tether direction analysis
# =============================================================================

function report_tether_direction_alignment(lg)
    isempty(lg.syslog) && return
    idxs = unique([1, cld(length(lg.syslog), 2),
        length(lg.syslog)])
    for idx in idxs
        sl = lg.syslog[idx]
        p1 = [sl.X[1], sl.Y[1], sl.Z[1]]
        ple12 = [sl.X[12], sl.Y[12], sl.Z[12]]
        ple14 = [sl.X[14], sl.Y[14], sl.Z[14]]
        p_le_mid = (ple12 .+ ple14) ./ 2
        bridle_dir = p1 .- p_le_mid
        midLE_to_KCU = norm(bridle_dir) > 0 ?
                       bridle_dir / norm(bridle_dir) : bridle_dir
        p39 = [sl.X[39], sl.Y[39], sl.Z[39]]
        seg90_dir = p39 .- p1
        seg90_unit = norm(seg90_dir) > 0 ?
                     seg90_dir / norm(seg90_dir) : seg90_dir
        @info "Direction (sample $idx)" midLE_to_KCU seg90_unit
    end
end

# =============================================================================
# Line stretch analysis
# =============================================================================

"""
    compute_line_stretch(lg, sam; kwargs...)

Compute signed line stretch ratios over a time window.
Returns per-category stretch matrices and per-pulley
combined stretch vectors.
"""
function compute_line_stretch(lg, sam;
    window_seconds::Real=50.0,
    segment_l0_adjustments=nothing,
    tether_length=nothing)
    sl = hasproperty(lg, :syslog) ? lg.syslog : lg
    if isempty(sl)
        @warn "compute_line_stretch: empty log"
        return (window=(0.0, 0.0),
            ratio=Dict{Symbol,Matrix{Float64}}())
    end

    segments = sam.sys_struct.segments
    base_l0 = [Float64(seg.l0) for seg in segments]

    tether_seg_idxs = if !isempty(sam.sys_struct.tethers) &&
                         !isempty(sam.sys_struct.tethers[1].segment_idxs)
        Int.(sam.sys_struct.tethers[1].segment_idxs)
    else
        collect(90:95)
    end
    tether_seg_idxs = [
        idx for idx in tether_seg_idxs
        if 1 <= idx <= length(segments)]
    tether_seg_set = Set(tether_seg_idxs)
    nominal_total = sum(
        l0 for (i, l0) in enumerate(base_l0)
        if i in tether_seg_set && isfinite(l0) && l0 > 0)

    # Per-sample tether scaling
    tether_len_series = Float64[]
    if tether_length === nothing
        tether_len_series = fill(NaN, length(sl))
        if hasproperty(sl[1], :l_tether)
            for (idx, state) in enumerate(sl)
                lt_vec = getproperty(state, :l_tether)
                if !isempty(lt_vec)
                    tether_len_series[idx] =
                        float(lt_vec[1])
                end
            end
        end
    elseif tether_length isa AbstractVector
        tether_len_series = [
            float(tether_length[min(i,
                length(tether_length))])
            for i in 1:length(sl)]
    else
        tether_len_series = fill(
            float(tether_length), length(sl))
    end

    tether_scale = ones(Float64, length(sl))
    if nominal_total > 0 && !isempty(tether_seg_set)
        last_scale = 1.0
        for i in 1:length(sl)
            tl = tether_len_series[i]
            if isfinite(tl) && tl > 0
                last_scale = tl / nominal_total
            end
            tether_scale[i] = last_scale
        end
    end

    t_end = sl[end].time
    t_start = t_end - window_seconds
    start_idx = 1
    for i in 1:length(sl)
        if sl[i].time >= t_start
            start_idx = i
            break
        end
    end
    window = (sl[start_idx].time, t_end)
    window_span = window[2] - window[1]
    n_samples = length(sl) - start_idx + 1
    sample_times = [sl[i].time for i in start_idx:length(sl)]
    l0_adj = segment_l0_adjustments === nothing ?
             Dict{Int,Any}() : segment_l0_adjustments

    pulleys = sam.sys_struct.pulleys
    pulley_seg_set = Set{Int}()
    for p in pulleys
        push!(pulley_seg_set, Int(p.segment_idxs[1]))
        push!(pulley_seg_set, Int(p.segment_idxs[2]))
    end

    seg_len_for_pulleys = Dict(
        seg => fill(NaN, n_samples)
        for seg in pulley_seg_set)
    seg_l0_for_pulleys = Dict(
        seg => fill(NaN, n_samples)
        for seg in pulley_seg_set)

    bridle_seg_idxs = [
        i for i in 47:86 if !(i in pulley_seg_set)]
    steering_tape_idxs = [
        V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX]
    power_tape_idxs = [V3_DEPOWER_IDX]

    # Precompute pulley segment lengths
    if !isempty(pulley_seg_set)
        @inbounds for (si, li) in
                      enumerate(start_idx:length(sl))
            state = sl[li]
            X, Y, Z = state.X, state.Y, state.Z
            for seg_idx in pulley_seg_set
                seg = segments[seg_idx]
                l0 = base_l0[seg_idx]
                if seg_idx in tether_seg_set
                    l0 *= tether_scale[li]
                end
                adj = get(l0_adj, seg_idx, nothing)
                if adj !== nothing
                    if adj isa AbstractVector
                        l0 += adj[min(si, length(adj))]
                    elseif adj isa Real
                        l0 += adj
                    end
                end
                (!isfinite(l0) || l0 <= 0) && continue
                p1, p2 = seg.point_idxs
                dx = X[p2] - X[p1]
                dy = Y[p2] - Y[p1]
                dz = Z[p2] - Z[p1]
                len = sqrt(dx * dx + dy * dy + dz * dz)
                if isfinite(len)
                    seg_len_for_pulleys[seg_idx][si] = len
                    seg_l0_for_pulleys[seg_idx][si] = l0
                end
            end
        end
    end

    categories = (
        (:tubular_frame,
            "Tubular frame", 1:19),
        (:te_wires_and_diagonals,
            "TE wires and diagonals", 20:46),
        (:bridles,
            "Bridles", bridle_seg_idxs),
        (:steering_tapes,
            "Steering tapes", steering_tape_idxs),
        (:power_tape,
            "Power tape", power_tape_idxs),
        (:tether,
            "Tether", 90:95),
    )

    ratio_by_category = Dict{Symbol,Matrix{Float64}}()

    for (key, label, seg_idxs) in categories
        ratios = fill(NaN, n_samples, length(seg_idxs))
        l0_used = fill(NaN, n_samples, length(seg_idxs))
        @inbounds for (si, li) in
                      enumerate(start_idx:length(sl))
            state = sl[li]
            X, Y, Z = state.X, state.Y, state.Z
            for (ci, seg_idx) in enumerate(seg_idxs)
                seg = segments[seg_idx]
                l0 = base_l0[seg_idx]
                if seg_idx in tether_seg_set
                    l0 *= tether_scale[li]
                end
                adj = get(l0_adj, seg_idx, nothing)
                if adj !== nothing
                    if adj isa AbstractVector
                        l0 += adj[min(si, length(adj))]
                    elseif adj isa Real
                        l0 += adj
                    end
                end
                (!isfinite(l0) || l0 <= 0) && continue
                p1, p2 = seg.point_idxs
                dx = X[p2] - X[p1]
                dy = Y[p2] - Y[p1]
                dz = Z[p2] - Z[p1]
                len = sqrt(dx * dx + dy * dy + dz * dz)
                if isfinite(len)
                    ratios[si, ci] = (len - l0) / l0
                    l0_used[si, ci] = l0
                end
            end
        end

        finite_mask = isfinite.(ratios)
        if any(finite_mask)
            elong_mask = finite_mask .& (ratios .> 0)
            comp_mask = finite_mask .& (ratios .< 0)
            me = any(elong_mask) ?
                 mean(ratios[elong_mask]) : NaN
            mc = any(comp_mask) ?
                 mean(ratios[comp_mask]) : NaN
            me_abs = any(elong_mask) ?
                     mean(ratios[elong_mask] .*
                          l0_used[elong_mask]) : NaN
            mc_abs = any(comp_mask) ?
                     mean(ratios[comp_mask] .*
                          l0_used[comp_mask]) : NaN

            max_e, max_e_info, max_e_abs = NaN, missing, NaN
            if any(elong_mask)
                m = copy(ratios)
                m[.!elong_mask] .= -Inf
                max_e, li = findmax(m)
                ci = CartesianIndices(size(ratios))[li]
                r, c = ci[1], ci[2]
                max_e_info = (; sample=r,
                    segment=seg_idxs[c],
                    time=sample_times[r])
                max_e_abs = max_e * l0_used[r, c]
            end

            max_c, max_c_info, max_c_abs = NaN, missing, NaN
            if any(comp_mask)
                m = copy(ratios)
                m[.!comp_mask] .= Inf
                max_c, li = findmin(m)
                ci = CartesianIndices(size(ratios))[li]
                r, c = ci[1], ci[2]
                max_c_info = (; sample=r,
                    segment=seg_idxs[c],
                    time=sample_times[r])
                max_c_abs = max_c * l0_used[r, c]
            end

            e_mean = any(elong_mask) ?
                     "mean = $(round(me_abs, digits=4))" *
                     " [m], $(round(me*100, digits=4)) [%]" :
                     "mean = n/a"
            e_max = any(elong_mask) ?
                    "max  = $(round(max_e_abs, digits=4))" *
                    " [m], $(round(max_e*100, digits=4))" *
                    " [%] seg=$(max_e_info.segment)," *
                    " t=$(round(max_e_info.time, digits=2))" :
                    "max  = n/a"
            c_mean = any(comp_mask) ?
                     "mean = $(round(mc_abs, digits=4))" *
                     " [m], $(round(mc*100, digits=4)) [%]" :
                     "mean = n/a"
            c_max = any(comp_mask) ?
                    "max  = $(round(max_c_abs, digits=4))" *
                    " [m], $(round(max_c*100, digits=4))" *
                    " [%] seg=$(max_c_info.segment)," *
                    " t=$(round(max_c_info.time, digits=2))" :
                    "max  = n/a"

            msg = """
              Elongation
                $e_mean
                $e_max
              Compression
                $c_mean
                $c_max
            """
            @info "$label, last " *
                  "$(round(window_span, digits=2)) s\n$msg"
        else
            @warn "$label: no finite values in window"
        end
        ratio_by_category[key] = ratios
    end

    # Pulley combined stretch
    pulley_ratio = Dict{Int,Vector{Float64}}()
    if !isempty(pulleys)
        np = length(pulleys)
        pmat = fill(NaN, n_samples, np)
        pl0 = fill(NaN, n_samples, np)
        for (ci, pulley) in enumerate(pulleys)
            s1, s2 = Int.(pulley.segment_idxs)
            len1 = get(seg_len_for_pulleys, s1,
                fill(NaN, n_samples))
            len2 = get(seg_len_for_pulleys, s2,
                fill(NaN, n_samples))
            l01 = get(seg_l0_for_pulleys, s1,
                fill(NaN, n_samples))
            l02 = get(seg_l0_for_pulleys, s2,
                fill(NaN, n_samples))
            total_l0 = l01 .+ l02
            total_len = len1 .+ len2
            ratios = (total_len .- total_l0) ./ total_l0
            pmat[:, ci] .= ratios
            pl0[:, ci] .= total_l0
            pulley_ratio[Int(pulley.idx)] = ratios
        end
        ratio_by_category[:pulleys] = pmat

        finite_mask = isfinite.(pmat)
        if any(finite_mask)
            elong_mask = finite_mask .& (pmat .> 0)
            comp_mask = finite_mask .& (pmat .< 0)

            me = any(elong_mask) ?
                 mean(pmat[elong_mask]) : NaN
            mc = any(comp_mask) ?
                 mean(pmat[comp_mask]) : NaN
            me_abs = any(elong_mask) ?
                     mean(pmat[elong_mask] .*
                          pl0[elong_mask]) : NaN
            mc_abs = any(comp_mask) ?
                     mean(pmat[comp_mask] .*
                          pl0[comp_mask]) : NaN

            max_e, max_e_abs = NaN, NaN
            max_e_info = missing
            if any(elong_mask)
                m = copy(pmat)
                m[.!elong_mask] .= -Inf
                max_e, li = findmax(m)
                ci = CartesianIndices(size(pmat))[li]
                r, c = ci[1], ci[2]
                max_e_abs = max_e * pl0[r, c]
                max_e_info = (; sample=r,
                    pulley=Int(pulleys[c].idx),
                    segments=Tuple(
                        Int.(pulleys[c].segment_idxs)),
                    time=sample_times[r])
            end

            max_c, max_c_abs = NaN, NaN
            max_c_info = missing
            if any(comp_mask)
                m = copy(pmat)
                m[.!comp_mask] .= Inf
                max_c, li = findmin(m)
                ci = CartesianIndices(size(pmat))[li]
                r, c = ci[1], ci[2]
                max_c_abs = max_c * pl0[r, c]
                max_c_info = (; sample=r,
                    pulley=Int(pulleys[c].idx),
                    segments=Tuple(
                        Int.(pulleys[c].segment_idxs)),
                    time=sample_times[r])
            end

            e_s = any(elong_mask) ?
                  "mean=$(round(me_abs, digits=4))[m]" *
                  " $(round(me*100, digits=4))[%]" *
                  " | max=$(round(max_e_abs, digits=4))" *
                  "[m] pulley=$(max_e_info.pulley)" : "n/a"
            c_s = any(comp_mask) ?
                  "mean=$(round(mc_abs, digits=4))[m]" *
                  " $(round(mc*100, digits=4))[%]" *
                  " | max=$(round(max_c_abs, digits=4))" *
                  "[m] pulley=$(max_c_info.pulley)" : "n/a"

            @info "Pulleys, last " *
                  "$(round(window_span, digits=2))s" *
                  "\n  Elong: $e_s\n  Comp: $c_s"
        end
    end

    return (window=window, ratio=ratio_by_category,
        pulley_ratio=pulley_ratio)
end

# =============================================================================
# Main execution
# =============================================================================

# Set to "" to auto-select the latest dated directory and log file.
log_name = ""
# log_name =
#     "vw8_lt_270_circles_udp_042__2026_03_06_14_58_08/" *
#     "circles_from_initial_state__up_42_us_15_vw_8_lt_269_g_0_run_002_date_2026_03_06_15_04_40"

log_name =
    "vw8_lt_270_circles_udp_034__2026_03_06_18_56_10/" *
    "circles_from_initial_state__up_34_us_10_vw_8_lt_269_g_0_run_001_date_2026_03_06_18_56_10"

# add an if log is empty, use dir name that has the last date.
# then inside that dir, use the file_name with the last date
if isempty(strip(log_name))
    function last_timestamp_token(name::AbstractString)
        token = ""
        for m in eachmatch(
            r"[0-9]{4}(?:_[0-9]{2}){5}",
            name)
            token = m.match
        end
        return token
    end

    dirs = filter(name -> isdir(joinpath(DATA_DIR, name)),
        readdir(DATA_DIR))
    isempty(dirs) &&
        error("No log directories found in $DATA_DIR")
    latest_dir = last(sort(dirs; by=name ->
        (last_timestamp_token(name), name)))

    log_dir = joinpath(DATA_DIR, latest_dir)
    files = filter(name -> isfile(joinpath(log_dir, name)),
        readdir(log_dir))
    arrow_files = filter(name -> endswith(name, ".arrow"),
        files)
    candidates = isempty(arrow_files) ? files : arrow_files
    isempty(candidates) &&
        error("No log files found in $log_dir")

    latest_file = last(sort(candidates; by=name ->
        (last_timestamp_token(name), name)))
    log_name = joinpath(latest_dir, splitext(latest_file)[1])
    @info "Using latest log_name" log_name
end

lg, sam, u_dp, us, v_wind, lt =
    load_log_and_system(log_name=log_name)

# Stretch analysis with commanded tape adjustments
seg87_nom = Float64(
    sam.sys_struct.segments[V3_STEERING_LEFT_IDX].l0)
seg88_nom = Float64(
    sam.sys_struct.segments[V3_DEPOWER_IDX].l0)
seg89_nom = Float64(
    sam.sys_struct.segments[V3_STEERING_RIGHT_IDX].l0)
left_target_l0, right_target_l0 =
    steering_percentage_to_lengths(us * 100.0)
power_target_l0 = depower_percentage_to_length(u_dp * 100.0)
segment_l0_adjustments = Dict(
    V3_STEERING_LEFT_IDX => left_target_l0 - seg87_nom,
    V3_DEPOWER_IDX => power_target_l0 - seg88_nom,
    V3_STEERING_RIGHT_IDX => right_target_l0 - seg89_nom,
)

stretch_info = compute_line_stretch(lg, sam;
    window_seconds=50.0,
    segment_l0_adjustments=segment_l0_adjustments)

# Plots
fig_time = plot_time_series(lg, sam)
time_series_pdf_path = let
    log_file, log_dir, _ = resolve_log_file(log_name, DATA_DIR)
    joinpath(log_dir, "$(log_file)_time_series.pdf")
end
CairoMakie.save(time_series_pdf_path, fig_time)
@info "Saved time series PDF" time_series_pdf_path

scene = replay(lg, sam.sys_struct;
    autoplay=false, loop=true, show_panes=false)
style_replay_scene!(scene;
    segment_linewidth=REPLAY_SEGMENT_LINEWIDTH,
    other_linewidth=REPLAY_OTHER_LINEWIDTH,
    thick_segment_ids=REPLAY_THICK_SEGMENT_IDS,
    thick_segment_linewidth=REPLAY_THICK_SEGMENT_LINEWIDTH,
    thick_segment_color=REPLAY_THICK_SEGMENT_COLOR)
fig_wing = print_and_plot_wing(lg, sam, is_print=true)

scr1 = display(fig_time)
wait(scr1)
scr2 = display(scene)
wait(scr2)
scr3 = display(fig_wing)
wait(scr3)

# print_gk_window_summary(lg, sam; udp=u_dp, window_sec=1.0)

nothing
