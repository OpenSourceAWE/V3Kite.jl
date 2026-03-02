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
    V3_DEPOWER_IDX, V3_STEERING_GAIN
using GLMakie
using LinearAlgebra
using Statistics

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = joinpath(
    dirname(@__DIR__),
    "processed_data")

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

function load_log_and_system(; log_name)
    m = match(
        r"_up_([0-9]+)_us_([0-9._-]+)_vw_([0-9]+)_lt_([0-9]+)",
        log_name)
    m === nothing && error(
        "Could not parse up/us/vw/lt from: $log_name")
    up = parse(Float64, m.captures[1])
    us_tokens = split(m.captures[2], "_")
    us_vals = parse.(Float64, us_tokens)
    v_wind = parse(Int, m.captures[3])
    lt = parse(Int, m.captures[4])
    @info "Parsed tags" up = up / 100 us = us_vals ./ 100 v_wind lt

    config = V3SimConfig(
        struc_yaml_path="struc_geometry.yaml",
        aero_yaml_path="aero_geometry.yaml",
        vsm_settings_path="vsm_settings.yaml",
        v_wind=Float64(v_wind),
        tether_length=Float64(lt),
        wing_type=REFINE,
    )
    sam, sys = create_v3_model(config)
    apply_geom_adjustments!(sys, V3GeomAdjustConfig(
        reduce_te=true))

    log_file, log_dir, log_path = resolve_log_file(
        log_name, DATA_DIR)
    @info "Resolved log file" log_path
    lg = load_log(log_file; path=log_dir)
    return lg, sam, up, us_vals, v_wind, lt
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

function plot_time_series(lg, sam)
    return plot(sam.sys_struct, lg;
        plot_turn_rates=false, plot_reelout=false,
        plot_twist=false,
        plot_yaw_rate_paper=false,
        yaw_rate_paper_ylims=(-90.0, 90.0),
        yaw_rate_paper_compare=false,
        plot_v_app=true, plot_kite_vel=false,
        plot_gk=false,
        plot_aoa=true,
        ylims=Dict(:aoa => (0.0, 15.0), :gk => (0.0, 15.0)),
        plot_heading=false, plot_elevation=true,
        plot_azimuth=true, plot_winch_force=false,
        plot_set_values=false, plot_us=true,
        plot_tether_actual=false,
        plot_turn_radius=true,
        turn_radius_ylims=(0.0, 40.0),
        plot_cs=true, cs_ylims=(0.0, 0.02))
end

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
#     "zenith_2019_batch_2026_02_23_15_09_48/" *
#     "hold_at_zenith_then_circles__up_18_us_0_vw_9_lt_268_el_50_g_0_run_001_date_2026_02_23_15_09_53"

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

lg, sam, up, us, v_wind, lt =
    load_log_and_system(log_name=log_name)

# Stretch analysis with commanded tape adjustments
up_fraction = up / 100
us_fraction = us / 100
seg87_nom = Float64(
    sam.sys_struct.segments[V3_STEERING_LEFT_IDX].l0)
seg88_nom = Float64(
    sam.sys_struct.segments[V3_DEPOWER_IDX].l0)
seg89_nom = Float64(
    sam.sys_struct.segments[V3_STEERING_RIGHT_IDX].l0)
steering_tape_change = V3_STEERING_GAIN * us_fraction
power_target_l0 = 0.2 + 5.0 * up_fraction
segment_l0_adjustments = Dict(
    V3_STEERING_LEFT_IDX => steering_tape_change,
    V3_DEPOWER_IDX => power_target_l0 - seg88_nom,
    V3_STEERING_RIGHT_IDX => -steering_tape_change,
)

stretch_info = compute_line_stretch(lg, sam;
    window_seconds=50.0,
    segment_l0_adjustments=segment_l0_adjustments)

# Plots
fig_time = plot_time_series(lg, sam)
scene = replay(lg, sam.sys_struct;
    autoplay=false, loop=true, show_panes=false)
fig_wing = print_and_plot_wing(lg, sam, is_print=true)

scr1 = display(fig_time)
wait(scr1)
scr2 = display(scene)
wait(scr2)
scr3 = display(fig_wing)
wait(scr3)

nothing
