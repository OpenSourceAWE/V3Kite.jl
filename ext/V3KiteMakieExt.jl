# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3Kite Makie Extension

Provides visualization functions for V3 kite simulation results when GLMakie is available.
"""
module V3KiteMakieExt

using V3Kite
using GLMakie
using LinearAlgebra
using LaTeXStrings
import V3Kite: SymbolicAWEModels
using VortexStepMethod: calculate_projected_area

const PLOT_COLORS = [:blue, :green, :orange,
    :purple, :cyan, :magenta]

"""
Draw photogrammetry group lines and scatter points.
LE lines are opaque+thick, strut inner points are
transparent.
"""
function _draw_extra_groups!(ax, coords, groups;
        point_size=8)
    te_idxs = Set{Int}()
    for (gname, indices) in groups
        is_le = gname == "LE"
        is_strut = startswith(gname, "strut")
        clr = is_le ? :red : (:red, 0.6)
        lw = is_le ? 5 : 3
        for i in 1:(length(indices)-1)
            c1 = coords[indices[i]]
            c2 = coords[indices[i+1]]
            lines!(ax, [c1[1], c2[1]],
                [c1[2], c2[2]];
                color=clr, linewidth=lw)
        end
        if is_strut
            push!(te_idxs, indices[1])
        end
    end

    # Strut inner indices (transparent)
    strut_inner = Set{Int}()
    for (gname, indices) in groups
        if startswith(gname, "strut")
            for idx in indices[2:end]
                push!(strut_inner, idx)
            end
        end
    end
    # Opaque points (LE, TE, other)
    opaque = [i for i in eachindex(coords)
              if i ∉ strut_inner]
    scatter!(ax,
        [coords[i][1] for i in opaque],
        [coords[i][2] for i in opaque];
        markersize=point_size,
        color=:red, marker=:circle)
    # Transparent strut inner points
    if !isempty(strut_inner)
        inner = collect(strut_inner)
        scatter!(ax,
            [coords[i][1] for i in inner],
            [coords[i][2] for i in inner];
            markersize=point_size,
            color=(:red, 0.4), marker=:circle)
    end
end

"""
    plot_body_frame_local(sys_structs; extra_points, extra_groups, dir, labels)

Plot wing points in 2D body frame. Accepts single sys_struct or vector of them.
Extra points connected per-strut and LE.

# Arguments
- `sys_structs`: System structure or vector of system structures to plot
- `extra_points`: Optional vector of (x,y,z) tuples from load_extra_points
- `extra_groups`: Optional groups from load_extra_points
- `dir::Symbol`: Viewing direction (:side, :front, or :top)
- `labels`: Optional vector of labels for each sys_struct
- `point_idxs`: Optional vector of point indices to plot (default: WING points only)
- `point_size`: Size of simulation points (default: 10)
- `extra_point_size`: Size of extra points (default: 8)
- `figsize`: Figure size tuple (default: (800, 600))
- `legend`: Show legend (default: true)
- `title`: Show title (default: true)
- `show_point_idxs`: Show point index labels (default: true)
- `show_aoa`: Show geometric AoA panel below (default: false)
"""
function V3Kite.plot_body_frame_local(sys_structs;
                               extra_points=nothing,
                               extra_groups=nothing,
                               dir::Symbol=:front,
                               point_size=10,
                               extra_point_size=8,
                               figsize=(800, 600),
                               labels=nothing,
                               point_idxs=nothing,
                               legend=true,
                               title=true,
                               show_point_idxs=false,
                               show_aoa=false)
    # Normalize to vector
    structs = sys_structs isa Vector ? sys_structs : [sys_structs]
    n_structs = length(structs)

    # Default labels
    if isnothing(labels)
        labels = n_structs == 1 ? ["sim"] : ["sim_$i" for i in 1:n_structs]
    end

    # Set up axis labels
    if dir == :top
        xlabel, ylabel = "y [m]", "x [m]"
    elseif dir == :side
        xlabel, ylabel = "x [m]", "z [m]"
    else  # :front
        xlabel, ylabel = "y [m]", "z [m]"
    end

    aoa_figsize = show_aoa ?
        (figsize[1], figsize[2] + 150) : figsize
    fig = Figure(size=aoa_figsize)
    ax_title = title ? "Wing Points (Body Frame)" : ""
    ax = Axis(fig[1, 1]; xlabel, ylabel,
              title=ax_title, aspect=DataAspect())
    ax_aoa = nothing
    if show_aoa
        ax_aoa = Axis(fig[2, 1];
            xlabel="y [m]",
            ylabel="Geo. AoA [deg]")
        rowsize!(fig.layout, 2, Fixed(150))
    end

    function get_2d(pos_b)
        if dir == :top
            return (pos_b[2], pos_b[1])
        elseif dir == :side
            return (pos_b[1], pos_b[3])
        else
            return (pos_b[2], pos_b[3])
        end
    end

    # Collect all coordinates for auto-zoom
    all_x_vals = Float64[]
    all_y_vals = Float64[]
    sim_aoa_data = []

    # Plot each sys_struct
    for (s_idx, sys_struct) in enumerate(structs)
        points = sys_struct.points
        wings = sys_struct.wings
        segments = sys_struct.segments
        color = PLOT_COLORS[mod1(s_idx, length(PLOT_COLORS))]

        # Update pos_b for REFINE wing points
        for wing in wings
            if wing.wing_type == SymbolicAWEModels.REFINE
                R_w_b = V3Kite.calc_R_b_w(sys_struct)'
                for point in points
                    if point.wing_idx == wing.idx
                        point.pos_b .= R_w_b * (point.pos_w - wing.pos_w)
                    end
                end
            end
        end

        # Geometric AoA for WING point pairs
        if show_aoa
            wing_pts = sort(
                [p for p in points
                 if p.type == SymbolicAWEModels.WING],
                by=p -> p.idx)
            span_ys = Float64[]
            aoas = Float64[]
            le_pos = [wing_pts[i].pos_b
                      for i in 1:2:length(wing_pts)]
            n_le = length(le_pos)
            body_x = [1.0, 0.0, 0.0]
            for k in 1:n_le
                le = wing_pts[2k - 1]
                te = wing_pts[2k]
                chord_b = te.pos_b - le.pos_b
                y_airf = if k == 1
                    normalize(le_pos[2] - le_pos[1])
                elseif k == n_le
                    normalize(le_pos[n_le] - le_pos[n_le-1])
                else
                    normalize(le_pos[k+1] - le_pos[k-1])
                end
                z_loc = normalize(cross(body_x, y_airf))
                push!(aoas, rad2deg(atan(
                    dot(chord_b, z_loc),
                    dot(chord_b, body_x))))
                push!(span_ys,
                    (le.pos_b[2] + te.pos_b[2]) / 2)
            end
            push!(sim_aoa_data,
                (span_ys, aoas, color,
                 labels[s_idx]))
        end

        # Select points to plot: use point_idxs if provided, otherwise WING points
        if isnothing(point_idxs)
            plot_points = [p for p in points if p.type == SymbolicAWEModels.WING]
        else
            plot_points = [points[i] for i in point_idxs if i <= length(points)]
        end

        # Extract 2D coords based on viewing direction
        if dir == :top
            coords = [(p.pos_b[2], p.pos_b[1]) for p in plot_points]
        elseif dir == :side
            coords = [(p.pos_b[1], p.pos_b[3]) for p in plot_points]
        else  # :front
            coords = [(p.pos_b[2], p.pos_b[3]) for p in plot_points]
        end

        x_vals = [c[1] for c in coords]
        y_vals = [c[2] for c in coords]
        append!(all_x_vals, x_vals)
        append!(all_y_vals, y_vals)

        # Plot segments (skip TE 20-28 and diagonals 29-46)
        plot_point_idxs = Set(p.idx for p in plot_points)
        for seg in segments
            if 20 <= seg.idx <= 46
                continue
            end
            from_idx, to_idx = seg.point_idxs
            if from_idx in plot_point_idxs && to_idx in plot_point_idxs
                p1 = points[from_idx]
                p2 = points[to_idx]
                c1 = get_2d(p1.pos_b)
                c2 = get_2d(p2.pos_b)
                is_le = seg.idx <= 9
                lw = is_le ? 5 : 3
                clr = is_le ? color : (color, 0.5)
                lines!(ax, [c1[1], c2[1]], [c1[2], c2[2]];
                       color=clr, linewidth=lw)
            end
        end

        # Plot wing points
        scatter!(ax, x_vals, y_vals;
                 markersize=point_size, color=color, marker=:circle)

        # Add point labels only for first struct
        if show_point_idxs && s_idx == 1
            for (i, p) in enumerate(plot_points)
                px, py = coords[i]
                away_x, away_y = 0.0, 0.0
                for (j, _) in enumerate(plot_points)
                    if i != j
                        ox, oy = coords[j]
                        dx, dy = px - ox, py - oy
                        dist = sqrt(dx^2 + dy^2) + 0.01
                        away_x += dx / dist^2
                        away_y += dy / dist^2
                    end
                end
                away_len = sqrt(away_x^2 + away_y^2)
                if away_len > 0
                    away_x /= away_len
                    away_y /= away_len
                else
                    away_x, away_y = 1.0, 1.0
                end
                offset = (12 * sign(away_x), 12 * sign(away_y))
                align_x = away_x >= 0 ? :left : :right
                align_y = away_y >= 0 ? :bottom : :top
                text!(ax, px, py; text=string(p.idx), fontsize=12,
                      align=(align_x, align_y), offset=offset)
            end
        end
    end

    # Plot extra points with connections (use first struct's wing for transform)
    if !isnothing(extra_points) && !isnothing(extra_groups)
        wing = structs[1].wings[1]
        R_w_b = V3Kite.calc_R_b_w(structs[1])'
        extra_body = [R_w_b * (collect(p) - wing.pos_w) for p in extra_points]

        if dir == :top
            extra_coords = [(p[2], p[1]) for p in extra_body]
        elseif dir == :side
            extra_coords = [(p[1], p[3]) for p in extra_body]
        else
            extra_coords = [(p[2], p[3]) for p in extra_body]
        end

        _draw_extra_groups!(ax, extra_coords,
            extra_groups;
            point_size=extra_point_size)

        # Photogrammetry geometric AoA
        if show_aoa && !isnothing(ax_aoa)
            le_idxs = Int[]
            strut_groups =
                Tuple{String,Vector{Int}}[]
            for (gname, indices) in extra_groups
                if gname == "LE"
                    le_idxs = indices
                elseif startswith(gname, "strut")
                    push!(strut_groups,
                        (gname, indices))
                end
            end
            if !isempty(le_idxs)
                le_body = [extra_body[i]
                           for i in le_idxs]
                n_le = length(le_body)
                body_x = [1.0, 0.0, 0.0]
                photo_span = Float64[]
                photo_aoa = Float64[]
                for (_, indices) in strut_groups
                    te_b = extra_body[indices[1]]
                    le_b = extra_body[indices[end]]
                    chord = te_b - le_b
                    _, best_idx = findmin(
                        lp -> norm(lp - le_b), le_body)
                    y_airf = if best_idx == 1
                        normalize(le_body[2] - le_body[1])
                    elseif best_idx == n_le
                        normalize(
                            le_body[n_le] - le_body[n_le-1])
                    else
                        normalize(
                            le_body[best_idx+1] -
                            le_body[best_idx-1])
                    end
                    z_loc = normalize(cross(body_x, y_airf))
                    push!(photo_aoa, -rad2deg(atan(
                        dot(chord, z_loc),
                        dot(chord, body_x))))
                    push!(photo_span,
                        (te_b[2] + le_b[2]) / 2)
                end
                perm = sortperm(photo_span)
                lines!(ax_aoa,
                    photo_span[perm],
                    photo_aoa[perm];
                    color=:red, linewidth=2,
                    label="photogrammetry")
                scatter!(ax_aoa,
                    photo_span, photo_aoa;
                    color=:red, markersize=8)
            end
        end
    end

    # Auto-zoom with margin
    if !isempty(all_x_vals)
        x_min, x_max = extrema(all_x_vals)
        y_min, y_max = extrema(all_y_vals)
        margin_x = 0.15 * (x_max - x_min) + 0.3
        margin_y = 0.15 * (y_max - y_min) + 0.3
        if dir == :top
            limits!(ax, x_min - margin_x, x_max + margin_x,
                        y_max + margin_y, y_min - margin_y)
        else
            limits!(ax, x_min - margin_x, x_max + margin_x,
                        y_min - margin_y, y_max + margin_y)
        end
    end

    # Plot sim AoA curves
    if show_aoa && !isnothing(ax_aoa)
        for (span_ys, aoas, clr, lbl) in sim_aoa_data
            perm = sortperm(span_ys)
            lines!(ax_aoa,
                span_ys[perm], aoas[perm];
                color=clr, linewidth=2, label=lbl)
            scatter!(ax_aoa, span_ys, aoas;
                color=clr, markersize=8)
        end
        if dir == :front
            linkxaxes!(ax, ax_aoa)
        end
    end

    # Legend
    if legend
        legend_elements = [
            MarkerElement(color=PLOT_COLORS[mod1(i, length(PLOT_COLORS))],
                          marker=:circle, markersize=10)
            for i in 1:n_structs
        ]
        legend_labels = copy(labels)
        if !isnothing(extra_points)
            push!(legend_elements,
                  MarkerElement(color=:red, marker=:circle, markersize=10))
            push!(legend_labels, "photogrammetry")
        end
        Legend(fig[1, 2], legend_elements, legend_labels)
    end

    return fig
end

"""
    plot_geom_aoa_dist(sys_structs; extra_points,
        extra_groups, labels, figsize)

Plot geometric AoA distribution along the span.
Computes local AoA at each LE/TE pair from sim
sys_structs and optionally from photogrammetry data.

# Arguments
- `sys_structs`: System structure or vector of them
- `extra_points`: Optional photogrammetry points
- `extra_groups`: Optional photogrammetry groups
- `labels`: Optional vector of labels
- `figsize`: Figure size (default: (800, 300))
"""
function V3Kite.plot_geom_aoa_dist(sys_structs;
        extra_points=nothing,
        extra_groups=nothing,
        labels=nothing,
        figsize=(800, 300),
        title=true,
        legend=true)
    structs = sys_structs isa Vector ?
        sys_structs : [sys_structs]
    n_structs = length(structs)
    if isnothing(labels)
        labels = n_structs == 1 ? ["sim"] :
            ["sim_$i" for i in 1:n_structs]
    end

    fig = Figure(size=figsize)
    ax_title = title ? "Geometric AoA Distribution" : ""
    ax = Axis(fig[1, 1];
        xlabel="y [m]", ylabel="Geo. AoA [deg]",
        title=ax_title)

    # Sim AoA per sys_struct
    for (s_idx, sys_struct) in enumerate(structs)
        points = sys_struct.points
        color = PLOT_COLORS[
            mod1(s_idx, length(PLOT_COLORS))]

        # Update pos_b for REFINE wing points
        for wing in sys_struct.wings
            if wing.wing_type == SymbolicAWEModels.REFINE
                R_w_b = V3Kite.calc_R_b_w(sys_struct)'
                for point in points
                    if point.wing_idx == wing.idx
                        point.pos_b .= R_w_b * (
                            point.pos_w - wing.pos_w)
                    end
                end
            end
        end

        wing_pts = sort(
            [p for p in points
             if p.type == SymbolicAWEModels.WING],
            by=p -> p.idx)
        le_pos = [wing_pts[i].pos_b
                  for i in 1:2:length(wing_pts)]
        n_le = length(le_pos)
        body_x = [1.0, 0.0, 0.0]
        span_ys = Float64[]
        aoas = Float64[]
        for k in 1:n_le
            le = wing_pts[2k - 1]
            te = wing_pts[2k]
            chord_b = te.pos_b - le.pos_b
            y_airf = if k == 1
                normalize(le_pos[2] - le_pos[1])
            elseif k == n_le
                normalize(
                    le_pos[n_le] - le_pos[n_le-1])
            else
                normalize(
                    le_pos[k+1] - le_pos[k-1])
            end
            z_loc = normalize(cross(body_x, y_airf))
            push!(aoas, rad2deg(atan(
                dot(chord_b, z_loc),
                dot(chord_b, body_x))))
            push!(span_ys,
                (le.pos_b[2] + te.pos_b[2]) / 2)
        end
        perm = sortperm(span_ys)
        lines!(ax, span_ys[perm], aoas[perm];
            color, linewidth=2, label=labels[s_idx])
        scatter!(ax, span_ys, aoas;
            color, markersize=8)
    end

    # Photogrammetry AoA
    if !isnothing(extra_points) &&
            !isnothing(extra_groups)
        wing = structs[1].wings[1]
        R_w_b = V3Kite.calc_R_b_w(structs[1])'
        extra_body = [R_w_b * (collect(p) - wing.pos_w)
                      for p in extra_points]
        le_idxs = Int[]
        strut_groups = Tuple{String,Vector{Int}}[]
        for (gname, indices) in extra_groups
            if gname == "LE"
                le_idxs = indices
            elseif startswith(gname, "strut")
                push!(strut_groups,
                    (gname, indices))
            end
        end
        if !isempty(le_idxs)
            le_body = [extra_body[i]
                       for i in le_idxs]
            # Collect strut stations
            stations = [(
                te_b=extra_body[indices[1]],
                le_b=extra_body[indices[end]])
                for (_, indices) in strut_groups]
            # Sort by span position
            span_pos = [(s.te_b[2] + s.le_b[2]) / 2
                        for s in stations]
            perm = sortperm(span_pos)
            stations = stations[perm]
            span_pos = span_pos[perm]
            # Compute AoA per station
            body_x = [1.0, 0.0, 0.0]
            n_st = length(stations)
            photo_aoa = map(1:n_st) do i
                chord = stations[i].te_b -
                        stations[i].le_b
                y_airf = if i == 1
                    normalize(stations[2].le_b -
                        stations[1].le_b)
                elseif i == n_st
                    normalize(stations[n_st].le_b -
                        stations[n_st-1].le_b)
                else
                    normalize(
                        stations[i+1].le_b -
                        stations[i-1].le_b)
                end
                z_loc = normalize(
                    cross(body_x, y_airf))
                -rad2deg(atan(dot(chord, z_loc),
                    dot(chord, body_x)))
            end
            lines!(ax, span_pos, photo_aoa;
                color=:red, linewidth=2,
                label="photogrammetry")
            scatter!(ax, span_pos, photo_aoa;
                color=:red, markersize=8)
        end
    end

    if legend
        axislegend(ax; position=:rt)
    end
    return fig
end

"""
    plot_photogrammetry(points, groups; dir, kwargs...)

Plot photogrammetry points in 2D without a sys_struct.
Projects 3D points based on `dir` and draws group lines
and scatter points.

# Arguments
- `points`: Vector of (x, y, z) tuples
- `groups`: Vector of (group_name, indices) from
  `load_extra_points`
- `dir::Symbol`: Viewing direction (:side, :front, :top)
- `point_size`: Marker size (default: 8)
- `figsize`: Figure size (default: (800, 600))
"""
function V3Kite.plot_photogrammetry(points, groups;
        dir::Symbol=:front,
        point_size=8,
        figsize=(800, 600))
    xlabel, ylabel = if dir == :top
        ("x [m]", "y [m]")
    elseif dir == :side
        ("x [m]", "z [m]")
    else
        ("y [m]", "z [m]")
    end
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; xlabel, ylabel,
        aspect=DataAspect())
    coords = [dir == :top ? (p[1], p[2]) :
              dir == :side ? (p[1], p[3]) :
              (p[2], p[3]) for p in points]
    _draw_extra_groups!(ax, coords, groups;
        point_size)
    return fig
end

"""
    plot_yaw_rate_vs_steering(syslogs, tapes; labels, figsize)

Scatter plot of |yaw rate| vs |u_s * v_a| for one or more logs.

# Arguments
- `syslogs`: Single syslog or vector of syslogs
- `tapes`: Matching tape(s) with `.steering` and `.time` fields
- `labels`: Optional vector of series labels
- `figsize`: Figure size tuple (default: (600, 400))
"""
function V3Kite.plot_yaw_rate_vs_steering(
        syslogs, tapes;
        labels=nothing, figsize=(600, 400),
        min_steering=0.0, dt=0.01)
    logs = syslogs isa Vector ? syslogs : [syslogs]
    tps = tapes isa Vector ? tapes : [tapes]
    n = length(logs)

    if isnothing(labels)
        labels = n == 1 ? ["series"] :
            ["series_$i" for i in 1:n]
    end

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1];
        xlabel=L"|u_s \cdot v_a| \; [m/s]",
        ylabel=L"|\dot{\psi}| \; [rad/s]")

    has_data = false
    for (i, (lg, tape)) in enumerate(zip(logs, tps))
        sl = hasproperty(lg, :syslog) ? lg.syslog : lg

        # Unwrap heading and compute yaw rate
        hw = copy(sl.heading)
        for j in 2:length(hw)
            while hw[j] - hw[j-1] > pi
                hw[j] -= 2pi
            end
            while hw[j] - hw[j-1] < -pi
                hw[j] += 2pi
            end
        end
        yaw_rate = diff(hw) ./ dt

        us = tape.steering[2:end]
        mask = abs.(us) .> min_steering
        x = abs.(us[mask] .* sl.v_app[2:end][mask])
        y = abs.(yaw_rate[mask])
        isempty(x) && continue
        has_data = true
        color = PLOT_COLORS[mod1(i, length(PLOT_COLORS))]
        scatter!(ax, x, y; markersize=4, color=color,
            label=labels[i])

        # Best fit line through origin: gk = sum(x.*y) / sum(x.^2)
        gk = dot(x, y) / dot(x, x)
        x_fit = range(0, maximum(x); length=50)
        lines!(ax, collect(x_fit), gk .* collect(x_fit);
            color=color, linewidth=2,
            label="$(labels[i]) gk=$(round(gk; digits=2))")
    end

    if has_data
        axislegend(ax; position=:lt)
    end
    return fig
end

# =====================================================================
# plot_replay — custom time-series panels for V3 kite replay data
# =====================================================================

"""
    plot_replay(syss, logs; tape_lengths, suffixes, size)

Custom time-series plot for V3 kite replay data with panels for
tether force, apparent wind, wind speed, kite velocity, force
coefficient, steering gain, and steering input.
"""
function V3Kite.plot_replay(
        syss::Vector{<:SymbolicAWEModels.SystemStructure},
        logs::Vector{<:SymbolicAWEModels.KiteUtils.SysLog};
        tape_lengths=nothing,
        suffixes=nothing,
        size=(1200, 800))

    n = length(logs)
    actual_suffixes = if n == 1
        [""]
    elseif isnothing(suffixes)
        [" ($i)" for i in 1:n]
    else
        [" (" * s * ")" for s in suffixes]
    end

    panels = []

    # --- Tether force panel ---
    all_data, all_labels, all_times = [], [], []
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        wf = [sl.winch_force[k][1]
              for k in eachindex(sl.winch_force)]
        push!(all_data, wf)
        push!(all_labels, L"F_t" * actual_suffixes[i])
        push!(all_times, sl.time)
    end
    push!(panels, (data=all_data, labels=all_labels,
        times=all_times, ylabel=L"F_t \; [N]"))

    # --- Apparent wind speed panel ---
    all_data, all_labels, all_times = [], [], []
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        push!(all_data, collect(sl.v_app))
        push!(all_labels, L"v_a" * actual_suffixes[i])
        push!(all_times, sl.time)
    end
    push!(panels, (data=all_data, labels=all_labels,
        times=all_times, ylabel=L"v_a \; [m/s]"))

    # --- Wind speed panel ---
    all_data, all_labels, all_times = [], [], []
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        vw = [sl.v_wind_gnd[k][1]
              for k in eachindex(sl.v_wind_gnd)]
        push!(all_data, vw)
        push!(all_labels, L"v_w" * actual_suffixes[i])
        push!(all_times, sl.time)
    end
    push!(panels, (data=all_data, labels=all_labels,
        times=all_times, ylabel=L"v_w \; [m/s]"))

    # --- Wind direction panel ---
    all_data, all_labels, all_times = [], [], []
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        wd = [rad2deg(atan(sl.v_wind_gnd[k][2],
                           sl.v_wind_gnd[k][1]))
              for k in eachindex(sl.v_wind_gnd)]
        push!(all_data, wd)
        push!(all_labels,
            L"\theta_w" * actual_suffixes[i])
        push!(all_times, sl.time)
    end
    push!(panels, (data=all_data, labels=all_labels,
        times=all_times,
        ylabel=L"\theta_w \; [°]"))

    # --- Kite velocity panel ---
    all_data, all_labels, all_times = [], [], []
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        vk = [norm(sl.vel_kite[k])
              for k in eachindex(sl.vel_kite)]
        push!(all_data, vk)
        push!(all_labels, L"v_k" * actual_suffixes[i])
        push!(all_times, sl.time)
    end
    push!(panels, (data=all_data, labels=all_labels,
        times=all_times, ylabel=L"v_k \; [m/s]"))

    # --- Force coefficient panel ---
    all_data, all_labels, all_times = [], [], []
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        A_proj = calculate_projected_area(
            syss[i].wings[1].vsm_wing)
        cf = [sl.winch_force[k][1] /
              (0.5 * 1.225 * sl.v_app[k]^2 * A_proj)
              for k in eachindex(sl.winch_force)]
        push!(all_data, cf)
        push!(all_labels, L"C_F" * actual_suffixes[i])
        push!(all_times, sl.time)
    end
    push!(panels, (data=all_data, labels=all_labels,
        times=all_times, ylabel=L"C_F \; [-]"))

    # --- gk panel (only when tape_lengths provided) ---
    if !isnothing(tape_lengths)
        all_data, all_labels, all_times = [], [], []
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            tl = tape_lengths[i]

            # Interpolate tape steering to syslog times
            us_pct = similar(sl.time)
            for k in eachindex(us_pct)
                t = sl.time[k]
                idx = searchsortedfirst(tl.time, t)
                idx = clamp(idx, 1, length(tl.steering))
                us_pct[k] = tl.steering[idx]
            end

            # Unwrap heading, compute heading rate
            hw = copy(sl.heading)
            for j in 2:length(hw)
                while hw[j] - hw[j-1] > pi
                    hw[j] -= 2pi
                end
                while hw[j] - hw[j-1] < -pi
                    hw[j] += 2pi
                end
            end
            heading_rate = (diff(rad2deg.(hw))
                            ./ diff(sl.time))
            v_app = sl.v_app[2:end]
            us_seg = us_pct[2:end]

            gk = similar(heading_rate)
            for k in eachindex(gk)
                gk[k] = abs(us_seg[k]) > 0.01 ?
                    heading_rate[k] /
                        (v_app[k] * us_seg[k]) :
                    NaN
            end

            push!(all_data, gk)
            push!(all_labels,
                L"g_k" * actual_suffixes[i])
            push!(all_times, sl.time[2:end])
        end
        push!(panels, (data=all_data, labels=all_labels,
            times=all_times, ylabel=L"g_k \; [-]"))
    end

    # --- Steering panel (only when tape_lengths) ---
    if !isnothing(tape_lengths)
        all_data, all_labels, all_times = [], [], []
        for (i, _) in enumerate(logs)
            tl = tape_lengths[i]
            push!(all_data, collect(tl.steering .* 100))
            push!(all_labels,
                L"u_s" * actual_suffixes[i])
            push!(all_times, collect(tl.time))
        end
        push!(panels, (data=all_data, labels=all_labels,
            times=all_times,
            ylabel=L"u_s \; [\%]"))
    end

    # --- Render panels ---
    n_panels = length(panels)
    fig = Figure(; size)
    axes = Axis[]
    label_fontsize = 16
    ticklabelsize = 12

    for (i, panel) in enumerate(panels)
        if i == 1
            ax = Axis(fig[i, 1];
                ylabel=panel.ylabel,
                ylabelsize=label_fontsize,
                xticklabelsize=ticklabelsize,
                yticklabelsize=ticklabelsize)
        else
            ax = Axis(fig[i, 1];
                ylabel=panel.ylabel,
                ylabelsize=label_fontsize,
                xticklabelsvisible=false,
                xticklabelsize=ticklabelsize,
                yticklabelsize=ticklabelsize)
            linkxaxes!(axes[1], ax)
        end

        for (j, (data_series, label, time_vec)) in
                enumerate(zip(panel.data,
                    panel.labels, panel.times))
            if length(data_series) != length(time_vec)
                @warn "Skipping '$label': " *
                    "len $(length(data_series)) " *
                    "!= $(length(time_vec))"
                continue
            end
            lines!(ax, time_vec, data_series;
                label=label)
        end

        if length(panel.data) > 1
            axislegend(ax; position=:rt,
                labelsize=10, patchsize=(10, 5))
        end

        push!(axes, ax)
    end

    # x-label on bottom axis only
    axes[end].xlabel = L"t \; [s]"
    axes[end].xlabelsize = label_fontsize
    axes[end].xticklabelsvisible = true

    Makie.resize_to_layout!(fig)
    return fig
end

# =====================================================================
# plot_sphere_trajectory — trajectories on unit sphere with body axes
# =====================================================================

"""
    plot_sphere_trajectory(logs; radius, colors, labels, kwargs...)

Plot kite trajectories on a unit sphere. Adds body-frame axes
(red=x, green=y, blue=z) at the final position of each trajectory.
"""
function V3Kite.plot_sphere_trajectory(
        logs::Vector{<:SymbolicAWEModels.KiteUtils.SysLog};
        radius=1.0,
        colors=nothing,
        labels=nothing,
        sphere_alpha=0.2,
        linewidth=2.0,
        size=(800, 800))

    fig = Figure(; size)
    ax = LScene(fig[1, 1], show_axis=false)
    ax.scene.camera.projection[] =
        Makie.orthographicprojection(
            -2f0, 2f0, -2f0, 2f0, -10f0, 10f0)

    # Semi-transparent sphere
    sphere_mesh = Sphere(Point3f(0, 0, 0),
        Float32(radius))
    mesh!(ax, sphere_mesh;
        color=(:gray, sphere_alpha), transparency=true)

    # Reference axes
    axis_len = radius * 1.3
    scatter!(ax, [Point3f(0, 0, 0)];
        color=:black, markersize=10)
    lines!(ax, [Point3f(0, 0, 0),
        Point3f(axis_len, 0, 0)];
        color=:red, linewidth=3)
    text!(ax, Point3f(axis_len * 1.1, 0, 0);
        text="X (az=0)", fontsize=14, color=:red)
    lines!(ax, [Point3f(0, 0, 0),
        Point3f(0, axis_len, 0)];
        color=:green, linewidth=3)
    text!(ax, Point3f(0, axis_len * 1.1, 0);
        text="Y (az=-90)", fontsize=14, color=:green)
    lines!(ax, [Point3f(0, 0, 0),
        Point3f(0, 0, axis_len)];
        color=:blue, linewidth=3)
    text!(ax, Point3f(0, 0, axis_len * 1.1);
        text="Z (el=90)", fontsize=14, color=:blue)

    default_colors = [:blue, :red, :green,
        :orange, :purple, :cyan]
    actual_colors = isnothing(colors) ?
        default_colors : colors

    arrow_scale = 0.3 * radius

    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        elevation = sl.elevation
        azimuth = sl.azimuth

        # ENU frame, azimuth_east convention
        x = radius .* cos.(elevation) .* cos.(azimuth)
        y = -radius .* cos.(elevation) .* sin.(azimuth)
        z = radius .* sin.(elevation)

        color = actual_colors[
            mod1(i, length(actual_colors))]
        label = isnothing(labels) ?
            "trajectory $i" : labels[i]
        lines!(ax, x, y, z; color, linewidth, label)

        # Body-frame axes at final position
        orient_last = sl.orient[end]
        R_b_w = SymbolicAWEModels.quaternion_to_rotation_matrix(
            orient_last)
        pos = Point3f(x[end], y[end], z[end])

        for (col, arrow_color) in enumerate(
                (:red, :green, :blue))
            dir = Vec3f(R_b_w[:, col]...) * arrow_scale
            arrows3d!(ax, [pos], [dir];
                color=arrow_color, shaftradius=0.01 * radius,
                tipradius=0.02 * radius,
                tiplength=0.05 * radius)
        end
    end

    if length(logs) > 1 || !isnothing(labels)
        Legend(fig[1, 2], ax)
    end
    return fig
end

# =====================================================================
# plot_2d_trajectory — y vs z colored by gradient
# =====================================================================

"""
    plot_2d_trajectory(logs; gradient=:vel, tapes=nothing,
        labels, colormap, size,
        show_steering, show_winch_force, show_v_app,
        show_drag_coeff)

Plot kite y vs z position colored by a gradient quantity,
with optional time-series subplots below.

# Arguments
- `gradient=:vel`: color by `:vel` or `:steering`
- `tapes`: vector of named tuples with `time`, `steering`
- `labels`: legend labels per log
- `colormap=:viridis`: colormap for trajectory
- `size=(800, 600)`: figure size
- `show_steering`: steering panel (default: `!isnothing(tapes)`)
- `show_winch_force=true`: winch force panel
- `show_v_app=true`: apparent wind speed panel
- `show_drag_coeff=false`: drag coefficient (C_D) from `var_01`
- `show_lift_coeff=false`: lift coefficient (C_L) from `var_02`
  C_D and C_L share a single panel when both are enabled.
- `show_lift_drag_ratio=true`: C_L/C_D ratio panel
  (C_D = wing + tether + bridle + KCU)
- `show_te_force=false`: mean TE segment force panel from `var_03`
- `show_heading=false`: heading angle panel
- `show_bridle_pitch=false`: bridle pitch angle panel from `var_08`
- `show_aoa=false`: AoA panel (wing and bridle, sim and data)
- `show_wing_vel=false`: kite ground speed panel
- `show_yaw=false`: yaw angle panel from `var_05`
- `show_pitch=true`: pitch angle panel from `var_06`
- `show_roll=false`: roll angle panel from `var_07`
  Yaw/pitch/roll share a single panel when any are enabled.
- `t_start=nothing`: start time in seconds from log start
- `t_end=nothing`: end time in seconds from log start
"""
function V3Kite.plot_2d_trajectory(
        logs::Vector{<:SymbolicAWEModels.KiteUtils.SysLog};
        gradient::Symbol=:vel,
        tapes=nothing,
        labels=nothing,
        colormap=:viridis,
        size=(800, 600),
        show_steering=nothing,
        show_winch_force=false,
        show_v_app=false,
        show_drag_coeff=false,
        show_lift_coeff=false,
        show_lift_drag_ratio=false,
        show_te_force=false,
        show_heading=false,
        show_bridle_pitch=true,
        show_aoa=true,
        show_wing_vel=true,
        show_depower=false,
        show_yaw=false,
        show_pitch=false,
        show_roll=false,
        t_start=nothing,
        t_end=nothing,
        twin_time_axes::Bool=false)

    if gradient == :steering && isnothing(tapes)
        error("tapes required for gradient=:steering")
    end
    show_steering = something(show_steering,
        !isnothing(tapes))
    if show_steering && isnothing(tapes)
        error("tapes required for show_steering=true")
    end

    # Count time-series panels for dynamic figure height
    has_euler = show_yaw || show_pitch || show_roll
    has_aoa = show_aoa && length(logs) >= 2
    n_panels = show_steering + show_depower +
        show_winch_force + show_v_app +
        show_drag_coeff + show_lift_coeff +
        show_lift_drag_ratio + show_te_force +
        show_heading + show_wing_vel + has_euler +
        show_bridle_pitch + has_aoa
    panel_height = 75
    fig_size = (size[1],
        size[2] + n_panels * panel_height)

    # Compute per-log index ranges for time filtering
    _time_range(t) = begin
        t0 = t[1]
        i1 = isnothing(t_start) ? 1 :
            searchsortedfirst(t, t0 + t_start)
        i2 = isnothing(t_end) ? length(t) :
            searchsortedlast(t, t0 + t_end)
        i1:i2
    end
    log_ranges = [_time_range(collect(lg.syslog.time))
                  for lg in logs]
    tape_ranges = isnothing(tapes) ? nothing :
        [_time_range(collect(Float64, tp.time))
         for tp in tapes]

    fig = Figure(; size=fig_size)
    ax = Axis(fig[1, 1];
        xlabel=L"y \; [m]", ylabel=L"z \; [m]",
        aspect=DataAspect())

    # Collect all gradient values for consistent range
    all_vals = Float64[]
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        rng = log_ranges[i]
        if gradient == :vel
            for k in rng
                push!(all_vals,
                    norm(sl.vel_kite[k]))
            end
        elseif gradient == :steering
            trng = tape_ranges[i]
            append!(all_vals,
                tapes[i].steering[trng] .* 100)
        else
            error("Unknown gradient: $gradient")
        end
    end
    vmin, vmax = extrema(all_vals)

    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        rng = log_ranges[i]
        y_pos = [sl.Y[k][1] for k in rng]
        z_pos = [sl.Z[k][1] for k in rng]
        vals = if gradient == :vel
            [norm(sl.vel_kite[k]) for k in rng]
        else
            collect(Float64,
                tapes[i].steering[tape_ranges[i]]
                .* 100)
        end

        label = isnothing(labels) ?
            "trajectory $i" : labels[i]
        n = min(length(y_pos), length(vals))
        lw = i == 1 ? 4.0 : 2.5
        lines!(ax, y_pos[1:n], z_pos[1:n];
            color=vals[1:n], colormap,
            colorrange=(vmin, vmax),
            linewidth=lw,
            label=i == 1 ? label : nothing)
        # Overlay dotted line for non-primary traces
        # (CairoMakie ignores linestyle with vector color)
        if i > 1
            lines!(ax, y_pos[1:n], z_pos[1:n];
                color=:white, linewidth=lw,
                linestyle=Makie.Linestyle(
                    [0, 2, 5, 7]))
            lines!(ax, Float64[], Float64[];
                color=:black, linewidth=lw,
                linestyle=Makie.Linestyle(
                    [0, 2, 5, 7]),
                label)
        end
    end

    cb_label = if gradient == :vel
        L"v_k \; [m/s]"
    else
        L"steering \; [\%]"
    end
    Colorbar(fig[1, 2]; colormap,
        colorrange=(vmin, vmax), label=cb_label)
    colsize!(fig.layout, 2, Fixed(40))

    # --- Time-series panels ---
    next_row = 1
    time_axes = Axis[]
    use_twin = twin_time_axes && length(logs) >= 2
    top_axes = Axis[]

    function _twin_panel!(fig, row, ylabel)
        ax = Axis(fig[row, 1]; ylabel,
            xticklabelsvisible=false)
        push!(time_axes, ax)
        if use_twin
            ax_top = Axis(fig[row, 1];
                xaxisposition=:top,
                xticklabelsvisible=false,
                yticklabelsvisible=false,
                ylabelvisible=false)
            linkyaxes!(ax, ax_top)
            push!(top_axes, ax_top)
        end
        return ax
    end

    if show_steering
        next_row += 1
        ax_st = _twin_panel!(fig, next_row,
            L"steering \; [\%]")
        for (i, tp) in enumerate(tapes)
            trng = tape_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_st
            lines!(target,
                collect(Float64, tp.time)[trng],
                collect(Float64,
                    tp.steering .* 100)[trng];
                linewidth=lw, linestyle=ls)
        end
    end

    if show_depower
        if isnothing(tapes)
            error("tapes required for show_depower=true")
        end
        next_row += 1
        ax_dp = _twin_panel!(fig, next_row,
            L"depower \; [\%]")
        for (i, tp) in enumerate(tapes)
            trng = tape_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_dp
            lines!(target,
                collect(Float64, tp.time)[trng],
                collect(Float64,
                    tp.depower .* 100)[trng];
                linewidth=lw, linestyle=ls)
        end
    end

    if show_winch_force
        next_row += 1
        ax_wf = _twin_panel!(fig, next_row,
            L"F_t \; [kN]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            wf = [sl.winch_force[k][1] / 1000
                  for k in rng]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_wf
            lines!(target,
                collect(sl.time)[rng], wf;
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_wf, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_v_app
        next_row += 1
        ax_va = _twin_panel!(fig, next_row,
            L"v_{app} \; [m/s]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_va
            lines!(target,
                collect(sl.time)[rng],
                collect(sl.v_app)[rng];
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_va, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_drag_coeff
        next_row += 1
        ax_cd = _twin_panel!(fig, next_row,
            L"C_D \; [-]")
        cd_vars = [
            (:var_01, "wing", :blue),
            (:var_09, "tether", :orange),
            (:var_10, "bridle", :green),
            (:var_11, "kcu", :red),
        ]
        leg_cd = Tuple{Vector, String}[]
        for (var, name, clr) in cd_vars
            plotted = false
            for (i, lg) in enumerate(logs)
                sl = lg.syslog
                rng = log_ranges[i]
                vals = collect(
                    getproperty(sl, var))[rng]
                all(iszero, vals) && continue
                lw = i == 1 ? 2.0 : 1.5
                ls = i == 1 ? :solid : :dash
                target = (use_twin && i == 2) ?
                    top_axes[end] : ax_cd
                lines!(target,
                    collect(sl.time)[rng], vals;
                    linewidth=lw, linestyle=ls,
                    color=clr)
                plotted = true
            end
            if plotted
                push!(leg_cd, (
                    [LineElement(color=clr,
                        linewidth=2)], name))
            end
        end
        if !isempty(leg_cd)
            Legend(fig[next_row, 2],
                first.(leg_cd), last.(leg_cd);
                labelsize=10, patchsize=(10, 5))
        end
        hlines!(ax_cd, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_lift_coeff
        next_row += 1
        ax_cl = _twin_panel!(fig, next_row,
            L"C_L \; [-]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_cl
            lines!(target,
                collect(sl.time)[rng],
                collect(sl.var_02)[rng];
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_cl, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_lift_drag_ratio
        next_row += 1
        ax_ld = _twin_panel!(fig, next_row,
            L"C_L / C_D \; [-]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            cd_wing = collect(sl.var_01)[rng]
            cd_teth = collect(sl.var_09)[rng]
            cd_brdl = collect(sl.var_10)[rng]
            cd_kcu  = collect(sl.var_11)[rng]
            cd = cd_wing .+ cd_teth .+ cd_brdl .+ cd_kcu
            cl = collect(sl.var_02)[rng]
            ratio = [abs(d) > 1e-6 ? l / d : NaN
                     for (l, d) in zip(cl, cd)]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_ld
            lines!(target,
                collect(sl.time)[rng], ratio;
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_ld, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_te_force
        next_row += 1
        ax_te = _twin_panel!(fig, next_row,
            L"\bar{F}_{TE} \; [N]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_te
            lines!(target,
                collect(sl.time)[rng],
                collect(sl.var_03)[rng];
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_te, [0]; linewidth=0.5,
            color=:gray70)
    end

    # --- Heading panel ---
    if show_heading
        next_row += 1
        ax_hd = _twin_panel!(fig, next_row,
            L"\psi \; [°]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_hd
            lines!(target,
                collect(sl.time)[rng],
                rad2deg.(collect(sl.heading)[rng]);
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_hd, [0]; linewidth=0.5,
            color=:gray70)
    end

    # --- Wing velocity panel ---
    if show_wing_vel
        next_row += 1
        ax_wv = _twin_panel!(fig, next_row,
            L"v_k \; [m/s]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            vk = [norm(sl.vel_kite[k]) for k in rng]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_wv
            lines!(target,
                collect(sl.time)[rng], vk;
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_wv, [0]; linewidth=0.5,
            color=:gray70)
    end

    # --- Euler angles panel (yaw / pitch / roll) ---
    if show_yaw || show_pitch || show_roll
        angle_vars = [
            (show_yaw, :var_05, "yaw",
                :blue, L"\psi \; [°]"),
            (show_pitch, :var_06, "pitch",
                :orange, L"\theta \; [°]"),
            (show_roll, :var_07, "roll",
                :green, L"\phi \; [°]"),
        ]
        active = filter(x -> x[1], angle_vars)
        single_angle = length(active) == 1
        ylabel_euler = single_angle ?
            active[1][5] : L"angle \; [°]"
        next_row += 1
        ax_euler = _twin_panel!(fig, next_row,
            ylabel_euler)
        if single_angle
            # Single angle: let Makie auto-cycle color
            var = active[1][2]
            for (i, lg) in enumerate(logs)
                sl = lg.syslog
                rng = log_ranges[i]
                vals = rad2deg.(
                    collect(getproperty(sl, var))[rng])
                lw = i == 1 ? 2.0 : 1.5
                ls = i == 1 ? :solid : :dash
                target = (use_twin && i == 2) ?
                    top_axes[end] : ax_euler
                lines!(target,
                    collect(sl.time)[rng], vals;
                    linewidth=lw, linestyle=ls)
            end
        else
            # Multiple angles: explicit colors + legend
            leg_elems = []
            leg_labels = String[]
            for (_, var, name, clr, _) in active
                for (i, lg) in enumerate(logs)
                    sl = lg.syslog
                    rng = log_ranges[i]
                    vals = rad2deg.(collect(
                        getproperty(sl, var))[rng])
                    lw = i == 1 ? 2.0 : 1.5
                    ls = i == 1 ? :solid : :dash
                    target = (use_twin && i == 2) ?
                        top_axes[end] : ax_euler
                    lines!(target,
                        collect(sl.time)[rng], vals;
                        linewidth=lw, linestyle=ls,
                        color=clr)
                end
                push!(leg_elems, [LineElement(
                    color=clr, linewidth=2)])
                push!(leg_labels, name)
            end
            if !isempty(leg_labels)
                Legend(fig[next_row, 2], leg_elems,
                    leg_labels;
                    labelsize=10, patchsize=(10, 5))
            end
        end
        hlines!(ax_euler, [0];
            linewidth=0.5, color=:gray70)
    end

    # --- Bridle pitch angle panel ---
    if show_bridle_pitch
        next_row += 1
        ax_bp = _twin_panel!(fig, next_row,
            L"\beta_{br} \; [°]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_bp
            lines!(target,
                collect(sl.time)[rng],
                rad2deg.(collect(sl.var_08)[rng]);
                linewidth=lw, linestyle=ls)
        end
        hlines!(ax_bp, [0]; linewidth=0.5,
            color=:gray70)
    end

    # --- AoA panel (wing + bridle, sim and data) ---
    if show_aoa && length(logs) >= 2
        sl_sim = logs[1].syslog
        sl_data = logs[2].syslog
        rng_sim = log_ranges[1]
        rng_data = log_ranges[2]
        t_sim_aoa = collect(sl_sim.time)[rng_sim]
        t_data_aoa = collect(sl_data.time)[rng_data]
        next_row += 1
        ax_aoa = _twin_panel!(fig, next_row,
            L"\alpha \; [°]")
        c_wing = Makie.wong_colors()[1]
        c_bridle = Makie.wong_colors()[2]
        # Sim traces on bottom axis
        lines!(ax_aoa, t_sim_aoa,
            rad2deg.(collect(sl_sim.var_12)[rng_sim]);
            linewidth=2.0, color=c_wing)
        lines!(ax_aoa, t_sim_aoa,
            rad2deg.(collect(sl_sim.var_04)[rng_sim]);
            linewidth=2.0, color=c_bridle)
        # Data traces on top axis (or same when !use_twin)
        data_ax = use_twin ? top_axes[end] : ax_aoa
        lines!(data_ax, t_data_aoa,
            rad2deg.(collect(sl_data.AoA)[rng_data]);
            linewidth=1.5, linestyle=:dash,
            color=c_wing)
        lines!(data_ax, t_data_aoa,
            rad2deg.(collect(
                sl_data.var_04)[rng_data]);
            linewidth=1.5, linestyle=:dash,
            color=c_bridle)
        hlines!(ax_aoa, [0]; linewidth=0.5,
            color=:gray70)
        leg_entries = [
            [LineElement(color=c_wing, linewidth=2)],
            [LineElement(color=c_bridle, linewidth=2)],
        ]
        Legend(fig[next_row, 2], leg_entries,
            ["wing AoA", "bridle AoA"];
            labelsize=10, patchsize=(10, 5))
    end

    # Final axis gets x label and visible tick labels
    if !isempty(time_axes)
        linkxaxes!(time_axes...)
        time_axes[end].xticklabelsvisible = true
        time_axes[end].xlabel = use_twin ?
            L"t_{sim} \; [s]" : L"t \; [s]"
        if use_twin && !isempty(top_axes)
            linkxaxes!(top_axes...)
            top_axes[1].xticklabelsvisible = true
            top_axes[1].xlabel = L"t_{data} \; [s]"
        end
        rowsize!(fig.layout, 1, Fixed(size[2] * 0.5))
    end

    if length(logs) > 1 || !isnothing(labels)
        traj_elems = []
        traj_labels = String[]
        for (i, _) in enumerate(logs)
            lbl = isnothing(labels) ?
                "trajectory $i" : labels[i]
            lw = i == 1 ? 4.0 : 2.5
            ls = i == 1 ? :solid :
                Makie.Linestyle([0, 2, 5, 7])
            clr = i == 1 ? :black : :black
            push!(traj_elems, [LineElement(;
                color=clr, linewidth=lw,
                linestyle=ls)])
            push!(traj_labels, lbl)
        end
        Legend(fig[next_row + 1, 1],
            traj_elems, traj_labels;
            orientation=:horizontal)
    end

    return fig
end

end # module
