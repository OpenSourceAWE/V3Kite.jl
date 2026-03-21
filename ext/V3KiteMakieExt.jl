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
const FRAME_COLORS = [:black, :red, :blue,
    :green, :orange, :purple, :cyan, :magenta]

"""
Draw photogrammetry group lines and scatter points.
LE lines are opaque+thick, strut inner points are
transparent.
"""
function _draw_extra_groups!(ax, coords, groups;
        point_size=8, strut_alpha=0.6,
        skip_ungrouped=false)
    te_idxs = Set{Int}()
    for (gname, indices) in groups
        is_le = gname == "LE"
        is_strut = startswith(gname, "strut")
        clr = is_le ? :orange : (:orange, strut_alpha)
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
    # Grouped indices (all points belonging to a group)
    grouped = Set{Int}()
    if skip_ungrouped
        for (_, indices) in groups
            union!(grouped, indices)
        end
    end
    # Opaque points (LE, TE, other)
    opaque = [i for i in eachindex(coords)
              if i ∉ strut_inner &&
                 (!skip_ungrouped || i ∈ grouped)]
    scatter!(ax,
        [coords[i][1] for i in opaque],
        [coords[i][2] for i in opaque];
        markersize=point_size,
        color=:orange, marker=:circle)
    # Transparent strut inner points
    if !isempty(strut_inner)
        inner = collect(strut_inner)
        scatter!(ax,
            [coords[i][1] for i in inner],
            [coords[i][2] for i in inner];
            markersize=point_size,
            color=(:orange, 0.4), marker=:circle)
    end
end

"""Draw text with a white halo (CairoMakie-safe)."""
function _halo_text!(ax, x, y; text, fontsize=12,
        color=:black, halo_width=3, kw...)
    text!(ax, x, y; text, fontsize,
        color=:white, strokecolor=:white,
        strokewidth=halo_width, kw...)
    text!(ax, x, y; text, fontsize, color, kw...)
end

"""
Draw incidence angle overlay: chord line, bridle
lines, scatter points, angle arc, and labels.
"""
function _draw_incidence!(ax, kcu_2d, cr_2d,
        te_2d, le_2d; color=:purple,
        radius_scale=0.3, show_labels=true,
        origin_label="KCU")
    # LE→TE chord line (semi-transparent)
    lines!(ax,
        [le_2d[1], te_2d[1]],
        [le_2d[2], te_2d[2]];
        color=(color, 0.4), linewidth=2)
    # KCU→CR and CR→TE solid lines
    lines!(ax,
        [kcu_2d[1], cr_2d[1]],
        [kcu_2d[2], cr_2d[2]];
        color=color, linewidth=2)
    lines!(ax,
        [cr_2d[1], te_2d[1]],
        [cr_2d[2], te_2d[2]];
        color=color, linewidth=2)
    # Scatter on all 4 points
    scatter!(ax,
        [kcu_2d[1], cr_2d[1], te_2d[1], le_2d[1]],
        [kcu_2d[2], cr_2d[2], te_2d[2], le_2d[2]];
        markersize=10, color=color)
    if show_labels
        _halo_text!(ax, kcu_2d[1], kcu_2d[2];
            text=origin_label, fontsize=12,
            color=color,
            align=(:right, :top), offset=(-6, -4))
        _halo_text!(ax, cr_2d[1], cr_2d[2];
            text="CR", fontsize=12, color=color,
            align=(:left, :bottom), offset=(6, 8))
        _halo_text!(ax, te_2d[1], te_2d[2];
            text="TE", fontsize=12, color=color,
            align=(:right, :bottom),
            offset=(-8, 8))
        _halo_text!(ax, le_2d[1], le_2d[2];
            text="LE", fontsize=12, color=color,
            align=(:left, :bottom), offset=(6, 8))
    end
    # Angle arc at CR
    v_kcu = normalize([kcu_2d[1] - cr_2d[1],
                       kcu_2d[2] - cr_2d[2]])
    v_te  = normalize([te_2d[1] - cr_2d[1],
                       te_2d[2] - cr_2d[2]])
    th1 = atan(v_kcu[2], v_kcu[1])
    th2 = atan(v_te[2], v_te[1])
    dth = th2 - th1
    dth > pi  && (dth -= 2pi)
    dth < -pi && (dth += 2pi)
    arm_kcu = norm([kcu_2d[1] - cr_2d[1],
                    kcu_2d[2] - cr_2d[2]])
    arm_te  = norm([te_2d[1] - cr_2d[1],
                    te_2d[2] - cr_2d[2]])
    radius = radius_scale * min(arm_kcu, arm_te)
    ths = range(th1, th1 + dth; length=30)
    arc_x = cr_2d[1] .+ radius .* cos.(ths)
    arc_y = cr_2d[2] .+ radius .* sin.(ths)
    lines!(ax, arc_x, arc_y;
        color=color, linewidth=2)
    th_mid = th1 + dth / 2
    deg_str = "$(round(abs(rad2deg(dth));
        digits=1))°"
    _halo_text!(ax,
        cr_2d[1] + radius * 1.5 * cos(th_mid),
        cr_2d[2] + radius * 1.5 * sin(th_mid);
        text=deg_str, fontsize=14, color=color,
        halo_width=5)
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
- `figsize`: Figure size tuple (default: (560, 420))
- `legend`: Show legend (default: true)
- `title`: Show title (default: true)
- `show_point_idxs`: Show point index labels (default: true)
- `show_twist`: Show twist panel below (default: false)
- `show_incidence`: Show incidence angle overlay on :side views (default: false)
- `show_kcu`: Show KCU point (sys_struct point 1) with legend entry (default: false)
- `show_camera`: Show camera point (ungrouped extra point) with legend entry (default: false)
"""
function V3Kite.plot_body_frame_local(sys_structs;
                               extra_points=nothing,
                               extra_groups=nothing,
                               dir::Symbol=:front,
                               point_size=10,
                               extra_point_size=8,
                               figsize=(560, 420),
                               labels=nothing,
                               point_idxs=nothing,
                               legend=true,
                               legend_position=:right,
                               title=true,
                               show_point_idxs=false,
                               show_twist=false,
                               show_incidence=false,
                               show_kcu=false,
                               show_camera=false,
                               annotation="")
    # Normalize to vector
    structs = sys_structs isa Vector ? sys_structs : [sys_structs]
    n_structs = length(structs)

    # Default labels
    if isnothing(labels)
        labels = n_structs == 1 ?
            ["simulation"] :
            ["simulation $i"
             for i in 1:n_structs]
    end

    # Set up axis labels
    if dir == :top
        xlabel = L"y \; [m]"
        ylabel = L"x \; [m]"
    elseif dir == :side
        xlabel = L"x \; [m]"
        ylabel = L"z \; [m]"
    else  # :front
        xlabel = L"y \; [m]"
        ylabel = L"z \; [m]"
    end

    twist_figsize = show_twist ?
        (figsize[1], figsize[2] + 150) : figsize
    fig = Figure(size=twist_figsize)
    ax_title = title ? "Wing Points (Body Frame)" : ""
    ax = Axis(fig[1, 1]; xlabel, ylabel,
              title=ax_title, aspect=DataAspect())
    ax_twist = nothing
    if show_twist
        ax_twist = Axis(fig[2, 1];
            xlabel=L"y \; [m]",
            ylabel=L"\theta \; [°]")
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
                wing.pos_w .= points[1].pos_w
                for point in points
                    if point.wing_idx == wing.idx
                        point.pos_b .= R_w_b * (point.pos_w - wing.pos_w)
                    end
                end
            end
        end

        # Twist for WING point pairs
        if show_twist
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

    # Plot KCU (point 1) for each sys_struct
    if show_kcu
        for sys_struct in structs
            wing = sys_struct.wings[1]
            R_w_b = V3Kite.calc_R_b_w(sys_struct)'
            kcu_b = R_w_b *
                (sys_struct.points[1].pos_w - wing.pos_w)
            kx, ky = get_2d(kcu_b)
            push!(all_x_vals, kx)
            push!(all_y_vals, ky)
            scatter!(ax, [kx], [ky];
                color=:green, markersize=12,
                marker=:circle)
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

        _s_alpha = show_incidence && dir == :side ?
            0.4 : 0.6
        _draw_extra_groups!(ax, extra_coords,
            extra_groups;
            point_size=extra_point_size,
            strut_alpha=_s_alpha,
            skip_ungrouped=!show_camera)

        # Plot camera point (first ungrouped extra point)
        if show_camera
            grouped = Set{Int}()
            for (_, indices) in extra_groups
                union!(grouped, indices)
            end
            for i in eachindex(extra_coords)
                if i ∉ grouped
                    cx, cy = extra_coords[i]
                    push!(all_x_vals, cx)
                    push!(all_y_vals, cy)
                    scatter!(ax, [cx], [cy];
                        color=:black, markersize=12,
                        marker=:circle)
                    break
                end
            end
        end

        # Photogrammetry twist
        if show_twist && !isnothing(ax_twist)
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
                lines!(ax_twist,
                    photo_span[perm],
                    photo_aoa[perm];
                    color=:orange, linewidth=2,
                    label="photogrammetry")
                scatter!(ax_twist,
                    photo_span, photo_aoa;
                    color=:orange, markersize=8)
            end
        end
    end

    # Incidence angle overlay (side view only)
    if show_incidence && dir == :side
        # Sim incidence from first sys_struct
        sys1 = structs[1]
        pts1 = sys1.points
        wing1 = sys1.wings[1]
        R_w_b = V3Kite.calc_R_b_w(sys1)'
        kcu_b = R_w_b * (pts1[1].pos_w - wing1.pos_w)
        le_3_b = pts1[10].pos_b
        te_3_b = pts1[11].pos_b
        le_4_b = pts1[12].pos_b
        te_4_b = pts1[13].pos_b
        cr_b = chord_ref_mid(
            le_3_b, te_3_b, le_4_b, te_4_b)
        le_center_b = (le_3_b + le_4_b) / 2
        te_mid_b = (te_3_b + te_4_b) / 2
        _draw_incidence!(ax,
            get_2d(kcu_b), get_2d(cr_b),
            get_2d(te_mid_b), get_2d(le_center_b);
            color=:purple, radius_scale=0.3)

        # Photogrammetry incidence (when extra data given)
        if !isnothing(extra_points) &&
                !isnothing(extra_groups)
            eb = [R_w_b * (collect(p) - wing1.pos_w)
                  for p in extra_points]
            s3_idx = nothing
            s4_idx = nothing
            for (gname, indices) in extra_groups
                gname == "strut3" && (s3_idx = indices)
                gname == "strut4" && (s4_idx = indices)
            end
            if !isnothing(s3_idx) && !isnothing(s4_idx)
                p_te3 = eb[s3_idx[1]]
                p_le3 = eb[s3_idx[end]]
                p_te4 = eb[s4_idx[1]]
                p_le4 = eb[s4_idx[end]]
                p_cr = chord_ref_mid(
                    p_le3, p_te3, p_le4, p_te4)
                p_te_mid = (p_te3 + p_te4) / 2
                p_le_ctr = (p_le3 + p_le4) / 2
                # KCU = first ungrouped point
                grouped = Set{Int}()
                for (_, idxs) in extra_groups
                    union!(grouped, idxs)
                end
                p_kcu = nothing
                for i in eachindex(eb)
                    if i ∉ grouped
                        p_kcu = eb[i]
                        break
                    end
                end
                if !isnothing(p_kcu)
                    _draw_incidence!(ax,
                        get_2d(p_kcu), get_2d(p_cr),
                        get_2d(p_te_mid),
                        get_2d(p_le_ctr);
                        color=:darkorange,
                        radius_scale=0.45,
                        origin_label="C")
                end
            end
        end
    end

    # Set axis limits
    if dir == :front || dir == :top
        if !isempty(all_y_vals)
            y_min, y_max = extrema(all_y_vals)
            margin_y = 0.15 * (y_max - y_min) + 0.3
            if dir == :top
                limits!(ax, -5.3, 5.3,
                    y_max + margin_y,
                    y_min - margin_y)
            else
                limits!(ax, -5.3, 5.3,
                    y_min - margin_y,
                    y_max + margin_y)
            end
        end
    elseif !isempty(all_x_vals)
        x_min, x_max = extrema(all_x_vals)
        y_min, y_max = extrema(all_y_vals)
        margin_x = 0.15 * (x_max - x_min) + 0.3
        margin_y = 0.15 * (y_max - y_min) + 0.3
        limits!(ax, x_min - margin_x, x_max + margin_x,
                    y_min - margin_y, y_max + margin_y)
    end

    # Plot sim twist curves
    if show_twist && !isnothing(ax_twist)
        for (span_ys, aoas, clr, lbl) in sim_aoa_data
            perm = sortperm(span_ys)
            lines!(ax_twist,
                span_ys[perm], aoas[perm];
                color=clr, linewidth=2, label=lbl)
            scatter!(ax_twist, span_ys, aoas;
                color=clr, markersize=8)
        end
        if dir == :front
            linkxaxes!(ax, ax_twist)
        end
    end

    # Legend
    if legend
        legend_elements = [
            MarkerElement(
                color=PLOT_COLORS[
                    mod1(i, length(PLOT_COLORS))],
                marker=:circle, markersize=10)
            for i in 1:n_structs
        ]
        legend_labels = copy(labels)
        if show_kcu
            push!(legend_elements,
                  MarkerElement(color=:green,
                      marker=:circle, markersize=10))
            push!(legend_labels, "KCU")
        end
        if !isnothing(extra_points)
            push!(legend_elements,
                  MarkerElement(color=:orange,
                      marker=:circle, markersize=10))
            push!(legend_labels, "photogrammetry")
        end
        if show_camera
            push!(legend_elements,
                  MarkerElement(color=:black,
                      marker=:circle, markersize=10))
            push!(legend_labels, "camera")
        end
        if legend_position == :top
            Legend(fig[1, 1], legend_elements,
                legend_labels;
                orientation=:horizontal,
                padding=(4, 4, 2, 2),
                margin=(0, 0, 0, 0),
                valign=:top, halign=:center,
                tellwidth=false, tellheight=false)
        else
            Legend(fig[1, 2], legend_elements,
                legend_labels)
        end
    end

    if !isempty(annotation)
        text!(ax, annotation;
            position=Point2f(1, 1),
            space=:relative,
            align=(:right, :top),
            offset=(-6, -6),
            fontsize=14, font=:bold)
    end

    return fig
end

"""
    plot_twist_dist(sys_structs; extra_points,
        extra_groups, labels, figsize)

Plot twist distribution along the span.
Computes local twist at each LE/TE pair from sim
sys_structs and optionally from photogrammetry data.

# Arguments
- `sys_structs`: System structure or vector of them
- `extra_points`: Optional photogrammetry points
- `extra_groups`: Optional photogrammetry groups
- `labels`: Optional vector of labels
- `figsize`: Figure size (default: (560, 210))
- `wingtips`: Include wingtip struts (default: false)
- `limits`: Twist axis limits in deg (default: (-7, 10))
"""
function V3Kite.plot_twist_dist(sys_structs;
        extra_points=nothing,
        extra_groups=nothing,
        labels=nothing,
        figsize=(560, 210),
        title=true,
        legend=true,
        wingtips=false,
        limits=(-7, 10),
        annotation="")
    structs = sys_structs isa Vector ?
        sys_structs : [sys_structs]
    n_structs = length(structs)
    if isnothing(labels)
        labels = n_structs == 1 ?
            ["simulation"] :
            ["simulation $i"
             for i in 1:n_structs]
    end

    fig = Figure(size=figsize)
    ax_title = title ? "Twist Distribution" : ""
    ax = Axis(fig[1, 1];
        xlabel=L"y \; [m]",
        ylabel=L"\theta \; [°]",
        title=ax_title, limits=((-5.3, 5.3), limits))

    # Sim AoA per sys_struct
    for (s_idx, sys_struct) in enumerate(structs)
        points = sys_struct.points
        color = PLOT_COLORS[
            mod1(s_idx, length(PLOT_COLORS))]

        # Update pos_b for REFINE wing points
        for wing in sys_struct.wings
            if wing.wing_type == SymbolicAWEModels.REFINE
                R_w_b = V3Kite.calc_R_b_w(sys_struct)'
                wing.pos_w .= points[1].pos_w
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
        k_range = wingtips ? (1:n_le) :
            (2:(n_le - 1))
        for k in k_range
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
                color=:orange, linewidth=2,
                label="photogrammetry")
            scatter!(ax, span_pos, photo_aoa;
                color=:orange, markersize=8)
        end
    end

    if legend
        axislegend(ax; position=:rt)
    end

    if !isempty(annotation)
        text!(ax, annotation;
            position=Point2f(1, 1),
            space=:relative,
            align=(:right, :top),
            offset=(-6, -6),
            fontsize=14, font=:bold)
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
        (L"x \; [m]", L"y \; [m]")
    elseif dir == :side
        (L"x \; [m]", L"z \; [m]")
    else
        (L"y \; [m]", L"z \; [m]")
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
        xlabel=L"|u_{\text{s}} \cdot v_{\text{a}}| \; [m/s]",
        ylabel=L"|\dot{\psi}| \; [rad/s]",
        xlabelsize=18, ylabelsize=18)

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
        times=all_times, ylabel=L"F_{\text{t}} \; [N]"))

    # --- Apparent wind speed panel ---
    all_data, all_labels, all_times = [], [], []
    for (i, lg) in enumerate(logs)
        sl = lg.syslog
        push!(all_data, collect(sl.v_app))
        push!(all_labels, L"v_a" * actual_suffixes[i])
        push!(all_times, sl.time)
    end
    push!(panels, (data=all_data, labels=all_labels,
        times=all_times, ylabel=L"v_{\text{a}} \; [m/s]"))

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
        times=all_times, ylabel=L"v_{\text{w}} \; [m/s]"))

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
        ylabel=L"\theta_{\text{w}} \; [°]"))

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
        times=all_times, ylabel=L"v_{\text{k}} \; [m/s]"))

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
        times=all_times, ylabel=L"C_{\text{F}} \; [-]"))

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
                L"g_{\text{k}}" * actual_suffixes[i])
            push!(all_times, sl.time[2:end])
        end
        push!(panels, (data=all_data, labels=all_labels,
            times=all_times, ylabel=L"g_{\text{k}} \; [-]"))
    end

    # --- Steering panel (only when tape_lengths) ---
    if !isnothing(tape_lengths)
        all_data, all_labels, all_times = [], [], []
        for (i, _) in enumerate(logs)
            tl = tape_lengths[i]
            push!(all_data, collect(tl.steering .* 100))
            push!(all_labels,
                L"u_{\text{s}}" * actual_suffixes[i])
            push!(all_times, collect(tl.time))
        end
        push!(panels, (data=all_data, labels=all_labels,
            times=all_times,
            ylabel=L"u_{\text{s}} \; [\%]"))
    end

    # --- Render panels ---
    n_panels = length(panels)
    fig = Figure(; size)
    axes = Axis[]
    label_fontsize = 18
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
                labelsize=14, patchsize=(14, 7))
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
# Shared time-range helper for trajectory / panels
# =====================================================================

"""Compute per-log and per-tape index ranges for time filtering."""
function _compute_ranges(logs, tapes, t_start, t_end)
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
    return log_ranges, tape_ranges
end

# =====================================================================
# plot_2d_trajectory — y vs z colored by gradient
# =====================================================================

"""
    plot_2d_trajectory(logs; gradient=:vel, tapes=nothing,
        labels, colormap, size, t_start, t_end,
        frame_indexes)

Plot kite y vs z position colored by a gradient quantity.

# Arguments
- `gradient=:vel`: color by `:vel` or `:steering`
- `tapes`: vector of named tuples with `time`, `steering`
- `labels`: legend labels per log
- `colormap=:viridis`: colormap for trajectory
- `size=(800, 600)`: figure size
- `t_start=nothing`: start time in seconds from log start
- `t_end=nothing`: end time in seconds from log start
- `frame_indexes=nothing`: vector of `(frame_nr, syslog_idx)`
  tuples; plots a colored dot on the trajectory for each
  frame with a legend entry
"""
function V3Kite.plot_2d_trajectory(
        logs::Vector{<:SymbolicAWEModels.KiteUtils.SysLog};
        gradient::Symbol=:vel,
        tapes=nothing,
        labels=nothing,
        colormap=:viridis,
        size=(560, 420),
        t_start=nothing,
        t_end=nothing,
        frame_indexes=nothing)

    if gradient == :steering && isnothing(tapes)
        error("tapes required for gradient=:steering")
    end

    log_ranges, tape_ranges = _compute_ranges(
        logs, tapes, t_start, t_end)

    fig = Figure(; size)
    ax = Axis(fig[1, 1];
        xlabel=L"y \; [m]", ylabel=L"z \; [m]",
        xlabelsize=20, ylabelsize=20,
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
                tapes[i].steering[trng])
        else
            error("Unknown gradient: $gradient")
        end
    end
    vmin, vmax = extrema(all_vals)

    # Draw non-primary traces first so sim (i==1) is on top
    for i in reverse(eachindex(logs))
        lg = logs[i]
        sl = lg.syslog
        rng = log_ranges[i]
        y_pos = [sl.Y[k][1] for k in rng]
        z_pos = [sl.Z[k][1] for k in rng]
        vals = if gradient == :vel
            [norm(sl.vel_kite[k]) for k in rng]
        else
            collect(Float64,
                tapes[i].steering[tape_ranges[i]])
        end

        n = min(length(y_pos), length(vals))
        lw = 4.0
        lines!(ax, y_pos[1:n], z_pos[1:n];
            color=vals[1:n], colormap,
            colorrange=(vmin, vmax),
            linewidth=lw)
        if i > 1
            lines!(ax, y_pos[1:n], z_pos[1:n];
                color=:white, linewidth=lw,
                linestyle=Makie.Linestyle(
                    [0, 2, 5, 7]))
        end
    end

    cb_label = if gradient == :vel
        L"v_{\text{k}} \; [m/s]"
    else
        L"u_{\text{s}} \; [-]"
    end
    Colorbar(fig[2, 1]; colormap,
        colorrange=(vmin, vmax), label=cb_label,
        labelsize=20, vertical=false,
        flipaxis=false)
    rowgap!(fig.layout, 1, 8)

    # Combined legend: sim/data lines + frame markers
    legend_elems = []
    legend_labels = String[]
    for (i, _) in enumerate(logs)
        lbl = isnothing(labels) ?
            "trajectory $i" : labels[i]
        lw = i == 1 ? 4.0 : 2.5
        ls = i == 1 ? :solid :
            Makie.Linestyle([0, 2, 5, 7])
        push!(legend_elems, [LineElement(;
            color=:black, linewidth=lw,
            linestyle=ls)])
        push!(legend_labels, lbl)
    end
    if !isnothing(frame_indexes) &&
            !isempty(frame_indexes)
        sl1 = logs[1].syslog
        for (j, (frame_nr, syslog_idx)) in
                enumerate(frame_indexes)
            clr = FRAME_COLORS[
                mod1(j, length(FRAME_COLORS))]
            t_frame = collect(sl1.time)[syslog_idx]
            for lg in logs
                sl = lg.syslog
                t_all = collect(sl.time)
                idx = argmin(
                    abs.(t_all .- t_frame))
                y = sl.Y[idx][1]
                z = sl.Z[idx][1]
                scatter!(ax, [y], [z];
                    color=clr, markersize=14,
                    marker=:circle)
            end
            push!(legend_elems,
                [MarkerElement(color=clr,
                    marker=:circle,
                    markersize=10),
                 LineElement(color=clr,
                    linewidth=2,
                    linestyle=:dash)])
            push!(legend_labels,
                "frame $frame_nr")
        end
    end
    if !isempty(legend_labels)
        Legend(fig[1, 1], legend_elems,
            legend_labels; labelsize=14,
            halign=:left, valign=:bottom,
            tellwidth=false, tellheight=false,
            margin=(0, 20, 22, 20))
    end

    return fig
end

# =====================================================================
# plot_2d_panels — time-series subplots
# =====================================================================

"""
    plot_2d_panels(logs; tapes, labels, kwargs...)

Time-series panel figure (steering, winch force, v_app, etc.)
extracted from `plot_2d_trajectory`.

# Arguments
- `tapes`: vector of named tuples with `time`, `steering`,
  `depower`
- `labels`: legend labels per log
- `size`: figure `(width, height)` — auto-computed when
  `nothing` (default width 600)
- `panel_height=60`: height per panel in pixels
- `show_*` flags: toggle individual panels
- `t_start`, `t_end`: time range in seconds from log start
- `twin_time_axes`: use separate x axes for each log
- `frame_indexes=nothing`: vector of `(frame_nr, syslog_idx)`
  tuples; draws vertical lines on all panels at the
  corresponding times, using the same colors as trajectory
  frame markers
"""
function V3Kite.plot_2d_panels(
        logs::Vector{<:SymbolicAWEModels.KiteUtils.SysLog};
        tapes=nothing,
        labels=nothing,
        size=nothing,
        panel_height::Int=120,
        show_steering=nothing,
        show_winch_force=true,
        show_v_app=true,
        show_tether_len=false,
        show_drag_coeff=false,
        show_lift_coeff=false,
        show_lift_drag_ratio=false,
        show_te_force=false,
        show_heading=false,
        show_bridle_pitch=false,
        show_aoa=true,
        show_wing_vel=false,
        show_depower=false,
        show_yaw=false,
        show_pitch=false,
        show_roll=false,
        show_cop=false,
        t_start=nothing,
        t_end=nothing,
        twin_time_axes::Bool=false,
        frame_indexes=nothing)

    show_steering = something(show_steering,
        !isnothing(tapes))
    if show_steering && isnothing(tapes)
        error("tapes required for show_steering=true")
    end

    has_euler = show_yaw || show_pitch || show_roll
    has_aoa = show_aoa && length(logs) >= 2
    n_panels = show_steering + show_depower +
        show_winch_force + show_v_app +
        show_tether_len +
        show_drag_coeff + show_lift_coeff +
        show_lift_drag_ratio + show_te_force +
        show_heading + show_wing_vel + has_euler +
        show_bridle_pitch + has_aoa + show_cop

    fig_size = isnothing(size) ?
        (800, n_panels * panel_height) : size

    log_ranges, tape_ranges = _compute_ranges(
        logs, tapes, t_start, t_end)

    fig = Figure(; size=fig_size)
    rowgap!(fig.layout, 4)
    cur_row = 0
    time_axes = Axis[]
    use_twin = twin_time_axes && length(logs) >= 2
    top_axes = Axis[]

    function _twin_panel!(fig, row, ylabel)
        ax = Axis(fig[row, 1]; ylabel,
            ylabelsize=20,
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
        cur_row += 1
        ax_st = _twin_panel!(fig, cur_row,
            L"u_{\text{s}} \; [-]")
        for (i, tp) in enumerate(tapes)
            trng = tape_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_st
            lines!(target,
                collect(Float64, tp.time)[trng],
                collect(Float64,
                    tp.steering)[trng];
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
    end

    if show_depower
        if isnothing(tapes)
            error(
                "tapes required for show_depower=true")
        end
        cur_row += 1
        ax_dp = _twin_panel!(fig, cur_row,
            L"u_{\text{d}} \; [\%]")
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
    end

    if show_winch_force
        cur_row += 1
        ax_wf = _twin_panel!(fig, cur_row,
            L"F_{\text{t}} \; [kN]")
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_wf, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_v_app
        cur_row += 1
        ax_va = _twin_panel!(fig, cur_row,
            L"v_{\text{app}} \; [m/s]")
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_va, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_tether_len
        cur_row += 1
        ax_tl = _twin_panel!(fig, cur_row,
            L"l_{\text{t}} \; [m]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lt = [sl.l_tether[k][1] for k in rng]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_tl
            lines!(target,
                collect(sl.time)[rng], lt;
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
    end

    if show_drag_coeff
        cur_row += 1
        ax_cd = _twin_panel!(fig, cur_row,
            L"C_{\text{D}} \; [-]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_cd
            lines!(target,
                collect(sl.time)[rng],
                collect(sl.var_01)[rng];
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_cd, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_lift_coeff
        cur_row += 1
        ax_cl = _twin_panel!(fig, cur_row,
            L"C_{\text{L}} \; [-]")
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_cl, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_lift_drag_ratio
        cur_row += 1
        ax_ld = _twin_panel!(fig, cur_row,
            L"C_L / C_{\text{D}} \; [-]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            cd = collect(sl.var_01)[rng]
            cl = collect(sl.var_02)[rng]
            ratio = [abs(d) > 1e-6 ? l / d : NaN
                     for (l, d) in zip(cl, cd)]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_ld
            lines!(target,
                collect(sl.time)[rng], ratio;
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_ld, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_te_force
        cur_row += 1
        ax_te = _twin_panel!(fig, cur_row,
            L"\bar{F}_{\text{TE}} \; [N]")
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_te, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_heading
        cur_row += 1
        ax_hd = _twin_panel!(fig, cur_row,
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_hd, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_wing_vel
        cur_row += 1
        ax_wv = _twin_panel!(fig, cur_row,
            L"v_{\text{k}} \; [m/s]")
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_wv, [0]; linewidth=0.5,
            color=:gray70)
    end

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
            active[1][5] : L"\text{angle} \; [°]"
        cur_row += 1
        ax_euler = _twin_panel!(fig, cur_row,
            ylabel_euler)
        if single_angle
            var = active[1][2]
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
                    color=:black)
            end
        else
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
                        collect(sl.time)[rng],
                        vals;
                        linewidth=lw, linestyle=ls,
                        color=clr)
                end
                push!(leg_elems, [LineElement(
                    color=clr, linewidth=2)])
                push!(leg_labels, name)
            end
            if !isempty(leg_labels)
                Legend(fig[cur_row, 1], leg_elems,
                    leg_labels; labelsize=14,
                    patchsize=(18, 9),
                    halign=:right, valign=:top,
                    tellwidth=false,
                    tellheight=false,
                    margin=(0, 0, 0, 0))
            end
        end
        hlines!(ax_euler, [0];
            linewidth=0.5, color=:gray70)
    end

    if show_bridle_pitch
        cur_row += 1
        ax_bp = _twin_panel!(fig, cur_row,
            L"\beta_{\text{br}} \; [°]")
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
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_bp, [0]; linewidth=0.5,
            color=:gray70)
    end

    if show_aoa && length(logs) >= 2
        sl_sim = logs[1].syslog
        sl_data = logs[2].syslog
        rng_sim = log_ranges[1]
        rng_data = log_ranges[2]
        t_sim_aoa = collect(sl_sim.time)[rng_sim]
        t_data_aoa = collect(sl_data.time)[rng_data]
        cur_row += 1
        ax_aoa = _twin_panel!(fig, cur_row,
            L"\alpha \; [°]")
        c_wing = Makie.wong_colors()[1]
        c_kite = Makie.wong_colors()[2]
        # Sim traces on bottom axis
        lines!(ax_aoa, t_sim_aoa,
            rad2deg.(collect(
                sl_sim.var_12)[rng_sim]);
            linewidth=2.0, color=c_wing)
        lines!(ax_aoa, t_sim_aoa,
            rad2deg.(collect(
                sl_sim.var_04)[rng_sim]);
            linewidth=2.0, color=c_kite)
        # Data traces on top axis (or same)
        data_ax = use_twin ? top_axes[end] : ax_aoa
        lines!(data_ax, t_data_aoa,
            rad2deg.(collect(
                sl_data.var_12)[rng_data]);
            linewidth=1.5, linestyle=:dash,
            color=c_wing)
        lines!(data_ax, t_data_aoa,
            rad2deg.(collect(
                sl_data.var_04)[rng_data]);
            linewidth=1.5, linestyle=:dash,
            color=c_kite)
        hlines!(ax_aoa, [0]; linewidth=0.5,
            color=:gray70)
        leg_entries = [
            [LineElement(color=c_wing, linewidth=2)],
            [LineElement(
                color=c_kite, linewidth=2)],
        ]
        Legend(fig[cur_row, 1], leg_entries,
            [L"\text{wing}", L"\text{kite}"];
            labelsize=14, patchsize=(18, 9),
            halign=:right, valign=:top,
            tellwidth=false, tellheight=false,
            margin=(0, 0, 0, 0))
    end

    if show_cop
        cur_row += 1
        ax_cop = _twin_panel!(fig, cur_row,
            L"x_{\text{cop}} \; [m]")
        for (i, lg) in enumerate(logs)
            sl = lg.syslog
            rng = log_ranges[i]
            lw = i == 1 ? 2.0 : 1.5
            ls = i == 1 ? :solid : :dash
            target = (use_twin && i == 2) ?
                top_axes[end] : ax_cop
            lines!(target,
                collect(sl.time)[rng],
                collect(sl.var_13)[rng];
                linewidth=lw, linestyle=ls,
                    color=:black)
        end
        hlines!(ax_cop, [0]; linewidth=0.5,
            color=:gray70)
    end

    # Final axis gets x label and visible tick labels
    if !isempty(time_axes)
        linkxaxes!(time_axes...)
        time_axes[end].xticklabelsvisible = true
        time_axes[end].xlabel = use_twin ?
            L"t_{\text{sim}} \; [s]" : L"t \; [s]"
        time_axes[end].xlabelsize = 20
        if use_twin && !isempty(top_axes)
            linkxaxes!(top_axes...)
            top_axes[1].xticklabelsvisible = true
            top_axes[1].xlabel =
                L"t_{\text{data}} \; [s]"
            top_axes[1].xlabelsize = 20
        end
    end

    # Frame index vertical lines on all panels
    if !isnothing(frame_indexes) &&
            !isempty(frame_indexes) &&
            !isempty(time_axes)
        sl = logs[1].syslog
        for (j, (_, syslog_idx)) in
                enumerate(frame_indexes)
            clr = FRAME_COLORS[
                mod1(j, length(FRAME_COLORS))]
            t = collect(sl.time)[syslog_idx]
            for tax in time_axes
                vlines!(tax, [t]; color=clr,
                    linewidth=1.5, linestyle=:dash)
            end
        end
    end


    return fig
end

end # module
