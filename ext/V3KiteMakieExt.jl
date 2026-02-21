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
import V3Kite: SymbolicAWEModels

const PLOT_COLORS = [:blue, :green, :orange, :purple, :cyan, :magenta]

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
                               show_point_idxs=true)
    # Normalize to vector
    structs = sys_structs isa Vector ? sys_structs : [sys_structs]
    n_structs = length(structs)

    # Default labels
    if isnothing(labels)
        labels = n_structs == 1 ? ["sim"] : ["sim_$i" for i in 1:n_structs]
    end

    # Set up axis labels
    if dir == :top
        xlabel, ylabel = "x [m]", "y [m]"
    elseif dir == :side
        xlabel, ylabel = "x [m]", "z [m]"
    else  # :front
        xlabel, ylabel = "y [m]", "z [m]"
    end

    fig = Figure(size=figsize)
    ax_title = title ? "Wing Points (Body Frame)" : ""
    ax = Axis(fig[1, 1]; xlabel, ylabel,
              title=ax_title, aspect=DataAspect())

    function get_2d(pos_b)
        if dir == :top
            return (pos_b[1], pos_b[2])
        elseif dir == :side
            return (pos_b[1], pos_b[3])
        else
            return (pos_b[2], pos_b[3])
        end
    end

    # Collect all coordinates for auto-zoom
    all_x_vals = Float64[]
    all_y_vals = Float64[]

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

        # Select points to plot: use point_idxs if provided, otherwise WING points
        if isnothing(point_idxs)
            plot_points = [p for p in points if p.type == SymbolicAWEModels.WING]
        else
            plot_points = [points[i] for i in point_idxs if i <= length(points)]
        end

        # Extract 2D coords based on viewing direction
        if dir == :top
            coords = [(p.pos_b[1], p.pos_b[2]) for p in plot_points]
        elseif dir == :side
            coords = [(p.pos_b[1], p.pos_b[3]) for p in plot_points]
        else  # :front
            coords = [(p.pos_b[2], p.pos_b[3]) for p in plot_points]
        end

        x_vals = [c[1] for c in coords]
        y_vals = [c[2] for c in coords]
        append!(all_x_vals, x_vals)
        append!(all_y_vals, y_vals)

        # Plot segments (skip diagonals 29-46)
        plot_point_idxs = Set(p.idx for p in plot_points)
        for seg in segments
            if 29 <= seg.idx <= 46
                continue
            end
            from_idx, to_idx = seg.point_idxs
            if from_idx in plot_point_idxs && to_idx in plot_point_idxs
                p1 = points[from_idx]
                p2 = points[to_idx]
                c1 = get_2d(p1.pos_b)
                c2 = get_2d(p2.pos_b)
                lines!(ax, [c1[1], c2[1]], [c1[2], c2[2]];
                       color=(color, 0.5), linewidth=3)
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
            extra_coords = [(p[1], p[2]) for p in extra_body]
        elseif dir == :side
            extra_coords = [(p[1], p[3]) for p in extra_body]
        else
            extra_coords = [(p[2], p[3]) for p in extra_body]
        end

        # Draw lines connecting points within each group
        for (gname, indices) in extra_groups
            for i in 1:(length(indices)-1)
                c1 = extra_coords[indices[i]]
                c2 = extra_coords[indices[i+1]]
                lines!(ax, [c1[1], c2[1]], [c1[2], c2[2]];
                       color=(:red, 0.6), linewidth=2)
            end
        end

        # Plot all extra points as circles
        ex_x = [c[1] for c in extra_coords]
        ex_y = [c[2] for c in extra_coords]
        scatter!(ax, ex_x, ex_y;
                 markersize=extra_point_size, color=:red, marker=:circle)
    end

    # Auto-zoom with margin
    if !isempty(all_x_vals)
        x_min, x_max = extrema(all_x_vals)
        y_min, y_max = extrema(all_y_vals)
        margin_x = 0.15 * (x_max - x_min) + 0.3
        margin_y = 0.15 * (y_max - y_min) + 0.3
        limits!(ax, x_min - margin_x, x_max + margin_x,
                    y_min - margin_y, y_max + margin_y)
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

end # module
