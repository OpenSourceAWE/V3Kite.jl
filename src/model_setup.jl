# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite model setup utilities.
Functions for adjusting tether length, elevation, and other model parameters.
"""

"""
    V3GeomAdjustConfig

Configuration for wing geometry adjustments (tip reduction, trailing
edge shortening, depower tape reduction, and tether length).
"""
Base.@kwdef mutable struct V3GeomAdjustConfig
    reduce_tip::Bool = false
    tip_reduction::Float64 = 0.4
    tip_segments::Vector{Int} = [47, 48, 57, 58]

    reduce_te::Bool = false
    te_frac::Float64 = 0.95
    te_segments::UnitRange{Int} = 20:28

    reduce_depower::Bool = false
    depower_reduction::Float64 = 0.2
    depower_offset::Float64 = 0.0       # added to depower (0..1)
    steering_dp_offset::Float64 = 0.0   # Δdp per abs(steering), normalized

    reduce_steering::Bool = false
    steering_reduction::Float64 = 0.2

    tether_length::Union{Nothing,Float64} = nothing
end

"""
    apply_geom_adjustments!(sys, config::V3GeomAdjustConfig)

Apply wing geometry adjustments to a `SystemStructure`:
tip leading-edge reduction, trailing-edge wire shortening,
and tether length repositioning.
"""
function apply_geom_adjustments!(sys, config::V3GeomAdjustConfig)
    if config.reduce_tip
        for idx in config.tip_segments
            sys.segments[idx].l0 -= config.tip_reduction
        end
    end
    if config.reduce_te
        for idx in config.te_segments
            sys.segments[idx].l0 *= config.te_frac
        end
    end
    return nothing
end

"""
    l0_diffs_to_move_point(sys, point_idx, target_idx) -> Dict{Int,Float64}

Rest-length changes (`Δl0`) for every segment connected to
`point_idx` that would place it on top of point `target_idx`,
keeping each segment's other endpoint fixed.

For each connected segment, `Δl0` is the distance from its
fixed endpoint to the target position minus the current `l0`.
Returns a mapping from segment index to `Δl0`.
"""
function l0_diffs_to_move_point(sys, point_idx, target_idx)
    target = sys.points[target_idx].pos_w
    diffs = Dict{Int,Float64}()
    for seg in sys.segments
        a, b = seg.point_idxs
        a == point_idx || b == point_idx || continue
        other = a == point_idx ? b : a
        new_l0 = norm(sys.points[other].pos_w - target)
        diffs[seg.idx] = new_l0 - seg.l0
    end
    return diffs
end

"""
    distribute_wing_drag!(sys, area, drag_coeff)

Divide `area` equally over all wing points and set each
point's `drag_coeff`. This lets the solver account for
parasitic drag distributed along the span.
"""
function distribute_wing_drag!(sys, area, drag_coeff)
    wing_pts = [p for p in sys.points if p.type == WING]
    n = length(wing_pts)
    n > 0 || error("No wing points found")
    area_per_point = area / n
    for p in wing_pts
        p.area = area_per_point
        p.drag_coeff = drag_coeff
    end
    return nothing
end

"""
    distribute_wing_mass!(sys, mass; dist=0.75)

Distribute wing mass over LE-TE pairs proportional to chord
length. `dist` controls the LE/TE split (0.75 = 75% on LE).
"""
function distribute_wing_mass!(sys, mass; dist=0.75)
    wing_pts = sort(
        [p for p in sys.points if p.type == WING],
        by=p -> p.idx)
    n = length(wing_pts)
    iseven(n) || error(
        "Expected even number of wing points, got $n")
    pairs = [(wing_pts[i], wing_pts[i+1])
             for i in 1:2:n]
    chords = [norm(le.pos_b - te.pos_b)
              for (le, te) in pairs]
    total_chord = sum(chords)
    for (i, (le, te)) in enumerate(pairs)
        pair_mass = mass * chords[i] / total_chord
        le.extra_mass = pair_mass * dist
        te.extra_mass = pair_mass * (1 - dist)
    end
    return nothing
end

"""
    set_v3_body_damping!(sys, body_damping, point_37_38_damping)

Apply the V3 two-region body-frame damping pattern: `body_damping`
on points 1:38 and the `point_37_38_damping` override on 37:38.
"""
function set_v3_body_damping!(sys, body_damping,
                              point_37_38_damping)
    SymbolicAWEModels.set_body_frame_damping(
        sys, body_damping, 1:38)
    SymbolicAWEModels.set_body_frame_damping(
        sys, point_37_38_damping, 37:38)
    return nothing
end

"""
    generate_drag_adjusted_polars(drag_factor; data_path, src_dir, dst_dir)

Read 2D polar CSVs, multiply the `Cd` column by `drag_factor`, and
write the adjusted polars to `dst_dir`.
"""
function generate_drag_adjusted_polars(drag_factor;
        data_path=v3_data_path(),
        src_dir="2D_polars_CFD_NF_combined",
        dst_dir="2D_polars_drag_adjusted")
    src = joinpath(data_path, src_dir)
    dst = joinpath(data_path, dst_dir)
    mkpath(dst)
    for f in readdir(src)
        endswith(f, ".csv") || continue
        df = CSV.read(joinpath(src, f), DataFrame)
        df.Cd .*= drag_factor
        CSV.write(joinpath(dst, f), df)
    end
    return nothing
end

