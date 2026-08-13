# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite model setup utilities.
Functions for adjusting tether length, elevation, and other model parameters.
"""

"""
    V3BridleConfig

Material of the bridle lines and the canopy membranes, shared by the geometry
generator that emits them ([`V3Kite.SurfplanAdapter.V3BeamTopology`](@ref)) and
the run that loads them ([`apply_bridle_material!`](@ref)), so a sweep needs
neither a re-export nor a recompile.
"""
Base.@kwdef struct V3BridleConfig
    """
    Fraction of tensile stiffness a segment keeps under compression, making
    lines and fabric nearly tension-only. `0` carries no compressive force at all.
    """
    compression_frac::Float64 = 0.01
    """
    The same for the damping term. The default `1.0` leaves damping unaffected by
    compression, so a line's damping ratio jumps the moment it goes slack; setting
    this to `compression_frac` keeps one ratio on both branches.
    """
    compression_damping_frac::Float64 = 1.0
    "`unit_damping` per `unit_stiffness` on the lines and tapes only, not the canopy [s]"
    bridle_rel_damping::Float64 = 0.01
    """
    Fraction of line tension a bridle sheave passes on, the rest opposing rope
    travel. The default is SymbolicAWEModels' own sealed ball-bearing sheave; no
    V3 pulley has been measured.
    """
    pulley_efficiency::Float64 = 0.95
end

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
tip leading-edge reduction and trailing-edge wire shortening.

Both were fitted against the particle lattice and address segments by position, so
they are skipped with a warning on a structure whose segments are laid out
differently — the beam geometry from [`V3Kite.SurfplanAdapter`](@ref) must not
inherit a correction fitted for another structure. A beam wing is recognised by its
`TimoshenkoJoint`s; being the larger structure, it has a segment at every one of
these indices, so an in-range check alone would let the corrections land on canopy
membranes instead.
"""
function apply_geom_adjustments!(sys, config::V3GeomAdjustConfig)
    in_range(idxs) = all(idx -> idx in eachindex(sys.segments), idxs)
    if !isempty(sys.timoshenko_joints) &&
            (config.reduce_tip || config.reduce_te)
        @warn "Skipping the tip/TE reductions on a beam wing: they are indices " *
              "into the particle lattice, whose wing segments this has none of"
        return nothing
    end
    if config.reduce_tip
        if in_range(config.tip_segments)
            for idx in config.tip_segments
                sys.segments[idx].l0 -= config.tip_reduction
            end
        else
            @warn "Skipping tip reduction: no segments at those indices" config.tip_segments
        end
    end
    if config.reduce_te
        if in_range(config.te_segments)
            for idx in config.te_segments
                sys.segments[idx].l0 *= config.te_frac
            end
        else
            @warn "Skipping TE reduction: no segments at those indices" config.te_segments
        end
    end
    return nothing
end

"""
    distribute_wing_drag!(sys, area, drag_coeff)

Divide `area` equally over all wing points and set each
point's `drag_coeff`. This lets the solver account for
parasitic drag distributed along the span.
"""
function distribute_wing_drag!(sys, area, drag_coeff)
    wing_pts = [p for p in sys.points if p.is_wing_node]
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
        [p for p in sys.points if p.is_wing_node],
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
    tether_point_idxs(sys) -> Vector{Int}

Sorted indices of every point on a winched tether: the endpoints
of all segments those tethers reference. A winch-less tether is a
bridle line that was split into segments, not part of the tether.
"""
function tether_point_idxs(sys)
    idxs = Set{Int}()
    winched = Set(idx for winch in sys.winches for idx in winch.tether_idxs)
    for tether in sys.tethers, segment_idx in tether.segment_idxs
        tether.idx in winched || continue
        i, j = sys.segments[segment_idx].point_idxs
        push!(idxs, i, j)
    end
    return sort!(collect(idxs))
end

"""
    set_body_frame_damping!(sys, damping)

Apply body-frame `damping` to every point except tether points
(see [`tether_point_idxs`](@ref)). A tether-skipping replacement for
`SymbolicAWEModels.set_body_frame_damping`.
"""
function set_body_frame_damping!(sys, damping)
    skip = Set(tether_point_idxs(sys))
    keep = [i for i in eachindex(sys.points) if !(i in skip)]
    SymbolicAWEModels.set_body_frame_damping(sys, damping, keep)
    return nothing
end

"""
    set_v3_body_damping!(sys, body_damping, point_37_38_damping)

Apply the V3 two-region body-frame damping pattern: `body_damping`
on all non-tether points and the `point_37_38_damping` override on
points 37:38.
"""
function set_v3_body_damping!(sys, body_damping,
                              point_37_38_damping)
    set_body_frame_damping!(sys, body_damping)
    SymbolicAWEModels.set_body_frame_damping(
        sys, point_37_38_damping, 37:38)
    return nothing
end

"""
    tether_bridle_segments(sys) -> Vector{Int}

Indices of the tether and bridle segments, i.e. every segment except the wing
frame (LE tubes, struts, TE wires, diagonals), which is stiff in compression by
design and would dominate any compression metric. A segment belongs to the wing
frame when both of its endpoints are wing nodes.
"""
function tether_bridle_segments(sys)
    points = sys.points
    return [seg.idx for seg in sys.segments
            if !(points[seg.point_idxs[1]].is_wing_node &&
                 points[seg.point_idxs[2]].is_wing_node)]
end

"""
    set_damping_per_stiffness!(sys, seg_idxs, ratio)

Damp the segments `seg_idxs` proportionally to their stiffness:
`unit_damping = ratio * unit_stiffness` [N·s], the relation the `materials`
section of `struc_geometry.yaml` uses to derive `unit_damping` from
`damping_per_stiffness`. Segments carrying a callable force law instead of a
stiffness are skipped, as there is no `unit_stiffness` to scale.

`unit_damping` is a flattened MTK parameter re-synced from the live
`SystemStructure` before every `step!`, so this takes effect on the next step
without rebuilding the model.

[`init`](@ref) calls this for its `damping_per_stiffness` keyword before
settling, which is the usual way in: the run is then damped from the very first
step, and settling and flight never differ in it. Calling it on an already
settled model instead steps the damping mid-run, which the short bridle segments
tolerate badly (see `examples/simple_parking.jl`).

See [`tether_bridle_segments`](@ref) for the usual choice of `seg_idxs`.
"""
function set_damping_per_stiffness!(sys, seg_idxs, ratio)
    segments = sys.segments
    for idx in seg_idxs
        segments[idx].unit_stiffness isa Real || continue
        segments[idx].unit_damping = ratio * segments[idx].unit_stiffness
    end
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

