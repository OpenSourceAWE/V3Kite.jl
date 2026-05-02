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
    adjust_tether_length!(sam::SymbolicAWEModel, tether_length_raw; tether_point_idxs=V3_TETHER_POINT_IDXS)

Update the winch rest length, reposition tether points in CAD/body frames,
and reapply the main transform so the wing stays at the requested tether radius.

# Arguments
- `sam`: SymbolicAWEModel to modify
- `tether_length_raw`: Target tether length in meters
- `tether_point_idxs`: Indices of tether points (default: V3_TETHER_POINT_IDXS = 39:44)
"""
function adjust_tether_length!(sam::SymbolicAWEModel, tether_length_raw;
                               tether_point_idxs=V3_TETHER_POINT_IDXS)
    tether_length = float(tether_length_raw)
    sys = sam.sys_struct
    set = sam.set

    if !isempty(set.l_tethers)
        set.l_tethers[1] = tether_length
    end

    n_points = length(tether_point_idxs)
    for (n, p_idx) in enumerate(tether_point_idxs)
        pos = (0.0, 0.0, -n * tether_length / n_points)
        sys.points[p_idx].pos_cad .= pos
        sys.points[p_idx].pos_b .= pos
    end

    if !isempty(sys.transforms)
        transform = sys.transforms[1]
        if !isempty(sys.wings) && norm(sys.wings[1].pos_w) > 0
            target_pos = normalize(sys.wings[1].pos_w) * tether_length
            transform.elevation = KiteUtils.calc_elevation(target_pos)
            transform.azimuth = KiteUtils.azimuth_east(target_pos)
        end
        SymbolicAWEModels.reinit!([transform], sys)
    end

    if !isempty(sys.tethers)
        sys.tethers[1].len = tether_length
        sys.tethers[1].stretched_len = tether_length
    end
    if !isempty(sys.winches)
        winch = sys.winches[1]
        winch.vel = 0.0
        winch.brake = true
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

"""
    adjust_elevation!(sam::SymbolicAWEModel, elevation_deg)

Update the transform elevation to the specified value in degrees.

# Arguments
- `sam`: SymbolicAWEModel to modify
- `elevation_deg`: Target elevation angle in degrees
"""
function adjust_elevation!(sam::SymbolicAWEModel, elevation_deg)
    sys = sam.sys_struct

    if !isempty(sys.transforms)
        transform = sys.transforms[1]
        transform.elevation = deg2rad(elevation_deg)
        SymbolicAWEModels.reinit!([transform], sys)
    end
    return nothing
end
