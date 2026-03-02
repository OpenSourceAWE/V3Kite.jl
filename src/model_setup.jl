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
Base.@kwdef struct V3GeomAdjustConfig
    reduce_tip::Bool = false
    tip_reduction::Float64 = 0.4
    tip_segments::Vector{Int} = [47, 48, 57, 58]

    reduce_te::Bool = false
    te_frac::Float64 = 0.95
    te_segments::UnitRange{Int} = 20:28

    reduce_depower::Bool = false
    depower_reduction::Float64 = 0.2

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
    if !isnothing(config.tether_length)
        wf = norm(sys.winches[1].force)
        wf = isnan(wf) ? 0.0 : wf
        stiffness = sys.segments[end].unit_stiffness
        n_segs = length(V3_TETHER_POINT_IDXS)
        seg_len = config.tether_length / n_segs *
            (1 + wf / stiffness)
        for (n, i) in enumerate(V3_TETHER_POINT_IDXS)
            sys.points[i].pos_cad .= [
                0.0, 0.0, -n * seg_len]
        end
        for i in 90:95
            sys.segments[i].l0 = seg_len
        end
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

    if !isempty(sys.winches)
        winch = sys.winches[1]
        winch.tether_len = tether_length
        winch.tether_vel = 0.0
        winch.brake = true
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
