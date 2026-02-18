# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite model setup utilities.
Functions for adjusting tether length, elevation, and other model parameters.
"""

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

"""
    segment_stretch_stats(sys_struct::SystemStructure)

Calculate segment stretch statistics for the system structure.

# Returns
- `(max_stretch, mean_stretch, max_idx)`: Maximum relative stretch, mean relative stretch,
  and the index of the segment with maximum stretch
"""
function segment_stretch_stats(sys_struct::SymbolicAWEModels.SystemStructure)
    @unpack segments, points = sys_struct

    stretches = Float64[]
    for seg in segments
        p1_idx, p2_idx = seg.point_idxs
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        actual_len = norm(p2.pos_w - p1.pos_w)
        rest_len = seg.l0
        if rest_len > 0
            relative_stretch = (actual_len - rest_len) / rest_len
            push!(stretches, relative_stretch)
        end
    end

    if isempty(stretches)
        return (0.0, 0.0, 0)
    end

    max_stretch = maximum(stretches)
    mean_stretch = mean(stretches)
    max_idx = argmax(stretches)

    return (max_stretch, mean_stretch, max_idx)
end
