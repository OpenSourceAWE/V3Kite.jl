@with_kw mutable struct V3KITE <: AbstractKiteModel
    "Reference to the settings struct"
    set::Settings
    "Reference to the KCU model (Kite Control Unit) as implemented in the package KitePodModels"
    kcu::KCU
    sam::SymbolicAWEModel
    sys::SystemStructure
    "Reference to the atmospheric model as implemented in the package AtmosphericModels"
    am::AtmosphericModel = AtmosphericModel(set)
    
end

# Output functions

"""
    lift_drag(s::V3KITE) -> (lift, drag)

Return the aerodynamic lift and drag forces [N] of the kite wing.
Lift is the component of `aero_force_b` perpendicular to the apparent wind;
drag is the VSM wing drag plus bridle drag, both along the apparent wind direction.
"""
function lift_drag(s::V3KITE)
    wing = s.sys.wings[1]
    sys  = s.sys
    va_b = wing.va_b
    v_app = norm(va_b)
    v_app < 1e-6 && return (0.0, 0.0)
    va_hat = va_b / v_app

    # Wing: VSM drag (body frame) and lift (perpendicular component)
    vsm_drag  = dot(wing.aero_force_b, va_hat)
    lift_force = norm(wing.aero_force_b .- vsm_drag .* va_hat)

    # Bridle drag: non-tether, non-wing, non-KCU points (world frame)
    tether_pts = _tether_point_idxs(sys)
    bridle_w   = zeros(3)
    for p in sys.points
        p.idx in tether_pts && continue
        p.type == WING        && continue
        p.idx == 1            && continue  # KCU
        bridle_w .+= p.drag_force
    end
    va_hat_w    = calc_R_b_w(sys) * va_hat
    bridle_drag = dot(bridle_w, va_hat_w)

    return (lift_force, vsm_drag + bridle_drag)
end

"""
    unstretched_length(s::V3KITE) -> Float64

Return the current unstretched tether length [m].
"""
function unstretched_length(s::V3KITE)
    s.sys.tethers[1].len
end

"""
    v_wind_kite(s::V3KITE) -> Vector{Float64}

Return the wind velocity vector [m/s] at the kite's current height,
applying the atmospheric wind profile to the ground-level wind vector.
"""
function v_wind_kite(s::V3KITE)
    height = s.sys.wings[1].pos_w[3]
    calc_wind_factor(s.am, height) * s.set.wind_vec
end

"""
    pos_kite(s::V3KITE) -> Vector{Float64}

Return the centre-of-pressure position of the kite from the four mid-span wing points
(indices 10–13: LE/TE pairs). The centre of pressure is at ~30 % chord, so each pair
is weighted 0.7 × LE + 0.3 × TE, then averaged over both sides.
"""
function pos_kite(s::V3KITE)
    pts = s.sys.points
    (0.7 .* pts[10].pos_w .+ 0.3 .* pts[11].pos_w .+
     0.7 .* pts[12].pos_w .+ 0.3 .* pts[13].pos_w) ./ 2
end

"""
    tether_length(s::V3KITE) -> Float64

Calculate and return the real, stretched tether length [m] by summing
the Euclidean distances between the endpoints of each tether segment.
"""
function tether_length(s::V3KITE)
    tether = s.sys.tethers[1]
    len = 0.0
    for seg_idx in tether.segment_idxs
        seg = s.sys.segments[seg_idx]
        p1 = s.sys.points[seg.point_idxs[1]].pos_w
        p2 = s.sys.points[seg.point_idxs[2]].pos_w
        len += norm(p2 .- p1)
    end
    return len
end

"""
    calc_height(s::V3KITE) -> Float64

Return the height [m] of the kite's centre-of-pressure above ground
(z-component of `pos_kite`).
"""
function calc_height(s::V3KITE)
    pos_kite(s)[3]
end

"""
    calc_elevation(s::V3KITE) -> Float64

Determine the elevation angle of the kite in radian.
"""
function calc_elevation(s::V3KITE)
    KiteUtils.calc_elevation(pos_kite(s))
end

"""
    upwind_dir(s::V3KITE) -> Float64
    upwind_dir(v_wind_gnd) -> Float64

Return the upwind direction [rad] in the horizontal plane, measured clockwise
from North (NED convention). Computed from the ground-level wind vector so that
the result points into the wind (i.e. opposite to the wind direction).
Returns `NaN` if the wind vector has no horizontal component.
"""
function upwind_dir(s::V3KITE)
    upwind_dir(s.set.wind_vec)
end
function upwind_dir(v_wind_gnd)
    if v_wind_gnd[1] == 0.0 && v_wind_gnd[2] == 0.0
        return NaN
    end
    wind_dir = atan(v_wind_gnd[2], v_wind_gnd[1])
    -(wind_dir + π/2)
end

"""
    calc_azimuth(s::V3KITE) -> Float64

Determine the azimuth angle of the kite in wind reference frame in radian.
Positive anti-clockwise when seen from above.
"""
function calc_azimuth(s::V3KITE)
    azn = KiteUtils.azimuth_north(pos_kite(s))
    azn2azw(azn; upwind_dir = upwind_dir(s))
end

"""
    calc_azimuth_east(s::V3KITE) -> Float64

Determine the azimuth_east angle of the kite in radian.
Positive clockwise when seen from above.
"""
function calc_azimuth_east(s::V3KITE)
    KiteUtils.azimuth_east(pos_kite(s))
end

"""
    calc_azimuth_north(s::V3KITE) -> Float64

Determine the azimuth_north angle of the kite in radian.
Positive anti-clockwise when seen from above.
"""
function calc_azimuth_north(s::V3KITE)
    KiteUtils.azimuth_north(pos_kite(s))
end

