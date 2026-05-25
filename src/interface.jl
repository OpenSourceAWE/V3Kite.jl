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

"""
    lift_drag(s::V3KITE) -> (lift, drag)

Return the aerodynamic lift and drag forces [N] of the kite wing.
Lift is the component of `aero_force_b` perpendicular to the apparent wind;
drag is the component along the apparent wind direction.
"""
function lift_drag(s::V3KITE)
    wing = s.sys.wings[1]
    va_b = wing.va_b
    v_app = norm(va_b)
    v_app < 1e-6 && return (0.0, 0.0)
    va_hat = va_b / v_app
    drag_force = dot(wing.aero_force_b, va_hat)
    lift_force = norm(wing.aero_force_b .- drag_force .* va_hat)
    return (lift_force, drag_force)
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
