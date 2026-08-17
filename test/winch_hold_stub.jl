# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Length-holding torque source for the tests, built only from V3Kite's own
public interface (`force_to_torque`, `unstretched_length`, `reel_out_speed`,
`winch_force`). Unlike `examples/winch_adapter.jl`, this does not depend on
WinchControllers.jl. Its names are deliberately distinct from that adapter's
(`hold_torque!`/`LengthHoldController`, not `winch_torque!`/
`WinchPosController`), so a session that has loaded both never has one
silently shadow the other.

    lhc = length_hold_controller(s)
    step!(s; rel_depower, set_torque = hold_torque!(lhc, s, l0))
"""

"""
    length_hold_acc_limit(max_acc) -> Float64

Acceleration limit [m/s²] for the rate limiter in [`hold_torque!`](@ref). A
non-positive `max_acc` means unlimited, not a frozen drum.
"""
length_hold_acc_limit(max_acc) = max_acc > 0 ? Float64(max_acc) : Inf

"""
    LengthHoldController(; kp_pos=0.5, kp_speed=30.0)

State of the cascaded length-holding torque controller: `kp_pos` is the outer
proportional gain, length error [m] to speed setpoint [m/s]; `kp_speed` the
inner one, speed error [m/s] to torque correction [N·m]. `v_sp_prev` carries
the rate-limited speed setpoint between steps.
"""
Base.@kwdef mutable struct LengthHoldController
    kp_pos::Float64 = 0.5
    kp_speed::Float64 = 30.0
    v_sp_prev::Float64 = 0.0
end

"""
    length_hold_controller(s::V3KITE) -> LengthHoldController

The length-holding torque controller for `s`. One per model.
"""
length_hold_controller(s::V3KITE) = LengthHoldController()

"""
    hold_torque!(lhc::LengthHoldController, s::V3KITE, set_length;
                 acceleration_limit=length_hold_acc_limit(s.set.max_acc)) -> torque

Winch torque [N·m] holding `set_length`, for `step!`'s `set_torque`.
"""
function hold_torque!(lhc::LengthHoldController, s::V3KITE, set_length;
                      acceleration_limit = length_hold_acc_limit(s.set.max_acc))
    v_sp = lhc.kp_pos * (set_length - unstretched_length(s))
    dv_max = acceleration_limit * s.dt
    v_sp = clamp(v_sp, lhc.v_sp_prev - dv_max, lhc.v_sp_prev + dv_max)
    lhc.v_sp_prev = v_sp
    return force_to_torque(winch_force(s), s.sys) +
        lhc.kp_speed * (v_sp - reel_out_speed(s))
end
