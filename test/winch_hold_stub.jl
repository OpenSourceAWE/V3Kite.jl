# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Length-holding torque source for the tests, built only from V3Kite's own
public interface (`force_to_torque`, `unstretched_length`, `reel_out_speed`).
Unlike `examples/winch_adapter.jl`, this does not depend on WinchControllers.jl:
a cascaded P controller (outer: length error -> speed setpoint, inner: speed
error -> torque correction) replaces WinchControllers' PI. Gains mirror V3's
tuning in `data/wc_settings.yaml` (`winch_pos_kp`, `winch_speed_k`).

    wpc = winch_pos_controller(s)
    step!(s; rel_depower, set_torque = winch_torque!(wpc, s, l0))
"""

"""
    winch_acc_limit(max_acc) -> Float64

Acceleration limit [m/s²] for the rate limiter in [`winch_torque!`](@ref). A
non-positive `max_acc` means unlimited, not a frozen drum.
"""
winch_acc_limit(max_acc) = max_acc > 0 ? Float64(max_acc) : Inf

"""
    WinchPosController(; kp_pos=0.5, kp_speed=30.0)

State of the cascaded length-holding torque controller: `kp_pos` is the outer
proportional gain, length error [m] to speed setpoint [m/s]; `kp_speed` the
inner one, speed error [m/s] to torque correction [N·m]. `v_sp_prev` carries
the rate-limited speed setpoint between steps.
"""
Base.@kwdef mutable struct WinchPosController
    kp_pos::Float64 = 0.5
    kp_speed::Float64 = 30.0
    v_sp_prev::Float64 = 0.0
end

"""
    winch_pos_controller(s::V3KITE) -> WinchPosController

The length-holding torque controller for `s`. One per model.
"""
winch_pos_controller(s::V3KITE) = WinchPosController()

"""
    winch_torque!(wpc::WinchPosController, s::V3KITE, set_length; v_ff=0.0,
                  speed_limit=Inf, acceleration_limit=winch_acc_limit(s.set.max_acc)) -> torque

Winch torque [N·m] holding `set_length`, for `step!`'s `set_torque`. `v_ff`
[m/s] is a speed feed-forward added to the outer loop's setpoint.
"""
function winch_torque!(wpc::WinchPosController, s::V3KITE, set_length;
                       v_ff = 0.0, speed_limit = Inf,
                       acceleration_limit = winch_acc_limit(s.set.max_acc))
    v_sp = v_ff + wpc.kp_pos * (set_length - unstretched_length(s))
    v_sp = clamp(v_sp, -speed_limit, speed_limit)
    dv_max = acceleration_limit * s.dt
    v_sp = clamp(v_sp, wpc.v_sp_prev - dv_max, wpc.v_sp_prev + dv_max)
    wpc.v_sp_prev = v_sp
    return force_to_torque(winch_force(s), s.sys) +
        wpc.kp_speed * (v_sp - reel_out_speed(s))
end
