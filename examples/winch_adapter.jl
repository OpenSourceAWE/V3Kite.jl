# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Glue between V3Kite (the plant: `step!` takes a torque) and WinchControllers'
scalar-only length loop, so neither package depends on the other:

    wpc = winch_pos_controller(s)
    step!(s; rel_depower, set_torque = winch_torque!(wpc, s, l0))
"""

using WinchControllers: WCSettings, WinchPosController, winch_position_torque!,
    winch_acc_limit

"""
    winch_pos_controller(s::V3KITE) -> WinchPosController

The cascaded length controller for `s`, at its own `dt`. One per model.
"""
winch_pos_controller(s::V3KITE) =
    WinchPosController(WCSettings(true; dt = s.dt); dt = s.dt)

"""
    winch_torque!(wpc, s::V3KITE, set_length; v_ff=0.0, speed_limit=Inf,
                  acceleration_limit=winch_acc_limit(s.set.max_acc)) -> torque

Winch torque [N·m] holding `set_length`, for `step!`'s `set_torque`. `v_ff`
[m/s] is the speed feed-forward; see `winch_position_torque!` for both.
"""
function winch_torque!(wpc::WinchPosController, s::V3KITE, set_length;
                       v_ff = 0.0, speed_limit = Inf,
                       acceleration_limit = winch_acc_limit(s.set.max_acc))
    r, G, friction = drum_params(s)
    return winch_position_torque!(wpc, set_length, unstretched_length(s),
                                  reel_out_speed(s), winch_force(s),
                                  r, G, friction, s.dt,
                                  speed_limit, acceleration_limit; v_ff)
end
