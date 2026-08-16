# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plant/controller adapter, used by V3Kite's own examples and tests. V3Kite is the
plant: `step!` takes a winch torque. Holding a length is a control problem, so
that loop lives in WinchControllers.jl and is the caller's to own. This file is
the glue: it reads the plant scalars off a `V3KITE` and hands them to
WinchControllers' scalar-only functions, so neither package depends on the
other. `include` it and build a controller once per run:

    wpc = winch_pos_controller(s)
    step!(s; rel_depower, set_torque = winch_torque!(wpc, s, l0))
"""

using WinchControllers: WCSettings, WinchPosController, winch_position_torque!,
    winch_acc_limit

"""
    winch_pos_controller(s::V3KITE) -> WinchPosController

The cascaded length controller a run drives `s` with, at the model's own `dt`.
One per model. Gains come from the file named by the `wc_settings:` field of the
active system YAML, loaded by WinchControllers' own `WCSettings(true; dt)`.
"""
winch_pos_controller(s::V3KITE) =
    WinchPosController(WCSettings(true; dt = s.dt); dt = s.dt)

"""
    winch_torque!(wpc::WinchPosController, s::V3KITE, set_length;
                  v_ff=0.0, speed_limit=Inf,
                  acceleration_limit=winch_acc_limit(s.set.max_acc)) -> torque

Run WinchControllers.jl's cascaded length loop against `s` and return the winch
torque [N·m] for `step!`'s `set_torque`. Replaces the `set_length` keyword
`step!` used to carry before the winch controllers moved out of V3Kite.

`acceleration_limit` defaults to the plant's own `winch: max_acc:`, where
`step!` used to default it to `Inf`. `v_ff` [m/s] is the speed feed-forward: a
caller integrating a speed setpoint into `set_length` should pass that same
speed, or the outer P loop has to rediscover it from a length error at a cost
of `1/kp_pos` of lag.
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
