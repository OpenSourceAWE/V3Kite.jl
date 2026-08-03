# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Settings of the two winch controllers — the cascaded length controller
(`winch_position_torque!`, the `winch_pos_*`/`winch_speed_*`/`winch_ff_scale`
fields) and force mode (`winch_force_torque!`, the `winch_force_*`/`winch_len_kp`/
`winch_damp` fields) — loaded from a YAML file (`wc_settings.yaml`) rather than
hard-coded. A run uses one mode or the other; the two share only the drum. These are new
parameters with no historical counterpart; tune on `examples/simple_parking.jl`
(constant-length parking) and change them by at most 10 % per iteration.
"""
@with_kw mutable struct WC_Settings @deftype Float64
    "Outer proportional gain: tether length error [m] → reel-out speed setpoint [1/s]"
    winch_pos_kp = 0.5
    "Inner speed-loop proportional gain: speed error [m/s] → winch torque [N·m·s/m]"
    winch_speed_k = 30.0
    "Inner speed-loop integral time [s] (larger = weaker integral action)"
    winch_speed_ti = 2.0
    "Saturation of the inner speed loop's torque correction [N·m]"
    winch_torque_limit = 500.0
    """
    Scale on the load term of the force feed-forward,
    `τ = -ff_scale·r/G·F + friction + Δτ`. The friction compensation is never
    scaled: it cancels the drum's own friction, not the load, and flips sign with
    the reel-out direction.

    `1.0` (the default) cancels the measured tether force exactly, making the drum
    perfectly stiff at any PI gain; below 1.0 the holding torque falls short and
    the drum pays out. The compliance is transient unless `winch_speed_ti` is well
    above the timescale you want the winch to yield on, since the inner PI
    integrates the resulting speed error away.
    """
    winch_ff_scale = 1.0
    """
    Force mode only (`winch_force_torque!`): time constant of the low-pass that
    turns the measured winch force into the reference force [s]. The drum yields to
    everything faster than this and holds everything slower, so this is the winch's
    compliance timescale — it should sit above the lap rate to let the kite trade
    line length for speed within a lap.
    """
    winch_force_tau = 10.0
    "Force mode only: length-trim gain, length error [m] → reference force [N/m]"
    winch_len_kp = 100.0
    """
    Force mode only: viscous damping on the drum, reel-out speed [m/s] → reference
    force [N·s/m]. Without it force mode has no velocity feedback at all — the drum
    is a free mass on the `winch_len_kp` spring and its speed runs away (measured:
    3.5 m/s of payout, `v_app` 49.6 m/s, run lost at t = 19.8 s). This is what makes
    the winch compliant rather than free-wheeling; it opposes fast reel-out without
    pinning the length the way the position cascade does.
    """
    winch_damp = 500.0
    "Force mode only: floor on the reference force, keeps the tether taut [N]"
    winch_force_min = 100.0
end

"""
    WC_Settings(filename::String) -> WC_Settings

Load winch-controller settings from the YAML file `filename` (looked up under the
active data path, i.e. `joinpath(get_data_path(), filename)`). The file must have a
top-level `wc_settings:` mapping whose keys are the field names of `WC_Settings`;
any missing key falls back to the struct default. Call `set_data_path(v3_data_path())`
(as `init` does) before this so the lookup resolves to `data/`.
"""
function WC_Settings(filename::String)
    dict = YAML.load_file(joinpath(get_data_path(), filename))["wc_settings"]
    WC_Settings(; (Symbol(k) => Float64(v) for (k, v) in dict)...)
end
