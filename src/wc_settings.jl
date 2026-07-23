# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Settings of the cascaded winch length controller (see `_winch_position_torque!`),
loaded from a YAML file (`wc_settings.yaml`) rather than hard-coded. These are new
parameters with no historical counterpart; tune on `examples/simple_parking.jl`
(constant-length parking) and, per CLAUDE.md, change them by at most 10 % per
iteration.
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
