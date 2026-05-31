# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite steering and depower calibration constants and conversion
functions. Based on KCU (Kite Control Unit) documentation for the
TU Delft V3 kite.

Tape reductions (shortening of steering/depower tapes) are applied
through `V3GeomAdjustConfig`, not through global constants.
"""

# =============================================================================
# Official measured values (KCU documentation)
# =============================================================================

"""Base neutral steering tape length from KCU documentation (m)"""
const V3_STEERING_L0_BASE = 1.6

"""Maximum differential steering (m) at |u_s| = 1"""
const V3_STEERING_GAIN = 1.4

"""Base neutral depower tape length from 2025 KCU calibration (m)"""
const V3_DEPOWER_L0_BASE = 0.192

"""Legacy linear depower range (m) for 0-100%"""
const V3_DEPOWER_GAIN = 5.0

"""Quadratic coefficient for `udp^2` in the 2025 depower calibration"""
const UDP_2025_QUADRATIC_A = -1.7237

"""Linear coefficient for `udp` in the 2025 depower calibration"""
const UDP_2025_QUADRATIC_B = 6.623795

"""Constant coefficient in the 2025 depower calibration"""
const UDP_2025_QUADRATIC_C = V3_DEPOWER_L0_BASE

# V3 Kite segment indices for data/python_yamls/struc_geometry_julia_generated.yaml
"""Left steering tape segment index"""
const V3_STEERING_LEFT_IDX = 91

"""Right steering tape segment index"""
const V3_STEERING_RIGHT_IDX = 89

"""Depower tape segment index"""
const V3_DEPOWER_IDX = 90

"""
    steering_percentage_to_lengths(percentage;
        l0_base=V3_STEERING_L0_BASE,
        gain=V3_STEERING_GAIN)

Convert steering percentage to left/right tape lengths (m).
Positive percentage shortens left tape (right turn when sitting
on kite). Full gain per side.

# Arguments
- `percentage`: Steering percentage in range [-100, 100]
- `l0_base`: Base neutral steering tape length (m)
- `gain`: Maximum differential (m) at |percentage| = 100

# Returns
- `(L_left, L_right)`: Left and right tape lengths in meters
"""
function steering_percentage_to_lengths(percentage;
        l0_base=V3_STEERING_L0_BASE,
        gain=V3_STEERING_GAIN)
    u_s = percentage / 100.0
    L_left = l0_base - gain * u_s
    L_right = l0_base + gain * u_s
    return L_left, L_right
end

"""
    udp_to_depower_tape_length_m(udp)

Convert Ch. 9 2025 depower command to depower-tape length in meters.

# Arguments
- `udp`: Depower command in range [0, 1]

# Returns
- Depower tape length in meters
"""
function udp_to_depower_tape_length_m(udp)
    return UDP_2025_QUADRATIC_A * udp^2 +
        UDP_2025_QUADRATIC_B * udp +
        UDP_2025_QUADRATIC_C
end

"""
    depower_percentage_to_length(percentage;
        l0_base=V3_DEPOWER_L0_BASE,
        gain=V3_DEPOWER_GAIN)

Convert depower percentage to tape length (m) using the 2025
quadratic UDP calibration. The `gain` keyword is kept for API
compatibility and is not used by this calibration.

# Arguments
- `percentage`: Depower percentage in range [0, 100]
- `l0_base`: Base neutral depower tape length (m)
- `gain`: Deprecated legacy linear gain

# Returns
- Depower tape length in meters
"""
function depower_percentage_to_length(percentage;
        l0_base=V3_DEPOWER_L0_BASE,
        gain=V3_DEPOWER_GAIN)
    udp = percentage / 100.0
    return udp_to_depower_tape_length_m(udp) +
        (l0_base - V3_DEPOWER_L0_BASE)
end

"""
    steering_length_to_percentage(L_left, L_right;
        gain=V3_STEERING_GAIN)

Convert left/right tape lengths back to steering percentage.
Inverse of `steering_percentage_to_lengths`.

# Arguments
- `L_left`: Left tape length (m)
- `L_right`: Right tape length (m)
- `gain`: Maximum differential (m)

# Returns
- Steering percentage in range [-100, 100]

# Notes
The inverse only depends on the gain, not on l0_base:
- L_left = l0_base - gain * u_s
- L_right = l0_base + gain * u_s
- u_s = (L_right - L_left) / (2 * gain)
"""
function steering_length_to_percentage(L_left, L_right;
        gain=V3_STEERING_GAIN)
    u_s = (L_right - L_left) / (2.0 * gain)
    return u_s * 100.0
end

"""
    depower_length_to_percentage(length;
        l0_base=V3_DEPOWER_L0_BASE,
        gain=V3_DEPOWER_GAIN)

Convert depower tape length back to percentage.
Inverse of the 2025 quadratic `depower_percentage_to_length`.

# Arguments
- `length`: Depower tape length (m)
- `l0_base`: Base neutral depower tape length (m)
- `gain`: Deprecated legacy linear gain

# Returns
- Depower percentage in range [0, 100]
"""
function depower_length_to_percentage(length;
        l0_base=V3_DEPOWER_L0_BASE,
        gain=V3_DEPOWER_GAIN)
    target = length - (l0_base - V3_DEPOWER_L0_BASE)
    a = UDP_2025_QUADRATIC_A
    b = UDP_2025_QUADRATIC_B
    c = UDP_2025_QUADRATIC_C - target
    disc = b^2 - 4.0 * a * c
    disc < 0 && throw(DomainError(
        length, "depower tape length is outside the 2025 UDP calibration range"))
    r1 = (-b + sqrt(disc)) / (2.0 * a)
    r2 = (-b - sqrt(disc)) / (2.0 * a)
    udp = if 0.0 <= r1 <= 1.0
        r1
    elseif 0.0 <= r2 <= 1.0
        r2
    else
        throw(DomainError(
            length, "depower tape length is outside the 2025 UDP calibration range"))
    end
    return udp * 100.0
end

"""
    build_geom_suffix(depower_tape, steering_left,
        steering_right, tip_reduction, te_frac)

Build geometry filename suffix from tape lengths.

# Arguments
- `depower_tape`: Depower tape length (m)
- `steering_left`: Left steering tape length (m)
- `steering_right`: Right steering tape length (m)
- `tip_reduction`: Tip leading edge reduction (m)
- `te_frac`: Trailing edge wire factor
"""
function build_geom_suffix(depower_tape,
        steering_left, steering_right,
        tip_reduction, te_frac)
    dp = round(depower_tape, digits=2)
    sl = round(steering_left, digits=2)
    sr = round(steering_right, digits=2)
    return "dp$(dp)_sl$(sl)_sr$(sr)" *
        "_tip$(tip_reduction)_te$(te_frac)"
end

# =============================================================================
# Normalized control functions (require V3GeomAdjustConfig)
# =============================================================================

"""
    set_steering!(sys, steering,
        config::V3GeomAdjustConfig; min_l0=0.0)

Set the steering input, accounting for steering tape reduction
from the geometry config.

# Arguments
- `sys`: SystemStructure from the kite model
- `steering`: Relative steering, must be between -1.0 .. 1.0
              (-1.0 = full left, 0.0 = neutral, 1.0 = full right)
- `config`: Geometry adjustment config
- `min_l0`: Minimum tape length clamp (m)
"""
function set_steering!(sys, steering,
        config::V3GeomAdjustConfig; min_l0=0.0)
    reduction = config.reduce_steering ?
        config.steering_reduction : 0.0
    L_left, L_right = steering_percentage_to_lengths(
        steering * 100.0;
        l0_base=V3_STEERING_L0_BASE - reduction)
    sys.segments[V3_STEERING_LEFT_IDX].l0 =
        max(min_l0, L_left)
    sys.segments[V3_STEERING_RIGHT_IDX].l0 =
        max(min_l0, L_right)
    return nothing
end

"""
    get_steering(sys, config::V3GeomAdjustConfig)

Get the current steering value as a normalized input.

# Arguments
- `sys`: SystemStructure from the kite model
- `config`: Geometry adjustment config

# Returns
- Relative steering in range -1.0 .. 1.0
"""
function get_steering(sys, ::V3GeomAdjustConfig)
    L_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    L_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
    return steering_length_to_percentage(
        L_left, L_right) / 100.0
end

"""
    set_depower!(sys, udp, steering,
        config::V3GeomAdjustConfig)

Set the UDP depower input, accounting for depower tape reduction,
depower offset, and steering-based depower offset from the
geometry config.

# Arguments
- `sys`: SystemStructure from the kite model
- `udp`: Relative depower, must be between 0.0 .. 1.0
             (0.0 = no depower, 1.0 = full depower)
- `steering`: Relative steering in -1.0 .. 1.0 (used for
              steering-dependent depower offset)
- `config`: Geometry adjustment config
"""
function set_depower!(sys, udp, steering,
        config::V3GeomAdjustConfig)
    dp = udp + config.depower_offset +
        config.steering_dp_offset * abs(steering)
    reduction = config.reduce_depower ?
        config.depower_reduction : 0.0
    L_depower = depower_percentage_to_length(
        dp * 100.0;
        l0_base=V3_DEPOWER_L0_BASE - reduction)
    sys.segments[V3_DEPOWER_IDX].l0 = L_depower
    return dp
end

"""
    get_depower(sys, config::V3GeomAdjustConfig)

Get the current depower value as a normalized input.

# Arguments
- `sys`: SystemStructure from the kite model
- `config`: Geometry adjustment config

# Returns
- Relative depower in range 0.0 .. 1.0
"""
function get_depower(sys, config::V3GeomAdjustConfig)
    reduction = config.reduce_depower ?
        config.depower_reduction : 0.0
    L_depower = sys.segments[V3_DEPOWER_IDX].l0
    return depower_length_to_percentage(
        L_depower;
        l0_base=V3_DEPOWER_L0_BASE - reduction) / 100.0
end
