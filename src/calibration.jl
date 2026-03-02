# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite steering and depower calibration constants and conversion functions.
Based on KCU (Kite Control Unit) documentation for the TU Delft V3 kite.

## Calibration Structure

The V3 kite calibration uses a base + delta pattern:
- **Base values**: Official measured values from KCU documentation
- **Delta**: Empirical correction factor (default: -0.2m)
- **Effective values**: base + delta (used in simulation)

This structure allows:
1. Transparent documentation of official measurements
2. Easy adjustment of the empirical delta for tuning
3. Backward compatibility with existing code using effective values
"""

# =============================================================================
# Official measured values (KCU documentation)
# =============================================================================

"""Base neutral steering tape length from KCU documentation (m)"""
const V3_STEERING_L0_BASE = 1.6

"""Maximum differential steering (m) at |u_s| = 1"""
const V3_STEERING_GAIN = 1.4

"""Base neutral depower tape length from KCU documentation (m)"""
const V3_DEPOWER_L0_BASE = 0.2

"""Depower range (m) for 0-100%"""
const V3_DEPOWER_GAIN = 5.0

# =============================================================================
# Default calibration delta
# =============================================================================

"""
Default calibration delta (m).
This empirical correction accounts for the difference between measured
KCU values and effective simulation values.
"""
const V3_DEFAULT_DELTA = -0.2

# =============================================================================
# Effective values (backward compatible exports)
# =============================================================================

"""Effective neutral steering tape length (m): V3_STEERING_L0_BASE + V3_DEFAULT_DELTA"""
const V3_STEERING_L0 = V3_STEERING_L0_BASE + V3_DEFAULT_DELTA  # = 1.4

"""Effective neutral depower tape length (m): V3_DEPOWER_L0_BASE + V3_DEFAULT_DELTA"""
const V3_DEPOWER_L0 = V3_DEPOWER_L0_BASE + V3_DEFAULT_DELTA    # = 0.0

# V3 Kite segment indices
"""Tether point indices in the V3 kite model"""
const V3_TETHER_POINT_IDXS = 39:44

"""Left steering tape segment index"""
const V3_STEERING_LEFT_IDX = 87

"""Right steering tape segment index"""
const V3_STEERING_RIGHT_IDX = 89

"""Depower tape segment index"""
const V3_DEPOWER_IDX = 88

"""
    steering_percentage_to_lengths(percentage;
        l0_base=V3_STEERING_L0_BASE, gain=V3_STEERING_GAIN, delta=V3_DEFAULT_DELTA)

Convert steering percentage to left/right tape lengths (m).
Percentage convention: negative = left turn, positive = right turn.
Uses half-gain on each side for symmetric actuation.

# Arguments
- `percentage`: Steering percentage in range [-100, 100]
- `l0_base`: Base neutral steering tape length (m)
- `gain`: Maximum differential (m) at |percentage| = 100
- `delta`: Calibration delta adjustment (m)

# Returns
- `(L_left, L_right)`: Left and right tape lengths in meters

# Notes
The effective neutral length is computed as `l0 = l0_base + delta`.
For backward compatibility, uses the effective default V3_STEERING_L0 = 1.4m.
"""
function steering_percentage_to_lengths(percentage;
                                        l0_base=V3_STEERING_L0_BASE,
                                        gain=V3_STEERING_GAIN,
                                        delta=V3_DEFAULT_DELTA)
    l0 = l0_base + delta
    u_s = percentage / 100.0
    L_left = l0 - (gain / 2.0) * u_s
    L_right = l0 + (gain / 2.0) * u_s
    return L_left, L_right
end

"""
    csv_steering_percentage_to_lengths(percentage;
        l0_base=V3_STEERING_L0_BASE, gain=V3_STEERING_GAIN, delta=V3_DEFAULT_DELTA)

Convert CSV steering percentage to left/right tape lengths (m).
Uses opposite sign convention and full gain (matches CSV data format from flight tests).

# Arguments
- `percentage`: Steering percentage from CSV in range [-100, 100]
- `l0_base`: Base neutral steering tape length (m)
- `gain`: Maximum differential (m) at |percentage| = 100
- `delta`: Calibration delta adjustment (m)

# Returns
- `(L_left, L_right)`: Left and right tape lengths in meters
"""
function csv_steering_percentage_to_lengths(percentage;
                                            l0_base=V3_STEERING_L0_BASE,
                                            gain=V3_STEERING_GAIN,
                                            delta=V3_DEFAULT_DELTA)
    l0 = l0_base + delta
    u_s = percentage / 100.0
    L_left = l0 + gain * u_s
    L_right = l0 - gain * u_s
    return L_left, L_right
end

"""
    depower_percentage_to_length(percentage;
        l0_base=V3_DEPOWER_L0_BASE, gain=V3_DEPOWER_GAIN, delta=V3_DEFAULT_DELTA)

Convert depower percentage to tape length (m).

# Arguments
- `percentage`: Depower percentage in range [0, 100]
- `l0_base`: Base neutral depower tape length (m)
- `gain`: Depower range (m) for 0-100%
- `delta`: Calibration delta adjustment (m)

# Returns
- Depower tape length in meters
"""
function depower_percentage_to_length(percentage;
                                      l0_base=V3_DEPOWER_L0_BASE,
                                      gain=V3_DEPOWER_GAIN,
                                      delta=V3_DEFAULT_DELTA)
    l0 = l0_base + delta
    u_p = percentage / 100.0
    return l0 + gain * u_p
end

"""
    steering_length_to_percentage(L_left, L_right; gain=V3_STEERING_GAIN)

Convert left/right tape lengths back to steering percentage.
Inverse of `steering_percentage_to_lengths`.

# Arguments
- `L_left`: Left tape length (m)
- `L_right`: Right tape length (m)
- `gain`: Maximum differential (m)

# Returns
- Steering percentage in range [-100, 100]

# Notes
The inverse only depends on the gain, not on l0 or delta, because:
- L_left = l0 - (gain/2) * u_s
- L_right = l0 + (gain/2) * u_s
- u_s = (L_right - L_left) / gain
"""
function steering_length_to_percentage(L_left, L_right; gain=V3_STEERING_GAIN)
    u_s = (L_right - L_left) / gain
    return u_s * 100.0
end

"""
    depower_length_to_percentage(length;
        l0_base=V3_DEPOWER_L0_BASE, gain=V3_DEPOWER_GAIN, delta=V3_DEFAULT_DELTA)

Convert depower tape length back to percentage.
Inverse of `depower_percentage_to_length`.

# Arguments
- `length`: Depower tape length (m)
- `l0_base`: Base neutral depower tape length (m)
- `gain`: Depower range (m)
- `delta`: Calibration delta adjustment (m)

# Returns
- Depower percentage in range [0, 100]
"""
function depower_length_to_percentage(length;
                                      l0_base=V3_DEPOWER_L0_BASE,
                                      gain=V3_DEPOWER_GAIN,
                                      delta=V3_DEFAULT_DELTA)
    l0 = l0_base + delta
    u_p = (length - l0) / gain
    return u_p * 100.0
end

"""
    build_geom_suffix(depower_l0, tip_reduction, te_frac)

Build geometry filename suffix from configuration parameters.
Used for parametric geometry variations.

# Arguments
- `depower_l0`: Depower tape neutral length
- `tip_reduction`: Tip leading edge reduction (m)
- `te_frac`: Trailing edge wire factor

# Returns
- String suffix for geometry filenames
"""
function build_geom_suffix(depower_l0, tip_reduction, te_frac)
    return "depower$(depower_l0)_tip$(tip_reduction)_te$(te_frac)"
end

# =============================================================================
# Normalized control functions (KiteModels.jl compatible interface)
# =============================================================================

"""
    set_steering!(sys, steering)

Set the steering input. Internally converts to tape segment lengths.

# Arguments
- `sys`: SystemStructure from the kite model
- `steering`: Relative steering, must be between -1.0 .. 1.0
              (-1.0 = full left, 0.0 = neutral, 1.0 = full right)
"""
function set_steering!(sys, steering)
    L_left, L_right = steering_percentage_to_lengths(steering * 100.0)
    sys.segments[V3_STEERING_LEFT_IDX].l0 = L_left
    sys.segments[V3_STEERING_RIGHT_IDX].l0 = L_right
    return nothing
end

"""
    get_steering(sys)

Get the current steering value as a normalized input.

# Arguments
- `sys`: SystemStructure from the kite model

# Returns
- Relative steering in range -1.0 .. 1.0
"""
function get_steering(sys)
    L_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    L_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
    return steering_length_to_percentage(L_left, L_right) / 100.0
end

"""
    set_depower!(sys, depower)

Set the depower input. Internally converts to tape segment length.

# Arguments
- `sys`: SystemStructure from the kite model
- `depower`: Relative depower, must be between 0.0 .. 1.0
             (0.0 = no depower, 1.0 = full depower)
"""
function set_depower!(sys, depower)
    L_depower = depower_percentage_to_length(depower * 100.0)
    sys.segments[V3_DEPOWER_IDX].l0 = L_depower
    return nothing
end

"""
    set_depower!(sys, depower, config::V3GeomAdjustConfig)

Set the depower input, accounting for depower tape reduction
from the geometry config.

# Arguments
- `sys`: SystemStructure from the kite model
- `depower`: Relative depower, must be between 0.0 .. 1.0
- `config`: Geometry adjustment config with optional
            depower reduction
"""
function set_depower!(sys, depower, config::V3GeomAdjustConfig)
    reduction = config.reduce_depower ?
        config.depower_reduction : 0.0
    L_depower = depower_percentage_to_length(
        depower * 100.0;
        l0_base=V3_DEPOWER_L0_BASE - reduction)
    sys.segments[V3_DEPOWER_IDX].l0 = L_depower
    return nothing
end

"""
    get_depower(sys)

Get the current depower value as a normalized input.

# Arguments
- `sys`: SystemStructure from the kite model

# Returns
- Relative depower in range 0.0 .. 1.0
"""
function get_depower(sys)
    L_depower = sys.segments[V3_DEPOWER_IDX].l0
    return depower_length_to_percentage(L_depower) / 100.0
end
