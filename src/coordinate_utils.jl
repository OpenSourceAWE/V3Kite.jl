# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Coordinate transformation and heading calculation utilities for the V3 kite.
"""

const COURSE_RATE_WINDOW_SEC = 10.0

"""
    wrap_to_pi(angle)

Wrap angle to [-π, π) range.
"""
function wrap_to_pi(angle)
    return mod(angle + π, 2π) - π
end

"""
    euler_to_quaternion(roll, pitch, yaw)

Convert Euler angles (in radians) to quaternion.
Converts from NED to ENU frame:
  X_ENU = Y_NED (East)
  Y_ENU = X_NED (North)
  Z_ENU = -Z_NED (Up = -Down)

# Arguments
- `roll`: Roll angle in radians
- `pitch`: Pitch angle in radians
- `yaw`: Yaw angle in radians

# Returns
- Quaternion [w, x, y, z]
"""
function euler_to_quaternion(roll, pitch, yaw)
    rot_ned = RotZYX(yaw, pitch, roll)
    R_ned_to_enu = [0.0 1.0 0.0;
                    1.0 0.0 0.0;
                    0.0 0.0 -1.0]
    rot_enu = R_ned_to_enu * Matrix(rot_ned)
    q = SymbolicAWEModels.rotation_matrix_to_quaternion(rot_enu)
    return q
end

function _calc_heading_from_rotation(R_b_w, pos_w)
    e_x = R_b_w[:, 1]
    # Tangential sphere frame (same as scalar_eqs.jl)
    z = normalize(pos_w)
    y = normalize([-pos_w[2], pos_w[1], 0.0])
    x = cross(y, z)
    return atan(dot(e_x, y), dot(e_x, x))
end

"""
    calc_csv_heading(roll, pitch, yaw, pos_w)

Heading from EKF NED Euler angles and kite position,
using the tangential sphere frame. Adds π because
`euler_to_quaternion` produces a body x-axis that
points opposite to SymbolicAWEModels' convention.
"""
function calc_csv_heading(roll, pitch, yaw, pos_w)
    quat = euler_to_quaternion(roll, pitch, yaw)
    R = SymbolicAWEModels.quaternion_to_rotation_matrix(
        quat)
    return wrap_to_pi(_calc_heading_from_rotation(R, pos_w) + π)
end

"""
    calc_R_b_w(sys_struct::SystemStructure)

Calculate the body-to-world rotation matrix for PARTICLE_DYNAMICS wing type.

# Arguments
- `sys_struct`: System structure with PARTICLE_DYNAMICS wing

# Returns
- 3x3 rotation matrix R_b_w
"""
function calc_R_b_w(sys_struct::SymbolicAWEModels.SystemStructure)
    @unpack points, wings = sys_struct
    wing = wings[1]
    R_b_w, origin = SymbolicAWEModels.calc_particle_dynamics_wing_frame(
        points,
        wing.z_ref_points,
        wing.y_ref_points,
        wing.idx
    )
    return R_b_w
end

"""
    calc_turn_rate(syslog; source=:heading, dt=0.01)

Frame-transport-corrected turn rate series for `syslog`,
returned as a vector aligned to `syslog.time[2:end]`.

`source` selects the angle whose derivative is taken:
`:heading` (uses `syslog.heading`) or `:course` (uses
`syslog.course`). Both angles live in the rotating
tangent-sphere frame, so the same correction
`χ̇ − φ̇·sin(β)` applies (φ = azimuth, β = elevation).

Accepts either a `SysLog` or its inner syslog table.
"""
function calc_turn_rate(syslog; source=:heading, dt=0.01)
    sl = hasproperty(syslog, :syslog) ? syslog.syslog :
        syslog
    if source === :heading
        ang = copy(sl.heading)
    elseif source === :course
        ang = copy(sl.course)
    else
        error("Unknown source: $source " *
            "(expected :heading or :course)")
    end
    _unwrap_inplace!(ang)
    rate = diff(ang) ./ dt
    az = copy(sl.azimuth)
    _unwrap_inplace!(az)
    az_dot = diff(az) ./ dt
    return rate .- az_dot .* sin.(sl.elevation[2:end])
end

function _unwrap_inplace!(a)
    for j in 2:length(a)
        while a[j] - a[j-1] > π
            a[j] -= 2π
        end
        while a[j] - a[j-1] < -π
            a[j] += 2π
        end
    end
    return a
end
