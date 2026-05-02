# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Coordinate transformation and heading calculation utilities for the V3 kite.
"""

"""
    wrap_to_pi(angle)

Wrap angle to [-π, π] range.
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

"""
    calc_heading(R_b_w, pos_w)

Heading in the tangential sphere frame, matching
`wing.heading` from SymbolicAWEModels. Zero when the
nose (body x-axis) points toward the ground station
along the great circle.

# Arguments
- `R_b_w`: Rotation matrix from body to world frame
- `pos_w`: Kite position in world (ENU) frame
"""
function calc_heading(R_b_w, pos_w)
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
    return wrap_to_pi(calc_heading(R, pos_w) + π)
end

"""
    calc_R_b_w(sys_struct::SystemStructure)

Calculate the body-to-world rotation matrix for REFINE wing type.

# Arguments
- `sys_struct`: System structure with REFINE wing

# Returns
- 3x3 rotation matrix R_b_w
"""
function calc_R_b_w(sys_struct::SymbolicAWEModels.SystemStructure)
    @unpack points, wings = sys_struct
    wing = wings[1]
    R_b_w, origin = SymbolicAWEModels.calc_refine_wing_frame(
        points,
        wing.z_ref_points,
        wing.y_ref_points,
        wing.origin_idx
    )
    return R_b_w
end
