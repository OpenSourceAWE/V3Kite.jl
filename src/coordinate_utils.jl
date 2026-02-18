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
    euler_to_quaternion(roll_deg, pitch_deg, yaw_deg)

Convert Euler angles (in degrees) to quaternion.
Converts from NED to ENU frame:
  X_ENU = Y_NED (East)
  Y_ENU = X_NED (North)
  Z_ENU = -Z_NED (Up = -Down)

# Arguments
- `roll_deg`: Roll angle in degrees
- `pitch_deg`: Pitch angle in degrees
- `yaw_deg`: Yaw angle in degrees

# Returns
- Quaternion [w, x, y, z]
"""
function euler_to_quaternion(roll_deg, pitch_deg, yaw_deg)
    roll_rad = deg2rad(roll_deg)
    pitch_rad = deg2rad(pitch_deg)
    yaw_rad = deg2rad(yaw_deg)
    rot_ned = RotZYX(yaw_rad, pitch_rad, roll_rad)
    R_ned_to_enu = [0.0 1.0 0.0;
                    1.0 0.0 0.0;
                    0.0 0.0 -1.0]
    rot_enu = R_ned_to_enu * Matrix(rot_ned)
    q = SymbolicAWEModels.rotation_matrix_to_quaternion(rot_enu)
    return q
end

"""
    calc_heading(sys_struct::SystemStructure, R_b_w)

Calculate heading angle from rotation matrix, wrapped to [-π, π].

# Arguments
- `sys_struct`: System structure
- `R_b_w`: Rotation matrix from body to world frame

# Returns
- Heading angle in radians, wrapped to [-π, π]
"""
function calc_heading(sys_struct::SymbolicAWEModels.SystemStructure, R_b_w)
    e_x = R_b_w[:, 1]
    wind_norm = [1,0,0]
    minus_e_x = -e_x
    proj_on_wind = dot(minus_e_x, wind_norm) * wind_norm
    e_x_perp = minus_e_x - proj_on_wind
    wind_cross_z = [wind_norm[2], -wind_norm[1], 0]
    heading_x = dot(e_x_perp, wind_cross_z)
    heading_z = e_x_perp[3]
    heading = atan(heading_x, heading_z)
    return wrap_to_pi(heading)
end

"""
    calc_csv_heading(roll_deg, pitch_deg, yaw_deg, sys_struct)

Calculate heading from CSV Euler angles, wrapped to [-π, π].

# Arguments
- `roll_deg`: Roll angle from CSV in degrees
- `pitch_deg`: Pitch angle from CSV in degrees
- `yaw_deg`: Yaw angle from CSV in degrees
- `sys_struct`: System structure

# Returns
- Heading angle in radians, wrapped to [-π, π]
"""
function calc_csv_heading(roll_deg, pitch_deg, yaw_deg, sys_struct)
    quat = euler_to_quaternion(roll_deg, pitch_deg, yaw_deg)
    R = SymbolicAWEModels.quaternion_to_rotation_matrix(quat)
    heading = calc_heading(sys_struct, R)
    return wrap_to_pi(heading + π)
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
    @unpack points, wings, wind_vec_gnd = sys_struct
    wing = wings[1]
    R_b_w, origin = SymbolicAWEModels.calc_refine_wing_frame(
        points,
        wing.z_ref_points,
        wing.y_ref_points,
        wing.origin_idx
    )
    return R_b_w
end
