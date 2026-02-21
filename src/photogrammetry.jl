# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Photogrammetry data loading and comparison utilities.
Functions for loading camera-measured structural points and comparing with simulation.
"""

"""
    load_extra_points(csv_path::String, sys_struct; body_offset=[0.3, 0.0, 0.2])

Load extra points from CSV and transform from camera frame to simulation frame.
Returns (transformed_points, groups) where groups is Vector of (group_name, indices).

CSV has columns: group, idx_in_group, x, y, z.
Alignment: CSV strut3/strut4 LE centers align with sim points 10, 12.

# Arguments
- `csv_path`: Path to CSV file with photogrammetry points
- `sys_struct`: System structure for coordinate transformation
- `body_offset`: Offset in body frame [x, y, z] (default: [0.3, 0.0, 0.2])

# Returns
- `(transformed_points, groups)`: Tuple of transformed point tuples and group definitions
"""
function load_extra_points(csv_path::String, sys_struct; body_offset=[0.3, 0.0, 0.2])
    df = CSV.read(csv_path, DataFrame)

    # CSV strut centers: strut3[1]/strut4[1] are at TE, [end] are at LE
    strut3 = [[r.x, r.y, r.z] for r in eachrow(df) if r.group == "strut3"]
    strut4 = [[r.x, r.y, r.z] for r in eachrow(df) if r.group == "strut4"]
    csv_le_center = (strut3[end] + strut4[end]) / 2
    csv_te_center = (strut3[1] + strut4[1]) / 2

    # Sim reference: points 10, 12 (center LE)
    sim_p10 = collect(sys_struct.points[10].pos_w)
    sim_p12 = collect(sys_struct.points[12].pos_w)
    sim_le_center = (sim_p10 + sim_p12) / 2

    # Direction vectors
    csv_span = normalize(strut4[end] - strut3[end])

    # CSV basis: y=spanwise, z from wing center geometry, x from cross
    csv_y = csv_span
    csv_wing_center = (csv_le_center + csv_te_center) / 2
    csv_z = normalize(csv_wing_center - csv_y * 0.84/2)
    csv_x = cross(csv_y, csv_z)

    # Sim basis: directly from wing rotation matrix
    R_b_w = calc_R_b_w(sys_struct)
    sim_x = R_b_w[:, 1]
    sim_y = R_b_w[:, 2]
    sim_z = R_b_w[:, 3]

    # Rotation: R * csv_basis = sim_basis
    csv_basis = hcat(csv_x, csv_y, csv_z)
    sim_basis = hcat(sim_x, sim_y, sim_z)
    R = sim_basis * csv_basis'

    # Translation: align LE centers
    T = sim_le_center - R * csv_le_center + R_b_w * body_offset

    # Transform all points (including camera origin marker at zeros)
    all_pts = [[row.x, row.y, row.z] for row in eachrow(df)]
    push!(all_pts, zeros(3))
    transformed = [Tuple(R * p + T) for p in all_pts]

    # Build group indices (1-based)
    groups = Vector{Tuple{String, Vector{Int}}}()
    current_group = ""
    current_indices = Int[]
    for (i, row) in enumerate(eachrow(df))
        if row.group != current_group
            if !isempty(current_indices)
                push!(groups, (current_group, copy(current_indices)))
            end
            current_group = row.group
            current_indices = [i]
        else
            push!(current_indices, i)
        end
    end
    if !isempty(current_indices)
        push!(groups, (current_group, current_indices))
    end

    return transformed, groups
end

export load_extra_points
