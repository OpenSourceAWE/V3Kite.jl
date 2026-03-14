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

    # CSV strut centers: find matching LE/TE pairs by y-coordinate
    strut3 = [[r.x, r.y, r.z] for r in eachrow(df) if r.group == "strut3"]
    strut4 = [[r.x, r.y, r.z] for r in eachrow(df) if r.group == "strut4"]

    # LE pair: highest matching index (iterate downward)
    le_idx = nothing
    for i in min(length(strut3), length(strut4)):-1:1
        if abs(strut3[i][2] - strut4[i][2]) < 0.3
            le_idx = i
            break
        end
    end
    isnothing(le_idx) &&
        error("No matching LE pair in strut3/strut4")
    if le_idx != length(strut3) || le_idx != length(strut4)
        @warn("LE match at index $le_idx, not at array end " *
              "(strut3=$(length(strut3)), strut4=$(length(strut4)))")
    end
    strut3_le = strut3[le_idx]
    strut4_le = strut4[le_idx]

    # TE pair: lowest matching index (iterate upward)
    te_idx = nothing
    for i in 1:min(length(strut3), length(strut4))
        if abs(strut3[i][2] - strut4[i][2]) < 0.3
            te_idx = i
            break
        end
    end
    isnothing(te_idx) &&
        error("No matching TE pair in strut3/strut4")
    if te_idx != 1
        @warn("TE match at index $te_idx, not at index 1")
    end

    csv_le_center = (strut3_le + strut4_le) / 2
    csv_te_center = (strut3[te_idx] + strut4[te_idx]) / 2

    # Sim reference: points 10, 12 (center LE)
    sim_p10 = collect(sys_struct.points[10].pos_w)
    sim_p12 = collect(sys_struct.points[12].pos_w)
    sim_le_center = (sim_p10 + sim_p12) / 2

    # Direction vectors
    csv_span = normalize(strut4_le - strut3_le)

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
