# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Flight data loading and processing utilities for V3 kite validation.
Loads HDF5 flight data files with ekf_output and flight_data groups.
"""

# Video frame mapping constants
# (video_frame 7182 = UTC 15:36:31.0)
const VIDEO_FRAME_REF = 7182
const UTC_REF_SECONDS = 15*3600 + 36*60 + 31.0
const VIDEO_FPS = 29.97

"""
    parse_time_to_seconds(time_str::String)

Parse "HH:MM:SS" or "HH:MM:SS.sss" to seconds since midnight.
"""
function parse_time_to_seconds(time_str::String)
    parts = split(time_str, ":")
    h = parse(Int, parts[1])
    m = parse(Int, parts[2])
    s = parse(Float64, parts[3])
    return h * 3600.0 + m * 60.0 + s
end

"""
    utc_to_video_frame(utc_seconds::Float64)

Convert UTC time (seconds since midnight) to video frame number.
"""
function utc_to_video_frame(utc_seconds::Float64)
    return round(Int,
        VIDEO_FRAME_REF +
        (utc_seconds - UTC_REF_SECONDS) * VIDEO_FPS)
end

"""
    unix_to_utc_seconds(unix_timestamp::Float64)

Convert Unix timestamp to UTC seconds since midnight.
"""
function unix_to_utc_seconds(unix_timestamp::Float64)
    dt = Dates.unix2datetime(unix_timestamp)
    return Dates.hour(dt)*3600 + Dates.minute(dt)*60 +
           Dates.second(dt) + Dates.millisecond(dt)/1000
end

"""
    load_flight_data(h5_path::String)

Load flight data from an HDF5 file containing `ekf_output`
and `flight_data` groups.

Datasets from `ekf_output` are prefixed with `ekf_` (e.g.,
`kite_roll` becomes `ekf_kite_roll`). Datasets from
`flight_data` keep their original names. Non-Float64 datasets
(strings, integers) are skipped.

Returns a NamedTuple with all datasets as Float64 vectors.
"""
function load_flight_data(h5_path::String)
    @info "Loading flight data from: $h5_path"
    data = Dict{Symbol, Vector{Float64}}()

    h5open(h5_path, "r") do fid
        # ekf_output datasets with "ekf_" prefix
        if haskey(fid, "ekf_output")
            for name in keys(fid["ekf_output"])
                ds = read(fid["ekf_output"][name])
                if eltype(ds) <: Real
                    data[Symbol("ekf_", name)] =
                        convert(Vector{Float64}, ds)
                end
            end
        end

        # flight_data datasets without prefix
        if haskey(fid, "flight_data")
            for name in keys(fid["flight_data"])
                ds = read(fid["flight_data"][name])
                if eltype(ds) <: Real
                    data[Symbol(name)] =
                        convert(Vector{Float64}, ds)
                end
            end
        end
    end

    col_names = Tuple(keys(data))
    return NamedTuple{col_names}(
        Tuple(data[k] for k in col_names))
end

"""
    find_indices_by_utc(data, start_utc, end_utc)

Find row indices corresponding to a UTC time range.

# Arguments
- `data`: NamedTuple with `unix_time` or `time` field
- `start_utc`: Start time as "HH:MM:SS" or "HH:MM:SS.sss"
- `end_utc`: End time as "HH:MM:SS" or "HH:MM:SS.sss"

# Returns
- `(start_idx, end_idx)` tuple of row indices
"""
function find_indices_by_utc(data,
        start_utc::String,
        end_utc::Union{String, Nothing}=nothing)
    start_sec = parse_time_to_seconds(start_utc)
    end_sec = isnothing(end_utc) ? Inf :
        parse_time_to_seconds(end_utc)

    start_idx = nothing
    end_idx = nothing

    time_col = hasproperty(data, :unix_time) ?
        data.unix_time : data.time

    for (i, unix_t) in enumerate(time_col)
        utc_sec = unix_to_utc_seconds(unix_t)
        if isnothing(start_idx) && utc_sec >= start_sec
            start_idx = i
        end
        if utc_sec <= end_sec
            end_idx = i
        end
    end

    if isnothing(start_idx) || isnothing(end_idx)
        error("Could not find UTC range " *
              "$start_utc to $end_utc in data")
    end

    return start_idx, end_idx
end

"""
    limit_by_utc(data, start_utc, end_utc)

Limit data to UTC time range. Normalizes time to start at 0
and adds a video_frame column.

# Arguments
- `data`: NamedTuple with time and unix_time fields
- `start_utc`: Start time as "HH:MM:SS" or "HH:MM:SS.sss"
- `end_utc`: End time as "HH:MM:SS" or "HH:MM:SS.sss"

# Returns
- `(data, start_idx)`: Sliced NamedTuple and starting index
"""
function limit_by_utc(data,
        start_utc::String,
        end_utc::Union{String, Nothing}=nothing)
    start_idx, end_idx = find_indices_by_utc(
        data, start_utc, end_utc)
    end_str = isnothing(end_utc) ? "end" : end_utc
    @info "UTC range $start_utc to $end_str" *
        " -> rows $start_idx to $end_idx"

    # Slice each vector
    sliced = NamedTuple{keys(data)}(
        Tuple(v[start_idx:end_idx] for v in data))

    # Video frames from unix timestamps
    video_frames = [
        Float64(utc_to_video_frame(
            unix_to_utc_seconds(t)))
        for t in sliced.unix_time]

    # Normalize time to start at 0
    t0 = sliced.time[1]
    normed_time = sliced.time .- t0

    sliced = merge(sliced, (
        time=normed_time, video_frame=video_frames))

    @info "Video frame range:" *
        " $(video_frames[1]) to $(video_frames[end])"

    return sliced, start_idx
end

"""
    add_distance_column(data)

Add distance and cumulative_distance columns to flight data.
Calculates 3D Euclidean distance between consecutive kite
positions using ekf_kite_position_x/y/z.
"""
function add_distance_column(data)
    n = length(data.time)
    distances = zeros(Float64, n)
    cumulative_distances = zeros(Float64, n)

    for i in 2:n
        dx = data.ekf_kite_position_x[i] -
            data.ekf_kite_position_x[i-1]
        dy = data.ekf_kite_position_y[i] -
            data.ekf_kite_position_y[i-1]
        dz = data.ekf_kite_position_z[i] -
            data.ekf_kite_position_z[i-1]
        distances[i] = sqrt(dx^2 + dy^2 + dz^2)
        cumulative_distances[i] =
            cumulative_distances[i-1] + distances[i]
    end

    return merge(data, (distance=distances,
        cumulative_distance=cumulative_distances))
end

"""
    find_closest_trajectory_index(data, query_pos;
        start_idx=1)

Find the data index whose (x, y, z) position is closest to
`query_pos`. Searches forward from `start_idx` to avoid
matching already-passed trajectory segments.
"""
function find_closest_trajectory_index(data, query_pos;
        start_idx=1)
    best_idx = start_idx
    best_dist2 = Inf
    for i in start_idx:length(data.time)
        dx = data.ekf_kite_position_x[i] - query_pos[1]
        dy = data.ekf_kite_position_y[i] - query_pos[2]
        dz = data.ekf_kite_position_z[i] - query_pos[3]
        d2 = dx^2 + dy^2 + dz^2
        if d2 < best_dist2
            best_dist2 = d2
            best_idx = i
        end
    end
    return best_idx
end

"""
    get_row_at_distance(data, target_dist)

Interpolate all fields of `data` at a given cumulative
distance. Uses `searchsortedlast` on `cumulative_distance`
to find the bracketing interval and linearly interpolates.

Returns a NamedTuple with the same fields as `data`.
"""
function get_row_at_distance(data, target_dist)
    cd = data.cumulative_distance
    idx = searchsortedlast(cd, target_dist)
    idx = clamp(idx, 1, length(cd) - 1)
    d0, d1 = cd[idx], cd[idx + 1]
    alpha = (d1 > d0) ?
        clamp((target_dist - d0) / (d1 - d0), 0.0, 1.0) :
        0.0
    ks = keys(data)
    vals = Tuple(
        (1 - alpha) * data[k][idx] +
        alpha * data[k][idx + 1]
        for k in ks)
    return NamedTuple{ks}(vals)
end

"""
    interpolate_flight_data(data, n_substeps)

Linearly interpolate flight data to create `n_substeps`
points between each pair of original data points.
"""
function interpolate_flight_data(data, n_substeps)
    n_original = length(data.time)
    n_interp = (n_original - 1) * n_substeps + 1

    interp_data = Dict{Symbol, Vector}()

    for field in keys(data)
        eltype_field = eltype(data[field])
        interp_values = Vector{eltype_field}(
            undef, n_interp)

        for i in 1:(n_original-1)
            start_idx = (i-1) * n_substeps + 1
            val_i = data[field][i]
            val_next = data[field][i+1]

            for j in 0:(n_substeps-1)
                idx = start_idx + j
                alpha = j / n_substeps

                if ismissing(val_i) || ismissing(val_next)
                    interp_values[idx] = missing
                else
                    interp_values[idx] =
                        (1 - alpha) * val_i +
                        alpha * val_next
                end
            end
        end

        interp_values[end] = data[field][end]
        interp_data[field] = interp_values
    end

    col_names = Tuple(keys(data))
    return NamedTuple{col_names}(
        Tuple(interp_data[k] for k in col_names))
end

"""
    update_sys_struct_from_data!(sys, row;
        extra_vel_body_x=0.0,
        config=V3GeomAdjustConfig())

Update system structure from a single data row.
Updates wing orientation from Euler angles and position
via transform system. Sets steering/depower from CSV data
using the provided geometry config.

# Arguments
- `sys`: System structure to update
- `row`: NamedTuple with roll, pitch, yaw, x, y, z, vx,
  vy, vz, steering, depower fields
- `extra_vel_body_x`: Extra velocity in body x direction
- `config`: Geometry adjustment config for tape reductions
"""
function update_sys_struct_from_data!(sys, row;
        extra_vel_body_x=0.0,
        config::V3GeomAdjustConfig=V3GeomAdjustConfig())
    @unpack wings, points, winches, segments, transforms = sys
    wing = wings[1]
    transform = transforms[1]

    sys.set.wind_vec = KiteUtils.MVec3(row.wind_vec)

    # calc target heading from data
    data_pos = [row.x, row.y, row.z]
    quat = euler_to_quaternion(
        row.roll, row.pitch, row.yaw)
    data_heading = wrap_to_pi(calc_heading(
        SymbolicAWEModels.quaternion_to_rotation_matrix(
            quat), data_pos) + π)
    R_b_w = calc_R_b_w(sys)

    # calc needed transform
    transform.elevation = KiteUtils.calc_elevation(data_pos)
    transform.azimuth = KiteUtils.azimuth_east(data_pos)
    transform.heading = data_heading
    SymbolicAWEModels.reinit!([transform], sys)

    # apply vel with optional extra velocity in body x
    data_vel = [row.vx, row.vy, row.vz]
    extra_vel_w = extra_vel_body_x * R_b_w[:, 1]
    wing.vel_w .= data_vel + extra_vel_w
    for point in points
        transform_frac =
            point.pos_w ⋅ normalize(wing.pos_w) /
            norm(wing.pos_w)
        point.vel_w .= transform_frac *
            (data_vel + extra_vel_w)
    end

    # update winch
    winches[1].brake = true

    # Set steering/depower from CSV data
    set_steering!(sys, row.steering, config;
        min_l0=0.01)
    set_depower!(sys, row.depower, 0.0, config)
end
