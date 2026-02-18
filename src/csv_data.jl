# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
CSV flight data loading and processing utilities for V3 kite validation.
"""

# Video frame mapping constants (video_frame 7182 = UTC 15:36:31.0)
const VIDEO_FRAME_REF = 7182
const UTC_REF_SECONDS = 15*3600 + 36*60 + 31.0  # UTC 15:36:31.0 in seconds since midnight
const VIDEO_FPS = 29.97

"""
    parse_time_to_seconds(time_str::String)

Parse "HH:MM:SS" or "HH:MM:SS.sss" to seconds since midnight.

# Arguments
- `time_str`: Time string in format "HH:MM:SS" or "HH:MM:SS.sss"

# Returns
- Seconds since midnight as Float64
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

# Arguments
- `utc_seconds`: UTC time as seconds since midnight

# Returns
- Video frame number as Int
"""
function utc_to_video_frame(utc_seconds::Float64)
    return round(Int, VIDEO_FRAME_REF + (utc_seconds - UTC_REF_SECONDS) * VIDEO_FPS)
end

"""
    unix_to_utc_seconds(unix_timestamp::Float64)

Convert Unix timestamp to UTC seconds since midnight.

# Arguments
- `unix_timestamp`: Unix timestamp (seconds since epoch)

# Returns
- Seconds since midnight as Float64
"""
function unix_to_utc_seconds(unix_timestamp::Float64)
    dt = Dates.unix2datetime(unix_timestamp)
    return Dates.hour(dt)*3600 + Dates.minute(dt)*60 +
           Dates.second(dt) + Dates.millisecond(dt)/1000
end

"""
    load_flight_data(csv_path::String)

Load and parse CSV flight data using space/multi-space delimiter.

# Arguments
- `csv_path`: Path to CSV file

# Returns
- DataFrame with all CSV columns
"""
function load_flight_data(csv_path::String)
    @info "Loading CSV data from: $csv_path"
    df = CSV.read(csv_path, DataFrame;
                  delim=' ',
                  silencewarnings=true,
                  normalizenames=true,
                  types=Float64,
                  strict=false)
    return df
end

"""
    find_csv_indices_by_utc(df, start_utc::String, end_utc::String)

Find CSV row indices corresponding to UTC time range.

# Arguments
- `df`: DataFrame with time data
- `start_utc`: Start time as "HH:MM:SS" or "HH:MM:SS.sss"
- `end_utc`: End time as "HH:MM:SS" or "HH:MM:SS.sss"

# Returns
- `(start_idx, end_idx)`: Tuple of start and end row indices
"""
function find_csv_indices_by_utc(df, start_utc::String, end_utc::String)
    start_sec = parse_time_to_seconds(start_utc)
    end_sec = parse_time_to_seconds(end_utc)

    start_idx = nothing
    end_idx = nothing

    # Use unix_time column (EKF CSV) or time column (old CSV)
    time_col = hasproperty(df, :unix_time) ? df.unix_time : df.time

    for (i, unix_t) in enumerate(time_col)
        if ismissing(unix_t)
            continue
        end
        utc_sec = unix_to_utc_seconds(unix_t)
        if isnothing(start_idx) && utc_sec >= start_sec
            start_idx = i
        end
        if utc_sec <= end_sec
            end_idx = i
        end
    end

    if isnothing(start_idx) || isnothing(end_idx)
        error("Could not find UTC range $start_utc to $end_utc in CSV")
    end

    return start_idx, end_idx
end

"""
    limit_by_utc(df, start_utc::String, end_utc::String)

Limit DataFrame to UTC time range and convert to named tuple.
Normalizes time column to start at 0. Adds video_frame column.

# Arguments
- `df`: DataFrame with time data
- `start_utc`: Start time as "HH:MM:SS" or "HH:MM:SS.sss"
- `end_utc`: End time as "HH:MM:SS" or "HH:MM:SS.sss"

# Returns
- `(data, start_idx)`: Named tuple with data and the starting row index
"""
function limit_by_utc(df, start_utc::String, end_utc::String)
    start_idx, end_idx = find_csv_indices_by_utc(df, start_utc, end_utc)
    @info "UTC range $start_utc to $end_utc -> rows $start_idx to $end_idx"

    # Slice the dataframe
    limited_df = df[start_idx:end_idx, :]

    # EKF CSV has both time (relative) and unix_time (absolute)
    # Use unix_time for video frame calculation, then normalize time column
    if hasproperty(limited_df, :unix_time)
        # Calculate video_frame from unix timestamps
        video_frames = [Float64(utc_to_video_frame(unix_to_utc_seconds(t)))
                        for t in limited_df.unix_time]
        # Normalize time column to start at 0
        t0 = limited_df.time[1]
        limited_df.time .= limited_df.time .- t0
    else
        # Old CSV: time column contains unix timestamps
        video_frames = [Float64(utc_to_video_frame(unix_to_utc_seconds(t)))
                        for t in limited_df.time]
        t0 = limited_df.time[1]
        limited_df.time .= limited_df.time .- t0
    end

    # Convert to named tuple and add video_frame
    col_names = Tuple(Symbol(name) for name in names(limited_df))
    data = NamedTuple{col_names}(Tuple(eachcol(limited_df)))
    data = merge(data, (video_frame=video_frames,))

    # Print video frame range
    @info "Video frame range: $(video_frames[1]) to $(video_frames[end])"

    return data, start_idx
end

"""
    add_distance_column(data)

Add distance and cumulative_distance columns to CSV data.
Calculates 3D Euclidean distance between consecutive kite positions.

# Arguments
- `data`: Named tuple with ekf_kite_position_x/y/z columns

# Returns
- Named tuple with added distance and cumulative_distance columns
"""
function add_distance_column(data)
    n = length(data.time)
    distances = zeros(Float64, n)
    cumulative_distances = zeros(Float64, n)

    # Use EKF position columns
    for i in 2:n
        dx = data.ekf_kite_position_x[i] - data.ekf_kite_position_x[i-1]
        dy = data.ekf_kite_position_y[i] - data.ekf_kite_position_y[i-1]
        dz = data.ekf_kite_position_z[i] - data.ekf_kite_position_z[i-1]
        distances[i] = sqrt(dx^2 + dy^2 + dz^2)
        cumulative_distances[i] = cumulative_distances[i-1] + distances[i]
    end

    # Add new fields to the named tuple
    return merge(data, (distance=distances, cumulative_distance=cumulative_distances))
end

"""
    interpolate_csv_data(data, n_substeps)

Linearly interpolate CSV data to create n_substeps points between each pair
of original data points. Handles missing values by propagating them.

# Arguments
- `data`: Named tuple with CSV data
- `n_substeps`: Number of substeps between each pair of original points

# Returns
- Named tuple with interpolated data
"""
function interpolate_csv_data(data, n_substeps)
    n_original = length(data.time)
    n_interp = (n_original - 1) * n_substeps + 1

    # Create interpolated arrays for each field
    interp_data = Dict{Symbol, Vector}()

    for field in keys(data)
        # Determine element type (handle Union{Float64, Missing})
        eltype_field = eltype(data[field])
        interp_values = Vector{eltype_field}(undef, n_interp)

        for i in 1:(n_original-1)
            # Starting index in interpolated array
            start_idx = (i-1) * n_substeps + 1

            val_i = data[field][i]
            val_next = data[field][i+1]

            # Interpolate between data[i] and data[i+1]
            for j in 0:(n_substeps-1)
                idx = start_idx + j
                alpha = j / n_substeps

                # Handle missing values
                if ismissing(val_i) || ismissing(val_next)
                    interp_values[idx] = missing
                else
                    interp_values[idx] = (1 - alpha) * val_i + alpha * val_next
                end
            end
        end

        # Add the last point
        interp_values[end] = data[field][end]
        interp_data[field] = interp_values
    end

    # Convert back to named tuple
    col_names = Tuple(keys(data))
    return NamedTuple{col_names}(Tuple(interp_data[k] for k in col_names))
end

"""
    update_sys_struct_from_csv!(sys, row; extra_vel_body_x=0.0)

Update system structure from a single CSV row.
Updates wing orientation from Euler angles and position via transform system.

# Arguments
- `sys`: System structure to update
- `row`: Named tuple with CSV row data (roll, pitch, yaw, x, y, z, vx, vy, vz, etc.)
- `extra_vel_body_x`: Extra velocity in body x direction (default: 0.0)
"""
function update_sys_struct_from_csv!(sys, row; extra_vel_body_x=0.0)
    @unpack wings, points, winches, segments, transforms = sys
    wing = wings[1]
    transform = transforms[1]

    # calc target heading from CSV
    quat = euler_to_quaternion(row.roll, row.pitch, row.yaw)
    csv_heading = calc_heading(sys,
        SymbolicAWEModels.quaternion_to_rotation_matrix(quat)) + pi
    wing.R_b_w = calc_R_b_w(sys)
    curr_heading = calc_heading(sys, wing.R_b_w)

    # calc needed transform
    csv_pos = [row.x, row.y, row.z]
    curr_pos = wing.pos_w
    delta_pos = csv_pos - curr_pos
    # apply transform
    transform.elevation = KiteUtils.calc_elevation(csv_pos)
    transform.azimuth = KiteUtils.azimuth_east(csv_pos)
    transform.heading = csv_heading
    SymbolicAWEModels.reinit!([transform], sys)

    # apply vel with optional extra velocity in body -z direction (upward)
    csv_vel = [row.vx, row.vy, row.vz]
    extra_vel_w = extra_vel_body_x * wing.R_b_w[:, 1]
    wing.vel_w .= csv_vel + extra_vel_w
    for point in points
        transform_frac = point.pos_w ⋅ normalize(wing.pos_w) / norm(wing.pos_w)
        point.vel_w .= transform_frac * (csv_vel + extra_vel_w)
    end

    # update winch
    winches[1].brake = true

    # Convert CSV percentages to tape lengths
    L_left, L_right = csv_steering_percentage_to_lengths(row.steering)
    L_depower = depower_percentage_to_length(row.depower)

    segments[V3_STEERING_LEFT_IDX].l0 = L_left
    segments[V3_STEERING_RIGHT_IDX].l0 = L_right
    segments[V3_DEPOWER_IDX].l0 = L_depower
end
