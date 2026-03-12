# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Analysis for Circular Flight Runs

Loads saved circular batch logs, computes steady-state
metrics (yaw rate, gk, Cs, turn radius), and writes a
summary CSV.

Usage:
    julia --project=examples examples/batch_load_circles.jl
"""

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX
using LinearAlgebra
using Statistics
using Dates
using StaticArrays

WINDOW_SEC = 100.0

# =============================================================================
# Tag parsing
# =============================================================================

function parse_up_us_vw_lt(log_name)
    m = match(
        r"_up_([0-9]+)_us_([0-9._-]+)" *
        r"_vw_([0-9]+)_lt_([0-9]+)", log_name)
    m === nothing && return nothing
    up_raw = parse(Float64, m.captures[1])
    us_raw = parse(Float64, split(m.captures[2], "_")[1])
    v_wind = parse(Int, m.captures[3])
    lt = parse(Int, m.captures[4])
    return up_raw / 100, us_raw / 100, v_wind, lt
end

function udp_tag_from_log_name(log_name::AbstractString, fallback_udp::Real)
    m = match(r"_udp_([0-9]{3})", log_name)
    if m !== nothing
        return String(m.captures[1])
    end
    return lpad(string(Int(round(fallback_udp * 100))), 3, '0')
end

function find_initial_state_geometry(; lt::Int, udp_tag::AbstractString,
    v_wind::Real, data_root::String=v3_data_path())
    udp_tag_s = String(udp_tag)
    pat = Regex("^struc_geometry_initial_state_lt_$(lt)_vw_([0-9]+)_udp_$(udp_tag_s)\\.yaml" * "\$")
    target_vw = Int(round(v_wind * 10))
    candidates = Tuple{String,Int}[]
    for name in readdir(data_root)
        m = match(pat, name)
        m === nothing && continue
        push!(candidates, (name, parse(Int, m.captures[1])))
    end
    isempty(candidates) && return nothing

    sort!(candidates; by=x -> (abs(x[2] - target_vw), x[2], x[1]))
    struc_name = first(candidates)[1]
    aero_name = replace(struc_name,
        "struc_geometry_" => "aero_geometry_")
    isfile(joinpath(data_root, aero_name)) || return nothing
    return (struc_yaml_path=struc_name,
        aero_yaml_path=aero_name)
end

function effective_v_wind_from_log(lg, fallback::Real)
    sl = hasproperty(lg, :syslog) ? lg.syslog : lg
    if hasproperty(sl, :v_wind_gnd)
        vw_col = getproperty(sl, :v_wind_gnd)
        if !isempty(vw_col)
            vw0 = vw_col[1]
            v = NaN
            if vw0 isa Number
                v = abs(float(vw0))
            elseif !isempty(vw0)
                v = abs(float(vw0[1]))
            end
            if isfinite(v) && v > 0
                return v
            end
        end
    end
    return float(fallback)
end

function load_log_compatible(log_name::AbstractString,
    batch_dir::AbstractString)
    arrow_path = joinpath(batch_dir, String(log_name) * ".arrow")
    isfile(arrow_path) || error("Log file not found: $arrow_path")
    filesize(arrow_path) > 0 || throw(ArgumentError(
        "Log file is empty: $arrow_path"))
    try
        return load_log(String(log_name); path=String(batch_dir))
    catch err
        if err isa KeyError && getfield(err, :key) in (:X, :Y, :Z, :time)
            @warn "Log schema differs from SysLog expectations; loading Arrow table directly" log_name arrow_path
            table = V3Kite.KiteUtils.Arrow.Table(arrow_path)
            haskey(table, :time) || throw(ArgumentError(
                "Unsupported Arrow schema (missing :time): $arrow_path"))
            return table
        end
        rethrow(err)
    end
end

# =============================================================================
# System construction
# =============================================================================

function build_sys(; v_wind=10.0, tether_length=150.0,
    up=0.0, log_name="")
    struc_yaml_path = "struc_geometry.yaml"
    aero_yaml_path = "aero_geometry.yaml"
    geom_adjust_cfg = V3GeomAdjustConfig(
        reduce_tip=true, reduce_te=true)

    if occursin("circles_from_initial_state", log_name)
        udp_tag = udp_tag_from_log_name(log_name, up)
        geom = find_initial_state_geometry(;
            lt=Int(round(tether_length)),
            udp_tag, v_wind)
        if geom !== nothing
            struc_yaml_path = geom.struc_yaml_path
            aero_yaml_path = geom.aero_yaml_path
            geom_adjust_cfg = nothing
            @info "Using initial-state geometry" log_name struc_yaml_path aero_yaml_path
        else
            @warn "Initial-state geometry not found; falling back to base geometry + model_setup adjustments" log_name
        end
    end

    config = V3SimConfig(
        struc_yaml_path=struc_yaml_path,
        aero_yaml_path=aero_yaml_path,
        vsm_settings_path="vsm_settings.yaml",
        v_wind=v_wind,
        tether_length=tether_length,
        wing_type=REFINE,
    )
    _, sys = create_v3_model(config)
    if geom_adjust_cfg !== nothing
        apply_geom_adjustments!(sys, geom_adjust_cfg)
    end
    return sys
end

# =============================================================================
# Math helpers
# =============================================================================

function unwrap_phase!(vals; period=2pi, thresh=pi)
    isempty(vals) && return vals
    offset = 0.0
    prev = vals[1]
    for i in 2:length(vals)
        delta = vals[i] - prev
        if delta > thresh
            offset -= period
        elseif delta < -thresh
            offset += period
        end
        prev = vals[i]
        vals[i] += offset
    end
    return vals
end

function gradient_uniform(y, ts)
    n = length(y)
    grad = Vector{Float64}(undef, n)
    n == 0 && return grad
    if n == 1
        grad[1] = 0.0
        return grad
    end
    grad[1] = (y[2] - y[1]) / ts
    for i in 2:(n-1)
        grad[i] = (y[i+1] - y[i-1]) / (2 * ts)
    end
    grad[n] = (y[n] - y[n-1]) / ts
    return grad
end

function moving_average_same(x, window)
    n = length(x)
    (window <= 1 || n == 0) && return Float64.(x)
    left = div(window, 2)
    right = window - 1 - left
    padded = zeros(Float64, n + left + right)
    padded[(left+1):(left+n)] .= x
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        s = 0.0
        for k in 0:(window-1)
            s += padded[i+k]
        end
        out[i] = s / window
    end
    return out
end

function midle_to_kcu_dir(sl, k; eps=1e-12)
    Xk, Yk, Zk = sl.X[k], sl.Y[k], sl.Z[k]
    (length(Xk) < 14 || length(Yk) < 14 ||
     length(Zk) < 14) && return nothing
    p1 = SVector{3}(Xk[1], Yk[1], Zk[1])
    ple12 = SVector{3}(Xk[12], Yk[12], Zk[12])
    ple14 = SVector{3}(Xk[14], Yk[14], Zk[14])
    p_le_mid = (ple12 + ple14) / 2
    dir = p1 - p_le_mid
    n = norm(dir)
    return n > eps ? dir / n : nothing
end

function calc_ref_area(sys)
    isempty(sys.wings) && return NaN
    wing = sys.wings[1]
    hasproperty(wing, :vsm_aero) || return NaN
    panels = wing.vsm_aero.panels
    isempty(panels) && return NaN
    return sum(p.chord * p.width for p in panels)
end

# =============================================================================
# Derived quantities
# =============================================================================

function calculate_cs(sl, sys; rho=1.225, eps=1e-12)
    (!hasproperty(sl, :X) || !hasproperty(sl, :Y) ||
     !hasproperty(sl, :Z)) && return Float64[], Float64[]
    s_ref = calc_ref_area(sys)
    (!isfinite(s_ref) || s_ref <= eps) &&
        return Float64[], Float64[]
    n = length(sl.time)
    cs = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        va = sl.vel_kite[k] - sl.v_wind_kite[k]
        va_norm = norm(va)
        if va_norm <= eps
            cs[k] = NaN
            continue
        end
        drag_dir = -va / va_norm
        up_dir = midle_to_kcu_dir(sl, k; eps)
        if up_dir === nothing
            cs[k] = NaN
            continue
        end
        up_dir = -up_dir
        side_raw = cross(drag_dir, up_dir)
        sn = norm(side_raw)
        if sn <= eps
            cs[k] = NaN
            continue
        end
        side_dir = side_raw / sn
        R = SymbolicAWEModels.quaternion_to_rotation_matrix(
            sl.orient[k])
        Fw = R * sl.aero_force_b[k]
        cs[k] = dot(Fw, side_dir) /
                (0.5 * rho * va_norm^2 * s_ref)
    end
    return cs, sl.time
end

function compute_turn_radius(sl_in, sys;
    smooth_window=10, eps=1e-12)
    sl = hasproperty(sl_in, :syslog) ?
         sl_in.syslog : sl_in
    n = length(sl.time)
    (n < 2 || isempty(sl.vel_kite) ||
     isempty(sl.orient)) && return nothing
    (length(sl.vel_kite) < n ||
     length(sl.orient) < n) && return nothing
    ts = mean(diff(sl.time))
    ts = isfinite(ts) && ts > eps ? ts : eps
    vx = [sl.vel_kite[k][1] for k in 1:n]
    vy = [sl.vel_kite[k][2] for k in 1:n]
    vz = [sl.vel_kite[k][3] for k in 1:n]
    ax = gradient_uniform(vx, ts)
    ay = gradient_uniform(vy, ts)
    az = gradient_uniform(vz, ts)
    if smooth_window > 1
        ax = moving_average_same(ax, smooth_window)
        ay = moving_average_same(ay, smooth_window)
        az = moving_average_same(az, smooth_window)
    end
    radius = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        v = SVector{3}(vx[k], vy[k], vz[k])
        a = SVector{3}(ax[k], ay[k], az[k])
        vn = norm(v)
        if !isfinite(vn) || vn <= eps
            radius[k] = NaN
            continue
        end
        vh = v / vn
        at = dot(a, vh) * vh
        omega = cross(a - at, v) / (vn^2)
        on = norm(omega)
        if !isfinite(on) || on <= eps
            radius[k] = NaN
            continue
        end
        icr = cross(v, omega) / (on^2)
        R = SymbolicAWEModels.quaternion_to_rotation_matrix(
            sl.orient[k])
        ex = SVector{3}(R[:, 1])
        det = ex[1] * icr[2] - ex[2] * icr[1]
        if !isfinite(det) || abs(det) <= eps
            radius[k] = NaN
        else
            radius[k] = -(det < 0 ? -1.0 : 1.0) *
                        norm(icr)
        end
    end
    return radius, sl.time
end

function compute_ekf_yaw_and_rate(sl_in, sys; eps=1e-12)
    sl = hasproperty(sl_in, :syslog) ?
         sl_in.syslog : sl_in
    n = length(sl.time)
    (n < 2 || isempty(sl.vel_kite)) && return nothing
    (!hasproperty(sl, :X) || !hasproperty(sl, :Y) ||
     !hasproperty(sl, :Z)) && return nothing
    (length(sys.wings) == 0 || length(sl.X) < n ||
     length(sl.Y) < n || length(sl.Z) < n) &&
        return nothing
    kite_idx = sys.wings[1].origin_idx
    yaw = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        pos = SVector{3}(sl.X[k][kite_idx],
            sl.Y[k][kite_idx], sl.Z[k][kite_idx])
        vel = SVector{3}(sl.vel_kite[k])
        npos, nvel = norm(pos), norm(vel)
        if npos > eps && nvel > eps
            rad = pos / npos
            tv = vel - dot(vel, rad) * rad
            ntv = norm(tv)
            if ntv > eps
                tvu = tv / ntv
                uz = rad
                uy_raw = SVector(-pos[2], pos[1], 0.0)
                nuy = norm(uy_raw)
                if nuy > eps
                    uy = uy_raw / nuy
                    ux = cross(uz, uy)
                    nux = norm(ux)
                    if nux > eps
                        ux = ux / nux
                        uy = cross(uz, ux)
                        R_up = @SMatrix [
                            ux[1] uy[1] uz[1];
                            ux[2] uy[2] uz[2];
                            ux[3] uy[3] uz[3]]
                        hv = R_up' * tvu
                        yaw[k] = atan(hv[2], hv[1])
                        continue
                    end
                end
            end
        end
        yaw[k] = k > 1 ? yaw[k-1] : NaN
    end
    yaw_uw = copy(yaw)
    unwrap_phase!(yaw_uw)
    ts = mean(diff(sl.time))
    ts = isfinite(ts) && ts > eps ? ts : eps
    yr = gradient_uniform(yaw_uw, ts)
    yr = moving_average_same(yr, 10)
    return yaw_uw, rad2deg.(yr)
end

function unwrap_heading(heading)
    hw = copy(heading)
    for j in 2:length(hw)
        while hw[j] - hw[j-1] > pi
            hw[j] -= 2pi
        end
        while hw[j] - hw[j-1] < -pi
            hw[j] += 2pi
        end
    end
    return hw
end

function heading_rate(sl)
    hw = unwrap_heading(sl.heading)
    rates = diff(rad2deg.(hw)) ./ diff(sl.time)
    return rates, sl.time[1:end-1]
end

function steering_command(sl, sys)
    n = length(sl.time)
    if hasproperty(sl, :X) && hasproperty(sl, :Y) &&
       hasproperty(sl, :Z)
        seg_left = sys.segments[V3_STEERING_LEFT_IDX]
        seg_right = sys.segments[V3_STEERING_RIGHT_IDX]
        li_l, lj_l = seg_left.point_idxs
        li_r, lj_r = seg_right.point_idxs
        us_cmd = zeros(Float64, n)
        @inbounds for k in 1:n
            p1l = SVector{3}(sl.X[k][li_l], sl.Y[k][li_l],
                sl.Z[k][li_l])
            p2l = SVector{3}(sl.X[k][lj_l], sl.Y[k][lj_l],
                sl.Z[k][lj_l])
            p1r = SVector{3}(sl.X[k][li_r], sl.Y[k][li_r],
                sl.Z[k][li_r])
            p2r = SVector{3}(sl.X[k][lj_r], sl.Y[k][lj_r],
                sl.Z[k][lj_r])
            us_cmd[k] = steering_length_to_percentage(
                norm(p2l - p1l), norm(p2r - p1r)) / 100.0
        end
        return us_cmd
    end

    # Older/reduced logs may not contain node positions; use logged steering directly.
    if hasproperty(sl, :steering)
        return Float64.(sl.steering)
    end
    if hasproperty(sl, :set_steering)
        return Float64.(sl.set_steering)
    end

    us_cmd = zeros(Float64, n)
    @warn "No steering information found in log; assuming zero steering"
    return us_cmd
end

function aoa_deg_series(sl)
    if hasproperty(sl, :AoA)
        return rad2deg.(Float64.(sl.AoA))
    end
    if hasproperty(sl, :alpha3)
        return rad2deg.(Float64.(sl.alpha3))
    end
    if hasproperty(sl, :alpha4)
        return rad2deg.(Float64.(sl.alpha4))
    end
    return fill(NaN, length(sl.time))
end

function gk_series(sl, sys)
    hr, _ = heading_rate(sl)
    us_cmd = steering_command(sl, sys)
    va = sl.v_app[2:end]
    us_seg = us_cmd[2:end]
    gk = similar(hr)
    @inbounds for k in eachindex(gk)
        gk[k] = abs(us_seg[k]) > 1e-8 ?
                hr[k] / (va[k] * us_seg[k]) : NaN
    end
    return gk, sl.time[2:end]
end

function gk_paper_series(sl, sys)
    n = length(sl.time)
    yaw = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        va_enu = sl.v_wind_kite[k] .- sl.vel_kite[k]
        va_ned = SVector{3}(va_enu[2], va_enu[1],
            -va_enu[3])
        yaw[k] = atan(va_ned[2], va_ned[1])
    end
    for k in 2:n
        dp = yaw[k] - yaw[k-1]
        if dp > pi
            yaw[k] -= 2pi
        elseif dp < -pi
            yaw[k] += 2pi
        end
    end
    yr = diff(rad2deg.(yaw)) ./ diff(sl.time)
    us_cmd = steering_command(sl, sys)
    us_seg = us_cmd[2:end]
    va = sl.v_app[2:end]
    gk = similar(yr)
    @inbounds for k in eachindex(gk)
        gk[k] = abs(us_seg[k]) > 1e-8 ?
                yr[k] / (va[k] * us_seg[k]) : NaN
    end
    return gk, sl.time[2:end]
end

function mean_last_window(values, times;
    window_sec=WINDOW_SEC)
    @assert length(values) == length(times)
    t_end = times[end]
    mask = times .>= (t_end - window_sec)
    any(mask) || (mask = trues(length(times)))
    data = values[mask]
    data = data[isfinite.(data)]
    return isempty(data) ? NaN : mean(data)
end

function mean_at_time(values, times, target_time;
    window_half=0.5)
    @assert length(values) == length(times)
    mask = (times .>= (target_time - window_half)) .&
           (times .<= (target_time + window_half))
    any(mask) || return NaN
    data = values[mask]
    data = data[isfinite.(data)]
    return isempty(data) ? NaN : mean(data)
end

# =============================================================================
# Log analysis
# =============================================================================

function analyze_log(lg, sys; window_sec=WINDOW_SEC)
    sl = hasproperty(lg, :syslog) ? lg.syslog : lg
    length(sl.time) < 2 && return (
        aero_force=NaN, v_app=NaN,
        yaw_rate=NaN, yaw_rate_paper=NaN,
        gk=NaN, gk_paper=NaN, kite_vel=NaN,
        aoa=NaN, elevation=NaN, azimuth=NaN,
        cs=NaN, turn_radius=NaN,
        usva_at=Dict{Int,Float64}(),
        yaw_rate_at=Dict{Int,Float64}(),
        va_at=Dict{Int,Float64}(),
        aoa_at=Dict{Int,Float64}())

    az = [sl.aero_force_b[i][3]
          for i in eachindex(sl.aero_force_b)]
    aero_force = mean_last_window(az, sl.time;
        window_sec)
    v_app = mean_last_window(sl.v_app, sl.time;
        window_sec)
    yr_deg, yr_time = heading_rate(sl)
    yaw_rate = mean_last_window(yr_deg, yr_time;
        window_sec)
    ekf = compute_ekf_yaw_and_rate(lg, sys)
    yaw_rate_paper = if ekf === nothing
        yaw_rate
    else
        mean_last_window(ekf[2], sl.time; window_sec)
    end
    gk_v, gk_t = gk_series(sl, sys)
    gk = mean_last_window(gk_v, gk_t; window_sec)
    gkp_v, gkp_t = gk_paper_series(sl, sys)
    gk_paper = mean_last_window(gkp_v, gkp_t;
        window_sec)
    vk = [norm(v) for v in sl.vel_kite]
    kite_vel = mean_last_window(vk, sl.time; window_sec)
    aoa_deg = aoa_deg_series(sl)
    aoa = mean_last_window(aoa_deg, sl.time; window_sec)
    elev_deg = rad2deg.(sl.elevation)
    elevation = mean_last_window(elev_deg, sl.time;
        window_sec)
    az_deg = rad2deg.(sl.azimuth)
    azimuth = mean_last_window(az_deg, sl.time;
        window_sec)
    cs_v, cs_t = calculate_cs(sl, sys)
    cs = if isempty(cs_v) || isempty(cs_t)
        NaN
    else
        abs(mean_last_window(cs_v, cs_t; window_sec))
    end
    tr_res = compute_turn_radius(sl, sys)
    turn_radius = tr_res === nothing ? NaN :
                  mean_last_window(abs.(tr_res[1]), tr_res[2];
        window_sec)

    us_cmd = steering_command(sl, sys)
    usva = us_cmd .* sl.v_app
    usva_at = Dict{Int,Float64}()
    yaw_rate_at = Dict{Int,Float64}()
    va_at = Dict{Int,Float64}()
    aoa_at = Dict{Int,Float64}()
    for t_sec in 3:10
        usva_at[t_sec] = mean_at_time(
            usva, sl.time, Float64(t_sec))
        yaw_rate_at[t_sec] = mean_at_time(
            yr_deg, yr_time, Float64(t_sec))
        va_at[t_sec] = mean_at_time(
            sl.v_app, sl.time, Float64(t_sec))
        aoa_at[t_sec] = mean_at_time(
            aoa_deg, sl.time, Float64(t_sec))
    end

    return (aero_force=aero_force, v_app=v_app,
        yaw_rate=yaw_rate, yaw_rate_paper=yaw_rate_paper,
        gk=gk, gk_paper=gk_paper, kite_vel=kite_vel,
        aoa=aoa, elevation=elevation, azimuth=azimuth,
        cs=cs, turn_radius=turn_radius,
        usva_at=usva_at, yaw_rate_at=yaw_rate_at,
        va_at=va_at, aoa_at=aoa_at)
end

# =============================================================================
# Batch loading
# =============================================================================

function find_log_names(batch_dir)
    isdir(batch_dir) || error("Not found: $batch_dir")
    names = String[]
    for file in readdir(batch_dir; join=true)
        isfile(file) || continue
        endswith(file, ".txt") && continue
        if endswith(file, ".arrow") && filesize(file) == 0
            @warn "Skipping empty log file" file
            continue
        end
        name = splitext(basename(file))[1]
        parse_up_us_vw_lt(name) === nothing && continue
        push!(names, name)
    end
    return sort(unique(names))
end

function resolve_batch_dir(batch_name)
    batch_dir = joinpath("processed_data", batch_name)
    isdir(batch_dir) && return batch_dir
    legacy_dir = joinpath("processed_data", "v3_kite", batch_name)
    if isdir(legacy_dir)
        @warn "Using legacy batch path" batch_dir = legacy_dir
        return legacy_dir
    end
    return batch_dir
end

function write_csv(path, rows)
    base = "vw,up,us,lt,aero_force,v_app," *
           "yaw_rate,yaw_rate_paper,gk,gk_paper," *
           "kite_vel,aoa,elevation,azimuth,cs,turn_radius"
    tc = String[]
    for t in 4:9
        push!(tc, "usva_$t")
        push!(tc, "yaw_rate_$t")
        push!(tc, "va$t")
        push!(tc, "aoa$t")
    end
    header = base * "," * join(tc, ",")
    open(path, "w") do io
        println(io, header)
        for r in rows
            bv = [r.vw, r.up, r.us, r.lt,
                r.aero_force, r.v_app,
                r.yaw_rate, r.yaw_rate_paper,
                r.gk, r.gk_paper, r.kite_vel,
                r.aoa, r.elevation, r.azimuth,
                r.cs, r.turn_radius]
            tv = Float64[]
            for t in 3:10
                push!(tv, r.usva_at[t])
                push!(tv, r.yaw_rate_at[t])
                push!(tv, r.va_at[t])
                push!(tv, r.aoa_at[t])
            end
            println(io, join(vcat(bv, tv), ","))
        end
    end
end

function main(; batch_name::AbstractString="")
    batch_name = strip(String(batch_name))
    if isempty(batch_name)
        batch_name = isempty(ARGS) ? "" : strip(ARGS[1])
    end
    # # batch_name = "circular_2025_batch_2026_01_11_11_29_19"
    # batch_name = "zenith_2025_batch_2026_03_03_11_26_48"
    # batch_name = "circles_from_initial_state_2019_2026_03_04_22_20_39"
    # batch_name = "circles_from_initial_state_2025_2026_03_05_08_51_36"
    # batch_name = "circles_from_initial_state_2019_0352026_03_05_15_39_37"
    if isempty(batch_name)
        print("Enter batch folder name: ")
        batch_name = strip(readline())
    end
    isempty(batch_name) && error("Batch name required.")

    batch_dir = resolve_batch_dir(batch_name)
    log_names = find_log_names(batch_dir)
    isempty(log_names) && error("No logs in: $batch_dir")

    rows = NamedTuple[]
    sys_cache = Dict{Tuple{Int,Int,Int},
        SymbolicAWEModels.SystemStructure}()

    for log_name in log_names
        tags = parse_up_us_vw_lt(log_name)
        tags === nothing && continue
        up, us, vw, lt = tags
        lg = try
            load_log_compatible(log_name, batch_dir)
        catch err
            @warn "Skipping unreadable log" log_name error = err
            continue
        end
        v_wind_ref = effective_v_wind_from_log(
            lg, Float64(vw))
        up_tag = Int(round(up * 100))
        vw_ref_tag = Int(round(v_wind_ref * 10))
        sys = get!(sys_cache, (vw_ref_tag, lt, up_tag)) do
            build_sys(v_wind=v_wind_ref,
                tether_length=Float64(lt),
                up=up, log_name=log_name)
        end
        m = analyze_log(lg, sys)
        push!(rows, (
            vw=vw, up=up, us=us, lt=lt,
            aero_force=m.aero_force, v_app=m.v_app,
            yaw_rate=m.yaw_rate,
            yaw_rate_paper=m.yaw_rate_paper,
            gk=m.gk, gk_paper=m.gk_paper,
            kite_vel=m.kite_vel, aoa=m.aoa,
            elevation=m.elevation, azimuth=m.azimuth,
            cs=m.cs, turn_radius=m.turn_radius,
            usva_at=m.usva_at,
            yaw_rate_at=m.yaw_rate_at,
            va_at=m.va_at,
            aoa_at=m.aoa_at))
    end

    isempty(rows) && error("No readable logs in: $batch_dir")
    sort!(rows, by=r -> (r.vw, r.up, r.us, r.lt))

    out_path = joinpath(batch_dir,
        "circles_batch_analysis.csv")
    write_csv(out_path, rows)
    @info "Wrote CSV" path = out_path rows = length(rows)
end

main(
    batch_name="_vw_8_lt_270_udp_sweep_2026_03_09"
)
# main(
#     batch_name="vw8_lt_270_circles_udp_032__2026_03_06_14_27_33"
# )
# main(
#     batch_name="vw8_lt_270_circles_udp_042__2026_03_06_14_58_08"
# )

nothing
