# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Analysis for Circular Flight Runs

Loads saved circular batch logs, computes steady-state
metrics (yaw rate, gk, Cs, turn radius), and writes a
summary CSV.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_GAIN
using LinearAlgebra
using Statistics
using Dates
using StaticArrays
using REPL.TerminalMenus
using GLMakie

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = dirname(@__DIR__)

BATCH_DATA_PATH = joinpath(PROJECT_DIR, "processed_data",
    "circles_batch_2026_05_30_22_18_57")
DEFAULT_STRUC_YAML_PATH = joinpath(PROJECT_DIR, "data/python_yamls/struc_geometry_julia_generated.yaml")
DEFAULT_AERO_YAML_PATH = joinpath(PROJECT_DIR, "data/aero_geometry.yaml")
DEFAULT_VSM_SETTINGS_PATH = joinpath(PROJECT_DIR, "data/vsm_settings.yaml")

# =============================================================================
# Tag parsing
# =============================================================================

function parse_udp_us_vw_lt(log_name)
    m = match(
        r"_(?:udp|up)_([0-9]+)_us_([0-9._-]+)" *
        r"_vw_([0-9]+)_lt_([0-9]+)", log_name)
    m === nothing && return nothing
    any(isnothing, m.captures) && return nothing
    udp_raw = parse(Float64, something(m.captures[1]))
    us_raw = parse(Float64, split(something(m.captures[2]), "_")[1])
    v_wind = parse(Int, something(m.captures[3]))
    lt = parse(Int, something(m.captures[4]))
    return udp_raw / 100, us_raw / 100, v_wind, lt
end

# =============================================================================
# System construction
# =============================================================================

function build_sys(; v_wind=10.0, tether_length=150.0)
    config = V3SimConfig(
        struc_yaml_path=DEFAULT_STRUC_YAML_PATH,
        aero_yaml_path=DEFAULT_AERO_YAML_PATH,
        vsm_settings_path=DEFAULT_VSM_SETTINGS_PATH,
        v_wind=v_wind,
        tether_length=tether_length,
        wing_type=REFINE,
    )
    _, sys = create_v3_model(config)
    apply_geom_adjustments!(sys, V3GeomAdjustConfig(
        tether_length=tether_length))
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
    return wing.vsm_aero.projected_area
end

# =============================================================================
# Derived quantities
# =============================================================================

function calculate_cs(sl, sys; rho=1.225, eps=1e-12)
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

function compute_turn_radius(sl_in, _sys;
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
        r = norm(icr)
        radius[k] = isfinite(r) ? r : NaN
    end
    return radius, sl.time
end

function compute_ekf_yaw_and_rate(sl_in, sys; eps=1e-12)
    sl = hasproperty(sl_in, :syslog) ?
         sl_in.syslog : sl_in
    n = length(sl.time)
    (n < 2 || isempty(sl.vel_kite)) && return nothing
    (length(sys.wings) == 0 || length(sl.X) < n ||
     length(sl.Y) < n || length(sl.Z) < n) &&
        return nothing
    kite_idx = sys.wings[1].idx
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

function steering_command(sl, sys; steering_l0=nothing)
    seg = sys.segments[V3_STEERING_LEFT_IDX]
    pi_, pj = seg.point_idxs
    n = length(sl.time)
    slen = zeros(Float64, n)
    @inbounds for k in 1:n
        p1 = SVector{3}(sl.X[k][pi_], sl.Y[k][pi_],
            sl.Z[k][pi_])
        p2 = SVector{3}(sl.X[k][pj], sl.Y[k][pj],
            sl.Z[k][pj])
        slen[k] = norm(p2 - p1)
    end
    base_l0 = isnothing(steering_l0) ? slen[1] : steering_l0
    us_cmd = similar(slen)
    @inbounds for k in eachindex(us_cmd)
        d = slen[k] - base_l0
        us_cmd[k] = abs(d) > 1e-6 ?
                    d / V3_STEERING_GAIN : 0.0
    end
    return us_cmd
end

function gk_series(sl, sys)
    hr, _ = heading_rate(sl)
    us_cmd = steering_command(sl, sys; steering_l0=1.6)
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
    window_sec=COURSE_RATE_WINDOW_SEC)
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

function analyze_log(lg, sys; window_sec=COURSE_RATE_WINDOW_SEC)
    sl = lg.syslog
    length(sl.time) < 2 && return (
        aero_force=NaN, v_app=NaN,
        yaw_rate=NaN, yaw_rate_paper=NaN,
        gk=NaN, gk_paper=NaN, kite_vel=NaN,
        aoa=NaN, elevation=NaN, azimuth=NaN,
        cs=NaN, turn_radius=NaN,
        usva_at=Dict{Int,Float64}(),
        yaw_rate_at=Dict{Int,Float64}())

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
    aoa_deg = rad2deg.(sl.AoA)
    aoa = mean_last_window(aoa_deg, sl.time; window_sec)
    elev_deg = rad2deg.(sl.elevation)
    elevation = mean_last_window(elev_deg, sl.time;
        window_sec)
    az_deg = rad2deg.(sl.azimuth)
    azimuth = mean_last_window(az_deg, sl.time;
        window_sec)
    cs_v, cs_t = calculate_cs(sl, sys)
    cs = abs(mean_last_window(cs_v, cs_t; window_sec))
    tr_res = compute_turn_radius(sl, sys)
    turn_radius = tr_res === nothing ? NaN :
                  mean_last_window(tr_res[1], tr_res[2];
        window_sec)

    us_cmd = steering_command(sl, sys)
    usva = us_cmd .* sl.v_app
    usva_mean = mean_last_window(
        abs.(usva), sl.time; window_sec)
    dt_sample = length(sl.time) > 1 ?
                mean(diff(sl.time)) : 0.01
    course_rate = calc_turn_rate(lg;
        source=:course, dt=dt_sample)
    course_rate_mean = mean_last_window(
        abs.(course_rate), sl.time[2:end]; window_sec)

    usva_at = Dict{Int,Float64}()
    yaw_rate_at = Dict{Int,Float64}()
    for t_sec in 3:10
        usva_at[t_sec] = mean_at_time(
            usva, sl.time, Float64(t_sec))
        yaw_rate_at[t_sec] = mean_at_time(
            yr_deg, yr_time, Float64(t_sec))
    end

    return (aero_force=aero_force, v_app=v_app,
        yaw_rate=yaw_rate, yaw_rate_paper=yaw_rate_paper,
        gk=gk, gk_paper=gk_paper, kite_vel=kite_vel,
        aoa=aoa, elevation=elevation, azimuth=azimuth,
        cs=cs, turn_radius=turn_radius,
        usva=usva_mean, course_rate=course_rate_mean,
        usva_at=usva_at, yaw_rate_at=yaw_rate_at)
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
        name = splitext(basename(file))[1]
        parse_udp_us_vw_lt(name) === nothing && continue
        push!(names, name)
    end
    return sort(unique(names))
end

function resolve_batch_dir(batch_name)
    isdir(batch_name) && return batch_name
    project_path = joinpath(PROJECT_DIR, batch_name)
    isdir(project_path) && return project_path

    batch_dir = joinpath(PROJECT_DIR, "processed_data", batch_name)
    isdir(batch_dir) && return batch_dir
    legacy_dir = joinpath(PROJECT_DIR, "processed_data",
        "v3_kite", batch_name)
    if isdir(legacy_dir)
        @warn "Using legacy batch path" batch_dir = legacy_dir
        return legacy_dir
    end
    return batch_dir
end

function write_csv(path, rows)
    base = "vw,udp,us,lt,aero_force,v_app," *
           "yaw_rate,yaw_rate_paper,gk,gk_paper," *
           "kite_vel,aoa,elevation,azimuth,cs,turn_radius"
    tc = String[]
    for t in 3:10
        push!(tc, "usva_$t")
        push!(tc, "yaw_rate_$t")
    end
    header = base * "," * join(tc, ",")
    open(path, "w") do io
        println(io, header)
        for r in rows
            bv = [r.vw, r.udp, r.us, r.lt,
                r.aero_force, r.v_app,
                r.yaw_rate, r.yaw_rate_paper,
                r.gk, r.gk_paper, r.kite_vel,
                r.aoa, r.elevation, r.azimuth,
                r.cs, r.turn_radius]
            tv = Float64[]
            for t in 3:10
                push!(tv, r.usva_at[t])
                push!(tv, r.yaw_rate_at[t])
            end
            println(io, join(vcat(bv, tv), ","))
        end
    end
end

function last_timestamp_token(name::AbstractString)
    token = ""
    for m in eachmatch(
        r"[0-9]{4}(?:_[0-9]{2}){5}", name)
        token = m.match
    end
    return token
end

function classify_swept(rows; params=(:udp, :us, :vw, :lt))
    defaults = Dict{Symbol,Any}()
    for p in params
        counts = Dict{Any,Int}()
        for r in rows
            v = getproperty(r, p)
            counts[v] = get(counts, v, 0) + 1
        end
        defaults[p] = argmax(counts)
    end
    labels = Symbol[]
    for r in rows
        diffs = Symbol[]
        for p in params
            getproperty(r, p) == defaults[p] || push!(diffs, p)
        end
        if isempty(diffs)
            push!(labels, :defaults)
        elseif length(diffs) == 1
            push!(labels, diffs[1])
        else
            push!(labels, :combo)
        end
    end
    return labels, defaults
end

function plot_usva_vs_course_rate(rows;
    window_sec=COURSE_RATE_WINDOW_SEC)
    finite_rows = [r for r in rows
                   if isfinite(r.usva) &&
                   isfinite(r.course_rate)]

    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1];
        xlabel=L"|u_{\text{s}} \cdot v_{\text{a}}| \; [m/s]",
        ylabel=L"|\dot{\chi}| \; [rad/s]",
        xlabelsize=18, ylabelsize=18,
        title="last $(window_sec)s mean, one dot per run")

    isempty(finite_rows) && return fig

    labels, defaults = classify_swept(finite_rows)
    palette = Makie.wong_colors()
    group_order = Symbol[]
    for g in labels
        g in group_order || push!(group_order, g)
    end
    sort!(group_order;
        by=g -> (g === :defaults ? 0 :
                 g === :combo ? 99 : 1, string(g)))
    group_color = Dict(g => palette[mod1(i, length(palette))]
                       for (i, g) in enumerate(group_order))

    fmt_v(v) = v isa AbstractFloat ?
               string(round(v; digits=3)) : string(v)
    sweep_params = (:udp, :us, :vw, :lt)

    all_x = Float64[]
    all_y = Float64[]
    for g in group_order
        idxs = findall(==(g), labels)
        xs = Float64[finite_rows[i].usva for i in idxs]
        ys = Float64[finite_rows[i].course_rate for i in idxs]
        lbl = if g === :defaults
            "defaults"
        elseif g === :combo
            "combo"
        else
            "sweep $(g)"
        end
        scatter!(ax, xs, ys; markersize=12,
            color=group_color[g],
            label=lbl)
        append!(all_x, xs)
        append!(all_y, ys)

        txts = String[]
        for i in idxs
            r = finite_rows[i]
            if g === :defaults
                push!(txts, "default")
            elseif g === :combo
                diffs = String[]
                for p in sweep_params
                    getproperty(r, p) == defaults[p] && continue
                    push!(diffs,
                        "$(p)=$(fmt_v(getproperty(r, p)))")
                end
                push!(txts, join(diffs, ","))
            else
                push!(txts, fmt_v(getproperty(r, g)))
            end
        end
        text!(ax, xs, ys; text=txts, fontsize=9,
            offset=(6, 6))
    end

    default_idx = findfirst(==(:defaults), labels)
    if default_idx !== nothing && !isempty(all_x)
        dx = finite_rows[default_idx].usva
        dy = finite_rows[default_idx].course_rate
        if dx > 0
            gk = dy / dx
            x_fit = range(0, maximum(all_x); length=50)
            lines!(ax, collect(x_fit), gk .* collect(x_fit);
                color=:black, linewidth=2,
                label="gk=$(round(gk; digits=3)) (defaults)")
        end
    end
    axislegend(ax; position=:lt)
    return fig
end

function select_batch_interactively(root)
    isdir(root) || error(
        "$root does not exist. " *
        "Run examples/batch_run_circles.jl first " *
        "to generate logs.")
    dirs = filter(name -> isdir(joinpath(root, name)),
        readdir(root))
    isempty(dirs) && error(
        "No batch directories found in $root. " *
        "Run examples/batch_run_circles.jl first " *
        "to generate logs.")
    dirs_sorted = sort(dirs;
        by=name -> (last_timestamp_token(name), name),
        rev=true)
    menu = RadioMenu(dirs_sorted, pagesize=15)
    choice = request(
        "Select batch directory (newest first):", menu)
    choice == -1 && error("Selection cancelled")
    return dirs_sorted[choice]
end

function main()
    batch_name = isempty(ARGS) ? "" : strip(ARGS[1])
    batch_dir = if !isempty(batch_name)
        resolve_batch_dir(batch_name)
    elseif !isempty(BATCH_DATA_PATH)
        BATCH_DATA_PATH
    else
        selected_batch = select_batch_interactively(
            joinpath(PROJECT_DIR, "processed_data"))
        resolve_batch_dir(selected_batch)
    end
    log_names = find_log_names(batch_dir)
    isempty(log_names) && error("No logs in: $batch_dir")

    rows = NamedTuple[]
    sys_cache = Dict{Tuple{Int,Int},
        SymbolicAWEModels.SystemStructure}()

    for log_name in log_names
        tags = parse_udp_us_vw_lt(log_name)
        tags === nothing && continue
        udp, us, vw, lt = tags
        sys = get!(sys_cache, (vw, lt)) do
            build_sys(v_wind=Float64(vw),
                tether_length=Float64(lt))
        end
        lg = load_log(log_name; path=batch_dir)
        m = analyze_log(lg, sys)
        push!(rows, (
            vw=vw, udp=udp, us=us, lt=lt,
            aero_force=m.aero_force, v_app=m.v_app,
            yaw_rate=m.yaw_rate,
            yaw_rate_paper=m.yaw_rate_paper,
            gk=m.gk, gk_paper=m.gk_paper,
            kite_vel=m.kite_vel, aoa=m.aoa,
            elevation=m.elevation, azimuth=m.azimuth,
            cs=m.cs, turn_radius=m.turn_radius,
            usva=m.usva, course_rate=m.course_rate,
            usva_at=m.usva_at,
            yaw_rate_at=m.yaw_rate_at))
    end

    sort!(rows, by=r -> (r.vw, r.udp, r.us, r.lt))

    out_path = joinpath(batch_dir,
        "circles_batch_analysis.csv")
    write_csv(out_path, rows)
    @info "Wrote CSV" path = out_path rows = length(rows)

    fig = plot_usva_vs_course_rate(rows)
    plot_path = joinpath(batch_dir,
        "circles_batch_usva_vs_course_rate.png")
    @info "Saving plot" path = plot_path
    save(plot_path, fig)
    display(fig)
end

main()
nothing
