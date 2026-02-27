# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Analysis for Zenith-Circle Runs

Loads saved zenith-circle batch logs, computes steady-state
metrics, and writes a summary CSV.

Usage:
    julia --project=examples examples/batch_load_zenith.jl
"""

using V3Kite
using LinearAlgebra
using Statistics
using Dates
using StaticArrays

include(joinpath(@__DIR__, "_shared.jl"))

WINDOW_SEC = 200.0

# =============================================================================
# Tag parsing
# =============================================================================

function parse_tags(log_name)
    m = match(
        r"(?:hold_at_zenith_then_circles|zenith_circle)__up_([0-9]+)" *
        r"_us_([0-9._-]+)_vw_([0-9]+)" *
        r"_lt_([0-9]+)_el_([0-9]+|yaml)" *
        r"_g_([0-9]+|yaml)",
        log_name)
    m === nothing && return nothing
    up_raw = parse(Float64, m.captures[1])
    us_raw = parse(Float64, split(m.captures[2], "_")[1])
    v_wind = parse(Int, m.captures[3])
    lt = parse(Int, m.captures[4])
    set_elev = m.captures[5] == "yaml" ?
               nothing : parse(Float64, m.captures[5])
    g_earth = m.captures[6] == "yaml" ?
              nothing : parse(Float64, m.captures[6]) / 10
    return up_raw / 100, us_raw / 100, v_wind, lt,
    set_elev, g_earth
end

# =============================================================================
# System construction
# =============================================================================

function build_sys(; v_wind=10.0, tether_length=150.0,
    g_earth=nothing, te_edge_scale=0.95)
    config = V3SimConfig(
        struc_yaml_path="struc_geometry.yaml",
        aero_yaml_path="aero_geometry.yaml",
        vsm_settings_path="vsm_settings.yaml",
        v_wind=v_wind,
        tether_length=tether_length,
        wing_type=REFINE,
    )
    sam, sys = create_v3_model(config)
    scale_te_edge_rest_lengths!(sys; scale=te_edge_scale)
    if g_earth !== nothing
        sam.set.g_earth = g_earth
    end
    return sys
end

# =============================================================================
# Analysis helpers
# =============================================================================

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

function calc_ref_area(sys)
    isempty(sys.wings) && return NaN
    wing = sys.wings[1]
    hasproperty(wing, :vsm_aero) || return NaN
    panels = wing.vsm_aero.panels
    isempty(panels) && return NaN
    return sum(p.chord * p.width for p in panels)
end

function mid_te_position(sl, k)
    Xk, Yk, Zk = sl.X[k], sl.Y[k], sl.Z[k]
    (length(Xk) < 11 || length(Yk) < 11 ||
     length(Zk) < 11) && return nothing
    pte10 = SVector{3}(Xk[10], Yk[10], Zk[10])
    pte11 = SVector{3}(Xk[11], Yk[11], Zk[11])
    return (pte10 + pte11) / 2
end

function mid_le_position(sl, k)
    Xk, Yk, Zk = sl.X[k], sl.Y[k], sl.Z[k]
    (length(Xk) < 14 || length(Yk) < 14 ||
     length(Zk) < 14) && return nothing
    ple12 = SVector{3}(Xk[12], Yk[12], Zk[12])
    ple14 = SVector{3}(Xk[14], Yk[14], Zk[14])
    return (ple12 + ple14) / 2
end

function compute_projected_area(sl, k, sys; eps=1e-12)
    Xk, Yk, Zk = sl.X[k], sl.Y[k], sl.Z[k]
    (length(Xk) < 7 || length(Yk) < 7 ||
     length(Zk) < 7) && return NaN
    p_le = mid_le_position(sl, k)
    p_te = mid_te_position(sl, k)
    (p_le === nothing || p_te === nothing) && return NaN
    chord_dir = p_te - p_le
    cn = norm(chord_dir)
    cn <= eps && return NaN
    cu = chord_dir / cn
    length(sl.orient) < k && return NaN
    R = SymbolicAWEModels.quaternion_to_rotation_matrix(
        sl.orient[k])
    by = SVector{3}(R[1, 2], R[2, 2], R[3, 2])
    byn = norm(by)
    byn <= eps && return NaN
    byu = by / byn
    byu = byu - dot(byu, cu) * cu
    byn2 = norm(byu)
    byn2 <= eps && return NaN
    byu = byu / byn2
    nodes = [SVector{3}(Xk[i], Yk[i], Zk[i])
             for i in 2:7]
    proj = [SVector{2}(dot(n - p_le, cu),
        dot(n - p_le, byu)) for n in nodes]
    area = 0.0
    for i in 1:length(proj)-1
        area += proj[i][1] * proj[i+1][2] -
                proj[i+1][1] * proj[i][2]
    end
    return abs(area) / 2.0
end

function analyze_log(lg, sys; window_sec=WINDOW_SEC)
    sl = lg.syslog
    length(sl.time) < 2 && return (
        aero_force=NaN, v_app=NaN, kite_vel=NaN,
        aoa=NaN, final_elevation=NaN, azimuth=NaN,
        proj_area=NaN, ref_area=NaN)
    aero_z = [sl.aero_force_b[i][3]
              for i in eachindex(sl.aero_force_b)]
    aero_force = mean_last_window(aero_z, sl.time;
        window_sec)
    v_app = mean_last_window(sl.v_app, sl.time;
        window_sec)
    vk = [norm(v) for v in sl.vel_kite]
    kite_vel = mean_last_window(vk, sl.time; window_sec)
    aoa_deg = rad2deg.(sl.AoA)
    aoa = mean_last_window(aoa_deg, sl.time; window_sec)
    elev_deg = rad2deg.(sl.elevation)
    final_elevation = mean_last_window(
        elev_deg, sl.time; window_sec)
    az_deg = rad2deg.(sl.azimuth)
    azimuth = mean_last_window(az_deg, sl.time;
        window_sec)
    proj_area = compute_projected_area(
        sl, length(sl.time), sys)
    ref_area = calc_ref_area(sys)
    return (aero_force=aero_force, v_app=v_app,
        kite_vel=kite_vel, aoa=aoa,
        final_elevation=final_elevation,
        azimuth=azimuth, proj_area=proj_area,
        ref_area=ref_area)
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
        parse_tags(name) === nothing && continue
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
    header = "vw,up,us,lt,aero_force,v_app,kite_vel," *
             "aoa,set_elevation,final_elevation,azimuth," *
             "proj_area,ref_area,g_earth"
    open(path, "w") do io
        println(io, header)
        for r in rows
            println(io, join([
                    r.vw, r.up, r.us, r.lt,
                    r.aero_force, r.v_app, r.kite_vel,
                    r.aoa, r.set_elevation,
                    r.final_elevation, r.azimuth,
                    r.proj_area, r.ref_area, r.g_earth
                ], ","))
        end
    end
end

function main()
    batch_name = isempty(ARGS) ? "" : strip(ARGS[1])
    # batch_name = "zenith_2019_batch_2026_02_23_11_13_35"
    if isempty(batch_name)
        print("Enter batch folder name: ")
        batch_name = strip(readline())
    end
    isempty(batch_name) && error("Batch name required.")

    batch_dir = resolve_batch_dir(batch_name)
    log_names = find_log_names(batch_dir)
    isempty(log_names) && error("No logs in: $batch_dir")

    rows = NamedTuple[]
    sys_cache = Dict{Tuple{Int,Int,Float64},
        SymbolicAWEModels.SystemStructure}()

    for log_name in log_names
        tags = parse_tags(log_name)
        tags === nothing && continue
        up, us, vw, lt, set_elev, g_earth = tags
        g_eff = isnothing(g_earth) ? 0.0 : g_earth
        key = (vw, lt, g_eff)
        sys = get!(sys_cache, key) do
            build_sys(v_wind=Float64(vw),
                tether_length=Float64(lt),
                g_earth=g_earth)
        end
        lg = load_log(log_name; path=batch_dir)
        m = analyze_log(lg, sys)
        push!(rows, (
            vw=vw, up=up, us=us, lt=lt,
            aero_force=m.aero_force, v_app=m.v_app,
            kite_vel=m.kite_vel, aoa=m.aoa,
            set_elevation=isnothing(set_elev) ?
                          NaN : set_elev,
            final_elevation=m.final_elevation,
            azimuth=m.azimuth, proj_area=m.proj_area,
            ref_area=m.ref_area, g_earth=g_eff))
    end

    sort!(rows, by=r -> (
        r.vw, r.up, r.us, r.lt,
        r.set_elevation, r.g_earth))

    out_path = joinpath(batch_dir,
        "hold_at_zenith_then_circles_batch_analysis.csv")
    write_csv(out_path, rows)
    @info "Wrote CSV" path = out_path rows = length(rows)
end

main()
nothing
