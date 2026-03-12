#!/usr/bin/env julia
# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Find a robust initial state with one single simulation phase.

Compared to `test_initialize_state.jl`, this script is intentionally simpler:
- one phase only (single ramped simulation loop),
- enforced per-step constraints:
  - KCU + tether node `y = 0`,
  - wing symmetry about the x-z plane (`y -> -y` mirror).

Usage:
    julia --project=. examples/find_initial_state.jl
"""

using V3Kite
using LinearAlgebra: norm
using Dates
using Serialization

const KCU_POINT_IDX = 1
const TETHER_POINT_IDXS = Int[V3_TETHER_POINT_IDXS...]

# Hardcoded from data/struc_geometry.yaml (left side y>=0 mirrored to right y<=0).
const WING_SYMMETRY_PAIRS = [
    (2, 20), (3, 21),
    (4, 18), (5, 19),
    (6, 16), (7, 17),
    (8, 14), (9, 15),
    (10, 12), (11, 13),
]

Base.@kwdef mutable struct SimpleInitConfig
    source_struc_path::String = "struc_geometry.yaml"
    source_aero_path::String = "aero_geometry.yaml"
    vsm_settings_path::String = "vsm_settings.yaml"
    n_panels::Int = 36

    # ###### 2025
    # tether_length::Float64 = 271 #269.0 #271
    # elevation::Float64 = 30.0
    # azimuth::Float64 = 0.0
    # heading::Float64 = 0.0
    # upwind_dir::Float64 = -90.0
    # g_earth::Float64 = 9.81

    # start_wind::Float64 = 7.6 #7.6
    # target_wind::Float64 = 7.6 #8.4

    # t_target_udp::Union{Nothing,Float64} = 5
    # start_udp::Float64 = 0.4
    # target_udp::Float64 = 0.42 #0.19 #0.42
    # target_steering::Float64 = 0.0

    # runtime::Float64 = 10
    # fps::Float64 = 120.0
    # vsm_interval::Int = 1

    # t_final_damping::Union{Nothing,Float64} = 8
    # world_damping_start::Float64 = 10.0
    # world_damping_end::Float64 = 0.0
    # body_damping_start::Vector{Float64} = [10.0, 50.0, 100.0]
    # body_damping_end::Vector{Float64} = [0.0, 0.0, 20.0]

    # ###### 2019
    # tether_length::Float64 = 269
    # elevation::Float64 = 75.0
    # azimuth::Float64 = 0.0
    # heading::Float64 = 0.0
    # upwind_dir::Float64 = -90.0
    # g_earth::Float64 = 0.0

    # start_wind::Float64 = 8.4
    # target_wind::Float64 = 8.4

    # t_target_udp::Union{Nothing,Float64} = 15
    # start_udp::Float64 = 0.4
    # target_udp::Float64 = 0.25
    # target_steering::Float64 = 0.0

    # runtime::Float64 = 35
    # fps::Float64 = 360.0
    # vsm_interval::Int = 1

    # t_final_damping::Union{Nothing,Float64} = 20
    # world_damping_start::Float64 = 10.0
    # world_damping_end::Float64 = 0.0
    # body_damping_start::Vector{Float64} = [10.0, 50.0, 100.0]
    # body_damping_end::Vector{Float64} = [0.0, 0.0, 20.0]

    ###### 22019-2025 MIX
    tether_length::Float64 = 270
    elevation::Float64 = 75.0
    azimuth::Float64 = 0.0
    heading::Float64 = 0.0
    upwind_dir::Float64 = -90.0
    g_earth::Float64 = 0.0

    start_wind::Float64 = 8.0
    target_wind::Float64 = 8.0

    t_target_udp::Union{Nothing,Float64} = 13
    start_udp::Float64 = 0.4
    target_udp::Float64 = 0.24
    target_steering::Float64 = 0.0

    runtime::Float64 = 15
    fps::Float64 = 360.0
    vsm_interval::Int = 1

    t_final_damping::Union{Nothing,Float64} = 15
    world_damping_start::Float64 = 10.0
    world_damping_end::Float64 = 0.0
    body_damping_start::Vector{Float64} = [10.0, 50.0, 100.0]
    body_damping_end::Vector{Float64} = [0.0, 0.0, 20.0]


    use_heading_controller::Bool = false
    heading_ctrl_p::Float64 = 1.0
    heading_ctrl_i::Float64 = 0.1
    heading_ctrl_d::Float64 = 0.0
    heading_ctrl_max_steering::Float64 = 0.2

    symmetry_tolerance::Float64 = 1e-8
    y_zero_tolerance::Float64 = 1e-10

    geom::V3GeomAdjustConfig = V3GeomAdjustConfig(
        reduce_tip=true,
        reduce_te=true,
        reduce_depower_tape_by=0.0,
        reduce_steering_tapes_by=0.0,
        tether_length=269.0)
end

@inline lerp(a, b, f) = a + (b - a) * f

@inline function udp_to_l0(udp)
    return (200.0 + 5000.0 * udp) / 1000.0
end

@inline function set_udp!(sys, udp, gc)
    sys.segments[V3_DEPOWER_IDX].l0 = udp_to_l0(udp) - gc.reduce_depower_tape_by
    return nothing
end

@inline function point_by_idx(sys, idx::Int)
    if 1 <= idx <= length(sys.points) && sys.points[idx].idx == idx
        return sys.points[idx]
    end
    i = findfirst(p -> p.idx == idx, sys.points)
    isnothing(i) && throw(ArgumentError("Point index $idx not found in system"))
    return sys.points[i]
end

@inline function symmetry_error(a, b)
    return max(
        abs(a[1] - b[1]),
        abs(a[2] + b[2]),
        abs(a[3] - b[3]),
    )
end

@inline function mirror_right_from_left!(left, right)
    right.pos_w[1] = left.pos_w[1]
    right.pos_w[2] = -left.pos_w[2]
    right.pos_w[3] = left.pos_w[3]

    right.vel_w[1] = left.vel_w[1]
    right.vel_w[2] = -left.vel_w[2]
    right.vel_w[3] = left.vel_w[3]

    right.pos_b[1] = left.pos_b[1]
    right.pos_b[2] = -left.pos_b[2]
    right.pos_b[3] = left.pos_b[3]
end

function check_initial_symmetry(sys; tol=1e-8)
    max_err = 0.0
    for (lidx, ridx) in WING_SYMMETRY_PAIRS
        left = point_by_idx(sys, lidx)
        right = point_by_idx(sys, ridx)
        max_err = max(max_err, symmetry_error(left.pos_cad, right.pos_cad))
    end
    if max_err > tol
        @warn "Initial CAD geometry is not perfectly symmetric" max_err tol
    else
        @info "Initial CAD geometry symmetry check passed" max_err tol
    end
    return max_err
end

function enforce_kcu_tether_y_zero!(sys; tol=1e-10)
    n_changed = 0
    for idx in (KCU_POINT_IDX, TETHER_POINT_IDXS...)
        p = point_by_idx(sys, idx)
        if abs(p.pos_w[2]) > tol
            p.pos_w[2] = 0.0
            n_changed += 1
        end
        if abs(p.vel_w[2]) > tol
            p.vel_w[2] = 0.0
            n_changed += 1
        end
        if abs(p.pos_b[2]) > tol
            p.pos_b[2] = 0.0
            n_changed += 1
        end
    end
    return n_changed
end

function enforce_wing_symmetry!(sys; tol=1e-8)
    max_err = 0.0
    n_changed = 0
    for (lidx, ridx) in WING_SYMMETRY_PAIRS
        left = point_by_idx(sys, lidx)
        right = point_by_idx(sys, ridx)
        err = symmetry_error(left.pos_w, right.pos_w)
        max_err = max(max_err, err)
        if err > tol
            mirror_right_from_left!(left, right)
            n_changed += 1
        end
    end
    return max_err, n_changed
end

function lock_winch!(sys, tether_length)
    for winch in sys.winches
        winch.brake = true
        winch.tether_len = tether_length
        winch.tether_vel = 0.0
        winch.set_value = 0.0
    end
    return nothing
end

function build_model(cfg::SimpleInitConfig)
    sim_cfg = V3SimConfig(
        struc_yaml_path=cfg.source_struc_path,
        aero_yaml_path=cfg.source_aero_path,
        vsm_settings_path=cfg.vsm_settings_path,
        v_wind=cfg.start_wind,
        upwind_dir=cfg.upwind_dir,
        tether_length=cfg.tether_length,
        elevation=cfg.elevation,
        damping_pattern=cfg.body_damping_start,
        wing_type=REFINE,
        n_panels=cfg.n_panels,
    )

    sam, sys = create_v3_model(sim_cfg)
    apply_geom_adjustments!(sys, cfg.geom)
    init!(sam; remake=false, ignore_l0=false, remake_vsm=true)
    apply_vsm_solver_settings!(sys)

    if !isempty(sys.transforms)
        tr = sys.transforms[1]
        tr.elevation = deg2rad(cfg.elevation)
        tr.azimuth = deg2rad(cfg.azimuth)
        tr.heading = deg2rad(cfg.heading)
        SymbolicAWEModels.reinit!([tr], sys)
    end

    return sam, sys
end

function build_log_name(cfg::SimpleInitConfig, timestamp)
    up_t = Int(round(cfg.target_udp * 100))
    us_t = Int(round(cfg.target_steering * 100))
    vw_t = Int(round(cfg.target_wind))
    lt_t = Int(round(cfg.tether_length))
    return "find_initial_state__up_$(up_t)_us_$(us_t)_vw_$(vw_t)_lt_$(lt_t)_date_$(timestamp)"
end

@inline function resolve_yaml_path(path::String, data_root::String)
    return isabspath(path) ? path : joinpath(data_root, path)
end

function build_initial_state_yaml_names(cfg::SimpleInitConfig)
    lt_tag = Int(round(cfg.tether_length))
    vw_tag = Int(round(cfg.target_wind * 10))
    udp_tag = lpad(string(Int(round(cfg.target_udp * 100))), 3, '0')
    struc_name = "struc_geometry_initial_state_lt_$(lt_tag)_vw_$(vw_tag)_udp_$(udp_tag).yaml"
    aero_name = "aero_geometry_initial_state_lt_$(lt_tag)_vw_$(vw_tag)_udp_$(udp_tag).yaml"
    return struc_name, aero_name
end

function build_initial_state_snapshot_name(cfg::SimpleInitConfig)
    lt_tag = Int(round(cfg.tether_length))
    vw_tag = Int(round(cfg.target_wind * 10))
    udp_tag = lpad(string(Int(round(cfg.target_udp * 100))), 3, '0')
    return "initial_state_snapshot_lt_$(lt_tag)_vw_$(vw_tag)_udp_$(udp_tag).jls"
end

@inline vec3(v) = Float64[v[1], v[2], v[3]]
@inline vec4(v) = Float64[v[1], v[2], v[3], v[4]]
@inline mat3(R) = Float64[R[i, j] for i in 1:3, j in 1:3]

function snapshot_pid_state(pid)
    isnothing(pid) && return nothing
    state = Dict{String,Any}()
    for fn in fieldnames(typeof(pid))
        v = getfield(pid, fn)
        state[string(fn)] = v isa Number ? Float64(v) : v
    end
    return state
end

function build_initial_state_snapshot_payload(
    cfg::SimpleInitConfig,
    sys;
    heading_pid=nothing,
    heading_target=NaN)

    point_states = [
        (
            idx=Int(p.idx),
            pos_w=vec3(p.pos_w),
            vel_w=vec3(p.vel_w),
        )
        for p in sys.points
    ]

    winch_states = [
        (
            idx=Int(w.idx),
            tether_len=Float64(w.tether_len),
            tether_vel=Float64(w.tether_vel),
            set_value=Float64(w.set_value),
            brake=Bool(w.brake),
        )
        for w in sys.winches
    ]

    group_states = [
        (
            idx=Int(g.idx),
            twist=Float64(g.twist),
            twist_rate=Float64(g.twist_ω),
        )
        for g in sys.groups
    ]

    wing_states = [
        (
            idx=Int(w.idx),
            wing_type=string(w.wing_type),
            pos_w=vec3(w.pos_w),
            vel_w=vec3(w.vel_w),
            Q_b_to_w=vec4(w.Q_b_to_w),
            R_b_to_w=mat3(w.R_b_to_w),
            omega_b=vec3(w.ω_b),
            turn_rate=vec3(w.turn_rate),
            elevation=Float64(w.elevation),
            azimuth=Float64(w.azimuth),
            heading=Float64(w.heading),
        )
        for w in sys.wings
    ]

    controller_state = (
        use_heading_controller=cfg.use_heading_controller,
        heading_target=Float64(heading_target),
        target_steering=Float64(cfg.target_steering),
        heading_pid=snapshot_pid_state(heading_pid),
    )

    return (
        format_version=1,
        created_at=string(Dates.now()),
        cfg=(
            source_struc_path=cfg.source_struc_path,
            source_aero_path=cfg.source_aero_path,
            vsm_settings_path=cfg.vsm_settings_path,
            tether_length=Float64(cfg.tether_length),
            target_wind=Float64(cfg.target_wind),
            target_udp=Float64(cfg.target_udp),
            target_steering=Float64(cfg.target_steering),
            upwind_dir=Float64(cfg.upwind_dir),
            g_earth=Float64(cfg.g_earth),
        ),
        points=point_states,
        winches=winch_states,
        groups=group_states,
        wings=wing_states,
        controller=controller_state,
    )
end

@inline function format_angle_deg(val_rad::Float64)
    deg = rad2deg(val_rad)
    rounded = abs(deg) < 1e-10 ? 0.0 : round(deg, digits=4)
    return string(rounded)
end

function update_transform_angles_in_struc_yaml!(
    struc_yaml_path::String,
    sys)
    isempty(sys.transforms) && return nothing

    transform_angles_by_idx = Dict{Int,NamedTuple{(:elevation, :azimuth, :heading),Tuple{Float64,Float64,Float64}}}()
    for tr in sys.transforms
        elev = float(tr.elevation)
        azim = float(tr.azimuth)
        head = float(tr.heading)

        # Prefer converged wing angles for transforms that drive a wing.
        if !isnothing(tr.wing_idx) &&
           1 <= tr.wing_idx <= length(sys.wings)
            wing = sys.wings[tr.wing_idx]
            elev = float(wing.elevation)
            azim = float(wing.azimuth)
            head = float(wing.heading)
        end

        transform_angles_by_idx[Int(tr.idx)] = (
            elevation=elev,
            azimuth=azim,
            heading=head)
    end

    lines = readlines(struc_yaml_path)
    in_transforms_section = false
    in_transforms_data = false
    current_transform_idx = nothing
    n_updates = 0

    for (i, line) in enumerate(lines)
        if occursin(r"^transforms:", line)
            in_transforms_section = true
            in_transforms_data = false
            current_transform_idx = nothing
            continue
        elseif in_transforms_section && occursin(r"^\w+:", line)
            in_transforms_section = false
            in_transforms_data = false
            current_transform_idx = nothing
        end

        if !in_transforms_section
            continue
        end

        if occursin(r"^\s*data:\s*$", line)
            in_transforms_data = true
            continue
        end
        in_transforms_data || continue

        idx_match = match(r"^(\s*-\s*idx:\s*)(\d+)(\s*(#.*)?)$", line)
        if idx_match !== nothing
            current_transform_idx = parse(Int, idx_match.captures[2])
            continue
        end
        isnothing(current_transform_idx) && continue
        haskey(transform_angles_by_idx, current_transform_idx) || continue

        angles = transform_angles_by_idx[current_transform_idx]
        key_to_val = Dict(
            "elevation" => format_angle_deg(angles.elevation),
            "azimuth" => format_angle_deg(angles.azimuth),
            "heading" => format_angle_deg(angles.heading),
        )
        for (key, val) in key_to_val
            rx = Regex(
                "^\\s*(" * key * ":\\s*)([-+]?\\d+\\.?\\d*(?:[eE][-+]?\\d+)?)(\\s*(#.*)?)\$")
            m = match(
                rx,
                strip(line))
            if m !== nothing
                indent = match(r"^(\s*)", line).captures[1]
                suffix = isnothing(m.captures[3]) ? "" : m.captures[3]
                lines[i] = indent * m.captures[1] * val * suffix
                n_updates += 1
                break
            end
        end
    end

    open(struc_yaml_path, "w") do io
        for line in lines
            println(io, line)
        end
    end
    @info "Updated transform angles in structural YAML" path = struc_yaml_path n_updates
    return nothing
end

function save_initial_state_yamls!(cfg::SimpleInitConfig, sys)
    data_root = v3_data_path()
    source_struc = resolve_yaml_path(cfg.source_struc_path, data_root)
    source_aero = resolve_yaml_path(cfg.source_aero_path, data_root)
    struc_name, aero_name = build_initial_state_yaml_names(cfg)
    dest_struc = joinpath(dirname(source_struc), struc_name)
    dest_aero = joinpath(dirname(source_aero), aero_name)

    SymbolicAWEModels.update_yaml_from_sys_struct!(
        sys,
        source_struc, dest_struc,
        source_aero, dest_aero)
    update_transform_angles_in_struc_yaml!(dest_struc, sys)

    @info "Saved initial-state geometry YAMLs" dest_struc dest_aero
    return dest_struc, dest_aero
end

function save_initial_state_snapshot!(
    cfg::SimpleInitConfig,
    sys;
    heading_pid=nothing,
    heading_target=NaN)
    data_root = v3_data_path()
    source_struc = resolve_yaml_path(cfg.source_struc_path, data_root)
    snapshot_name = build_initial_state_snapshot_name(cfg)
    snapshot_path = joinpath(dirname(source_struc), snapshot_name)
    payload = build_initial_state_snapshot_payload(
        cfg, sys;
        heading_pid,
        heading_target)

    open(snapshot_path, "w") do io
        serialize(io, payload)
    end
    @info "Saved initial-state restart snapshot" snapshot_path
    return snapshot_path
end

function run_one_phase!(cfg::SimpleInitConfig)
    sam, sys = build_model(cfg)
    check_initial_symmetry(sys; tol=cfg.symmetry_tolerance)

    sam.set.g_earth = cfg.g_earth
    sys.set.upwind_dir = cfg.upwind_dir
    lock_winch!(sys, cfg.tether_length)

    n_steps = max(1, Int(round(cfg.runtime * cfg.fps)))
    dt = cfg.runtime / n_steps
    t_final_damping = isnothing(cfg.t_final_damping) ?
                      cfg.runtime : float(cfg.t_final_damping)
    t_target_udp = isnothing(cfg.t_target_udp) ?
                   cfg.runtime : float(cfg.t_target_udp)
    logger, sys_state = create_logger(sam, n_steps + 1)

    heading_pid = if cfg.use_heading_controller
        Ti = cfg.heading_ctrl_i > 0 ? 1.0 / cfg.heading_ctrl_i : false
        Td = cfg.heading_ctrl_d > 0 ? cfg.heading_ctrl_d : false
        umax = abs(cfg.heading_ctrl_max_steering)
        create_heading_pid(;
            K=cfg.heading_ctrl_p,
            Ti,
            Td,
            dt,
            umin=-umax,
            umax=umax)
    else
        nothing
    end
    heading_target = deg2rad(cfg.heading)

    max_sym_err_seen = 0.0
    n_constraint_events = 0

    @info "Starting one-phase initialization" n_steps dt runtime = cfg.runtime t_final_damping t_target_udp
    for step in 1:n_steps
        frac = step / n_steps
        t = step * dt
        damping_frac = t_final_damping <= 0 ?
                       1.0 : clamp(t / t_final_damping, 0.0, 1.0)
        udp_frac = t_target_udp <= 0 ?
                   1.0 : clamp(t / t_target_udp, 0.0, 1.0)

        stage_wind = lerp(cfg.start_wind, cfg.target_wind, frac)
        stage_udp = lerp(cfg.start_udp, cfg.target_udp, udp_frac)
        stage_world_damping = lerp(
            cfg.world_damping_start,
            cfg.world_damping_end,
            damping_frac)
        stage_body_damping = cfg.body_damping_start .+
                             (cfg.body_damping_end .- cfg.body_damping_start) .* damping_frac

        sys.set.v_wind = stage_wind
        SymbolicAWEModels.set_world_frame_damping(sys, stage_world_damping)
        SymbolicAWEModels.set_body_frame_damping(sys, stage_body_damping)

        set_udp!(sys, stage_udp, cfg.geom)
        if isnothing(heading_pid)
            set_steering!(sys, cfg.target_steering, cfg.geom)
        else
            steer_corr = heading_pid(
                heading_target,
                sys.wings[1].heading,
                0.0)
            set_steering!(
                sys,
                clamp(cfg.target_steering + steer_corr, -1.0, 1.0),
                cfg.geom)
        end
        lock_winch!(sys, cfg.tether_length)

        if !sim_step!(sam; set_values=[0.0], dt, vsm_interval=cfg.vsm_interval)
            error("Initialization became unstable at step $step/$n_steps")
        end

        n_changed_y = enforce_kcu_tether_y_zero!(
            sys; tol=cfg.y_zero_tolerance)
        sym_err, n_changed_sym = enforce_wing_symmetry!(
            sys; tol=cfg.symmetry_tolerance)
        max_sym_err_seen = max(max_sym_err_seen, sym_err)

        if n_changed_y > 0 || n_changed_sym > 0
            n_constraint_events += 1
            SymbolicAWEModels.reinit!(sam, sam.prob, SymbolicAWEModels.FBDF())
        end

        log_state!(logger, sys_state, sam, t)

        if should_report(step, n_steps)
            elev_deg = rad2deg(sys.wings[1].elevation)
            alpha_deg = rad2deg(sys_state.AoA)
            @info "t = $(round(t, digits=2))"
            println("| elev         = $(round(elev_deg, digits=2))deg")
            println("│ udp          = $(round(stage_udp, digits=3))")
            println("│ alpha        = $(round(alpha_deg, digits=2))")
            println("│ wind         = $(round(stage_wind, digits=3))")
            println("│ symmetry_err = $(round(sym_err, digits=8))")
        end
    end

    st = SysState(sam)
    update_sys_state!(st, sam)
    wing = sys.wings[1]
    elevation_deg = rad2deg(wing.elevation)
    azimuth_deg = rad2deg(wing.azimuth)
    heading_deg = rad2deg(wing.heading)

    println("One-phase initialization summary")
    println("  wind [m/s]          : ", round(sys.set.v_wind, digits=4))
    println("  elevation [deg]     : ", round(elevation_deg, digits=4))
    println("  azimuth [deg]       : ", round(azimuth_deg, digits=4))
    println("  heading [deg]       : ", round(heading_deg, digits=4))
    println("  tether_len [m]      : ", round(sys.winches[1].tether_len, digits=4))
    println("  udp [-]             : ", round((1000.0 * sys.segments[V3_DEPOWER_IDX].l0 - 200.0) / 5000.0, digits=4))
    println("  kite speed [m/s]    : ", round(norm(st.vel_kite), digits=5))
    println("  max symmetry err [m]: ", round(max_sym_err_seen, digits=8))
    println("  constraint events   : ", n_constraint_events)

    ts = Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
    save_root = joinpath(dirname(@__DIR__), "processed_data")
    save_dir = joinpath(save_root, "find_initial_state_" * ts)
    isdir(save_dir) || mkpath(save_dir)
    log_name = build_log_name(cfg, ts)
    save_log(logger, log_name; path=save_dir)
    @info "Saved initialization log" log_name path = joinpath(save_dir, log_name * ".arrow")
    save_initial_state_yamls!(cfg, sys)
    save_initial_state_snapshot!(
        cfg, sys;
        heading_pid,
        heading_target)

    return sam
end

cfg = SimpleInitConfig()
@info "Running find_initial_state with defaults from test_initialize_state.jl"
run_one_phase!(cfg)
nothing
