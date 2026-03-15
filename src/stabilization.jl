# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Wing settling/stabilization functions.
Runs a damped simulation to find equilibrium wing geometry and writes
the settled positions back to YAML files.
"""

"""
    V3SettleConfig

Configuration for wing settling simulation.
"""
Base.@kwdef mutable struct V3SettleConfig
    # Geometry paths
    source_struc_path::String = "struc_geometry.yaml"
    source_aero_path::String = "aero_geometry.yaml"
    vsm_settings_path::String = "vsm_settings.yaml"

    # Simulation parameters
    num_steps::Int = 8000
    num_substeps::Int = 1
    dt::Float64 = 0.01

    # Damping
    world_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 0.0]
    min_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 0.0]
    decay_steps::Int = 2000
    body_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 20.0]

    # Flight condition
    v_wind::Float64 = 10.72
    elevation::Float64 = 70.0
    azimuth::Float64 = 0.0
    heading::Float64 = 0.0
    tether_length::Float64 = 240.0

    # Control
    heading_setpoint::Float64 = -1.562
    heading_kp::Float64 = 0.5
    heading_tau_i::Union{Float64,Bool} = false

    # Geometry modifications
    geom::V3GeomAdjustConfig = V3GeomAdjustConfig(
        reduce_tip=true, reduce_te=true,
        tether_length=240.0)

    # Depower ramp (power-zone settling only)
    start_depower::Union{Nothing,Float64} = nothing
    end_depower::Union{Nothing,Float64} = nothing

    # Model options
    n_panels::Int = 36
    fix_sphere_idxs::Vector{Int} = Int[]
end

"""
    settle_wing(config::V3SettleConfig; data_path=nothing,
                show_progress=true, remake=false,
                init_row=nothing,
                power_zone=false) -> (sam, syslog)

Run a damped settling simulation to find equilibrium wing
geometry.

Always returns a fresh model loaded from the settled YAML, so
the caller gets clean settings (normal gravity, no settling
damping).

When `remake=false` and the destination YAML already exists,
the simulation is skipped and the settled geometry is loaded
from file.

When `power_zone=true`, runs power-zone settling with gravity
using `init_row` for position/orientation/controls. When
`power_zone=false`, runs zero-gravity geometry settling; pass
`init_row` to set steering/depower from flight data.
"""
function settle_wing(config::V3SettleConfig;
                     data_path=nothing,
                     show_progress=true,
                     remake=false,
                     init_row=nothing,
                     power_zone::Bool=false)
    if isnothing(data_path)
        data_path = v3_data_path()
    end

    gc = config.geom
    gc.tether_length = config.tether_length

    dp_reduction = gc.reduce_depower ?
        gc.depower_reduction : 0.0
    st_reduction = gc.reduce_steering ?
        gc.steering_reduction : 0.0
    dp_norm = if !isnothing(config.end_depower)
        config.end_depower / 100.0
    elseif !isnothing(init_row)
        init_row.depower
    else
        0.0
    end
    st_norm = isnothing(init_row) ? 0.0 : init_row.steering
    depower_tape = depower_percentage_to_length(
        dp_norm * 100.0;
        l0_base=V3_DEPOWER_L0_BASE - dp_reduction)
    L_left, L_right = steering_percentage_to_lengths(
        st_norm * 100.0;
        l0_base=V3_STEERING_L0_BASE - st_reduction)
    tip_red = gc.reduce_tip ? gc.tip_reduction : 0.0
    te_f = gc.reduce_te ? gc.te_frac : 1.0
    suffix = build_geom_suffix(depower_tape,
        L_left, L_right, tip_red, te_f)
    suffix *= "_vapp$(round(config.v_wind, digits=2))" *
        "_lt$(Int(round(config.tether_length)))"
    if power_zone
        suffix *= "_pz"
    end
    dest_struc = joinpath(
        data_path, "struc_geometry_$(suffix).yaml")
    source_struc = joinpath(
        data_path, config.source_struc_path)
    source_aero = joinpath(
        data_path, config.source_aero_path)

    # Run settling simulation if needed
    syslog = nothing
    if remake || !isfile(dest_struc)
        if power_zone
            syslog = _run_power_zone_settling!(config;
                data_path, show_progress,
                source_struc, source_aero,
                dest_struc, init_row)
        else
            syslog = _run_zero_g_settling!(config;
                data_path, show_progress,
                source_struc, source_aero,
                dest_struc, init_row)
        end
    end

    # Always load a fresh model from settled YAML
    @info "Loading settled geometry" dest_struc
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    set.profile_law = 0
    if power_zone
        set.v_wind = init_row.wind_speed
    end

    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].n_panels = config.n_panels
    vsm_set.wings[1].geometry_file = source_aero

    sys = load_sys_struct_from_yaml(dest_struc;
        system_name=V3_MODEL_NAME, set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)
    sam = SymbolicAWEModel(set, sys)
    SymbolicAWEModels.init!(sam;
        remake=false, ignore_l0=false, remake_vsm=true)

    return sam, syslog
end

"""
Set up a settling model: settings, VSM, sys struct, damping,
SAM creation, geometry adjustments, init, and lock tether.
Returns `(sam, sys, gc)`.
"""
function _setup_settling_model(config::V3SettleConfig;
        g_earth, data_path, source_struc, source_aero)
    gc = config.geom
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.g_earth = g_earth
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    set.profile_law = 0

    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].n_panels = config.n_panels
    vsm_set.wings[1].geometry_file = source_aero

    sys = load_sys_struct_from_yaml(source_struc;
        system_name=V3_MODEL_NAME, set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)

    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping)
    SymbolicAWEModels.set_body_frame_damping(
        sys, config.body_damping)

    sam = SymbolicAWEModel(set, sys)
    apply_geom_adjustments!(sys, gc)
    SymbolicAWEModels.init!(
        sam; remake=false, ignore_l0=false, remake_vsm=true)

    @info "Settling REFINE wing" config.num_steps config.dt total_time=config.num_steps * config.dt

    for winch in sys.winches
        winch.brake = true
        winch.tether_len = config.tether_length
    end

    return sam, sys, gc
end

"""Run zero-gravity settling to find equilibrium geometry."""
function _run_zero_g_settling!(config::V3SettleConfig;
        data_path, show_progress,
        source_struc, source_aero,
        dest_struc,
        init_row=nothing)
    sam, sys, gc = _setup_settling_model(config;
        g_earth=0.0, data_path, source_struc, source_aero)

    # Set initial transform from config
    sys.transforms[1].elevation = deg2rad(config.elevation)
    sys.transforms[1].azimuth = deg2rad(config.azimuth)
    sys.transforms[1].heading = deg2rad(config.heading)

    # Set control from init_row if provided
    if !isnothing(init_row)
        set_steering!(sys, init_row.steering, gc)
        set_depower!(sys, init_row.depower, 0.0, gc)
    end

    logger, sys_state = create_logger(sam, config.num_steps)

    # Save original transform values
    saved_el = sys.transforms[1].elevation
    saved_az = sys.transforms[1].azimuth
    saved_hd = sys.transforms[1].heading

    @info "Starting zero-g settling..."
    wing = sys.wings[1]
    for step in 1:config.num_steps
        t = step * config.dt

        damping = max(config.world_damping *
            (1.0 - step / config.decay_steps),
            config.min_damping)
        SymbolicAWEModels.set_world_frame_damping(
            sys, damping)

        if !sim_step!(sam; dt=config.dt, vsm_interval=1)
            @error "Simulation failed" step t
            break
        end

        log_state!(logger, sys_state, sam, t)

        if show_progress && step % 20 == 0
            @info "Step $step/$(config.num_steps)" damping=round(damping, digits=1) elevation=round(rad2deg(wing.elevation), digits=2) heading=round(rad2deg(wing.heading), digits=2)
        end
    end

    # Reset transform to saved values
    sys.transforms[1].elevation = saved_el
    sys.transforms[1].azimuth = saved_az
    sys.transforms[1].heading = saved_hd
    SymbolicAWEModels.reinit!(
        sys.transforms, sys; update_vel=false)

    @info "Updating YAML with settled positions..."
    SymbolicAWEModels.update_yaml_from_sys_struct!(
        sys, source_struc, dest_struc,
        source_aero, source_aero)

    syslog = save_and_load_log(logger, "settle_refine_wing")
    @info "Settling complete" dest_struc
    return syslog
end

"""Run power-zone settling initialized from flight data."""
function _run_power_zone_settling!(config::V3SettleConfig;
        data_path, show_progress,
        source_struc, source_aero,
        dest_struc,
        init_row)
    sam, sys, gc = _setup_settling_model(config;
        g_earth=9.81, data_path, source_struc, source_aero)
    sam.set.v_wind = init_row.wind_speed

    update_sys_struct_from_data!(sys, init_row; config=gc)

    # Override initial depower if ramp is configured
    if !isnothing(config.start_depower)
        set_depower!(sys, config.start_depower / 100.0, 0.0, gc)
    end

    SymbolicAWEModels.reinit!(
        sam, sam.prob, SymbolicAWEModels.FBDF())

    for idx in config.fix_sphere_idxs
        sys.points[idx].fix_sphere = true
    end

    total_steps = config.num_steps * config.num_substeps
    logger, sys_state = create_logger(sam, total_steps)

    @info "Starting power-zone settling..." num_substeps=config.num_substeps
    wing = sys.wings[1]
    failed = false
    for step in 1:config.num_steps
        damping = max(config.world_damping *
            (1.0 - step / config.decay_steps),
            config.min_damping)
        SymbolicAWEModels.set_world_frame_damping(
            sys, damping)

        # Ramp depower linearly over settling steps
        if !isnothing(config.start_depower)
            dp_end = isnothing(config.end_depower) ?
                init_row.depower * 100.0 :
                config.end_depower
            frac = (step - 1) / max(config.num_steps - 1, 1)
            dp = config.start_depower +
                frac * (dp_end - config.start_depower)
            set_depower!(sys, dp / 100.0, 0.0, gc)
        end

        SymbolicAWEModels.reposition!(
            sys.transforms, sys)
        SymbolicAWEModels.reinit!(
            sam, sam.prob, SymbolicAWEModels.FBDF())

        for sub in 1:config.num_substeps
            global_step = (step - 1) * config.num_substeps + sub
            t = global_step * config.dt

            if !sim_step!(sam; dt=config.dt, vsm_interval=1)
                @error "Simulation failed" step sub t
                failed = true
                break
            end

            log_state!(logger, sys_state, sam, t)

            if show_progress && global_step % 20 == 0
                @info "Step $step/$(config.num_steps)" substep=sub damping=round(damping, digits=1) elevation=round(rad2deg(wing.elevation), digits=2) heading=round(rad2deg(wing.heading), digits=2)
            end
        end
        failed && break
    end

    @info "Updating YAML with settled positions..."
    SymbolicAWEModels.update_yaml_from_sys_struct!(
        sys, source_struc, dest_struc,
        source_aero, source_aero)

    syslog = save_and_load_log(logger, "settle_refine_wing")
    @info "Settling complete" dest_struc
    return syslog
end
