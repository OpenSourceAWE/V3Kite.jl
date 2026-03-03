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
    dt::Float64 = 0.01

    # Damping
    world_damping::Float64 = 1000.0
    min_damping::Float64 = 50.0
    decay_steps::Int = 2000
    body_damping::Float64 = 300.0

    # Flight condition
    v_wind::Float64 = 10.72
    elevation::Float64 = 70.0
    tether_length::Float64 = 240.0

    # Control
    steering_pct::Float64 = 0.0
    depower_pct::Float64 = 39.37
    heading_setpoint::Float64 = -1.562
    heading_kp::Float64 = 0.5
    heading_tau_i::Union{Float64,Bool} = false

    # Geometry modifications
    geom::V3GeomAdjustConfig = V3GeomAdjustConfig(
        reduce_tip=true, reduce_te=true,
        tether_length=240.0)

    # Model options
    n_panels::Int = 36
end

"""
    settle_wing(config::V3SettleConfig; v_app=config.v_wind,
                tether_length=config.tether_length,
                data_path=nothing, show_progress=true,
                remake=false) -> (sam, syslog)

Run a damped settling simulation to find equilibrium wing geometry.

Always returns a fresh model loaded from the settled YAML, so the
caller gets clean settings (normal gravity, no settling damping).

When `remake=false` and the destination YAML already exists, the
simulation is skipped and the settled geometry is loaded from file.
"""
function settle_wing(config::V3SettleConfig;
                     v_app=config.v_wind,
                     tether_length=config.tether_length,
                     elevation=config.elevation,
                     data_path=nothing,
                     show_progress=true,
                     remake=false)
    if isnothing(data_path)
        data_path = v3_data_path()
    end

    gc = config.geom
    suffix = build_geom_suffix(
        V3_DEPOWER_L0, gc.tip_reduction, gc.te_frac)
    suffix *= "_vapp$(round(v_app, digits=2))_lt$(Int(round(tether_length)))"
    dest_struc = joinpath(
        data_path, "struc_geometry_$(suffix).yaml")
    dest_aero = joinpath(
        data_path, "aero_geometry_$(suffix).yaml")
    source_struc = joinpath(
        data_path, config.source_struc_path)
    source_aero = joinpath(
        data_path, config.source_aero_path)

    # Run settling simulation if needed
    syslog = nothing
    if remake || !isfile(dest_struc)
        syslog = _run_settling_sim!(config;
            v_app, tether_length, elevation,
            data_path, show_progress,
            source_struc, source_aero,
            dest_struc, dest_aero, gc)
    end

    # Always load a fresh model from settled YAML
    @info "Loading settled geometry" dest_struc
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.v_wind = v_app
    set.l_tether = tether_length
    set.profile_law = 0

    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].n_panels = config.n_panels
    vsm_set.wings[1].geometry_file = dest_aero

    sys = load_sys_struct_from_yaml(dest_struc;
        system_name=V3_MODEL_NAME, set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)
    sam = SymbolicAWEModel(set, sys)
    SymbolicAWEModels.init!(sam;
        remake=false, ignore_l0=false, remake_vsm=true)

    return sam, syslog
end

"""Run the damped settling simulation and write results to YAML."""
function _run_settling_sim!(config::V3SettleConfig;
        v_app, tether_length, elevation,
        data_path, show_progress,
        source_struc, source_aero,
        dest_struc, dest_aero, gc)

    set_data_path(data_path)
    set = Settings("system.yaml")
    set.g_earth = 0.0
    set.v_wind = v_app
    set.l_tether = tether_length
    set.profile_law = 0

    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].n_panels = config.n_panels
    vsm_set.wings[1].geometry_file = source_aero

    sys = load_sys_struct_from_yaml(source_struc;
        system_name=V3_MODEL_NAME, set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)

    # Override geom tether_length from kwarg
    gc = V3GeomAdjustConfig(;
        reduce_tip=gc.reduce_tip,
        tip_reduction=gc.tip_reduction,
        tip_segments=gc.tip_segments,
        reduce_te=gc.reduce_te,
        te_frac=gc.te_frac,
        te_segments=gc.te_segments,
        reduce_depower=gc.reduce_depower,
        depower_reduction=gc.depower_reduction,
        tether_length=tether_length)

    # Set transform from heading/elevation
    sys.transforms[1].elevation = deg2rad(elevation)
    sys.transforms[1].azimuth = 0.0
    sys.transforms[1].heading = 0.0

    # Set damping
    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping)
    SymbolicAWEModels.set_body_frame_damping(
        sys, config.body_damping)

    # Create and init model
    sam = SymbolicAWEModel(set, sys)

    # Apply geometry modifications
    apply_geom_adjustments!(sys, gc)
    SymbolicAWEModels.init!(
        sam; remake=false, ignore_l0=false, remake_vsm=true)
    @show sys.winches[1].tether_len sys.segments[90].l0 set.l_tether

    @info "Settling REFINE wing" config.num_steps config.dt total_time=config.num_steps * config.dt

    # Lock tether
    for winch in sys.winches
        winch.brake = true
        winch.tether_len = tether_length
    end

    # Set steering/depower
    set_steering!(sys, config.steering_pct / 100.0)
    set_depower!(sys, config.depower_pct / 100.0, gc)

    # Logger
    logger, sys_state = create_logger(sam, config.num_steps)

    # Simulation loop
    @info "Starting settling simulation..."
    wing = sys.wings[1]
    for step in 1:config.num_steps
        t = step * config.dt

        # Damping decay
        damping = max(config.world_damping *
            (1.0 - step / config.decay_steps),
            config.min_damping)
        SymbolicAWEModels.set_world_frame_damping(
            sys, damping)

        # Step
        if !sim_step!(sam; dt=config.dt, vsm_interval=1)
            @error "Simulation failed" step t
            break
        end

        log_state!(logger, sys_state, sam, t)

        if show_progress && step % 20 == 0
            @info "Step $step/$(config.num_steps)" damping=round(damping, digits=1) elevation=round(rad2deg(wing.elevation), digits=2) heading=round(rad2deg(wing.heading), digits=2)
        end
    end

    # Write settled geometry to YAML
    @info "Updating YAML with settled positions..."
    SymbolicAWEModels.update_yaml_from_sys_struct!(
        sys, source_struc, dest_struc,
        source_aero, dest_aero)

    syslog = save_and_load_log(logger, "settle_refine_wing")
    @info "Settling complete" dest_struc dest_aero
    return syslog
end

"""
    V3InitConfig

Configuration for robust coupled-state initialization.
This is additive functionality and does not alter `settle_wing`.
"""
Base.@kwdef mutable struct V3InitConfig
    # Geometry paths
    source_struc_path::String = "struc_geometry.yaml"
    source_aero_path::String = "aero_geometry.yaml"
    vsm_settings_path::String = "vsm_settings.yaml"
    n_panels::Int = 36

    # Target operating point
    tether_length::Float64 = 240.0
    elevation::Float64 = 70.0
    azimuth::Float64 = 0.0
    heading::Float64 = 0.0
    upwind_dir::Float64 = -90.0
    g_earth::Float64 = 9.81
    target_wind::Float64 = 8.0
    target_depower::Float64 = 0.4
    target_steering::Float64 = 0.0

    # Continuation start point
    start_wind::Float64 = 0.0
    start_depower::Float64 = 0.4
    start_steering::Float64 = 0.0

    # Continuation and steady-state solve settings
    n_stages::Int = 6
    settle_time::Float64 = 1.0
    settle_dt::Float64 = 0.02
    vsm_interval::Int = 1

    # Final convergence checks and optional refinement
    max_extra_stages::Int = 3
    extra_settle_time_factor::Float64 = 1.5
    convergence_tol_elevation_deg::Float64 = 0.25
    convergence_tol_azimuth_deg::Float64 = 0.25
    convergence_tol_heading_deg::Float64 = 0.5
    convergence_tol_kite_speed::Float64 = 0.4
    enforce_convergence::Bool = false

    # Damping schedule
    world_damping_start::Float64 = 500.0
    world_damping_end::Float64 = 50.0
    body_damping::Float64 = 300.0

    # Optional dynamic warmup
    warmup_time::Float64 = 0.0
    warmup_fps::Int = 120

    # Optional model choices
    use_quasi_static::Bool = false
    use_settled_geometry::Bool = false
    settle_wind::Union{Nothing,Float64} = nothing
    fallback_to_raw_geometry::Bool = true
    settle_remake::Bool = false
    settle_num_steps::Int = 2000
    settle_step_dt::Float64 = 0.01

    # Geometry modifications
    geom::V3GeomAdjustConfig = V3GeomAdjustConfig(
        reduce_tip=true, reduce_te=true,
        tether_length=240.0)
end

@inline _lerp(a, b, f) = a + (b - a) * f

function _geom_with_tether(config::V3InitConfig)
    gc = config.geom
    return V3GeomAdjustConfig(
        reduce_tip=gc.reduce_tip,
        tip_reduction=gc.tip_reduction,
        tip_segments=gc.tip_segments,
        reduce_te=gc.reduce_te,
        te_frac=gc.te_frac,
        te_segments=gc.te_segments,
        reduce_depower=gc.reduce_depower,
        depower_reduction=gc.depower_reduction,
        tether_length=config.tether_length)
end

function _validate_init_config(config::V3InitConfig)
    config.n_stages > 0 || throw(ArgumentError("n_stages must be > 0"))
    config.settle_time > 0 || throw(ArgumentError("settle_time must be > 0"))
    config.settle_dt > 0 || throw(ArgumentError("settle_dt must be > 0"))
    config.start_wind >= 0 ||
        throw(ArgumentError("start_wind must be >= 0"))
    config.target_wind >= 0 ||
        throw(ArgumentError("target_wind must be >= 0"))
    if config.warmup_time > 0 && config.warmup_fps <= 0
        throw(ArgumentError("warmup_fps must be > 0 when warmup_time > 0"))
    end
    if config.settle_wind !== nothing && config.settle_wind < 0
        throw(ArgumentError("settle_wind must be >= 0"))
    end
    0.0 <= config.start_depower <= 1.0 ||
        throw(ArgumentError("start_depower must be in [0, 1]"))
    0.0 <= config.target_depower <= 1.0 ||
        throw(ArgumentError("target_depower must be in [0, 1]"))
    -1.0 <= config.start_steering <= 1.0 ||
        throw(ArgumentError("start_steering must be in [-1, 1]"))
    -1.0 <= config.target_steering <= 1.0 ||
        throw(ArgumentError("target_steering must be in [-1, 1]"))
    config.max_extra_stages >= 0 ||
        throw(ArgumentError("max_extra_stages must be >= 0"))
    config.extra_settle_time_factor > 0 ||
        throw(ArgumentError("extra_settle_time_factor must be > 0"))
    config.convergence_tol_elevation_deg >= 0 ||
        throw(ArgumentError("convergence_tol_elevation_deg must be >= 0"))
    config.convergence_tol_azimuth_deg >= 0 ||
        throw(ArgumentError("convergence_tol_azimuth_deg must be >= 0"))
    config.convergence_tol_heading_deg >= 0 ||
        throw(ArgumentError("convergence_tol_heading_deg must be >= 0"))
    config.convergence_tol_kite_speed >= 0 ||
        throw(ArgumentError("convergence_tol_kite_speed must be >= 0"))
    return nothing
end

function _init_convergence_metrics(sam, config::V3InitConfig)
    sys = sam.sys_struct
    wing = sys.wings[1]
    st = SysState(sam)
    update_sys_state!(st, sam)

    elevation_err_deg = rad2deg(wing.elevation) - config.elevation
    azimuth_err_deg = rad2deg(wrap_to_pi(
        wing.azimuth - deg2rad(config.azimuth)))
    heading_err_deg = rad2deg(wrap_to_pi(
        wing.heading - deg2rad(config.heading)))

    return (
        elevation_err_deg=elevation_err_deg,
        azimuth_err_deg=azimuth_err_deg,
        heading_err_deg=heading_err_deg,
        kite_speed=norm(st.vel_kite),
        aoa_deg=rad2deg(st.AoA),
    )
end

function _init_is_converged(metrics, config::V3InitConfig)
    return abs(metrics.elevation_err_deg) <= config.convergence_tol_elevation_deg &&
           abs(metrics.azimuth_err_deg) <= config.convergence_tol_azimuth_deg &&
           abs(metrics.heading_err_deg) <= config.convergence_tol_heading_deg &&
           metrics.kite_speed <= config.convergence_tol_kite_speed
end

function _enforce_target_state!(sys, config::V3InitConfig, gc)
    sys.set.v_wind = config.target_wind
    sys.set.upwind_dir = config.upwind_dir
    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping_end)
    set_depower!(sys, config.target_depower, gc)
    set_steering!(sys, config.target_steering)
    for winch in sys.winches
        winch.brake = true
        winch.tether_len = config.tether_length
        winch.tether_vel = 0.0
        winch.set_value = 0.0
    end
    return nothing
end

function _build_init_model(config::V3InitConfig; data_path, show_progress)
    gc = _geom_with_tether(config)

    if config.use_settled_geometry
        settle_v_app = isnothing(config.settle_wind) ?
            max(config.start_wind, config.target_wind) :
            float(config.settle_wind)
        settle_cfg = V3SettleConfig(
            source_struc_path=config.source_struc_path,
            source_aero_path=config.source_aero_path,
            vsm_settings_path=config.vsm_settings_path,
            num_steps=config.settle_num_steps,
            dt=config.settle_step_dt,
            v_wind=settle_v_app,
            elevation=config.elevation,
            tether_length=config.tether_length,
            steering_pct=100.0 * config.start_steering,
            depower_pct=100.0 * config.start_depower,
            geom=gc,
            n_panels=config.n_panels)
        try
            sam, _ = settle_wing(
                settle_cfg;
                v_app=settle_v_app,
                tether_length=config.tether_length,
                elevation=config.elevation,
                data_path=data_path,
                show_progress=show_progress,
                remake=config.settle_remake)
            return sam, sam.sys_struct, gc
        catch err
            if !config.fallback_to_raw_geometry
                rethrow(err)
            end
            @warn "settle_wing failed; falling back to raw initialization model" settle_v_app err
        end
    end

    sim_cfg = V3SimConfig(
        struc_yaml_path=config.source_struc_path,
        aero_yaml_path=config.source_aero_path,
        vsm_settings_path=config.vsm_settings_path,
        v_wind=config.start_wind,
        upwind_dir=config.upwind_dir,
        tether_length=config.tether_length,
        elevation=config.elevation,
        damping_pattern=[0.0, 0.0, 20.0],
        wing_type=REFINE,
        n_panels=config.n_panels)
    sam, sys = create_v3_model(sim_cfg; data_path=data_path)
    apply_geom_adjustments!(sys, gc)
    init!(sam; remake=false, ignore_l0=false, remake_vsm=true)
    return sam, sys, gc
end

"""
    initialize_state(config::V3InitConfig; data_path=nothing,
                     show_progress=true) -> sam

Robustly initialize a coupled tether + kite state using continuation
and `find_steady_state!` at each stage.
"""
function initialize_state(config::V3InitConfig;
                          data_path=nothing,
                          show_progress=true)
    _validate_init_config(config)
    if isnothing(data_path)
        data_path = v3_data_path()
    end

    sam, sys, gc = _build_init_model(
        config; data_path, show_progress=false)

    # Place transform explicitly at requested orientation.
    if !isempty(sys.transforms)
        tr = sys.transforms[1]
        tr.elevation = deg2rad(config.elevation)
        tr.azimuth = deg2rad(config.azimuth)
        tr.heading = deg2rad(config.heading)
        SymbolicAWEModels.reinit!([tr], sys)
    end

    sam.set.g_earth = config.g_earth
    sys.set.upwind_dir = config.upwind_dir
    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping_start)
    SymbolicAWEModels.set_body_frame_damping(
        sys, config.body_damping)

    for winch in sys.winches
        winch.brake = true
        winch.tether_len = config.tether_length
        winch.tether_vel = 0.0
        winch.set_value = 0.0
    end

    set_depower!(sys, config.start_depower, gc)
    set_steering!(sys, config.start_steering)
    SymbolicAWEModels.reinit!(
        sam, sam.prob, SymbolicAWEModels.FBDF())

    if config.use_quasi_static
        sam.set.quasi_static = true
        init!(sam; remake=false,
            ignore_l0=false, remake_vsm=true)
    end

    n_stages = max(1, config.n_stages)
    for stage in 1:n_stages
        frac = stage / n_stages
        stage_wind = _lerp(
            config.start_wind, config.target_wind, frac)
        stage_depower = _lerp(
            config.start_depower, config.target_depower, frac)
        stage_steering = _lerp(
            config.start_steering, config.target_steering, frac)
        stage_world_damping = _lerp(
            config.world_damping_start,
            config.world_damping_end, frac)

        sys.set.v_wind = stage_wind
        sys.set.upwind_dir = config.upwind_dir
        SymbolicAWEModels.set_world_frame_damping(
            sys, stage_world_damping)
        set_depower!(sys, stage_depower, gc)
        set_steering!(sys, stage_steering)
        for winch in sys.winches
            winch.brake = true
            winch.tether_len = config.tether_length
            winch.tether_vel = 0.0
            winch.set_value = 0.0
        end

        if show_progress
            @info "Initialize stage" stage n_stages wind=round(stage_wind, digits=3) depower=round(stage_depower, digits=3) steering=round(stage_steering, digits=3) world_damping=round(stage_world_damping, digits=2)
        end

        try
            SymbolicAWEModels.find_steady_state!(
                sam;
                t=config.settle_time,
                dt=config.settle_dt,
                vsm_interval=config.vsm_interval)
        catch err
            throw(ErrorException(
                "initialize_state failed at stage " *
                "$stage/$n_stages (wind=$(round(stage_wind, digits=3)), " *
                "depower=$(round(stage_depower, digits=3)), " *
                "steering=$(round(stage_steering, digits=3))): " *
                sprint(showerror, err)))
        end
    end

    if config.use_quasi_static
        sam.set.quasi_static = false
        init!(sam; remake=false,
            ignore_l0=false, remake_vsm=true)
    end

    # Check convergence at the requested target state and, if needed,
    # run extra steady-state passes at final settings.
    _enforce_target_state!(sys, config, gc)
    converged = false
    metrics = _init_convergence_metrics(sam, config)
    for pass in 0:config.max_extra_stages
        metrics = _init_convergence_metrics(sam, config)
        converged = _init_is_converged(metrics, config)
        if show_progress || !converged
            @info "Initialization convergence" pass converged elevation_err_deg=round(metrics.elevation_err_deg, digits=4) azimuth_err_deg=round(metrics.azimuth_err_deg, digits=4) heading_err_deg=round(metrics.heading_err_deg, digits=4) kite_speed=round(metrics.kite_speed, digits=5) aoa_deg=round(metrics.aoa_deg, digits=4)
        end
        if converged || pass == config.max_extra_stages
            break
        end
        extra_settle_time = config.settle_time *
            config.extra_settle_time_factor
        try
            SymbolicAWEModels.find_steady_state!(
                sam;
                t=extra_settle_time,
                dt=config.settle_dt,
                vsm_interval=config.vsm_interval)
        catch err
            throw(ErrorException(
                "initialize_state failed in extra stage " *
                "$(pass + 1)/$(config.max_extra_stages) " *
                "(t=$(round(extra_settle_time, digits=4))): " *
                sprint(showerror, err)))
        end
    end

    if !converged
        msg = "initialize_state convergence criteria not met " *
              "(elev_err=$(round(metrics.elevation_err_deg, digits=4)) deg, " *
              "az_err=$(round(metrics.azimuth_err_deg, digits=4)) deg, " *
              "heading_err=$(round(metrics.heading_err_deg, digits=4)) deg, " *
              "kite_speed=$(round(metrics.kite_speed, digits=5)) m/s)"
        if config.enforce_convergence
            throw(ErrorException(msg))
        elseif show_progress
            @warn msg
        end
    end

    if config.warmup_time > 0 && config.warmup_fps > 0
        n_warmup = max(1, Int(round(
            config.warmup_time * config.warmup_fps)))
        dt_warmup = config.warmup_time / n_warmup
        for step in 1:n_warmup
            frac = step / n_warmup
            warm_damping = _lerp(
                config.world_damping_end, 0.0, frac)
            SymbolicAWEModels.set_world_frame_damping(
                sys, max(0.0, warm_damping))
            if !sim_step!(sam;
                set_values=[0.0],
                dt=dt_warmup,
                vsm_interval=config.vsm_interval)
                throw(ErrorException(
                    "initialize_state warmup unstable at " *
                    "step $step/$n_warmup"))
            end
        end
    end

    return sam
end

function initialize_state(; data_path=nothing,
                          show_progress=true,
                          kwargs...)
    config = V3InitConfig(; kwargs...)
    return initialize_state(config;
        data_path=data_path,
        show_progress=show_progress)
end
