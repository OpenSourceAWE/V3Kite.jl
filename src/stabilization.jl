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

Returns the model and simulation log. The settled positions are
written to destination YAML files derived from the geometry suffix.

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

    # Load settings
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.v_wind = v_app
    set.l_tether = tether_length
    set.profile_law = 0

    # Load VSM
    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].n_panels = config.n_panels

    # Early return if settled YAML exists and !remake
    if !remake && isfile(dest_struc)
        @info "Loading settled geometry from file" dest_struc
        vsm_set.wings[1].geometry_file = dest_aero
        sys = load_sys_struct_from_yaml(dest_struc;
            system_name=V3_MODEL_NAME, set,
            wing_type=SymbolicAWEModels.REFINE, vsm_set)
        sam = SymbolicAWEModel(set, sys)
        SymbolicAWEModels.init!(sam;
            remake=false, ignore_l0=false,
            remake_vsm=true)
        return sam, nothing
    end

    # Settling uses zero gravity
    set.g_earth = 0.0
    vsm_set.wings[1].geometry_file = source_aero

    # Load system structure
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
    SymbolicAWEModels.init!(
        sam; remake=false, ignore_l0=false, remake_vsm=true)

    # Apply geometry modifications after init
    # (so winch force is available for tether calculation)
    apply_geom_adjustments!(sys, gc)

    @info "Settling REFINE wing" config.num_steps config.dt total_time=config.num_steps * config.dt

    # Lock tether
    for winch in sys.winches
        winch.brake = true
    end

    # Set steering/depower
    set_steering!(sys, config.steering_pct / 100.0)
    set_depower!(sys, config.depower_pct / 100.0, gc)

    # Heading PID
    heading_pid = DiscretePID(;
        K=config.heading_kp,
        Ti=config.heading_tau_i,
        Td=false,
        Ts=config.dt,
        umin=-1.0, umax=1.0)

    # Logger
    logger, sys_state = create_logger(sam, config.num_steps)

    # Simulation loop
    @info "Starting settling simulation..."
    for step in 1:config.num_steps
        t = step * config.dt

        # Damping decay
        damping = max(config.world_damping *
            (1.0 - step / config.decay_steps),
            config.min_damping)
        SymbolicAWEModels.set_world_frame_damping(
            sys, damping)

        # Heading control
        wing = sys.wings[1]
        R_b_w = SymbolicAWEModels.calc_refine_wing_frame(
            sys.points, wing.z_ref_points,
            wing.y_ref_points, wing.origin_idx)[1]
        curr_heading = calc_heading(sys, R_b_w)
        delta = -wrap_to_pi(
            config.heading_setpoint - curr_heading)
        ctrl = DiscretePIDs.calculate_control!(
            heading_pid, 0.0, delta, 0.0)

        # Apply steering with heading correction
        L_left, L_right = steering_percentage_to_lengths(
            config.steering_pct)
        sys.segments[V3_STEERING_LEFT_IDX].l0 =
            L_left + V3_STEERING_GAIN * ctrl
        sys.segments[V3_STEERING_RIGHT_IDX].l0 =
            L_right - V3_STEERING_GAIN * ctrl

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
    return sam, syslog
end
