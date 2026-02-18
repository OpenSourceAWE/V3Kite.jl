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
    source_struc_path::String = "CORRECT_struc_geometry.yaml"
    source_aero_path::String = "CORRECT_aero_geometry.yaml"
    vsm_settings_path::String = "vsm_settings_reduced_for_coupling.yaml"

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
    te_frac::Float64 = 0.95
    tip_reduction::Float64 = 0.4
    tip_segments::Vector{Int} = [47, 48, 57, 58]
    te_segments::UnitRange{Int} = 20:28

    # Model options
    n_panels::Int = 36
end

"""
    settle_wing(config::V3SettleConfig; data_path=nothing,
                show_progress=true) -> (sam, syslog)

Run a damped settling simulation to find equilibrium wing geometry.

Returns the model and simulation log. The settled positions are
written to destination YAML files derived from the geometry suffix.
"""
function settle_wing(config::V3SettleConfig;
                     data_path=nothing,
                     show_progress=true)
    if isnothing(data_path)
        data_path = v3_data_path()
    end

    suffix = build_geom_suffix(
        V3_DEPOWER_L0, config.tip_reduction, config.te_frac)
    dest_struc = joinpath(
        data_path, "struc_geometry_$(suffix).yaml")
    dest_aero = joinpath(
        data_path, "aero_geometry_$(suffix).yaml")
    source_struc = joinpath(data_path, config.source_struc_path)
    source_aero = joinpath(data_path, config.source_aero_path)

    @info "Settling REFINE wing" config.num_steps config.dt total_time=config.num_steps * config.dt

    # Load settings
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.g_earth = 9.81
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    set.profile_law = 0

    # Load VSM
    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = source_aero
    vsm_set.wings[1].n_panels = config.n_panels

    # Load system structure
    sys = load_sys_struct_from_yaml(source_struc;
        system_name="v3", set,
        wing_type=SymbolicAWEModels.REFINE, vsm_set)

    # Set transform from heading/elevation
    h = config.heading_setpoint
    sys.transforms[1].elevation = deg2rad(
        config.elevation * cos(h))
    sys.transforms[1].azimuth = deg2rad(
        config.elevation * sin(h))
    sys.transforms[1].heading = h

    # Configure tether
    seg_len = config.tether_length / 6 *
        (1 + 1000 / sys.segments[end].unit_stiffness)
    for i in 39:44
        sys.points[i].pos_cad .= [
            0.0, 0.0, -(i - 38) * seg_len]
    end
    for i in 90:95
        sys.segments[i].l0 = seg_len
    end

    # Apply geometry modifications
    for idx in config.tip_segments
        sys.segments[idx].l0 -= config.tip_reduction
    end
    for idx in config.te_segments
        sys.segments[idx].l0 *= config.te_frac
    end

    # Set damping
    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping)
    SymbolicAWEModels.set_body_frame_damping(
        sys, config.body_damping)

    # Create and init model
    sam = SymbolicAWEModel(set, sys)
    SymbolicAWEModels.init!(
        sam; remake=true, ignore_l0=false, remake_vsm=true)

    # Lock tether
    for winch in sys.winches
        winch.brake = true
    end

    # Set steering/depower
    set_steering!(sys, config.steering_pct / 100.0)
    set_depower!(sys, config.depower_pct / 100.0)

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
        wing.R_b_w = SymbolicAWEModels.calc_refine_wing_frame(
            sys.points, wing.z_ref_points,
            wing.y_ref_points, wing.origin_idx)[1]
        curr_heading = calc_heading(sys, wing.R_b_w)
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
