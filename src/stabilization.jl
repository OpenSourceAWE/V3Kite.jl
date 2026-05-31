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
    # Per-point overrides applied AFTER body_damping
    body_damping_overrides::Vector{
        Tuple{UnitRange{Int}, Vector{Float64}}} =
        Tuple{UnitRange{Int}, Vector{Float64}}[]

    # Flight condition
    v_wind::Float64 = 10.72
    tether_length::Float64 = 240.0
    g_earth::Float64 = 9.81
    kcu_mass::Union{Nothing,Float64} = nothing

    # Geometry modifications
    geom::V3GeomAdjustConfig = V3GeomAdjustConfig(
        reduce_tip=true, reduce_te=true,
        tether_length=240.0)

    # Depower ramp
    start_depower::Union{Nothing,Float64} = nothing
    end_depower::Union{Nothing,Float64} = nothing

    course_correction_gain::Float64 = 0.3
    # :course tracks flight-data course (needs nonzero velocity);
    # :heading tracks the init_row's heading (use when settling
    # from rest, e.g. zenith hold).
    course_correction_mode::Symbol = :course

    # Model options
    fix_sphere_idxs::Vector{Int} = Int[]
end

"""
    settle_wing(config::V3SettleConfig, init_row;
                data_path=nothing,
                show_progress=true, remake=false)
    settle_wing(config::V3SettleConfig;
                position, velocity, attitude,
                steering, depower, wind_vec,
                data_path=nothing,
                show_progress=true, remake=false)
    -> (sam, syslog, settle_failed)

Run power-zone settling with gravity to find equilibrium wing
geometry matching the given flight state.

First form takes a prebuilt `init_row` NamedTuple with `x, y, z,
vx, vy, vz, roll, pitch, yaw, steering, depower, wind_vec`.
Second form builds it from explicit ENU vectors (`position`,
`velocity` in m / m·s⁻¹, `attitude` is `[roll, pitch, yaw]` rad).

Always returns a fresh model loaded from the settled binary, so
the caller gets clean settings (no settling damping). When
`remake=false` and the destination file already exists, the
simulation is skipped and the settled geometry is loaded from
file.
"""
function settle_wing(config::V3SettleConfig;
                     position, velocity, heading,
                     steering, depower, wind_vec,
                     kwargs...)
    init_row = (
        x=position[1], y=position[2], z=position[3],
        vx=velocity[1], vy=velocity[2], vz=velocity[3],
        heading=heading,
        steering=steering, depower=depower,
        wind_vec=wind_vec)
    return settle_wing(config, init_row; kwargs...)
end

function settle_wing(config::V3SettleConfig, init_row;
                     data_path=nothing,
                     show_progress=true,
                     remake=false)
    if isnothing(data_path)
        data_path = v3_data_path()
    end

    gc = config.geom
    gc.tether_length = config.tether_length

    dp_reduction = gc.reduce_depower ?
        gc.depower_reduction : 0.0
    st_reduction = gc.reduce_steering ?
        gc.steering_reduction : 0.0
    dp_norm = isnothing(config.end_depower) ?
        init_row.depower : config.end_depower / 100.0
    st_norm = init_row.steering
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
        "_lt$(Int(round(config.tether_length)))" *
        "_g$(Int(round(config.g_earth * 10)))"
    if !isnothing(config.kcu_mass)
        suffix *= "_kcu$(Int(round(config.kcu_mass * 10)))"
    end
    dest_struc = joinpath(
        data_path, "settled_$(suffix).bin")
    source_struc = joinpath(
        data_path, config.source_struc_path)
    source_aero = joinpath(
        data_path, config.source_aero_path)

    # Run settling simulation if needed
    syslog = nothing
    settle_failed = false
    if remake || !isfile(dest_struc)
        try
            syslog = _run_power_zone_settling!(
                config; data_path, show_progress,
                source_struc, source_aero,
                dest_struc, init_row)
        catch err
            is_interrupt = err isa InterruptException ||
                any(e isa InterruptException
                    for (e, _) in current_exceptions())
            if is_interrupt
                @warn "Settling interrupted"
                settle_failed = true
            elseif err isa ErrorException
                @warn "Settling failed" msg=err.msg
                settle_failed = true
            else
                rethrow(err)
            end
            try
                syslog = load_log("settle_particle_dynamics_wing")
            catch
            end
        end
    end

    # Load model from serialized sys_struct, or source
    # YAML if settling failed
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    set.g_earth = config.g_earth
    set.profile_law = 0
    set.wind_vec = KiteUtils.MVec3(init_row.wind_vec)

    if !settle_failed && isfile(dest_struc)
        @info "Loading settled geometry" dest_struc
        sys = deserialize(dest_struc)
        sys.set = set
        sam = SymbolicAWEModel(set, sys)
        SymbolicAWEModels.init!(sam;
            remake=false, remake_vsm=true,
            reinit_sys=false)
    else
        @info "Loading source geometry" source_struc
        vsm_path = joinpath(
            data_path, config.vsm_settings_path)
        vsm_set = VortexStepMethod.VSMSettings(
            vsm_path; data_prefix=false)
        vsm_set.wings[1].geometry_file = source_aero
        sys = load_sys_struct_from_yaml(source_struc;
            system_name=V3_MODEL_NAME, set,
            dynamics_type=SymbolicAWEModels.PARTICLE_DYNAMICS, vsm_set)
        sam = SymbolicAWEModel(set, sys)
        SymbolicAWEModels.init!(sam;
            remake=false, ignore_l0=false,
            remake_vsm=true)
    end

    return sam, syslog, settle_failed
end

"""
Set up a settling model: settings, VSM, sys struct, damping,
SAM creation, geometry adjustments, init, and lock tether.
Returns `(sam, sys, gc)`.
"""
function _setup_settling_model(config::V3SettleConfig;
        data_path, source_struc, source_aero)
    gc = config.geom
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.g_earth = config.g_earth
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    set.profile_law = 0

    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = source_aero

    sys = load_sys_struct_from_yaml(source_struc;
        system_name=V3_MODEL_NAME, set,
        dynamics_type=SymbolicAWEModels.PARTICLE_DYNAMICS, vsm_set)

    if !isnothing(config.kcu_mass)
        sys.points[1].extra_mass = config.kcu_mass
    end

    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping)
    SymbolicAWEModels.set_body_frame_damping(
        sys, config.body_damping)
    for (rng, damp) in config.body_damping_overrides
        SymbolicAWEModels.set_body_frame_damping(
            sys, damp, rng)
    end

    sam = SymbolicAWEModel(set, sys)
    apply_geom_adjustments!(sys, gc)
    sys.tethers[1].init_unstretched_len = gc.tether_length
    sys.tethers[1].init_stretched_len = gc.tether_length
    SymbolicAWEModels.init!(
        sam; remake=false, ignore_l0=false, remake_vsm=true)

    @info "Settling PARTICLE_DYNAMICS wing" config.num_steps config.dt total_time=config.num_steps * config.dt

    for winch in sys.winches
        winch.brake = true
    end

    return sam, sys, gc
end

"""Run power-zone settling initialized from flight data."""
function _run_power_zone_settling!(config::V3SettleConfig;
        data_path, show_progress,
        source_struc, source_aero,
        dest_struc,
        init_row)
    sam, sys, gc = _setup_settling_model(config;
        data_path, source_struc, source_aero)

    update_sys_struct_from_data!(sys, init_row; config=gc)

    data_pos = [init_row.x, init_row.y, init_row.z]
    data_vel = [init_row.vx, init_row.vy, init_row.vz]
    R_t_to_w = SymbolicAWEModels.calc_R_t_to_w(data_pos)
    target_course = atan(
        data_vel ⋅ R_t_to_w[:, 2],
        data_vel ⋅ R_t_to_w[:, 1])
    target_heading = sys.transforms[1].heading
    config.course_correction_mode in (:course, :heading) ||
        error("course_correction_mode must be :course or " *
              ":heading, got $(config.course_correction_mode)")

    # Override initial depower if ramp is configured
    if !isnothing(config.start_depower)
        set_depower!(sys, config.start_depower / 100.0, 0.0, gc)
    end

    SymbolicAWEModels.reinit!(
        sam, sam.prob, SymbolicAWEModels.FBDF())

    if hasproperty(init_row, :wind_vec)
        @assert isapprox(
            sam.set.wind_vec, init_row.wind_vec;
            atol=1e-6) "wind_vec mismatch " *
            "after settle init: " *
            "got $(sam.set.wind_vec), " *
            "expected $(init_row.wind_vec)"
    end

    for idx in config.fix_sphere_idxs
        sys.points[idx].fix_sphere = true
    end

    total_steps = config.num_steps * config.num_substeps
    logger, sys_state = create_logger(sam, total_steps)

    @info "Starting power-zone settling..." num_substeps=config.num_substeps
    wing = sys.wings[1]
    failed = false
    try
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
                frac = (step - 1) /
                    max(config.num_steps - 1, 1)
                dp = config.start_depower +
                    frac * (dp_end - config.start_depower)
                set_depower!(sys, dp / 100.0, 0.0, gc)
            end

            SymbolicAWEModels.reposition!(
                sys.transforms, sys)
            SymbolicAWEModels.reinit!(
                sam, sam.prob, SymbolicAWEModels.FBDF())

            for sub in 1:config.num_substeps
                global_step =
                    (step - 1) * config.num_substeps + sub
                t = global_step * config.dt

                if !sim_step!(sam; dt=config.dt,
                        vsm_interval=1)
                    @error "Simulation failed" step sub t
                    failed = true
                    break
                end

                log_state!(logger, sys_state, sam, t)

                if show_progress &&
                   should_report(global_step, total_steps)
                    @info "Step $step/$(config.num_steps)" substep=sub damping=round.(damping, digits=1) elevation=round(rad2deg(wing.elevation), digits=2) heading=round(rad2deg(wing.heading), digits=2)
                end
            end
            failed && break

            if config.course_correction_mode === :course
                target = target_course
                current = wing.course
            else
                target = target_heading
                current = wing.heading
            end
            diff = wrap_to_pi(target - current)
            delta_heading =
                config.course_correction_gain * diff
            old_heading = sys.transforms[1].heading
            sys.transforms[1].heading = wrap_to_pi(
                old_heading + delta_heading)

            # reposition! does not rotate vel_w
            transform = sys.transforms[1]
            base_pos = sys.points[
                transform.base_point_idx].pos_w
            k = normalize(wing.pos_w - base_pos)
            wing.vel_w .= SymbolicAWEModels.rotate_v_around_k(
                wing.vel_w, k, delta_heading)
            for point in sys.points
                point.transform_idx == transform.idx ||
                    continue
                point.vel_w .=
                    SymbolicAWEModels.rotate_v_around_k(
                        point.vel_w, k, delta_heading)
            end

            if show_progress &&
               should_report(step, config.num_steps) &&
               config.course_correction_mode === :course
                @info "Course correction step $step" target_course=round(rad2deg(target), digits=2) wing_course=round(rad2deg(current), digits=2) course_diff=round(rad2deg(diff), digits=2) old_heading=round(rad2deg(old_heading), digits=2) new_heading=round(rad2deg(sys.transforms[1].heading), digits=2)
            end
        end
    catch err
        if logger.index > 1
            @warn "Settling crashed, saving partial log" msg=sprint(showerror, err)
            save_log(logger, "settle_particle_dynamics_wing")
        end
        rethrow(err)
    end

    # Copy settled world positions into CAD slots so
    # that copy_cad_to_world! during init! restores
    # the settled state exactly.
    for point in sys.points
        point.pos_cad .= point.pos_w
    end
    for wing in sys.wings
        wing.pos_cad .= wing.pos_w
        wing.R_b_to_c .=
            SymbolicAWEModels.quaternion_to_rotation_matrix(
                wing.Q_b_to_w)
    end

    @info "Serializing settled sys_struct..."
    serialize(dest_struc, sys)

    syslog = save_and_load_log(
        logger, "settle_particle_dynamics_wing")
    @info "Settling complete" dest_struc
    return syslog
end
