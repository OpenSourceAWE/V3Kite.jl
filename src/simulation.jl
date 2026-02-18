# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite simulation functions.
Provides high-level functions for creating and running V3 kite simulations.
"""

const V3_MODEL_NAME = "v3"
const V3_QUAT_MODEL_NAME = "v3_quat"

"""
    v3_data_path()

Return the path to the V3 data directory bundled with V3Kite.jl.
"""
function v3_data_path()
    return joinpath(pkgdir(@__MODULE__), "data")
end

"""
    V3SimConfig

Configuration structure for V3 kite simulations.
"""
Base.@kwdef mutable struct V3SimConfig
    # Geometry paths (relative to data_path, or absolute)
    struc_yaml_path::String = "struc_geometry_depower0.0_tip0.4_te0.95.yaml"
    aero_yaml_path::String = "aero_geometry_depower0.0_tip0.4_te0.95.yaml"
    vsm_settings_path::String = "vsm_settings_reduced_for_coupling.yaml"

    # Simulation parameters
    sim_time::Float64 = 60.0
    fps::Int = 60

    # Wind parameters
    v_wind::Float64 = 10.0
    upwind_dir::Float64 = -90.0

    # Control parameters
    up::Float64 = 40.0           # Depower percentage [0, 100]
    us::Float64 = 0.0            # Steering percentage [-100, 100]
    tether_length::Float64 = 250.0
    elevation::Union{Nothing, Float64} = nothing

    # Ramp parameters
    ramp_start_time_up::Float64 = 0.0
    ramp_end_time_up::Float64 = 5.0
    ramp_start_time_us::Float64 = 0.0
    ramp_end_time_us::Float64 = 5.0

    # Damping parameters
    damping_pattern::Vector{Float64} = [0.0, 30.0, 60.0]

    # Model options
    wing_type::SymbolicAWEModels.WingType = SymbolicAWEModels.REFINE
    remake_cache::Bool = false
    n_panels::Int = 36

    # Winch control
    brake::Bool = true
end

export V3SimConfig

"""
    create_v3_model(config::V3SimConfig; data_path=nothing)
    create_v3_model(; kwargs...)

Create a V3 kite SymbolicAWEModel with the given configuration.

# Arguments
- `config::V3SimConfig`: Configuration struct
- `data_path`: Path to V3 data directory (default: V3Kite bundled data)

# Returns
- `(sam, sys)`: Tuple of SymbolicAWEModel and SystemStructure
"""
function create_v3_model(config::V3SimConfig; data_path=nothing)
    # Use bundled V3 data by default
    if isnothing(data_path)
        data_path = v3_data_path()
    end

    wing_type_str = config.wing_type == SymbolicAWEModels.REFINE ? "REFINE" : "QUATERNION"
    @info "Creating V3 kite model" wing_type=wing_type_str data_path struc_yaml=config.struc_yaml_path

    # Load settings
    set_data_path(data_path)
    set = Settings("system.yaml")
    set.v_wind = config.v_wind
    set.upwind_dir = config.upwind_dir
    set.l_tether = config.tether_length
    set.v_reel_outs[1] = 0.0

    # Load VSMSettings
    vsm_set_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(vsm_set_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = joinpath(data_path, config.aero_yaml_path)
    vsm_set.wings[1].n_panels = config.n_panels

    # Determine model name
    model_name = config.wing_type == SymbolicAWEModels.QUATERNION ?
        V3_QUAT_MODEL_NAME : V3_MODEL_NAME

    # Load system structure (use absolute path)
    struc_yaml_full = joinpath(data_path, config.struc_yaml_path)
    sys = load_sys_struct_from_yaml(struc_yaml_full;
        system_name=model_name, set, wing_type=config.wing_type, vsm_set)

    # Initialize damping
    SymbolicAWEModels.set_body_frame_damping(sys, config.damping_pattern, 1:38)

    # Create symbolic model
    sam = SymbolicAWEModel(set, sys)

    # Adjust tether length
    adjust_tether_length!(sam, config.tether_length)

    # Adjust elevation if provided
    if config.elevation !== nothing
        adjust_elevation!(sam, config.elevation)
    end

    return sam, sys
end

function create_v3_model(; kwargs...)
    config = V3SimConfig(; kwargs...)
    return create_v3_model(config)
end

"""
    run_v3_simulation(config::V3SimConfig; show_progress=true)
    run_v3_simulation(; kwargs...)

Run a V3 kite simulation with the given configuration.

# Arguments
- `config::V3SimConfig`: Configuration struct
- `show_progress`: Show progress updates (default: true)

# Returns
- `(sam, syslog, tape_data)`: Tuple of model, log, and tape length data
"""
function run_v3_simulation(config::V3SimConfig; show_progress=true)
    # Create model
    sam, sys = create_v3_model(config)

    # Initialize model
    wing_type_str = config.wing_type == SymbolicAWEModels.REFINE ? "REFINE" : "QUATERNION"
    @info "Initializing $wing_type_str model..."
    SymbolicAWEModels.init!(sam; remake=config.remake_cache, ignore_l0=false, remake_vsm=true)

    # Set winch brake
    sam.sys_struct.winches[1].brake = config.brake

    # Create logger
    n_steps = Int(round(config.fps * config.sim_time))
    dt = config.sim_time / n_steps
    logger = Logger(sam, n_steps + 1)
    sys_state = SysState(sam)
    sys_state.time = 0.0
    log!(logger, sys_state)

    # Store nominal segment lengths
    nominal_l0_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    nominal_l0_right = sys.segments[V3_STEERING_RIGHT_IDX].l0
    nominal_l0_depower = sys.segments[V3_DEPOWER_IDX].l0

    # Calculate target tape lengths
    L_left_target, L_right_target = steering_percentage_to_lengths(config.us)
    L_depower_target = depower_percentage_to_length(config.up)

    # Storage for tape lengths
    tape_times = Float64[]
    tape_steering_pct = Float64[]
    tape_depower_pct = Float64[]

    @info "Starting simulation" n_steps dt wing_type=wing_type_str
    sim_start_time = time()

    for step in 1:n_steps
        t = step * dt

        # Calculate ramp factors
        if t <= config.ramp_start_time_up
            power_ramp = 0.0
        elseif t >= config.ramp_end_time_up
            power_ramp = 1.0
        else
            power_ramp = (t - config.ramp_start_time_up) /
                         (config.ramp_end_time_up - config.ramp_start_time_up)
        end

        if t <= config.ramp_start_time_us
            steering_ramp = 0.0
        elseif t >= config.ramp_end_time_us
            steering_ramp = 1.0
        else
            steering_ramp = (t - config.ramp_start_time_us) /
                            (config.ramp_end_time_us - config.ramp_start_time_us)
        end

        # Apply ramped tape lengths
        sys.segments[V3_STEERING_LEFT_IDX].l0 =
            nominal_l0_left + steering_ramp * (L_left_target - nominal_l0_left)
        sys.segments[V3_STEERING_RIGHT_IDX].l0 =
            nominal_l0_right + steering_ramp * (L_right_target - nominal_l0_right)
        sys.segments[V3_DEPOWER_IDX].l0 =
            nominal_l0_depower + power_ramp * (L_depower_target - nominal_l0_depower)

        # Log tape percentages
        push!(tape_times, t)
        push!(tape_steering_pct, steering_ramp * config.us)
        push!(tape_depower_pct, power_ramp * config.up)

        # Advance simulation
        try
            next_step!(sam; set_values=[0.0], dt=dt, vsm_interval=1)
        catch err
            if err isa AssertionError
                @error "Simulation failed at step $step"
                break
            end
            rethrow(err)
        end

        # Log state
        update_sys_state!(sys_state, sam)
        sys_state.time = t
        log!(logger, sys_state)

        # Progress updates
        if show_progress && (step % max(1, div(n_steps, 10)) == 0 || step == n_steps)
            elapsed = time() - sim_start_time
            times_realtime = t / elapsed
            @info "  Step $step/$n_steps (t = $(round(t, digits=2)) s)" times_realtime=round(times_realtime, digits=2)
        end
    end

    # Report performance
    total_wall_time = time() - sim_start_time
    final_times_realtime = config.sim_time / total_wall_time
    @info "Simulation completed" wall_time=round(total_wall_time, digits=2) times_realtime=round(final_times_realtime, digits=2)

    # Save and load log
    log_name = "v3_sim_$(lowercase(wing_type_str))_$(Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS"))"
    save_log(logger, log_name)
    syslog = load_log(log_name)

    # Create tape data
    tape_data = (
        time = tape_times,
        steering = tape_steering_pct,
        depower = tape_depower_pct
    )

    return sam, syslog, tape_data
end

function run_v3_simulation(; kwargs...)
    config = V3SimConfig(; kwargs...)
    return run_v3_simulation(config)
end
