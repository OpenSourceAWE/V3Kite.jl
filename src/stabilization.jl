# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Wing settling/stabilization functions.
Runs a damped simulation to find equilibrium wing geometry and writes
the settled positions back to YAML files.
"""

"""
    MIN_SETTLE_DAMPING_PER_STIFFNESS

Lowest tether/bridle `damping_per_stiffness` [s] the settling run stays stable
at. Settling is a violent transient and needs the structural damping that the
settled model can then fly without: 0.0015 settles, 0.0014 and below diverge
(measured at `body_damping = [0, 0, 40]`, `min_damping = [0, 0, 32]`, 40 steps).
"""
const MIN_SETTLE_DAMPING_PER_STIFFNESS = 0.0015

"""
    V3SettleConfig

Configuration for wing settling simulation.
"""
Base.@kwdef mutable struct V3SettleConfig
    # Geometry paths
    source_struc_path::String = "struc_geometry.yaml"
    source_aero_path::String = "aero_geometry.yaml"
    vsm_settings_path::String = "vsm_settings.yaml"
    # System YAML pointing at the active settings file (loaded via `Settings`)
    system_yaml::String = "system.yaml"

    # Simulation parameters
    num_steps::Int = 1600
    num_substeps::Int = 1
    dt::Float64 = 0.05

    # Damping
    world_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 0.0]
    # Body-frame damping decays linearly over `decay_steps`, floored elementwise
    # at `min_damping`; world-frame damping decays all the way to zero
    min_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 20.0]
    decay_steps::Int = 400
    body_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 40.0]
    # Per-point overrides applied AFTER body_damping
    body_damping_overrides::Vector{
        Tuple{UnitRange{Int}, Vector{Float64}}} =
        Tuple{UnitRange{Int}, Vector{Float64}}[]
    # Structural damping of the tether and bridle segments, as a ratio of their
    # stiffness (see `set_damping_per_stiffness!`). Applied from the start of
    # settling, but never below `min_settle_damping_per_stiffness`; `nothing`
    # keeps the values loaded from `struc_geometry.yaml`.
    damping_per_stiffness::Union{Nothing, Float64} = nothing
    # Floor on the ratio DURING SETTLING only: below it the settling transient
    # diverges. `init` applies the unfloored ratio to the settled structure.
    min_settle_damping_per_stiffness::Float64 =
        MIN_SETTLE_DAMPING_PER_STIFFNESS

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
    aero_mode::SymbolicAWEModels.AbstractAeroModel =
        SymbolicAWEModels.AeroDirect()
    fix_sphere_idxs::Vector{Int} = Int[]
end

"""
    settle_damping_per_stiffness(config) -> Union{Nothing, Float64}

The tether/bridle `damping_per_stiffness` settling actually runs at:
`config.damping_per_stiffness` raised to
`config.min_settle_damping_per_stiffness`, or `nothing` when the ratio is not
set at all. The flown value can be lower — [`init`](@ref) applies it to the
settled structure, where there is no transient left to destabilize.
"""
function settle_damping_per_stiffness(config::V3SettleConfig)
    isnothing(config.damping_per_stiffness) && return nothing
    return max(config.damping_per_stiffness,
               config.min_settle_damping_per_stiffness)
end

"""
    settled_state_path(config, init_row; data_path=nothing,
                       cache_path=nothing) -> String

Path of the settled-state log for `config` and `init_row`.
Deterministic from the geometry adjustments, depower/steering, wind
speed, tether length, damping, the elevation implied by `init_row`
and gravity, so the same flight state always maps to the same file.
Settings are read from `data_path`; the file lives under
`cache_path` (default [`default_cache_path`](@ref)`(data_path)`).

The tether/bridle damping enters the name as well (as `_dps`, in
units of 1e-3 s), because it is applied from the start of settling
and so shapes the transient the settled state converges along.
What enters is [`settle_damping_per_stiffness`](@ref), the floored
value settling ran at, so every flown ratio below the floor shares
one settled state.

The aerodynamics enter the name because a structure settled under one
mode is not equilibrium under another.
"""
function settled_state_path(config::V3SettleConfig, init_row;
                            data_path=nothing, cache_path=nothing)
    if isnothing(data_path)
        data_path = v3_data_path()
    end
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))
    gc = config.geom
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
    el_deg = rad2deg(KiteUtils.calc_elevation(
        [init_row.x, init_row.y, init_row.z]))
    suffix *= "_vapp$(round(config.v_wind, digits=2))" *
        "_lt$(Int(round(config.tether_length)))" *
        "_el$(num_tag(round(el_deg, digits=1)))" *
        "_g$(Int(round(config.g_earth * 10)))" *
        "_sys$(splitext(basename(config.system_yaml))[1])"
    yaml_kcu_mass = Settings(joinpath(data_path, config.system_yaml)).kcu_mass
    resolved_kcu_mass = !isnothing(config.kcu_mass) ? config.kcu_mass :
        (yaml_kcu_mass != 0 ? yaml_kcu_mass : nothing)
    if !isnothing(resolved_kcu_mass)
        suffix *= "_kcu$(Int(round(resolved_kcu_mass * 10)))"
    end
    suffix *= "_bd$(num_tag(config.body_damping))"
    if !all(iszero, config.world_damping) || !all(iszero, config.min_damping)
        suffix *= "_wd$(num_tag(config.world_damping))" *
                  "_md$(num_tag(config.min_damping))"
    end
    settle_dps = settle_damping_per_stiffness(config)
    if !isnothing(settle_dps)
        # In units of 1e-3 s: `num_tag` rounds to 3 digits, which would map the
        # whole usable range of this ratio onto one or two tags.
        suffix *= "_dps$(num_tag(settle_dps * 1e3))"
    end
    aero_tag = SymbolicAWEModels.aero_mode_tag(config.aero_mode)
    aero_tag == DEFAULT_AERO_TAG || (suffix *= "_aero$(aero_tag)")
    return joinpath(cache_path, "settled_$(suffix).arrow")
end

"""
    save_settled_state(sam, path)

Write `sam`'s state to `path` as a one-row `Float64` log — everything the
integrator needs to restart, and nothing else. `Float64` because a `Float32`
state does not reproduce `integrator.u` on a bridle this stiff.
"""
function save_settled_state(sam, path)
    logger = Logger(sam, 1; precision=Float64)
    log!(logger, SysState(sam; precision=Float64))
    name, dir = basename(path), dirname(path)
    save_log(logger, splitext(name)[1], false; path=dir)
    return path
end

"""
    apply_settled_damping!(sys, config)

Re-apply the damping the settling loop ended on: world-frame damping decays to
zero over `config.decay_steps` and body-frame damping down to
`config.min_damping`. Both follow from `config`, so they are recomputed rather
than stored with the state.
"""
function apply_settled_damping!(sys, config::V3SettleConfig)
    decay = max(0.0, 1.0 - config.num_steps / config.decay_steps)
    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping .* decay)
    SymbolicAWEModels.set_body_frame_damping(
        sys, max.(config.body_damping .* decay, config.min_damping))
    return sys
end

"""
    apply_settled_state!(sys, config, path) -> Bool

Restore the settled state at `path` into `sys` and re-apply the settled
damping. Returns `false` when the file is missing or unreadable, so the caller
can fall back to the source geometry rather than crash.
"""
function apply_settled_state!(sys, config::V3SettleConfig, path)
    isfile(path) || return false
    log = try
        load_log(path)
    catch err
        err isa InterruptException && rethrow()
        @warn "Settled state unreadable, using source geometry" path err
        return false
    end
    isempty(log.syslog) && return false
    update_from_sysstate!(sys, log.syslog[1])
    apply_settled_damping!(sys, config)
    return true
end

"""
    load_settled_struct(config, init_row;
                        data_path=nothing, cache_path=nothing,
                        set=nothing)

Rebuild the settling `SystemStructure` from the source YAML and restore the
settled state logged for `config`/`init_row` (see
[`settled_state_path`](@ref)) onto it, assigning `set` when given. Errors when
the state is missing; run [`settle_wing`](@ref) first to create it.
"""
function load_settled_struct(config::V3SettleConfig, init_row;
                             data_path=nothing, cache_path=nothing,
                             set=nothing)
    isnothing(data_path) && (data_path = v3_data_path())
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))
    path = settled_state_path(config, init_row; data_path, cache_path)
    sys, yaml_set = build_settling_struct(config; data_path,
        source_struc = joinpath(data_path, config.source_struc_path),
        source_aero = joinpath(data_path, config.source_aero_path))
    apply_settled_state!(sys, config, path) ||
        error("No settled state at $path; run settle_wing first")
    sys.set = isnothing(set) ? yaml_set : set
    return sys
end

"""
    settle_wing(config::V3SettleConfig, init_row;
                data_path=nothing, cache_path=nothing,
                show_progress=true, remake=false)
    settle_wing(config::V3SettleConfig;
                position, velocity, attitude,
                steering, depower, wind_vec,
                data_path=nothing, cache_path=nothing,
                show_progress=true, remake=false)
    -> (sam, syslog, settle_failed)

Run power-zone settling with gravity to find equilibrium wing
geometry matching the given flight state.

First form takes a prebuilt `init_row` NamedTuple with `x, y, z,
vx, vy, vz, roll, pitch, yaw, steering, depower, wind_vec`.
Second form builds it from explicit ENU vectors (`position`,
`velocity` in m / m·s⁻¹, `attitude` is `[roll, pitch, yaw]` rad).

Always returns a fresh model loaded from the settled binary, so
the caller gets clean settings — clean meaning the *world*-frame
damping, which is only a settling aid and decays away. The
body-frame damping is a per-point field of the serialized
`SystemStructure` and therefore carries over into the returned
model on purpose; it is part of the cache key for that reason.

When `remake=false` and the destination file already exists, the
simulation is skipped and the settled geometry is loaded from
file. The cache file name encodes the settling inputs, including
the elevation implied by the initial position and a non-default
`aero_mode`, so runs that only differ in elevation or in
aerodynamics get their own file instead of sharing one. A cached
geometry whose aerodynamics disagree with `config.aero_mode` is
rejected and re-derived from the source YAML.

`data_path` is where the source geometry/settings YAMLs are read
from (default [`v3_data_path`](@ref)); `cache_path` is where
everything generated is written — the `settled_*.arrow` state, the
settling log, and the serialized model binary. It defaults to
[`default_cache_path`](@ref)`(data_path)`, which is `data_path`
for a development checkout and a depot scratch directory for an
installed V3Kite, and is created if missing. See [`init`](@ref)
and [`with_model_cache`](@ref).
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

"""
    V3RelaxConfig

Settings for [`relax_bridle!`](@ref). The defaults reach full stiffness on the
beam geometry in under a hundred steps.
"""
Base.@kwdef mutable struct V3RelaxConfig
    start_scale::Float64 = 1.0e-4
    growth::Float64 = 1.35
    settled_tol::Float64 = 50.0
    world_damping::Float64 = 200.0
    dt::Float64 = 0.02
    max_steps::Int = 600
    "Steps held at full stiffness while the damping decays back to its own value."
    hold_steps::Int = 200
    "VSM solve interval during the hold phase; 0 keeps the aero frozen."
    hold_vsm_interval::Int = 1
end

"""
    segment_stiffness_baseline(sys) -> (unit_stiffness, unit_damping)

Every segment's stiffness and damping, captured before a relaxation scales them.
Errors on a callable force law, which has no scalar to scale.
"""
function segment_stiffness_baseline(sys)
    stiffness = [segment.unit_stiffness for segment in sys.segments]
    all(value -> value isa Real, stiffness) || error(
        "relax_bridle!: a segment carries a callable unit_stiffness, which " *
        "cannot be scaled; relax the geometry with linear springs instead.")
    return stiffness, [segment.unit_damping for segment in sys.segments]
end

"""
    scale_segment_stiffness!(sys, baseline, scale)

Set every segment's stiffness and damping to `scale` times its baseline.
"""
function scale_segment_stiffness!(sys, baseline, scale)
    stiffness, damping = baseline
    for (idx, segment) in enumerate(sys.segments)
        segment.unit_stiffness = stiffness[idx] * scale
        segment.unit_damping = damping[idx] * scale
    end
    return nothing
end

"""
    relax_residual(sam) -> Float64

The largest absolute entry of the right-hand side at the current state, i.e. how
far the structure is from equilibrium in m/s² (and rad/s²).
"""
function relax_residual(sam)
    integ = sam.integrator
    du = similar(integ.u)
    integ.f(du, integ.u, integ.p, integ.t)
    return maximum(abs, du)
end

"""
    relax_bridle!(sam, sys, config=V3RelaxConfig(); prn=true)
        -> (reached_scale, steps, residual)

Settle a geometry whose bridle rest lengths disagree with its node positions,
by integrating it with every segment stiffness scaled down to
`config.start_scale` of nominal and handed back only as the structure settles.

The measured V3 bridle lengths and the measured node coordinates come from
different sources and are not consistent: several lines start at more than 100 %
strain, which puts the initial accelerations near 5·10⁷ m/s² and leaves the
implicit solver unable to complete even one step. Softening the springs brings
that into a range it can integrate; the ramp is gated on the residual falling
below `config.settled_tol` rather than run on a fixed schedule, because raising
the stiffness before the knots have moved just puts the strain energy back.
`config.world_damping` is applied to every point for the duration so the knots
settle instead of ringing, and is removed again before returning.

Call after `init!`. Returns the scale actually reached (1.0 on success), the
number of steps taken and the final residual.
"""
function relax_bridle!(sam, sys, config::V3RelaxConfig=V3RelaxConfig();
                       prn::Bool=true)
    baseline = segment_stiffness_baseline(sys)
    saved_damping = [copy(point.world_frame_damping) for point in sys.points]
    SymbolicAWEModels.set_world_frame_damping(
        sys, fill(config.world_damping, 3))

    scale = config.start_scale
    scale_segment_stiffness!(sys, baseline, scale)
    steps = 0
    residual = NaN
    while scale < 1.0 && steps < config.max_steps
        steps += 1
        try
            next_step!(sam; dt=config.dt, vsm_interval=0)
        catch exception
            prn && @warn "relax_bridle! stopped" steps scale exception
            break
        end
        residual = relax_residual(sam)
        residual < config.settled_tol || continue
        scale = min(1.0, scale * config.growth)
        scale_segment_stiffness!(sys, baseline, scale)
    end

    scale_segment_stiffness!(sys, baseline, 1.0)
    if scale >= 1.0
        for hold in 1:config.hold_steps
            decay = 1.0 - hold / config.hold_steps
            for (idx, point) in enumerate(sys.points)
                point.world_frame_damping .=
                    max.(saved_damping[idx], config.world_damping * decay)
            end
            try
                next_step!(sam; dt=config.dt,
                    vsm_interval=config.hold_vsm_interval)
            catch exception
                prn && @warn "relax_bridle! hold phase stopped" hold exception
                break
            end
            steps += 1
        end
    end
    for (idx, point) in enumerate(sys.points)
        point.world_frame_damping .= saved_damping[idx]
    end
    residual = relax_residual(sam)
    prn && @info "Bridle relaxed" scale steps residual
    return scale, steps, residual
end

"""
    num_tag(d) -> String

Filename-safe tag for a number, scalar or per-axis vector:
`num_tag([0.0, 0.0, 40.0]) == "0-0-40"`. Used to put damping coefficients and
the settling elevation into the settled-geometry cache key.
"""
num_tag(d::Real) = replace(string(round(Float64(d), digits=3)), r"\.0$" => "")
num_tag(d::AbstractVector) = join(num_tag.(d), "-")

"""
    default_cache_path(data_path) -> String

Where generated artifacts go when the caller names no `cache_path`: `data_path`
itself for a development checkout, and a scratch directory next to the depot for
a Pkg-INSTALLED V3Kite.

A package directory under `DEPOT_PATH/packages` is not ours to write to: it is
usually read-only, and `Pkg.gc` deletes the whole tree once no environment
references that version. A `scratchspaces` directory keyed by our UUID survives
reinstalling. A development checkout keeps caching in place, so `] dev` behaves
as before and existing `data/settled_*.arrow` states stay in use.
"""
function default_cache_path(data_path)
    abs_data = abspath(data_path)
    in_depot = any(DEPOT_PATH) do depot
        # Trailing separator: a bare prefix also matches a sibling `packages_old`.
        startswith(abs_data, joinpath(abspath(joinpath(depot, "packages")), ""))
    end
    in_depot || return data_path
    uuid = "4caac9c8-c726-438f-ab10-3553e918eab1"  # V3Kite, see Project.toml
    return joinpath(DEPOT_PATH[1], "scratchspaces", uuid, "v3kite_cache")
end

"""
    with_model_cache(f, cache_path)

Run `f()` with KiteUtils' data path pointed at `cache_path`, restoring whatever
the caller had afterwards.

Exists for exactly one write: `SymbolicAWEModels.init!` serializes the compiled
model under `KiteUtils.get_data_path()` and takes no path argument, so the global
data path is the only lever on where it lands. Without this the model binary
would still go to a read-only `data_path` while the geometry went to the cache.
Narrow on purpose: `init!` reads nothing else through the data path, so the
redirect cannot pull another file from the wrong directory.
"""
function with_model_cache(f, cache_path)
    mkpath(cache_path)
    previous = get_data_path()
    set_data_path(cache_path)
    try
        return f()
    finally
        set_data_path(previous)
    end
end

function settle_wing(config::V3SettleConfig, init_row;
                     data_path=nothing,
                     cache_path=nothing,
                     show_progress=true,
                     remake=false)
    if isnothing(data_path)
        data_path = v3_data_path()
    end
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))

    gc = config.geom
    gc.tether_length = config.tether_length

    cache_path != data_path && mkpath(cache_path)
    dest_struc = settled_state_path(
        config, init_row; data_path, cache_path)
    source_struc = joinpath(
        data_path, config.source_struc_path)
    source_aero = joinpath(
        data_path, config.source_aero_path)

    # Run settling simulation if needed
    syslog = nothing
    settle_failed = false
    if remake || !isfile(dest_struc)
        try
            syslog = run_power_zone_settling!(
                config; data_path, show_progress,
                source_struc, source_aero, dest_struc,
                log_path = cache_path, cache_path, init_row)
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
                syslog = load_log("settle_particle_dynamics_wing"; path=cache_path)
            catch
            end
        end
    end

    # Load model from the settled state, or source
    # YAML if settling failed
    set = Settings(joinpath(data_path, config.system_yaml))
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    set.g_earth = config.g_earth
    # profile_law is taken from settings.yaml (loaded via Settings above).
    set.wind_vec = KiteUtils.MVec3(init_row.wind_vec)

    # `nothing` until a cached state has actually been read back: the file
    # existing is not enough.
    sys = nothing
    if !settle_failed && isfile(dest_struc)
        @info "Loading settled state" dest_struc
        candidate, _ = build_settling_struct(config;
            data_path, source_struc, source_aero)
        apply_settled_state!(candidate, config, dest_struc) &&
            (sys = candidate)
    end

    if !isnothing(sys)
        sys.set = set
        sam = SymbolicAWEModel(set, sys)
        # Tape rest lengths are parameters, not state, so the log does not carry
        # them; the rebuilt structure needs them applied as settling did.
        apply_geom_adjustments!(sys, gc)
        sys.tethers[1].init_stretched_len = gc.tether_length
        with_model_cache(cache_path) do
            SymbolicAWEModels.init!(sam;
                remake=false, remake_vsm=true,
                reinit_sys=false)
        end
    else
        @info "Loading source geometry" source_struc
        vsm_path = joinpath(
            data_path, config.vsm_settings_path)
        vsm_set = VortexStepMethod.VSMSettings(
            vsm_path; data_prefix=false)
        vsm_set.wings[1].geometry_file = source_aero
        sys = load_sys_struct_from_yaml(source_struc;
            system_name=V3_MODEL_NAME, set,
            dynamics_type=SymbolicAWEModels.PARTICLE_DYNAMICS, vsm_set,
            aero_mode=config.aero_mode)
        sam = SymbolicAWEModel(set, sys)
        with_model_cache(cache_path) do
            SymbolicAWEModels.init!(sam;
                remake=false, ignore_l0=false,
                remake_vsm=true)
        end
    end

    return sam, syslog, settle_failed
end

"""
    build_settling_struct(config; data_path, source_struc, source_aero)

Build the settling `SystemStructure` from the source YAML: settings, VSM, the
structure itself, the resolved KCU mass and the starting damping. Returns
`(sys, set)`. Shared by [`setup_settling_model`](@ref) and the settled-state
cache, which rebuilds the same structure before restoring a state onto it.
"""
function build_settling_struct(config::V3SettleConfig;
        data_path, source_struc, source_aero)
    set = Settings(joinpath(data_path, config.system_yaml))
    set.g_earth = config.g_earth
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    # profile_law is taken from settings.yaml (loaded via Settings above).

    vsm_path = joinpath(data_path, config.vsm_settings_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = source_aero

    sys = load_sys_struct_from_yaml(source_struc;
        system_name=V3_MODEL_NAME, set,
        dynamics_type=SymbolicAWEModels.PARTICLE_DYNAMICS, vsm_set,
        aero_mode=config.aero_mode)

    # Explicit `config.kcu_mass` (used by parameter sweeps) takes priority;
    # otherwise fall back to the `kcu_mass` field of the active settings YAML
    # (0 means "not set", i.e. keep the geometry-file default).
    kcu_mass = !isnothing(config.kcu_mass) ? config.kcu_mass :
        (set.kcu_mass != 0 ? set.kcu_mass : nothing)
    if !isnothing(kcu_mass)
        sys.points[1].extra_mass = kcu_mass
    end

    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_damping)
    set_body_frame_damping!(sys, config.body_damping)
    for (rng, damp) in config.body_damping_overrides
        SymbolicAWEModels.set_body_frame_damping(
            sys, damp, rng)
    end
    settle_dps = settle_damping_per_stiffness(config)
    if !isnothing(settle_dps)
        set_damping_per_stiffness!(sys, tether_bridle_segments(sys),
                                   settle_dps)
    end
    SymbolicAWEModels.set_body_frame_damping(sys, config.body_damping)
    return sys, set
end

"""
Set up a settling model: settings, VSM, sys struct, damping,
SAM creation, geometry adjustments, init, and lock tether.
Returns `(sam, sys, gc)`.
"""
function setup_settling_model(config::V3SettleConfig;
        data_path, source_struc, source_aero, cache_path=data_path)
    gc = config.geom
    sys, set = build_settling_struct(config;
        data_path, source_struc, source_aero)

    sam = SymbolicAWEModel(set, sys)
    apply_geom_adjustments!(sys, gc)
    sys.tethers[1].init_stretched_len = gc.tether_length
    with_model_cache(cache_path) do
        SymbolicAWEModels.init!(
            sam; remake=false, ignore_l0=false, remake_vsm=true)
    end

    @info "Settling PARTICLE_DYNAMICS wing" config.num_steps config.dt total_time=config.num_steps * config.dt

    for winch in sys.winches
        winch.brake = true
    end

    return sam, sys, gc
end

"""Run power-zone settling initialized from flight data."""
function run_power_zone_settling!(config::V3SettleConfig;
        data_path, show_progress,
        source_struc, source_aero,
        dest_struc, cache_path=data_path, log_path=cache_path,
        init_row)
    sam, sys, gc = setup_settling_model(config;
        data_path, source_struc, source_aero, cache_path)

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
            decay = max(0.0, 1.0 - step / config.decay_steps)
            decayed(x) = max.(x .* decay, config.min_damping)
            damping = decayed(config.body_damping)
            SymbolicAWEModels.set_world_frame_damping(
                sys, config.world_damping .* decay)
            SymbolicAWEModels.set_body_frame_damping(sys, damping)

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
                sam, sam.prob, SymbolicAWEModels.FBDF(); prn=false)

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
            save_log(logger, "settle_particle_dynamics_wing"; path=log_path)
        end
        rethrow(err)
    end

    # A diverged run must not be cached: the resulting geometry fails the VSM
    # solve on reload, and because the cache is keyed on the *inputs* it would be
    # reused on every later run until someone deletes the file by hand.
    # `settle_wing` turns this into `settle_failed = true`.
    failed && error("Settling diverged before completing " *
                    "$(config.num_steps) steps; state not saved")

    # Final placement on target elev/azim/heading; the loop ends on a drifted sim_step!.
    SymbolicAWEModels.reposition!(sys.transforms, sys)

    @info "Saving settled state..."
    save_settled_state(sam, dest_struc)

    syslog = save_and_load_log(
        logger, "settle_particle_dynamics_wing"; path=log_path)
    @info "Settling complete" dest_struc
    return syslog
end
