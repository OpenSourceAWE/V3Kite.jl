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

Schedule of a wing settling run: how long it integrates, how its damping decays,
and how the depower ramps. What is settled and at what flight condition does not
live here — the geometry comes from the project file, the flight condition from
its `sim_settings:` and the model assembly from [`V3KiteConfig`](@ref) — because
the schedule is per-example while those are per-kite.

Loaded from the `settle_settings:` file of a project by
[`V3SettleConfig(filename)`](@ref).
"""
Base.@kwdef mutable struct V3SettleConfig
    "Project file the geometry, settings and kite come from"
    project::String = "system_psm.yaml"
    kite_set::V3KiteConfig = V3KiteConfig()

    num_steps::Int = 1600
    num_substeps::Int = 1
    dt::Float64 = 0.05

    """
    Damping the settling transient starts at. Both decay linearly over
    `decay_steps` and are floored elementwise at the sim damping the kite
    carries (`kite_set.body_sim_damping`, `kite_set.world_sim_damping`), so the schedule
    only says how the transient is killed and the kite file alone says what the
    returned model flies with.
    """
    world_start_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 0.0]
    decay_steps::Int = 400
    body_start_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 40.0]
    # Per-point overrides applied AFTER body_start_damping
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

    """
    The same for the rigid bodies, floored at the damping they fly with. Zero by
    default: only a beam wing has bodies that carry mass and are not the whole kite.
    `beam_angular_start_damping` ramps the bodies' spin damping rather than their
    translation.
    """
    beam_body_start_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 0.0]
    beam_world_start_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 0.0]
    beam_angular_start_damping::Union{Float64, Vector{Float64}} = [0.0, 0.0, 0.0]

    """
    Flight condition to settle at. These stay runtime values rather than moving
    to the settings YAML because a replay takes them from the row it settles
    onto — `v_wind` is that row's apparent wind, not a configured wind speed.
    """
    v_wind::Float64 = 10.72
    tether_length::Float64 = 240.0
    g_earth::Float64 = 9.81
    "Overrides the `kcu_mass` of the settings YAML; `nothing` keeps it"
    kcu_mass::Union{Nothing,Float64} = nothing

    "Depower ramped over the settling steps; `nothing` holds the row's depower"
    start_depower::Union{Nothing,Float64} = nothing
    end_depower::Union{Nothing,Float64} = nothing

    course_correction_gain::Float64 = 0.3
    """
    `:course` tracks the flight-data course and needs a nonzero velocity;
    `:heading` tracks the row's heading, for settling from rest.
    """
    course_correction_mode::Symbol = :course

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
mode is not equilibrium under another, and the source geometry enters it
because a state logged for one has the wrong number of points for
another. Both are left out at their default so that files written before
the key knew about them keep being found.
"""
function settled_state_path(config::V3SettleConfig, init_row;
                            data_path=nothing, cache_path=nothing)
    data_path = project_data_path(config.project, data_path)
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))
    gc = config.kite_set.geom
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
        "_sys$(splitext(basename(config.project))[1])"
    yaml_kcu_mass = Settings(project_path(config.project; data_path)).kcu_mass
    resolved_kcu_mass = !isnothing(config.kcu_mass) ? config.kcu_mass :
        (yaml_kcu_mass != 0 ? yaml_kcu_mass : nothing)
    if !isnothing(resolved_kcu_mass)
        suffix *= "_kcu$(Int(round(resolved_kcu_mass * 10)))"
    end
    suffix *= "_bd$(num_tag(config.body_start_damping))"
    sim_damping = config.kite_set.body_sim_damping
    if !all(iszero, config.world_start_damping) || !all(iszero, sim_damping)
        suffix *= "_wd$(num_tag(config.world_start_damping))" *
                  "_md$(num_tag(sim_damping))"
    end
    settle_dps = settle_damping_per_stiffness(config)
    if !isnothing(settle_dps)
        # In units of 1e-3 s: `num_tag` rounds to 3 digits, which would map the
        # whole usable range of this ratio onto one or two tags.
        suffix *= "_dps$(num_tag(settle_dps * 1e3))"
    end
    kite_set = config.kite_set
    beam_tag = vcat(beam_damping(kite_set.beam_body_damping, kite_set.body_sim_damping),
        kite_set.beam_world_damping, kite_set.beam_angular_damping,
        config.beam_body_start_damping, config.beam_world_start_damping,
        config.beam_angular_start_damping)
    all(iszero, beam_tag) || (suffix *= "_bb$(num_tag(beam_tag))")
    scale = kite_set.beam_joint_damping_scale
    scale == 1.0 || (suffix *= "_jd$(num_tag(scale))")
    aero_mode = resolve_aero_mode(config.kite_set)
    if !isnothing(aero_mode)
        aero_tag = SymbolicAWEModels.aero_mode_tag(aero_mode)
        aero_tag == DEFAULT_AERO_TAG || (suffix *= "_aero$(aero_tag)")
    end
    struc_tag = splitext(basename(
        project_entry(config.project, "structural_geometry"; data_path)))[1]
    struc_tag == DEFAULT_STRUC_TAG || (suffix *= "_$(struc_tag)")
    return joinpath(cache_path, "settled_$(suffix).arrow")
end

"""
    save_state_log(sam, path)

Write `sam`'s state to `path` as a one-row `Float64` log — everything the
integrator needs to restart, and nothing else. `Float64` because a `Float32`
state does not reproduce `integrator.u` on a bridle this stiff.
"""
function save_state_log(sam, path)
    logger = Logger(sam, 1; precision=Float64)
    log!(logger, SysState(sam; precision=Float64))
    name, dir = basename(path), dirname(path)
    save_log(logger, splitext(name)[1], false; path=dir)
    return path
end

"""
    read_state_log(path) -> Union{Nothing, SysState}

The single state written by [`save_state_log`](@ref), or `nothing` when `path` is
missing, unreadable or empty, so a caller can fall back to the placed geometry
rather than crash.
"""
function read_state_log(path)
    isfile(path) || return nothing
    log = try
        load_log(path)
    catch err
        err isa InterruptException && rethrow()
        @warn "State log unreadable, using the placed geometry" path err
        return nothing
    end
    return isempty(log.syslog) ? nothing : log.syslog[1]
end

"""
    reinit_integrator!(sam; prn=true)

Reset `sam`'s integrator from the current `SystemStructure`, reusing the solver
`init!` built it with. `SymbolicAWEModels.reinit!` takes that solver as a required
argument, and a bare `FBDF()` differentiates forward, which makes the monolith
backend compile its right-hand side a second time at `ForwardDiff.Dual`.
"""
reinit_integrator!(sam; prn=true) =
    SymbolicAWEModels.reinit!(sam, sam.prob, sam.integrator.alg; prn)

"""
    start_from_state!(sam, sys, path) -> Bool

Restore the state logged at `path` onto `sys` and push it onto `sam`'s
integrator, so a run starts where that log left off. Returns `false` when the log
is missing or unreadable.

Call after `init!`, not before: the log carries positions and velocities, and the
rest lengths they belong with are the ones `init!` computes, so restoring first
and skipping the recompute would pair a relaxed geometry with the rest lengths of
the YAML instead.
"""
function start_from_state!(sam, sys, path)
    state = read_state_log(path)
    isnothing(state) && return false
    update_from_sysstate!(sys, state)
    reinit_integrator!(sam)
    return true
end

"""
    relaxed_state_name(struc_yaml, depower) -> String

Log name, without extension, of the relaxed state of `struc_yaml` at `depower`
(a fraction): `relaxed_struc_geometry_beam_dp20`. The relaxation example writes
it and `V3KiteConfig.init_state` reads it, so the name lives here rather than in
both.

Only the depower is in the name because it alone changes the shape the bridle
relaxes into. The state is world-frame, so it is saved at the elevation and
tether length the relaxation ran at, and settling repositions it from there.
"""
relaxed_state_name(struc_yaml, depower) =
    "relaxed_$(splitext(basename(struc_yaml))[1])_dp$(num_tag(depower * 100))"

"""
    beam_damping(beam, point) -> Vector{Float64}

The beam bodies' body-frame damping: `beam` when the kite settings give one,
otherwise the `point` damping, which both resolve on the wing's own axes.
"""
beam_damping(beam, point) = isnothing(beam) ? copy(point) : beam

"""
    beam_body_idxs(sys) -> Vector{Int}

Indices of the structural bodies of `sys`, skipping the wings. A wing is a `Body`
too, and its angular damping is the wing's own, so the beam settings must not
reach it.
"""
beam_body_idxs(sys) =
    [idx for (idx, body) in enumerate(sys.bodies)
     if !SymbolicAWEModels.is_wing(body)]

"""
    apply_decayed_damping!(sys, config, decay)

Set the point and body damping `decay` of the way along the settling ramp: each
start value scaled by `decay`, floored elementwise at the damping the kite flies
with.
"""
function apply_decayed_damping!(sys, config::V3SettleConfig, decay)
    kite_set = config.kite_set
    SymbolicAWEModels.set_world_frame_damping(sys,
        max.(config.world_start_damping .* decay, kite_set.world_sim_damping))
    set_body_frame_damping!(sys,
        max.(config.body_start_damping .* decay, kite_set.body_sim_damping))
    beam_idxs = beam_body_idxs(sys)
    flight_body = beam_damping(kite_set.beam_body_damping, kite_set.body_sim_damping)
    SymbolicAWEModels.set_world_frame_damping(sys.bodies,
        max.(config.beam_world_start_damping .* decay, kite_set.beam_world_damping),
        beam_idxs)
    SymbolicAWEModels.set_body_frame_damping(sys.bodies,
        max.(config.beam_body_start_damping .* decay, flight_body), beam_idxs)
    SymbolicAWEModels.set_angular_damping(sys.bodies,
        max.(config.beam_angular_start_damping .* decay, kite_set.beam_angular_damping),
        beam_idxs)
    return sys
end

"""
    apply_settled_damping!(sys, config)

Re-apply the damping the settling loop ended on. It follows from `config`, so it
is recomputed rather than stored with the state.
"""
apply_settled_damping!(sys, config::V3SettleConfig) = apply_decayed_damping!(
    sys, config, max(0.0, 1.0 - config.num_steps / config.decay_steps))

"""
    apply_settled_state!(sys, config, path) -> Bool

Restore the settled state at `path` into `sys` and re-apply the settled
damping. Returns `false` when the file is missing or unreadable, so the caller
can fall back to the source geometry rather than crash.
"""
function apply_settled_state!(sys, config::V3SettleConfig, path)
    state = read_state_log(path)
    isnothing(state) && return false
    update_from_sysstate!(sys, state)
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
    data_path = project_data_path(config.project, data_path)
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))
    path = settled_state_path(config, init_row; data_path, cache_path)
    sys, yaml_set = build_settling_struct(config; data_path,
        source_struc = struc_geometry_path(config.project; data_path),
        source_aero = aero_geometry_path(config.project; data_path,
            aero_mode = resolve_aero_mode(config.kite_set)))
    apply_settled_state!(sys, config, path) ||
        error("No settled state at $path; run settle_wing first")
    sys.set = isnothing(set) ? yaml_set : set
    return sys
end

"""
    settle_wing(config::V3SettleConfig, init_row;
                data_path=nothing, cache_path=nothing, show_progress=true,
                remake_model=false, remake_settled_state=false)
    settle_wing(config::V3SettleConfig;
                position, velocity, attitude,
                steering, depower, wind_vec,
                data_path=nothing, cache_path=nothing, show_progress=true,
                remake_model=false, remake_settled_state=false)
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

When `remake_settled_state=false` and the destination file already
exists, the simulation is skipped and the settled geometry is
loaded from file; `remake_model` is the independent flag for the
serialized equations. The cache file name encodes the settling inputs, including
the elevation implied by the initial position and a non-default
`aero_mode`, so runs that only differ in elevation or in
aerodynamics get their own file instead of sharing one. A cached
geometry whose aerodynamics disagree with `config.kite_set.aero_mode` is
rejected and re-derived from the source YAML.

A kite whose `init_mode` is `:relaxed_state` starts the settling
from its `init_state` instead of from the placed geometry (see
[`relaxed_state_name`](@ref)), which is what makes a geometry whose
bridle rest lengths disagree with its node positions settleable at
all, and what makes one that agrees settle faster. The state is
restored before the flight state is applied, so the relaxed bridle
shape rides along into the target pose.

`data_path` is where the source geometry/settings YAMLs are read
from (default [`v3_data_path`](@ref)); `cache_path` is where
everything generated is written — the `settled_*.arrow` state, the
settling log, and the serialized model binary. It defaults to
[`default_cache_path`](@ref)`(data_path)`, a depot scratch
directory regardless of install mode, and is created if missing.
See [`init`](@ref) and [`with_model_cache`](@ref).
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
    "Damping held on the bodies, both frames, during the ramp and decayed after."
    body_damping::Float64 = 20.0
    dt::Float64 = 0.02
    max_steps::Int = 600
    "Steps held at full stiffness while the damping decays back to its own value."
    hold_steps::Int = 200
    "VSM solve interval during the hold phase; 0 keeps the aero frozen."
    hold_vsm_interval::Int = 1
    "Steps between progress lines."
    report_every::Int = 20
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
    damping_snapshot(components) -> Vector{Tuple}

Copy the world- and body-frame damping of every point or body, so a relaxation
can hand them back when it is done.
"""
damping_snapshot(components) =
    [(copy(item.world_frame_damping), copy(item.body_frame_damping))
     for item in components]

"""
    raise_damping!(components, saved, world, body)

Hold every component's damping at `world`/`body`, or at the value it started
with where that is already larger. `world`/`body` at zero restores the snapshot.
"""
function raise_damping!(components, saved, world, body)
    for (idx, item) in enumerate(components)
        item.world_frame_damping .= max.(saved[idx][1], world)
        item.body_frame_damping .= max.(saved[idx][2], body)
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
settle instead of ringing, and `config.body_damping` to every body in both
frames: a beam wing carries most of its mass in bodies, whose rigid-body motion
nothing else damps. Both decay away over the hold phase before returning.

Call after `init!`. Returns the scale actually reached (1.0 on success), the
number of steps taken and the final residual.
"""
function relax_bridle!(sam, sys, config::V3RelaxConfig=V3RelaxConfig();
                       prn::Bool=true)
    baseline = segment_stiffness_baseline(sys)
    saved_points = damping_snapshot(sys.points)
    saved_bodies = damping_snapshot(sys.bodies)
    raise_damping!(sys.points, saved_points, config.world_damping, 0.0)
    raise_damping!(sys.bodies, saved_bodies, config.body_damping,
        config.body_damping)

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
        prn && steps % config.report_every == 0 &&
            @info "relax_bridle! ramping" steps scale residual
        residual < config.settled_tol || continue
        scale = min(1.0, scale * config.growth)
        scale_segment_stiffness!(sys, baseline, scale)
    end

    scale_segment_stiffness!(sys, baseline, 1.0)
    if scale >= 1.0
        for hold in 1:config.hold_steps
            decay = 1.0 - hold / config.hold_steps
            raise_damping!(sys.points, saved_points,
                config.world_damping * decay, 0.0)
            raise_damping!(sys.bodies, saved_bodies,
                config.body_damping * decay, config.body_damping * decay)
            try
                next_step!(sam; dt=config.dt,
                    vsm_interval=config.hold_vsm_interval)
            catch exception
                prn && @warn "relax_bridle! hold phase stopped" hold exception
                break
            end
            steps += 1
            prn && hold % config.report_every == 0 &&
                @info "relax_bridle! holding" hold residual=relax_residual(sam)
        end
    end
    raise_damping!(sys.points, saved_points, 0.0, 0.0)
    raise_damping!(sys.bodies, saved_bodies, 0.0, 0.0)
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
    default_cache_path(data_path = nothing) -> String

Where generated artifacts go when the caller names no `cache_path`: always a
scratch directory next to the depot, keyed by V3Kite's own UUID — regardless
of `data_path` (a caller's project directory, possibly its own, outside
V3Kite entirely) and regardless of whether V3Kite itself is Pkg-installed or
a development checkout. The argument is accepted and ignored, so a caller that
has a `data_path` to hand can pass it and one that has none can leave it out.

A package directory under `DEPOT_PATH/packages` is not ours to write to: it is
usually read-only, and `Pkg.gc` deletes the whole tree once no environment
references that version. A `scratchspaces` directory keyed by our UUID
survives reinstalling and is writable regardless of install mode, so
`precompile.jl`'s warm-up artifacts and every runtime caller — dev'ed or
installed — agree on where to look.
"""
function default_cache_path(data_path = nothing)
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
                     remake_model=false,
                     remake_settled_state=false)
    data_path = project_data_path(config.project, data_path)
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))

    gc = config.kite_set.geom
    gc.tether_length = config.tether_length

    cache_path != data_path && mkpath(cache_path)
    dest_struc = settled_state_path(
        config, init_row; data_path, cache_path)
    source_struc = struc_geometry_path(config.project; data_path)
    source_aero = aero_geometry_path(config.project; data_path,
        aero_mode = resolve_aero_mode(config.kite_set))

    # Run settling simulation if needed
    syslog = nothing
    settle_failed = false
    settling_rebuilt = false
    if remake_settled_state || !isfile(dest_struc)
        try
            syslog = run_power_zone_settling!(
                config; data_path, show_progress,
                source_struc, source_aero, dest_struc,
                log_path = cache_path, cache_path, init_row)
            settling_rebuilt = remake_model
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
    set = Settings(project_path(config.project; data_path))
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    set.g_earth = config.g_earth
    # profile_law is taken from sim_settings_default.yaml (loaded via Settings above).
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
        sam = SymbolicAWEModel(set, sys; backend = config.kite_set.backend)
        # Tape rest lengths are parameters, not state, so the log does not carry
        # them; the rebuilt structure needs them applied as settling did.
        apply_geom_adjustments!(sys, gc)
        sys.tethers[1].init_stretched_len = gc.tether_length
        with_model_cache(cache_path) do
            # Settling already rebuilt into this cache, and the bin carries a
            # structure hash that rejects it if the settled structure differs.
            SymbolicAWEModels.init!(sam;
                remake=remake_model && !settling_rebuilt, remake_vsm=true,
                reinit_sys=false,
                analytic_jacobian=config.kite_set.analytic_jacobian)
        end
    else
        @info "Loading source geometry" source_struc
        vsm_path = vsm_settings_path(config.project; data_path)
        vsm_set = VortexStepMethod.VSMSettings(
            vsm_path; data_prefix=false)
        vsm_set.wings[1].geometry_file = source_aero
        sys = load_sys_struct_from_yaml(source_struc;
            system_name=v3_model_name(config.kite_set), set,
            dynamics_type=config.kite_set.wing_type, vsm_set,
            aero_mode=resolve_aero_mode(config.kite_set))
        sam = SymbolicAWEModel(set, sys; backend = config.kite_set.backend)
        with_model_cache(cache_path) do
            SymbolicAWEModels.init!(sam;
                remake=remake_model && !settling_rebuilt, ignore_l0=false,
                remake_vsm=true,
                analytic_jacobian=config.kite_set.analytic_jacobian)
        end
    end

    return sam, syslog, settle_failed
end

"""
    build_settling_struct(config; data_path, source_struc, source_aero)

Build the settling `SystemStructure` from the source YAML: settings, VSM, the
structure itself, the resolved KCU mass, the kite's material and the starting
damping. Returns `(sys, set)`. Shared by [`setup_settling_model`](@ref) and the
settled-state cache, which rebuilds the same structure before restoring a state
onto it.

The material goes on here and not only in [`create_v3_model`](@ref) because a
structure settled on the YAML's bridle stiffness is not an equilibrium of the one
the run flies.
"""
function build_settling_struct(config::V3SettleConfig;
        data_path, source_struc, source_aero)
    set = Settings(project_path(config.project; data_path))
    set.g_earth = config.g_earth
    set.v_wind = config.v_wind
    set.l_tether = config.tether_length
    # profile_law is taken from sim_settings_default.yaml (loaded via Settings above).

    vsm_path = vsm_settings_path(config.project; data_path)
    vsm_set = VortexStepMethod.VSMSettings(
        vsm_path; data_prefix=false)
    vsm_set.wings[1].geometry_file = source_aero

    sys = load_sys_struct_from_yaml(source_struc;
        system_name=v3_model_name(config.kite_set), set,
        dynamics_type=config.kite_set.wing_type, vsm_set,
        aero_mode=resolve_aero_mode(config.kite_set))

    # Explicit `config.kcu_mass` (used by parameter sweeps) takes priority;
    # otherwise fall back to the `kcu_mass` field of the active settings YAML
    # (0 means "not set", i.e. keep the geometry-file default).
    kcu_mass = !isnothing(config.kcu_mass) ? config.kcu_mass :
        (set.kcu_mass != 0 ? set.kcu_mass : nothing)
    if !isnothing(kcu_mass)
        sys.points[1].extra_mass = kcu_mass
    end

    apply_kite_material!(sys, config.kite_set)
    SymbolicAWEModels.set_world_frame_damping(
        sys, config.world_start_damping)
    set_body_frame_damping!(sys, config.body_start_damping)
    for (rng, damp) in config.body_damping_overrides
        SymbolicAWEModels.set_body_frame_damping(
            sys, damp, rng)
    end
    settle_dps = settle_damping_per_stiffness(config)
    if !isnothing(settle_dps)
        set_damping_per_stiffness!(sys, tether_bridle_segments(sys),
                                   settle_dps)
    end
    return sys, set
end

"""
Set up a settling model: settings, VSM, sys struct, damping,
SAM creation, geometry adjustments, init, and lock tether.
Returns `(sam, sys, gc)`.
"""
function setup_settling_model(config::V3SettleConfig;
        data_path, source_struc, source_aero, cache_path=nothing)
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))
    gc = config.kite_set.geom
    sys, set = build_settling_struct(config;
        data_path, source_struc, source_aero)

    sam = SymbolicAWEModel(set, sys; backend = config.kite_set.backend)
    apply_geom_adjustments!(sys, gc)
    sys.tethers[1].init_stretched_len = gc.tether_length
    with_model_cache(cache_path) do
        SymbolicAWEModels.init!(sam; remake=config.kite_set.remake_model,
            ignore_l0=false, remake_vsm=true,
            analytic_jacobian=config.kite_set.analytic_jacobian)
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
        dest_struc, cache_path=nothing, log_path=nothing,
        init_row)
    isnothing(cache_path) && (cache_path = default_cache_path(data_path))
    isnothing(log_path) && (log_path = cache_path)
    sam, sys, gc = setup_settling_model(config;
        data_path, source_struc, source_aero, cache_path)

    if !isnothing(config.kite_set.init_state)
        state_path = project_file(config.project, config.kite_set.init_state;
            data_path)
        start_from_state!(sam, sys, state_path) ||
            error("No relaxed state at $state_path; run the relaxation " *
                  "example for $(basename(source_struc)) first")
        @info "Settling from a relaxed state" state_path
    end

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

    reinit_integrator!(sam)

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
            apply_decayed_damping!(sys, config, decay)

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
            reinit_integrator!(sam; prn=false)

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
                    @info "Step $step/$(config.num_steps)" substep=sub damping=round.(sys.points[1].body_frame_damping, digits=1) elevation=round(rad2deg(wing.elevation), digits=2) heading=round(rad2deg(wing.heading), digits=2)
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
    save_state_log(sam, dest_struc)

    syslog = save_and_load_log(
        logger, "settle_particle_dynamics_wing"; path=log_path)
    @info "Settling complete" dest_struc
    return syslog
end
