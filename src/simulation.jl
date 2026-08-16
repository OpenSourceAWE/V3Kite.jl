# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite simulation functions.
Provides high-level functions for creating and running V3 kite simulations.
"""

const V3_MODEL_NAME = "v3"
const V3_RIGID_DYNAMICS_MODEL_NAME = "v3_rigid_dynamics"

"""
Aero-mode tag left out of the settled-geometry cache key, so files written
before the key knew about aerodynamics keep being found. See
[`settled_state_path`](@ref).
"""
const DEFAULT_AERO_TAG = SymbolicAWEModels.aero_mode_tag(
    SymbolicAWEModels.AeroDirect())

"""
Source-geometry tag left out of the settled-state cache key, for the same reason
[`DEFAULT_AERO_TAG`](@ref) is. See [`settled_state_path`](@ref).
"""
const DEFAULT_STRUC_TAG = "struc_geometry"

"""
    v3_data_path()

Return the path to the V3 data directory bundled with V3Kite.jl.
"""
function v3_data_path()
    return joinpath(pkgdir(@__MODULE__), "data")
end

"""
    V3KiteConfig

Which kite a run flies and how its model is assembled — the part of a project
file KiteUtils has no concept of. The geometry files come from the project
file's `structural_geometry:`, `aero_geometry:` and `vsm_settings:` keys and the
flight condition from its `sim_settings:`, so neither is repeated here.

Loaded from the `kite_settings:` file of a project by
[`V3KiteConfig(filename)`](@ref); see `data/kite_settings_beam.yaml`.
"""
Base.@kwdef mutable struct V3KiteConfig
    wing_type::SymbolicAWEModels.WingType = SymbolicAWEModels.PARTICLE_DYNAMICS
    "`nothing` resolves by `wing_type`: `AeroDirect` for particle, default for rigid"
    aero_mode::Union{Nothing, SymbolicAWEModels.AbstractAeroModel} = nothing
    """
    `KernelBackend` assembles one kernel per component, which a beam wing wants —
    its bodies and joints make the monolithic build the dominant cost.
    """
    backend::SymbolicAWEModels.ModelBackend = SymbolicAWEModels.MonolithBackend()
    "Steps between VSM aero solves; tuned together with `aero_mode`"
    vsm_interval::Int = 1

    """
    Damping the simulation runs with, and the floor settling decays down to.
    Body-frame damping damps a node against its wing frame, which only `DYNAMIC`
    points have — on a beam wing, whose wing nodes are `BODY_STATIC` points
    riding bodies, it reaches nothing and only `world_sim_damping` acts.
    """
    body_sim_damping::Vector{Float64} = [0.0, 0.0, 20.0]
    world_sim_damping::Vector{Float64} = [0.0, 0.0, 0.0]

    geom::V3GeomAdjustConfig = V3GeomAdjustConfig()
    bridle::V3BridleConfig = V3BridleConfig()

    """
    Mass [kg] redistributed over the wing nodes by chord, `0` keeping what the
    geometry carries. A beam wing keeps its mass in the bodies, so setting this
    there counts gravity twice.
    """
    wing_mass::Float64 = 0.0
    "Leading-edge share of `wing_mass`"
    wing_mass_le_frac::Float64 = 0.75
    "Parasitic drag coefficient spread over the wing nodes, `0` adding none"
    wing_drag_coeff::Float64 = 0.0

    """
    SurfplanAdapter export the beam was built from. `nothing` keeps the linear
    Breukels bending the YAML carries; a directory swaps in the
    curvature-softening Comer-Levy law.
    """
    adapter_dir::Union{Nothing, String} = nothing

    """
    `:settle` runs power-zone settling; `:relaxed_state` skips it and flies
    `init_state` directly, which is what a geometry already relaxed at the
    depower it is flown at needs.
    """
    init_mode::Symbol = :settle
    """
    Relaxed state to start from, see [`relaxed_state_name`](@ref). Settling
    starts from it whenever it is set, which is what makes a geometry whose
    bridle rest lengths disagree with its node positions settleable at all.
    """
    init_state::Union{Nothing, String} = nothing

    """
    Rebuild the serialized model (`data/model_*.bin`) instead of reading it back.
    Needed whenever the equations change; the settled state is untouched.
    """
    remake_model::Bool = false
    """
    Re-run settling instead of loading the cached `data/settled_*.arrow`. Needed
    when the geometry or the flight condition changes.
    """
    remake_settled_state::Bool = false
    brake::Bool = true
end

export V3KiteConfig


"""
    resolve_aero_mode(kite::V3KiteConfig)

`kite.aero_mode` when set, otherwise the default for its wing type: `AeroDirect`
for a particle wing, and the model's own for a rigid one.
"""
function resolve_aero_mode(kite::V3KiteConfig)
    isnothing(kite.aero_mode) || return kite.aero_mode
    kite.wing_type == SymbolicAWEModels.RIGID_DYNAMICS && return nothing
    return SymbolicAWEModels.AeroDirect()
end

"""
    create_v3_model(project::String; data_path=nothing, kite=nothing)
        -> (sam, sys)

Build the `SymbolicAWEModel` a project file describes: its geometry from the
`structural_geometry:`, `aero_geometry:` and `vsm_settings:` keys, its flight
condition from `sim_settings:`, and how the model is assembled from
`kite_settings:` (pass `kite` to override that file).

`data_path` holds the data directory the project file and everything it points
at are read from, and becomes the active data path. Pass `settings` to override
the project's own, which is what rebuilding a structure to match a recorded log
needs — the log's wind and tether length are not the project's.
"""
function create_v3_model(project::String; data_path=nothing, kite=nothing,
                         settings=nothing)
    isnothing(data_path) && (data_path = v3_data_path())
    set_data_path(data_path)
    isnothing(kite) && (kite = load_kite(project; data_path))

    set = isnothing(settings) ? Settings(project) : settings
    set.v_reel_outs[1] = 0.0

    struc_path = struc_geometry_path(project; data_path)
    @info "Creating V3 kite model" project struc=basename(struc_path) wing_type=kite.wing_type

    vsm_set = VortexStepMethod.VSMSettings(
        vsm_settings_path(project; data_path); data_prefix=false)
    vsm_set.wings[1].geometry_file = aero_geometry_path(project;
        data_path, aero_mode = resolve_aero_mode(kite))

    model_name = kite.wing_type == SymbolicAWEModels.RIGID_DYNAMICS ?
        V3_RIGID_DYNAMICS_MODEL_NAME : V3_MODEL_NAME
    sys = load_sys_struct_from_yaml(struc_path;
        system_name=model_name, set, dynamics_type=kite.wing_type,
        vsm_set, aero_mode=resolve_aero_mode(kite))

    SymbolicAWEModels.set_body_frame_damping(sys, kite.body_sim_damping)
    SymbolicAWEModels.set_world_frame_damping(sys, kite.world_sim_damping)

    sam = SymbolicAWEModel(set, sys; backend = kite.backend)

    # `l_tethers: [0]` is KiteUtils' "not set" sentinel; overriding with it would
    # collapse the tether the geometry was placed with.
    if !isempty(sys.tethers) && set.l_tether > 0
        sys.tethers[1].init_stretched_len = set.l_tether
    end
    isempty(sys.transforms) ||
        (sys.transforms[1].elevation = deg2rad(set.elevation))

    return sam, sys
end

"""
    apply_kite_material!(sys, kite::V3KiteConfig) -> sys

Set the bridle material a kite carries onto a loaded structure, and swap the
YAML's linear Breukels bending for the curvature-softening Comer-Levy law when
`kite.adapter_dir` names the export the beam was built from. Both are no-ops on
a geometry that has neither bridle tethers nor beam joints.
"""
function apply_kite_material!(sys, kite::V3KiteConfig)
    if !isnothing(kite.adapter_dir) && isdir(kite.adapter_dir)
        SurfplanAdapter.apply_comer_bending!(sys, kite.adapter_dir,
            SurfplanAdapter.V3BeamTopology(bridle = kite.bridle))
        @info "Comer-Levy bending applied" joints=length(sys.timoshenko_joints)
    end
    SurfplanAdapter.apply_bridle_material!(sys, kite.bridle)
    return sys
end

"""
    build_v3_model(project; data_path=nothing, remake_model=nothing,
                   remake_settled_state=nothing, kite=nothing,
                   settle=nothing) -> (sam, sys)

Bring up the model a project file describes, ready to step. Both `remake` flags
default to the project's `remake_model` / `remake_settled_state`; they are
independent caches, the serialized equations and the settled geometry.

`init_mode: settle` runs power-zone settling at the flight condition of the
project's `sim_settings:`, starting from the kite's `init_state` when it has one.
`init_mode: relaxed_state` skips settling and restores that state onto the loaded
geometry instead, for a kite already relaxed at the depower it is flown at —
restoring after `init!` rather than before, so the rest lengths stay the ones the
state was relaxed against.
"""
function build_v3_model(project; data_path=nothing, remake_model=nothing,
                        remake_settled_state=nothing, kite=nothing, settle=nothing)
    isnothing(data_path) && (data_path = v3_data_path())
    set_data_path(data_path)
    isnothing(kite) && (kite = load_kite(project; data_path))
    isnothing(remake_model) && (remake_model = kite.remake_model)
    isnothing(remake_settled_state) &&
        (remake_settled_state = kite.remake_settled_state)
    set = Settings(project)

    kite.init_mode in (:settle, :relaxed_state) ||
        error("init_mode must be :settle or :relaxed_state, got " *
              "$(kite.init_mode) in the kite_settings of $project")

    if kite.init_mode === :relaxed_state
        isnothing(kite.init_state) &&
            error("init_mode is :relaxed_state but no init_state is set in " *
                  "the kite_settings of $project")
        sam, sys = create_v3_model(project; data_path, kite)
        apply_kite_material!(sys, kite)
        set_depower!(sys, set.depower / 100.0, 0.0, kite.geom)
        set_steering!(sys, 0.0, kite.geom)
        SymbolicAWEModels.init!(sam; remake=remake_model, ignore_l0=false,
                                remake_vsm=true)
        sys.winches[1].brake = kite.brake

        state_path = joinpath(data_path, kite.init_state)
        start_from_state!(sam, sys, state_path) ||
            error("No relaxed state at $state_path; run " *
                  "examples/relax_bridle.jl for this geometry first")
        @info "Started from the relaxed state" state_path
        return sam, sys
    end

    isnothing(settle) && (settle = load_settle(project; data_path, kite))
    settle.v_wind = set.v_wind
    settle.tether_length = set.l_tether
    settle.g_earth = set.g_earth
    settle.start_depower = set.depower + 10.0

    el_rad = deg2rad(set.elevation)
    position = [cos(el_rad) * set.l_tether, 0.0, sin(el_rad) * set.l_tether]
    sam, _, failed = settle_wing(settle;
        position, velocity=[0.0, 0.0, 0.0], heading=0.0, steering=0.0,
        depower=set.depower / 100.0, wind_vec=[set.v_wind, 0.0, 0.0],
        data_path, remake_model, remake_settled_state)
    failed && error("Settling failed for $project")
    sys = sam.sys_struct
    apply_kite_material!(sys, kite)
    sys.winches[1].brake = kite.brake
    return sam, sys
end
