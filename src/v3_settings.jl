# Copyright (c) 2026 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Settings files of a V3 project, loaded into structs the way `RippleSettings`
already is.

A KiteUtils project file (`system*.yaml`) is a `system:` section of pointers to
other files. KiteUtils ships accessors for the keys it defines
(`sim_settings`, `structural_geometry`, `aero_geometry`, `vsm_settings`) and
returns each as a bare filename, leaving the parse to the consumer. The three
accessors here are siblings for the keys V3Kite adds.
"""

"""
    project_entry(project, key; data_path=nothing, default=nothing) -> String

Filename under `key` in the `system:` section of `project`. This is KiteUtils'
`sim_settings(project)` with two differences: it reaches keys KiteUtils does not
define, and it takes `data_path` explicitly rather than reading the global one,
which settling needs because it resolves paths against a data path while the
active one points at the model cache. It defaults to [`v3_data_path`](@ref), as
`settle_wing` and `create_v3_model` do, not to the active path.

Errors naming the project file when the key is absent and no `default` is given,
since a missing pointer is a mistake in the file rather than something to guess.
"""
function project_entry(project, key; data_path=nothing, default=nothing)
    isnothing(data_path) && (data_path = v3_data_path())
    dict = YAML.load_file(joinpath(data_path, project))
    system = get(dict, "system", nothing)
    isnothing(system) && error("$project has no `system:` section")
    if !haskey(system, key)
        isnothing(default) &&
            error("$project has no `$key:` entry in its system section")
        return default
    end
    return system[key]
end

"""
    struc_geometry_path(project; data_path=nothing) -> String

Absolute path of the project's `structural_geometry:` file.
"""
struc_geometry_path(project; data_path=nothing) =
    joinpath(something(data_path, v3_data_path()),
        project_entry(project, "structural_geometry"; data_path))

"""
    vsm_settings_path(project; data_path=nothing) -> String

Absolute path of the project's `vsm_settings:` file.
"""
vsm_settings_path(project; data_path=nothing) =
    joinpath(something(data_path, v3_data_path()),
        project_entry(project, "vsm_settings"; data_path))

"""
    has_surface_tables(geometry_path) -> Bool

Whether an aero geometry carries the per-node `Cp`/`cf` tables `AeroPressure`
builds its station-point map from, rather than lift/drag/moment polars alone.
"""
function has_surface_tables(geometry_path)
    dict = YAML.load_file(geometry_path)
    airfoils = get(dict, "wing_airfoils", nothing)
    isnothing(airfoils) && return false
    rows = get(airfoils, "data", nothing)
    isnothing(rows) && return false
    return any(rows) do row
        row isa AbstractVector && length(row) >= 3 &&
            row[3] isa AbstractDict && haskey(row[3], "cp_file")
    end
end

"""
    aero_geometry_path(project; data_path=nothing, aero_mode=nothing) -> String

Absolute path of the project's `aero_geometry:` file, erroring when it is
missing. The surface-resolved tables `AeroPressure` needs are generated rather
than tracked, so a fresh checkout hits this and is told which example writes
them.

Given the `aero_mode` the project will be flown on, a geometry that cannot carry
that mode is rejected here rather than deep in the model build: `AeroPressure`
has no surface to work on unless the geometry carries
[`has_surface_tables`](@ref).
"""
function aero_geometry_path(project; data_path=nothing, aero_mode=nothing)
    entry = project_entry(project, "aero_geometry"; data_path)
    path = joinpath(something(data_path, v3_data_path()), entry)
    isfile(path) || error(
        "No aero geometry at $path (from `aero_geometry:` in $project). " *
        "The surface-resolved tables are generated, not tracked — run " *
        "examples/v3beam_aero_geometry.jl to write them.")
    if aero_mode isa SymbolicAWEModels.AeroPressure && !has_surface_tables(path)
        error("$project flies `aero_mode: pressure` on `aero_geometry: $entry`, " *
              "which carries polars but no per-node Cp/cf tables. Run " *
              "examples/v3beam_aero_geometry.jl and point `aero_geometry:` " *
              "at nf_aero_geometry.yaml.")
    end
    return path
end

"""Filename of the [`V3KiteConfig`](@ref) file of `project`."""
kite_settings(project) = project_entry(project, "kite_settings")

"""Filename of the [`HeadingSettings`](@ref) file of `project`."""
heading_settings(project) = project_entry(project, "heading_settings")

"""Filename of the [`V3SettleConfig`](@ref) file of `project`."""
settle_settings(project) = project_entry(project, "settle_settings")

"""
    select_project(options; prompt="Which kite?", default=1) -> String

Ask which project file to run and return its filename. `options` are
`label => "system_*.yaml"` pairs, so one example covers several kites instead of
being copied per kite.

`ENV["V3KITE_PROJECT"]` skips the menu: it selects the one option whose filename
contains it, so `psm` or `beam` is enough. `bin/run_julia --psm` / `--beam` set it.
An unmatched value errors rather than falling back, so a typo cannot silently run
the other kite.

Without a terminal — a scripted run, a test, CI — the `default` option is taken
and reported rather than asked for, and leaving the menu takes it too.
"""
function select_project(options::AbstractVector{<:Pair};
                        prompt="Which kite?", default::Int=1)
    projects = [String(last(option)) for option in options]
    wanted = lowercase(strip(get(ENV, "V3KITE_PROJECT", "")))
    if !isempty(wanted)
        matched = findall(project -> occursin(wanted, lowercase(project)), projects)
        length(matched) == 1 || error(
            "V3KITE_PROJECT=\"$wanted\" matches $(length(matched)) of $projects; " *
            "give a fragment that picks exactly one")
        @info "$prompt taken from V3KITE_PROJECT" project=projects[only(matched)]
        return projects[only(matched)]
    end
    if !(stdin isa Base.TTY)
        @info "$prompt not asked (no terminal)" project=projects[default]
        return projects[default]
    end
    labels = ["$(first(option))  [$(last(option))]" for option in options]
    choice = request("\n$prompt", RadioMenu(labels, pagesize=8))
    choice == -1 && (choice = default)
    return projects[choice]
end

"""
    parse_backend(name) -> ModelBackend

`"kernel"` or `"monolith"`. Unknown names error here rather than as a
`MethodError` deep in the model build.
"""
function parse_backend(name)
    name isa SymbolicAWEModels.ModelBackend && return name
    key = Symbol(lowercase(String(name)))
    key === :kernel && return SymbolicAWEModels.KernelBackend()
    key === :monolith && return SymbolicAWEModels.MonolithBackend()
    error("Unknown backend `$name`; expected kernel or monolith")
end

"""
    parse_aero_mode(name) -> Union{Nothing, AbstractAeroModel}

`"direct"`, `"continuous"` or `"pressure"`; `nothing` resolves by wing type.
`AeroPressure` needs a geometry carrying per-node `Cp`/`cf` tables, which
`v3beam_aero_geometry.jl` writes.
"""
function parse_aero_mode(name)
    isnothing(name) && return nothing
    name isa SymbolicAWEModels.AbstractAeroModel && return name
    key = Symbol(lowercase(String(name)))
    key === :direct && return SymbolicAWEModels.AeroDirect()
    key === :continuous && return SymbolicAWEModels.ContinuousAero()
    key === :pressure && return SymbolicAWEModels.AeroPressure()
    error("Unknown aero_mode `$name`; expected direct, continuous or pressure")
end

"""
    parse_wing_type(name) -> WingType

`"particle_dynamics"` or `"rigid_dynamics"`.
"""
function parse_wing_type(name)
    name isa SymbolicAWEModels.WingType && return name
    key = Symbol(lowercase(String(name)))
    key === :particle_dynamics && return SymbolicAWEModels.PARTICLE_DYNAMICS
    key === :rigid_dynamics && return SymbolicAWEModels.RIGID_DYNAMICS
    error("Unknown wing_type `$name`; expected particle_dynamics or " *
          "rigid_dynamics")
end

"""
    fill_struct(T, dict; source="", converters=Dict()) -> T

Build the `Base.@kwdef` struct `T` from `dict`, whose keys are field names.
Absent keys keep the struct default, so a settings file only has to carry what
it changes; a key that is not a field of `T` errors naming `source`, so a typo
is caught when the file loads rather than silently leaving the default in place.

`converters` maps a field name to a function turning its YAML value into the
field type — enums, nested structs, anything `convert` cannot do on its own.
"""
function fill_struct(T, dict; source="", converters=Dict{Symbol, Any}())
    isnothing(dict) && return T()
    kwargs = Dict{Symbol, Any}()
    for (key, value) in dict
        name = Symbol(key)
        if !(name in fieldnames(T))
            where_from = isempty(source) ? "" : " in $source"
            known = join(fieldnames(T), ", ")
            error("`$name`$where_from is not a field of $(nameof(T)); " *
                  "known fields: $known")
        end
        converter = get(converters, name, nothing)
        kwargs[name] = isnothing(converter) ?
            convert_setting(fieldtype(T, name), value) : converter(value)
    end
    return T(; kwargs...)
end

"""
    convert_setting(T, value)

`convert` a YAML scalar to a field type, with the cases plain `convert` cannot
do: a `Symbol` written as a string, a `UnitRange` written as a list, and the
small unions the config structs use, which `convert` will not disambiguate.
"""
convert_setting(::Type{Symbol}, value) = Symbol(value)
convert_setting(::Type{UnitRange{Int}}, value) = value[1]:value[end]
convert_setting(::Type{Union{Float64, Vector{Float64}}}, value) =
    value isa AbstractVector ? convert(Vector{Float64}, value) : Float64(value)
convert_setting(::Type{Union{Nothing, Float64}}, value) =
    isnothing(value) ? nothing : Float64(value)
convert_setting(::Type{Union{Nothing, String}}, value) =
    isnothing(value) ? nothing : String(value)
convert_setting(T, value) = convert(T, value)

"""
    V3GeomAdjustConfig(dict; source="") -> V3GeomAdjustConfig

The `geom:` block of a [`V3KiteConfig`](@ref) file.
"""
V3GeomAdjustConfig(dict::AbstractDict; source="") =
    fill_struct(V3GeomAdjustConfig, dict; source)

"""
    V3BridleConfig(dict; source="") -> V3BridleConfig

The `bridle:` block of a [`V3KiteConfig`](@ref) file.
"""
V3BridleConfig(dict::AbstractDict; source="") =
    fill_struct(V3BridleConfig, dict; source)

"""
    settings_block(filename, section; data_path=nothing) -> (Dict, String)

The `section:` mapping of a settings file resolved under `data_path`, and the
file's basename for error messages. Errors when the section is absent, that
being a file pointed at by the wrong key.
"""
function settings_block(filename, section; data_path=nothing)
    path = isabspath(filename) ? filename :
        joinpath(something(data_path, v3_data_path()), filename)
    dict = YAML.load_file(path)
    haskey(dict, section) ||
        error("$(basename(path)) has no top-level `$section:` section")
    return dict[section], basename(path)
end

"""
    V3KiteConfig(filename; data_path=nothing) -> V3KiteConfig

Load the kite variant from `filename`, whose top-level `kite_settings:` mapping
holds the field names of [`V3KiteConfig`](@ref) plus the nested `geom:` and
`bridle:` blocks.
"""
function V3KiteConfig(filename::String; data_path=nothing)
    dict, source = settings_block(filename, "kite_settings"; data_path)
    return fill_struct(V3KiteConfig, dict; source, converters=Dict{Symbol, Any}(
        :backend => parse_backend,
        :aero_mode => parse_aero_mode,
        :wing_type => parse_wing_type,
        :geom => d -> V3GeomAdjustConfig(d; source),
        :bridle => d -> V3BridleConfig(d; source)))
end

"""
    V3SettleConfig(filename; data_path=nothing) -> V3SettleConfig

Load the settling schedule from `filename`, whose top-level `settle_settings:`
mapping holds the field names of [`V3SettleConfig`](@ref). The `project` and
`kite` fields are not in the file — [`load_kite`](@ref) fills them from the
project this schedule was reached through.
"""
function V3SettleConfig(filename::String; data_path=nothing)
    dict, source = settings_block(filename, "settle_settings"; data_path)
    return fill_struct(V3SettleConfig, dict; source)
end

"""
    load_kite(project; data_path=nothing) -> V3KiteConfig

The [`V3KiteConfig`](@ref) a project file points at with its `kite_settings:`
key.
"""
load_kite(project; data_path=nothing) =
    V3KiteConfig(project_entry(project, "kite_settings"; data_path); data_path)

"""
    load_settle(project; data_path=nothing, kite=nothing) -> V3SettleConfig

The [`V3SettleConfig`](@ref) a project file points at with its
`settle_settings:` key, carrying the project and its kite so the settling
functions can reach the geometry and the model options from the schedule alone.
A project with no `settle_settings:` key gets the struct defaults, which is
enough for a kite that never settles.
"""
function load_settle(project; data_path=nothing, kite=nothing)
    isnothing(kite) && (kite = load_kite(project; data_path))
    filename = project_entry(project, "settle_settings"; data_path, default="")
    config = isempty(filename) ? V3SettleConfig() :
        V3SettleConfig(filename; data_path)
    config.project = project
    config.kite = kite
    return config
end

"""
    HeadingSettings

Gains of the heading PID, what `WC_Settings` is to the winch loop. The
controller was retuned in every example that closed it — K between 1.0 and 1.2,
`Td` 0 or 0.15 — with only `simple_sinus.jl` documenting its tuning, so the
numbers live in a file where the divergence is visible.

The setpoint is not here: a sinusoid, a constant or a course to track is the
maneuver an example runs, not a property of the controller.
"""
Base.@kwdef mutable struct HeadingSettings
    "Proportional gain; the output is a steering fraction, [-1, 1]"
    K::Float64 = 1.2
    "Integral time [s]; `false` disables integral action"
    Ti::Union{Bool, Float64} = false
    "Derivative time [s]; `false` disables derivative action"
    Td::Union{Bool, Float64} = 0.15
    """
    Maximum gain of the derivative filter: the D path is `K*Td*s / (1 + s*Td/N)`,
    so it amplifies measurement noise by up to `N*K` above `N/(2*pi*Td)` [Hz].
    """
    N::Float64 = 10.0
    "Saturation of the steering command, applied symmetrically [-]"
    max_steering::Float64 = 0.175

    """
    Apparent wind speed the gain applies at. A kite's turn rate goes roughly with
    `u_s * v_app`, so scheduling `K * v_app_ref / v_app` keeps the closed loop
    roughly invariant as the apparent wind changes. `0` disables the schedule.
    """
    v_app_ref::Float64 = 0.0
    "Lower clamp on `v_app` in the schedule, limiting the gain boost [m/s]"
    v_app_min::Float64 = 5.0

    """
    Seconds over which `Td` is ramped to zero. Sustained derivative action damps
    the fast mode but floors the settled tracking error, so ramping it away keeps
    the initial transient damped without paying for it afterwards. `0` holds
    `Td`.
    """
    td_ramp_time::Float64 = 0.0
end

"""
    HeadingSettings(filename; data_path=nothing) -> HeadingSettings

Load the heading-PID gains from `filename`, whose top-level `heading_settings:`
mapping holds the field names of [`HeadingSettings`](@ref).
"""
function HeadingSettings(filename::String; data_path=nothing)
    dict, source = settings_block(filename, "heading_settings"; data_path)
    return fill_struct(HeadingSettings, dict; source)
end

"""
    load_heading(project; data_path=nothing) -> HeadingSettings

The [`HeadingSettings`](@ref) a project file points at with its
`heading_settings:` key. A project without one gets the struct defaults.
"""
function load_heading(project; data_path=nothing)
    filename = project_entry(project, "heading_settings"; data_path, default="")
    return isempty(filename) ? HeadingSettings() :
        HeadingSettings(filename; data_path)
end

"""
    heading_pid(settings::HeadingSettings, dt) -> DiscretePID

Build the heading PID `settings` describes. Apply the gain schedule and the `Td`
ramp with [`schedule_heading_pid!`](@ref) each step.
"""
heading_pid(settings::HeadingSettings, dt) =
    create_heading_pid(; K = settings.K, Ti = settings.Ti, Td = settings.Td,
        N = settings.N, dt, umin = -settings.max_steering,
        umax = settings.max_steering)

"""
    schedule_heading_pid!(pid, settings, t, v_app, setpoint, measurement)

Apply the two things a heading PID needs per step: the `1/v_app` gain schedule
and the `Td` ramp. Both are no-ops at their disabled defaults, so an example that
wants neither can still call this.
"""
function schedule_heading_pid!(pid, settings::HeadingSettings, t, v_app,
                               setpoint, measurement)
    if settings.td_ramp_time > 0
        set_Td!(pid, settings.Td *
            (1 - ramp_factor(t, 0.0, settings.td_ramp_time)))
    end
    if settings.v_app_ref > 0
        scaled = settings.K * settings.v_app_ref /
            max(v_app, settings.v_app_min)
        set_K!(pid, scaled, setpoint, measurement)
    end
    return pid
end

"""
    V3ReplayConfig

Which recorded maneuver `flight_replay.jl` puts through the simulator, and the
corrections it applies on the way. The wing itself is not here: that is the
project's `kite_settings:`, so the same maneuver can be replayed on any wing.
"""
Base.@kwdef mutable struct V3ReplayConfig
    "Flight year, selecting the recording and its depower calibration"
    year::Int = 2025
    "Maneuver to replay; `\$(section)_\$(year)` keys into `maneuvers`"
    section::String = "straight_right"
    "Substeps per 10 Hz data sample; 2 gives dt = 0.05 s"
    n_substeps::Int = 2
    "Settle the wing on the first data row before replaying"
    settle::Bool = true
    "Depower offset of the 2019 flight; the 2025 one is the kite file's"
    depower_offset_2019::Float64 = 0.07
    "Scales every recorded steering command"
    steering_multiplier::Float64 = 1.0
    "Proportional gain of the heading feedback; 0 replays open loop"
    heading_K::Float64 = 0.0
    "Integral time of the heading feedback [s]"
    heading_Ti::Float64 = 0.0
    "Proportional gain of the lateral position feedback; 0 disables it"
    lateral_K::Float64 = 0.0
    "Constant added to every steering command [%]"
    steering_offset::Float64 = 1.5
    "Drive the steering off flown distance instead of off time"
    distance_based_steering::Bool = false
    "Slope of the photogrammetry angle-of-attack offset [deg per % depower]"
    aoa_offset_slope::Float64 = -0.6831
    "Intercept of the photogrammetry angle-of-attack offset [deg]"
    aoa_offset_intercept::Float64 = 28.74
    "Where the wind speed comes from: `ekf` or `lidar`"
    wind_source_speed::Symbol = :ekf
    "Where the wind direction and vertical component come from: `ekf` or `lidar`"
    wind_source_dir::Symbol = :lidar
    "Write the figures instead of only displaying them"
    save_figs::Bool = true
    "Directory the figures go to, relative to the repository root when not absolute"
    figures_dir::String = "output"
    "UTC window `[start, end]` of every maneuver, keyed by `\$(section)_\$(year)`"
    maneuvers::Dict{String, Any} = Dict{String, Any}()
end

"""
    V3ReplayConfig(filename; data_path=nothing) -> V3ReplayConfig

Load the replay setup from `filename`, whose top-level `replay_settings:`
mapping holds the field names of [`V3ReplayConfig`](@ref).
"""
function V3ReplayConfig(filename::String; data_path=nothing)
    dict, source = settings_block(filename, "replay_settings"; data_path)
    return fill_struct(V3ReplayConfig, dict; source)
end

"""
    load_replay(project; data_path=nothing) -> V3ReplayConfig

The [`V3ReplayConfig`](@ref) a project file points at with its
`replay_settings:` key. A project without one gets the struct defaults.
"""
function load_replay(project; data_path=nothing)
    filename = project_entry(project, "replay_settings"; data_path, default="")
    return isempty(filename) ? V3ReplayConfig() :
        V3ReplayConfig(filename; data_path)
end

"""
    replay_maneuver(config::V3ReplayConfig) -> (name, start_utc, end_utc)

The maneuver `config` selects: its `\$(section)_\$(year)` name and the UTC
window `maneuvers:` gives it. Errors on a section the file does not list.
"""
function replay_maneuver(config::V3ReplayConfig)
    name = "$(config.section)_$(config.year)"
    haskey(config.maneuvers, name) || error(
        "`$name` is not in `maneuvers:`; known: " *
        join(sort(collect(keys(config.maneuvers))), ", "))
    window = config.maneuvers[name]
    return name, String(window["start"]), String(window["end"])
end
