# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Value of `default_turbulence` meaning "no opinion, use the settings YAML".
"""
const DEFAULT_TURBULENCE_KEYWORD = "default"

"""
    is_default_turbulence_keyword(value) -> Bool

`true` for the string `"default"` in any casing, `false` for anything else (numbers included).
"""
is_default_turbulence_keyword(value) =
    value isa AbstractString && lowercase(strip(value)) == DEFAULT_TURBULENCE_KEYWORD

"""
    gui_yaml_path(data_path = get_data_path()) -> Union{String, Nothing}

Path of the `gui.yaml` working copy in `data_path`, creating it from `gui.yaml.default` if it does
not exist yet. Returns `nothing` if neither exists.
"""
function gui_yaml_path(data_path = get_data_path())
    gui_yaml = joinpath(data_path, "gui.yaml")
    if !isfile(gui_yaml)
        gui_yaml_default = gui_yaml * ".default"
        if isfile(gui_yaml_default)
            cp(gui_yaml_default, gui_yaml)
        else
            println("Missing $gui_yaml and fallback $gui_yaml_default")
            return nothing
        end
    end
    return gui_yaml
end

"""
    raw_default_turbulence(data_path = get_data_path())

The `default_turbulence` setting of `data/gui.yaml` exactly as stored, without converting or
validating it, and without printing anything. `nothing` if it cannot be read.

Exists so callers can tell the keyword `"default"` apart from an unreadable file — both of which
[`get_default_turbulence`](@ref) reports as `nothing`.
"""
function raw_default_turbulence(data_path = get_data_path())
    gui_yaml = gui_yaml_path(data_path)
    isnothing(gui_yaml) && return nothing
    dict = try
        YAML.load_file(gui_yaml)
    catch
        return nothing
    end
    haskey(dict, "gui") || return nothing
    return get(dict["gui"], "default_turbulence", nothing)
end

"""
    get_default_turbulence(data_path = get_data_path()) -> Union{Float64, Nothing}

Read `default_turbulence` from `data/gui.yaml`, the turbulence level `init` applies as
`set.use_turbulence`. If the file does not exist it is created from `gui.yaml.default`.

Returns `nothing` both for the keyword `"default"` — meaning `init` should keep the
`environment.use_turbulence` of the settings YAML — and if the value cannot be read at all. The two
are distinguished by their output: the keyword is silent, an unreadable value prints why.
"""
function get_default_turbulence(data_path = get_data_path())
    gui_yaml = gui_yaml_path(data_path)
    isnothing(gui_yaml) && return nothing

    dict = try
        YAML.load_file(gui_yaml)
    catch
        println("Could not parse $gui_yaml")
        return nothing
    end
    if !haskey(dict, "gui") || !haskey(dict["gui"], "default_turbulence")
        println("Could not read current default_turbulence in $gui_yaml")
        return nothing
    end
    raw = dict["gui"]["default_turbulence"]
    # Checked first: Float64("default") would otherwise hit the catch below and misreport a valid setting.
    is_default_turbulence_keyword(raw) && return nothing
    try
        return Float64(raw)
    catch
        println("Could not read current default_turbulence in $gui_yaml")
        return nothing
    end
end

"""
    ask_default_turbulence() -> Union{Float64, Nothing}

Prompt for a turbulence level in `[0.0, 1.0]`, re-asking on invalid input; `nothing` on empty input
(cancel).
"""
function ask_default_turbulence()
    while true
        input = Base.prompt("default_turbulence in [0.0, 1.0]")
        (isnothing(input) || isempty(strip(input))) && return nothing
        value = tryparse(Float64, strip(input))
        if !isnothing(value) && 0.0 <= value <= 1.0
            return value
        end
        println("Enter a number between 0.0 and 1.0, e.g. 0.5.")
    end
end

"""
    set_default_turbulence([value]; data_path = get_data_path()) -> Union{Float64, Nothing}

Persist `default_turbulence` in `data/gui.yaml` and return the new value. Without an argument, a
menu asks for `"default"` or a specific value via [`ask_default_turbulence`](@ref); leaving the menu
cancels. Values outside `[0.0, 1.0]` are rejected. Returns `nothing` when nothing was written.

`0.0` disables turbulence, `1.0` is the Cabauw-calibrated reference level — the setting scales the
turbulence, it is not an absolute intensity. The value takes effect at the next [`init`](@ref).

`value` may also be `"default"`, which makes [`init`](@ref) keep the `environment.use_turbulence`
of the active settings YAML instead of overriding it. Use it whenever turbulence belongs to the
scenario rather than to this checkout — a numeric value here silently wins over the settings YAML.

A wind field is stored per `(grid, use_turbulence, ground wind speed)`, so a value with no matching
`data/windfield_*.npz` makes the next run generate one (~1.2 GB, tens of seconds). Note the filename
keeps only one decimal of `use_turbulence`, so e.g. 0.30 and 0.34 would share a file.
"""
function set_default_turbulence(value::Union{Nothing, Real, AbstractString} = nothing;
                                data_path = get_data_path())
    gui_yaml = gui_yaml_path(data_path)
    isnothing(gui_yaml) && return nothing

    current = get_default_turbulence(data_path)

    if isnothing(value)
        # Re-check the raw value: `current` is `nothing` for both "default" and an unreadable file.
        current_str = if is_default_turbulence_keyword(raw_default_turbulence(data_path))
            DEFAULT_TURBULENCE_KEYWORD
        elseif isnothing(current)
            "not set"
        else
            "$current"
        end
        options = [DEFAULT_TURBULENCE_KEYWORD, "specific value in [0.0, 1.0]...", "quit"]
        choice = request("\nSelect default_turbulence (current: $current_str): ",
                         RadioMenu(options, pagesize = 8))
        if choice == 1
            value = DEFAULT_TURBULENCE_KEYWORD
        elseif choice == 2
            picked = ask_default_turbulence()
            if isnothing(picked)
                println("Cancelled.")
                return nothing
            end
            value = picked
        else
            println("Cancelled.")
            return nothing
        end
    end

    is_default = is_default_turbulence_keyword(value)
    if value isa AbstractString && !is_default
        println("Invalid value: $value. Use a number between 0.0 and 1.0, " *
                "or \"$DEFAULT_TURBULENCE_KEYWORD\".")
        return nothing
    end

    new_value = is_default ? DEFAULT_TURBULENCE_KEYWORD : Float64(value)
    if !is_default
        if new_value < 0.0 || new_value > 1.0
            println("Value out of range. Please use a value between 0.0 and 1.0")
            return nothing
        end
        if round(new_value, digits = 1) != new_value
            # calc_full_name formats this with "%.1f", so two values differing only in the second decimal share a file.
            println("Warning: the wind-field filename keeps only one decimal of this value; " *
                    "$new_value shares its file with $(round(new_value, digits = 1)).")
        end
    end

    # KiteUtils >= 0.11.13; a YAML.jl round-trip would drop every comment in the file.
    lines = KiteUtils.readfile(gui_yaml)
    new_lines, updated = KiteUtils.update_yaml_scalar(lines, "default_turbulence:", new_value)
    if !updated
        new_lines, updated = KiteUtils.insert_yaml_scalar_in_section(lines, "gui:",
                                                           "default_turbulence:", new_value)
        if !updated
            println("Could not update default_turbulence in $gui_yaml")
            return nothing
        end
    end

    KiteUtils.writefile(new_lines, gui_yaml)
    println("default_turbulence set to: $new_value")
    # `new_value > 0` would throw on the keyword, hence the branch order.
    if is_default
        println("init() now takes use_turbulence from the settings YAML.")
    elseif new_value > 0
        println("Takes effect at the next init(); a missing wind field is generated on first use.")
    end
    return new_value
end
