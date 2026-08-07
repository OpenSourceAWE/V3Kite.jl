# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Persistent turbulence preference.
#
# The default turbulence level lives in `data/gui.yaml`, not in the settings YAML: it is a
# per-checkout preference rather than a property of a simulation scenario. `gui.yaml` is the
# gitignored working copy of the tracked `gui.yaml.default`, the same convention V3Kite uses for
# `Manifest-v1.12.toml`. `init` reads it and copies it into `set.use_turbulence`.
#
# A numeric value therefore *shadows* `environment.use_turbulence` in the settings YAML: editing the
# YAML has no effect while `gui.yaml` holds a number. `DEFAULT_TURBULENCE_KEYWORD` ("default") is the
# opt-out: `get_default_turbulence` maps it to `nothing`, which `init` already treats as "leave
# `set.use_turbulence` alone", so the settings YAML stays authoritative unless this file
# deliberately overrides it. It is what `gui.yaml.default` ships, so a fresh checkout does not
# shadow anything.
#
# `update_yaml_scalar`/`insert_yaml_scalar_in_section` are ported from `KiteModels.jl`
# (`src/KiteModels.jl`), where the same `get_default_turbulence`/`set_default_turbulence` pair
# lives. They edit the file line by line because a `YAML.jl` round-trip would drop every comment.

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
    update_yaml_scalar(lines, key, value) -> (lines, updated)

Replace the value of the first line whose stripped form starts with `key`, keeping the original
indentation and trailing comment. `updated` is `false` if no such line exists.
"""
function update_yaml_scalar(lines::Vector{String}, key::AbstractString, value)
    value_str = repr(value)
    result = String[]
    updated = false
    pattern = Regex("^(\\s*" * escape_string(key) * "\\s*)([^#]*?)(\\s*(?:#.*)?)\$")
    for line in lines
        stripped = lstrip(line)
        if !updated && startswith(stripped, key)
            matched = match(pattern, line)
            if isnothing(matched)
                push!(result, key * " " * value_str)
            else
                prefix, _, suffix = matched.captures
                push!(result, prefix * value_str * suffix)
            end
            updated = true
        else
            push!(result, line)
        end
    end
    return result, updated
end

"""
    insert_yaml_scalar_in_section(lines, section, key, value) -> (lines, true)

Insert `key value` into `section`, indented like the section's existing children. Appends the
section itself if it is not present at all.
"""
function insert_yaml_scalar_in_section(lines::Vector{String}, section::AbstractString,
                                       key::AbstractString, value)
    value_str = repr(value)
    result = String[]
    in_section = false
    inserted = false
    section_indent = 0
    child_indent = "    "
    section_found = false

    for line in lines
        stripped = lstrip(line)
        indent = length(line) - length(stripped)

        if !inserted && in_section && !isempty(stripped)
            if indent <= section_indent
                push!(result, child_indent * key * " " * value_str)
                inserted = true
                in_section = false
            elseif indent > section_indent
                child_indent = line[begin:indent]
            end
        end

        push!(result, line)

        if !inserted && startswith(stripped, section)
            in_section = true
            section_found = true
            section_indent = indent
            child_indent = line[begin:indent] * "    "
        end
    end

    # Still inside the section at end of file: append the key there.
    if !inserted && in_section
        push!(result, child_indent * key * " " * value_str)
        inserted = true
    end

    # Only add a new section if it was never found.
    if !inserted && !section_found
        push!(result, section)
        push!(result, child_indent * key * " " * value_str)
    end
    return result, true
end

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
    # Checked before the numeric conversion: `Float64("default")` would land in the `catch` below
    # and print an error for what is a perfectly valid setting.
    is_default_turbulence_keyword(raw) && return nothing
    try
        return Float64(raw)
    catch
        println("Could not read current default_turbulence in $gui_yaml")
        return nothing
    end
end

"""
    set_default_turbulence([value]; data_path = get_data_path()) -> Union{Float64, Nothing}

Persist `default_turbulence` in `data/gui.yaml` and return the new value. Without an argument the
current value is shown and a new one is read from the terminal; a blank line cancels. Values
outside `[0.0, 1.0]` are rejected. Returns `nothing` when nothing was written.

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
        # `current` is `nothing` for the keyword as well as for an unreadable file, so read the
        # raw setting back for the prompt: showing "not set" for a deliberate "default" would be
        # misleading.
        if is_default_turbulence_keyword(raw_default_turbulence(data_path))
            println("Current default_turbulence: $DEFAULT_TURBULENCE_KEYWORD " *
                    "(init uses use_turbulence from the settings YAML)")
        elseif isnothing(current)
            println("Current default_turbulence is not set.")
        else
            println("Current default_turbulence: $current")
        end
        print("Enter new default_turbulence [0.0..1.0] or " *
              "\"$DEFAULT_TURBULENCE_KEYWORD\" (blank to cancel): ")
        input = strip(readline())
        if isempty(input)
            println("Cancelled.")
            return nothing
        end
        value = if is_default_turbulence_keyword(input)
            DEFAULT_TURBULENCE_KEYWORD
        else
            try
                parse(Float64, input)
            catch
                println("Invalid number: $input")
                return nothing
            end
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
            # calc_full_name formats this with "%.1f", so two values that differ only in the second
            # decimal name the same file and silently share one field.
            println("Warning: the wind-field filename keeps only one decimal of this value; " *
                    "$new_value shares its file with $(round(new_value, digits = 1)).")
        end
    end

    lines = KiteUtils.readfile(gui_yaml)
    new_lines, updated = update_yaml_scalar(lines, "default_turbulence:", new_value)
    if !updated
        new_lines, updated = insert_yaml_scalar_in_section(lines, "gui:", "default_turbulence:",
                                                           new_value)
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
