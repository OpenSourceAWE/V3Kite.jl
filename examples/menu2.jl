# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3Kite.jl Simple Example Menu

Interactive menu to select and run the `simple_*` V3Kite examples.

Usage:
    julia --project=examples examples/menu2.jl
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using REPL.TerminalMenus
using V3Kite: set_default_turbulence

# (label, script) pairs; `nothing` as script means "call the action directly" instead of
# including a file. A `"name = include(...)"` string cannot express that: evaluating
# `set_default_turbulence = set_default_turbulence()` would bind the returned number over the
# function, so the second invocation would fail.
files = sort(filter(f -> startswith(f, "simple_") && endswith(f, ".jl"), readdir(@__DIR__)))
actions = Tuple{String, Union{String, Nothing}}[(f[1:end-3], f) for f in files]
push!(actions, ("reel_out_v3", "reel_out_v3.jl"))
push!(actions, ("reel_out_v3_plots", "reel_out_v3_plots.jl"))
push!(actions, ("steering_test_v3", "steering_test_v3.jl"))
push!(actions, ("steering_test_v3_plots", "steering_test_v3_plots.jl"))
push!(actions, ("set_default_turbulence", nothing))

options = [isnothing(script) ? label : "$(label) = include(\"$(script)\")"
           for (label, script) in actions]
push!(options, "quit")

function example_menu()
    active = true
    while active
        menu = RadioMenu(options, pagesize=8)
        choice = request("\nChoose example to run or `q` to quit: ", menu)

        if choice != -1 && choice != length(options)
            _, script = actions[choice]
            if isnothing(script)
                set_default_turbulence()
            else
                eval(Meta.parse(options[choice]))
            end
        else
            println("Left menu. Press <ctrl><d> to quit Julia!")
            active = false
        end
    end
end

example_menu()
