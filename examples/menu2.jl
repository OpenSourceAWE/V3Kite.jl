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

files = sort(filter(f -> startswith(f, "simple_") && endswith(f, ".jl"), readdir(@__DIR__)))
options = [string(f[1:end-3], " = include(\"", f, "\")") for f in files]
push!(options, "quit")

function example_menu()
    active = true
    while active
        menu = RadioMenu(options, pagesize=8)
        choice = request("\nChoose example to run or `q` to quit: ", menu)

        if choice != -1 && choice != length(options)
            eval(Meta.parse(options[choice]))
        else
            println("Left menu. Press <ctrl><d> to quit Julia!")
            active = false
        end
    end
end

example_menu()
