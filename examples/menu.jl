# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3Kite.jl Example Menu

Interactive menu to select and run V3Kite examples.

Usage:
    julia --project=examples examples/menu.jl
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using REPL.TerminalMenus

options = [
    "v3kite = include(\"v3kite.jl\")",
    "reel_out_v3 = include(\"../examples_2d/reel_out_v3.jl\")",
    "open_loop = include(\"open_loop.jl\")",
    "realtime = include(\"realtime.jl\")",
    "flight_replay = include(\"flight_replay.jl\")",
    "batch_run_circles = include(\"batch_run_circles.jl\")",
    "batch_load_circles = include(\"batch_load_circles.jl\")",
    "batch_load_zenith = include(\"batch_load_zenith.jl\")",
    "batch_run_zenith_then_circles = include(\"batch_run_zenith_then_circles.jl\")",
    "load_and_plot = include(\"load_and_plot.jl\")",
    "photogrammetry_aoa = include(\"photogrammetry_aoa.jl\")",
    "plot_wind_sources = include(\"plot_wind_sources.jl\")",
    "quit",
]

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
