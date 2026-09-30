# Copyright (c) 2026 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

# Settle and fly the particle and the beam V3 kite through the same heading
# manoeuvre, from a cache of this paper's own under `output/`. Draws the paper's
# figures to `figures/` and writes the numbers its text quotes to `results.tex`.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "examples"))

using V3Kite
using GLMakie
using MakieControlPlots
using CSV
using DataFrames
using Printf
using Statistics

OUTPUT = joinpath(@__DIR__, "output")
FIGURES = joinpath(@__DIR__, "figures")
MODELS = ["psm" => "system_psm.yaml", "beam" => "system_beam.yaml"]
LABELS = ["particle", "beam"]

function model_size(sam, sys)
    return (n_points = length(sys.points),
        n_dynamic_points = count(point -> point.type == SymbolicAWEModels.DYNAMIC,
            sys.points),
        n_segments = length(sys.segments),
        n_pulleys = length(sys.pulleys),
        n_states = length(sam.integrator.u))
end

function save_structure_figure(sys, name; half_width = 5.5)
    wing_points = [point.pos_w for point in sys.points if point.is_wing_node]
    center = sum(wing_points) / length(wing_points) .- [0.0, 0.0, 2.0]
    limits = Tuple(Iterators.flatten((c - half_width, c + half_width) for c in center))
    fig = Figure(size = (800, 500))
    ax = Axis3(fig[1, 1]; aspect = :data, limits, azimuth = 0.15pi,
        elevation = 0.3pi, viewmode = :fitzoom, protrusions = 0)
    hidedecorations!(ax)
    hidespines!(ax)
    plot!(ax, sys; show_orient = false, plot_aero = false, plot_airfoils = false,
        beam_opacity = 0.5)
    path = joinpath(FIGURES, "structure_$name.png")
    GLMakie.save(path, fig; px_per_unit = 3)
    image = GLMakie.FileIO.load(path)
    drawn = findall(pixel -> pixel != eltype(image)(1, 1, 1), image)
    rows, cols = extrema(index[1] for index in drawn), extrema(index[2] for index in drawn)
    GLMakie.save(path, image[first(rows):last(rows), first(cols):last(cols)])
end

function flight_metrics(syslog)
    sl = syslog.syslog
    second_half = (length(sl.time) ÷ 2):length(sl.time)
    force = first.(sl.winch_force)[second_half]
    tracking_error = rad2deg.(wrap_to_pi.(sl.heading .- sl.bearing))[second_half]
    return (mean_force = mean(force), peak_force = maximum(force),
        mean_aoa = mean(rad2deg.(sl.AoA[second_half])),
        heading_rms = sqrt(mean(tracking_error .^ 2)))
end

function settle_metrics(syslog)
    sl = syslog.syslog
    force = first.(sl.winch_force)
    return (settle_time = sl.time[end], settle_end_force = force[end],
        settle_end_force_rate = (force[end] - force[end - 1]) /
            (sl.time[end] - sl.time[end - 1]))
end

function run_case(name, project)
    cache_path = joinpath(OUTPUT, "cache_$name")
    set_data_path(v3_data_path())
    build_time = @elapsed sam, sys = build_v3_model(project; cache_path,
        remake_settled_state = true)
    settle_log = load_log("settle_particle_dynamics_wing"; path = cache_path)
    save_structure_figure(sys, name)
    size = model_size(sam, sys)

    logger, wall_time, completed = fly_heading_sine(sam, sys, project)
    save_log(logger, "flight_$name"; path = OUTPUT)
    flight_log = load_log("flight_$name"; path = OUTPUT)
    case = (; name, size..., build_time, wall_time, completed,
        sim_time = Settings(project).sim_time, flight_metrics(flight_log)...,
        settle_metrics(settle_log)...)
    return (; case, sys, flight_log, settle_log)
end

"`\\newcommand{\\<name><Model>}{<value>}` for every quantity the text quotes."
function write_results(path, cases)
    formats = (n_points = "%d", n_dynamic_points = "%d", n_segments = "%d",
        n_pulleys = "%d", n_states = "%d", build_time = "%.0f",
        mean_force = "%.0f", peak_force = "%.0f", mean_aoa = "%.1f",
        heading_rms = "%.1f", settle_time = "%.0f", settle_end_force = "%.0f",
        settle_end_force_rate = "%.0f")
    open(path, "w") do io
        println(io, "% Written by run_simulations.jl; do not edit.")
        for case in eachrow(cases)
            model = uppercasefirst(case.name)
            for (key, format) in pairs(formats)
                value = Printf.format(Printf.Format(format), case[key])
                macro_name = replace(titlecase(replace(String(key), "_" => " ")), " " => "")
                println(io, "\\newcommand{\\$(macro_name)$(model)}{$value}")
            end
            cost = @sprintf("%.2f", case.wall_time / case.sim_time)
            println(io, "\\newcommand{\\CostPerSecond$(model)}{$cost}")
        end
    end
end

mkpath(OUTPUT)
mkpath(FIGURES)
runs = [run_case(name, project) for (name, project) in MODELS]
cases = DataFrame([run.case for run in runs])
CSV.write(joinpath(OUTPUT, "cases.csv"), cases)
write_results(joinpath(@__DIR__, "results.tex"), cases)
println(cases)

syss = [run.sys for run in runs]
GLMakie.save(joinpath(FIGURES, "beam_rigidities.png"), plot_beam_rigidities(last(syss));
    px_per_unit = 3)
GLMakie.save(joinpath(FIGURES, "twist.png"),
    plot_twist_dist(syss; labels = LABELS, title = false, limits = (-10, 10));
    px_per_unit = 3)
GLMakie.save(joinpath(FIGURES, "flight.png"),
    MakieControlPlots.plot(syss, [run.flight_log for run in runs];
        plot_default = false, plot_winch_force = true, plot_v_app = true,
        plot_heading = true, plot_course = false, plot_aoa = true,
        suffixes = LABELS, size = (700, 700), ticklabelsize = 16,
        label_fontsize = 20, legendsize = 14); px_per_unit = 3)
GLMakie.save(joinpath(FIGURES, "settling.png"),
    MakieControlPlots.plot(syss, [run.settle_log for run in runs];
        plot_default = false, plot_winch_force = true, plot_v_app = true,
        plot_heading = false, plot_course = false, suffixes = LABELS,
        size = (700, 400), ticklabelsize = 16, label_fontsize = 20, legendsize = 14);
    px_per_unit = 3)
