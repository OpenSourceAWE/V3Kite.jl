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
    limits = Tuple(Iterators.flatten((coordinate - half_width, coordinate + half_width)
        for coordinate in center))
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
    states = syslog.syslog
    second_half = (length(states.time) ÷ 2):length(states.time)
    force = first.(states.winch_force)
    tracking_error = rad2deg.(wrap_to_pi.(states.heading .- states.bearing))[second_half]
    peak_force = maximum(force[second_half])
    return (mean_force = mean(force[second_half]), peak_force,
        start_force = force[1] / 1000,
        transient_time = states.time[findfirst(<=(peak_force), force)],
        mean_aoa = mean(rad2deg.(states.AoA[second_half])),
        heading_rms = sqrt(mean(tracking_error .^ 2)))
end

"Least-squares slope of `values` against `times`."
function slope(times, values)
    time_offsets = times .- mean(times)
    return sum(time_offsets .* (values .- mean(values))) / sum(time_offsets .^ 2)
end

function settle_metrics(syslog; window = 1.0)
    states = syslog.syslog
    force = first.(states.winch_force)
    last_window = findall(>=(states.time[end] - window), states.time)
    return (settle_time = states.time[end], settle_end_force = force[end],
        settle_end_force_rate = slope(states.time[last_window], force[last_window]))
end

"Rigidity figures the text quotes, from the rows of [`beam_rigidities`](@ref)."
function rigidity_metrics(sys)
    elements = beam_rigidities(sys)
    leading_edge = sort(filter(element -> element.member === :leading_edge, elements);
        by = element -> element.span)
    struts = filter(element -> element.member === :strut, elements)
    spans = [element.span for element in leading_edge]
    leading_edge_ei = [element.EI for element in leading_edge]
    stiffest = leading_edge[argmax(leading_edge_ei)]
    softest = leading_edge[argmin(leading_edge_ei)]
    strut_ratios = [linear_interpolation(spans, leading_edge_ei, strut.span) / strut.EI
        for strut in struts]
    shear_ratios = [12 * element.EI / (element.shear_coeff * element.GA * element.length^2)
        for element in leading_edge]
    return (leading_edge_max_ei = stiffest.EI, leading_edge_max_radius = stiffest.radius,
        leading_edge_min_ei = softest.EI, leading_edge_min_radius = softest.radius,
        strut_ratio_min = minimum(strut_ratios), strut_ratio_max = maximum(strut_ratios),
        shear_ratio_min = minimum(shear_ratios), shear_ratio_max = maximum(shear_ratios))
end

"Value at `x` of the piecewise-linear curve through sorted `xs` and `ys`, clamped."
function linear_interpolation(xs, ys, x)
    x <= first(xs) && return first(ys)
    x >= last(xs) && return last(ys)
    i = searchsortedlast(xs, x)
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])
end

function run_case(name, project)
    cache_path = joinpath(OUTPUT, "cache_$name")
    set_data_path(v3_data_path())
    build_time = @elapsed sam, sys = build_v3_model(project; cache_path,
        remake_settled_state = true)
    settle_log = load_log("settle_particle_dynamics_wing"; path = cache_path)
    save_structure_figure(sys, name)
    counts = model_size(sam, sys)

    logger, wall_time, completed = fly_heading_sine(sam, sys, project)
    flight_log = save_and_load_log(logger, "flight_$name"; path = OUTPUT)
    case = (; name, counts..., build_time, wall_time, completed,
        sim_time = Settings(project).sim_time, flight_metrics(flight_log)...,
        settle_metrics(settle_log)...)
    return (; case, sys, flight_log, settle_log)
end

"`\\newcommand{\\<Key><suffix>}{<value>}` for each key of `formats`, from `values`."
function write_macros(io, values, formats; suffix = "")
    for (key, format) in pairs(formats)
        value = Printf.format(Printf.Format(format), values[key])
        macro_name = replace(titlecase(replace(String(key), "_" => " ")), " " => "")
        println(io, "\\newcommand{\\$(macro_name)$(suffix)}{$value}")
    end
end

"Write every quantity the text quotes, per model and for the beam's rigidities."
function write_results(path, cases, rigidities)
    case_formats = (n_points = "%d", n_dynamic_points = "%d", n_segments = "%d",
        n_pulleys = "%d", n_states = "%d", build_time = "%.0f", cost_per_second = "%.2f",
        mean_force = "%.0f", peak_force = "%.0f", start_force = "%.1f",
        transient_time = "%.1f", mean_aoa = "%.1f", heading_rms = "%.1f",
        settle_time = "%.0f", settle_end_force = "%.0f", settle_end_force_rate = "%.0f")
    rigidity_formats = (leading_edge_max_ei = "%.0f", leading_edge_max_radius = "%.3f",
        leading_edge_min_ei = "%.0f", leading_edge_min_radius = "%.3f",
        strut_ratio_min = "%.1f", strut_ratio_max = "%.1f",
        shear_ratio_min = "%.2f", shear_ratio_max = "%.2f")
    open(path, "w") do io
        println(io, "% Written by run_simulations.jl; do not edit.")
        for case in eachrow(cases)
            values = merge(NamedTuple(case),
                (; cost_per_second = case.wall_time / case.sim_time))
            write_macros(io, values, case_formats; suffix = uppercasefirst(case.name))
        end
        write_macros(io, rigidities, rigidity_formats)
    end
end

mkpath(OUTPUT)
mkpath(FIGURES)
runs = [run_case(name, project) for (name, project) in MODELS]
cases = DataFrame([run.case for run in runs])
CSV.write(joinpath(OUTPUT, "cases.csv"), cases)
syss = [run.sys for run in runs]
write_results(joinpath(@__DIR__, "results.tex"), cases, rigidity_metrics(last(syss)))
println(cases)

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
