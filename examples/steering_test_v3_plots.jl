# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for steering_test_v3.jl results.

Loads the "tmp_steering" log and re-runs the identification of `src/turn_rate_id.jl`
on it, so the numbers printed here are the same ones steering_test_v3.jl
printed — without re-simulating.

Three figures, following the KPS4 original:

1. Time series: elevation, heading with the relay band, the steering command
   (`set_steering`) against the KCU's actual tape position (`steering`) and its
   delayed version used for the fit, the frame-corrected turn rate per apparent
   wind speed, and the AoA. The gap between command and actual is the
   slew-limited actuator and is expected to be large.
2. `G/G_mean`, the scatter of the single-parameter turn-rate gain about its
   mean — this is the quantity the ≤35 % criterion is about.
3. Delayed steering against the steering the fitted law needs to explain the
   measured turn rate (`est_steering` of the original). Overlaying curves mean
   the two-parameter law reproduces the run.

Plus the `|ψ̇|` vs `|u_s·v_a|` scatter from `V3Kite.plot_yaw_rate_vs_steering`,
whose slope is the same gain seen a different way and which is directly
comparable to the flight-data and circle-sweep logs.

Run from the REPL after (or instead of, if "tmp_steering" already exists)
running steering_test_v3.jl:

    include("steering_test_v3_plots.jl")
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using GLMakie
using MakieControlPlots
using LaTeXStrings
using V3Kite

# Must match the corresponding constants in steering_test_v3.jl.
DT               = 0.05/3
T_START          = 10.0
HEADING_OFFSET   = 10.0
START_STEERING   = 0.05
MIN_STEERING_FIT = START_STEERING / 2
MAX_REL_STD      = 0.35

@info "Loading simulation results..."
set_data_path(v3_data_path())
syslog = load_log("tmp_steering")
sl = syslog.syslog

created_at = log_created_at("tmp_steering")
fig_name = "V3 Kite Steering Response"
if !isnothing(created_at)
    fig_name *= " – " * replace(first(split(created_at, '.')), "T" => "_")
end

r = identify_turn_rate_law(sl; dt = DT, t_start = T_START,
                           min_steering = MIN_STEERING_FIT)
print("\n", format_turn_rate_report(r; max_rel_std = MAX_REL_STD))

# ==================== TIME SERIES ======================== #

# Analysis-window indices into the full log, so the logged channels line up with
# the arrays returned by identify_turn_rate_law (which start at index 2).
i0 = searchsortedfirst(sl.time, r.t_start)
rng = i0:searchsortedlast(sl.time, r.t_end)
band = fill(HEADING_OFFSET, length(rng))

@info "Plotting results..."
p1 = plotx(
    sl.time[rng],
    rad2deg.(sl.elevation[rng]),
    [rad2deg.(wrap_to_pi.(sl.heading[rng])), band, -band],
    [100.0 .* sl.set_steering[rng], 100.0 .* sl.steering[rng], 100.0 .* r.us_del],
    rad2deg.(r.rate) ./ r.v_app,
    rad2deg.(sl.AoA[rng]);
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 18,
    legendsize = 16,
    ylabels = [
        L"\mathrm{elevation}~[°]",
        L"\psi~[°]",
        L"u_{\mathrm{s}}~[\%]",
        L"\dot{\psi}/v_{\mathrm{a}}~[°/m]",
        L"\mathrm{AoA}~[°]",
    ],
    labels = [
        nothing,
        [L"\psi", L"+\psi_\mathrm{band}", L"-\psi_\mathrm{band}"],
        [L"u_{\mathrm{s,set}}", L"u_{\mathrm{s}}", L"u_{\mathrm{s,delayed}}"],
        nothing,
        nothing,
    ],
    fig = fig_name,
)
display(p1)
sleep(0.1)  # Allow Makie to render the plot before continuing

# =============== GAIN SCATTER AND FIT ==================== #

# both GLMakie and MakieControlPlots export `plot`, so it has to be qualified
p2 = MakieControlPlots.plot(r.time, r.G ./ r.G_mean;
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ylabel = L"G/G_{\mathrm{mean}}~[-]",
    fig = fig_name * " – turn rate law")
display(p2)
sleep(0.1)

p3 = plotx(r.time,
    [r.us_del, r.us_est];
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 18,
    legendsize = 16,
    ylabels = [L"u_{\mathrm{s}}~[-]"],
    labels = [[L"u_{\mathrm{s,delayed}}", L"u_{\mathrm{s,est}}"]],
    fig = fig_name * " – measured vs fitted steering")
display(p3)
sleep(0.1)

# Slope view, directly comparable to the flight-data and circle-sweep logs.
p4 = plot_yaw_rate_vs_steering(syslog; source = :heading, dt = DT,
                               min_steering = MIN_STEERING_FIT)
display(p4)
sleep(0.1)

nothing
