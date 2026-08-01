# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for simple_fig8.jl results.

Loads the "fig8_run" log saved by simple_fig8.jl. Produces two figures:

1. the flown pattern in the (azimuth, elevation) plane, with the reference
   lemniscate and the attractor track overlaid — the plot that shows at a glance
   whether the pattern is being flown or only approximated;
2. a stacked time-series figure: cross-track error, elevation (with the pattern
   centre), heading vs commanded course, steering command vs the KCU's actual
   tape-lagged value, and tether force.

Run from the REPL after (or instead of, if "fig8_run" already exists) running
simple_fig8.jl:

    include("simple_fig8_plots.jl")
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using GLMakie
using MakieControlPlots
using LaTeXStrings
using V3Kite

# Must match the corresponding constants in simple_fig8.jl (used only to
# redraw the reference path).
F8_A = 50.0
F8_B = 25.0
F8_C = 0.0
F8_D = 0.0

@info "Loading simulation results..."
set_data_path(v3_data_path())
syslog = load_log("fig8_run")
sl = syslog.syslog

created_at = log_created_at("fig8_run")
fig_name = "V3 Kite Figure-of-Eight"
if !isnothing(created_at)
    fig_name *= " – " * replace(first(split(created_at, '.')), "T" => "_")
end

# Skip the t=0 initial log entry (the guidance slots are only filled from the
# first `step!` call onward).
rng = 2:length(sl.time)

az_deg = rad2deg.(sl.azimuth[rng])
el_deg = rad2deg.(sl.elevation[rng])

# Reference path at the FINAL pattern centre (var_04), so the overlay matches
# the run even when the centre was walked (WALK_RATE > 0).
el_c_end = Float64(sl.var_04[end])
ref_az, ref_el = figure_eight_path(F8_A, F8_B, F8_C, F8_D, 0.0, el_c_end, 0.0, 361)

@info "Plotting the pattern..."
p1 = plotxy(
    [az_deg, ref_az, Float64.(sl.var_02[rng])],
    [el_deg, ref_el, Float64.(sl.var_03[rng])];
    xlabel = L"\mathrm{azimuth}~[°]",
    ylabel = L"\mathrm{elevation}~[°]",
    legend = [L"\mathrm{flown}", L"\mathrm{reference}", L"\mathrm{attractor}"],
    fig = fig_name * " – pattern",
)
display(p1)
sleep(0.1)

@info "Plotting the time series..."
p2 = plotx(
    sl.time[rng],
    sl.var_01[rng],
    [el_deg, Float64.(sl.var_04[rng])],
    [rad2deg.(sl.heading[rng]), rad2deg.(sl.bearing[rng])],
    (100.0 .* sl.steering[rng], 100.0 .* sl.set_steering[rng]),
    getindex.(sl.winch_force[rng], 1);
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 18,
    legendsize = 16,
    ylabels = [
        L"d~[°]",
        L"\mathrm{elevation}~[°]",
        L"\psi~[°]",
        L"u_{\mathrm{s}}~[\%]",
        L"F_{\mathrm{tether}}~[\mathrm{N}]",
    ],
    labels = [
        nothing,
        [L"\mathrm{kite}", L"\mathrm{pattern~centre}"],
        [L"\psi", L"\chi_{\mathrm{set}}"],
        [L"u_{\mathrm{s}}", L"u_{\mathrm{s,set}}"],
        nothing,
    ],
    fig = fig_name * " – time series",
)
display(p2)
sleep(0.1)

nothing
