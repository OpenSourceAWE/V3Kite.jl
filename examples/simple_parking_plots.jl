# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for simple_parking.jl results.

Loads the "tmp_run" log saved by simple_parking.jl and reproduces the same
plot as parking.jl, entirely from logged data — no re-simulation needed.
v_reelout, winch_force, elevation, heading and AoA come straight from the
syslog; the L/D ratios come from the SysState spare slots that `step!` fills
during simple_parking.jl's simulation loop (var_15 = L/D_wing, var_16 =
L/D_eff). The single-winch V3 model has no l_diff panel; the depower panel
instead shows the actual value (`depower`, the KCU's tape-lagged fraction)
against the command (`var_14`, written by `step!`), matching parking.jl. Both
come from the log, so nothing here needs to be kept in sync with
simple_parking.jl by hand.

Run from the REPL after (or instead of, if "tmp_run" already exists) running
simple_parking.jl:

    include("simple_parking_plots.jl")
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using GLMakie
using MakieControlPlots
using LaTeXStrings
using V3Kite

@info "Loading simulation results..."
set_data_path(v3_data_path())
syslog = load_log("tmp_run")
sl = syslog.syslog

created_at = log_created_at("tmp_run")
fig_name = "V3 Kite Parking"
if !isnothing(created_at)
    fig_name *= " – " * replace(first(split(created_at, '.')), "T" => "_")
end

# Skip the t=0 initial log entry (depower/var_14/var_15/var_16 are only
# filled from the first `step!` call onward).
rng = 2:length(sl.time)

@info "Plotting results..."
p = plotx(
    sl.time[rng],
    first.(sl.v_reelout[rng]),
    first.(sl.winch_force[rng]),
    rad2deg.(sl.elevation[rng]),
    rad2deg.(sl.heading[rng]),
    rad2deg.(sl.AoA[rng]),
    (sl.depower[rng], sl.var_14[rng]),
    (sl.var_15[rng], sl.var_16[rng]);
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 16,
    legendsize = 16,
    ylabels = [
        L"v_{\mathrm{ro}}~[\mathrm{m/s}]",
        L"F_{\mathrm{t}}~[\mathrm{N}]",
        L"\mathrm{elevation}~[°]",
        L"\mathrm{heading}~[°]",
        L"\mathrm{AoA}~[°]",
        L"u_{\mathrm{d}}~[-]",
        L"L/D~[-]",
    ],
    labels = [
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [L"u_{\mathrm{d}}", L"u_{\mathrm{d,set}}"],
        [L"L/D_{\mathrm{wing}}", L"L/D_{\mathrm{eff}}"],
    ],
    fig = fig_name,
)
display(p)
sleep(0.1)  # Allow Makie to render the plot before continuing

nothing
