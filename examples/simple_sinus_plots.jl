# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for simple_sinus.jl results.

Loads the "tmp_sinus" log saved by simple_sinus.jl. The heading setpoint is
recomputed from the logged time vector and MAX_HEADING/HEADING_PERIOD (kept
in sync with simple_sinus.jl) rather than logged, since the L/D spare slots
(var_15 = L/D_wing, var_16 = L/D_eff) already use the two free SysState
slots. The commanded steering (`set_steering`, logged by `step!`) is plotted
alongside heading, elevation, azimuth, AoA and L/D.

Run from the REPL after (or instead of, if "tmp_sinus" already exists)
running simple_sinus.jl:

    include("simple_sinus_plots.jl")
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using GLMakie
using MakieControlPlots
using LaTeXStrings
using V3Kite

# Must match the corresponding constants in simple_sinus.jl.
PROJECT          = "system_cabauw.yaml"
DEPOWER_SETPOINT = 0.25
MAX_HEADING      = 40.0     # deg
HEADING_PERIOD   = 30.0     # s

@info "Loading simulation results..."
set_data_path(v3_data_path())
syslog = load_log("tmp_sinus")
sl = syslog.syslog

# Skip the t=0 initial log entry (var_15/var_16 are only filled from the
# first `step!` call onward).
rng = 2:length(sl.time)

heading_setpoint = deg2rad(MAX_HEADING) .* sin.(2pi / HEADING_PERIOD .* sl.time[rng])

@info "Plotting results..."
p = plotx(
    sl.time[rng],
    rad2deg.(sl.elevation[rng]),
    rad2deg.(sl.azimuth[rng]),
    [rad2deg.(sl.heading[rng]), rad2deg.(heading_setpoint)],
    sl.set_steering[rng],
    rad2deg.(sl.AoA[rng]),
    (sl.var_15[rng], sl.var_16[rng]);
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 16,
    legendsize = 16,
    ylabels = [
        L"\mathrm{elevation}~[°]",
        L"\mathrm{azimuth}~[°]",
        L"\psi~[°]",
        L"u_{\mathrm{s}}~[-]",
        L"\mathrm{AoA}~[°]",
        L"L/D~[-]",
    ],
    labels = [
        nothing,
        nothing,
        [L"\psi", L"\psi_{\mathrm{ref}}"],
        nothing,
        nothing,
        [L"L/D_{\mathrm{wing}}", L"L/D_{\mathrm{eff}}"],
    ],
    fig = "V3 Kite Sinusoidal Heading Tracking",
)
display(p)
sleep(0.1)  # Allow Makie to render the plot before continuing

nothing
