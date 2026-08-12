# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for simple_sinus.jl results.

Loads the "tmp_sinus" log saved by simple_sinus.jl in the `output` folder
(`examples/../output`). The heading setpoint is
read back from the logged `bearing` field (set each step in simple_sinus.jl).
The steering panel shows both values `step!` logs: the command
(`set_steering`) and the KCU's actual, tape-rate-limited value (`steering`).
They are plotted alongside heading, elevation, azimuth, AoA and L/D.

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
PROJECT          = "system_reelout.yaml"
DEPOWER_SETPOINT = 0.26

@info "Loading simulation results..."
set_data_path(v3_data_path())
OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
syslog = load_log("tmp_sinus"; path=OUTPUT_DIR)
sl = syslog.syslog

created_at = log_created_at("tmp_sinus"; path=OUTPUT_DIR)
fig_name = "V3 Kite Sinusoidal Heading Tracking"
if !isnothing(created_at)
    fig_name *= " – " * replace(first(split(created_at, '.')), "T" => "_")
end

# Skip the t=0 initial log entry (var_15/var_16 are only filled from the
# first `step!` call onward).
rng = 2:length(sl.time)

@info "Plotting results..."
p = plotx(
    sl.time[rng],
    rad2deg.(sl.elevation[rng]),
    rad2deg.(sl.azimuth[rng]),
    [rad2deg.(sl.heading[rng]), rad2deg.(sl.bearing[rng])],
    (100.0 .* sl.steering[rng], 100.0 .* sl.set_steering[rng]),
    rad2deg.(sl.AoA[rng]),
    (sl.var_15[rng], sl.var_16[rng]);
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 18,
    legendsize = 16,
    ylabels = [
        L"\mathrm{elevation}~[°]",
        L"\mathrm{azimuth}~[°]",
        L"\psi~[°]",
        L"u_{\mathrm{s}}~[\%]",
        L"\mathrm{AoA}~[°]",
        L"L/D~[-]",
    ],
    labels = [
        nothing,
        nothing,
        [L"\psi", L"\psi_{\mathrm{ref}}"],
        [L"u_{\mathrm{s}}", L"u_{\mathrm{s,set}}"],
        nothing,
        [L"L/D_{\mathrm{wing}}", L"L/D_{\mathrm{eff}}"],
    ],
    fig = fig_name,
)
display(p)
sleep(0.1)  # Allow Makie to render the plot before continuing

nothing
