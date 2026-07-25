# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for reel_out_v3.jl results.

Loads the "tmp_reel_out" log saved by reel_out_v3.jl and reproduces the same
plot, entirely from logged data — no re-simulation needed.

Run from the REPL after (or instead of, if "tmp_reel_out" already exists)
running reel_out_v3.jl:

    include("reel_out_v3_plots.jl")
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using LinearAlgebra: norm
using MakieControlPlots: plotx
using V3Kite: set_data_path, load_log, v3_data_path

T_MIN = 0.0 # only plot results from T_MIN onwards

@info "Loading simulation results..."
set_data_path(v3_data_path())
syslog = load_log("tmp_reel_out")
sl = syslog.syslog
mask = sl.time .>= T_MIN

@info "Plotting results..."
p = plotx(sl.time[mask], first.(sl.v_reelout[mask]), first.(sl.winch_force[mask]),
    rad2deg.(sl.elevation[mask]), rad2deg.(sl.heading[mask]), norm.(sl.v_wind_kite[mask]);
    ysize= 16,
    ylabels=["v_reelout  [m/s]", "tether_force [N]", "elevation [deg]",
             "heading [deg]", "wind_speed_kite [m/s]"], fig="V3 Kite reel-out")
display(p)
sleep(0.1)  # Allow Makie to render the plot before continuing

nothing
