# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0
#
# Loads the log file written by reel_out_4p_torque_control.jl and
# recreates its reel-out speed/force/elevation/heading/wind plot from
# the saved data, without re-running the simulation.

using Pkg
if dirname(Pkg.project().path) != @__DIR__
    Pkg.activate(@__DIR__)
end

using V3Kite, LinearAlgebra
using MakieControlPlots

LOGFILE = "reel_out_4p_torque_control"
T_MIN = 10.0 # only plot results from T_MIN onwards

reel_log = load_log(LOGFILE)
sl = reel_log.syslog

mask         = sl.time .>= T_MIN
v_time       = sl.time[mask]
v_speed      = first.(sl.v_reelout)[mask]
v_force      = first.(sl.winch_force)[mask]
v_elevation  = rad2deg.(sl.elevation)[mask]
v_heading    = rad2deg.(wrap_to_pi.(sl.heading))[mask]
v_wind_speed = norm.(sl.v_wind_kite)[mask]

p = plotx(v_time, v_speed, v_force, v_elevation, v_heading, v_wind_speed;
    ylabels=["v_reelout  [m/s]", "tether_force [N]", "elevation [deg]", "heading [deg]", "wind at kite [m/s]"],
    fig="winch_KiteModels")
display(p)
nothing
