# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Example script demonstrating a simple reel-out maneuver of the V3 kite model.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers
tic()
using V3Kite
using LinearAlgebra
using MakieControlPlots
using Printf
toc("Loaded packages")

@info "reel_out_v3.jl: Simulating a simple reel-out maneuver of the V3 kite model."

# the following values can be changed to match your interest
dt    = 0.05/3
STEPS = 600*3
const PLOT = true
FRONT_VIEW = false
ZOOM = false
DEPOWER_SETPOINT = 0.27 # tuned so winch_force(15s) ≈ 1050 N, matching winch_KiteModels
REL_STEERING  = -0.0016 # tuned so heading(end) is between 0 and 2 degrees
TETHER_LENGTH = 150.0 # m
V_WIND        = 9.51  # m/s
T_MIN =  0.0          # only plot results from T_MIN onwards
AERO_MODE = AeroDirect()
# end of user parameter section #

@info "Initializing model..."
s = init(V_WIND, TETHER_LENGTH; system_yaml = "system_reelout.yaml",
                                depower_setpoint = DEPOWER_SETPOINT, dt,
                                sim_time = STEPS*dt, aero_mode = AERO_MODE)
toc("Initialized V3KITE instance")

function simulate(s, steps, plot=false)
    iter = 0
    for i in 1:steps
        dforce = 0.0
        if s.sys_state.time > 15.0
            dforce = +4.5
        end
        force = winch_force(s)
        winch = s.sys.winches[1]
        set_torque = -winch.drum_radius / winch.gear_ratio * force + dforce
        step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering = REL_STEERING, set_torque)
        iter += 1

        if plot
            reltime = i*dt-dt
            if mod(i, 5) == 1
                tether = s.sys.tethers[1]
                n_segs = length(tether.segment_idxs)
                pos = [s.sys.points[tether.start_point_idx].pos_w]
                for si in tether.segment_idxs
                    seg = s.sys.segments[si]
                    push!(pos, s.sys.points[seg.point_idxs[2]].pos_w)
                end
                if FRONT_VIEW
                    plot2d(pos, reltime; zoom=ZOOM, front=true,
                                        segments=n_segs, fig="front_view")
                else
                    plot2d(pos, reltime; zoom=ZOOM, front=false,
                                        segments=n_segs, fig="side_view")
                end
                sleep(0.001) # yield to let GLMakie's render task redraw
            end
        end
    end
    iter / steps
end

GC.enable(false)
try
    simulate(s, STEPS, true)
catch e
    @error "Simulation stopped early at t≈$(round(s.sys_state.time, digits=2))s" exception=(e, catch_backtrace())
finally
    GC.enable(true)
end

@info "Save the log"
save_log(s.logger, "tmp_reel_out"; colmeta=timestamp_colmeta())

if PLOT
    local p
    syslog = load_log("tmp_reel_out")
    sl = syslog.syslog
    mask = sl.time .>= T_MIN
    created_at = log_created_at("tmp_reel_out")
    fig_name = "winch"
    if !isnothing(created_at)
        fig_name *= " – " * replace(first(split(created_at, '.')), "T" => "_")
    end
    p = plotx(sl.time[mask], first.(sl.v_reelout[mask]), first.(sl.winch_force[mask]),
        rad2deg.(sl.elevation[mask]), rad2deg.(sl.heading[mask]), norm.(sl.v_wind_kite[mask]),
        rad2deg.(sl.AoA[mask]);
        ysize= 16,
        ylabels=["v_reelout  [m/s]", "tether_force [N]", "elevation [deg]",
                 "heading [deg]", "wind_speed_kite [m/s]", "AoA [deg]"], fig=fig_name)
    display(p)
end
@printf "\nMass kite: %.1f kg,  mass KCU: %.1f kg,  tether diameter: %.1f mm\n" s.set.mass s.sys.points[1].extra_mass s.set.d_tether
println("Number of states: $(states(s))")
nothing
