# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Example script demonstrating a simple reel-out maneuver of the V3 kite model.
"""

# TODO: Plot heading

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers
tic()
using V3Kite
using V3Kite.KitePodModels
using VortexStepMethod
using LinearAlgebra
using ControlPlots
using Printf
toc("Loaded packages")

@info "reel_out_v3.jl: Simulating a simple reel-out maneuver of the V3 kite model."

TETHER_LENGTH = 150.0 # m
V_WIND        = 9.51  # m/s

gc = V3GeomAdjustConfig()

data_path = v3_data_path()
set_data_path(data_path)
set = Settings("system.yaml")
set.g_earth = 9.81
set.wind_vec = [V_WIND, 0.0, 0.0]
set.l_tether = TETHER_LENGTH
set.profile_law = 0

# the following values can be changed to match your interest
dt = 0.05
STEPS = 600
const PLOT = true
FRONT_VIEW = false
ZOOM = false
PRINT = false
STATISTIC = false
ALPHA_ZERO = 8.8 
# end of user parameter section #

set.alpha_zero = ALPHA_ZERO

v_time = zeros(STEPS)
v_speed = zeros(STEPS)
v_force = zeros(STEPS)
v_elevation = zeros(STEPS)

kcu::KCU = KCU(set)

source_struc = joinpath(data_path, "struc_geometry.yaml")
source_aero = joinpath(data_path, "aero_geometry.yaml")
vsm_path = joinpath(data_path, "vsm_settings.yaml")
vsm_set = VortexStepMethod.VSMSettings(vsm_path; data_prefix=false)
vsm_set.wings[1].geometry_file = source_aero

@info "Creating V3 model..."
sys = load_sys_struct_from_yaml(source_struc;
    system_name=V3_MODEL_NAME, set,
    wing_type=REFINE, vsm_set)
sam = SymbolicAWEModel(set, sys)

# create an instance of the V3KITE struct
v3kite = V3KITE(set=set, kcu=kcu, sam=sam, sys=sys)
toc("Created V3KITE instance")

function simulate(integrator, steps, plot=false)
    iter = 0
    for i in 1:steps
        if PRINT
            lift, drag = lift_drag(v3kite)
            @printf "%.2f: " round(integrator.t, digits=2)
            println("lift, drag  [N]: $(round(lift, digits=2)), $(round(drag, digits=2))")
        end
        dforce = 0.0
        if integrator.t > 15.0
            dforce = +4.5
        end
        winch = v3kite.sys.winches[1]
        force = norm(winch.force)
        r = winch.drum_radius
        n = winch.gear_ratio
        set_torque = -r/n * force + dforce
        v_time[i] = integrator.t
        v_speed[i] = winch.vel
        v_force[i] = force
        v_elevation[i] = rad2deg(KiteUtils.calc_elevation(v3kite.sys.wings[1].pos_w))
        sim_step!(v3kite.sam; set_values=[set_torque], dt, vsm_interval=1)
        iter += 1

        if plot
            reltime = i*dt-dt
            if mod(i, 5) == 1
                tether = v3kite.sys.tethers[1]
                n_segs = length(tether.segment_idxs)
                pos = [v3kite.sys.points[tether.start_point_idx].pos_w]
                for si in tether.segment_idxs
                    seg = v3kite.sys.segments[si]
                    push!(pos, v3kite.sys.points[seg.point_idxs[2]].pos_w)
                end
                if FRONT_VIEW
                    plot2d(pos, reltime; zoom=ZOOM, front=true,
                                        segments=n_segs, fig="front_view")
                else
                    plot2d(pos, reltime; zoom=ZOOM, front=false,
                                        segments=n_segs, fig="side_view")
                end
            end
        end
    end
    iter / steps
end

@info "Initializing model..."
integrator = init!(v3kite.sam; remake=false, remake_vsm=true)
v3kite.sys.winches[1].brake = false

simulate(integrator, STEPS, true)

if PLOT
    local p
    p = plotx(v_time, v_speed, v_force, v_elevation; 
    ylabels=["v_reelout  [m/s]","tether_force [N]","elevation [deg]"], fig="winch")
    display(p)
end
nothing
