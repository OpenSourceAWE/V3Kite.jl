# Copyright (c) 2025 Jelle Poland, Bart van de Lint
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
ZOOM = true
PRINT = true
STATISTIC = false
ALPHA_ZERO = 8.8 
# end of user parameter section #

set.alpha_zero = ALPHA_ZERO

v_time = zeros(STEPS)
v_speed = zeros(STEPS)
v_force = zeros(STEPS)

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
    #     dforce = 0.0
    #     if kps4.t_0 > 15.0
    #         dforce = +4.5
    #     end
    #     force = norm(kps4.forces[1])
    #     r = set.drum_radius
    #     n = set.gear_ratio
    #     set_torque = -r/n * force + dforce
    #     v_time[i] = kps4.t_0
    #     v_speed[i] = kps4.v_reel_out
    #     v_force[i] = winch_force(kps4)
    #     next_step!(kps4, integrator; set_torque, dt)
    #     iter += kps4.iter
        
    #     if plot
    #         reltime = i*dt-dt
    #         if mod(i, 5) == 1
    #             if FRONT_VIEW
    #                 plot2d(kps4.pos, reltime; zoom=ZOOM, front=true,
    #                                         segments=set.segments, fig="front_view")
    #             else
    #                 plot2d(kps4.pos, reltime; zoom=ZOOM, front=false, xlim=(37, 78),
    #                                         segments=set.segments, fig="side_view")
    #             end
    #         end
    #     end
    end
    iter / steps
end

@info "Initializing model..."
integrator = init!(v3kite.sam; remake=false, remake_vsm=true)
v3kite.sys.winches[1].brake = false

simulate(integrator, STEPS, true)
