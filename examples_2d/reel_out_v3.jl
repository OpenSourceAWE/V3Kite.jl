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

gc = V3GeomAdjustConfig()

# the following values can be changed to match your interest
dt = 0.05/3
STEPS = 600*3
const PLOT = true
FRONT_VIEW = false
ZOOM = false
PRINT = true
STATISTIC = false
ALPHA_ZERO = 8.8 
TETHER_LENGTH = 150.0 # m
V_WIND        = 9.51  # m/s
# end of user parameter section #

data_path = v3_data_path()
set_data_path(data_path)
set = Settings("system.yaml")
set.g_earth = 9.81
set.wind_vec = [V_WIND, 0.0, 0.0]
set.l_tether = TETHER_LENGTH
set.profile_law = 3  # 3=EXPLOG, matches KiteModels default

set.alpha_zero = ALPHA_ZERO
set.mass = 6.2    # kite mass [kg]
set.d_tether = 4.0  # tether diameter [mm]
set.drum_radius = 0.1615  # radius of the drum                    [m]
set.gear_ratio = 6.2      # gear ratio of the winch               [-]
set.f_coulomb = 122.0     # coulomb friction                      [N]
set.c_vf = 30.6           # coefficient for the viscous friction  [Ns/m]

v_time = zeros(STEPS)
v_speed = zeros(STEPS)
v_force = zeros(STEPS)
v_elevation = zeros(STEPS)
v_heading = zeros(STEPS)
v_wind_speed = zeros(STEPS)

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
sys.points[1].extra_mass = 8.4   # KCU mass [kg]
sam = SymbolicAWEModel(set, sys)

# create an instance of the V3KITE struct
v3kite = V3KITE(set=set, kcu=kcu, sam=sam)
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
        force = winch_force(v3kite)
        r = set.drum_radius
        n = set.gear_ratio
        set_torque = -r/n * force + dforce
        # This should be the same as the above, but it isn't. Why?
        # set_torque = force_to_torque(-force, v3kite.sys) + dforce
        v_time[i] = integrator.t
        v_speed[i] = reel_out_speed(v3kite)
        v_force[i] = force
        v_elevation[i] = rad2deg(calc_elevation(v3kite))
        v_heading[i] = rad2deg(calc_heading(v3kite))
        v_wind_speed[i] = norm(v_wind_kite(v3kite))
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
    p = plotx(v_time, v_speed, v_force, v_elevation, v_heading, v_wind_speed;
    ysize= 11,
        ylabels=["v_reelout  [m/s]", "tether_force [N]", "elevation [deg]",
                 "heading [deg]", "wind_speed_kite [m/s]"], fig="winch")
    display(p)
end
@printf "\nMass kite: %.1f kg,  mass KCU: %.1f kg,  tether diameter: %.1f mm\n" set.mass sys.points[1].extra_mass set.d_tether
println("Number of states: $(states(v3kite))")
nothing
