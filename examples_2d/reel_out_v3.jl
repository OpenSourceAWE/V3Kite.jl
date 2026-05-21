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
dt::Float64 = 0.05
STEPS = 600
const PLOT = true
FRONT_VIEW = false
ZOOM = true
PRINT = false
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
toc("Created system structure")
sam = SymbolicAWEModel(set, sys)
toc("Created symbolic model")

# create an instance of the V3KITE struct
v3kite = V3KITE(set=set, kcu=kcu, sam=sam)
toc("Created V3KITE instance")
