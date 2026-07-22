# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Example script demonstrating a parking maneuver of the V3 kite.

The wing is settled at a fixed depower setting (REL_DEPOWER) and the
single winch is then braked, so the kite parks at a constant tether
length without any reel-out. 

Shows:
  - A plot of the results using MakieControlPlots
  - An interactive 3D replay of the logged trajectory
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using Timers
tic()
using V3Kite
using V3Kite.KitePodModels
using GLMakie
using MakieControlPlots
using LaTeXStrings
using LinearAlgebra
using Printf
toc("Loaded packages")

@info "parking.jl: Simulating a parking maneuver of the V3 kite."

# ==================== USER PARAMETERS ==================== #

SIM_TIME      = 10.0     # Total simulation time [s]
V_WIND        = 10.0     # Ground wind speed at 6 m height [m/s]
TETHER_LENGTH = 150.0    # Initial tether length [m]
ELEVATION     = 72.0     # Initial elevation angle [deg]
AZIMUTH       = 0.0      # Initial azimuth angle [deg]
REL_DEPOWER   = 0.25      # Depower setting held during parking [-]
TETHER_DIAM   = 5.0      # Tether diameter for this example [mm]
FPS           = 100      # Simulation/log frames per second (dt = 10 ms)
const PLOT    = true
REPLAY_LOG    = true     # Interactive 3D replay after simulation

# ========================================================== #

# ==================== SETTLING ==================== #

el_rad = deg2rad(ELEVATION)
az_rad = deg2rad(AZIMUTH)
position = [
    cos(el_rad) * cos(az_rad) * TETHER_LENGTH,
    cos(el_rad) * sin(az_rad) * TETHER_LENGTH,
    sin(el_rad) * TETHER_LENGTH,
]
velocity = [0.0, 0.0, 0.0]
heading = 0.0
wind_vec = [V_WIND, 0.0, 0.0]

settle_config = V3SettleConfig(
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    dt = 0.001,
    num_steps = 400,
    num_substeps = 5,
    body_damping = [0.0, 0.0, 40.0],
    start_depower = REL_DEPOWER * 100.0 + 10.0,
    course_correction_mode = :heading,
    course_correction_gain = 0.05,
    geom = V3GeomAdjustConfig(),
)

@info "Settling V3 model at rel_depower = $REL_DEPOWER..."
sam, settle_log, settle_failed = settle_wing(settle_config;
    position, velocity, heading,
    steering = 0.0, depower = REL_DEPOWER, wind_vec, remake = false)
settle_failed && error("Settling failed")
sys = sam.sys_struct

# ---- Local tether-diameter override (this example only) ----
# `sam.set` is this instance's Settings (built from system.yaml inside
# settle_wing, not the on-disk data/settings.yaml), so mutating it here
# affects only this run. We set the new diameter and propagate it to the
# main tether's segments. The DAE reads segment diameter/stiffness live
# (@register_symbolic), so this takes effect without rebuilding the
# model. Stiffness and damping scale with the cross-sectional area
# (∝ diameter²) to keep the tether physically consistent.
d_tether_old = sam.set.d_tether
d_ratio = TETHER_DIAM / d_tether_old
sam.set.d_tether = TETHER_DIAM
for tether in sys.tethers
    for seg_idx in tether.segment_idxs
        seg = sys.segments[seg_idx]
        seg.diameter       *= d_ratio        # drag area ∝ d, mass ∝ d²
        seg.unit_stiffness *= d_ratio^2      # E·A ∝ d²
        seg.unit_damping   *= d_ratio^2
    end
end
@info "Tether diameter set to $(TETHER_DIAM) mm (was $(round(d_tether_old, digits=1)) mm)"

# Brake the winch: park at constant tether length (no reel-out).
sys.winches[1].brake = true

# Convenience wrapper exposing the V3KITE query interface
# (reel_out_speed, winch_force, calc_elevation, calc_heading,
# lift_drag, total_drag).
v3kite = V3KITE(set = sam.set, kcu = KCU(sam.set), sam = sam)
toc("Settled V3 model")

# ==================== SIMULATION LOOP ==================== #

n_steps = Int(round(FPS * SIM_TIME))
dt = SIM_TIME / n_steps

logger, sys_state = create_logger(sam, n_steps)

# All plot data is stored in the syslog. Most quantities land in
# standard SysState fields (v_reelout, winch_force, elevation, heading)
# or in the computed slots filled by `log_state!` (AoA → var_04). The
# two L/D ratios have no standard field, so this example puts them in
# the free slots var_15 / var_16 before each `log_state!` call.

@info "Starting simulation with n_steps=$n_steps, dt=$dt"
sim_start_time = time()

for step in 1:n_steps
    t = step * dt

    if !sim_step!(sam; set_values = [0.0], dt, vsm_interval = 1)
        @error "Simulation failed" step
        break
    end

    lift, _ = lift_drag(v3kite)
    wing_drag, _, total_d = total_drag(v3kite)
    sys_state.var_15 = lift / wing_drag   # L/D  (wing lift / wing drag)
    sys_state.var_16 = lift / total_d     # L/D_eff (wing lift / total drag)

    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps)
        elapsed = time() - sim_start_time
        @info "Step $step/$n_steps (t=$(round(t, digits=2))s), times_rt = $(round(t / elapsed, digits=2))"
    end
end

report_performance(SIM_TIME, time() - sim_start_time)

# ==================== SAVE / LOAD LOG ==================== #

log_name = "parking_lt_$(Int(round(TETHER_LENGTH)))"
save_log(logger, log_name)
syslog = load_log(log_name)
sl = syslog.syslog

# Skip the t=0 initial log entry (var slots are only filled from the
# first simulation step onward). The syslog already has the right
# length even if the loop stopped early on a failed step.
rng = 2:length(sl.time)
t_plot = sl.time[rng]

# ==================== PLOTTING ==================== #

if PLOT
    @info "Plotting results..."
    p = plotx(
        t_plot,
        first.(sl.v_reelout[rng]),         # winch 1 reel-out speed
        first.(sl.winch_force[rng]),       # winch 1 force
        rad2deg.(sl.elevation[rng]),
        rad2deg.(sl.heading[rng]),
        rad2deg.(sl.var_04[rng]),          # kite AoA
        fill(REL_DEPOWER, length(rng)),    # depower held constant
        (sl.var_15[rng], sl.var_16[rng]);  # L/D_wing, L/D_eff
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
            nothing,
            [L"L/D_{\mathrm{wing}}", L"L/D_{\mathrm{eff}}"],
        ],
        fig = "V3 Kite Parking",
    )
    display(p)
    sleep(0.1)  # Allow Makie to render the plot before continuing
end

# ==================== INTERACTIVE REPLAY ==================== #

if REPLAY_LOG
    scene = replay(syslog, sam.sys_struct; show_panes = false)
    display(scene)
end

nothing
