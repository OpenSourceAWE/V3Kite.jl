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
REL_DEPOWER   = 0.3      # Depower setting held during parking [-]
FPS           = 120      # Simulation/log frames per second
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

# Plot data collected each step via the V3KITE query interface.
v_reelout_log   = Float64[]
winch_force_log = Float64[]
elevation_log   = Float64[]
heading_log     = Float64[]
aoa_log         = Float64[]
rel_depower_log = Float64[]
ld_wing_log     = Float64[]   # L/D  (wing lift / wing drag)
ld_eff_log      = Float64[]   # L/D_eff (wing lift / total drag)

@info "Starting simulation" n_steps dt
sim_start_time = time()

for step in 1:n_steps
    t = step * dt

    lift, _ = lift_drag(v3kite)
    wing_drag, _, total_d = total_drag(v3kite)

    push!(v_reelout_log,   reel_out_speed(v3kite))
    push!(winch_force_log, winch_force(v3kite))
    push!(elevation_log,   rad2deg(calc_elevation(v3kite)))
    push!(heading_log,     rad2deg(calc_heading(v3kite)))
    push!(aoa_log,         rad2deg(compute_kite_aoa(sys)))
    push!(rel_depower_log, REL_DEPOWER)
    push!(ld_wing_log,     lift / wing_drag)
    push!(ld_eff_log,      lift / total_d)

    if !sim_step!(sam; set_values = [0.0], dt, vsm_interval = 1)
        @error "Simulation failed" step
        break
    end

    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps)
        elapsed = time() - sim_start_time
        @info "Step $step/$n_steps (t=$(round(t, digits=2))s)" times_realtime=round(t / elapsed, digits=2)
    end
end

report_performance(SIM_TIME, time() - sim_start_time)

# ==================== SAVE / LOAD LOG ==================== #

log_name = "parking_lt_$(Int(round(TETHER_LENGTH)))"
save_log(logger, log_name)
syslog = load_log(log_name)
sl = syslog.syslog

# Trim the plot vectors to the number of logged steps (the loop may
# have stopped early on a failed step).
n_logged = min(length(sl.time), length(v_reelout_log) + 1)
t_plot = sl.time[2:n_logged]   # skip the t=0 initial log entry

# ==================== PLOTTING ==================== #

if PLOT
    @info "Plotting results..."
    p = plotx(
        t_plot,
        v_reelout_log[1:n_logged-1],
        winch_force_log[1:n_logged-1],
        elevation_log[1:n_logged-1],
        heading_log[1:n_logged-1],
        aoa_log[1:n_logged-1],
        rel_depower_log[1:n_logged-1],
        (ld_wing_log[1:n_logged-1], ld_eff_log[1:n_logged-1]);
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
