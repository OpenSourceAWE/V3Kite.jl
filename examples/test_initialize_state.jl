#!/usr/bin/env julia
# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Minimal smoke test for `initialize_state`.

Usage:
    julia --project=. examples/test_initialize_state.jl
"""

using V3Kite
using LinearAlgebra: norm
using Statistics: mean

const HOLD_TIME_S = 50.0
const HOLD_FPS = 360
const HOLD_WINDOW_S = 5.0
const HOLD_ELEV_TOL_DEG = 1.0
const BATCH_UP_CMD = 0.35
const START_RAMP_TIME_S = 1.0
const STARTUP_DECAY_TIME_S = 2.0
const STARTUP_DAMPING_PATTERN = [100.0, 500.0, 1000.0]
const NOMINAL_DAMPING_PATTERN = [0.0, 0.0, 20.0]

function damping_profile(t)
    if t < START_RAMP_TIME_S
        return STARTUP_DAMPING_PATTERN
    end
    if STARTUP_DECAY_TIME_S <= 0
        return NOMINAL_DAMPING_PATTERN
    end
    mix = clamp((t - START_RAMP_TIME_S) / STARTUP_DECAY_TIME_S, 0.0, 1.0)
    return STARTUP_DAMPING_PATTERN .+
           (NOMINAL_DAMPING_PATTERN .- STARTUP_DAMPING_PATTERN) .* mix
end

function mean_last_window(values, times; window_sec)
    t_end = times[end]
    mask = times .>= (t_end - window_sec)
    data = values[mask]
    return isempty(data) ? values[end] : mean(data)
end

function up_to_depower(up_cmd)
    # Match batch_run_zenith_than_circles_working.jl command mapping:
    # target_l0 = (200 + 5000 * up) / 1000 [m]
    target_l0 = (200.0 + 5000.0 * up_cmd) / 1000.0
    dep = depower_length_to_percentage(target_l0) / 100.0
    return clamp(dep, 0.0, 1.0)
end

const BATCH_EQUIV_DEPOWER = up_to_depower(BATCH_UP_CMD)

cfg = V3Kite.V3InitConfig(
    source_struc_path="struc_geometry.yaml",
    source_aero_path="aero_geometry.yaml",
    vsm_settings_path="vsm_settings.yaml",
    tether_length=269.0,
    elevation=50.0,
    azimuth=0.0,
    heading=0.0,
    upwind_dir=-90.0,
    g_earth=0.0,
    target_wind=8.4,
    target_depower=BATCH_EQUIV_DEPOWER,
    target_steering=0.0,
    start_wind=2.0,
    start_depower=BATCH_EQUIV_DEPOWER,
    start_steering=0.0,
    n_stages=10,
    settle_time=1.0,
    settle_dt=0.02,
    vsm_interval=1,
    max_extra_stages=6,
    extra_settle_time_factor=2.0,
    convergence_tol_elevation_deg=0.1,
    convergence_tol_azimuth_deg=0.1,
    convergence_tol_heading_deg=0.5,
    convergence_tol_kite_speed=0.5,
    enforce_convergence=false,
    world_damping_start=1200.0,
    world_damping_end=20.0,
    body_damping=300.0,
    warmup_time=2.0,
    warmup_fps=120,
    use_quasi_static=false,
    use_settled_geometry=false,
    settle_wind=4.0,
    fallback_to_raw_geometry=true,
    settle_remake=false,
    settle_num_steps=1200,
    settle_step_dt=0.01,
    geom=V3GeomAdjustConfig(
        reduce_tip=true,
        reduce_te=true,
        reduce_depower=false,
        tether_length=269.0),
)

@info "Initializing coupled state..."
@info "Batch-equivalent control mapping" up=BATCH_UP_CMD depower=BATCH_EQUIV_DEPOWER
# Use closure + invokelatest to avoid world-age issues for keyword dispatch
# in long-running REPL sessions.
sam = Base.invokelatest(() ->
    V3Kite.initialize_state(cfg; show_progress=true))

sys = sam.sys_struct
wing = sys.wings[1]
sys_state = SysState(sam)
update_sys_state!(sys_state, sam)
elevation_deg = rad2deg(wing.elevation)
azimuth_deg = rad2deg(wing.azimuth)
heading_deg = rad2deg(wing.heading)
elevation_err_deg = elevation_deg - cfg.elevation
azimuth_err_deg = rad2deg(wrap_to_pi(wing.azimuth - deg2rad(cfg.azimuth)))
heading_err_deg = rad2deg(wrap_to_pi(wing.heading - deg2rad(cfg.heading)))
kite_speed = norm(sys_state.vel_kite)

println("Initialization summary")
println("  wind [m/s]        : ", round(sys.set.v_wind, digits=4))
println("  upwind_dir [deg]  : ", round(sys.set.upwind_dir, digits=4))
println("  elevation [deg]   : ", round(elevation_deg, digits=4))
println("  azimuth [deg]     : ", round(azimuth_deg, digits=4))
println("  heading [deg]     : ", round(heading_deg, digits=4))
println("  elev err [deg]    : ", round(elevation_err_deg, digits=4))
println("  az err [deg]      : ", round(azimuth_err_deg, digits=4))
println("  heading err [deg] : ", round(heading_err_deg, digits=4))
println("  tether_len [m]    : ", round(sys.winches[1].tether_len, digits=4))
println("  depower [-]       : ", round(get_depower(sys), digits=4))
println("  steering [-]      : ", round(get_steering(sys), digits=4))
println("  apparent wind [m/s]: ", round(norm(wing.va_b), digits=4))
println("  kite speed [m/s]  : ", round(kite_speed, digits=5))
println("  AoA [deg]         : ", round(rad2deg(sys_state.AoA), digits=4))

@info "Running post-init zenith hold check..." hold_time=HOLD_TIME_S fps=HOLD_FPS
n_hold = max(1, Int(round(HOLD_TIME_S * HOLD_FPS)))
dt_hold = HOLD_TIME_S / n_hold
hold_t = Float64[]
hold_elev = Float64[]
for step in 1:n_hold
    t = step * dt_hold
    SymbolicAWEModels.set_body_frame_damping(sys, damping_profile(t))
    set_depower!(sys, cfg.target_depower, cfg.geom)
    set_steering!(sys, cfg.target_steering)
    for winch in sys.winches
        winch.brake = true
        winch.tether_len = cfg.tether_length
        winch.tether_vel = 0.0
        winch.set_value = 0.0
    end
    if !sim_step!(sam;
        set_values=[0.0],
        dt=dt_hold,
        vsm_interval=cfg.vsm_interval)
        error("Post-init hold unstable at step $step/$n_hold")
    end
    push!(hold_t, t)
    push!(hold_elev, rad2deg(sys.wings[1].elevation))
end

hold_elev_end = hold_elev[end]
hold_elev_mean_last = mean_last_window(hold_elev, hold_t;
    window_sec=HOLD_WINDOW_S)
hold_elev_err = hold_elev_mean_last - cfg.elevation

println("Post-init hold summary")
println("  hold_time [s]       : ", HOLD_TIME_S)
println("  elevation end [deg] : ", round(hold_elev_end, digits=4))
println("  elevation mean-last [deg]: ", round(hold_elev_mean_last, digits=4))
println("  elev err mean-last [deg] : ", round(hold_elev_err, digits=4))

if abs(hold_elev_err) > HOLD_ELEV_TOL_DEG
    error(
        "Post-init hold did not converge near target elevation. " *
        "mean-last=$(round(hold_elev_mean_last, digits=4)) deg, " *
        "target=$(round(cfg.elevation, digits=4)) deg, " *
        "error=$(round(hold_elev_err, digits=4)) deg, " *
        "tol=$(round(HOLD_ELEV_TOL_DEG, digits=4)) deg.")
end
