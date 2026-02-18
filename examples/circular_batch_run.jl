# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Run for Single-Phase Circular Flight

Runs multiple parameter combinations for the open-loop
v3 kite simulation with ramped steering and depower.

Usage:
    julia --project=examples examples/circular_batch_run.jl
"""

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX,
    V3_DEPOWER_IDX, V3_STEERING_GAIN
using GLMakie
using LinearAlgebra
using Statistics
using DiscretePIDs
using Dates
using StaticArrays: SVector

# =============================================================================
# Single-phase simulation function
# =============================================================================

"""
    run_circular(; kwargs...) -> (syslog, sam, heading_setpoint)

Run a single-phase open-loop simulation with ramped
steering and depower.
"""
function run_circular(;
        sim_time=300.0, fps=4,
        initial_damping=100.0,
        damping_pattern=[0.0, 1.0, 1.0],
        decay_time=10.0, min_damping=0.0,
        up=0.4, us=0.1,
        v_wind=15.4, v_wind_base=15.0,
        ramp_start_time_up=0.0,
        ramp_end_time_up=25.0,
        ramp_start_time_us=0.0,
        ramp_end_time_us=25.0,
        tether_length=150.0, elevation=nothing,
        tube_bending_resistance=0.0,
        save_subdir="", run_tag="")

    config = V3SimConfig(
        struc_yaml_path = "CORRECT_struc_geometry.yaml",
        aero_yaml_path = "CORRECT_aero_geometry.yaml",
        vsm_settings_path = "CORRECT_vsm_settings.yaml",
        v_wind = v_wind,
        tether_length = tether_length,
        elevation = elevation,
        damping_pattern = damping_pattern * initial_damping,
        wing_type = REFINE,
        brake = true,
    )
    sam, sys = create_v3_model(config)

    init!(sam; remake=false,
        ignore_l0=false, remake_vsm=true)
    sys.winches[1].brake = true

    n_steps = Int(round(fps * sim_time))
    dt = sim_time / n_steps
    logger, sys_state = create_logger(sam, n_steps)

    nom_left = sys.segments[V3_STEERING_LEFT_IDX].l0
    nom_dep = sys.segments[V3_DEPOWER_IDX].l0
    nom_right = sys.segments[V3_STEERING_RIGHT_IDX].l0

    steer_change = V3_STEERING_GAIN * us
    power_change = ((200 + 5000 * up) / 1000) - nom_dep
    vw_change = v_wind - v_wind_base

    heading_setpoint = Float64[0.0]
    sys.winches[1].set_value = 0.0

    @info "Starting simulation" n_steps dt
    sim_start = time()

    for step in 1:n_steps
        t = step * dt

        # Damping decay
        cd = max(
            initial_damping * (1.0 - t / decay_time),
            min_damping)
        SymbolicAWEModels.set_body_frame_damping(
            sys, damping_pattern * cd)

        # Power ramp
        prf = ramp_factor(t,
            ramp_start_time_up, ramp_end_time_up)
        sys.segments[V3_DEPOWER_IDX].l0 =
            nom_dep + power_change * prf

        # Steering ramp
        srf = ramp_factor(t,
            ramp_start_time_us, ramp_end_time_us)
        sys.segments[V3_STEERING_LEFT_IDX].l0 =
            nom_left + steer_change * srf
        sys.segments[V3_STEERING_RIGHT_IDX].l0 =
            nom_right - steer_change * srf

        push!(heading_setpoint, 0.0)

        # Wind speed
        sys.set.v_wind = v_wind_base + vw_change

        # Tube bending resistance
        if tube_bending_resistance != 0
            R = sys.wings[1].R_b_w
            f_pos = R * [0.0, tube_bending_resistance, 0.0]
            f_neg = R * [0.0, -tube_bending_resistance, 0.0]
            sys.points[2].disturb .= f_pos
            sys.points[3].disturb .= f_pos
            sys.points[20].disturb .= f_neg
            sys.points[21].disturb .= f_neg
        end

        if !sim_step!(sam;
                set_values=[0.0], dt, vsm_interval=1)
            @error "Simulation failed" step
            break
        end
        log_state!(logger, sys_state, sam, t)

        if should_report(step, n_steps)
            elapsed = time() - sim_start
            @info "Step $step/$n_steps" times_realtime=round(t/elapsed, digits=2)
        end
    end

    report_performance(sim_time, time() - sim_start)

    # Save
    lt_tag = Int(round(tether_length))
    save_log(logger, "tmp_run_refine_lt_$(lt_tag)")
    syslog = load_log("tmp_run_refine_lt_$(lt_tag)")

    save_root = joinpath("processed_data", "v3_kite")
    save_dir = isempty(save_subdir) ? save_root :
        joinpath(save_root, save_subdir)
    isdir(save_dir) || mkpath(save_dir)
    ts = Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")
    up_t = Int(round(up * 100))
    us_t = Int(round(us * 100))
    vw_t = Int(round(v_wind))
    ln = "circle__up_$(up_t)_us_$(us_t)" *
        "_vw_$(vw_t)_lt_$(lt_tag)"
    if !isempty(run_tag)
        ln *= "_" * run_tag
    end
    ln *= "_date_" * ts
    save_log(logger, ln; path=save_dir)

    return syslog, sam, heading_setpoint
end

# =============================================================================
# Batch sweep: 2025 kite parameters
# =============================================================================

us_vals = [0.05, 0.075, 0.1, 0.125, 0.15,
    0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325]
up_vals = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
vw_vals = [7.8]
lt_vals = [262]

batch_tag = "circular_2025_batch_" *
    Dates.format(Dates.now(), "yyyy_mm_dd_HH_MM_SS")

sim_time = 250.0
decay_time = 2.0
ramp_start_time_up = 0.1
ramp_end_time_up = 1.5
ramp_start_time_us = 3.0
ramp_end_time_us = 5.0
fps = 60
initial_damping = 200.0
damping_pattern = [0.0, 0.0, 20.0]
min_damping = 1.0
tube_bending_resistance = 0.0
elevation = 17.0

failed_runs = NamedTuple[]

for (run_id, (us, up, vw, lt)) in enumerate(
        Iterators.product(us_vals, up_vals, vw_vals, lt_vals))
    run_tag = "run_" * lpad(string(run_id), 3, '0')
    @info "Starting run" run_id us up vw lt
    try
        run_circular(;
            sim_time, fps,
            up, us, v_wind=vw, tether_length=lt,
            decay_time,
            ramp_start_time_up, ramp_end_time_up,
            ramp_start_time_us, ramp_end_time_us,
            initial_damping, damping_pattern, min_damping,
            elevation, tube_bending_resistance,
            save_subdir=batch_tag, run_tag)
    catch err
        @error "Run failed" run_id us up vw lt err
        push!(failed_runs, (run_id=run_id, us=us, up=up,
            vw=vw, lt=lt, error=err))
    end
    GC.gc()
end

if !isempty(failed_runs)
    fp = joinpath("processed_data", "v3_kite",
        batch_tag, "failed_runs.txt")
    open(fp, "w") do io
        for f in failed_runs
            println(io, "run_id=$(f.run_id) " *
                "us=$(f.us) up=$(f.up) " *
                "vw=$(f.vw) error=$(f.error)")
        end
    end
    @info "Wrote failure list" path=fp
end

n_total = length(collect(Iterators.product(
    us_vals, up_vals, vw_vals, lt_vals)))
@info "Batch completed" total=n_total failed=length(failed_runs)
