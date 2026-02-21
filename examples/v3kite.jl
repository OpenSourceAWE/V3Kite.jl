# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite Simulation Example

Heading PID control with sinusoidal setpoint, winch PID
control for constant tether length, and 3D visualization.

Usage:
    julia --project=examples examples/v3kite.jl
"""

using V3Kite
using V3Kite: V3_STEERING_LEFT_IDX, V3_STEERING_RIGHT_IDX,
    V3_DEPOWER_IDX
using GLMakie
using LinearAlgebra
using DiscretePIDs

# =============================================================================
# Configuration
# =============================================================================

SIM_TIME = 60.0
FPS = 60
MAX_HEADING = 10.0    # degrees
PERIOD = 30.0         # seconds
V_WIND = 15.4
TETHER_LENGTH = 250.0

# PID gains
HEADING_P = 0.0
HEADING_I = 0.1
HEADING_D = 0.0
WINCH_P = 1000.0
WINCH_I = 100.0
WINCH_D = 50.0

# =============================================================================
# Main simulation
# =============================================================================

function run_example()
    @info "V3 Kite Simulation Example"
    @info "Calibration:" steering_l0=V3_STEERING_L0 depower_l0=V3_DEPOWER_L0

    config = V3SimConfig(
        sim_time = SIM_TIME,
        fps = FPS,
        v_wind = V_WIND,
        tether_length = TETHER_LENGTH,
        up = 40.0,
        us = 0.0,
        wing_type = REFINE,
        brake = false,
        damping_pattern = [0.0, 30.0, 60.0],
    )

    @info "Creating V3 model..."
    sam, sys = create_v3_model(config)

    @info "Initializing model..."
    init!(sam; remake=config.remake_cache,
        ignore_l0=false, remake_vsm=true)

    n_steps = Int(round(FPS * SIM_TIME))
    dt = SIM_TIME / n_steps
    logger, sys_state = create_logger(sam, n_steps)

    # Heading PID (outputs steering tape delta in m)
    nominal_steering = get_steering(sys)
    max_heading_rad = deg2rad(MAX_HEADING)
    angular_freq = 2pi / PERIOD
    max_steering = 0.15

    heading_pid = create_heading_pid(;
        K = HEADING_P > 0 ? HEADING_P : 1.0,
        Ti = HEADING_I > 0 ? 1.0 / HEADING_I : false,
        Td = HEADING_D > 0 ? HEADING_D : false,
        dt, umin=-abs(max_steering),
        umax=abs(max_steering))

    # Winch PID
    nominal_tether_length = sys.winches[1].tether_len
    init_winch_torque!(sys)
    winch_pid = create_winch_pid(;
        K = WINCH_P,
        Ti = WINCH_I > 0 ? WINCH_P / WINCH_I : false,
        Td = WINCH_D > 0 ? WINCH_D / WINCH_P : false,
        dt)

    heading_setpoint = [0.0]

    @info "Starting simulation" n_steps dt
    sim_start = time()

    for step in 1:n_steps
        t = step * dt

        # PID heading control with sine wave setpoint
        target_rad = max_heading_rad *
            sin(angular_freq * t)
        current = sam.sys_struct.wings[1].heading
        steer_ctrl = heading_pid(target_rad, current, 0.0)
        push!(heading_setpoint, target_rad)

        set_steering!(sys, nominal_steering - steer_ctrl)

        # Winch PID
        tl = sys.winches[1].tether_len
        wf = winch_pid(nominal_tether_length, tl, 0.0)
        wt = force_to_torque(wf, sys)
        sys.winches[1].set_value = -wt

        @time if !sim_step!(sam;
                set_values=[-wt], dt, vsm_interval=1)
            @error "Simulation failed" step
            break
        end
        log_state!(logger, sys_state, sam, t)

        if should_report(step, n_steps)
            elapsed = time() - sim_start
            @info "Step $step/$n_steps" times_realtime=round(t/elapsed, digits=2)
        end
    end

    report_performance(SIM_TIME, time() - sim_start)

    save_log(logger, "v3kite_example")
    syslog = load_log("v3kite_example")

    return sam, syslog, heading_setpoint
end

# =============================================================================
# Main execution
# =============================================================================

@info "Running V3 Kite example..."
sam, syslog, heading_setpoint = run_example()

@info "Creating visualization..."
fig = plot(sam.sys_struct, syslog;
    plot_tether=true,
    setpoints=Dict(:heading => heading_setpoint))
display(fig)

@info "Example complete!"
nothing
