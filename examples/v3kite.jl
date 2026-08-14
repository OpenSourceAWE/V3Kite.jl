# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite Simulation Example

Heading PID tracking a sinusoidal setpoint on a settled, depowered wing, with the
winch braked at a constant tether length.

Which kite this flies and at what flight condition is the project file's, not the
script's, and the menu picks between them: `system_v3kite_psm.yaml` is the
particle lattice and `system_v3kite_beam.yaml` the Timoshenko-beam wing.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using MakieControlPlots
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

PROJECT = select_project(
    ["particle lattice" => "system_v3kite_psm.yaml",
     "Timoshenko-beam wing" => "system_v3kite_beam.yaml"];
    prompt = "Which wing model should fly?")

# The maneuver. The gains that track it are the project's heading settings.
MAX_HEADING = 40.0    # setpoint amplitude [deg]
PERIOD = 30.0         # setpoint period [s]

# =============================================================================
# Model
# =============================================================================

@info "V3 Kite Simulation Example" PROJECT
@info "Calibration:" steering_l0=V3_STEERING_L0_BASE depower_l0=V3_DEPOWER_L0_BASE

set_data_path(v3_data_path())
kite = load_kite(PROJECT)
heading = load_heading(PROJECT)
set = Settings(PROJECT)

sam, sys = build_v3_model(PROJECT)

n_steps = Int(round(set.sample_freq * set.sim_time))
dt = set.sim_time / n_steps
logger, sys_state = create_logger(sam, n_steps)

nominal_steering = V3Kite.get_steering(sys, kite.geom)
max_heading_rad = deg2rad(MAX_HEADING)
angular_freq = 2pi / PERIOD

pid = heading_pid(heading, dt)

# =============================================================================
# Simulation loop
# =============================================================================

@info "Starting simulation" n_steps dt
sim_start = time()

for step in 1:n_steps
    t = step * dt

    target_rad = max_heading_rad * sin(angular_freq * t)
    measured = sys.wings[1].heading
    schedule_heading_pid!(pid, heading, t, sys_state.v_app, target_rad, measured)
    steer_ctrl = pid(target_rad, measured, 0.0)
    sys_state.bearing = target_rad

    set_steering!(sys, nominal_steering + steer_ctrl, kite.geom)

    if !sim_step!(sam; dt, vsm_interval = kite.vsm_interval)
        @error "Simulation failed" step
        break
    end
    log_state!(logger, sys_state, sam, t)

    if should_report(step, n_steps)
        elapsed = time() - sim_start
        @info "Step $step/$n_steps" times_realtime=round(t/elapsed, digits=2)
    end
end

report_performance(set.sim_time, time() - sim_start)

log_name = "v3kite_$(splitext(PROJECT)[1])"
save_log(logger, log_name)
syslog = load_log(log_name)

# =============================================================================
# Visualization
# =============================================================================

@info "Creating visualization..."
fig = Makie.plot(sam.sys_struct, syslog;
    plot_tether=true,
    setpoints=Dict(:heading => syslog.syslog.bearing))
display(GLMakie.Screen(), fig)

scene = SymbolicAWEModels.replay(syslog, sam.sys_struct)
display(GLMakie.Screen(), scene)

@info "Example complete!"
nothing
