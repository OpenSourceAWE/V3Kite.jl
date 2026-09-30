# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite Simulation Example

Heading PID tracking a sinusoidal setpoint on a settled, depowered wing, with the
winch braked at a constant tether length.

Which kite this flies and at what flight condition is the project file's, not the
script's, and the menu picks between them: `system_psm.yaml` is the
particle lattice and `system_beam.yaml` the Timoshenko-beam wing.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using MakieControlPlots
using LaTeXStrings
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

PROJECT = select_project(
    ["particle lattice" => "system_psm.yaml",
     "Timoshenko-beam wing" => "system_beam.yaml"];
    prompt = "Which wing model should fly?")

# The maneuver. The gains that track it are the project's heading settings.
MAX_HEADING = 40.0    # setpoint amplitude [deg]
PERIOD = 30.0         # setpoint period [s]

# =============================================================================
# Simulation
# =============================================================================

@info "V3 Kite Simulation Example" PROJECT
@info "Calibration:" steering_l0=V3_STEERING_L0_BASE depower_l0=V3_DEPOWER_L0_BASE

set_data_path(v3_data_path())
sam, sys = build_v3_model(PROJECT)
logger, _, _ = fly_heading_sine(sam, sys, PROJECT; max_heading = MAX_HEADING,
    period = PERIOD)

log_name = "v3kite_$(splitext(PROJECT)[1])"
save_log(logger, log_name)
syslog = load_log(log_name)

# =============================================================================
# Visualization
# =============================================================================

@info "Creating visualization..."
states = syslog.syslog
plot_window = plotx(states.time,
    rad2deg.(states.elevation),
    rad2deg.(states.azimuth),
    [rad2deg.(wrap_to_pi.(states.heading)), rad2deg.(states.bearing),
        rad2deg.(states.course)],
    100.0 .* states.steering,
    rad2deg.(states.AoA),
    first.(states.winch_force);
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 18,
    legendsize = 16,
    ylabels = [L"\mathrm{elevation}~[°]", L"\mathrm{azimuth}~[°]",
               L"\psi, \chi~[°]", L"u_{\mathrm{s}}~[\%]", L"\mathrm{AoA}~[°]",
               L"F_{\mathrm{t}}~[\mathrm{N}]"],
    labels = [nothing, nothing, [L"\psi", L"\psi_{\mathrm{ref}}", L"\chi"],
              nothing, nothing, nothing],
    fig = "V3 Kite heading tracking – $(splitext(PROJECT)[1])")
display(plot_window)

scene = SymbolicAWEModels.replay(syslog, sam.sys_struct)
display(GLMakie.Screen(), scene)

@info "Example complete!"
nothing
