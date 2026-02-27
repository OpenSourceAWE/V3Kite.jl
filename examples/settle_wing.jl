# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Wing Settling / Stabilization

Runs a damped simulation to find equilibrium wing geometry and writes
the settled positions to YAML files. Plots the settling log afterward.

Usage:
    julia --project=examples examples/settle_wing.jl
"""

using V3Kite
using GLMakie
GLMakie.activate!()

# =============================================================================
# Configuration
# =============================================================================

config = V3SettleConfig(
    num_steps = 8000,
    dt = 0.01,
    v_wind = 10.72,
    elevation = 70.0,
    tether_length = 240.0,
    te_frac = 0.95,
    tip_reduction = 0.4,
)

# =============================================================================
# Run settling
# =============================================================================

@info "Running wing settling..."
sam, syslog = settle_wing(config)

# =============================================================================
# Plot
# =============================================================================

fig = plot(sam.sys_struct, syslog)
scene = replay(syslog, sam.sys_struct; show_panes=false)

display(fig)
display(scene)

nothing
