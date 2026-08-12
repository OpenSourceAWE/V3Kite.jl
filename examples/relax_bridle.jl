# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Relax a Bridle and Save the State

Writes the relaxed state of `STRUC_YAML` at `DEPOWER` to `data/`, where every
other example picks it up: `v3beam.jl` restores it directly, and `settle_wing`
starts from it through `V3SettleConfig.init_state_path`. The file is in git, so
this only has to run when the geometry or the depower changes.

Relaxation exists because the measured bridle lengths and the measured node
coordinates come from different upstream files and disagree — on the beam
geometry several lines start above 100 % strain, which puts the initial
accelerations near 5e7 m/s² and leaves the implicit solver unable to complete a
single step. `relax_bridle!` integrates with every segment stiffness scaled down
and hands it back as the structure settles.

Geometry-agnostic on purpose: point it at `struc_geometry.yaml` with the
particle backend below and it saves a relaxed start for the particle model too,
which is a shorter and better-conditioned settle than starting from the placed
geometry.

The state is world-frame, so it is saved at the `ELEVATION` and `TETHER_LENGTH`
set here. `settle_wing` repositions it onto its own flight state, `v3beam.jl`
does not, so keep these matching the run they feed.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using SymbolicAWEModels

# =============================================================================
# Configuration
# =============================================================================

STRUC_YAML = "struc_geometry_beam.yaml"
AERO_YAML = "aero_geometry.yaml"
VSM_SETTINGS = "vsm_settings.yaml"

# ContinuousAero + KernelBackend for the beam wing, AeroDirect + MonolithBackend
# for the particle one.
AERO_MODE = ContinuousAero()
BACKEND = KernelBackend()

BODY_DAMPING = [0.0, 0.0, 20.0]
WORLD_DAMPING = [0.0, 0.0, 0.0]

V_WIND = 15.4
TETHER_LENGTH = 250.0
ELEVATION = 70.0      # degrees
DEPOWER = 0.20        # fraction [0, 1]
STEERING = 0.0        # fraction [-1, 1]

# =============================================================================
# Relaxation
# =============================================================================

sam, sys = create_v3_model(V3SimConfig(
    struc_yaml_path = STRUC_YAML,
    aero_yaml_path = AERO_YAML,
    vsm_settings_path = VSM_SETTINGS,
    aero_mode = AERO_MODE,
    backend = BACKEND,
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    elevation = ELEVATION,
    damping_pattern = BODY_DAMPING,
    world_damping_pattern = WORLD_DAMPING,
))

geom = V3GeomAdjustConfig()
set_depower!(sys, DEPOWER, STEERING, geom)
set_steering!(sys, STEERING, geom)

init!(sam; remake=false, ignore_l0=false, remake_vsm=true)
sys.winches[1].brake = true

scale, steps, residual = relax_bridle!(sam, sys)
scale < 1.0 && error("Bridle relaxation did not reach full stiffness " *
                     "(scale=$scale, residual=$residual)")

state_path = joinpath(v3_data_path(),
    relaxed_state_name(STRUC_YAML, DEPOWER) * ".arrow")
save_state_log(sam, state_path)

@info "Saved the relaxed state" state_path steps residual
nothing
