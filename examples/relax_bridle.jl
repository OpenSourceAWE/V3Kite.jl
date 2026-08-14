# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Relax a Bridle and Save the State

Writes the relaxed state of a project's structural geometry at `DEPOWER` to
`data/`, where the runs pick it up through the `init_state` of their kite
settings — `init_mode: relaxed_state` flies it directly, and settling starts
from it. The file is in git, so this only has to run when the geometry or the
depower changes.

Relaxation exists because the measured bridle lengths and the measured node
coordinates come from different upstream files and disagree — on the beam
geometry several lines start above 100 % strain, which puts the initial
accelerations near 5e7 m/s² and leaves the implicit solver unable to complete a
single step. `relax_bridle!` integrates with every segment stiffness scaled down
and hands it back as the structure settles.

Geometry-agnostic on purpose: pick the particle project from the menu and it
saves a relaxed start for that model too, which is a shorter and
better-conditioned settle than starting from the placed geometry.

The state is world-frame, so it is saved at the elevation and tether length of
the project's settings. Settling repositions it onto its own flight state,
`init_mode: relaxed_state` does not, so keep the project matching the run it
feeds.
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

PROJECT = select_project(
    ["Timoshenko-beam wing" => "system_v3kite_beam.yaml",
     "particle lattice" => "system_v3kite_psm.yaml"];
    prompt = "Which wing model should be relaxed?")

DEPOWER = 0.20        # fraction [0, 1]
STEERING = 0.0        # fraction [-1, 1]

# =============================================================================
# Relaxation
# =============================================================================

set_data_path(v3_data_path())
kite = load_kite(PROJECT)
struc_yaml = basename(struc_geometry_path(PROJECT))

sam, sys = create_v3_model(PROJECT; kite)
apply_kite_material!(sys, kite)

set_depower!(sys, DEPOWER, STEERING, kite.geom)
set_steering!(sys, STEERING, kite.geom)

init!(sam; remake=false, ignore_l0=false, remake_vsm=true)
sys.winches[1].brake = true

scale, steps, residual = relax_bridle!(sam, sys)
scale < 1.0 && error("Bridle relaxation did not reach full stiffness " *
                     "(scale=$scale, residual=$residual)")

state_path = joinpath(v3_data_path(),
    relaxed_state_name(struc_yaml, DEPOWER) * ".arrow")
save_state_log(sam, state_path)

@info "Saved the relaxed state" state_path steps residual
nothing
