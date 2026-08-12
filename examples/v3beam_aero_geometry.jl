# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Emit the V3 Surface-Resolved Aero Geometry

Slices `V3_25.obj` into aero sections and runs NeuralFoil on each, writing the
airfoil contours, the polars and the per-node `Cp`/`cf` surface tables that
`AeroPressure` needs — `v3beam_replay.jl` loads the `geometry.yaml` this writes.
The stock `data/aero_geometry.yaml` comes from the same slice but carries only
lift/drag/moment polars, so `AeroPressure` has no surface to build its
station-point map on and refuses to load.

The mesh is taken from VortexStepMethod's own data directory rather than
`data/V3_25.obj`, because that copy is already rotated into slicer convention and
raised by 7.3 m, and `aero_geometry.yaml` was sliced from it — see
[`V3_ADAPTER_FRAME_OFFSET`](@ref), which is how the structural export follows it.
`obj_to_yaml` rotates but does not translate, so slicing the raised copy at
`rotation=I` is what puts the sections where the beam wing already is.

The output is a few hundred MB of `Cp`/`cf` tables and is not in git, unlike the
structural YAML and the relaxed state. `obj_to_yaml` reuses an existing
`geometry.yaml`, so rerunning is free; `FORCE` regenerates, which is the slow
NeuralFoil pass.

`DELTA_RANGE` is `nothing` because the V3 has no trailing-edge flap — it steers
by bridle-induced wing twist — so the section tables are functions of angle of
attack alone. Giving it a range multiplies the dataset by the number of
deflections.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using VortexStepMethod
using VortexStepMethod.ObjAdapter: obj_to_yaml
using VortexStepMethod.AirfoilAero: ShrinkWrap, NeuralFoilSolver

# =============================================================================
# Configuration
# =============================================================================

OUT_DIR = joinpath(v3_data_path(), "polars_neuralfoil_pressure")

N_SECTIONS = 37            # matches the section count of aero_geometry.yaml
ALPHA_RANGE = -10:2:30     # deg; every step is another Cp/cf table per section
DELTA_RANGE = nothing      # the V3 has no trailing-edge flap
WINGTIP_DISTANCE = 0.05
CREASE_FRAC = 0.75
TABLE_FORMAT = :arrow      # :csv is readable, :arrow loads ~10x faster
FORCE = false              # true reruns the NeuralFoil pass

# Reynolds number of the sliced sections at the flight condition v3beam.jl flies.
RHO = 1.225
MU = 1.81e-5
V_APP = 15.4
CHORD_REF = 2.32           # median chord of aero_geometry.yaml

# =============================================================================
# Emission
# =============================================================================

obj_path = joinpath(pkgdir(VortexStepMethod), "data", "TUDELFT_V3_KITE",
    "V3_25.obj")
isfile(obj_path) || error("No V3 mesh at $obj_path")

reynolds = RHO * V_APP * CHORD_REF / MU
@info "Slicing the V3 mesh" obj_path N_SECTIONS reynolds ALPHA_RANGE

yaml_path = obj_to_yaml(obj_path, OUT_DIR;
    n_sections = N_SECTIONS,
    Re = reynolds,
    alpha_range = ALPHA_RANGE,
    delta_range = DELTA_RANGE,
    aero_solver = NeuralFoilSolver(model_size = "large"),
    wrap_method = ShrinkWrap(),
    wingtip_distance = WINGTIP_DISTANCE,
    crease_frac = CREASE_FRAC,
    table_format = TABLE_FORMAT,
    force = FORCE)

@info "Wrote the surface-resolved aero geometry" yaml_path
@info "Point v3beam_replay.jl's AERO_YAML at it."
nothing
