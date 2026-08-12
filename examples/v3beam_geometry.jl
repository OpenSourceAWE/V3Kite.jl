# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Emit the V3 Beam-Wing Geometry

Reads the SurfplanAdapter export in `SURFPLAN_DIR`, joins it with the measured
2025 bridle, and writes `data/struc_geometry_beam.yaml` — the file `v3beam.jl`
and every other example load. That file is in git, so this only has to run when
one of the knobs below changes; nothing else regenerates it.

`BRIDLE_SEGMENTS` splits every bridle line into that many spring-damper segments,
a pulley's two legs included; only the three KCU tapes stay single, their length
being driven directly. It is the one knob that changes how many segments exist,
so it is also the one that forces the kernels to be recompiled on the first run
at a new value.

The other three decide what a line does as it goes slack. They are written into
the file here, and `apply_bridle_material!` sets them on a loaded structure, so
sweeping them needs neither this example nor a recompile.

Rerun `relax_bridle.jl` afterwards: the relaxed state it saves is keyed on this
file and stops matching the moment the segment count changes.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite

# =============================================================================
# Configuration
# =============================================================================

STRUC_YAML = "struc_geometry_beam.yaml"

SURFPLAN_DIR = joinpath(homedir(), "Code", "Kite", "SurfplanAdapter",
    "processed_data", "TUDELFT_V3_KITE")

BRIDLE_SEGMENTS = 1                  # spring-damper segments per bridle line
BRIDLE_DAMPING_PER_STIFFNESS = 0.001 # bridle unit_damping / unit_stiffness [s]
COMPRESSION_FRAC = 0.01              # stiffness left under compression
COMPRESSION_DAMPING_FRAC = 1.0       # damping left under compression

TETHER_LENGTH = 250.0
ELEVATION = 70.0      # degrees

# =============================================================================
# Emission
# =============================================================================

isdir(SURFPLAN_DIR) || error("No SurfplanAdapter export at $SURFPLAN_DIR")

beam_topology = V3BeamTopology(
    bridle_segments = BRIDLE_SEGMENTS,
    bridle_rel_damping = BRIDLE_DAMPING_PER_STIFFNESS,
    compression_frac = COMPRESSION_FRAC,
    compression_damping_frac = COMPRESSION_DAMPING_FRAC,
    tether_length = TETHER_LENGTH,
    elevation_deg = ELEVATION,
)

out_yaml = joinpath(v3_data_path(), STRUC_YAML)
counts = surfplan_to_struc(SURFPLAN_DIR, out_yaml;
    topo = beam_topology, wing_only = true)

@info "Wrote the beam geometry" out_yaml counts BRIDLE_SEGMENTS
@info "Rerun relax_bridle.jl for $STRUC_YAML before flying it."
nothing
