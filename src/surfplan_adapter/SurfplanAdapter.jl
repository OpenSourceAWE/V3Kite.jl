# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
    V3Kite.SurfplanAdapter

Turn a SurfplanAdapter export of the TU Delft V3 into a SymbolicAWEModels beam
`struc_geometry.yaml`, so the V3 can fly on a Timoshenko-beam wing instead of the
particle lattice.

The Surfplan `.txt` → adapter-YAML step stays a documented prerequisite (the
upstream Python `SurfplanAdapter`, run as
`python -m scripts.process_surfplan_files --kite_name=TUDELFT_V3_KITE`). This
submodule consumes the adapter's YAML outputs: it reads the particle-schema
`struc_geometry.yaml` (with real tube diameters from
`struc_geometry_all_in_surfplan.yaml` and the mass model from
`wing_mass_distribution.yaml`) and emits the beam `struc_geometry.yaml` — the
leading edge and struts as `Body` chains joined by `TimoshenkoJoint`s, with every
bridle branch end riding the nearest node body as a `BODY_STATIC` point.

The bridle itself comes from `data/bridle_geometry_full_fem.yaml`, the measured 2025
line system, not from the export: a Surfplan export carries the design bridle, cut off
below its main knots and without the KCU. The measured file has the KCU, the pulleys,
the M-line and the three tapes, which are emitted as `power_tape`, `steering_left` and
`steering_right` for [`set_depower!`](@ref) and [`set_steering!`](@ref) to drive.

The emitted wing is a `PARTICLE_DYNAMICS` wing: its shape follows `BODY_STATIC`
points that ride the beam bodies, so no `RIGID_DYNAMICS` path is involved and the
file is a drop-in replacement for the particle `struc_geometry.yaml`.

Entry points: [`read_adapter_geometry`](@ref), [`surfplan_to_struc`](@ref),
[`beam_joint_radii`](@ref) and [`apply_comer_bending!`](@ref).
"""
module SurfplanAdapter

using LinearAlgebra
import YAML
using SymbolicAWEModels: TimoshenkoJoint, TUBE_SHEAR_COEFF, tube_torsion_law,
    membrane_linear_rigidities, breukels_membrane_stiffness, comer_levy_bending_law,
    frame_quaternion_xy
using ..V3Kite: v3_data_path, V3BridleConfig

include("topology.jl")
include("read_adapter.jl")
include("beam_emit.jl")

"""Read `adapter_dir` with the export-dependent settings a `V3BeamTopology` carries."""
adapter_geometry(adapter_dir, topo::V3BeamTopology) =
    read_adapter_geometry(adapter_dir;
        leading_edge_ids_odd = topo.leading_edge_ids_odd,
        frame_offset = topo.frame_offset)

"""Read `topo.bridle_file` into the frame of the export `geom` came from."""
bridle_geometry(geom::AdapterGeometry, topo::V3BeamTopology) =
    read_bridle_geometry(topo.bridle_file, geom.pos;
        rotation_deg = topo.bridle_rotation_deg, lift = topo.bridle_lift,
        frame_offset = topo.frame_offset)

"""
    surfplan_to_struc(adapter_dir, out_yaml; topo=V3BeamTopology(), wing_only=true)
        -> NamedTuple

Read the SurfplanAdapter export in `adapter_dir`, join it with the measured bridle at
`topo.bridle_file`, and write the beam `struc_geometry.yaml` to `out_yaml`. When
`wing_only=true` a sibling `<out>_wing.yaml` (the wing subset at CAD, no transform) is
written too. Returns the full model's `(bodies, joints, points, segments)` counts.

The emitted file carries constant `EIy`/`EIz` per joint because YAML cannot hold a
callable; call [`apply_comer_bending!`](@ref) on the loaded structure to swap in the
curvature-softening Comer-Levy law.
"""
function surfplan_to_struc(adapter_dir, out_yaml; topo = V3BeamTopology(),
        wing_only = true)
    geom = adapter_geometry(adapter_dir, topo)
    bridle = bridle_geometry(geom, topo)
    tables = beam_tables(geom, topo)
    counts = write_model(out_yaml, tables, geom, bridle, topo; full = true)
    if wing_only
        wing_path = replace(out_yaml, r"\.yaml$" => "_wing.yaml")
        write_model(wing_path, tables, geom, bridle, topo; full = false)
    end
    return counts
end

"""
    apply_comer_bending!(sys, joint_radius, topo) -> sys
    apply_comer_bending!(sys, adapter_dir, topo) -> sys

Replace each beam `TimoshenkoJoint`'s constant bending rigidity with the
curvature-softening Comer-Levy law (`comer_levy_bending_law`) in place, so the
loaded (linear-YAML) beam gains the post-collapse bending branch. `joint_radius`
maps each joint name to its tube radius (from [`beam_joint_radii`](@ref)).
Axial/shear are refreshed from the same `E·t`; torsion and the resolved indices are
carried over from the loaded joint. Done by element replacement (not a
`SystemStructure` rebuild) so no reference re-resolution or tether re-expansion is
triggered.
"""
function apply_comer_bending!(sys, joint_radius::AbstractDict, topo::V3BeamTopology)
    pressure_pa = topo.pressure_bar * 1.0e5
    for (k, joint) in enumerate(sys.timoshenko_joints)
        haskey(joint_radius, joint.name) || continue
        radius = joint_radius[joint.name]
        et = resolve_membrane_stiffness(topo, radius)
        EA, GA, _ = membrane_linear_rigidities(radius, et)
        law = comer_levy_bending_law(radius, pressure_pa, et)
        new_joint = TimoshenkoJoint(joint.name, joint.body_a_ref, joint.body_b_ref;
            anchor_a = joint.anchor_a_b, anchor_b = joint.anchor_b_b,
            EA, GA, GJ = joint.GJ, EIy = law, EIz = law,
            shear_coeff = joint.shear_coeff, damping = joint.damping,
            rest_length = joint.rest_length, radius = joint.radius)
        new_joint.idx = joint.idx
        new_joint.body_a_idx = joint.body_a_idx
        new_joint.body_b_idx = joint.body_b_idx
        sys.timoshenko_joints[k] = new_joint
    end
    return sys
end

apply_comer_bending!(sys, adapter_dir::AbstractString, topo::V3BeamTopology) =
    apply_comer_bending!(sys, beam_joint_radii(adapter_dir; topo), topo)

"""
    apply_bridle_material!(sys, bridle::V3BridleConfig) -> sys
    apply_bridle_material!(sys, topo::V3BeamTopology) -> sys

Set the line material `bridle` carries onto an already-loaded structure, so
`compression_frac`, `compression_damping_frac` and `bridle_rel_damping` can be swept
without re-emitting the YAML. `bridle_segments` still needs a rewrite, it being the
one that changes how many segments exist and so living on the topology instead.

Applies what [`surfplan_to_struc`](@ref) would have emitted: both compression
fractions reach the bridle lines, the KCU tapes and the canopy membranes alike, while
`bridle_rel_damping` sets `unit_damping` on the lines and tapes only, the canopy
keeping the ratio it is emitted with. The winched tether is left untouched, its
material being the dyneema it is flown on rather than a bridle knob.
"""
function apply_bridle_material!(sys, bridle::V3BridleConfig)
    winched = Set(idx for winch in sys.winches for idx in winch.tether_idxs)
    on_tether = Dict{Int, Bool}()
    for tether in sys.tethers, segment_idx in tether.segment_idxs
        on_tether[segment_idx] = tether.idx in winched
    end
    tape_names = Set(Symbol.(TAPE_SEGMENT_NAMES))
    for segment in sys.segments
        get(on_tether, segment.idx, false) && continue
        segment.compression_frac = bridle.compression_frac
        segment.compression_damping_frac = bridle.compression_damping_frac
        haskey(on_tether, segment.idx) || segment.name in tape_names || continue
        segment.unit_stiffness isa Real || continue
        segment.unit_damping = bridle.bridle_rel_damping * segment.unit_stiffness
    end
    return sys
end

apply_bridle_material!(sys, topo::V3BeamTopology) =
    apply_bridle_material!(sys, topo.bridle)

"""
    beam_joint_radii(adapter_dir; topo=V3BeamTopology()) -> Dict{Symbol, Float64}

Per-joint tube radius map for the beam a SurfplanAdapter export in `adapter_dir`
produces under `topo`.
"""
beam_joint_radii(adapter_dir; topo = V3BeamTopology()) =
    beam_tables(adapter_geometry(adapter_dir, topo), topo).joint_radius

export V3BeamTopology, V3_ADAPTER_FRAME_OFFSET
export V3_ADAPTER_CHORD_ALIGN_DEG, V3_BRIDLE_FILE_LIFT
export AdapterGeometry, read_adapter_geometry
export BridleGeometry, read_bridle_geometry
export surfplan_to_struc, apply_comer_bending!, apply_bridle_material!
export beam_joint_radii

end
