# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Translation [m] from the SurfplanAdapter export frame into the CAD frame of the
other V3 data files. `V3_25.obj` was rotated into slicer convention and raised by
7.3 m when it was copied into the VortexStepMethod data directory, and
`cfd_aero_geometry.yaml` was sliced from that copy, so the structural export has to
follow it for the beam and the aero sections to line up.
"""
const V3_ADAPTER_FRAME_OFFSET = [0.0, 0.0, 7.3]

"""
Yaw [deg] about the span axis that the upstream SurfplanAdapter bakes into every
export (`compute_midspan_chord_alignment_rotation_about_y`): the angle
`atan2(chord_z, chord_x)` of the mid-span rib chord, applied to the ribs and to the
bridle so that "zero pitch" means "mid-span chord". For the V3 that chord droops
0.994530° in the raw Surfplan frame.

The measured bridle file is raw, so it has to be rotated by this to join an export.
`cfd_aero_geometry.yaml` is chord-aligned as well, which is why the rotation is added to
the bridle rather than removed from the export.
"""
const V3_ADAPTER_CHORD_ALIGN_DEG = -0.994530

"""
Lift [m] baked into the `z` of the measured bridle file, which places the KCU
(`bridle_point_node`) at its origin. Removed before
[`V3_ADAPTER_CHORD_ALIGN_DEG`](@ref) is applied, since the rotation is about the raw
Surfplan origin.
"""
const V3_BRIDLE_FILE_LIFT = 7.25

"""
    V3BeamTopology

Configuration for translating a SurfplanAdapter export of the TU Delft V3 into a
SymbolicAWEModels beam model. It captures what the adapter YAML does not carry:
the tube pressure and fabric properties, the bridle-to-KCU termination, and the
chordwise beam/receiver discretisation.

The bridle does not come from the export. A Surfplan export carries the *design*
bridle, truncated 0.2 m below its two main knots, with no KCU and no pulleys; the V3
was flown on a different one. `bridle_file` therefore points at the measured 2025
bridle (`data/bridle_geometry_full_fem.yaml`), which carries the KCU, the pulleys,
the M-line and the three tapes as real lines. The `Power Tape` and the two
`Steering Tape`s are emitted as `power_tape`, `steering_left` and `steering_right`,
the segments [`set_depower!`](@ref) and [`set_steering!`](@ref) drive.

`bridle_rotation_deg` and `bridle_lift` place that file in the export's frame, see
[`V3_ADAPTER_CHORD_ALIGN_DEG`](@ref) and [`V3_BRIDLE_FILE_LIFT`](@ref). The result is
checked against the export's own wing nodes, so a mismatched pair of files fails at
read time instead of silently building a skewed kite.

`pressure_bar` is the inflation pressure of the leading edge and struts, and sets
every emitted rigidity through the Breukels correlations. The measured V3 bridle
file vendored from awegroup/TUDELFT_V3_KITE records `pressure: 0.3 [bar]`, which is
where the default comes from.

The emitted `EA`, `GA`, `EI0` and `GJ` are the Breukels linear rigidities
(`tube_linear_rigidities`) at that pressure, so all four share one provenance.

`membrane_stiffness` is the tube fabric `E·t` [N/m] feeding the *optional*
Comer-Levy curvature-softening bending law, which only
[`apply_comer_bending!`](@ref) installs. `:from_breukels` (the default) derives it
per tube radius from the Breukels linear bending stiffness
(`EI0_breukels/(π·r³)`), so that law's linear regime matches the emitted one; a
`Float64` fixes it to a measured value. No fabric coupon data exists for the V3,
which is why there is no `:from_fabric` option here.

`areal_density` [kg/m²] is the tube fabric weight used by the area-based mass model
to back out tube radii from element masses; `:from_mass_file` (the default) reads
`inflatable_tube_density_kg_per_m2` from the export's `wing_mass_distribution.yaml`,
a `Float64` fixes it.

Every section's chord is a single Timoshenko element between a leading- and a
trailing-edge node. `chord_control_fractions` are the along-chord positions (0 =
leading edge, 1 = trailing edge) of the massless pressure receivers emitted per
section for the `AeroPressure` coupling to distribute surface pressure onto; each
is a `BODY_STATIC` point riding that element. They must be ascending and within
`[0, 1]`.

The receivers also carry the spanwise canopy net: each interior fraction (0 and 1
are already tied by the leading-edge beam and the `te` segments) gets a membrane
segment to the matching receiver on the neighbouring station, so a strut cannot bow
sideways between its chord ends. Every added fraction therefore costs
`stations - 1` extra segments in the compiled system, not just an aero receiver.

`cell_diagonals` cross-braces every cell of that net instead of bracing each bay
with the export's single full-chord `dia` pair. A quadrilateral cell with only
edge members carries no shear, so without it the fabric resists racking at bay
scale alone. Costs two segments per cell.

`le_tip_joints` is the number of Timoshenko elements the first and last
leading-edge sections are split into; the tube bends sharply there, so the default
`2` adds one intermediate node in each while interior sections keep a single
element.

`bridle` holds the line and membrane material, see [`V3BridleConfig`](@ref).

`bridle_segments` is the number of spring-damper segments each bridle line is split
into. The lines are emitted as winch-less SymbolicAWEModels tethers, which generate
that many segments and the `DYNAMIC` nodes between them and then hold their length
fixed; `init_stretch_frac` carries the measured rest length, which the generator
would otherwise take from the placed geometry. A pulley's two legs are tethers as
well, with the sheave placed on the generated segment touching it on either side,
where the rope actually pays in and out. Only the three KCU tapes stay plain
segments, because [`set_depower!`](@ref) and [`set_steering!`](@ref) drive their `l0`
directly.

`frame_offset` moves the export into the CAD frame of the other V3 data files, see
[`V3_ADAPTER_FRAME_OFFSET`](@ref).

`tether_length` has to match the length the model is then run at. Loading a geometry
at a different length makes SymbolicAWEModels restretch the tether, and its
restretch carries the tether's downstream *points* but not a beam's bodies, so the
KCU moves and the wing stays behind — which surfaces much later as a non-converged
VSM solve, not as a placement error. Particle geometries are unaffected because
their wing points are `DYNAMIC`.
"""
Base.@kwdef struct V3BeamTopology
    bridle::V3BridleConfig = V3BridleConfig()
    bridle_file::String = joinpath(v3_data_path(), "bridle_geometry_full_fem.yaml")
    bridle_rotation_deg::Float64 = V3_ADAPTER_CHORD_ALIGN_DEG
    bridle_lift::Float64 = V3_BRIDLE_FILE_LIFT
    leading_edge_ids_odd::Bool = true
    frame_offset::Vector{Float64} = copy(V3_ADAPTER_FRAME_OFFSET)
    chord_control_fractions::Vector{Float64} = collect(0.0:0.1:1.0)
    cell_diagonals::Bool = true
    le_tip_joints::Int = 2
    bridle_segments::Int = 1
    pressure_bar::Float64 = 0.3
    membrane_stiffness::Union{Symbol, Float64} = :from_breukels
    areal_density::Union{Symbol, Float64} = :from_mass_file
    youngs_modulus::Float64 = 5.5e10
    line_density::Float64 = 970.0
    min_tube_radius::Float64 = 0.03
    max_tube_radius::Float64 = 0.15
    damping_ratio::Float64 = 1.0
    kcu_mass::Float64 = 23.25
    kcu_area::Float64 = 0.48
    kcu_drag_coeff::Float64 = 0.83
    tether_length::Float64 = 250.0
    tether_diameter_mm::Float64 = 4.0
    tether_segments::Int = 6
    elevation_deg::Float64 = 70.0
    target_wing_mass::Union{Nothing, Float64} = nothing
end
