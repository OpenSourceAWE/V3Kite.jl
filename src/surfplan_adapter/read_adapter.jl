# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
    AdapterGeometry

Parsed SurfplanAdapter structural export: wing particle positions and the wing
connectivity with per-element length/mass. When present, it also carries the real
spanwise tube diameters from `struc_geometry_all_in_surfplan.yaml` and the mass model
from `wing_mass_distribution.yaml`.

The export's own bridle is not read. It is the design bridle, truncated below its main
knots and without the KCU; [`BridleGeometry`](@ref) supplies the flown one instead.

Tube diameters are stored as spanwise samples sorted by `abs_span`, because the two
adapter files use different particle numbering, so diameters can only be matched to
beam elements by span position, not by id. Leading-edge samples are
`(abs_span, diameter)`; strut samples keep both tapered ends as
`(abs_span, diameter_le, diameter_te)` so the radius can vary along the chord.

`leading_edge_polyline` is the fine leading-edge centre curve as `(span, pos)` pairs
sorted by signed span, from the same tube table; it lets the beam place subdivided
leading-edge nodes on the real curved tube rather than on the straight chord between
two stations.

`tube_areal_density` [kg/m²] is the areal weight of the leading-edge/strut fabric
from the export's mass model, which feeds the area-based radius fallback.
"""
struct AdapterGeometry
    pos::Dict{Int, Vector{Float64}}
    wing_ids::Vector{Int}
    wing_connections::Vector{Tuple{String, Int, Int}}
    wing_element::Dict{String, @NamedTuple{l0::Float64, m::Float64}}
    leading_edge_diameters::Vector{Tuple{Float64, Float64}}
    leading_edge_polyline::Vector{Tuple{Float64, Vector{Float64}}}
    strut_diameters::Vector{Tuple{Float64, Float64, Float64}}
    target_wing_mass::Union{Nothing, Float64}
    tube_areal_density::Float64
end

"""
    BridleGeometry

Measured bridle line system, read from `data/bridle_geometry_full_fem.yaml` and placed
in the export's frame. `pos` holds every node the bridle references, including the KCU
(`kcu_id`) and the few wing nodes a pulley hangs from, so the emitter never has to tell
the two id spaces apart. Nodes the file carries but no line uses are dropped.

`connections` keeps each line's node list as the file gives it: two nodes for a plain
line, three for a rope through a pulley (`[end, pulley, end]`), whose halves each get
`line[name].l0 / 2`.

`attachments` are the nodes that hang off the wing — those that occur in exactly one
connection, which is what a branch end means here. Everything else is free.
"""
struct BridleGeometry
    pos::Dict{Int, Vector{Float64}}
    kcu_id::Int
    attachments::Vector{Int}
    connections::Vector{Tuple{String, Vector{Int}}}
    line::Dict{String, @NamedTuple{l0::Float64, d::Float64}}
end

"""Rotate `p` about the span axis by `angle_deg`, the yaw a SurfplanAdapter export
carries (see [`V3_ADAPTER_CHORD_ALIGN_DEG`](@ref))."""
function rotate_about_span(p, angle_deg)
    s, c = sincosd(angle_deg)
    return [c * p[1] + s * p[3], p[2], -s * p[1] + c * p[3]]
end

"""
    read_bridle_geometry(path, wing_pos; rotation_deg, lift, frame_offset, tol=1e-3)
        -> BridleGeometry

Read the measured bridle file at `path` into the frame of an export whose wing
particles are `wing_pos`.

The file is in raw Surfplan axes with `z` lifted by `lift`, so each node goes through
`rotate_about_span(p - [0, 0, lift], rotation_deg) + frame_offset`. Wing nodes the two
files have in common must then coincide to within `tol`; a rotation about the span axis
leaves `y` untouched, so equal `y` is what pairs them, and the pairs agree to machine
precision when the files belong together. That check is what catches a mismatched pair.

Duplicate connection rows are dropped, and both `bridle_particles` and `wing_particles`
are kept because a pulley can hang directly off a wing node.
"""
function read_bridle_geometry(path, wing_pos;
        rotation_deg = V3_ADAPTER_CHORD_ALIGN_DEG,
        lift = V3_BRIDLE_FILE_LIFT,
        frame_offset = V3_ADAPTER_FRAME_OFFSET,
        tol = 1.0e-3)
    data = YAML.load_file(path)
    place(p) = rotate_about_span(Float64.(p) .- [0.0, 0.0, lift], rotation_deg) .+
        frame_offset

    pos = Dict{Int, Vector{Float64}}()
    wing_nodes = Int[]
    for row in data["wing_particles"]["data"]
        id = Int(row[1]); pos[id] = place(row[2:4]); push!(wing_nodes, id)
    end
    for row in data["bridle_particles"]["data"]
        pos[Int(row[1])] = place(row[2:4])
    end
    kcu_id = 0
    haskey(pos, kcu_id) && error("bridle file node $kcu_id is both a particle and " *
        "the KCU (bridle_point_node)")
    pos[kcu_id] = place(data["bridle_point_node"])

    errors = Float64[]
    for id in wing_nodes
        same_span = [q for q in values(wing_pos) if abs(q[2] - pos[id][2]) <= 1.0e-6]
        isempty(same_span) && continue
        push!(errors, minimum(norm(pos[id] .- q) for q in same_span))
    end
    length(errors) >= 8 || error("bridle file $path shares only " *
        "$(length(errors)) wing nodes with the export; they do not describe the " *
        "same kite")
    maximum(errors) <= tol || error("bridle file $path does not match the export: " *
        "shared wing nodes miss by up to $(round(maximum(errors); digits = 4)) m. " *
        "Check bridle_rotation_deg and bridle_lift.")

    connections = Tuple{String, Vector{Int}}[]
    seen = Set{Tuple{String, Vector{Int}}}()
    for row in data["bridle_connections"]["data"]
        entry = (String(row[1]), Int.(row[2:end]))
        entry in seen && continue
        push!(seen, entry); push!(connections, entry)
    end

    line = Dict{String, @NamedTuple{l0::Float64, d::Float64}}()
    for row in data["bridle_lines"]["data"]
        line[String(row[1])] = (l0 = Float64(row[2]), d = Float64(row[3]))
    end
    for (name, _) in connections
        haskey(line, name) ||
            error("bridle connection $name has no entry in bridle_lines")
    end

    degree = Dict{Int, Int}()
    for (_, nodes) in connections, id in nodes
        degree[id] = get(degree, id, 0) + 1
    end
    attachments = sort([id for (id, d) in degree if d == 1 && id != kcu_id])
    filter!(entry -> haskey(degree, entry.first), pos)

    return BridleGeometry(pos, kcu_id, attachments, connections, line)
end

"""Interpolate a spanwise diameter sample set at `span` (clamped to the ends);
`nothing` when no samples were loaded."""
function sample_at(samples, span)
    isempty(samples) && return nothing
    span <= samples[1][1] && return samples[1][2]
    span >= samples[end][1] && return samples[end][2]
    for k in 2:length(samples)
        span <= samples[k][1] || continue
        (x0, d0), (x1, d1) = samples[k - 1], samples[k]
        return d0 + (span - x0) / (x1 - x0) * (d1 - d0)
    end
    return samples[end][2]
end

"""Interpolate a `(span, pos)` polyline (sorted by signed span) at signed spanwise
`span`, clamped to the ends; `nothing` when the polyline is empty."""
function sample_position(polyline, span)
    isempty(polyline) && return nothing
    span <= polyline[1][1] && return polyline[1][2]
    span >= polyline[end][1] && return polyline[end][2]
    for k in 2:length(polyline)
        span <= polyline[k][1] || continue
        (x0, p0), (x1, p1) = polyline[k - 1], polyline[k]
        return p0 .+ (span - x0) / (x1 - x0) .* (p1 .- p0)
    end
    return polyline[end][2]
end

"""Read the leading-edge spanwise diameter samples from the tube table of a
`struc_geometry_all_in_surfplan.yaml`; empty when the file/table is absent.

The table holds the whole perimeter tube as one chain: out along the leading edge,
around a wingtip, then back along the trailing edge. Only the leading-edge runs are
kept — rows whose two nodes both carry the leading-edge parity
(`leading_edge_ids_odd`) — since the trailing-edge runs retrace the same spans with
their own diameters, and the wingtip rows join the two chains.

Returns `(diameters, polyline)`: `diameters` are `(abs_span, diameter)` samples,
`polyline` the unique leading-edge node positions as `(signed_span, pos)` sorted by
signed span."""
function leading_edge_samples(all_in, pos_all, leading_edge_ids_odd)
    empty = (diameters = Tuple{Float64, Float64}[],
        polyline = Tuple{Float64, Vector{Float64}}[])
    haskey(all_in, "leading_edge_tubes") || return empty
    leading_edge_id(id) = leading_edge_ids_odd ? isodd(id) : iseven(id)
    samples = Tuple{Float64, Float64}[]
    nodes = Dict{Int, Vector{Float64}}()
    for row in all_in["leading_edge_tubes"]["data"]
        ci, cj, diameter = Int(row[2]), Int(row[3]), Float64(row[4])
        (leading_edge_id(ci) && leading_edge_id(cj)) || continue
        push!(samples, (abs((pos_all[ci][2] + pos_all[cj][2]) / 2), diameter))
        nodes[ci] = pos_all[ci]; nodes[cj] = pos_all[cj]
    end
    isempty(samples) && error("no leading-edge runs in leading_edge_tubes; " *
        "leading_edge_ids_odd=$leading_edge_ids_odd may be inverted")
    polyline = sort!([(pos[2], pos) for pos in values(nodes)]; by = first)
    return (diameters = sort!(samples; by = first), polyline = polyline)
end

"""Read the strut spanwise diameter samples, keeping both tapered ends
`(abs_span, diameter_le, diameter_te)`, from a
`struc_geometry_all_in_surfplan.yaml`; empty when the file/table is absent.

Sampled by absolute span rather than by particle id because the V3 export mirrors
the strut rows onto shifted node ids, so only `|y|` locates a strut reliably."""
function strut_samples(all_in, pos_all)
    haskey(all_in, "strut_tubes") || return Tuple{Float64, Float64, Float64}[]
    samples = Tuple{Float64, Float64, Float64}[]
    for row in all_in["strut_tubes"]["data"]
        ci, diam_le, diam_te = Int(row[2]), Float64(row[4]), Float64(row[5])
        push!(samples, (abs(pos_all[ci][2]), diam_le, diam_te))
    end
    return sort!(samples; by = first)
end

"""Interpolate a strut diameter at spanwise position `span` and along-chord
fraction `frac` (0 = leading edge, 1 = trailing edge). The two tapered ends are each
interpolated over span (clamped at the ends), then blended by `frac`; `nothing` when
no samples were loaded."""
function sample_strut_at(samples, span, frac)
    isempty(samples) && return nothing
    blend(le, te) = le + frac * (te - le)
    span <= samples[1][1] && return blend(samples[1][2], samples[1][3])
    span >= samples[end][1] && return blend(samples[end][2], samples[end][3])
    for k in 2:length(samples)
        span <= samples[k][1] || continue
        (x0, le0, te0), (x1, le1, te1) = samples[k - 1], samples[k]
        t = (span - x0) / (x1 - x0)
        return blend(le0 + t * (le1 - le0), te0 + t * (te1 - te0))
    end
    return blend(samples[end][2], samples[end][3])
end

"""
    read_adapter_geometry(adapter_dir; native_file, tube_file, mass_file,
        leading_edge_ids_odd, frame_offset) -> AdapterGeometry

Parse a SurfplanAdapter export directory into an [`AdapterGeometry`](@ref).
`native_file` is the particle-schema `struc_geometry.yaml`. `tube_file`
(`struc_geometry_all_in_surfplan.yaml`) and `mass_file`
(`wing_mass_distribution.yaml`) are optional; when the tube file is missing, tube
radii fall back to the area-based mass model, and when the mass file is missing no
mass target is applied and the areal density falls back to 0.53 kg/m².

`leading_edge_ids_odd` is the leading-edge node parity of `tube_file`, used to pick
the leading-edge runs out of the perimeter tube chain (see `leading_edge_samples`).

`frame_offset` [m] is added to every particle position, moving the whole export into
the CAD frame the rest of the V3 data uses. It defaults to `V3_ADAPTER_FRAME_OFFSET`,
which is the translation the V3 `.obj` picked up on its way into the VSM data
directory, so the emitted beam and `aero_geometry.yaml` end up in the same frame.
"""
function read_adapter_geometry(adapter_dir;
        native_file = "struc_geometry.yaml",
        tube_file = "struc_geometry_all_in_surfplan.yaml",
        mass_file = "wing_mass_distribution.yaml",
        leading_edge_ids_odd = true,
        frame_offset = V3_ADAPTER_FRAME_OFFSET)
    adapter = YAML.load_file(joinpath(adapter_dir, native_file))
    shifted(row) = Float64.(row[2:4]) .+ frame_offset
    pos = Dict{Int, Vector{Float64}}()
    wing_ids = Int[]
    for row in adapter["wing_particles"]["data"]
        id = Int(row[1]); pos[id] = shifted(row); push!(wing_ids, id)
    end

    wing_element = Dict{String, @NamedTuple{l0::Float64, m::Float64}}()
    for row in adapter["wing_elements"]["data"]
        wing_element[String(row[1])] = (l0 = Float64(row[2]), m = Float64(row[5]))
    end
    wing_connections = [(String(r[1]), Int(r[2]), Int(r[3]))
                        for r in adapter["wing_connections"]["data"]]

    leading_edge_diameters = Tuple{Float64, Float64}[]
    leading_edge_polyline = Tuple{Float64, Vector{Float64}}[]
    strut_diameters = Tuple{Float64, Float64, Float64}[]
    tube_path = joinpath(adapter_dir, tube_file)
    if isfile(tube_path)
        all_in = YAML.load_file(tube_path)
        pos_all = Dict{Int, Vector{Float64}}()
        for row in all_in["wing_particles"]["data"]
            pos_all[Int(row[1])] = shifted(row)
        end
        leading_edge = leading_edge_samples(all_in, pos_all, leading_edge_ids_odd)
        leading_edge_diameters = leading_edge.diameters
        leading_edge_polyline = leading_edge.polyline
        strut_diameters = strut_samples(all_in, pos_all)
    end

    target_wing_mass = nothing
    areal_density = 0.53
    mass_path = joinpath(adapter_dir, mass_file)
    if isfile(mass_path)
        mass_data = YAML.load_file(mass_path)
        model = get(mass_data, "mass_model", Dict())
        target = get(model, "target_total_wing_mass_kg", nothing)
        target_wing_mass = target === nothing ? nothing : Float64(target)
        density = get(model, "inflatable_tube_density_kg_per_m2", nothing)
        density === nothing || (areal_density = Float64(density))
    end

    return AdapterGeometry(pos, sort(wing_ids), wing_connections, wing_element,
        leading_edge_diameters, leading_edge_polyline, strut_diameters,
        target_wing_mass, areal_density)
end
