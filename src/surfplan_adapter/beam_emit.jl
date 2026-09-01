# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""Lumped half-element isotropic node inertia contribution."""
half_inertia(mass_half, len_half, radius) =
    mass_half * (len_half^2 / 3 + radius^2 / 2)

"""Tube fabric membrane stiffness `E·t` [N/m] at `radius` under `topo`:
`:from_breukels` derives it from the Breukels linear rigidity, otherwise the fixed
`Float64`."""
resolve_membrane_stiffness(topo, radius) =
    topo.membrane_stiffness === :from_breukels ?
    breukels_membrane_stiffness(radius, topo.pressure_bar) :
    Float64(topo.membrane_stiffness)

"""Tube fabric areal density [kg/m²] under `topo`: `:from_mass_file` uses
`file_density` (from the export's mass model), otherwise the fixed `Float64`."""
resolve_areal_density(topo, file_density) =
    topo.areal_density === :from_mass_file ? file_density : Float64(topo.areal_density)

"""Equivalent constant radius of a tapered element, from `radius_at(x)` sampled over
the relative position `x ∈ [0, 1]`. Elements in series add bending compliance and
`EI ∝ r³`, so the equivalent is the harmonic mean of `r³`; sampling the midpoint
instead overstates the stiffness of a strongly tapered element."""
function harmonic_radius(radius_at, n_samples = 16)
    inverse_cube = sum(radius_at((k - 0.5) / n_samples)^-3 for k in 1:n_samples)
    return (inverse_cube / n_samples)^(-1 / 3)
end

le_body_name(i) = Symbol("wing_le_body_$i")
le_sub_body_name(i, j) = Symbol("wing_le_sub_body_$(i)_$j")
te_body_name(i) = Symbol("wing_te_body_$i")

"""Validate an along-chord fraction list: ascending and within `[0, 1]`. Returns the
list as a `Vector{Float64}`."""
function checked_chord_fractions(fractions, label)
    all(f -> 0.0 <= f <= 1.0, fractions) ||
        error("$label must lie in [0, 1]; got $fractions")
    issorted(fractions) || error("$label must be ascending; got $fractions")
    return collect(Float64, fractions)
end

"""
    beam_tables(geom, topo) -> NamedTuple

Build the wing beam from an [`AdapterGeometry`](@ref): sort the LE/TE stations by
descending span, give each chord one Timoshenko element between a leading- and a
trailing-edge node with a tapered tube radius (`harmonic_radius` over the real
spanwise/along-chord diameters when available, otherwise the area-based mass model),
lump node masses/inertia, and assemble the node `Body` and `TimoshenkoJoint` rows.
Returns everything the emitter needs; the bridle is separate
(see [`BridleGeometry`](@ref)).

Descending span is required, not cosmetic: `VortexStepMethod.refine!` sorts sections
that way, but SymbolicAWEModels rebuilds a beam wing's sections with
`sort_sections=false`, so whatever order the twist surfaces are emitted in is the
order the panels are built in. Ascending span gives reversed panels and the VSM
solve does not converge. `y_ref` is ordered against the stations to keep body y
pointing along +y.
"""
function beam_tables(geom, topo)
    pos = geom.pos
    parity_le(id) = topo.leading_edge_ids_odd ? isodd(id) : iseven(id)
    by_descending_span(ids) = sort(ids; by = id -> pos[id][2], rev = true)
    le_ids = by_descending_span([id for id in geom.wing_ids if parity_le(id)])
    te_ids = by_descending_span([id for id in geom.wing_ids if !parity_le(id)])
    n = length(le_ids)
    n == length(te_ids) ||
        error("LE/TE station count mismatch: $n vs $(length(te_ids))")

    le_pos = [pos[id] for id in le_ids]
    te_pos = [pos[id] for id in te_ids]
    le_tangent(i) = normalize(le_pos[min(i + 1, n)] - le_pos[max(i - 1, 1)])
    chord(i) = te_pos[i] - le_pos[i]

    control_fractions = checked_chord_fractions(topo.chord_control_fractions,
        "chord_control_fractions")

    le_tip_joints = topo.le_tip_joints
    le_tip_joints >= 1 || error("le_tip_joints must be >= 1; got $le_tip_joints")
    le_section_joints(i) = (i == 1 || i == n - 1) ? le_tip_joints : 1

    strut_edge_key(i) = minmax(le_ids[i], te_ids[i])
    strut_keys = Set(strut_edge_key(i) for i in 1:n)
    le_edge_keys = Set(minmax(le_ids[i], le_ids[i + 1]) for i in 1:n - 1)

    edge_mass = Dict{Tuple{Int, Int}, Float64}()
    node_mass = Dict{Int, Float64}()
    for (name, ci, cj) in geom.wing_connections
        mass = geom.wing_element[name].m
        key = minmax(ci, cj)
        edge_mass[key] = mass
        (key in strut_keys || key in le_edge_keys) && continue
        node_mass[ci] = get(node_mass, ci, 0.0) + mass / 2
        node_mass[cj] = get(node_mass, cj, 0.0) + mass / 2
    end
    le_node_mass = [get(node_mass, id, 0.0) for id in le_ids]
    te_node_mass = [get(node_mass, id, 0.0) for id in te_ids]

    areal_density = resolve_areal_density(topo, geom.tube_areal_density)
    clamp_radius(r) = clamp(r, topo.min_tube_radius, topo.max_tube_radius)
    mass_radius(mass, len) = clamp_radius(mass / (2π * len * areal_density))
    function le_edge_radius(i, frac0, frac1)
        if isempty(geom.leading_edge_diameters)
            p0 = le_pos[i] .+ frac0 .* (le_pos[i + 1] .- le_pos[i])
            p1 = le_pos[i] .+ frac1 .* (le_pos[i + 1] .- le_pos[i])
            section_mass = edge_mass[minmax(le_ids[i], le_ids[i + 1])]
            return mass_radius(section_mass * (frac1 - frac0), norm(p1 - p0))
        end
        span_at(x) = abs(le_pos[i][2] +
            (frac0 + x * (frac1 - frac0)) * (le_pos[i + 1][2] - le_pos[i][2]))
        return harmonic_radius(x -> clamp_radius(
            sample_at(geom.leading_edge_diameters, span_at(x)) / 2))
    end

    le_span_at(i, frac) = le_pos[i][2] + frac * (le_pos[i + 1][2] - le_pos[i][2])
    le_edge_position(i, frac) = isempty(geom.leading_edge_polyline) ?
        le_pos[i] .+ frac .* (le_pos[i + 1] .- le_pos[i]) :
        sample_position(geom.leading_edge_polyline, le_span_at(i, frac))
    function le_edge_tangent(i, frac)
        step = 0.01 * (le_pos[i + 1][2] - le_pos[i][2])
        (isempty(geom.leading_edge_polyline) || abs(step) < 1e-9) &&
            return normalize(le_pos[i + 1] .- le_pos[i])
        span = le_span_at(i, frac)
        ahead = sample_position(geom.leading_edge_polyline, span + step)
        behind = sample_position(geom.leading_edge_polyline, span - step)
        return normalize(ahead .- behind)
    end

    le_names = Symbol[]; le_poss = Vector{Float64}[]; le_station_of = Int[]
    le_sec = Int[]; le_fr = Float64[]
    le_node_tube_mass = Float64[]; le_node_tube_inertia = Float64[]
    le_edge_specs = NamedTuple[]
    function add_le_node!(name, p, station_idx, section, frac)
        push!(le_names, name); push!(le_poss, p); push!(le_station_of, station_idx)
        push!(le_sec, section); push!(le_fr, frac)
        push!(le_node_tube_mass, 0.0); push!(le_node_tube_inertia, 0.0)
        return length(le_names)
    end
    prev = add_le_node!(le_body_name(1), le_pos[1], 1, 0, 0.0)
    for i in 1:n - 1
        s = le_section_joints(i)
        section_mass = edge_mass[minmax(le_ids[i], le_ids[i + 1])]
        for j in 1:s
            frac0, frac1 = (j - 1) / s, j / s
            cur = j < s ?
                add_le_node!(le_sub_body_name(i, j), le_edge_position(i, frac1),
                    0, i, frac1) :
                add_le_node!(le_body_name(i + 1), le_pos[i + 1], i + 1, 0, 0.0)
            radius = le_edge_radius(i, frac0, frac1)
            len = norm(le_poss[cur] .- le_poss[prev])
            sub_mass = section_mass / s
            le_node_tube_mass[prev] += sub_mass / 2
            le_node_tube_mass[cur] += sub_mass / 2
            contribution = half_inertia(sub_mass / 2, len / 2, radius)
            le_node_tube_inertia[prev] += contribution
            le_node_tube_inertia[cur] += contribution
            push!(le_edge_specs, (a = prev, b = cur, radius = radius, len = len,
                section = i, j = j, s = s))
            prev = cur
        end
    end
    le_inter_idxs = [c for c in 1:length(le_names) if le_station_of[c] == 0]
    le_beam_inertia = zeros(n)
    for c in 1:length(le_names)
        le_station_of[c] == 0 && continue
        le_node_mass[le_station_of[c]] += le_node_tube_mass[c]
        le_beam_inertia[le_station_of[c]] += le_node_tube_inertia[c]
    end

    strut_radius = zeros(n)
    strut_len = zeros(n)
    strut_node_mass = zeros(n)
    strut_node_inertia = zeros(n)
    for i in 1:n
        len = norm(chord(i))
        strut_mass = get(edge_mass, strut_edge_key(i), 0.0)
        strut_len[i] = len
        strut_radius[i] = isempty(geom.strut_diameters) ?
            mass_radius(strut_mass, len) :
            harmonic_radius(x -> clamp_radius(sample_strut_at(
                geom.strut_diameters, abs(le_pos[i][2]), x) / 2))
        strut_node_mass[i] = strut_mass / 2
        strut_node_inertia[i] =
            half_inertia(strut_mass / 2, len / 2, strut_radius[i])
    end

    target = topo.target_wing_mass === nothing ? geom.target_wing_mass :
        topo.target_wing_mass
    current = sum(le_node_mass) + sum(te_node_mass) + 2 * sum(strut_node_mass) +
        sum(le_node_tube_mass[c] for c in le_inter_idxs; init = 0.0)
    if target !== nothing && current > 0
        scale = target / current
        le_node_mass .*= scale; te_node_mass .*= scale; le_beam_inertia .*= scale
        strut_node_mass .*= scale; strut_node_inertia .*= scale
        for c in le_inter_idxs
            le_node_tube_mass[c] *= scale
            le_node_tube_inertia[c] *= scale
        end
    end

    node_name(i, k) = k == 1 ? le_body_name(i) : te_body_name(i)
    node_pos(i, k) = k == 1 ? le_pos[i] : te_pos[i]
    node_mass_of(i, k) = strut_node_mass[i] +
        (k == 1 ? le_node_mass[i] : te_node_mass[i])
    node_inertia_of(i, k) = strut_node_inertia[i] +
        (k == 1 ? le_beam_inertia[i] : 0.0)

    body_rows = Vector{Any}[]
    body_frame = Dict{Symbol, Tuple{Vector{Float64}, Vector{Float64}}}()
    for i in 1:n
        q = frame_quaternion_xy(chord(i), le_tangent(i))
        for k in 1:2
            name = node_name(i, k)
            p = node_pos(i, k)
            push!(body_rows, [String(name), node_mass_of(i, k),
                fill(node_inertia_of(i, k), 3), p, "DYNAMIC", q])
            body_frame[name] = (p, q)
        end
    end
    for c in le_inter_idxs
        i, frac = le_sec[c], le_fr[c]
        q = frame_quaternion_xy((1 - frac) .* chord(i) .+ frac .* chord(i + 1),
            le_edge_tangent(i, frac))
        p = le_poss[c]
        push!(body_rows, [String(le_names[c]), le_node_tube_mass[c],
            fill(le_node_tube_inertia[c], 3), p, "DYNAMIC", q])
        body_frame[le_names[c]] = (p, q)
    end

    joint_rows = Vector{Any}[]
    joint_radius = Dict{Symbol, Float64}()
    function beam_joint!(name, a, b, radius, len, mass_a, mass_b, inertia_a, inertia_b)
        EA, GA, EI0, GJ0 = tube_linear_rigidities(radius, topo.pressure_bar)
        # Rayleigh β from ζ = βω/2, anchored at the transverse mode
        # ω = sqrt(12EI/(L³m)): ζ rises with frequency, so anchoring the softest
        # mode of interest leaves everything stiffer at least as well damped.
        bend_stiffness = 12 * EI0 / len^3
        omega_bend = sqrt(bend_stiffness / min(mass_a, mass_b))
        beta = 2 * topo.damping_ratio / omega_bend
        joint_radius[Symbol(name)] = radius
        push!(joint_rows, [name, String(a), String(b), EA, GA, GJ0, EI0, EI0,
            TUBE_SHEAR_COEFF, beta, radius])
    end
    le_node_full_mass(c) = le_station_of[c] == 0 ? le_node_tube_mass[c] :
        node_mass_of(le_station_of[c], 1)
    le_node_full_inertia(c) = le_station_of[c] == 0 ? le_node_tube_inertia[c] :
        node_inertia_of(le_station_of[c], 1)
    for e in le_edge_specs
        name = e.s == 1 ? "le_beam_$(e.section)" : "le_beam_$(e.section)_$(e.j)"
        beam_joint!(name, le_names[e.a], le_names[e.b], e.radius, e.len,
            le_node_full_mass(e.a), le_node_full_mass(e.b),
            le_node_full_inertia(e.a), le_node_full_inertia(e.b))
    end
    for i in 1:n
        beam_joint!("strut_beam_$i", node_name(i, 1), node_name(i, 2),
            strut_radius[i], strut_len[i],
            node_mass_of(i, 1), node_mass_of(i, 2),
            node_inertia_of(i, 1), node_inertia_of(i, 2))
    end

    wing_pt = Dict{Int, String}()
    wing_body = Dict{Int, String}()
    for i in 1:n
        wing_pt[le_ids[i]] = "wing_le_$i"; wing_pt[te_ids[i]] = "wing_te_$i"
        wing_body[le_ids[i]] = String(le_body_name(i))
        wing_body[te_ids[i]] = String(te_body_name(i))
    end

    mid = cld(n, 2)

    control_specs = NamedTuple[]
    for i in 1:n
        chord_vec = chord(i)
        for (j, frac) in enumerate(control_fractions)
            push!(control_specs, (name = "wing_ctrl_$(i)_$j",
                pos = le_pos[i] .+ frac .* chord_vec, station = i,
                joint = "strut_beam_$i", frac = frac))
        end
    end

    return (; n, le_ids, te_ids, le_pos, te_pos, body_rows, joint_rows, joint_radius,
        wing_pt, wing_body, body_frame, mid, control_specs)
end

"""
    tape_segment_name(bridle, name, nodes) -> String

Rename the measured bridle's three KCU tapes to the segment names V3Kite drives:
`Power Tape` becomes `power_tape`, and each `Steering Tape` becomes `steering_left`
or `steering_right` after the span of its non-KCU end. Any other line keeps its own
name, lowercased with separators folded to `_`.
"""
function tape_segment_name(bridle, name, nodes)
    plain = lowercase(replace(name, r"[^A-Za-z0-9]+" => "_"))
    name == "Power Tape" && return "power_tape"
    name == "Steering Tape" || return plain
    far = only(filter(!=(bridle.kcu_id), nodes))
    return bridle.pos[far][2] < 0 ? "steering_left" : "steering_right"
end

"""
Nearest beam joint (element) to a world position `p`, by distance to the element's
line SEGMENT between its two node bodies (not just the midpoint). This picks the
element `p` actually projects onto, so the bridle rides that element's deformed
centerline (corotational Hermite) with a small transverse offset — the midpoint
metric could pick a nearby wrong element whose projection clamps to an endpoint.
"""
function nearest_beam_joint(tables, p)
    best, best_dist = String(tables.joint_rows[1][1]), Inf
    for row in tables.joint_rows
        pos_a = tables.body_frame[Symbol(row[2])][1]
        pos_b = tables.body_frame[Symbol(row[3])][1]
        chord = pos_b .- pos_a
        len2 = dot(chord, chord)
        frac = len2 < 1e-12 ? 0.0 : clamp(dot(p .- pos_a, chord) / len2, 0.0, 1.0)
        dist = norm(pos_a .+ frac .* chord .- p)
        dist < best_dist && ((best, best_dist) = (String(row[1]), dist))
    end
    return best
end

function fmt_num(x)
    x isa Integer && return string(x)
    v = Float64(x)
    v == 0 && return "0"
    r = abs(v) < 0.1 ? round(v; sigdigits = 4) : round(v; digits = 4)
    isinteger(r) ? string(Int(r)) : string(r)
end
fmt_cell(x) = x === nothing ? "nothing" :
    x isa AbstractString ? String(x) :
    x isa AbstractVector ? string("[", join(fmt_cell.(x), ", "), "]") : fmt_num(x)
fmt_row(row) = string("[", join(fmt_cell.(row), ", "), "]")
fmt_ref(r) = r isa AbstractVector ?
    string("[", join(fmt_ref.(r), ", "), "]") : string(r)

"""Emit a `variables` block from `name => fields` pairs. Each is a multi-variable
whose `fields` fill the columns they are named after wherever `name` is written, so
the columns have to follow the field order given here."""
function emit_variables(io, variables)
    println(io, "variables:")
    for (name, fields) in variables
        println(io, "  ", name, ":")
        for (field, value) in fields
            println(io, "    ", field, ": ", fmt_num(value))
        end
    end
    println(io)
end

"""Emit a column-aligned `name: {headers, data}` block. A row is narrower than
`headers` when a cell names a multi-variable that fills several columns, and rows
may differ in width when they use multi-variables of different widths, so every
column is measured over the rows that reach it."""
function emit_table(io, name, headers, rows)
    ncol = min(length(headers), maximum(length(r) for r in rows))
    reaching(j) = [r for r in rows if j <= length(r)]
    is_vec = [any(r[j] isa AbstractVector for r in reaching(j)) for j in 1:ncol]
    is_num = [!is_vec[j] && all(r[j] isa Real for r in reaching(j))
              for j in 1:ncol]
    comp_width = Dict{Int, Vector{Int}}()
    for j in 1:ncol
        is_vec[j] || continue
        widths = zeros(Int, maximum(length(r[j]) for r in reaching(j)))
        for r in reaching(j), (k, c) in enumerate(r[j])
            widths[k] = max(widths[k], length(fmt_cell(c)))
        end
        comp_width[j] = widths
    end
    cellstr(j, cell) = is_vec[j] ?
        string("[", join([lpad(fmt_cell(c), comp_width[j][k])
            for (k, c) in enumerate(cell)], ", "), "]") : fmt_cell(cell)
    colw = [maximum(length(cellstr(j, r[j])) for r in reaching(j))
            for j in 1:ncol]
    padcell(j, s) = is_num[j] ? lpad(s, colw[j]) : rpad(s, colw[j])
    println(io, name, ":")
    println(io, "  headers: ", fmt_row(headers))
    println(io, "  data:")
    for r in rows
        println(io, "    - [", rstrip(join([padcell(j, cellstr(j, r[j]))
            for j in 1:min(ncol, length(r))], ", ")), "]")
    end
    println(io)
end

const POINT_HEADERS = ["name", "pos_cad", "type", "wing_idx", "transform_idx",
    "extra_mass", "area", "drag_coeff", "body_idx", "joint"]
const SEG_HEADERS = ["name", "point_i", "point_j", "l0", "diameter_mm",
    "unit_stiffness", "unit_damping", "compression_frac",
    "compression_damping_frac"]

"""Tether columns. `n_segments` through `compression_damping_frac` are consecutive so
that one `variables:` entry can fill them all from a single cell; `diameter_mm` and
the two `init_*` columns differ per line and stay written out per row."""
const TETHER_HEADERS = ["name", "start_point", "end_point", "n_segments",
    "youngs_modulus", "damping_per_stiffness", "density", "compression_frac",
    "compression_damping_frac", "diameter_mm", "init_stretched_length",
    "init_stretch_frac"]

"""Tether columns a `variables:` entry fills, in header order."""
const TETHER_MATERIAL_COLUMNS = TETHER_HEADERS[4:9]

"""
    tether_material(; n_segments, youngs_modulus, damping_per_stiffness, density,
                    compression_frac, compression_damping_frac)

A `variables:` entry covering [`TETHER_MATERIAL_COLUMNS`](@ref), ordered so it expands
over them. Every column has to be given, since a variable fills a fixed column span.
"""
function tether_material(; kwargs...)
    fields = Dict(String(key) => value for (key, value) in kwargs)
    issetequal(keys(fields), TETHER_MATERIAL_COLUMNS) ||
        error("tether_material needs exactly $(join(TETHER_MATERIAL_COLUMNS, ", "))")
    return [column => fields[column] for column in TETHER_MATERIAL_COLUMNS]
end

"""The three KCU tapes, whose `l0` V3Kite drives; they stay plain segments."""
const TAPE_SEGMENT_NAMES = ("power_tape", "steering_left", "steering_right")

"""Canopy membrane segment row. `l0` is left `nothing` so the loader takes the rest
length from the loaded geometry; `unit_stiffness` scales with `rest_length` so every
canopy spring ends up at the same 1000 N/m rate, and `compression_frac` (from
`topo.bridle.compression_frac`) leaves it mostly tension-only like fabric."""
function canopy_seg_row(name, point_i, point_j, rest_length, topo)
    unit_stiffness = 1000.0 * rest_length
    return Any[name, point_i, point_j, nothing, 1.0, unit_stiffness,
        0.01 * unit_stiffness, topo.bridle.compression_frac,
        topo.bridle.compression_damping_frac]
end

"""Point row with no mass, drag, body or joint anchor."""
plain_point_row(name, pos, type, transform_idx) =
    Any[name, pos, type, 1, transform_idx, 0.0, 0.0, 0.0, "nothing", "nothing"]

"""
    write_model(path, tables, geom, bridle, topo; full)

Write the SymbolicAWEModels beam `struc_geometry.yaml` to `path`. `full=true` writes
the flying model (bridle, KCU, tapes, pulleys, tether, winch, transform); `full=false`
writes the wing-only subset placed at CAD with no transform.

`topo.cell_diagonals` decides how the canopy is braced: per cell of the net, or
with the export's single full-chord `dia` pair per bay.
"""
function write_model(path, tables, geom, bridle, topo; full)
    tf = full ? 1 : "nothing"
    n, mid = tables.n, tables.mid
    wing_pt, wing_body = tables.wing_pt, tables.wing_body
    le_ids, te_ids = tables.le_ids, tables.te_ids
    bridle_pt(id) = id == bridle.kcu_id ? "kcu" : "bridle_$id"

    point_rows = Vector{Any}[]
    for id in sort([le_ids; te_ids])
        push!(point_rows, [wing_pt[id], geom.pos[id], "BODY_STATIC", 1, tf, 0.0,
            0.0, 0.0, wing_body[id], "nothing"])
    end
    for spec in tables.control_specs
        push!(point_rows, [spec.name, spec.pos, "BODY_STATIC", 1, tf, 0.0, 0.0, 0.0,
            "nothing", spec.joint])
    end

    seg_rows = Vector{Any}[]
    seg_name_seen = Dict{String, Int}()
    unique_seg_name = function (base)
        count = get(seg_name_seen, base, 0) + 1
        seg_name_seen[base] = count
        return count == 1 ? base : "$(base)_$count"
    end
    push_seg! = row -> push!(seg_rows, row)
    for (name, ci, cj) in geom.wing_connections
        (startswith(name, "te") || startswith(name, "dia")) || continue
        # The per-cell crosses brace every cell, which the full-bay X cannot.
        topo.cell_diagonals && startswith(name, "dia") && continue
        push_seg!(canopy_seg_row(unique_seg_name(name), wing_pt[ci], wing_pt[cj],
            norm(geom.pos[ci] - geom.pos[cj]), topo))
    end
    receivers = [[spec for spec in tables.control_specs if spec.station == i]
                 for i in 1:n]
    for i in 1:n - 1, (j, spec) in enumerate(receivers[i])
        (spec.frac == 0.0 || spec.frac == 1.0) && continue
        neighbour = receivers[i + 1][j]
        push_seg!(canopy_seg_row("spanwise_$(i)_$j", spec.name, neighbour.name,
            norm(spec.pos - neighbour.pos), topo))
    end
    """Canopy net nodes down station `i`'s chord, leading to trailing edge."""
    function net_nodes(i)
        nodes = [(wing_pt[le_ids[i]], geom.pos[le_ids[i]])]
        for spec in receivers[i]
            0.0 < spec.frac < 1.0 && push!(nodes, (spec.name, spec.pos))
        end
        push!(nodes, (wing_pt[te_ids[i]], geom.pos[te_ids[i]]))
        return nodes
    end
    if topo.cell_diagonals
        for i in 1:n - 1
            here, there = net_nodes(i), net_nodes(i + 1)
            for k in 1:length(here) - 1
                push_seg!(canopy_seg_row("cross_$(i)_$(k)a", here[k][1],
                    there[k + 1][1], norm(here[k][2] - there[k + 1][2]), topo))
                push_seg!(canopy_seg_row("cross_$(i)_$(k)b", here[k + 1][1],
                    there[k][1], norm(here[k + 1][2] - there[k][2]), topo))
            end
        end
    end

    body_ref = full ? 1 : "nothing"
    body_emit = [[r..., body_ref, body_ref] for r in tables.body_rows]
    pulley_rows = Vector{Any}[]
    tether_rows = Vector{Any}[]
    if full
        kcu_pos = bridle.pos[bridle.kcu_id]
        attached = Set(bridle.attachments)
        for id in sort(collect(keys(bridle.pos)))
            id == bridle.kcu_id && continue
            if id in attached
                push!(point_rows, [bridle_pt(id), bridle.pos[id], "BODY_STATIC", 1, 1,
                    0.0, 0.0, 0.0, "nothing",
                    nearest_beam_joint(tables, bridle.pos[id])])
            else
                push!(point_rows, plain_point_row(bridle_pt(id), bridle.pos[id],
                    "DYNAMIC", 1))
            end
        end
        push!(point_rows, ["kcu", kcu_pos, "DYNAMIC", 1, 1, topo.kcu_mass,
            topo.kcu_area, topo.kcu_drag_coeff, "nothing", "nothing"])
        push!(point_rows, plain_point_row("ground_anchor",
            kcu_pos .- [0.0, 0.0, topo.tether_length], "STATIC", 1))

        tape_seg_row = function (name, point_i, point_j, l0, diameter)
            unit_stiffness = topo.youngs_modulus * π * (diameter / 2)^2
            return Any[name, point_i, point_j, l0, diameter * 1000,
                unit_stiffness, topo.bridle.bridle_rel_damping * unit_stiffness,
                topo.bridle.compression_frac, topo.bridle.compression_damping_frac]
        end
        push!(tether_rows, Any["tether", "kcu", "ground_anchor", "dyneema",
            topo.tether_diameter_mm, topo.tether_length, nothing])
        bridle_tether_row = function (name, ci, cj, l0, diameter)
            stretched = norm(bridle.pos[ci] .- bridle.pos[cj])
            return Any[name, bridle_pt(ci), bridle_pt(cj), "bridle_line",
                diameter * 1000, nothing, l0 / stretched]
        end
        for (name, nodes) in bridle.connections
            line = bridle.line[name]
            base = unique_seg_name(tape_segment_name(bridle, name, nodes))
            if length(nodes) == 2
                if base in TAPE_SEGMENT_NAMES
                    push_seg!(tape_seg_row(base, bridle_pt(nodes[1]),
                        bridle_pt(nodes[2]), line.l0, line.d))
                else
                    push!(tether_rows, bridle_tether_row(base, nodes[1],
                        nodes[2], line.l0, line.d))
                end
                continue
            end
            length(nodes) == 3 ||
                error("bridle line $name has $(length(nodes)) nodes; expected 2 or 3")
            halves = ["$(base)_a", "$(base)_b"]
            for (k, half) in enumerate(halves)
                push!(tether_rows, bridle_tether_row(half, nodes[k],
                    nodes[k + 1], line.l0 / 2, line.d))
            end
            push!(pulley_rows, [base, "$(halves[1])_seg_$(topo.bridle_segments)",
                "$(halves[2])_seg_1", "DYNAMIC", topo.bridle.pulley_efficiency])
        end
    end

    origin = full ? "kcu" : wing_pt[le_ids[mid]]
    z_ref = full ? ["kcu", wing_pt[le_ids[mid]]] :
        [wing_pt[te_ids[mid]], wing_pt[le_ids[mid]]]
    y_ref = [wing_pt[le_ids[min(n, mid + 1)]], wing_pt[le_ids[max(1, mid - 1)]]]

    open(path, "w") do io
        println(io, "# Auto-generated ", full ? "beam model" : "wing-only beam model",
            " from a SurfplanAdapter export")
        println(io, "# by V3Kite.SurfplanAdapter. Edits here are overwritten.\n")
        full && emit_variables(io,
            ["dyneema" => tether_material(
                 n_segments = topo.tether_segments,
                 youngs_modulus = topo.youngs_modulus,
                 damping_per_stiffness = 0.002,
                 density = topo.line_density,
                 compression_frac = 0.1,
                 compression_damping_frac = 1.0),
             "bridle_line" => tether_material(
                 n_segments = topo.bridle_segments,
                 youngs_modulus = topo.youngs_modulus,
                 damping_per_stiffness = topo.bridle.bridle_rel_damping,
                 density = topo.line_density,
                 compression_frac = topo.bridle.compression_frac,
                 compression_damping_frac = topo.bridle.compression_damping_frac)])
        emit_table(io, "bodies",
            ["name", "mass", "inertia_principal", "pos", "type", "Q_b_to_w",
             "transform_idx", "wing"], body_emit)
        emit_table(io, "timoshenko_joints",
            ["name", "body_a", "body_b", "EA", "GA", "GJ", "EIy", "EIz",
             "shear_coeff", "damping", "radius"],
            tables.joint_rows)
        station_points(i) = [wing_pt[le_ids[i]]; wing_pt[te_ids[i]];
            [spec.name for spec in tables.control_specs if spec.station == i]]
        flap_rows = [["flap_$i", 1, "KINEMATIC", station_points(i),
            [String(le_body_name(i)), String(te_body_name(i))], [0.0, 1.0, 0.0]]
            for i in 1:n]
        emit_table(io, "twist_surfaces",
            ["name", "wing", "type", "points", "flap_bodies", "flap_axis"], flap_rows)
        emit_table(io, "points", POINT_HEADERS, point_rows)
        emit_table(io, "segments", SEG_HEADERS, seg_rows)
        isempty(pulley_rows) ||
            emit_table(io, "pulleys",
                ["name", "segment_i", "segment_j", "type", "efficiency"], pulley_rows)
        if full
            emit_table(io, "tethers", TETHER_HEADERS, tether_rows)
            emit_table(io, "winches", ["name", "tether_idxs", "winch_point"],
                [["winch", ["tether"], "ground_anchor"]])
        end
        println(io, "wings:\n  data:\n    - idx: 1\n      type: PARTICLE_DYNAMICS")
        println(io, "      origin_idx: ", fmt_ref(origin))
        println(io, "      z_ref_points: ", fmt_ref(z_ref))
        println(io, "      y_ref_points: ", fmt_ref(y_ref))
        println(io, "      twist_surfaces: ", fmt_ref(["flap_$i" for i in 1:n]))
        if full
            println(io, "\ntransforms:\n  data:\n    - idx: 1")
            println(io, "      elevation: ", fmt_num(topo.elevation_deg))
            println(io, "      azimuth: 0.0\n      heading: 0.0\n      wing_idx: 1")
            println(io, "      base_pos: [0.0, 0.0, 0.0]")
            println(io, "      base_point_idx: ground_anchor")
        end
    end
    return (bodies = length(tables.body_rows), joints = length(tables.joint_rows),
        points = length(point_rows), segments = length(seg_rows))
end
