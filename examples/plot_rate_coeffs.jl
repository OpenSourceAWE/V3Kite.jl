# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plot the identified turn-rate-law coefficients `c1`, `c2` and steering `delay`
against `rel_depower`, one line per `body_damping`, straight from
`data/turn_rate_coeffs.yaml` -- no simulation, just the table
`examples/build_turn_rate_table.jl` builds and [`turn_rate_coeffs`](@ref) looks
up (see PlanC1C2.md). Uses `MakieControlPlots.plotx` for the three linked,
stacked panels.

Reads the YAML directly, the same way `build_turn_rate_table.jl` does, rather
than through V3Kite's internal table cache, so it always shows exactly what is
on disk regardless of whether `reload_turn_rate_table!()` has been called this
session.

`plotx` shares one x-array across every panel and every line in it, so the
`rel_depower` grid is the union of every non-legacy entry's depower, and a
`body_damping` missing a point on that grid (e.g. `[10,10,40]` has no 0.40 row
-- it was left as the STEP 3 hold-out for `[0,0,40]` only) gets `NaN` there,
which `lines!` renders as a gap rather than a misleading straight line across
the missing point.

**Legacy rows are not shown.** A legacy row (identified at conditions other
than the table's current ones, e.g. a different tether length -- see
PlanC1C2.md "Conditions") is never used by `turn_rate_coeffs` as an
interpolation neighbour, and at present every `body_damping` with more than one
depower identified is fully non-legacy -- the only legacy row left,
`[20,20,40]`/0.25, is the sole point for that damping, and `plotx` only draws
lines: an isolated point with nothing to connect it to would not render at all.
A `@info` line names whatever is skipped instead of silently dropping it.

Run with `include("examples/plot_rate_coeffs.jl")`.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using MakieControlPlots
using LaTeXStrings
using YAML

path = joinpath(v3_data_path(), "turn_rate_coeffs.yaml")
dict = YAML.load_file(path)
conditions = dict["conditions"]

_is_legacy(e) = any(haskey(e, String(k)) && e[String(k)] != v for (k, v) in conditions)

entries = [(body_damping = Float64.(e["body_damping"]), depower = Float64(e["depower"]),
            c1 = Float64(e["c1"]), c2 = Float64(e["c2"]), delay = Float64(e["delay"]),
            legacy = _is_legacy(e)) for e in dict["entries"]]

legacy = filter(e -> e.legacy, entries)
if !isempty(legacy)
    skipped = join(("$(Int.(round.(e.body_damping)))/$(e.depower)" for e in legacy), ", ")
    @info "plot_rate_coeffs.jl: skipping legacy row(s) $skipped -- an isolated " *
          "point has nothing to connect it to under plotx's line-only rendering."
end

current = filter(e -> !e.legacy, entries)
dampings = sort(unique(e.body_damping for e in current); by = string)
depowers = sort(unique(e.depower for e in current))

_series(field) = [Float64[
        (idx = findfirst(e -> e.body_damping == bd && e.depower == dp, current);
         isnothing(idx) ? NaN : getfield(current[idx], field))
        for dp in depowers]
    for bd in dampings]

labels = ["[" * join(Int.(round.(bd)), ", ") * "]" for bd in dampings]

plotx(depowers, _series(:c1), _series(:c2), _series(:delay);
    xlabel = L"\mathrm{rel\_depower}~[-]",
    ylabels = [L"c_1~[\mathrm{1/m}]", L"c_2~[-]", L"\mathrm{delay}~[\mathrm{s}]"],
    labels = [labels, nothing, nothing],
    fig = "V3 turn-rate coefficients vs depower",
    disp = true)
sleep(0.1)  # Allow Makie to render the plot before continuing

nothing
