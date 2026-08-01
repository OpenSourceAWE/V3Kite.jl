# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Fill in `data/turn_rate_coeffs.yaml` with depower values V3Kite has not yet
identified, so `turn_rate_coeffs` can interpolate instead of throwing (see
PlanC1C2.md STEP 2). One `examples/steering_test_v3.jl`-style sweep per
`(body_damping, depower)` cell — settle, hold a constant tether length, relay-
oscillate the heading with a stepped steering amplitude, then fit
`identify_turn_rate_law` on the log — factored into [`_run_turn_rate_sweep`](@ref)
and driven over a grid by [`build_turn_rate_table`](@ref) instead of running
once for a single hard-coded configuration.

`TETHER_LENGTH` is fixed at 200 m for every cell, matching `conditions:` in
the YAML file (see "Conditions: the whole table is built at 200 m tether" in
PlanC1C2.md) -- **do not** change it here without also updating that block, or
every row this script writes becomes a legacy row on the next load.

Writes incrementally: after every cell the whole output file is re-read,
that cell's row is inserted or replaces a non-passing one, and the file is
rewritten -- so a crash or a diverged run costs at most one cell, not the
whole grid. Re-running with `remake=false` (the default) skips any cell whose
row already has `outcome ∈ (:sweep_done, :time_limit)` *at the table's current
conditions* -- a legacy row never counts as done, and `_write_turn_rate_entry!`
separately refuses to let a non-passing re-run demote a row that already
passed (see its docstring; this is the fix for a real incident, not a
hypothetical). Once it writes to `data/turn_rate_coeffs.yaml`, the file's
formatting changes from the hand-authored layout to `YAML.write_file`'s
(unordered keys, quoted strings) -- that is expected and harmless.

`include("examples/build_turn_rate_table.jl")` only loads the definitions --
it does **not** run anything, because this script gets called repeatedly with
different arguments (the full grid, then a single-cell retry at a different
`max_steering_cap`, etc.) rather than once with fixed parameters like a normal
`examples/*.jl` script. Call `build_turn_rate_table()` yourself once it is
loaded:

    build_turn_rate_table()                          # the whole STEP 2 grid
    build_turn_rate_table(depowers = [0.25],          # one cell, capped lower
        body_dampings = [[0.0, 0.0, 40.0]], max_steering_cap = 0.175)

expect an hour or more of wall time for the full grid (10 cells, each several
thousand `step!` calls, plus a settling-cache miss on every new
`(body_damping, depower)` pair) and some cells to *expectedly* fail: the plant
does not survive every depower at every amplitude (PlanFig8.md Finding 10).
After it finishes, call `reload_turn_rate_table!()` (or restart Julia) and
re-run `test/test_fig8_controller.jl`.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
import KiteUtils   # for KiteUtils.syslog
using YAML
using Printf

# ==================== FIXED CONDITIONS (PlanC1C2.md) ==================== #
# Every cell uses the same system, wind and tether length; only body_damping
# and depower vary. Keep these in sync with `conditions:` in
# data/turn_rate_coeffs.yaml.

const PROJECT         = "system_reelout.yaml"
const V_WIND           = 9.51
const TETHER_LENGTH    = 200.0
const DT               = 0.05 / 3
const SIM_TIME         = 260.0    # headroom over steering_test_v3.jl's 200 s:
                                   # a MAX_STEERING_CAP sweep needs more levels

# Relay controller (see examples/steering_test_v3.jl for the rationale).
const T_START          = 10.0
const HEADING_OFFSET   = 10.0
const START_STEERING   = 0.05
const STEERING_STEP    = 0.025
const CYCLES_PER_LEVEL = 2
# Upper bound the amplitude sweep searches up to; every depower tried so far
# diverges well before this (PlanFig8.md Finding 10), so it is a safety cap,
# not a target.
const MAX_STEERING_CAP = 0.50

const MIN_ELEVATION    = 50.0
const MIN_STEERING_FIT = START_STEERING / 2

const OUT_FILE = "turn_rate_coeffs.yaml"

"""
    _run_turn_rate_sweep(body_damping, depower; max_steering_cap=MAX_STEERING_CAP) -> NamedTuple

One steering-amplitude sweep at `TETHER_LENGTH` / `V_WIND` / `PROJECT` for the
given `(body_damping, depower)` -- the same relay-controller maneuver as
`examples/steering_test_v3.jl`, generalized over the grid `build_turn_rate_table`
drives.

`max_steering_cap` overrides the default `MAX_STEERING_CAP` for this one sweep
-- lower it to avoid a known divergence at the default cap, e.g. matching the
legacy 150 m identification's own 0.175 ceiling for the cleanest apples-to-apples
tether-length comparison (see PlanC1C2.md "Conditions").

Returns `(; outcome, u_s_max, min_elevation, fit)`. `outcome` is `:sweep_done`
(reached `max_steering_cap`), `:time_limit`, `:low_elevation`, or `:error`
(the solver diverged -- identification still runs on whatever was logged before
that). `fit` is the `identify_turn_rate_law` result, or `nothing` if even that
failed (e.g. too few samples before `T_START`).
"""
function _run_turn_rate_sweep(body_damping, depower; max_steering_cap::Real = MAX_STEERING_CAP)
    @info "build_turn_rate_table: body_damping = $body_damping, depower = $depower, " *
          "max_steering_cap = $max_steering_cap"
    s = init(V_WIND, TETHER_LENGTH; body_damping, depower_setpoint = depower,
        sim_time = SIM_TIME, dt = DT, system_yaml = PROJECT)
    l0 = s.sys_state.l_tether[1]

    steering = START_STEERING
    rel_steering = 0.0
    heading = 0.0
    cycles = 0
    min_elevation = Inf
    outcome = :time_limit

    try
        for _ in 1:s.steps
            t = s.sys_state.time + s.dt
            if T_START <= t < T_START + s.dt
                rel_steering = -steering
            end
            last_heading = heading
            if t > T_START + s.dt
                heading = wrap_to_pi(s.sys_state.heading)
                if rad2deg(heading) < -HEADING_OFFSET
                    rel_steering = steering
                elseif rad2deg(heading) > HEADING_OFFSET
                    rel_steering = -steering
                    if rad2deg(last_heading) <= HEADING_OFFSET
                        cycles += 1
                        if cycles >= CYCLES_PER_LEVEL
                            if steering >= max_steering_cap - 1e-9
                                outcome = :sweep_done
                                break
                            end
                            cycles = 0
                            steering = min(steering + STEERING_STEP, max_steering_cap)
                            @info @sprintf("  t = %6.2f s: steering amplitude -> %.3f", t, steering)
                        end
                    end
                end
            end

            step!(s; rel_depower = depower, rel_steering, set_length = l0)

            el = rad2deg(s.sys_state.elevation)
            min_elevation = min(min_elevation, el)
            if el < MIN_ELEVATION
                @warn @sprintf("  elevation %.2f° below floor %.1f° at t = %.2f s, stopping",
                    el, MIN_ELEVATION, t)
                outcome = :low_elevation
                break
            end
        end
    catch e
        outcome = :error
        @warn "build_turn_rate_table: run diverged" body_damping depower exception = (e, catch_backtrace())
    end

    sl = KiteUtils.syslog(s.logger)
    fit = try
        identify_turn_rate_law(sl; dt = DT, t_start = T_START, min_steering = MIN_STEERING_FIT)
    catch e
        @warn "build_turn_rate_table: identification failed, no coefficients recorded" body_damping depower exception = (e, catch_backtrace())
        nothing
    end

    return (; outcome, u_s_max = steering, min_elevation, fit)
end

"""
    _entry_key(e) -> (Vector{Float64}, Float64)

`(body_damping, depower)` of a YAML entry dict, for matching against existing
rows.
"""
_entry_key(e) = (Float64.(e["body_damping"]), Float64(e["depower"]))

"""
    _entry_is_legacy(e, conditions) -> Bool

`true` if entry `e` overrides any of the table's `conditions` (e.g. a per-entry
`l_tether` left over from before the table settled on one tether length for
everything -- see PlanC1C2.md). A legacy row is never treated as "already
passing" for skip/overwrite purposes: the whole point of re-running that cell
is to replace it with one at the *current* conditions, not to leave the old one
in place because it happens to also say `outcome: sweep_done`.
"""
function _entry_is_legacy(e, conditions)
    any(haskey(e, String(k)) && e[String(k)] != v for (k, v) in conditions)
end

"""
    _write_turn_rate_entry!(path, entry; remake=false) -> Bool

Insert `entry` into the `entries:` list of the YAML file at `path`. Without
`remake=true`, an existing row is kept rather than replaced whenever replacing
it would make things worse or gain nothing:

- the existing row **passed** (`outcome ∈ (:sweep_done, :time_limit)`) but
  `entry` did **not** -- a re-run is never allowed to demote a working value
  (legacy or not) to a broken one; this is what a fixed, previously-broken
  version of this function failed to do, and it cost a good 150 m row on the
  first real run of this script (see PlanC1C2.md);
- the existing row passed **and** is already at the table's current
  `conditions` (non-legacy) -- nothing to gain from replacing it.

Otherwise the existing row is replaced: a legacy row is promoted once the
re-run at current conditions passes, and a non-passing row (legacy or not) is
always replaced by the latest attempt, passing or not, since keeping an older
failure over a newer one serves no purpose. Returns whether the file was
written.
"""
function _write_turn_rate_entry!(path, entry::Dict; remake::Bool = false)
    dict = YAML.load_file(path)
    entries = dict["entries"]
    conditions = dict["conditions"]
    key = _entry_key(entry)
    idx = findfirst(e -> _entry_key(e) == key, entries)

    if isnothing(idx)
        push!(entries, entry)
    elseif remake
        entries[idx] = entry
    else
        old = entries[idx]
        old_passing = Symbol(get(old, "outcome", "")) in (:sweep_done, :time_limit)
        new_passing = Symbol(get(entry, "outcome", "")) in (:sweep_done, :time_limit)
        if old_passing && !new_passing
            @warn "build_turn_rate_table: keeping existing PASSING row for $key -- the " *
                  "re-run did not pass (outcome = $(get(entry, "outcome", missing))). Not " *
                  "overwriting a working value with a failed one; pass remake=true to force it in."
            return false
        elseif old_passing && !_entry_is_legacy(old, conditions)
            @info "build_turn_rate_table: keeping existing passing row for $key (remake=false)"
            return false
        else
            entries[idx] = entry
        end
    end
    YAML.write_file(path, dict)
    return true
end

"""
    build_turn_rate_table(; depowers, body_dampings, out="turn_rate_coeffs.yaml",
                          remake=false, max_steering_cap=MAX_STEERING_CAP)

Run [`_run_turn_rate_sweep`](@ref) for every `(body_damping, depower)` pair not
already a passing row in `data/out`, writing each result as it completes. See
this file's module docstring for wall-time expectations and the resume/rewrite
behaviour.

`depowers`/`body_dampings` default to the STEP 2 grid of PlanC1C2.md: depower
0.25/0.35/0.45/0.55 (0.40 is deliberately left out -- it is already on file and
is the STEP 3 hold-out validation point) at `body_damping` `[0,0,40]` and
`[10,10,40]` (`[20,20,40]` is deliberately not swept -- see PlanC1C2.md).

`max_steering_cap` is the same amplitude ceiling for every cell in this call --
pass a narrower `depowers`/`body_dampings` selection to retry just one cell at a
different cap (e.g. one that previously diverged at the default), rather than
recomputing the whole grid at the lower cap.
"""
function build_turn_rate_table(;
        depowers = [0.25, 0.35, 0.45, 0.55],
        body_dampings = [[0.0, 0.0, 40.0], [10.0, 10.0, 40.0]],
        out::String = OUT_FILE,
        remake::Bool = false,
        max_steering_cap::Real = MAX_STEERING_CAP)
    set_data_path(v3_data_path())
    path = joinpath(v3_data_path(), out)
    isfile(path) || error("build_turn_rate_table: $path not found; " *
                           "seed it first (see PlanC1C2.md STEP 1)")

    results = NamedTuple[]
    for bd in body_dampings, dp in depowers
        dict = YAML.load_file(path)
        idx = findfirst(e -> _entry_key(e) == (Float64.(bd), dp), dict["entries"])
        if !remake && !isnothing(idx) &&
           !_entry_is_legacy(dict["entries"][idx], dict["conditions"]) &&
           Symbol(get(dict["entries"][idx], "outcome", "")) in (:sweep_done, :time_limit)
            @info "build_turn_rate_table: skipping body_damping=$bd, depower=$dp (already passing at current conditions)"
            continue
        end

        r = _run_turn_rate_sweep(bd, dp; max_steering_cap)
        entry = Dict{String, Any}(
            "body_damping" => Float64.(bd), "depower" => dp,
            "outcome" => String(r.outcome), "u_s_max" => r.u_s_max,
            "min_elevation" => r.min_elevation,
        )
        if !isnothing(r.fit)
            entry["c1"] = r.fit.c1
            entry["c2"] = r.fit.c2
            entry["delay"] = r.fit.delay_sec
            entry["c1_rel_std"] = abs(r.fit.se1 / r.fit.c1)
            entry["g_rel_std"] = r.fit.G_rel_std
        end
        _write_turn_rate_entry!(path, entry; remake)
        push!(results, (; body_damping = bd, depower = dp, r...))

        @printf("  %-16s depower=%.2f  outcome=%-12s  u_s_max=%.3f  min_el=%.1f°%s\n",
            string(bd), dp, r.outcome, r.u_s_max, r.min_elevation,
            isnothing(r.fit) ? "  (no fit)" : @sprintf("  c1=%.4f", r.fit.c1))
    end

    # Tether-length sensitivity check (PlanC1C2.md "Conditions"): once
    # [0,0,40]/0.25 has a PASSING row at the table's current (200 m)
    # conditions, compare its c1 against the original 150 m identification
    # (0.3159, examples/steering_test_v3.jl, 2026-07-26) -- the number the
    # whole "l_tether is a fixed condition, not a table dimension" decision
    # rests on. A legacy or non-passing row for this cell (e.g. a diverged
    # re-run) means this check has nothing trustworthy to compare yet, and is
    # silently skipped rather than reporting a bad number.
    dict = YAML.load_file(path)
    idx_025 = findfirst(e -> _entry_key(e) == ([0.0, 0.0, 40.0], 0.25), dict["entries"])
    if !isnothing(idx_025) &&
       !_entry_is_legacy(dict["entries"][idx_025], dict["conditions"]) &&
       Symbol(get(dict["entries"][idx_025], "outcome", "")) in (:sweep_done, :time_limit)
        c1_150 = 0.3159
        c1_200 = Float64(dict["entries"][idx_025]["c1"])
        @printf("\nTether-length sensitivity at body_damping=[0,0,40], depower=0.25: c1(200 m)/c1(150 m) = %.4f (%.1f%%)\n",
                c1_200 / c1_150, 100 * (c1_200 / c1_150 - 1))
        abs(c1_200 / c1_150 - 1) > 0.10 &&
            @warn "Tether-length sensitivity exceeds 10% -- re-read PlanC1C2.md " *
                  "\"Conditions\" before trusting the rest of the grid."
    end

    return results
end

@info "build_turn_rate_table.jl: definitions loaded -- call build_turn_rate_table() " *
      "yourself (see this file's module docstring for the full-grid vs. single-cell forms)."
