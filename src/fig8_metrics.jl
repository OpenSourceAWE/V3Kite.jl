# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Headless quantitative metrics for figure-of-eight runs (see
`examples/simple_fig8.jl` and the success criteria in PlanFig8.md). Ported from
an earlier `fig8_metrics.jl`, adapted to the V3 log layout:

- `set_steering` is dimensionless `rel_steering` here, not meters, so the
  high-frequency content is reported in rel-steering units — establish a V3
  baseline rather than importing the earlier thresholds.
- the V3 has one winch, logged in `winch_force[1]`.
- `var_01` carries the cross-track error [deg] (see `simple_fig8.jl` for the
  full slot mapping).

No plotting dependency, so this runs in headless sweeps.
"""

"""
    fig8_metrics(sl; t_start=0.0, settle_time=10.0, hf_window=0.5)

Compute figure-of-eight quality metrics of syslog `sl` over the settled window
(`t_start + settle_time` to the end). `hf_window` [s] is the moving-average
width subtracted out to isolate high-frequency content. Returns `nothing` if the
window is empty.

The elevation floor is reported both over the settled window and over the
**whole run** (`min_elevation_all`) — the latter is the success criterion,
because a floor breach during the entry transient still counts as a breach.

Steering is reported on BOTH sides of the actuator, and the pair is the point:
`steering_sat_frac`/`max_steering_used` describe the command, while
`tape_rate_frac`/`max_steering_delivered` describe what the KCU tape actually
did. `v_steering` [1/s] is the tape's rate limit and must match the `kcu:`
section of the settings YAML in use (0.2 for every V3 settings file shipped
here); it is a keyword rather than a constant so a log from a differently
configured KCU can still be scored correctly.
"""
function fig8_metrics(sl; t_start = 0.0, settle_time = 10.0, hf_window = 0.5,
                      lap_frac = 0.5, min_excursion = deg2rad(5.0),
                      az_center = nothing, v_steering = 0.2)
    stats_start = t_start + settle_time
    settled = findall(>=(stats_start), sl.time)
    isempty(settled) && return nothing

    d = Float64.(sl.var_01[settled])
    t_all = Float64.(sl.time)
    dt = length(t_all) > 1 ? mean(diff(t_all)) : 0.01

    fp = Float64.(getindex.(sl.winch_force[settled], 1))

    # Turn rate recomputed from the logged headings — robust to any stale
    # heading_rate in the log.
    heading_rate_deg =
        [0.0; rad2deg.(wrap2pi.(diff(Float64.(sl.heading)))) ./ diff(t_all)]
    psi_dot_settled = heading_rate_deg[settled]

    # Residual std after removing a centered moving average of width
    # hf_window [s] — isolates ringing/chatter from the commanded maneuver.
    function hf_std(x)
        n = max(1, round(Int, hf_window / dt))
        half = n ÷ 2
        ma = similar(x)
        for i in eachindex(x)
            lo = max(1, i - half)
            hi = min(length(x), i + half)
            ma[i] = mean(@view x[lo:hi])
        end
        return std(x .- ma)
    end

    steering_hf = hf_std(Float64.(sl.set_steering[settled]))
    turnrate_hf = hf_std(psi_dot_settled)

    # Laps: a lemniscate crosses its centre azimuth twice per lap. Counting bare
    # sign changes of (azimuth - centre) is not enough — a kite stuck in a limit
    # cycle jitters across the centre and reports dozens of phantom laps (a run
    # that never reached the pattern scored 42.5). Require the azimuth to reach
    # out to `lap_frac` of its own half-range on alternating sides before
    # counting a crossing, so only genuine lobe-to-lobe excursions count.
    az = Float64.(sl.azimuth[settled])
    # Count crossings of the PATTERN's centre azimuth when it is known. Using
    # the flown mean instead lets a kite orbiting a circle off to one side score
    # laps it never flew: a run that circled at azimuth 29-45°, never crossing
    # the pattern centre at 0°, reported 14 laps.
    az_c = az_center === nothing ? mean(az) : az_center
    half_range = maximum(abs.(az .- az_c))
    # The band needs an ABSOLUTE floor as well as a relative one: scaling it by
    # the kite's own azimuth range normalises away exactly the thing being
    # tested, so a ±1.4° limit cycle still crossed a purely relative band and
    # scored 42.5 laps. A real lemniscate swings ±A (tens of degrees).
    band = max(lap_frac * half_range, min_excursion)
    crossings = 0
    side = 0
    for a in az
        if a - az_c > band && side <= 0
            side = 1
            crossings += 1
        elseif a - az_c < -band && side >= 0
            side = -1
            crossings += 1
        end
    end
    laps = max(0, crossings - 1) / 2   # the first arrival is not yet a crossing

    # Steering saturation: fraction of the settled window spent within 2% of
    # the largest commanded magnitude (a proxy for the clamp, which is not
    # logged).
    us = abs.(Float64.(sl.set_steering[settled]))
    us_max = maximum(us)
    sat_frac = us_max > 0 ? count(>=(0.98 * us_max), us) / length(us) : 0.0

    # TAPE rate saturation — the honest companion to `sat_frac` above.
    #
    # `set_steering` is the COMMAND and `steering` is what the KCU tape actually
    # reached; on the V3 they are nothing like each other. The tape is rate
    # limited to `v_steering` [1/s] (the `kcu:` section of the settings YAML),
    # and measured on a 2026-08-02 figure-eight run the command sat on its
    # ±0.300 clamp 88% of the time while the tape reached it in 0% of samples —
    # it was slewing at its limit for 67% of them instead. Reading `sat_frac`
    # alone therefore says "out of steering authority" when the truth is "out of
    # steering RATE", and the two call for opposite fixes: more amplitude makes
    # a rate-limited actuator worse, because a larger command means longer
    # slewing and more phase lag (measured: 62.7° at the loop's own 0.181 Hz,
    # against 24.7° for the identified small-signal dead time).
    act = Float64.(sl.steering[settled])
    tape_rate = length(act) > 1 ? abs.(diff(act)) ./ dt : Float64[]
    tape_rate_frac = isempty(tape_rate) ? 0.0 :
        count(>=(0.95 * v_steering), tape_rate) / length(tape_rate)
    max_tape_rate = isempty(tape_rate) ? 0.0 : maximum(tape_rate)
    max_steering_delivered = maximum(abs.(act))

    return (;
        stats_start,
        laps,
        rms_d = sqrt(mean(d .^ 2)),
        mean_d = mean(d),
        max_d = maximum(d),
        min_elevation_settled = rad2deg(minimum(sl.elevation[settled])),
        min_elevation_all = rad2deg(minimum(sl.elevation)),
        mean_force = mean(fp),
        std_force = std(fp),
        cv_force = std(fp) / mean(fp),
        max_turn_rate = maximum(abs.(psi_dot_settled)),
        steering_hf_std = steering_hf,
        turnrate_hf_std = turnrate_hf,
        max_steering_used = us_max,
        steering_sat_frac = sat_frac,
        max_steering_delivered,
        tape_rate_frac,
        max_tape_rate,
        v_steering,
    )
end

"""
    print_fig8_metrics(sl; kwargs...)

[`fig8_metrics`](@ref) plus a human-readable summary and a pass/fail line
against the PlanFig8.md success criteria. Returns the metrics NamedTuple (or
`nothing`, with a `@warn`, if unavailable).
"""
function print_fig8_metrics(sl; t_start = 0.0, settle_time = 10.0,
                            hf_window = 0.5, min_elevation = 10.0,
                            max_rms_d = 3.0, max_d_limit = 8.0, min_laps = 3.0,
                            lap_frac = 0.5, min_excursion = deg2rad(5.0),
                            az_center = nothing, v_steering = 0.2)
    m = fig8_metrics(sl; t_start, settle_time, hf_window, lap_frac, min_excursion,
                     az_center, v_steering)
    if m === nothing
        @warn "No settled samples — no figure-eight metrics."
        return nothing
    end
    @printf("Fig8 (settled from t=%.1fs): laps=%.1f | RMS d=%.2f° mean=%.2f° max=%.2f°\n",
            m.stats_start, m.laps, m.rms_d, m.mean_d, m.max_d)
    @printf("  elevation: min settled=%.1f° min WHOLE RUN=%.1f° | peak ψ̇=%.0f°/s\n",
            m.min_elevation_settled, m.min_elevation_all, m.max_turn_rate)
    @printf("  tether force: mean=%.0fN std=%.0fN (CV=%.1f%%)\n",
            m.mean_force, m.std_force, 100 * m.cv_force)
    @printf("  steering: peak |u_s|=%.3f, %.0f%% of time within 2%% of it | HF std: steering=%.4f turnrate=%.2f°/s\n",
            m.max_steering_used, 100 * m.steering_sat_frac,
            m.steering_hf_std, m.turnrate_hf_std)
    @printf("  tape: delivered peak |u_s|=%.3f of %.3f commanded | RATE-limited %.0f%% of the time (peak %.3f/s of %.3f/s)\n",
            m.max_steering_delivered, m.max_steering_used,
            100 * m.tape_rate_frac, m.max_tape_rate, m.v_steering)

    checks = [
        ("laps >= $(min_laps)",            m.laps >= min_laps),
        ("RMS d < $(max_rms_d)°",          m.rms_d < max_rms_d),
        ("max d < $(max_d_limit)°",        m.max_d < max_d_limit),
        ("min elevation > $(min_elevation)° (whole run)",
                                           m.min_elevation_all > min_elevation),
    ]
    failed = [name for (name, ok) in checks if !ok]
    if isempty(failed)
        @info "Success criteria: all $(length(checks)) passed."
    else
        @warn "Success criteria FAILED: " * join(failed, ", ")
    end
    return m
end
