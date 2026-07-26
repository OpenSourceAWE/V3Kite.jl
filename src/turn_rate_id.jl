# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Turn-rate-law identification from a V3 kite log.

Fits the standard two-parameter turn-rate law

    ψ̇ = c1 · v_a · u_s + c2 / v_a · sin(ψ) · cos(β)

to a logged run, where `ψ` is the heading, `β` the elevation, `v_a` the apparent
wind speed and `u_s` the *actual* (KCU tape-lagged) steering input. The first
term is the steering response, the second the gravity-induced turn towards the
zenith-down direction. `c1` has unit 1/m (`u_s` is dimensionless because
`init` sets `set.cs_4p = 1.0`), `c2` is dimensionless.

Ported from the analysis part of `steering_test_4p.jl` in KiteModels.jl, with
three changes that the V3 model requires:

1. The turn rate comes from `calc_turn_rate` (`coordinate_utils.jl`), not from a plain
   `diff(heading)/dt`. Heading lives in the rotating tangent-sphere frame, so
   the frame-transport term `−φ̇·sin(β)` has to be subtracted; at the ~70°
   elevation of these runs `sin(β) ≈ 0.94`, so leaving it out biases both
   coefficients.
2. The fit uses `sl.steering` (what the KCU tape actually reached), never
   `sl.set_steering` (the command). With `v_steering = 0.2` s⁻¹ and
   `steering_gain = 3.0` the actuator is strongly slew-limited during a
   bang-bang reversal, and fitting the command would identify the actuator
   rather than the kite.
3. Besides the relative scatter of the single-parameter gain
   `G = ψ̇ / (v_a·u_s)`, the least-squares fit also reports standard errors on
   `c1`/`c2`, from `σ²·(AᵀA)⁻¹`. Those are optimistic — the residuals of a
   fitted flight path are strongly autocorrelated, so the effective sample
   count is far below `n` — but they do expose an ill-conditioned fit, which
   the bare `A \\ b` of the original script cannot.

Used by `examples/steering_test_v3.jl` (which simulates, then reports) and by
`examples/steering_test_v3_plots.jl` (which re-runs the identical analysis on
the log already on disk, so both print the same numbers):

    r = identify_turn_rate_law(sl; dt = DT, t_start = T_START)
    print(format_turn_rate_report(r))
"""

using Printf

"""
    _corr(a, b) -> Float64

Pearson correlation coefficient of `a` and `b`. Returns 0.0 if either input is
constant.
"""
function _corr(a::AbstractVector, b::AbstractVector)
    da = a .- mean(a)
    db = b .- mean(b)
    den = sqrt(sum(abs2, da) * sum(abs2, db))
    return den < eps() ? 0.0 : sum(da .* db) / den
end

"""
    estimate_delay(u, y, dt; t_max=10.0) -> (d, corr)

Delay of the response `y` behind the input `u`, in samples: the shift `d ≥ 0`
that maximises the correlation between `u[1:end-d]` and `y[1+d:end]`, searched
over `0 .. t_max/dt`. Also returns the correlation at that shift, which is the
one number that says whether the delay estimate means anything (a weak peak,
say below ~0.5, means the shift is fitting noise).

Only non-negative delays are searched: the kite cannot respond before the tape
moves, so a negative peak would indicate a sign or alignment error rather than
a physical lead. Such a case shows up as `d = 0` with a poor correlation.

This replaces `StatsBase.crosscor` used by the KiteModels version — the sign
convention here is explicit, and the examples environment does not carry
StatsBase.
"""
function estimate_delay(u::AbstractVector, y::AbstractVector, dt::Real;
                        t_max::Real = 10.0)
    n = length(u)
    @assert n == length(y) "estimate_delay: inputs must have equal length"
    d_max = min(n - 2, round(Int, t_max / dt))
    best_d, best_c = 0, -Inf
    for d in 0:d_max
        c = _corr(view(u, 1:n-d), view(y, 1+d:n))
        if c > best_c
            best_c, best_d = c, d
        end
    end
    return best_d, best_c
end

"""
    shift_delay(u, d) -> Vector{Float64}

`u` delayed by `d` samples: `d` leading zeros, the tail dropped so the length is
preserved. Used to align the steering input with the delayed turn-rate response
before fitting.
"""
function shift_delay(u::AbstractVector, d::Int)
    d <= 0 && return collect(float.(u))
    n = length(u)
    d >= n && return zeros(Float64, n)
    return [zeros(Float64, d); float.(u[1:n-d])]
end

"""
    turn_rate_gain(us, rate, v_app; min_steering) -> NamedTuple

Single-parameter turn-rate gain `G = ψ̇ / (v_a · u_s)` [1/m] and its scatter,
over the samples with `abs(us) > min_steering`. Samples below that threshold are
dropped because `G` is a ratio that blows up as `u_s` crosses zero — which the
slew-limited tape does slowly, so a low threshold keeps a lot of near-singular
samples.

Returns `(; G, mask, mean, std, rel_std, n)`, where `G` is the full-length
series with the masked-out entries set to `NaN` (ready to plot) and the
statistics are taken over the surviving samples.
"""
function turn_rate_gain(us::AbstractVector, rate::AbstractVector,
                        v_app::AbstractVector; min_steering::Real = 0.025)
    G = rate ./ v_app ./ us
    mask = abs.(us) .> min_steering
    G[.!mask] .= NaN
    kept = filter(!isnan, G)
    n = length(kept)
    G_mean = n > 0 ? mean(kept) : NaN
    G_std = n > 1 ? std(kept) : NaN
    return (G = G, mask = mask, mean = G_mean, std = G_std,
            rel_std = abs(G_std / G_mean), n = n)
end

"""
    fit_c1_c2(v_app, psi, beta, psi_dot, us) -> NamedTuple

Least-squares fit of `c1`, `c2` in

    c1 · v_a · u_s + c2 / v_a · sin(ψ) · cos(β) − ψ̇ = 0

`A · c = b`, with the columns of `A` the two regressors and `b = psi_dot`;
solved by QR (`A \\ b`), i.e. minimising the sum of squared residuals.

Returns `(; c1, c2, se1, se2, rms, cond, n)`: the coefficients, their standard
errors from `σ̂²·(AᵀA)⁻¹` with `σ̂² = RSS/(n−2)`, the residual RMS [rad/s], and
the condition number of `A`. A large `cond` means the two regressors are
collinear over this data set — the `c2` term is only excited when `ψ` swings
well away from zero, so a narrow heading band gives a well-determined `c1` and
an almost arbitrary `c2`.
"""
function fit_c1_c2(v_app::AbstractVector, psi::AbstractVector,
                   beta::AbstractVector, psi_dot::AbstractVector,
                   us::AbstractVector)
    col1 = v_app .* us
    col2 = sin.(psi) .* cos.(beta) ./ v_app
    A = [col1 col2]
    c = A \ psi_dot
    resid = psi_dot .- A * c
    n = length(psi_dot)
    dof = max(n - 2, 1)
    sigma2 = sum(abs2, resid) / dof
    cov = sigma2 .* inv(A' * A)
    return (c1 = c[1], c2 = c[2],
            se1 = sqrt(abs(cov[1, 1])), se2 = sqrt(abs(cov[2, 2])),
            rms = sqrt(sum(abs2, resid) / n), cond = cond(A), n = n)
end

"""
    est_steering(c1, c2, v_app, psi, beta, psi_dot) -> Vector

The steering input the fitted law needs to explain the measured turn rate,

    u_s = ψ̇/(c1·v_a) − c2/(c1·v_a²)·sin(ψ)·cos(β)

Plotted against the actual (delayed) `u_s` this is the visual goodness-of-fit
check of the original script: the two curves overlaying means the law
reproduces the run.
"""
function est_steering(c1, c2, v_app::AbstractVector, psi::AbstractVector,
                      beta::AbstractVector, psi_dot::AbstractVector)
    return psi_dot ./ (v_app .* c1) .- c2 ./ (c1 .* v_app .^ 2) .*
           sin.(psi) .* cos.(beta)
end

"""
    identify_turn_rate_law(sl; dt, t_start=0.0, min_steering=0.025,
                           t_max_delay=10.0) -> NamedTuple

Full identification for the syslog table `sl` (from `KiteUtils.syslog(logger)`
or `load_log(name).syslog`), sampled at `dt` [s].

Only samples with `time >= t_start` enter the analysis, which is how the
pre-excitation part of the run (constant zero steering, no information about
either coefficient) is kept out of the fit. `min_steering` is the `G` mask
threshold, `t_max_delay` [s] the delay search range.

The steering input is delayed by the identified transport delay before the fit,
so `c1` is not diluted by the phase lag between tape motion and turn rate.

Returns a NamedTuple with the analysis window arrays (`time`, `us`, `us_del`,
`rate`, `v_app`, `psi`, `beta`, `G`, `us_est`), the delay (`delay_samples`,
`delay_sec`, `delay_corr`), the gain statistics (`G_mean`, `G_std`,
`G_rel_std`, `n_gain`) and the fit (`c1`, `c2`, `se1`, `se2`, `rms`, `cond`,
`n_fit`). Suitable for `format_turn_rate_report`.
"""
function identify_turn_rate_law(sl; dt::Real, t_start::Real = 0.0,
                                min_steering::Real = 0.025,
                                t_max_delay::Real = 10.0)
    # calc_turn_rate returns a series aligned to sl.time[2:end], so every other
    # signal is taken from index 2 on as well.
    rate_all = calc_turn_rate(sl; source = :heading, dt)
    rng = 2:length(sl.time)
    time = collect(sl.time[rng])
    keep = findall(>=(t_start), time)
    isempty(keep) && error("identify_turn_rate_law: no samples at t >= $t_start s")

    time = time[keep]
    rate = rate_all[keep]
    us = collect(sl.steering[rng][keep])
    v_app = collect(sl.v_app[rng][keep])
    psi = wrap_to_pi.(sl.heading[rng][keep])
    beta = collect(sl.elevation[rng][keep])

    d, d_corr = estimate_delay(us, rate ./ v_app, dt; t_max = t_max_delay)
    us_del = shift_delay(us, d)

    gain = turn_rate_gain(us_del, rate, v_app; min_steering)
    fit = fit_c1_c2(v_app, psi, beta, rate, us_del)
    us_est = est_steering(fit.c1, fit.c2, v_app, psi, beta, rate)

    return (time = time, us = us, us_del = us_del, rate = rate, v_app = v_app,
            psi = psi, beta = beta, G = gain.G, us_est = us_est,
            delay_samples = d, delay_sec = d * dt, delay_corr = d_corr,
            G_mean = gain.mean, G_std = gain.std, G_rel_std = gain.rel_std,
            n_gain = gain.n, min_steering = min_steering,
            c1 = fit.c1, c2 = fit.c2, se1 = fit.se1, se2 = fit.se2,
            rms = fit.rms, cond = fit.cond, n_fit = fit.n,
            t_start = first(time), t_end = last(time))
end

"""
    format_turn_rate_report(r; max_rel_std=0.35) -> String

Human-readable report of an `identify_turn_rate_law` result `r`, including the
verdict on the identification success criterion: the relative standard
deviation of `G` below `max_rel_std`.
"""
function format_turn_rate_report(r; max_rel_std::Real = 0.35)
    io = IOBuffer()
    @printf(io, "turn-rate law identification, t = %.2f .. %.2f s, %d samples\n",
            r.t_start, r.t_end, length(r.time))
    @printf(io, "  steering delay     : %8.2f s   (correlation %.3f)\n",
            r.delay_sec, r.delay_corr)
    @printf(io, "  |u_s| range in fit : %8.3f .. %.3f  (G mask at %.3f)\n",
            minimum(abs.(r.us)), maximum(abs.(r.us)), r.min_steering)
    @printf(io, "gain G = psi_dot/(v_a*u_s), %d of %d samples above the mask\n",
            r.n_gain, length(r.time))
    @printf(io, "  G                  : %8.4f 1/m ± %.2f %%\n",
            r.G_mean, 100 * r.G_rel_std)
    @printf(io, "                     : %8.3f °/m ± %.2f %%\n",
            rad2deg(r.G_mean), 100 * r.G_rel_std)
    @printf(io, "least squares fit of psi_dot = c1*v_a*u_s + c2/v_a*sin(psi)*cos(beta)\n")
    @printf(io, "  c1                 : %8.4f 1/m ± %.4f  (%.2f %%)\n",
            r.c1, r.se1, 100 * abs(r.se1 / r.c1))
    @printf(io, "  c2                 : %8.4f [-] ± %.4f  (%.2f %%)\n",
            r.c2, r.se2, 100 * abs(r.se2 / r.c2))
    @printf(io, "  residual RMS       : %8.4f rad/s (%.3f °/s)\n",
            r.rms, rad2deg(r.rms))
    @printf(io, "  cond(A)            : %8.1f\n", r.cond)
    ok = isfinite(r.G_rel_std) && r.G_rel_std < max_rel_std
    @printf(io, "  %s identification: G scatter %.2f %% %s %.0f %%\n",
            ok ? "PASS" : "FAIL", 100 * r.G_rel_std, ok ? "<" : ">=",
            100 * max_rel_std)
    return String(take!(io))
end
