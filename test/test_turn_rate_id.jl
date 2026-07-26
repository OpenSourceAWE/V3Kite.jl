# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Unit tests for src/turn_rate_id.jl. Everything here runs on synthetic signals
# — no simulation, no log on disk — so the whole file is fast.
#
# The end-to-end tests use a log that satisfies the turn-rate law *exactly*: a
# smooth heading history is chosen first, its exact discrete turn rate is taken,
# and the steering is then solved for so that
#
#     psi_dot = c1*v_a*u_s + c2/v_a*sin(psi)*cos(beta)
#
# holds sample by sample at the indices the identification actually aligns.
# `identify_turn_rate_law` must therefore recover `c1`/`c2` to machine precision
# with a zero residual — anything else is an alignment or sign error.

using Pkg
if ! ("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test
using V3Kite

"""
    _consistent_log(; c1, c2, dt, n, v_a, beta, amp, freq) -> (sl, us_true)

Synthetic syslog table that satisfies the turn-rate law exactly, plus the
steering series `us_true` aligned the way `identify_turn_rate_law` aligns it
(i.e. `sl.steering[2:end]`). The azimuth is constant so that the frame-transport
term of `calc_turn_rate` vanishes and the turn rate is a plain `diff(heading)/dt`.
"""
function _consistent_log(; c1 = 0.05, c2 = -0.2, dt = 0.05, n = 601,
                           v_a = 20.0, beta = deg2rad(70.0),
                           amp = 0.5, freq = 0.1)
    time = collect((0:n-1) .* dt)
    psi = amp .* sin.(2π * freq .* time)          # stays well inside ±π
    rate = diff(psi) ./ dt                        # exact discrete turn rate
    # Solve the law for the steering at the aligned indices (2:n).
    us_true = (rate .- (c2 / v_a) .* sin.(psi[2:n]) .* cos(beta)) ./ (c1 * v_a)
    sl = (time     = time,
          heading  = psi,
          azimuth  = zeros(n),
          elevation = fill(beta, n),
          v_app    = fill(v_a, n),
          steering = [0.0; us_true])
    return sl, us_true
end

@testset "Turn Rate Identification" begin

    @testset "Correlation" begin
        a = [1.0, 3.0, 2.0, 5.0, 4.0]
        @test V3Kite._corr(a, a) ≈ 1.0
        @test V3Kite._corr(a, -a) ≈ -1.0
        @test V3Kite._corr(a, 2 .* a .+ 3) ≈ 1.0     # scale/offset invariant
        @test V3Kite._corr(a, fill(2.0, 5)) == 0.0   # constant input, no NaN
        @test V3Kite._corr(fill(2.0, 5), a) == 0.0
    end

    @testset "shift_delay" begin
        u = [1, 2, 3, 4]
        @test shift_delay(u, 0) == [1.0, 2.0, 3.0, 4.0]
        @test eltype(shift_delay(u, 0)) == Float64
        @test shift_delay(u, 1) == [0.0, 1.0, 2.0, 3.0]
        @test shift_delay(u, 3) == [0.0, 0.0, 0.0, 1.0]
        @test length(shift_delay(u, 2)) == length(u)   # length is preserved
        # Non-positive delays are a no-op, delays past the end blank the series
        @test shift_delay(u, -2) == [1.0, 2.0, 3.0, 4.0]
        @test shift_delay(u, 4) == zeros(4)
        @test shift_delay(u, 99) == zeros(4)
    end

    @testset "estimate_delay" begin
        dt = 0.1
        u = [sin(0.3k) + 0.5cos(0.11k) for k in 1:200]   # not self-similar
        for d in (0, 1, 5, 17)
            dd, cc = estimate_delay(u, shift_delay(u, d), dt; t_max = 5.0)
            @test dd == d
            @test cc > 0.99
        end
        # Delays beyond t_max/dt are not searched
        d, _ = estimate_delay(u, shift_delay(u, 17), dt; t_max = 1.0)
        @test d <= round(Int, 1.0 / dt)
        # Only non-negative shifts are searched: a response that *leads* the
        # input reports d = 0 with a degraded correlation rather than a
        # negative lead. Shown on a single bump, since a periodic input
        # correlates with itself at a positive shift too.
        bump(k0) = [exp(-((k - k0) / 10)^2) for k in 1:200]
        d, cc = estimate_delay(bump(100), bump(95), dt; t_max = 5.0)
        @test d == 0
        @test cc < 1.0
        # ... whereas the same bump lagging by 5 samples is found exactly
        d, cc = estimate_delay(bump(100), bump(105), dt; t_max = 5.0)
        @test d == 5
        @test cc > 0.99
        # The delay is returned in samples, seconds are the caller's business
        d, _ = estimate_delay(u, shift_delay(u, 5), dt; t_max = 5.0)
        @test d * dt ≈ 0.5
        @test_throws AssertionError estimate_delay(u, u[1:end-1], dt)
    end

    @testset "turn_rate_gain" begin
        G0 = 0.04
        us = collect(-0.2:0.01:0.2)
        v_app = fill(20.0, length(us))
        rate = G0 .* v_app .* us                 # exactly one constant gain
        g = turn_rate_gain(us, rate, v_app; min_steering = 0.025)
        @test g.n == count(abs.(us) .> 0.025)
        @test g.mean ≈ G0
        @test g.std ≈ 0.0 atol=1e-12
        @test g.rel_std ≈ 0.0 atol=1e-12
        # The masked-out samples are NaN in the full-length series, so the
        # result stays plottable against the original time base
        @test length(g.G) == length(us)
        @test g.mask == (abs.(us) .> 0.025)
        @test all(isnan, g.G[.!g.mask])
        @test !any(isnan, g.G[g.mask])
        @test count(!isnan, g.G) == g.n

        # A higher threshold keeps strictly fewer samples
        g2 = turn_rate_gain(us, rate, v_app; min_steering = 0.1)
        @test g2.n < g.n
        @test g2.mean ≈ G0

        # Everything masked out: no statistics, but no exception either
        g3 = turn_rate_gain(us, rate, v_app; min_steering = 10.0)
        @test g3.n == 0
        @test isnan(g3.mean)
        @test isnan(g3.std)
        @test all(isnan, g3.G)
    end

    @testset "fit_c1_c2" begin
        c1_true, c2_true = 0.05, -0.2
        v_app = fill(20.0, 400)
        psi = [0.5sin(0.05k) for k in 1:400]
        beta = fill(deg2rad(70.0), 400)
        us = [0.2cos(0.05k) for k in 1:400]
        psi_dot = c1_true .* v_app .* us .+
                  (c2_true ./ v_app) .* sin.(psi) .* cos.(beta)

        f = fit_c1_c2(v_app, psi, beta, psi_dot, us)
        @test f.c1 ≈ c1_true
        @test f.c2 ≈ c2_true
        @test f.n == 400
        @test f.rms ≈ 0.0 atol=1e-12      # the data is exactly consistent
        @test f.se1 ≈ 0.0 atol=1e-12
        @test f.se2 ≈ 0.0 atol=1e-12
        @test isfinite(f.cond) && f.cond > 1.0

        # Adding noise leaves the coefficients close but the residual and the
        # standard errors strictly positive.
        noise = [1e-3 * sin(2.7k) for k in 1:400]
        fn = fit_c1_c2(v_app, psi, beta, psi_dot .+ noise, us)
        @test fn.c1 ≈ c1_true rtol=0.05
        @test fn.rms > 0.0
        @test fn.se1 > 0.0 && fn.se2 > 0.0
    end

    @testset "est_steering" begin
        # est_steering inverts the law, so it must return the steering the
        # turn rate was built from.
        c1_true, c2_true = 0.05, -0.2
        v_app = fill(20.0, 200)
        psi = [0.5sin(0.05k) for k in 1:200]
        beta = fill(deg2rad(70.0), 200)
        us = [0.2cos(0.05k) for k in 1:200]
        psi_dot = c1_true .* v_app .* us .+
                  (c2_true ./ v_app) .* sin.(psi) .* cos.(beta)
        @test est_steering(c1_true, c2_true, v_app, psi, beta, psi_dot) ≈ us
    end

    @testset "identify_turn_rate_law" begin
        dt = 0.05
        c1_true, c2_true = 0.05, -0.2
        sl, us_true = _consistent_log(; c1 = c1_true, c2 = c2_true, dt, n = 601)

        r = identify_turn_rate_law(sl; dt, t_start = 0.0, t_max_delay = 3.0)

        # The law is recovered exactly from exactly consistent data
        @test r.c1 ≈ c1_true
        @test r.c2 ≈ c2_true
        @test r.rms ≈ 0.0 atol=1e-12
        @test r.us_est ≈ r.us_del

        # calc_turn_rate drops the first sample, so everything starts at index 2
        @test length(r.time) == length(sl.time) - 1
        @test r.time == sl.time[2:end]
        @test r.us == us_true
        @test r.t_start == sl.time[2]
        @test r.t_end == sl.time[end]
        @test length(r.rate) == length(r.time)
        @test length(r.G) == length(r.time)
        @test r.n_fit == length(r.time)

        # No lag was built in, so none is identified and us_del == us
        @test r.delay_samples == 0
        @test r.delay_sec == 0.0
        @test r.delay_corr > 0.99
        @test r.us_del == r.us

        # The turn rate is the frame-transport-corrected one; the azimuth is
        # constant here, so it reduces to diff(heading)/dt
        @test r.rate ≈ diff(sl.heading) ./ dt

        @test r.min_steering == 0.025
        @test r.n_gain == count(abs.(r.us_del) .> r.min_steering)
        @test isfinite(r.G_mean)
    end

    @testset "identify_turn_rate_law Delay" begin
        # A steering series that leads the turn rate by a known number of
        # samples must be re-aligned before the fit, otherwise c1 is diluted.
        dt = 0.05
        d_true = 8
        c1_true, c2_true = 0.05, -0.2
        sl0, us_true = _consistent_log(; c1 = c1_true, c2 = c2_true, dt, n = 601)
        # Feed the identification a log whose steering happens d_true samples
        # *earlier* than the response it produced.
        lead = [us_true[1 + d_true:end]; fill(0.0, d_true)]
        sl = merge(sl0, (steering = [0.0; lead],))

        r = identify_turn_rate_law(sl; dt, t_start = 0.0, t_max_delay = 3.0)
        @test r.delay_samples == d_true
        @test r.delay_sec ≈ d_true * dt
        @test r.delay_corr > 0.99
        # Re-aligned, the fit is back on the true coefficients over the part of
        # the record that the shift did not blank out.
        @test r.c1 ≈ c1_true rtol=0.05
    end

    @testset "identify_turn_rate_law Window" begin
        dt = 0.05
        sl, _ = _consistent_log(; dt, n = 601)
        t_start = 10.0

        r = identify_turn_rate_law(sl; dt, t_start, t_max_delay = 3.0)
        @test r.t_start >= t_start
        @test r.t_start - dt < t_start          # the first sample at or past it
        @test all(>=(t_start), r.time)
        @test length(r.time) == count(>=(t_start), sl.time[2:end])
        @test r.c1 ≈ 0.05                        # still exact on the sub-window

        # A window that starts after the end of the log is an error, not an
        # empty fit
        @test_throws ErrorException identify_turn_rate_law(sl; dt, t_start = 1e6)
    end

    @testset "format_turn_rate_report" begin
        dt = 0.05
        sl, _ = _consistent_log(; dt, n = 601)
        r = identify_turn_rate_law(sl; dt, t_start = 0.0, t_max_delay = 3.0)

        rep = format_turn_rate_report(r)
        @test rep isa String
        @test occursin("turn-rate law identification", rep)
        @test occursin("steering delay", rep)
        @test occursin("gain G = psi_dot/(v_a*u_s)", rep)
        @test occursin("least squares fit", rep)
        @test occursin("residual RMS", rep)
        @test occursin("cond(A)", rep)
        @test endswith(rep, "\n")

        # The verdict follows the G-scatter criterion in both directions
        @test occursin("PASS", format_turn_rate_report(r;
                       max_rel_std = 2 * r.G_rel_std))
        @test occursin("FAIL", format_turn_rate_report(r;
                       max_rel_std = 0.5 * r.G_rel_std))
        # A non-finite scatter is a FAIL, not an exception
        @test occursin("FAIL", format_turn_rate_report(
                       merge(r, (G_rel_std = NaN,))))
    end

end
