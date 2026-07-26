# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Unit tests for src/ripple_metrics.jl. Everything here runs on synthetic
# signals — no simulation, no log on disk — so the whole file is fast.

using Pkg
if ! ("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test
using V3Kite

@testset "Ripple Metrics" begin

    @testset "RippleSettings Defaults" begin
        rs = RippleSettings()
        @test rs.t_start == 2.0
        @test rs.detrend_window == 0.5
        @test rs.f_min == 0.5
        @test rs.f_max == 30.0
    end

    @testset "RippleSettings From YAML" begin
        old_path = get_data_path()
        try
            # Bundled file: must exist and be self-consistent.
            set_data_path(v3_data_path())
            @test isfile(joinpath(v3_data_path(), "ripple_settings.yaml"))
            rs = RippleSettings("ripple_settings.yaml")
            @test rs.t_start >= 0.0
            @test rs.detrend_window > 0.0
            @test 0.0 <= rs.f_min < rs.f_max

            # A partial file overrides only the keys it lists; the rest fall
            # back to the struct defaults.
            tmp = mktempdir()
            write(joinpath(tmp, "partial.yaml"),
                  "ripple_settings:\n  t_start: 3.5\n  f_max: 12.0\n")
            set_data_path(tmp)
            rs = RippleSettings("partial.yaml")
            @test rs.t_start == 3.5
            @test rs.f_max == 12.0
            @test rs.detrend_window == RippleSettings().detrend_window
            @test rs.f_min == RippleSettings().f_min
        finally
            set_data_path(old_path)
        end
    end

    @testset "Centred Moving Average" begin
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        # half = 0 is the identity
        @test V3Kite._centred_ma(y, 0) == y
        # A constant is passed through unchanged (any window)
        @test V3Kite._centred_ma(fill(7.0, 5), 2) ≈ fill(7.0, 5)
        # Truncated (not zero-padded) at the edges
        @test V3Kite._centred_ma(y, 1) ≈ [1.5, 2.0, 3.0, 4.0, 4.5]
        # A linear ramp is reproduced exactly wherever the window is full
        @test V3Kite._centred_ma(y, 1)[2:4] ≈ y[2:4]
    end

    @testset "Peak Frequency" begin
        fs = 100.0
        t = collect(0:(1/fs):10)
        x = sin.(2π * 5.0 .* t)
        f_peak, f_res = V3Kite._peak_frequency(x, fs, 0.5, 30.0)
        @test f_res ≈ fs / length(t)
        @test abs(f_peak - 5.0) <= f_res      # within one bin of the true 5 Hz

        # The strongest component inside the band wins, not the strongest overall
        x2 = sin.(2π * 5.0 .* t) .+ 3 .* sin.(2π * 20.0 .* t)
        f_peak, f_res = V3Kite._peak_frequency(x2, fs, 0.5, 10.0)
        @test abs(f_peak - 5.0) <= f_res

        # Too few samples, and an empty search band, both yield NaN
        @test isnan(V3Kite._peak_frequency(x[1:5], fs, 0.5, 30.0)[1])
        @test isnan(V3Kite._peak_frequency(x, fs, 60.0, 80.0)[1])  # above Nyquist
    end

    # Reference signal for the metrics tests: a 5 Hz, 0.5-unit-amplitude ripple
    # on a constant trim of 10, sampled at 100 Hz for 10 s.
    fs = 100.0
    f_ripple = 5.0
    amp = 0.5
    trim = 10.0
    time = collect(0:(1/fs):10)
    ripple = amp .* sin.(2π * f_ripple .* time)
    rs = RippleSettings(t_start = 2.0, detrend_window = 0.5,
                        f_min = 0.5, f_max = 30.0)

    @testset "Ripple Metrics Window" begin
        r = ripple_metrics(time, trim .+ ripple, rs)
        @test r.t_start == 2.0
        @test r.t_end == 10.0
        @test r.n == length(time) - 200      # 2 s of 100 Hz samples dropped
        @test r.fs ≈ fs
        @test r.f_res ≈ r.fs / r.n
        @test r.mean ≈ trim atol=1e-3        # raw mean = the trim value

        @test_throws ErrorException ripple_metrics(time, trim .+ ripple,
                                                   RippleSettings(t_start = 99.0))
    end

    @testset "Ripple Metrics Amplitude" begin
        r = ripple_metrics(time, trim .+ ripple, rs)
        # The detrend attenuates the ripple itself (a 0.5 s moving average
        # passes ~12 % of a 5 Hz sine), so the reported values are low by that
        # much and no more — see the module docstring.
        @test 0.80 <= r.rms / (amp / sqrt(2)) <= 0.95      # 0.876 as it stands
        @test 0.80 <= r.pkpk / (2 * amp) <= 0.98           # 0.938 as it stands
        @test abs(r.f_peak - f_ripple) <= r.f_res

        # Exactly linear in the amplitude
        r2 = ripple_metrics(time, trim .+ 2 .* ripple, rs)
        @test r2.rms ≈ 2 * r.rms
        @test r2.pkpk ≈ 2 * r.pkpk
        @test r2.f_peak ≈ r.f_peak
    end

    @testset "Ripple Metrics Trim Invariance" begin
        r = ripple_metrics(time, trim .+ ripple, rs)
        # Shifting the trim moves `mean` and nothing else
        r_shift = ripple_metrics(time, trim + 3.0 .+ ripple, rs)
        @test r_shift.mean ≈ r.mean + 3.0
        @test r_shift.rms ≈ r.rms
        @test r_shift.pkpk ≈ r.pkpk
        @test r_shift.f_peak ≈ r.f_peak

        # A slow drift is removed by the detrend, so the ripple measurement
        # survives it (the moving average is truncated at the very end of the
        # record, which is what the residual tolerance covers).
        r_drift = ripple_metrics(time, trim .+ 0.2 .* time .+ ripple, rs)
        @test r_drift.rms ≈ r.rms rtol=0.01
        @test abs(r_drift.f_peak - r.f_peak) <= r.f_res
    end

    @testset "Ripple Metrics Constant Signal" begin
        r = ripple_metrics(time, fill(trim, length(time)), rs)
        @test r.rms ≈ 0.0 atol=1e-12
        @test r.pkpk ≈ 0.0 atol=1e-12
        @test r.mean ≈ trim
    end

    @testset "aoa_ripple" begin
        sl = (time = time, AoA = deg2rad.(trim .+ ripple))
        r = aoa_ripple(sl; rs)
        # The AoA is logged in radians and reported in degrees
        @test r.mean ≈ trim atol=1e-3
        # ... i.e. identical to running ripple_metrics on the degree signal
        @test r.rms ≈ ripple_metrics(time, trim .+ ripple, rs).rms
        @test abs(r.f_peak - f_ripple) <= r.f_res

        # The default `rs` comes from data/ripple_settings.yaml
        old_path = get_data_path()
        try
            set_data_path(v3_data_path())
            r_default = aoa_ripple(sl)
            @test r_default.rms > 0.0
            @test r_default.t_start ≈ RippleSettings("ripple_settings.yaml").t_start
        finally
            set_data_path(old_path)
        end
    end

    @testset "format_ripple_report" begin
        r = aoa_ripple((time = time, AoA = deg2rad.(trim .+ ripple)); rs)

        # Bare report: ripple only
        rep = format_ripple_report(r)
        @test rep isa String
        @test occursin("AoA ripple", rep)
        @test occursin("ripple RMS", rep)
        @test occursin("PSD peak", rep)
        @test occursin("mean AoA (trim)", rep)
        @test !occursin("mean elevation", rep)
        @test !occursin("solver cost", rep)
        @test !occursin("wall clock", rep)
        @test endswith(rep, "\n")

        # `sl` adds the parked trim state
        sl = (time = time,
              elevation = fill(deg2rad(70.0), length(time)),
              winch_force = [[500.0] for _ in time])
        rep = format_ripple_report(r; sl)
        @test occursin("mean elevation", rep)
        @test occursin("70.000", rep)
        @test occursin("mean tether force", rep)
        @test occursin("500.0 N", rep)

        # `stats` adds the solver cost, `t_loop`/`n_steps` the wall clock
        stats = (naccept = 100, nreject = 3, nf = 250,
                 njacs = 7, nw = 9, nsolve = 42)
        rep = format_ripple_report(r; stats, t_loop = 5.0, n_steps = 200)
        @test occursin("solver cost", rep)
        @test occursin("accepted steps", rep)
        @test occursin("wall clock", rep)
        @test occursin("mean per step!", rep)
        @test occursin("real-time factor", rep)

        # A NaN spectral peak degrades to "n/a" instead of throwing
        rep = format_ripple_report(merge(r, (f_peak = NaN,)))
        @test occursin("n/a", rep)
    end

end
