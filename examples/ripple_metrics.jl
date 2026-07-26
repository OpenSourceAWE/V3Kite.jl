# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Quantify the small ~5.5 Hz angle-of-attack oscillation seen in the parking
examples (see PlanSuppressOscillations.md). In the baseline it is a very lightly
damped mode rung by the initial condition rather than a limit cycle, so the
whole-window RMS mixes a large early amplitude with a small late one — compare
runs at equal `t_start`, or narrow the window to a late interval.

The signal is `sl.AoA` from `KiteUtils.syslog(s.logger)` — the VSM wing AoA filled by
`update_sys_state!`, and the only AoA the `init`/`step!` path logs (`var_04` is
written by `log_state!`, which `step!` never calls).

The parked AoA sits at a non-zero trim value that also drifts slowly while the
elevation is still creeping, so a plain RMS would measure the trim point rather
than the ripple. Everything here is therefore computed on the *detrended*
signal: the AoA minus a centred moving average (`detrend_window`), restricted to
`t >= t_start`.

Included by `simple_parking.jl` (which adds solver cost and wall clock) and by
`simple_parking_plots.jl` (which recomputes the AoA metrics from a log already on
disk, without re-simulating):

    include(joinpath(@__DIR__, "ripple_metrics.jl"))
    r = aoa_ripple(sl)
    print(format_ripple_report(r; sl))

Note that the moving-average detrend also attenuates the ripple itself (a 0.5 s
window passes ~11 % of a 5 Hz sine, so the reported RMS is ~11 % low). That bias
is identical for every run, so it does not affect comparisons.

Tunables live in `ripple_settings.yaml`, not in this file.
"""

using Parameters
using Printf
using Statistics: mean
using YAML

@with_kw mutable struct RippleSettings @deftype Float64
    "Start of the analysis window [s]; discards the post-settling transient"
    t_start = 2.0
    "Length of the centred moving average subtracted before the RMS [s]"
    detrend_window = 0.5
    "Lowest frequency considered when searching the spectral peak [Hz]"
    f_min = 0.5
    "Highest frequency considered when searching the spectral peak [Hz]"
    f_max = 30.0
end

"""
    RippleSettings(filename::String) -> RippleSettings

Load the ripple-analysis settings from the YAML file `filename`, looked up next
to this script (`examples/`). The file must have a top-level `ripple_settings:`
mapping whose keys are field names of `RippleSettings`; any missing key falls
back to the struct default.
"""
function RippleSettings(filename::String)
    dict = YAML.load_file(joinpath(@__DIR__, filename))["ripple_settings"]
    RippleSettings(; (Symbol(k) => Float64(v) for (k, v) in dict)...)
end

"""
    _centred_ma(y, half) -> Vector{Float64}

Centred moving average of `y` over `2*half+1` samples, with the window truncated
(not zero-padded) at the array edges.
"""
function _centred_ma(y::AbstractVector, half::Int)
    n = length(y)
    ma = Vector{Float64}(undef, n)
    for i in 1:n
        ma[i] = mean(@view y[max(1, i - half):min(n, i + half)])
    end
    return ma
end

"""
    _peak_frequency(x, fs, f_min, f_max) -> (f_peak, f_res)

Frequency [Hz] of the largest periodogram peak of `x` (sampled at `fs` Hz) inside
`[f_min, f_max]`, and the frequency resolution `fs/N`. A Hann window suppresses
the leakage from the finite record; the DFT is evaluated directly on the
one-sided frequency grid, which is cheap for the few hundred samples involved and
avoids an FFTW dependency in the examples environment.
"""
function _peak_frequency(x::AbstractVector, fs::Real, f_min::Real, f_max::Real)
    n = length(x)
    f_res = fs / n
    n < 8 && return (NaN, f_res)
    # Hann window, mean-removed so the window does not fold the DC bin into the
    # low-frequency bins we care about.
    xw = (x .- mean(x)) .* [0.5 * (1 - cos(2π * (i - 1) / n)) for i in 1:n]
    k_lo = max(1, ceil(Int, f_min / f_res))
    k_hi = min(n ÷ 2, floor(Int, f_max / f_res))
    k_hi < k_lo && return (NaN, f_res)
    best_k, best_p = k_lo, -1.0
    for k in k_lo:k_hi
        re = 0.0
        im = 0.0
        ω = 2π * k / n
        for i in 1:n
            s, c = sincos(ω * (i - 1))
            re += xw[i] * c
            im -= xw[i] * s
        end
        p = re^2 + im^2
        if p > best_p
            best_p, best_k = p, k
        end
    end
    return (best_k * f_res, f_res)
end

"""
    ripple_metrics(time, y, rs::RippleSettings) -> NamedTuple

Ripple metrics of the signal `y` (any unit) sampled at times `time` [s]:
detrended RMS and peak-to-peak, the spectral peak of the detrended signal, and
the mean of the *raw* signal over the same window (the trim value, kept so a
damping change can be checked for trim invariance).

Returns `(; rms, pkpk, mean, f_peak, f_res, fs, n, t_start, t_end)`.
"""
function ripple_metrics(time::AbstractVector, y::AbstractVector, rs::RippleSettings)
    i0 = searchsortedfirst(time, rs.t_start)
    i0 > length(time) && error("analysis window starts after the end of the log " *
                               "(t_start = $(rs.t_start) s, log ends at $(last(time)) s)")
    win = i0:length(time)
    dt = (time[end] - time[i0]) / (length(win) - 1)
    fs = 1 / dt
    # The moving average is taken over the whole record so that the start of the
    # analysis window still sees a full, centred window.
    resid = view(y, win) .- view(_centred_ma(y, round(Int, 0.5 * rs.detrend_window * fs)), win)
    f_peak, f_res = _peak_frequency(resid, fs, rs.f_min, rs.f_max)
    return (rms = sqrt(mean(abs2, resid)),
            pkpk = maximum(resid) - minimum(resid),
            mean = mean(view(y, win)),
            f_peak = f_peak,
            f_res = f_res,
            fs = fs,
            n = length(win),
            t_start = time[i0],
            t_end = time[end])
end

"""
    aoa_ripple(sl; rs=RippleSettings("ripple_settings.yaml")) -> NamedTuple

`ripple_metrics` applied to the AoA of the syslog table `sl` (as returned by
`KiteUtils.syslog(s.logger)` or `load_log(name).syslog`), converted to degrees. The
returned `mean`, `rms` and `pkpk` are therefore in degrees.
"""
aoa_ripple(sl; rs::RippleSettings = RippleSettings("ripple_settings.yaml")) =
    ripple_metrics(sl.time, rad2deg.(sl.AoA), rs)

"""
    format_ripple_report(r; sl=nothing, stats=nothing, t_loop=nothing, n_steps=nothing) -> String

Human-readable report of the AoA ripple metrics `r` (from `aoa_ripple`).

Optional extras, printed when given:
- `sl`      — syslog table, adds the parked trim state (mean elevation and tether
              force over the same window) for the trim-invariance check.
- `stats`   — `s.sam.integrator.stats`, the solver cost that actually explains the
              slowdown. Much less noisy than the wall clock.
- `t_loop`  — wall-clock duration of the simulation loop [s]; with `n_steps` it
              gives the mean time per `step!` and the real-time factor.
- `n_steps` — number of `step!` calls in that loop.
"""
function format_ripple_report(r; sl = nothing, stats = nothing,
                              t_loop = nothing, n_steps = nothing)
    io = IOBuffer()
    @printf(io, "AoA ripple, t = %.2f .. %.2f s, %d samples at %.1f Hz\n",
            r.t_start, r.t_end, r.n, r.fs)
    @printf(io, "  ripple RMS         : %8.4f °  (detrended)\n", r.rms)
    @printf(io, "  ripple pk-pk       : %8.4f °  (detrended)\n", r.pkpk)
    if isnan(r.f_peak)
        @printf(io, "  PSD peak           : %8s\n", "n/a")
    else
        @printf(io, "  PSD peak           : %8.2f Hz (resolution %.2f Hz, Nyquist %.1f Hz)\n",
                r.f_peak, r.f_res, r.fs / 2)
    end
    @printf(io, "  mean AoA (trim)    : %8.3f °\n", r.mean)
    if !isnothing(sl)
        i0 = searchsortedfirst(sl.time, r.t_start)
        win = i0:length(sl.time)
        @printf(io, "  mean elevation     : %8.3f °\n",
                mean(rad2deg.(view(sl.elevation, win))))
        @printf(io, "  mean tether force  : %8.1f N\n",
                mean(first.(view(sl.winch_force, win))))
    end
    if !isnothing(stats)
        @printf(io, "solver cost\n")
        @printf(io, "  accepted steps     : %8d  (rejected %d)\n",
                stats.naccept, stats.nreject)
        @printf(io, "  f evaluations      : %8d\n", stats.nf)
        @printf(io, "  Jacobians          : %8d  (W factorizations %d, lin. solves %d)\n",
                stats.njacs, stats.nw, stats.nsolve)
    end
    if !isnothing(t_loop) && !isnothing(n_steps) && n_steps > 0
        @printf(io, "wall clock\n")
        @printf(io, "  mean per step!     : %8.2f ms\n", 1000 * t_loop / n_steps)
        @printf(io, "  real-time factor   : %8.3f  (%.2f s simulated in %.2f s)\n",
                r.t_end / t_loop, r.t_end, t_loop)
    end
    return String(take!(io))
end
