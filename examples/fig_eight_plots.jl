# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

"""
    fig_eight_plots.jl — Plotting for test_figure_eight.jl results.

All plotted time series come from the syslog (see PlanPlotting.md):
dedicated SysState fields (elevation, azimuth, heading, bearing = chi_set,
attractor = pursued attractor point, steering/set_steering = rel_steering
measured/setpoint, depower, AoA, winch_force) plus the generic slots
var_01 = d (cross-track error) [deg], var_02 = L/D_wing, var_03 = L/D_eff,
and (A1-15 only) var_04/var_05 = back-channel rel_steering setpoint/measured.
sl.sys_state encodes the phase (0 = park, 1 = dive, 2 = hold, 3 = fig8
engaged).

Include it after running test_figure_eight.jl in the same REPL session to
(re)plot from the in-memory `syslog`:

    include("fig_eight_plots.jl")

or run it standalone (fresh session or `julia --project fig_eight_plots.jl`),
which loads the "fig8_reference" log from the default wing's data path. That
name is deliberately NOT "fig8_run": simple_fig8.jl saves its own runs under
"fig8_run" and would otherwise overwrite the reference log on every run (as it
did once, on 2026-08-01).
"""

using GLMakie, MakieControlPlots, KiteUtils, Printf
using Statistics: mean
using V3Kite: figure_eight_path

# ==================== PLOTTING FUNCTION ==================== #

"""
    plot_fig_eight_results(wing_type, sl; kwargs...)

Plot figure-eight simulation results from the syslog `sl` (see the file
docstring for where each series lives). `wing_type` is `:a1_15` or `:ram`.
The keyword arguments only carry the reference-path geometry and the
statistics settle time, which are not part of the log.
"""
function plot_fig_eight_results(wing_type, sl;
    f8_a=50.0, f8_b=20.0, f8_c=0.0, f8_d=0.0, f8_el_center=40.0,
    settle_time=10.0
)
    if wing_type == :ram
        _plot_fig_eight_ram(sl; f8_a, f8_b, f8_c, f8_d, f8_el_center)
    else
        @error "Unknown wing_type: $wing_type (expected :a1_15 or :ram)"
        return
    end

    # Cross-track statistics (phase code 3 = figure-eight guidance engaged)
    i_engage = findfirst(==(3), sl.sys_state)
    if i_engage !== nothing
        fig8_start_time = sl.time[i_engage]
        @printf("Figure-eight guidance engaged at t=%.1f s\n", fig8_start_time)
        stats_start = fig8_start_time + settle_time
        settled = findall(>=(stats_start), sl.time)
        if !isempty(settled)
            d = Float64.(sl.var_01[settled])
            @printf("Cross-track error (t ≥ %.0f s): mean %.2f°, RMS %.2f°, max %.2f°\n",
                    stats_start, mean(d), sqrt(mean(d .^ 2)), maximum(d))
        else
            @warn "No samples after t ≥ $(stats_start)s — SIM_TIME may be too short"
        end
    else
        @warn "Figure-eight guidance never engaged — no cross-track statistics."
    end

    @printf("Minimum elevation reached: %.2f°\n", minimum(rad2deg.(sl.elevation)))
    return nothing
end

# ==================== RAM AIR KITE BRANCH PLOTS ==================== #

function _plot_fig_eight_ram(sl; f8_a, f8_b, f8_c, f8_d, f8_el_center)
    steering_labels = [L"rs_{\mathrm{set}}~[-]", L"rs~[-]"]
    chi_labels = [L"\chi_{\mathrm{set}}~[°]", L"\mathrm{heading}~[°]"]
    elevation_labels = [L"\mathrm{target}~[°]", L"\mathrm{elevation}~[°]"]
    azimuth_labels = [L"\mathrm{target}~[°]", L"\mathrm{azimuth}~[°]"]
    ld_labels = [L"L/D_{\mathrm{wing}}", L"L/D_{\mathrm{eff}}"]

    # Recompute the heading rate from the logged headings rather than using
    # sl.heading_rate: logs recorded before the 2026-07-17 interface.jl fix
    # carry a ~194x-too-small rate (integrator/log clock mix-up in
    # update_sys_state!), and the recomputation is exact for new logs too.
    heading_rate_deg = [0.0; rad2deg.(wrap2pi.(diff(Float64.(sl.heading)))) ./
                             diff(Float64.(sl.time))]

    steering_sets = [sl.set_steering, sl.steering]

    p = plotx(
        sl.time,
        [rad2deg.(getindex.(sl.attractor, 2)), rad2deg.(sl.elevation)],
        [rad2deg.(getindex.(sl.attractor, 1)), rad2deg.(sl.azimuth)],
        [rad2deg.(sl.bearing), rad2deg.(sl.heading)],
        heading_rate_deg,
        steering_sets,
        sl.var_01,
        (getindex.(sl.winch_force, 1), getindex.(sl.winch_force, 2),
         getindex.(sl.winch_force, 3));
        xlabel=L"\mathrm{Time}~[\mathrm{s}]",
        ylabels=[
            L"\mathrm{elevation}~[°]",
            L"\mathrm{azimuth}~[°]",
            L"\chi~[°]",
            L"\dot{\psi}~[°/\mathrm{s}]",
            L"\mathrm{rel\_steering}~[-]",
            L"d~[°]",
            L"F_{\mathrm{t}}~[\mathrm{N}]",
        ],
        labels=[elevation_labels, azimuth_labels, chi_labels, nothing, steering_labels,
                nothing,
                [L"F_{\mathrm{P}}", L"F_{\mathrm{SL}}", L"F_{\mathrm{SR}}"]],
        ysize=16,
        legendsize=16,
        fig="Ram air kite figure-of-eight",
    )
    display(p)

    # Dedicated aero window (PlanPlotting.md, third step)
    _plot_aero(sl, ld_labels, "Ram air kite aero")

    # Flown path vs. desired path
    _plot_flight_path(f8_a, f8_b, f8_c, f8_d, f8_el_center, sl)
end

# ==================== AERO PLOT (shared) ==================== #

# Dedicated window for AoA, depower and L/D (PlanPlotting.md, third step).
function _plot_aero(sl, ld_labels, fig_name)
    p = plotx(
        sl.time,
        rad2deg.(sl.AoA),
        sl.depower,
        (sl.var_02, sl.var_03);
        xlabel=L"\mathrm{Time}~[\mathrm{s}]",
        ylabels=[
            L"\mathrm{AoA}~[°]",
            L"\mathrm{depower}~[\mathrm{m}]",
            L"L/D~[-]",
        ],
        labels=[nothing, nothing, ld_labels],
        ysize=16,
        legendsize=16,
        fig=fig_name,
    )
    display(p)
end

# ==================== FLIGHT PATH PLOT (shared) ==================== #

function _plot_flight_path(f8_a, f8_b, f8_c, f8_d, f8_el_center, sl)
    az_ref, el_ref = figure_eight_path(f8_a, f8_b, f8_c, f8_d,
                                       0.0, f8_el_center, 0.0, 361)
    p = plotxy(
        [collect(az_ref), rad2deg.(sl.azimuth)],
        [collect(el_ref), rad2deg.(sl.elevation)];
        xlabel="azimuth [°]",
        ylabel="elevation [°]",
        title="Flight path",
        aspect=:equal,
        legend=["desired", "flown"],
        linestyle=[:dash, :solid],
        disp=true,
        fig="Flight path",
    )
    return p
end

# ==================== ENTRY POINT ==================== #
#
# Runs on every include. If a `syslog` from a test_figure_eight.jl run
# exists in this REPL session it is plotted directly (PlanPlotting.md,
# second step); otherwise the "fig8_reference" log is loaded from the wing's data
# path. The figure-eight geometry and settle-time globals set by
# test_figure_eight.jl are used when defined, so both paths plot with the
# parameters of the run.

wing_type = :ram

if !isdefined(Main, :syslog) || !(Main.syslog isa KiteUtils.SysLog)
    @info "Loading simulation results for ram-air wing..."
    set_data_path("data")
    syslog = load_log("fig8_reference")
end
sl = syslog.syslog

# Reference-path geometry and settle time, in order of precedence: the
# globals of a test_figure_eight.jl run in this session, then the parameters
# test_figure_eight.jl stored in the log's metadata (fig8_log_meta.jl), then
# the wing's hardcoded defaults — the last only for logs written before that
# metadata existed, in which case the desired path may not match the run.
if !isdefined(Main, :read_fig8_meta)
    include("fig8_log_meta.jl")
end
f8_defaults = read_fig8_meta("fig8_reference")
if f8_defaults === nothing
    @warn "No figure-eight parameters in the log metadata — plotting the " *
          "desired path with the built-in defaults."
    f8_defaults = wing_type == :a1_15 ?
        (f8_a=50.0, f8_b=20.0, f8_c=0.0, f8_d=0.0, f8_el_center=40.0, settle_time=10.0) :
        (f8_a=40.0, f8_b=15.0, f8_c=0.0, f8_d=-7.0, f8_el_center=36.0, settle_time=10.0)
end

plot_fig_eight_results(wing_type, sl;
    f8_a=isdefined(Main, :F8_A) ? F8_A : f8_defaults.f8_a,
    f8_b=isdefined(Main, :F8_B) ? F8_B : f8_defaults.f8_b,
    f8_c=isdefined(Main, :F8_C) ? F8_C : f8_defaults.f8_c,
    f8_d=isdefined(Main, :F8_D) ? F8_D : f8_defaults.f8_d,
    f8_el_center=isdefined(Main, :SET_ELEVATION_FIG_EIGHT) ?
        SET_ELEVATION_FIG_EIGHT : f8_defaults.f8_el_center,
    settle_time=isdefined(Main, :SETTLE_TIME) ? SETTLE_TIME : f8_defaults.settle_time)
