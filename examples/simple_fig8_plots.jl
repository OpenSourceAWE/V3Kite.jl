# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

"""
Plotting for simple_fig8.jl results.

Loads the "fig8_run" log saved by simple_fig8.jl. Produces two figures:

1. the flown pattern in the (azimuth, elevation) plane, with the reference
   lemniscate and the attractor track overlaid — the plot that shows at a glance
   whether the pattern is being flown or only approximated;
2. a stacked time-series figure: cross-track error, elevation (with the pattern
   centre), heading and course vs the commanded course, the course-tracking
   error, steering command vs the KCU's actual tape-lagged value, and tether
   force.

The guidance commands a COURSE (direction of travel) while the inner loop
regulates HEADING (where the nose points), so the angle panel shows all three:
`chi` is what the kite actually flew, `chi_set` the command, and the
`chi - psi` gap is the kite's drift angle (~13° on the V3, see
`src/fig8_controller.jl`). Those three are plotted UNWRAPPED (a lap of the
pattern crosses ±180°, and the raw traces jump there); the error panel below
carries the same errors wrapped to ±180°.

Judge path following by `chi - chi_set`. The third curve in the error panel is
the error the PID actually regulated (`var_06`): the loop feeds back heading at
low apparent wind speed and course at high, blending in between
(`V_APP_HEADING`/`V_APP_COURSE` in simple_fig8.jl, weight logged in `var_08`),
so that curve rides on `psi - chi_set` at low speed and on `chi - chi_set` at
high. On logs written before the blend existed it coincides with
`psi - chi_set`.

Run from the REPL after (or instead of, if "fig8_run" already exists) running
simple_fig8.jl:

    include("simple_fig8_plots.jl")
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using GLMakie
using MakieControlPlots
using LaTeXStrings
using V3Kite

# Pattern geometry, used only to redraw the reference path. simple_fig8.jl
# `include`s this file at the end of a run, so whatever it defined wins and the
# overlay cannot drift out of sync with the run; these values are the fallback
# for running this script standalone against an existing "fig8_run" log.
@isdefined(F8_A) || (F8_A = 50.0)
@isdefined(F8_B) || (F8_B = 20.0)
@isdefined(F8_C) || (F8_C = 0.0)
@isdefined(F8_D) || (F8_D = 0.0)

@info "Loading simulation results..."
set_data_path(v3_data_path())
syslog = load_log("fig8_run")
sl = syslog.syslog

created_at = log_created_at("fig8_run")
fig_name = "V3 Kite Figure-of-Eight"
if !isnothing(created_at)
    fig_name *= " – " * replace(first(split(created_at, '.')), "T" => "_")
end

# Skip the t=0 initial log entry (the guidance slots are only filled from the
# first `step!` call onward).
rng = 2:length(sl.time)

az_deg = rad2deg.(sl.azimuth[rng])
el_deg = rad2deg.(sl.elevation[rng])

# Reference path at the FINAL pattern centre (var_04), so the overlay matches
# the run even when the centre was walked (WALK_RATE > 0).
el_c_end = Float64(sl.var_04[end])
ref_az, ref_el = figure_eight_path(F8_A, F8_B, F8_C, F8_D, 0.0, el_c_end, 0.0, 361)

# --- angles for the psi/chi panel ---------------------------------------- #
# Plotted UNWRAPPED: the raw ±180° traces jump at the branch cut, the three
# curves cross it at different times, and the pattern makes them cross it every
# lap. The course is unwrapped by integrating its wrapped increments; heading
# and the command are then placed on the branch nearest the unwrapped course at
# each sample, so all three stay continuous *and* their vertical distances are
# still exactly the wrapped errors plotted in the panel below.
unwrap_angle(a) = first(a) .+ cumsum(vcat(0.0, wrap_to_pi.(diff(a))))
onto(ref_u, ref_w, a) = ref_u .+ wrap_to_pi.(a .- ref_w)

psi    = Float64.(sl.heading[rng])
# +π puts the logged course into the same convention as heading and the
# guidance (0 = towards zenith); see the note in simple_fig8.jl. Without it the
# course trace sits 180° away from everything else in the panel.
chi    = wrap_to_pi.(Float64.(sl.course[rng]) .+ pi)
chiset = Float64.(sl.bearing[rng])   # chi_cmd, the course actually tracked
chi_u  = unwrap_angle(chi)

# Tracking errors, both wrapped to ±180°. chi - chi_set is the path-following
# error the guidance cares about; psi - chi_set is what the PID regulates, and
# the offset between them is the kite's drift angle (~13° on the V3).
err_course  = rad2deg.(wrap_to_pi.(chi .- chiset))
err_heading = rad2deg.(wrap_to_pi.(psi .- chiset))

@info "Plotting the pattern..."
p1 = plotxy(
    [az_deg, ref_az, Float64.(sl.var_02[rng])],
    [el_deg, ref_el, Float64.(sl.var_03[rng])];
    xlabel = L"\mathrm{azimuth}~[°]",
    ylabel = L"\mathrm{elevation}~[°]",
    legend = [L"\mathrm{flown}", L"\mathrm{reference}", L"\mathrm{attractor}"],
    fig = fig_name * " – pattern",
)
display(p1)
sleep(0.1)

@info "Plotting the time series..."
p2 = plotx(
    sl.time[rng],
    sl.var_01[rng],
    [el_deg, Float64.(sl.var_04[rng])],
    [rad2deg.(onto(chi_u, chi, psi)), rad2deg.(chi_u),
     rad2deg.(onto(chi_u, chi, chiset))],
    [err_course, err_heading, Float64.(sl.var_06[rng])],
    (100.0 .* sl.steering[rng], 100.0 .* sl.set_steering[rng]),
    getindex.(sl.winch_force[rng], 1);
    xlabel = L"\mathrm{time}~[\mathrm{s}]",
    ysize = 18,
    legendsize = 16,
    ylabels = [
        L"d~[°]",
        L"\mathrm{elevation}~[°]",
        L"\psi,~\chi~[°]",
        L"\Delta\chi~[°]",
        L"u_{\mathrm{s}}~[\%]",
        L"F_{\mathrm{tether}}~[\mathrm{N}]",
    ],
    labels = [
        nothing,
        [L"\mathrm{kite}", L"\mathrm{pattern~centre}"],
        [L"\psi", L"\chi", L"\chi_{\mathrm{set}}"],
        [L"\chi - \chi_{\mathrm{set}}", L"\psi - \chi_{\mathrm{set}}",
         L"\mathrm{regulated}"],
        [L"u_{\mathrm{s}}", L"u_{\mathrm{s,set}}"],
        nothing,
    ],
    fig = fig_name * " – time series",
)
display(p2)
sleep(0.1)

nothing
