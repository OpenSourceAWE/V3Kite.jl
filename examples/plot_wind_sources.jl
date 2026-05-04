# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Plot EKF vs lidar wind components for a UTC range, without
running any simulation. Loads the H5 flight data, slices by
UTC, computes both wind vectors at the kite's altitude per
timestep, and stacks E/N/U axes.
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using CairoMakie
GLMakie.activate!()

# =============================================================================
# Configuration
# =============================================================================

h5_path = joinpath(v3_data_path(),
    "flight_data", "ekf_awe_2025-10-09.h5")
start_utc = "15:36:29.0"
end_utc   = "15:40:29.0"

# =============================================================================
# Load + slice
# =============================================================================

full_data = load_flight_data(h5_path)
data, _ = limit_by_utc(full_data, start_utc, end_utc)
n = length(data.time)

# =============================================================================
# Compute ekf and lidar wind per timestep
# =============================================================================

ekf_wind = Array{Float64}(undef, 3, n)
lid_wind = Array{Float64}(undef, 3, n)
for i in eachindex(data.time)
    raw = NamedTuple{keys(data)}(
        Tuple(data[k][i] for k in keys(data)))
    alt = raw.ekf_kite_position_z
    ekf_wind[:, i] = compute_wind_vec(raw, alt;
        speed_source=:ekf, dir_source=:ekf)
    lid_wind[:, i] = compute_wind_vec(raw, alt;
        speed_source=:lidar, dir_source=:lidar)
end

# =============================================================================
# Plot
# =============================================================================

function build_fig()
    fig = Figure(size=(900, 800))
    labels = ("wind x (E) [m/s]",
              "wind y (N) [m/s]",
              "wind z (U) [m/s]")
    for (row, lbl) in enumerate(labels)
        ax = Axis(fig[row, 1]; ylabel=lbl,
            xlabelsize=18, ylabelsize=18,
            title=row == 1 ?
                "$start_utc – $end_utc" : "")
        lines!(ax, data.time, ekf_wind[row, :];
            color=:blue, label="ekf")
        lines!(ax, data.time, lid_wind[row, :];
            color=:red, label="lidar")
        row == 1 && axislegend(ax; position=:rt)
    end
    ax_v = Axis(fig[4, 1];
        xlabel="time [s]", ylabel="v_y kite [m/s]",
        xlabelsize=18, ylabelsize=18)
    lines!(ax_v, data.time, data.ekf_kite_velocity_y;
        color=:black)
    return fig
end

fig = build_fig()
display(fig)

CairoMakie.activate!()
pdf_fig = build_fig()
pdf_path = "wind_sources_$(replace(start_utc, ":" => ""))" *
    "_$(replace(end_utc, ":" => "")).pdf"
@info "Saving $pdf_path"
save(pdf_path, pdf_fig)
GLMakie.activate!()
