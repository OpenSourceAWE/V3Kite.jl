# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite: Batch Visualization for Circular Flight Runs

Loads the summary CSV from circular_batch_load.jl and
generates scatter plots of yaw rate vs us*v_app and
Cs vs us for each depower setting.

Usage:
    julia --project=examples examples/circular_batch_plot.jl
"""

using CSV, DataFrames
using Statistics
using Printf
using GLMakie
using Colors: distinguishable_colors

function resolve_batch_dir(batch_name)
    batch_dir = joinpath("processed_data", batch_name)
    isdir(batch_dir) && return batch_dir
    legacy_dir = joinpath("processed_data", "v3_kite", batch_name)
    if isdir(legacy_dir)
        @warn "Using legacy batch path" batch_dir=legacy_dir
        return legacy_dir
    end
    return batch_dir
end

function load_batch_csv(batch_name)
    batch_dir = resolve_batch_dir(batch_name)
    csv_path = joinpath(batch_dir,
        "circles_batch_analysis.csv")
    isfile(csv_path) || error("CSV not found: $csv_path")
    df = CSV.read(csv_path, DataFrame)
    return df, batch_dir
end

function plot_batch(df; batch_dir)
    df = filter(row ->
        isfinite(row.us) && isfinite(row.v_app) &&
        isfinite(row.yaw_rate) &&
        isfinite(row.yaw_rate_paper) &&
        isfinite(row.up), df)
    if "lt" in names(df)
        df = filter(row -> isfinite(row.lt), df)
    end

    ups = sort(unique(df.up))
    palette = distinguishable_colors(length(ups))

    fig = Figure(size=(800, 350))
    ax1 = Axis(fig[1, 1],
        xlabel="us*v_app [m/s]",
        ylabel="yaw_rate_paper [deg/s]")
    ax2 = Axis(fig[1, 2],
        xlabel="us [-]", ylabel="CS [-]")

    handles, labels = Any[], String[]
    plotted = false

    for (i, up_val) in enumerate(ups)
        rows = df[df.up .== up_val, :]
        x = rows.us .* rows.v_app
        m1 = isfinite.(x) .&
            isfinite.(rows.yaw_rate_paper)
        m2 = isfinite.(rows.us) .& isfinite.(rows.cs)
        color = palette[i]
        label = @sprintf("up=%.3f", up_val)

        if any(m1)
            h = scatter!(ax1,
                x[m1], rows.yaw_rate_paper[m1];
                color, markersize=8)
            push!(handles, h)
            push!(labels, label)
            plotted = true
        end
        if any(m2)
            scatter!(ax2,
                rows.us[m2], rows.cs[m2];
                color, markersize=8)
        end
    end

    # Dynamic time-series data
    usva_cols = filter(
        x -> startswith(string(x), "usva_"), names(df))
    yr_cols = filter(
        x -> occursin(r"^yaw_rate_\d+$", string(x)),
        names(df))

    if !isempty(usva_cols) && !isempty(yr_cols)
        for (i, up_val) in enumerate(ups)
            rows = df[df.up .== up_val, :]
            color = palette[i]
            dx, dy = Float64[], Float64[]
            for row in eachrow(rows)
                for (uc, yc) in zip(usva_cols, yr_cols)
                    xv, yv = row[uc], row[yc]
                    if isfinite(xv) && isfinite(yv)
                        push!(dx, xv); push!(dy, yv)
                    end
                end
            end
            if !isempty(dx)
                h = scatter!(ax1, dx, dy;
                    color, markersize=4,
                    marker=:circle, alpha=0.5)
                if i == 1
                    push!(handles, h)
                    push!(labels, "dynamic")
                end
                plotted = true
            end
        end
    end

    if plotted && !isempty(handles)
        axislegend(ax1, handles, labels; position=:rb)
    end

    lt_tag = ""
    if "lt" in names(df)
        lt_vals = unique(df.lt)
        if length(lt_vals) == 1
            lt_tag = "_lt_$(Int(round(lt_vals[1])))"
        end
    end
    out_path = joinpath(batch_dir,
        "circles_batch_plot$(lt_tag).png")
    save(out_path, fig)
    @info "Saved plot" path=out_path
end

function main()
    batch_name = isempty(ARGS) ? "" : strip(ARGS[1])
    # batch_name = "circular_2025_batch_2026_01_11_11_29_19"
    if isempty(batch_name)
        print("Enter batch folder name: ")
        batch_name = strip(readline())
    end
    isempty(batch_name) && error("Batch name required.")
    df, batch_dir = load_batch_csv(batch_name)
    plot_batch(df; batch_dir)
end

main()
