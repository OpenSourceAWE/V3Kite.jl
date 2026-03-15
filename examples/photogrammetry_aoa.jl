# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Photogrammetry Bridle AoA

Computes the geometric angle between the mid-chord direction and
the LE-bridle vector from photogrammetry data for each frame CSV.

Usage:
    julia --project=examples examples/photogrammetry_aoa.jl
"""

using V3Kite, LinearAlgebra, GLMakie, CairoMakie
GLMakie.activate!()

# Load flight data for depower lookup
data_path = v3_data_path()
h5_path = joinpath(data_path,
    "flight_data", "ekf_awe_2025-10-09.h5")
full_data = load_flight_data(h5_path)
flight_data, _ = limit_by_utc(full_data, "15:30:00")

# Two calibration frames with known measurements
TARGET_FRAMES = [7182, 17611]
STEERING_FRAMES = [7182, 7362]
ALL_FRAMES = union(TARGET_FRAMES, STEERING_FRAMES)

csvs = filter(
    f -> occursin("frame", f) && endswith(f, ".csv"),
    readdir(data_path; join=true))

measured_dp = Float64[]
measured_offset = Float64[]
steering_dp = Float64[]
steering_offset = Float64[]
steering_vals = Float64[]
steering_chords = Float64[]
plot_data = nothing

function process_frame_csv(csv, flight_data)
    pts, groups = load_extra_points(csv)

    s3_idx = nothing
    s4_idx = nothing
    for (name, idx) in groups
        name == "strut3" && (s3_idx = idx)
        name == "strut4" && (s4_idx = idx)
    end
    if isnothing(s3_idx) || isnothing(s4_idx)
        @warn "Skipping $(basename(csv)): " *
            "missing strut group"
        return nothing
    end

    # TE = first index, LE = last index in each strut
    te_3 = collect(pts[s3_idx[1]])
    le_3 = collect(pts[s3_idx[end]])
    te_4 = collect(pts[s4_idx[1]])
    le_4 = collect(pts[s4_idx[end]])

    # Camera origin (raw camera frame)
    cam_pos = [0.0, 0.0, 0.0]

    # Build body frame from CSV geometry
    # Spanwise: strut4 LE - strut3 LE = +y
    y_body = normalize(le_4 - le_3)

    # Intermediate z from camera to quarter chord
    qc = quarter_chord_mid(le_3, te_3, le_4, te_4)
    z_temp = normalize(qc - cam_pos)

    # Orthogonalize
    x_body = normalize(cross(y_body, z_temp))
    z_body = normalize(cross(x_body, y_body))
    R_body = hcat(x_body, y_body, z_body)

    # Chord direction (LE->TE, averaged over both struts)
    chord_w = normalize(
        normalize(te_3 - le_3) +
        normalize(te_4 - le_4))

    # Incidence: angle of chord in body xz plane
    incidence = rad2deg(atan(
        dot(chord_w, z_body),
        dot(chord_w, x_body)))
    offset = -incidence

    # Smallest angle between chord and bridle z axis
    check = rad2deg(acos(clamp(
        abs(dot(chord_w, z_body)), 0.0, 1.0)))

    # Look up depower from flight data at this frame
    m = match(r"frame_(\d+)", basename(csv))
    frame_num = parse(Int, m[1])
    _, closest = findmin(
        abs.(flight_data.video_frame .- frame_num))
    depower = flight_data.kcu_actual_depower[closest]
    steering = flight_data.kcu_actual_steering[closest]

    println("$(basename(csv)): offset = " *
            "$(round(offset; digits=2))deg, " *
            "chord-bridle = " *
            "$(round(check; digits=2))deg, " *
            "depower = $(round(depower; digits=1))%, " *
            "steering = $(round(steering; digits=1))%")

    le_center = (le_3 + le_4) / 2
    chord_len = (norm(te_3 - le_3) +
        norm(te_4 - le_4)) / 2
    return (; pts, groups, R_body,
        le_center, cam_pos, qc,
        le_3, te_3, le_4, te_4,
        te_mid=(te_3 + te_4) / 2,
        offset, depower, steering, chord_len)
end

for csv in csvs
    m = match(r"frame_(\d+)", basename(csv))
    isnothing(m) && continue
    frame_num = parse(Int, m[1])
    frame_num in ALL_FRAMES || continue

    result = process_frame_csv(csv, flight_data)
    isnothing(result) && continue

    if frame_num in TARGET_FRAMES
        push!(measured_dp, result.depower)
        push!(measured_offset, result.offset)
    end
    if frame_num in STEERING_FRAMES
        push!(steering_dp, result.depower)
        push!(steering_offset, result.offset)
        push!(steering_vals, result.steering)
        push!(steering_chords, result.chord_len)
    end

    if frame_num == 7182
        global plot_data = result
    end
end

"""Draw text with a white halo (CairoMakie-safe)."""
function halo_text!(ax, x, y; text, fontsize=12,
        color=:black, halo_width=3, kw...)
    text!(ax, x, y; text, fontsize,
        color=:white, strokecolor=:white,
        strokewidth=halo_width, kw...)
    text!(ax, x, y; text, fontsize, color, kw...)
end

function add_bridle_overlay!(ax, kcu_2d, qc_2d,
        te_2d, offset_deg)
    # KCU->QC and QC->TE lines
    lines!(ax,
        [kcu_2d[1], qc_2d[1]],
        [kcu_2d[2], qc_2d[2]];
        color=:black, linewidth=2)
    lines!(ax,
        [qc_2d[1], te_2d[1]],
        [qc_2d[2], te_2d[2]];
        color=:black, linewidth=2)
    scatter!(ax,
        [kcu_2d[1], qc_2d[1], te_2d[1]],
        [kcu_2d[2], qc_2d[2], te_2d[2]];
        markersize=10, color=:black)
    halo_text!(ax, kcu_2d[1], kcu_2d[2];
        text="KCU", fontsize=12, color=:black,
        align=(:right, :top), offset=(-6, -4))
    halo_text!(ax, qc_2d[1], qc_2d[2];
        text="QC", fontsize=12, color=:black,
        align=(:right, :bottom), offset=(-8, 8))
    halo_text!(ax, te_2d[1], te_2d[2];
        text="TE", fontsize=12, color=:black,
        align=(:right, :bottom), offset=(-8, 8))

    # Angle arc at QC
    v_kcu = normalize([kcu_2d[1] - qc_2d[1],
                       kcu_2d[2] - qc_2d[2]])
    v_te = normalize([te_2d[1] - qc_2d[1],
                      te_2d[2] - qc_2d[2]])
    th1 = atan(v_kcu[2], v_kcu[1])
    th2 = atan(v_te[2], v_te[1])
    dth = th2 - th1
    dth > pi && (dth -= 2pi)
    dth < -pi && (dth += 2pi)
    arm_kcu = norm([kcu_2d[1] - qc_2d[1],
                    kcu_2d[2] - qc_2d[2]])
    arm_te = norm([te_2d[1] - qc_2d[1],
                   te_2d[2] - qc_2d[2]])
    radius = 0.3 * min(arm_kcu, arm_te)
    ths = range(th1, th1 + dth, length=30)
    arc_x = qc_2d[1] .+ radius .* cos.(ths)
    arc_y = qc_2d[2] .+ radius .* sin.(ths)
    lines!(ax, arc_x, arc_y;
        color=:purple, linewidth=2)

    th_mid = th1 + dth / 2
    halo_text!(ax,
        qc_2d[1] + radius * 1.5 * cos(th_mid),
        qc_2d[2] + radius * 1.5 * sin(th_mid);
        text="$(round(abs(rad2deg(dth)); digits=1))\u00b0",
        fontsize=14, color=:purple)
end

# --- Single side-view plot with bridle geometry ---
to_body(p) = plot_data.R_body' *
    (p - plot_data.le_center)
to_side(b) = (b[1], b[3])

# Transform all points to body frame
body_pts = [Tuple(plot_data.R_body' *
    (collect(p) - plot_data.le_center))
    for p in plot_data.pts]

# Compute 2D coords for bridle overlay
kcu_2d = to_side(to_body(plot_data.cam_pos))
qc_2d = to_side(to_body(collect(plot_data.qc)))
te_2d = to_side(to_body(plot_data.te_mid))

fig = plot_photogrammetry(body_pts,
    plot_data.groups; dir=:side)
ax = content(fig[1, 1])
add_bridle_overlay!(ax, kcu_2d, qc_2d, te_2d,
    plot_data.offset)
autolimits!(ax)
display(fig)

# Save PDF with CairoMakie
FIGURES_DIR = joinpath(@__DIR__, "..", "..",
    "Torque2026", "figures")
CairoMakie.activate!()
fig_pdf = plot_photogrammetry(body_pts,
    plot_data.groups; dir=:side)
ax_pdf = content(fig_pdf[1, 1])
add_bridle_overlay!(ax_pdf, kcu_2d, qc_2d, te_2d,
    plot_data.offset)
autolimits!(ax_pdf)
mkpath(FIGURES_DIR)
save(joinpath(FIGURES_DIR,
    "bridle_incidence_side.pdf"), fig_pdf)
save("bridle_incidence_side.pdf", fig_pdf)
GLMakie.activate!()

# Linear depower-to-offset model: offset = a * u + b
# Two points → direct solve
o1, o2 = measured_offset
u1, u2 = measured_dp
a = (o2 - o1) / (u2 - u1)
b = o1 - a * u1

println("\nLinear model: offset = a * u + b")
println("  a = $(round(a; digits=4)) deg/%")
println("  b = $(round(b; digits=2)) deg")

# Steering effect: offset difference between two frames
Δoffset = steering_offset[2] - steering_offset[1]
Δsteering = abs(steering_vals[2] - steering_vals[1])
dp_steer_deg = -Δoffset / Δsteering

println("\nSteering effect:")
println("  Δoffset = $(round(Δoffset; digits=2)) deg")
println("  Δsteering = $(round(Δsteering; digits=1)) %")
println("  dp_steer_offset = " *
    "$(round(dp_steer_deg; digits=4)) deg/%")

# Convert to %/% via triangle geometry
chord = (steering_chords[1] + steering_chords[2]) / 2
z_height = chord * sin(deg2rad(Δoffset))
z_per_pct = z_height / Δsteering
dp_steer_pct = z_per_pct / (V3_DEPOWER_GAIN / 100)

println("  chord = $(round(chord; digits=3)) m")
println("  z_per_pct = $(round(z_per_pct; digits=5)) m/%")
println("  dp_steer_pct = " *
    "$(round(dp_steer_pct; digits=4)) %/%")

