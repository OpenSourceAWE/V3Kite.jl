# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Photogrammetry Kite AoA

Computes the geometric angle between the chord reference direction
and the tether vector from photogrammetry data for each frame CSV.

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
ALL_FRAMES = TARGET_FRAMES

csvs = filter(
    f -> occursin("frame", f) && endswith(f, ".csv"),
    readdir(data_path; join=true))

measured_dp = Float64[]
measured_offset = Float64[]
frame_results = Dict{Int, Any}()

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

    # Intermediate z from camera to chord ref point
    cr = chord_ref_mid(le_3, te_3, le_4, te_4)
    z_temp = normalize(cr - cam_pos)

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
    return (; pts, groups, R_body, chord_w,
        le_center, cam_pos, cr,
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
        frame_results[frame_num] = result
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

"""Draw a 2D reference frame on a side-view axis."""
function draw_frame_axes!(ax, origin, x_dir, z_dir,
        scale; x_label=L"x", z_label=L"z")
    ox, oz = origin
    perp = 0.35 * scale
    # x-axis (red arrow)
    arrows2d!(ax, [ox], [oz],
        [x_dir[1] * scale], [x_dir[2] * scale];
        color=:red, shaftwidth=2, tipwidth=10)
    halo_text!(ax,
        ox + x_dir[1] * 0.7 * scale - x_dir[2] * perp,
        oz + x_dir[2] * 0.7 * scale + x_dir[1] * perp;
        text=x_label, fontsize=22, color=:black,
        align=(:center, :center))
    # z-axis (blue arrow)
    arrows2d!(ax, [ox], [oz],
        [z_dir[1] * scale], [z_dir[2] * scale];
        color=:blue, shaftwidth=2, tipwidth=10)
    halo_text!(ax,
        ox + z_dir[1] * 0.7 * scale - z_dir[2] * perp,
        oz + z_dir[2] * 0.7 * scale + z_dir[1] * perp;
        text=z_label, fontsize=22, color=:black,
        align=(:center, :center))
end

function add_bridle_overlay!(ax, kcu_2d, cr_2d,
        te_2d, le_2d, offset_deg)
    # LE->TE chord line (see-through)
    lines!(ax,
        [le_2d[1], te_2d[1]],
        [le_2d[2], te_2d[2]];
        color=(:black, 0.4), linewidth=2)
    # KCU->CR and CR->TE lines
    lines!(ax,
        [kcu_2d[1], cr_2d[1]],
        [kcu_2d[2], cr_2d[2]];
        color=:black, linewidth=2)
    lines!(ax,
        [cr_2d[1], te_2d[1]],
        [cr_2d[2], te_2d[2]];
        color=:black, linewidth=2)
    scatter!(ax,
        [kcu_2d[1], cr_2d[1], te_2d[1], le_2d[1]],
        [kcu_2d[2], cr_2d[2], te_2d[2], le_2d[2]];
        markersize=10, color=:black)
    halo_text!(ax, kcu_2d[1], kcu_2d[2];
        text="KCU", fontsize=12, color=:black,
        align=(:right, :top), offset=(-6, -4))
    halo_text!(ax, cr_2d[1], cr_2d[2];
        text="CR", fontsize=12, color=:black,
        offset=(4, 4))
    halo_text!(ax, te_2d[1], te_2d[2];
        text="TE", fontsize=12, color=:black,
        offset=(-20, 4))
    halo_text!(ax, le_2d[1], le_2d[2];
        text="LE", fontsize=12, color=:black,
        offset=(-10, -20))

    # Angle arc at CR
    v_kcu = normalize([kcu_2d[1] - cr_2d[1],
                       kcu_2d[2] - cr_2d[2]])
    v_te = normalize([te_2d[1] - cr_2d[1],
                      te_2d[2] - cr_2d[2]])
    th1 = atan(v_kcu[2], v_kcu[1])
    th2 = atan(v_te[2], v_te[1])
    dth = th2 - th1
    dth > pi && (dth -= 2pi)
    dth < -pi && (dth += 2pi)
    arm_kcu = norm([kcu_2d[1] - cr_2d[1],
                    kcu_2d[2] - cr_2d[2]])
    arm_te = norm([te_2d[1] - cr_2d[1],
                   te_2d[2] - cr_2d[2]])
    radius = 0.3 * min(arm_kcu, arm_te)
    ths = range(th1, th1 + dth, length=30)
    arc_x = cr_2d[1] .+ radius .* cos.(ths)
    arc_y = cr_2d[2] .+ radius .* sin.(ths)
    lines!(ax, arc_x, arc_y;
        color=:purple, linewidth=2)

    th_mid = th1 + dth / 2
    halo_text!(ax,
        cr_2d[1] + radius * 1.5 * cos(th_mid),
        cr_2d[2] + radius * 1.5 * sin(th_mid);
        text="$(round(abs(rad2deg(dth)); digits=1))\u00b0",
        fontsize=14, color=:purple)
end

# --- Side-view plots with chord ref geometry per frame ---
FIGURES_DIR = joinpath(@__DIR__, "..", "..",
    "T26-BART", "figures")
mkpath(FIGURES_DIR)

function build_frame_plot!(ax, fd)
    to_body(p) = fd.R_body' * (p - fd.cam_pos)
    to_side(b) = (b[1], b[3])

    body_pts = [Tuple(fd.R_body' *
        (collect(p) - fd.cam_pos))
        for p in fd.pts]

    kcu_2d = to_side(to_body(fd.cam_pos))
    cr_2d = to_side(to_body(collect(fd.cr)))
    te_2d = to_side(to_body(fd.te_mid))
    le_2d = to_side(to_body(fd.le_center))

    # Plot photogrammetry lines (side view: x, z)
    strut_inner = Set{Int}()
    for (gname, indices) in fd.groups
        is_le = gname == "LE"
        is_strut = startswith(gname, "strut")
        clr = is_le ? :orange : (:orange, 0.6)
        lw = is_le ? 5 : 3
        for i in 1:(length(indices)-1)
            c1 = body_pts[indices[i]]
            c2 = body_pts[indices[i+1]]
            lines!(ax,
                [c1[1], c2[1]], [c1[3], c2[3]];
                color=clr, linewidth=lw)
        end
        if is_strut
            for idx in indices[2:end]
                push!(strut_inner, idx)
            end
        end
    end
    # Opaque points (non-strut-inner)
    opaque = [i for i in eachindex(body_pts)
              if i ∉ strut_inner]
    scatter!(ax,
        [body_pts[i][1] for i in opaque],
        [body_pts[i][3] for i in opaque];
        color=:orange, markersize=8)
    # Transparent strut inner points
    if !isempty(strut_inner)
        inner = collect(strut_inner)
        scatter!(ax,
            [body_pts[i][1] for i in inner],
            [body_pts[i][3] for i in inner];
            color=(:orange, 0.4), markersize=8)
    end

    add_bridle_overlay!(ax, kcu_2d, cr_2d,
        te_2d, le_2d, fd.offset)

    frame_scale = 0.3 * fd.chord_len
    chord_body = fd.R_body' * fd.chord_w
    x_vec = normalize([chord_body[1], chord_body[3]])
    x_chord = (x_vec[1], x_vec[2])
    z_chord = (-x_chord[2], x_chord[1])

    draw_frame_axes!(ax, kcu_2d,
        (1.0, 0.0), (0.0, 1.0), frame_scale;
        x_label=L"x_\textrm{k}", z_label=L"z_\textrm{k}")
    draw_frame_axes!(ax, le_2d,
        x_chord, z_chord, frame_scale;
        x_label=L"x_\textrm{c}", z_label=L"z_\textrm{c}")
    autolimits!(ax)
end

# Combined figure with all frames side by side
sorted_frames = sort(collect(keys(frame_results)))
n_frames = length(sorted_frames)
function build_combined(sorted_frames, frame_results)
    n = length(sorted_frames)
    fig = Figure(size=(280 * n, 700))
    axes = Axis[]
    for (col, frame_num) in enumerate(sorted_frames)
        fd = frame_results[frame_num]
        show_ylabel = col == 1
        ax = Axis(fig[1, col];
            xlabel=L"x \; [m]",
            ylabel=show_ylabel ? L"z \; [m]" : "",
            xlabelsize=18, ylabelsize=18,
            title="Frame $frame_num",
            aspect=DataAspect(),
            yticklabelsvisible=show_ylabel,
            yticksvisible=show_ylabel)
        build_frame_plot!(ax, fd)
        push!(axes, ax)
    end
    if length(axes) > 1
        linkyaxes!(axes...)
    end
    colgap!(fig.layout, 10)
    return fig
end

combined = build_combined(sorted_frames,
    frame_results)
display(combined)

# Save PDF
CairoMakie.activate!()
combined_pdf = build_combined(sorted_frames,
    frame_results)
fname = "chord_ref_incidence_combined.pdf"
@info "Saving $fname"
save(fname, combined_pdf)
save(joinpath(FIGURES_DIR, fname), combined_pdf)
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


