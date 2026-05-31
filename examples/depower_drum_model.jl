using V3Kite
using GLMakie

len_0pct = V3_DEPOWER_L0_BASE
len_100pct = udp_to_depower_tape_length_m(1.0)
stroke = len_100pct - len_0pct

drum_diameter = 0.05
tape_thickness = 0.0015

function linear_length(pct)
    return len_0pct + stroke * pct / 100
end

function drum_length(pct; D=drum_diameter, t=tape_thickness, L=stroke)
    stack = (-D + sqrt(D^2 + 20 * t / pi)) / 2
    r_outer = D / 2 + stack
    theta_max = 2pi * stack / t
    theta = theta_max * pct / 100
    return len_0pct + r_outer * theta - t * theta^2 / (4pi)
end

function ld_2019(u_dp)
    return -0.00011847 * (100 * u_dp)^2 + 0.0605081 * (100 * u_dp) + 0.22
end

function ld_2025(u_dp)
    return udp_to_depower_tape_length_m(u_dp)
end

pcts = range(0, 100; length=201)
u_dp = pcts ./ 100
len_lin = linear_length.(pcts)
len_drum = drum_length.(pcts)
len_2019 = ld_2019.(u_dp)
len_2025 = ld_2025.(u_dp)
err = len_drum .- len_lin
err_2019 = len_2019 .- len_lin
err_2025 = len_2025 .- len_lin

fig = Figure(size=(900, 700))

ax1 = Axis(fig[1, 1];
    title="Depower percentage vs tape length",
    xlabel="depower [%]",
    ylabel="tape length [m]")
lines!(ax1, pcts, len_lin; label="linear (current)", linewidth=2,
    linestyle=:dash)
lines!(ax1, pcts, len_drum;
    label="drum model (D=$(drum_diameter)m, t=$(tape_thickness)m)",
    linewidth=2)
lines!(ax1, pcts, len_2019; label="ld_2019", linewidth=2, color=:blue)
lines!(ax1, pcts, len_2025; label="ld_2025", linewidth=2, color=:green)
axislegend(ax1; position=:lt)

ax2 = Axis(fig[2, 1];
    title="Controller error (actual − commanded length, linear controller)",
    xlabel="commanded depower [%]",
    ylabel="Δ length [m]")
lines!(ax2, pcts, err; color=:red, linewidth=2, label="drum model")
lines!(ax2, pcts, err_2019; color=:blue, linewidth=2, label="ld_2019")
lines!(ax2, pcts, err_2025; color=:green, linewidth=2, label="ld_2025")
hlines!(ax2, [0]; color=:gray, linestyle=:dot)
axislegend(ax2; position=:lt)

display(fig)

stack = (-drum_diameter +
    sqrt(drum_diameter^2 + 20 * tape_thickness / pi)) / 2
wraps = stack / tape_thickness
println("tape stack thickness at 0% : $(round(stack * 1000, digits=2)) mm")
println("outer drum radius at 0%   : ",
    "$(round((drum_diameter/2 + stack) * 1000, digits=2)) mm")
println("wraps between 0% and 100% : $(round(wraps, digits=2))")
sim_len_30 = linear_length(30)
real_len_30 = drum_length(30)
err_30 = real_len_30 - sim_len_30
err_30_pct = err_30 / (stroke / 100)
println()
println("scenario at 30% command (winch angle = 30% of theta_max):")
println("  sim tape length (linear)  : $(round(sim_len_30, digits=4)) m")
println("  real tape length (drum)   : $(round(real_len_30, digits=4)) m")
println("  error (real - sim)        : ",
    "$(round(err_30 * 1000, digits=2)) mm ",
    "($(round(err_30_pct, digits=2)) %-points)")

max_err_pct = maximum(abs.(err)) / (stroke / 100)
println("max deviation             : ",
    "$(round(maximum(abs.(err)) * 1000, digits=2)) mm ",
    "($(round(max_err_pct, digits=2)) %-points)")
