using GLMakie

len_0pct = 0.2
len_100pct = 5.2
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

pcts = range(0, 100; length=201)
len_lin = linear_length.(pcts)
len_drum = drum_length.(pcts)
err = len_drum .- len_lin

fig = Figure(size=(900, 700))

ax1 = Axis(fig[1, 1];
    title="Depower percentage vs tape length",
    xlabel="depower [%]",
    ylabel="tape length [m]")
lines!(ax1, pcts, len_lin; label="linear (current)", linewidth=2)
lines!(ax1, pcts, len_drum;
    label="drum model (D=$(drum_diameter)m, t=$(tape_thickness)m)",
    linewidth=2, linestyle=:dash)
axislegend(ax1; position=:lt)

ax2 = Axis(fig[2, 1];
    title="Controller error (actual − commanded length, linear controller)",
    xlabel="commanded depower [%]",
    ylabel="Δ length [m]")
lines!(ax2, pcts, err; color=:red, linewidth=2)
hlines!(ax2, [0]; color=:gray, linestyle=:dot)

display(fig)

stack = (-drum_diameter +
    sqrt(drum_diameter^2 + 20 * tape_thickness / pi)) / 2
wraps = stack / tape_thickness
println("tape stack thickness at 0% : $(round(stack * 1000, digits=2)) mm")
println("outer drum radius at 0%   : ",
    "$(round((drum_diameter/2 + stack) * 1000, digits=2)) mm")
println("wraps between 0% and 100% : $(round(wraps, digits=2))")
err_30 = drum_length(30) - linear_length(30)
err_30_pct = err_30 / (stroke / 100)
println("deviation at 30%          : ",
    "$(round(err_30 * 1000, digits=2)) mm ",
    "($(round(err_30_pct, digits=2)) %-points)")

max_err_pct = maximum(abs.(err)) / (stroke / 100)
println("max deviation             : ",
    "$(round(maximum(abs.(err)) * 1000, digits=2)) mm ",
    "($(round(max_err_pct, digits=2)) %-points)")
