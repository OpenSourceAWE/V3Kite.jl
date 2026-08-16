# Copyright (c) 2026 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Kite Turn Diagnostics

Flies the same sinusoidal heading maneuver as `v3kite.jl` and instruments the
chain a turn has to travel: steering input -> differential twist -> yaw moment
-> body rate -> heading rate. A link that is dead or an order of magnitude down
localises the fault to one stage instead of "it does not turn".

Reports the yaw moment twice: VSM's own ([`aero_moment_z`](@ref)) and the one the
structure actually receives (`wing.aero_moment_b`). Under `AeroPressure` those
are different quantities, and a gap between them is load that the aero produced
but the scatter onto points never delivered.

Also cross-checks the rigid-body kinematics: the wing's own `ω_b` against the
rate implied by differencing its frame, and `wing.heading` (the field the
controller reads) against the heading recomputed from the live wing frame. Those
agreeing rules the kinematics out; them disagreeing means the loop is closing on
a stale signal.

Prints a table plus an end-of-run summary; writes no log and opens no window, so
it runs headless. `bin/run_julia --beam` (or `--psm`) picks the wing, and
`V3KITE_AERO_MODE=continuous` overrides the project's aero mode, which is how the
same maneuver is compared across couplings.

The loop closes on the heading recomputed from the live wing frame, not on
`wing.heading`: the `KernelBackend` never writes that field, so a controller
reading it sees a constant and saturates instead of flying the maneuver.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using SymbolicAWEModels
using LinearAlgebra
using Printf
using Statistics

PROJECT = select_project(
    ["Timoshenko-beam wing" => "system_v3kite_beam.yaml",
     "particle lattice" => "system_v3kite_psm.yaml"];
    prompt = "Which wing model should fly?")

MAX_HEADING = 40.0    # setpoint amplitude [deg]
PERIOD = 30.0         # setpoint period [s]
REPORT_EVERY = 5      # steps between table rows

"""
    frame_rate(R_prev, R_now, dt) -> KVec3

Body-frame angular velocity [rad/s] implied by the wing frame moving from
`R_prev` to `R_now` over `dt`, as the axis-angle of `R_prev' * R_now`. Compared
against the model's own `ω_b` this catches a frame that moves without the state
following, or the reverse.
"""
function frame_rate(R_prev, R_now, dt)
    rel = R_prev' * R_now
    angle = acos(clamp((tr(rel) - 1) / 2, -1, 1))
    axis = [rel[3, 2] - rel[2, 3], rel[1, 3] - rel[3, 1], rel[2, 1] - rel[1, 2]]
    len = norm(axis)
    len < 1e-12 && return zeros(3)
    return (angle / dt) .* (axis ./ len)
end

"""
    yaw_axis(wing) -> KVec3

Body-frame direction the kite yaws about: the tether line, i.e. the unit vector
from the ground anchor to the wing origin, expressed in the wing frame. The turn
that steering has to produce is the moment about this axis, which is not the
body `z` axis once the kite is off zenith.
"""
function yaw_axis(wing)
    radial = Vector(wing.pos_w)
    norm(radial) < 1e-9 && return [0.0, 0.0, 1.0]
    return wing.R_b_to_w' * normalize(radial)
end

"""
    live_heading(wing) -> Float64

Heading [rad] recomputed from the wing's current frame and position, the signal
`wing.heading` would carry if every backend wrote it.
"""
live_heading(wing) =
    V3Kite.calc_heading_from_rotation(wing.R_b_to_w, wing.pos_w)

set_data_path(v3_data_path())
kite = load_kite(PROJECT)
heading = load_heading(PROJECT)
set = Settings(PROJECT)

aero_override = get(ENV, "V3KITE_AERO_MODE", "")
isempty(aero_override) ||
    (kite.aero_mode = V3Kite.parse_aero_mode(aero_override))

gain_override = get(ENV, "V3KITE_HEADING_K", "")
isempty(gain_override) || (heading.K = parse(Float64, gain_override))
time_override = get(ENV, "V3KITE_SIM_TIME", "")

@info "V3 Kite Turn Diagnostics" PROJECT aero_mode=kite.aero_mode

sam, sys = build_v3_model(PROJECT; kite)
wing = sys.wings[1]

sim_time = isempty(time_override) ? set.sim_time : parse(Float64, time_override)
n_steps = Int(round(set.sample_freq * sim_time))
dt = sim_time / n_steps
logger, sys_state = create_logger(sam, n_steps)

nominal_steering = V3Kite.get_steering(sys, kite.geom)
max_heading_rad = deg2rad(MAX_HEADING)
angular_freq = 2pi / PERIOD
pid = heading_pid(heading, dt)

time_s = Float64[]
steer_cmd = Float64[]
twist_diff = Float64[]
moment_yaw = Float64[]
moment_vsm = Float64[]
moment_norm = Float64[]
rate_yaw = Float64[]
heading_rate = Float64[]
heading_field = Float64[]
heading_live = Float64[]
kinematics_gap = Float64[]

heading_prev = live_heading(wing)
frame_prev = copy(wing.R_b_to_w)

@printf("%5s %7s %7s %8s %8s %7s %8s %9s %9s %9s %9s %8s %8s\n",
        "step", "t[s]", "cmd[°]", "hfld[°]", "hlive[°]", "g_k", "steer[m]",
        "dtwist[°]", "Mapp[Nm]", "Mvsm[Nm]", "|M|[Nm]", "ωyaw[°/s]",
        "ḣead[°/s]")

failed_at = 0
for step in 1:n_steps
    t = step * dt

    target_rad = max_heading_rad * sin(angular_freq * t)
    measured = live_heading(wing)
    schedule_heading_pid!(pid, heading, t, sys_state.v_app, target_rad, measured)
    steer_ctrl = pid(target_rad, measured, 0.0)
    sys_state.bearing = target_rad
    set_steering!(sys, nominal_steering + steer_ctrl, kite.geom)

    if !sim_step!(sam; dt, vsm_interval = kite.vsm_interval)
        @error "Simulation failed" step
        global failed_at = step
        break
    end
    log_state!(logger, sys_state, sam, t)

    axis = yaw_axis(wing)
    moment = Vector(wing.aero_moment_b)
    omega = Vector(wing.ω_b)
    omega_frame = frame_rate(frame_prev, wing.R_b_to_w, dt)
    live = live_heading(wing)
    head_rate = rad2deg(wrap_to_pi(live - heading_prev) / dt)
    gap = norm(omega - omega_frame) / max(norm(omega_frame), 1e-6) * 100
    yaw_rate = rad2deg(dot(omega_frame, axis))

    push!(time_s, t)
    push!(steer_cmd, steer_ctrl)
    push!(twist_diff, rad2deg(V3Kite.differential_twist(sys)))
    push!(moment_yaw, dot(moment, axis))
    push!(moment_vsm, aero_moment_z(sys))
    push!(moment_norm, norm(moment))
    push!(rate_yaw, yaw_rate)
    push!(heading_rate, head_rate)
    push!(heading_field, rad2deg(wing.heading))
    push!(heading_live, rad2deg(live))
    push!(kinematics_gap, gap)

    if step % REPORT_EVERY == 0 || step <= 3
        @printf("%5d %7.2f %7.2f %8.2f %8.2f %7.3f %8.4f %9.3f %9.1f %9.1f %9.1f %8.3f %8.3f\n",
                step, t, rad2deg(target_rad), heading_field[end],
                heading_live[end], pid.K, steer_ctrl, twist_diff[end],
                moment_yaw[end], moment_vsm[end], moment_norm[end],
                rate_yaw[end], head_rate)
    end

    global heading_prev = live
    global frame_prev = copy(wing.R_b_to_w)
end

"""
    span(label, values, unit)

One summary line: the range a diagnostic covered and its mean magnitude. A stage
whose range is ~0 is the one that broke the chain.
"""
function span(label, values, unit)
    isempty(values) && return
    @printf("  %-26s %9.4f .. %-9.4f  mean|·| %9.4f %s\n",
            label, minimum(values), maximum(values),
            mean(abs.(values)), unit)
end

println("\n===== turn chain =====")
span("steering command", steer_cmd, "m")
span("differential twist", twist_diff, "deg")
span("applied moment about yaw", moment_yaw, "Nm")
span("VSM yaw moment", moment_vsm, "Nm")
span("applied moment magnitude", moment_norm, "Nm")
span("body rate about yaw (frame)", rate_yaw, "deg/s")
span("heading rate", heading_rate, "deg/s")

println("\n===== kinematics cross-check =====")
span("wing.heading (controller)", heading_field, "deg")
span("heading from live frame", heading_live, "deg")
span("|ω_b - ω_frame| / |ω_frame|", kinematics_gap, "%")
if !isempty(heading_field)
    field_span = maximum(heading_field) - minimum(heading_field)
    live_span = maximum(heading_live) - minimum(heading_live)
    field_span < 0.01 < live_span && println(
        "  wing.heading never moves while the frame's heading spans ",
        "$(round(live_span, digits=1))°: the controller is closing on a field ",
        "this backend does not write.")
end
if !isempty(kinematics_gap)
    worst = maximum(kinematics_gap)
    println(worst < 5.0 ?
        "  ω_b tracks the wing frame; kinematics are consistent." :
        "  ω_b disagrees with the frame by $(round(worst, digits=1))%; " *
        "suspect the pose fit or the rate state.")
end

if length(twist_diff) > 2 && std(twist_diff) > 1e-9
    println("\n===== stage gains (least squares, no lag correction) =====")
    gain(x, y) = std(x) < 1e-12 ? NaN : cov(x, y) / var(x)
    @printf("  steering -> twist        %10.4f deg/m\n",
            gain(steer_cmd, twist_diff))
    @printf("  twist    -> yaw moment   %10.4f Nm/deg\n",
            gain(twist_diff, moment_yaw))
    @printf("  yaw moment -> body rate  %10.6f (deg/s)/Nm\n",
            gain(moment_yaw, rate_yaw))
    @printf("  body rate -> heading rate%10.4f -\n",
            gain(rate_yaw, heading_rate))
end

failed_at > 0 && @warn "Run ended early" failed_at
nothing
