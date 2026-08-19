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

Quantifies the flow-curvature damping the continuous aero modes do not have.
`set_particle_panel_va!` gives each section the mean of its LE/TE apparent wind,
so a section rotating about its own spanwise axis produces no force at all and
the thin-airfoil increment `Δcm = -(π/4)q̂` is missing. The section rates are
recoverable from the live LE/TE point velocities, so the moment and the power
that increment would dissipate can be measured without carrying it in the RHS.

Sets that against the joints' Rayleigh damping at the same state. Both are
reported as dissipated power, which is the comparison that does not depend on
where a moment is taken. `V3KITE_JOINT_SCALE` overrides
`beam_joint_damping_scale`: measure at 1 so the section rates are free, since
a large scale suppresses the very rates the curvature term feeds on and makes
its own contribution look small. The Rayleigh term is linear in `beta`, so what
a larger scale would have dissipated at the same state is reported alongside.
`V3KITE_ANGULAR_DAMPING` sets `beam_angular_damping` on all three axes, which
tells the two artificial dampings apart: whether a run needs one, the other or
both is otherwise not visible from a run that carries both.

Also cross-checks the rigid-body kinematics: the wing's own `ω_b` against the
rate implied by differencing its frame, and `wing.heading` (the field the
controller reads) against the heading recomputed from the live wing frame. Those
agreeing rules the kinematics out; them disagreeing means the loop is closing on
a stale signal.

Prints a table plus an end-of-run summary and saves the log. `V3KITE_REPLAY=0`
skips the replay window so the run stays headless; the log is written either way.
`V3KITE_RECORD=<path>` records the run to that file at the log's own sample rate,
so the video plays at realtime; the extension picks the format (`.mp4`, `.gif`).
Recording from this run rather than from the saved log keeps it on the structure
the run actually built, which a cached model does not always reproduce. `bin/run_julia
--beam` (or `--psm`) picks the wing, and `V3KITE_AERO_MODE=continuous` overrides
the project's aero mode, which is how the same maneuver is compared across
couplings.

The loop closes on the heading recomputed from the live wing frame rather than on
`wing.heading`, so that the comparison of the two stays a diagnostic. Both
backends do write the field — on a particle wing it comes from
`write_wing_scalars!` — and it tracks the recomputed heading exactly.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using SymbolicAWEModels
using VortexStepMethod
using LinearAlgebra
using Printf
using Statistics

PROJECT = select_project(
    ["Timoshenko-beam wing" => "system_beam.yaml",
     "particle lattice" => "system_psm.yaml"];
    prompt = "Which wing model should fly?")

MAX_HEADING = 40.0    # setpoint amplitude [deg]
PERIOD = 30.0         # setpoint period [s]
REPORT_EVERY = 5      # steps between table rows
REFERENCE_SCALE = 30.0   # joint damping scale the counterfactual reports
REPLAY = get(ENV, "V3KITE_REPLAY", "1") != "0"
RECORD_PATH = get(ENV, "V3KITE_RECORD", "")

# `record` and `replay` live in the Makie extension, so both need GLMakie loaded.
(REPLAY || !isempty(RECORD_PATH)) && @eval using GLMakie, MakieControlPlots

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

"""
    section_pitch_rates(wing, points) -> (rates, chords, midpoints)

Per unrefined section: the rate [rad/s] it rotates about its own spanwise axis,
its chord [m], and its mid-chord point, all in the wing body frame, from the
live LE/TE structural points. Uniform wind cancels in the LE-minus-TE
difference, so the structural velocities alone carry the rate.
"""
function section_pitch_rates(wing, points)
    leading = Dict{Int64, Int64}()
    trailing = Dict{Int64, Int64}()
    for (point_idx, (section_idx, edge)) in wing.point_to_vsm_point
        (edge === :LE ? leading : trailing)[section_idx] = point_idx
    end
    sections = sort(collect(intersect(keys(leading), keys(trailing))))
    count = length(sections)
    frame = wing.R_b_to_w
    chords = zeros(count)
    midpoints = [zeros(3) for _ in 1:count]
    chord_dirs = [zeros(3) for _ in 1:count]
    velocities = [(zeros(3), zeros(3)) for _ in 1:count]
    for (k, section) in enumerate(sections)
        point_le = points[leading[section]]
        point_te = points[trailing[section]]
        pos_le = frame' * Vector(point_le.pos_w)
        pos_te = frame' * Vector(point_te.pos_w)
        chord_vec = pos_te .- pos_le
        chords[k] = norm(chord_vec)
        chords[k] < 1e-9 && continue
        chord_dirs[k] = chord_vec ./ chords[k]
        midpoints[k] = 0.5 .* (pos_le .+ pos_te)
        velocities[k] = (frame' * Vector(point_le.vel_w),
                         frame' * Vector(point_te.vel_w))
    end
    rates = zeros(count)
    for k in 1:count
        chords[k] < 1e-9 && continue
        neighbour = k < count ? k + 1 : k - 1
        spanwise = midpoints[neighbour] .- midpoints[k]
        norm(spanwise) < 1e-9 && continue
        normal = cross(chord_dirs[k], spanwise ./ norm(spanwise))
        norm(normal) < 1e-9 && continue
        vel_le, vel_te = velocities[k]
        rates[k] = section_pitch_rate(-vel_le, -vel_te, normal ./ norm(normal),
                                      chords[k])
    end
    return rates, chords, midpoints
end

"""
    curvature_damping(wing, points, rho) -> (moment, power)

Moment magnitude [Nm] and dissipated power [W] of the thin-airfoil pitch-rate
increment `Δcm = -(π/4)q̂` the continuous aero modes drop, summed over the
wing's panels. The dissipation is sign-definite in the section rate, so it does
not depend on which way the section normal points.
"""
function curvature_damping(wing, points, rho)
    isnothing(wing.point_to_vsm_point) && return (0.0, 0.0)
    rates, chords, midpoints = section_pitch_rates(wing, points)
    length(rates) < 2 && return (0.0, 0.0)
    v_rel = norm(wing.va_b)
    moment = 0.0
    power = 0.0
    for k in 1:(length(rates) - 1)
        rate = 0.5 * (rates[k] + rates[k + 1])
        chord = 0.5 * (chords[k] + chords[k + 1])
        width = norm(midpoints[k + 1] .- midpoints[k])
        delta_cm = VortexStepMethod.flow_curvature_cm(rate, chord, v_rel)
        panel = 0.5 * rho * v_rel^2 * chord^2 * width * delta_cm
        moment += abs(panel)
        power -= panel * rate
    end
    return moment, power
end

"""
    joint_damping_wrench(sys) -> (moment, power)

Rayleigh damping moment [Nm] the Timoshenko joints apply at their `a` nodes and
the power [W] they dissipate, at the live state. The model emits only the
combined elastic-plus-damping wrench, so the damping half is rebuilt here from
the library's own element frame and local wrench. Linear in each joint's `beta`,
so another damping scale is this scaled.
"""
function joint_damping_wrench(sys)
    moment = 0.0
    power = 0.0
    for joint in sys.timoshenko_joints
        body_a = sys.bodies[joint.body_a_idx]
        body_b = sys.bodies[joint.body_b_idx]
        rot_a = SymbolicAWEModels.quaternion_to_rotation_matrix(body_a.Q_b_to_w)
        rot_b = SymbolicAWEModels.quaternion_to_rotation_matrix(body_b.Q_b_to_w)
        x_a = Vector(body_a.pos_w) .+ rot_a * Vector(joint.anchor_a_b)
        x_b = Vector(body_b.pos_w) .+ rot_b * Vector(joint.anchor_b_b)
        e1, e2, e3, len = SymbolicAWEModels.timoshenko_element_frame(x_a, x_b,
                                                                    rot_a)
        element = [e1 e2 e3]
        omega_a = rot_a * Vector(body_a.ω_b)
        omega_b = rot_b * Vector(body_b.ω_b)
        vel_a = Vector(body_a.com_vel) .+ cross(omega_a, x_a .- Vector(body_a.com_w))
        vel_b = Vector(body_b.com_vel) .+ cross(omega_b, x_b .- Vector(body_b.com_w))
        relative = vel_b .- vel_a
        axis = element[:, 1]
        spin = dot(axis, omega_a) .* axis .+ cross(axis, relative) ./ len
        beta = joint.damping
        rate_a = element' * (omega_a .- spin)
        rate_b = element' * (omega_b .- spin)
        rigidities = (joint.EA, joint.GA, joint.GA, joint.GJ,
                      joint.EIy, joint.EIz)
        damp = SymbolicAWEModels.timoshenko_local_wrench(
            rigidities, joint.rest_length, joint.shear_coeff,
            beta * dot(relative, axis), beta .* rate_a, beta .* rate_b)
        force_a = element * damp[1]
        moment_a = element * damp[2]
        force_b = element * damp[3]
        moment_b = element * damp[4]
        moment += norm(moment_a)
        power -= dot(force_a, vel_a) + dot(moment_a, omega_a) +
                 dot(force_b, vel_b) + dot(moment_b, omega_b)
    end
    return moment, power
end

set_data_path(v3_data_path())
kite = load_kite(PROJECT)
heading = load_heading(PROJECT)
set = Settings(PROJECT)

aero_override = get(ENV, "V3KITE_AERO_MODE", "")
isempty(aero_override) ||
    (kite.aero_mode = V3Kite.parse_aero_mode(aero_override))

scale_override = get(ENV, "V3KITE_JOINT_SCALE", "")
isempty(scale_override) ||
    (kite.beam_joint_damping_scale = parse(Float64, scale_override))

angular_override = get(ENV, "V3KITE_ANGULAR_DAMPING", "")
isempty(angular_override) ||
    (kite.beam_angular_damping = fill(parse(Float64, angular_override), 3))

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
curvature_moment = Float64[]
curvature_power = Float64[]
rayleigh_moment = Float64[]
rayleigh_power = Float64[]

heading_prev = live_heading(wing)
frame_prev = copy(wing.R_b_to_w)

@printf("%5s %7s %7s %8s %8s %7s %8s %9s %9s %9s %9s %8s %8s %9s %9s\n",
        "step", "t[s]", "cmd[°]", "hfld[°]", "hlive[°]", "g_k", "steer[m]",
        "dtwist[°]", "Mapp[Nm]", "Mvsm[Nm]", "|M|[Nm]", "ωyaw[°/s]",
        "ḣead[°/s]", "Mcrv[Nm]", "Pcrv[W]")

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

    rho = SymbolicAWEModels.air_density(sam.sys_struct.am, wing.pos_w[3])
    curv_moment, curv_power = curvature_damping(wing, sys.points, rho)
    push!(curvature_moment, curv_moment)
    push!(curvature_power, curv_power)

    joint_moment, joint_power = joint_damping_wrench(sys)
    push!(rayleigh_moment, joint_moment)
    push!(rayleigh_power, joint_power)

    if step % REPORT_EVERY == 0 || step <= 3
        @printf("%5d %7.2f %7.2f %8.2f %8.2f %7.3f %8.4f %9.3f %9.1f %9.1f %9.1f %8.3f %8.3f %9.2f %9.2f\n",
                step, t, rad2deg(target_rad), heading_field[end],
                heading_live[end], pid.K, steer_ctrl, twist_diff[end],
                moment_yaw[end], moment_vsm[end], moment_norm[end],
                rate_yaw[end], head_rate, curv_moment, curv_power)
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

println("\n===== damping the beam gets, and the damping it does not =====")
scale = kite.beam_joint_damping_scale
counterfactual = REFERENCE_SCALE / scale
@printf("  joint damping scale flown  %.4g  (counterfactual x%.4g)\n",
        scale, REFERENCE_SCALE)
span("curvature moment", curvature_moment, "Nm")
span("curvature dissipation", curvature_power, "W")
span("Rayleigh moment", rayleigh_moment, "Nm")
span("Rayleigh dissipation", rayleigh_power, "W")
span("Rayleigh moment at ref", counterfactual .* rayleigh_moment, "Nm")
span("Rayleigh dissipation at ref", counterfactual .* rayleigh_power, "W")
if !isempty(curvature_power) && mean(rayleigh_power) != 0
    @printf("  %-26s %9.2f %%\n", "curvature / Rayleigh flown",
            100 * mean(curvature_power) / mean(rayleigh_power))
    @printf("  %-26s %9.2f %%\n", "curvature / Rayleigh at ref",
            100 * mean(curvature_power) / (counterfactual * mean(rayleigh_power)))
end
if !isempty(curvature_moment) && mean(moment_norm) > 0
    @printf("  %-26s %9.2f %%\n", "curvature / aero moment",
            100 * mean(curvature_moment) / mean(moment_norm))
end

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

log_name = "v3kite_diagnose_$(splitext(PROJECT)[1])"
save_log(logger, log_name)
@info "Log saved" log_name

if !isempty(RECORD_PATH)
    mkpath(dirname(RECORD_PATH))
    framerate = max(1, round(Int, 1 / dt))
    @info "Recording" RECORD_PATH framerate
    SymbolicAWEModels.record(load_log(log_name), sam.sys_struct, RECORD_PATH;
                             size=(1000, 800), framerate)
    @info "Recorded" kB=filesize(RECORD_PATH) ÷ 1024
end

if REPLAY
    scene = SymbolicAWEModels.replay(load_log(log_name), sam.sys_struct)
    display(GLMakie.Screen(), scene)
end
nothing
