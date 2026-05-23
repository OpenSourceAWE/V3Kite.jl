# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Real-Time V3 Kite with Keyboard Control

Interactive keyboard-controlled simulation with real-time 3D
visualization. Steering and power/depower via arrow keys.

Controls:
  Left Arrow:  Turn left
  Right Arrow: Turn right
  Down Arrow:  Power
  Up Arrow:    Depower
  ESC:         Stop

Usage:
    julia --project=examples examples/realtime.jl
"""

using Pkg
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using GLMakie
using LinearAlgebra
using Statistics
using Printf

# =============================================================================
# Configuration
# =============================================================================

TETHER_LENGTH = 262.0
ELEVATION = 70.0       # degrees
AZIMUTH = 0.0          # degrees

V_WIND = 7.6
UP = 0.25              # Depower fraction [0, 1]

# Steering targets (keyboard-driven)
STEERING_TARGET = 15.0     # Target % when key held
STEERING_RAMP_RATE = 20.0  # %/s ramp speed

SIM_TIME = 60.0
FPS = 120
DISPLAY_FPS = 10
vector_scale = 1.0

# Keyboard control
power_rate = 0.5           # Percentage per keypress
max_depower_pct = 50.0

# Recording
record_video = false
output_filename = joinpath(
    v3_data_path(), "v3_realtime.mp4")

# =============================================================================
# Settling setup (matches open_loop / flight_replay)
# =============================================================================

el_rad = deg2rad(ELEVATION)
az_rad = deg2rad(AZIMUTH)
position = [
    cos(el_rad) * cos(az_rad) * TETHER_LENGTH,
    cos(el_rad) * sin(az_rad) * TETHER_LENGTH,
    sin(el_rad) * TETHER_LENGTH,
]
velocity = [0.0, 0.0, 0.0]
heading = 0.0
wind_vec = [V_WIND, 0.0, 0.0]

settle_config = V3SettleConfig(
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    dt = 0.001,
    num_steps = 400,
    num_substeps = 5,
    body_damping = [0.0, 0.0, 40.0],
    start_depower = UP * 100.0 + 10.0,
    course_correction_mode = :heading,
    course_correction_gain = 0.05,
    geom = V3GeomAdjustConfig(),
)
gc = settle_config.geom

@info "Settling V3 model..."
sam, _settle_log, settle_failed = settle_wing(settle_config;
    position, velocity, heading,
    steering = 0.0, depower = UP, wind_vec)
settle_failed && error("Settling failed")
sys = sam.sys_struct
sys.winches[1].brake = true

dt = 1.0 / FPS
display_interval = max(1, round(Int, FPS / DISPLAY_FPS))

# =============================================================================
# Create visualization
# =============================================================================

@info "Creating 3D visualization..."
scene = plot(sys; vector_scale, size=(1400, 900))
display(scene)

progress_text = Observable("t = 0.0s")
text!(scene, progress_text, position=Point2f(1380, 40),
    space=:pixel, fontsize=20, color=:black,
    align=(:right, :top))

control_text = Observable(
    "Steering: 0.0% | Depower: 0.0%")
text!(scene, control_text, position=Point2f(20, 60),
    space=:pixel, fontsize=14, color=:darkgreen,
    align=(:left, :top))

instructions = """
Keyboard Controls:
← Turn Left   → Turn Right
↓ Power       ↑ Depower
ESC to Stop
"""
text!(scene, instructions, position=Point2f(20, 130),
    space=:pixel, fontsize=16, color=:darkblue,
    align=(:left, :top))

# =============================================================================
# Keyboard control
# =============================================================================

steering_target = Ref(0.0)   # Target: ±STEERING_TARGET
steering_pct = Ref(0.0)     # Current (ramped) value
depower_pct_delta = Ref(0.0)
stop_simulation = Ref(false)

on(events(scene).keyboardbutton) do event
    if event.action in (Keyboard.press, Keyboard.repeat)
        if event.key == Keyboard.left
            steering_target[] = STEERING_TARGET
        elseif event.key == Keyboard.right
            steering_target[] = -STEERING_TARGET
        elseif event.key == Keyboard.down
            depower_pct_delta[] = clamp(
                depower_pct_delta[] - power_rate,
                -max_depower_pct, max_depower_pct)
        elseif event.key == Keyboard.up
            depower_pct_delta[] = clamp(
                depower_pct_delta[] + power_rate,
                -max_depower_pct, max_depower_pct)
        elseif event.key == Keyboard.escape
            stop_simulation[] = true
        end
    elseif event.action == Keyboard.release
        if event.key in (Keyboard.left, Keyboard.right)
            steering_target[] = 0.0
        end
    end
end

# =============================================================================
# Simulation loop
# =============================================================================

n_steps = if record_video
    Int(round(FPS * SIM_TIME))
else
    typemax(Int)
end

if record_video
    logger, sys_state = create_logger(
        sam, Int(round(FPS * SIM_TIME)))
end

io = if record_video
    VideoStream(scene; framerate=DISPLAY_FPS)
else
    nothing
end

wing_points = [p for p in sys.points if p.type == WING]

@info "Starting real-time simulation..." dt FPS
start_time = time()
simulation_time = 0.0
last_t = 0.0

try
    for step in 1:n_steps
        stop_simulation[] && break

        global simulation_time, last_t
        t = step * dt

        # Ramp steering toward keyboard target
        max_delta = STEERING_RAMP_RATE * dt
        diff = steering_target[] - steering_pct[]
        steering_pct[] += clamp(diff, -max_delta, max_delta)

        set_steering!(sys, steering_pct[] / 100.0, gc)

        depower_val = UP + depower_pct_delta[] / 100.0
        set_depower!(sys, depower_val, 0.0, gc)

        step_start = time()
        if !sim_step!(sam;
                set_values=[0.0], dt, vsm_interval=1)
            @warn "Simulation crashed at t=$t"
            break
        end
        simulation_time += time() - step_start
        last_t = t

        if record_video
            log_state!(logger, sys_state, sam, t)
        end

        if step % display_interval == 0
            plot!(sys; vector_scale)
            progress_text[] = @sprintf("t = %.1fs", t)
            control_text[] = @sprintf(
                "Steering: %.1f%% | Depower: %.1f%%",
                steering_pct[], depower_val * 100)
            record_video && recordframe!(io)
            sleep(0.001)
        end

        target_elapsed = t
        actual_elapsed = time() - start_time
        sleep(max(0.0, target_elapsed - actual_elapsed))

        if step % (FPS * 5) == 0
            avg_pos = mean(
                [p.pos_w for p in wing_points])
            @printf(
                "  t=%.1fs z=%.1fm st=%.1f%% dp=%.1f%%\n",
                t, avg_pos[3], steering_pct[],
                depower_val * 100)
        end
    end
catch e
    e isa InterruptException || rethrow(e)
    @info "Stopped by user" t=round(last_t, digits=2)
end

if record_video
    save(output_filename, io)
    @info "Video saved" output_filename

    report_performance(SIM_TIME, simulation_time)

    save_log(logger, "realtime_v3")
    syslog = load_log("realtime_v3")
    replay_scene = replay(
        syslog, sys; autoplay=false, loop=true)
    display(replay_scene)
else
    total_elapsed = time() - start_time
    @info "Complete" wall=round(total_elapsed, digits=2) sim_only=round(simulation_time, digits=2)
end

nothing
