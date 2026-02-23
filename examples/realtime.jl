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

using V3Kite
using GLMakie
using LinearAlgebra
using Statistics
using Printf

# =============================================================================
# Configuration
# =============================================================================

TETHER_LENGTH = 262
ELEVATION = 20.0

# Geometry (must match settle output)
TE_FRAC = 0.95
TIP_REDUCTION = 0.4
GEOM_SUFFIX = build_geom_suffix(
    V3_DEPOWER_L0, TIP_REDUCTION, TE_FRAC)

# Base control values
UP = 0.42                  # Depower fraction (old 0-1)
V_WIND = 7.6
DAMPING_PATTERN = [0.0, 10.0, 20.0]

# Ramp timing (depower only)
RAMP_START_UP = 0.1
RAMP_END_UP = 1.5

# Steering targets (keyboard-driven)
STEERING_TARGET = 10.0     # Target % when key held
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
# Model setup
# =============================================================================

config = V3SimConfig(
    struc_yaml_path =
        "struc_geometry_$(GEOM_SUFFIX).yaml",
    aero_yaml_path =
        "aero_geometry_$(GEOM_SUFFIX).yaml",
    vsm_settings_path = "CORRECT_vsm_settings.yaml",
    sim_time = SIM_TIME,
    fps = FPS,
    v_wind = V_WIND,
    tether_length = TETHER_LENGTH,
    elevation = ELEVATION,
    wing_type = REFINE,
    brake = true,
    damping_pattern = DAMPING_PATTERN,
)

@info "Creating V3 model..."
sam, sys = create_v3_model(config)

@info "Initializing model..."
init!(sam; remake=false, ignore_l0=false, remake_vsm=true)
sys.winches[1].brake = true
sys.points[1].extra_mass = 2.0
sys.points[1].area = 0.2

dt = 1.0 / FPS
display_interval = max(1, round(Int, FPS / DISPLAY_FPS))

# Store nominal segment lengths
nominal_l0_left = sys.segments[V3_STEERING_LEFT_IDX].l0
nominal_l0_right =
    sys.segments[V3_STEERING_RIGHT_IDX].l0
nominal_l0_depower = sys.segments[V3_DEPOWER_IDX].l0

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

        # Apply steering
        L_left, L_right =
            steering_percentage_to_lengths(steering_pct[])
        sys.segments[V3_STEERING_LEFT_IDX].l0 = L_left
        sys.segments[V3_STEERING_RIGHT_IDX].l0 = L_right

        # Ramp depower
        rf_up = ramp_factor(t, RAMP_START_UP, RAMP_END_UP)
        up_pct = UP * 100 + depower_pct_delta[]
        L_dp = depower_percentage_to_length(up_pct)
        sys.segments[V3_DEPOWER_IDX].l0 =
            nominal_l0_depower +
            rf_up * (L_dp - nominal_l0_depower)

        # Step simulation
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

        # Update visualization
        if step % display_interval == 0
            plot!(sys; vector_scale)
            progress_text[] = @sprintf("t = %.1fs", t)
            control_text[] = @sprintf(
                "Steering: %.1f%% | Depower: %.1f%%",
                steering_pct[], up_pct * rf_up)
            record_video && recordframe!(io)
            sleep(0.001)
        end

        # Real-time pacing
        target_elapsed = t
        actual_elapsed = time() - start_time
        sleep(max(0.0, target_elapsed - actual_elapsed))

        # Status every 5 seconds
        if step % (FPS * 5) == 0
            avg_pos = mean(
                [p.pos_w for p in wing_points])
            @printf(
                "  t=%.1fs z=%.1fm st=%.1f%% dp=%.1f%%\n",
                t, avg_pos[3], steering_pct[], up_pct)
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
