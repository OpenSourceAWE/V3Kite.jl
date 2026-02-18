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

SIM_TIME = 60.0
FPS = 20
dt = 1.0 / FPS
realtime_factor = 1.0
plot_interval = 1
vector_scale = 1.0

# Control parameters
max_steering = 0.3
max_power_depower = 0.3
steering_rate = 0.001
power_rate = 0.002

# Damping
initial_damping = 10.0
decay_time = 2.0

# Recording
record_video = false
output_filename = joinpath(v3_data_path(), "v3_realtime.mp4")

# =============================================================================
# Model setup
# =============================================================================

config = V3SimConfig(
    struc_yaml_path = "struc_geometry_stable.yaml",
    aero_yaml_path = "aero_geometry_stable.yaml",
    sim_time = SIM_TIME,
    fps = FPS,
    wing_type = REFINE,
    remake_cache = false,
)

@info "Creating V3 model..."
sam, sys = create_v3_model(config)

# World frame damping
SymbolicAWEModels.set_world_frame_damping(
    sys, initial_damping, 1:38)

@info "Initializing model..."
init!(sam; remake=false, ignore_l0=false)

# Settle initial state
wing_points = [p for p in sys.points if p.type == WING]
[p.fix_static = true for p in sys.points if p.type == WING]
next_step!(sam; dt=10.0)
[p.fix_static = false for p in sys.points if p.type == WING]

# Store initial segment lengths
seg_left_init = sys.segments[V3_STEERING_LEFT_IDX].l0
seg_right_init = sys.segments[V3_STEERING_RIGHT_IDX].l0
seg_depower_init = sys.segments[V3_DEPOWER_IDX].l0

# =============================================================================
# Create visualization
# =============================================================================

@info "Creating 3D visualization..."
scene = plot(sys; vector_scale, size=(1400, 900))
display(scene)

progress_text = Observable("Progress: 0%")
text!(scene, progress_text, position=Point2f(1380, 40),
    space=:pixel, fontsize=20, color=:black,
    align=(:right, :top))

control_text = Observable("Steering: 0.00m | Power: 0.00m")
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

current_steering_delta = Ref(0.0)
current_power_delta = Ref(0.0)
stop_simulation = Ref(false)

on(events(scene).keyboardbutton) do event
    if event.action in (Keyboard.press, Keyboard.repeat)
        if event.key == Keyboard.left
            current_steering_delta[] = clamp(
                current_steering_delta[] - steering_rate,
                -max_steering, max_steering)
        elseif event.key == Keyboard.right
            current_steering_delta[] = clamp(
                current_steering_delta[] + steering_rate,
                -max_steering, max_steering)
        elseif event.key == Keyboard.down
            current_power_delta[] = clamp(
                current_power_delta[] - power_rate,
                -max_power_depower, max_power_depower)
        elseif event.key == Keyboard.up
            current_power_delta[] = clamp(
                current_power_delta[] + power_rate,
                -max_power_depower, max_power_depower)
        elseif event.key == Keyboard.escape
            stop_simulation[] = true
        end
    end
end

# =============================================================================
# Simulation loop
# =============================================================================

n_steps = Int(round(FPS * SIM_TIME))
logger, sys_state = create_logger(sam, n_steps)

io = record_video ? VideoStream(scene; framerate=FPS) : nothing

@info "Starting real-time simulation..." SIM_TIME dt
start_time = time()
simulation_time = 0.0

for step in 1:n_steps
    stop_simulation[] && break

    t = step * dt
    target_elapsed = t / realtime_factor

    # Damping decay
    if t <= decay_time
        damping = initial_damping * (1.0 - t / decay_time)
        SymbolicAWEModels.set_world_frame_damping(
            sys, damping, 1:38)
    else
        SymbolicAWEModels.set_world_frame_damping(
            sys, 0.0, 1:38)
    end

    # Apply control
    sys.segments[V3_STEERING_LEFT_IDX].l0 =
        seg_left_init + current_steering_delta[]
    sys.segments[V3_STEERING_RIGHT_IDX].l0 =
        seg_right_init - current_steering_delta[]
    sys.segments[V3_DEPOWER_IDX].l0 =
        seg_depower_init + current_power_delta[]

    control_text[] = @sprintf(
        "Steering: %.3fm | Power: %.3fm",
        current_steering_delta[], current_power_delta[])

    step_start = time()
    if !sim_step!(sam; dt, vsm_interval=1)
        @warn "Simulation crashed at t=$t"
        break
    end
    simulation_time += time() - step_start

    log_state!(logger, sys_state, sam, t)

    # Update visualization
    if step % plot_interval == 0
        plot!(sys; vector_scale)
        progress_text[] = @sprintf(
            "Progress: %d%%", round(Int, 100 * step / n_steps))
        record_video && recordframe!(io)
        sleep(0.001)
    end

    actual_elapsed = time() - start_time
    sleep(max(0.0, target_elapsed - actual_elapsed))

    if step % (n_steps ÷ 10) == 0
        avg_pos = mean([p.pos_w for p in wing_points])
        @printf("  %.0f%% (t=%.1fs, z=%.2fm)\n",
            100 * step / n_steps, t, avg_pos[3])
    end
end

if record_video
    save(output_filename, io)
    @info "Video saved" output_filename
end

total_elapsed = time() - start_time
@info "Complete" wall=round(total_elapsed, digits=2) sim_only=round(simulation_time, digits=2) speedup=round(SIM_TIME / simulation_time, digits=2)

# Save and replay
save_log(logger, "realtime_v3")
syslog = load_log("realtime_v3")
replay_scene = replay(syslog, sys; autoplay=false, loop=true)
display(replay_scene)

nothing
