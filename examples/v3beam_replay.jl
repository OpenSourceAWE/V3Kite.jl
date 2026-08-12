# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3 Beam-Wing Flight Data Replay

`flight_replay.jl` on the Timoshenko-beam wing: the same recorded steering,
depower and reel-out are played through the beam geometry instead of the particle
lattice. Kept apart from `flight_replay.jl` rather than switched inside it,
because almost nothing about the setup is shared — a different aerodynamic
coupling, a different backend, and none of the wing-lattice corrections.

`AeroPressure` distributes the VSM's per-section force over the structural points
as a surface traction, which is the transfer the beam wing is built for; the
particle lattice uses the direct one. The `KernelBackend` assembles one kernel
per component, the 23 bodies and 21 joints making the monolithic build the
dominant cost otherwise.

The tip and trailing-edge reductions stay off. Both are lengths subtracted from
numbered segments of the particle lattice, and the beam wing has no segments that
mean the same thing — its shape comes from the beam bodies, not from wire lengths.

`v3beam_aero_geometry.jl` writes the aero geometry this loads, and
`v3beam_geometry.jl` the structural one; neither runs here.

Settling starts from the relaxed state `relax_bridle.jl` writes, since the
measured bridle lengths and the measured node coordinates disagree badly enough
that no implicit solver takes a first step from the placed geometry.

Only the plain replay is here — no distance-based steering, no heading PID, no
comparison figures. `flight_replay.jl` has those, and they are worth extracting
to a shared driver once this settles reliably.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using SymbolicAWEModels
using GLMakie
using LazyArtifacts
using KiteUtils

# =============================================================================
# Configuration
# =============================================================================

STRUC_YAML = "struc_geometry_beam.yaml"
# The surface-resolved geometry v3beam_aero_geometry.jl writes. The stock
# aero_geometry.yaml carries lift/drag/moment polars only, which AeroPressure
# cannot build a station-point map from.
AERO_YAML = joinpath("polars_neuralfoil_pressure", "geometry.yaml")
VSM_SETTINGS = "vsm_settings.yaml"

AERO_MODE = AeroPressure()
BACKEND = KernelBackend()

# Depower the relaxed state was written at, see relax_bridle.jl.
RELAXED_DEPOWER = 0.20
RELAXED_STATE = relaxed_state_name(STRUC_YAML, RELAXED_DEPOWER) * ".arrow"

SECTION = "straight_right_2025"
START_UTC = "15:36:29.0"
END_UTC = "15:36:38.0"
N_SUBSTEPS = 2     # data is 10 Hz; 2 substeps gives dt = 0.05 s
VSM_INTERVAL = 1   # steps between VSM aero solves

DEPOWER_OFFSET = -0.07  # added to the recorded depower
STEERING_MULTIPLIER = 1.0
BODY_DAMPING = [0.0, 0.0, 20.0]
SETTLE_BODY_DAMPING = [0.0, 0.0, 40.0]
SETTLE_STEPS = 69
WIND_SOURCE_SPEED = :ekf   # :ekf or :lidar
WIND_SOURCE_DIR = :lidar   # :ekf or :lidar

# =============================================================================
# Flight data
# =============================================================================

datadir = joinpath(artifact"flight_data", "flight_data")
h5_path = joinpath(datadir, "ekf_awe_2025-10-09.h5")

full_data = load_flight_data(h5_path)
limited_data, _ = limit_by_utc(full_data, START_UTC, END_UTC)
data = interpolate_flight_data(add_distance_column(limited_data), N_SUBSTEPS)

"""
    beam_row(raw) -> NamedTuple

The flight state a beam replay needs: pose, velocity, tape settings, tether and
wind. Leaner than `flight_replay.jl`'s row, which also carries the coefficients
and angle-of-attack variants its figures compare against.
"""
function beam_row(raw)
    pos = [raw.ekf_kite_position_x, raw.ekf_kite_position_y,
        raw.ekf_kite_position_z]
    wind_vec = compute_wind_vec(raw, pos[3];
        speed_source = WIND_SOURCE_SPEED, dir_source = WIND_SOURCE_DIR)
    return (time = raw.time,
        x = pos[1], y = pos[2], z = pos[3],
        vx = raw.ekf_kite_velocity_x, vy = raw.ekf_kite_velocity_y,
        vz = raw.ekf_kite_velocity_z,
        heading = calc_csv_heading(raw.ekf_kite_roll, raw.ekf_kite_pitch,
            raw.ekf_kite_yaw, pos),
        tether_len = raw.ekf_tether_length,
        tether_vel = raw.tether_reelout_speed,
        steering = raw.kcu_actual_steering / 100.0,
        depower = raw.kcu_actual_depower / 100.0,
        v_app = raw.ekf_kite_apparent_windspeed,
        wind_vec = wind_vec)
end

row_at(step) = beam_row(NamedTuple{keys(data)}(
    Tuple(data[k][step] for k in keys(data))))

row1 = row_at(1)
n_steps = length(data.time)
dt = (data.time[2] - data.time[1])

@info "V3 beam-wing replay" SECTION n_steps dt tether_len=row1.tether_len depower=row1.depower

# =============================================================================
# Settling
# =============================================================================

geom = V3GeomAdjustConfig(
    reduce_tip = false, reduce_te = false,
    reduce_depower = false, reduce_steering = false,
    depower_offset = DEPOWER_OFFSET)

settle_config = V3SettleConfig(
    source_struc_path = STRUC_YAML,
    source_aero_path = AERO_YAML,
    vsm_settings_path = VSM_SETTINGS,
    init_state_path = RELAXED_STATE,
    backend = BACKEND,
    aero_mode = AERO_MODE,
    world_damping = 0.0,
    body_damping = SETTLE_BODY_DAMPING,
    min_damping = BODY_DAMPING,
    decay_steps = 50,
    v_wind = row1.v_app,
    tether_length = Float64(row1.tether_len),
    dt = 0.05,
    num_steps = SETTLE_STEPS,
    num_substeps = 1,
    start_depower = row1.depower * 100.0 + 10.0,
    course_correction_gain = 0.05,
    geom = geom)

sam, settle_log, settle_failed = settle_wing(settle_config, row1; remake = true)
settle_failed && error("Settling failed; the beam replay has nothing to fly")
sys = sam.sys_struct

# =============================================================================
# Replay loop
# =============================================================================

logger, sys_state = create_logger(sam, n_steps)

@info "Replaying flight data..."
replay_start = time()

for step in 1:n_steps
    row = row_at(step)
    sys.set.wind_vec = KiteUtils.MVec3(row.wind_vec)

    set_steering!(sys, clamp(row.steering, -1.0, 1.0) * STEERING_MULTIPLIER,
        geom; min_l0 = 0.01)
    set_depower!(sys, row.depower, row.steering, geom)

    winch = sys.winches[1]
    winch.brake = true
    winch.vel = row.tether_vel
    sys.tethers[1].len = row.tether_len
    sys.tethers[1].stretched_len = row.tether_len

    if !sim_step!(sam; dt, vsm_interval = VSM_INTERVAL)
        @error "Replay failed" step t=row.time
        break
    end
    log_state!(logger, sys_state, sam, step * dt)

    should_report(step, n_steps) && @info "Step $step/$n_steps" times_realtime=round(step * dt / (time() - replay_start), digits=2)
end

report_performance(n_steps * dt, time() - replay_start)

save_log(logger, "v3beam_replay")
syslog = load_log("v3beam_replay")

# =============================================================================
# Visualization
# =============================================================================

@info "Creating visualization..."
scene = SymbolicAWEModels.replay(syslog, sam.sys_struct)
display(GLMakie.Screen(), scene)

@info "Replay complete!"
nothing
