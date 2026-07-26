import SymbolicAWEModels: winch_force, tether_length
using Printf

# ===================== Winch position controller ===================== #
# Gains for the cascaded winch length controller (see `winch_position_torque!`)
# are stored in a `WC_Settings` struct loaded from `data/wc_settings.yaml`
# (see `src/wc_settings.jl`), not hard-coded here. The KCU actuator tape rate
# limits (`v_depower`/`v_steering`) live in the `kcu:` section of
# `data/settings.yaml` and are loaded into `Settings` like any other field.

"""
    WinchPosController

Runtime state of the cascaded winch length controller: an outer
proportional loop on the tether length error produces a speed setpoint
(saturated by `speed_limit` and rate-limited by `acceleration_limit` in
`step!`), and an inner PI loop (`speed_pid`) on the speed error produces a
winch-torque correction added to a force feed-forward. `v_sp_prev` carries
the rate-limited speed setpoint between steps.
"""
Base.@kwdef mutable struct WinchPosController
    "Inner speed PI controller; output is a winch torque correction [N·m]"
    speed_pid::DiscretePID
    "Outer proportional gain, length error [m] → speed setpoint [m/s]"
    kp_pos::Float64 = WC_Settings().winch_pos_kp
    "Rate-limited speed setpoint carried between steps [m/s]"
    v_sp_prev::Float64 = 0.0
end

@with_kw mutable struct V3KITE <: AbstractKiteModel
    "Reference to the settings struct"
    set::Settings
    "Reference to the KCU model (Kite Control Unit) as implemented in the package KitePodModels"
    kcu::KCU
    sam::SymbolicAWEModel
    "Reference to the atmospheric model as implemented in the package AtmosphericModels"
    am::AtmosphericModel = AtmosphericModel(set)
    "Geometry config used by the depower/steering tape conversions"
    gc::V3GeomAdjustConfig = V3GeomAdjustConfig()
    "Time step of the high-level `step!` loop [s]"
    dt::Float64 = NaN
    "Current simulation state, updated in place every `step!`"
    sys_state::Union{SysState, Nothing} = nothing
    "Log of the simulation run"
    logger::Union{Logger, Nothing} = nothing
    "Total number of simulation steps sized by `init`"
    steps::Int = 0
    "Integrator time at the previous `update_sys_state!` call [s]"
    t_prev::Float64 = NaN
    "Heading at the previous `update_sys_state!` call [rad]"
    heading_prev::Float64 = NaN
    "Wall-clock time at the previous realtime-factor print, `NaN` before the first [s]"
    last_step_time::Float64 = NaN
    "Winch length-controller state, set up by `init`"
    winch_ctrl::Union{WinchPosController, Nothing} = nothing
end

function Base.getproperty(s::V3KITE, name::Symbol)
    if name === :sys
        return getfield(s, :sam).sys_struct
    end
    return getfield(s, name)
end

function Base.propertynames(::V3KITE, private::Bool=false)
    return (fieldnames(V3KITE)..., :sys)
end

"""
    update_sys_state!(ss::SysState, s::V3KITE, zoom=1.0)

Update `ss` from the model (the `SymbolicAWEModel` method) and additionally
fill in two fields that method leaves wrong/unset for V3 (which has no twist
`groups`, unlike the group-actuated models the base method targets):
- `heading_rate` [rad/s], computed as a wrapped finite difference of the
  heading over the model's integrator clock.
- `depower` (`0..1`), the KCU's actual (tape-lag-limited) depower fraction —
  the base method leaves this at 0 since it is only filled from `groups`.

The previous heading and time are stored on the `V3KITE` (on the integrator
clock), so the `heading_rate` result is immune to callers restamping
`ss.time` with a log time between calls. On the first call (or if the
integrator time did not advance) `heading_rate` is left unchanged.
"""
function KiteUtils.update_sys_state!(ss::SysState, s::V3KITE, zoom=1.0)
    update_sys_state!(ss, s.sam, zoom)  # sets ss.time to the integrator time
    dt = ss.time - s.t_prev
    if dt > 0
        ss.heading_rate = wrap_to_pi(ss.heading - s.heading_prev) / dt
    end
    s.t_prev = ss.time
    s.heading_prev = ss.heading
    ss.depower = KitePodModels.get_depower(s.kcu)
    return nothing
end

# Output functions

"""
    lift_drag(s::V3KITE) -> (lift, drag)

Return aerodynamic lift and drag forces [N] from the shared
simulation-helper definitions.
"""
function lift_drag(s::V3KITE)
    return compute_lift_drag(s.sam)
end

"""
    total_drag(s::V3KITE) -> (wing_drag, tether_drag, total)

Return the total drag force [N] decomposed into:
- `wing_drag`: wing aerodynamic drag (VSM + wing-point parasitic)
- `tether_drag`: tether, bridle and KCU parasitic drag
- `total`: sum of both

Used for the L/D_wing vs.
L/D_eff comparison in examples/parking.jl.
"""
function total_drag(s::V3KITE)
    wing_drag = compute_drag(s.sam)
    tether_drag = compute_tether_drag(s.sam)
    return (wing_drag, tether_drag, wing_drag + tether_drag)
end

"""
    unstretched_length(s::V3KITE) -> Float64

Return the current unstretched tether length [m].
"""
function unstretched_length(s::V3KITE)
    s.sys.tethers[1].len
end

"""
    v_wind_kite(s::V3KITE) -> Vector{Float64}

Return the wind velocity vector [m/s] used at the kite position.

If `s.set.profile_law == 0`, this returns the ground-level wind vector
`s.set.wind_vec` unchanged. Otherwise, it computes the kite height from
`pos_kite(s)[3]`, evaluates `calc_wind_factor(s.am, height)`, and returns
the scaled vector `calc_wind_factor(s.am, height) * s.set.wind_vec`.
"""
function v_wind_kite(s::V3KITE)
    s.set.profile_law == 0 && return s.set.wind_vec
    height = pos_kite(s)[3]
    calc_wind_factor(s.am, height) * s.set.wind_vec
end

"""
    pos_kite(s::V3KITE) -> Vector{Float64}

Return the centre-of-pressure position of the kite from the four mid-span wing points
(indices 10–13: LE/TE pairs). The centre of pressure is at ~30 % chord, so each pair
is weighted 0.7 × LE + 0.3 × TE, then averaged over both sides.
"""
function pos_kite(s::V3KITE)
    pts = s.sys.points
    (0.7 .* pts[10].pos_w .+ 0.3 .* pts[11].pos_w .+
     0.7 .* pts[12].pos_w .+ 0.3 .* pts[13].pos_w) ./ 2
end

"""
    tether_length(s::V3KITE) -> Float64

Return the current stretched tether length [m] from the model's
`stretched_len` field for the primary tether.
"""
function tether_length(s::V3KITE)
    s.sam.sys_struct.tethers[1].stretched_len
end

"""
    calc_height(s::V3KITE) -> Float64

Return the height [m] of the kite's centre-of-pressure above ground
(z-component of `pos_kite`).
"""
function calc_height(s::V3KITE)
    pos_kite(s)[3]
end

"""
    calc_elevation(s::V3KITE) -> Float64

Determine the elevation angle of the kite in radian.
"""
function calc_elevation(s::V3KITE)
    KiteUtils.calc_elevation(pos_kite(s))
end

"""
    upwind_dir(s::V3KITE) -> Float64
    upwind_dir(v_wind_gnd) -> Float64

Return the upwind direction [rad] in the horizontal plane, measured clockwise
from North (NED convention). Computed from the ground-level wind vector so that
the result points into the wind (i.e. opposite to the wind direction).
Returns `NaN` if the wind vector has no horizontal component.
"""
function upwind_dir(s::V3KITE)
    upwind_dir(s.set.wind_vec)
end
function upwind_dir(v_wind_gnd)
    if v_wind_gnd[1] == 0.0 && v_wind_gnd[2] == 0.0
        return NaN
    end
    wind_dir = atan(v_wind_gnd[2], v_wind_gnd[1])
    -(wind_dir + π/2)
end

"""
    calc_azimuth(s::V3KITE) -> Float64

Determine the azimuth angle of the kite in wind reference frame in radian.
Positive anti-clockwise when seen from above.
"""
function calc_azimuth(s::V3KITE)
    azn = KiteUtils.azimuth_north(pos_kite(s))
    azn2azw(azn; upwind_dir = upwind_dir(s))
end

"""
    calc_azimuth_east(s::V3KITE) -> Float64

Determine the azimuth_east angle of the kite in radian.
Positive clockwise when seen from above.
"""
function calc_azimuth_east(s::V3KITE)
    KiteUtils.azimuth_east(pos_kite(s))
end

"""
    calc_azimuth_north(s::V3KITE) -> Float64

Determine the azimuth_north angle of the kite in radian.
Positive anti-clockwise when seen from above.
"""
function calc_azimuth_north(s::V3KITE)
    KiteUtils.azimuth_north(pos_kite(s))
end

"""
    kite_ref_frame(s::V3KITE)

Returns a tuple of the x, y, and z vectors of the kite reference frame.
"""
function kite_ref_frame(s::V3KITE)
    R_b_w = calc_R_b_w(s.sys)
    return R_b_w[:, 1], R_b_w[:, 2], R_b_w[:, 3]
end

function calc_orient_quat(s::V3KITE; viewer=false)
    if viewer
        x, _, z = kite_ref_frame(s)
        pos_kite_ = pos_kite(s)
        pos_before = pos_kite_ .+ z

        rotation = rot(pos_kite_, pos_before, -x)
    else
        x, y, z = kite_ref_frame(s) # in ENU reference
        x = enu2ned(x)
        y = enu2ned(y) 
        z = enu2ned(z)

        # reference frame for the orientation: NED (north, east, down)
        ax = @SVector [1, 0, 0]
        ay = @SVector [0, 1, 0]
        az = @SVector [0, 0, 1]
        rotation = rot3d(ax, ay, az, x, y, z)
    end
    q = QuatRotation(rotation)
    return Rotations.params(q)
end


"""
    orient_euler(s::V3KITE)

Calculate and return the orientation of the kite in euler angles (roll, pitch, yaw)
as SVector.
"""
function orient_euler(s::V3KITE)
    q = QuatRotation(calc_orient_quat(s))
    roll, pitch, yaw = quat2euler(q)
    SVector(roll, pitch, yaw)
end

"""
    calc_heading(s::V3KITE; upwind_dir_=upwind_dir(s), neg_azimuth=false)

Determine the heading angle of the kite in radian.
"""
function calc_heading(s::V3KITE; upwind_dir_=upwind_dir(s), neg_azimuth=false)
    orientation = orient_euler(s)
    elevation = calc_elevation(s)
    if neg_azimuth
        azimuth = -calc_azimuth(s)
    else
        azimuth = calc_azimuth(s)
    end
    heading = KiteUtils.calc_heading(orientation, elevation, azimuth; upwind_dir=upwind_dir_)
    # SymbolicAWEModels' body x-axis points opposite to the Xsens-sensor
    # convention `KiteUtils.calc_heading` assumes (same correction as
    # `calc_csv_heading` applies to EKF-derived orientation).
    wrap_to_pi(heading + π)
end

"""
    calc_course(s::V3KITE; neg_azimuth=false)

Return the course angle of the kite in radian as stored in
`sam.sys_struct.wings[1].course`. If `neg_azimuth=true`, return
the course using the negated azimuth sign convention.
"""
function calc_course(s::V3KITE; neg_azimuth=false)
    course = s.sam.sys_struct.wings[1].course
    neg_azimuth ? -course : course
end

"""
    cl_cd(s::V3KITE) -> (cl, cd)

Calculate lift and drag coefficients using the shared
simulation-helper definitions.
"""
function cl_cd(s::V3KITE)
    return compute_lift_coeff(s.sam), compute_drag_coeff(s.sam)
end

"""
    winch_force(s::V3KITE) -> Float64

Return the absolute value of the force at the winch as calculated during the last timestep.
"""
function winch_force(s::V3KITE)
    norm(s.sys.winches[1].force)
end

"""
    reel_out_speed(s::V3KITE)

Return the current reel-out speed of the winch in m/s.
"""
function reel_out_speed(s::V3KITE) s.sys.winches[1].vel end

"""
    states(s::V3KITE)

Return the number of states of the V3KITE model.
"""
function states(s::V3KITE)
    length(s.sam.prob.prob.u0)
end

"""
    spring_forces(s::V3KITE) -> Vector{Float64}

Return an array of scalar spring forces [N] from
`sam.sys_struct.segments[seg_idx].force` for all tether segments.
"""
function spring_forces(s::V3KITE)
    tether = s.sys.tethers[1]
    forces = zeros(Float64, length(tether.segment_idxs))
    for (i, seg_idx) in enumerate(tether.segment_idxs)
        forces[i] = s.sam.sys_struct.segments[seg_idx].force
    end
    return forces
end

# ============================ High-level interface ============================ #
# Two functions, `init` and `step!`, wrap the settle/simulate/log boilerplate
# (cf. examples/parking.jl) so that an example reduces to `init` plus a loop of
# `step!` calls.

"""
    winch_position_torque!(s::V3KITE, set_length, speed_limit,
                            acceleration_limit) -> torque

Cascaded winch length controller. Outer proportional loop on the tether
length error (`set_length - unstretched_length(s)`) yields a reel-out speed
setpoint, clamped to `±speed_limit` [m/s] and rate-limited by
`acceleration_limit` [m/s²]. The inner PI loop (`s.winch_ctrl.speed_pid`) on
the speed error yields a winch-torque correction, added to a force
feed-forward `force_to_torque(winch_force(s), s.sys)` (the steady/gravity
holding torque from the measured winch force). Returns the winch set torque
[N·m], applied directly as in `examples/reel_out_v3.jl`: at parking
equilibrium (speed setpoint 0, measured speed 0) the correction vanishes and
the output is exactly the holding torque.
"""
function winch_position_torque!(s::V3KITE, set_length, speed_limit,
                                 acceleration_limit)
    ctrl = s.winch_ctrl
    # Outer P loop: length error → speed setpoint.
    v_sp = ctrl.kp_pos * (set_length - unstretched_length(s))
    v_sp = clamp(v_sp, -speed_limit, speed_limit)
    dv_max = acceleration_limit * s.dt
    v_sp = clamp(v_sp, ctrl.v_sp_prev - dv_max, ctrl.v_sp_prev + dv_max)
    ctrl.v_sp_prev = v_sp
    # Inner PI loop: speed error → torque correction.
    dtau = ctrl.speed_pid(v_sp, reel_out_speed(s), 0.0)
    tau_ff = force_to_torque(winch_force(s), s.sys)
    return tau_ff + dtau
end

"""
    init(v_wind_gnd, l_tether; elevation=nothing, upwind_dir=-π/2,
         depower_setpoint=0.25, dt=nothing, sim_time=nothing,
         gc=V3GeomAdjustConfig(), wc=nothing, body_damping=[10.0, 10.0, 40.0],
         remake=false) -> V3KITE

Build and return a ready `V3KITE`, settled at a fixed depower equilibrium,
for a `step!` simulation loop (see `examples/simple_parking.jl`).

`v_wind_gnd` is the ground wind speed at the reference height [m/s] and
`l_tether` the initial tether length [m]. `elevation` [deg] falls back to the
settings value; `upwind_dir` [rad] sets the wind direction. `depower_setpoint`
is the initial `rel_depower` in `[0, 1]` (not meters). `dt` [s] and `sim_time`
[s] size the logger and step count, falling back to `1/set.sample_freq` and
`set.sim_time`. `gc` holds the geometry adjustments; `remake=true` forces
re-settling (ignoring the `data/settled_*.bin` cache). `wc` holds the winch
length-controller gains; when `nothing` (the default) they are loaded from the
file named in the `wc_settings` field of `system.yaml` (see `WC_Settings`).
`system_yaml` names the system-YAML file (default `"system.yaml"`) that points
at the active settings/wc-settings files; pass e.g. `"system2.yaml"` to use an
alternate config.

`body_damping` is the per-axis body-frame damping `[x, y, z]` in the wing frame,
applied to every point during settling. It acts on point velocity *relative to
the wing*, so it damps bridle vibration without slowing the kite's global motion
(unlike world-frame damping). It is baked into the settled geometry and carries
over into the returned model, so it also forms part of the settling cache key —
changing it produces a different `data/settled_*.bin` rather than silently
reusing the old one.

The in-plane terms matter as much as the normal one: the default used to be
`[0, 0, 40]`, which damps only normal to the wing surface and therefore could not
touch the lightly-damped ~5.5 Hz *in-plane* bridle mode that dominated the parked
solver cost. Adding the `[10, 10, ...]` terms cuts the parked AoA ripple from
0.144° to 0.020° peak-to-peak and the solver step count by 3.4×, with the settled
trim (AoA, elevation, tether force) unchanged to within 0.5 %.

Raising the in-plane terms further keeps helping (`[20, 20, 40]` is 5.4× on
solver steps) but the settling itself goes unstable at fixed `dt = 0.001`
somewhere between 15 and 20 on the `system_cabauw.yaml` configuration, so the
default keeps a margin. Pass a larger value explicitly if your configuration
tolerates it — a diverged settling now fails loudly rather than caching broken
geometry. See PlanSuppressOscillations.md for the sweep.

The settling stage mirrors `examples/parking.jl`. The KCU actuator model, the
winch position controller, the logger, `sys_state`, and `steps` are stored on
the returned instance, and the winch is left un-braked.
"""
function init(v_wind_gnd, l_tether;
              elevation = nothing,
              upwind_dir = -π/2,
              depower_setpoint = 0.25,
              dt = nothing,
              sim_time = nothing,
              gc = V3GeomAdjustConfig(),
              wc = nothing,
              system_yaml = "system.yaml",
              body_damping = [0.0, 0.0, 40.0],
              remake = false)
    # Elevation fallback comes from the on-disk settings.
    set_data_path(v3_data_path())
    # Winch-controller settings fall back to the file named in the `wc_settings`
    # field of system.yaml (needs the data path set above).
    isnothing(wc) && (wc = WC_Settings(wc_settings(system_yaml)))
    if isnothing(elevation)
        elevation = Settings(system_yaml).elevation
    end
    el_rad = deg2rad(elevation)
    wind_vec = wind_vec_from_angles(v_wind_gnd, upwind_dir, 0.0)

    # ---- Settling (see examples/parking.jl) ----
    position = [cos(el_rad) * l_tether, 0.0, sin(el_rad) * l_tether]
    settle_config = V3SettleConfig(
        v_wind = v_wind_gnd,
        tether_length = l_tether,
        dt = 0.001,
        num_steps = 400,
        num_substeps = 5,
        body_damping = body_damping,
        start_depower = depower_setpoint * 100.0 + 10.0,
        course_correction_mode = :heading,
        course_correction_gain = 0.05,
        geom = gc,
        system_yaml = system_yaml,
    )
    @info "init: settling V3 model at rel_depower = $depower_setpoint..."
    sam, _, settle_failed = settle_wing(settle_config;
        position, velocity = [0.0, 0.0, 0.0], heading = 0.0,
        steering = 0.0, depower = depower_setpoint, wind_vec, remake)
    settle_failed && error("Settling failed")
    sys = sam.sys_struct

    # Un-brake the winch (the settled binary is serialized with brake=true).
    sys.winches[1].brake = false

    set = sam.set
    set.wind_vec = wind_vec
    # `set.v_depower`/`set.v_steering` come from the `kcu:` section of
    # data/settings.yaml (loaded via Settings in settle_wing).
    set.cs_4p = 1.0

    isnothing(dt) && (dt = 1 / set.sample_freq)
    isnothing(sim_time) && (sim_time = set.sim_time)

    # KCU: start at the settled depower/steering so the first step introduces
    # no geometry jump.
    kcu = KCU(set)
    kcu.set_depower = depower_setpoint
    kcu.depower = depower_setpoint
    kcu.set_steering = 0.0
    kcu.steering = 0.0

    # Winch position controller (inner speed PI, torque output).
    speed_pid = DiscretePID(; K = wc.winch_speed_k, Ti = wc.winch_speed_ti,
        Td = false, Ts = dt,
        umin = -wc.winch_torque_limit, umax = wc.winch_torque_limit)
    winch_ctrl = WinchPosController(speed_pid = speed_pid,
        kp_pos = wc.winch_pos_kp, v_sp_prev = sys.winches[1].vel)

    steps = Int(round(sim_time / dt))
    logger, sys_state = create_logger(sam, steps)

    return V3KITE(set = set, kcu = kcu, sam = sam, gc = gc, dt = dt,
        sys_state = sys_state, logger = logger, steps = steps,
        winch_ctrl = winch_ctrl)
end

"""
    step!(s::V3KITE; rel_depower=0.0, rel_steering=0.0,
          v_wind_gnd=nothing, upwind_dir=nothing,
          set_torque=nothing, set_length=nothing,
          speed_limit=Inf, acceleration_limit=Inf, prn=false)

Advance the simulation by `s.dt`, update `s.sys_state` (including
`heading_rate`), and log it.

`rel_depower` (`0..1`) and `rel_steering` (`-1..1`) are routed through the KCU
actuator model (finite tape speed) before being applied to the geometry.
`v_wind_gnd` [m/s] and/or `upwind_dir` [rad], if given, update the live wind
vector `set.wind_vec` (read by the DAE via `get_wind_vec`) before the step.

The single winch runs in exactly one mode per step: if `set_length` [m] is
given, the cascaded position controller (`speed_limit` [m/s],
`acceleration_limit` [m/s²]) converts it to a set torque; otherwise
`set_torque` [N·m] is passed through; if neither is given, the measured
holding torque is applied. `prn` logs per-step lift/drag diagnostics; progress
is reported every 100 steps.

Each logged row carries both the command and the KCU's actual (tape-lagged)
value for both channels, so plot scripts never need to re-declare a setpoint:

| quantity | commanded          | actual     |
|:---------|:-------------------|:-----------|
| steering | `set_steering`     | `steering` |
| depower  | `var_14`           | `depower`  |

Depower uses a spare slot because `SysState` has no `set_depower` field; the
actual values are filled by `update_sys_state!`. The remaining spare slots
this method fills are `var_15` (L/D_wing) and `var_16` (L/D_eff).
"""
function step!(s::V3KITE; rel_depower = 0.0, rel_steering = 0.0,
               v_wind_gnd = nothing, upwind_dir = nothing,
               set_torque = nothing, set_length = nothing,
               speed_limit = Inf, acceleration_limit = Inf, prn = false)
    dt = s.dt
    t = s.sys_state.time + dt

    # Optional live wind update (read by the DAE via get_wind_vec).
    # set.v_wind/set.upwind_dir are kept in sync with set.wind_vec by
    # KiteUtils (use_wind_vec: true), so they fill in the omitted argument.
    if v_wind_gnd !== nothing || upwind_dir !== nothing
        vw = something(v_wind_gnd, s.set.v_wind)
        ud = upwind_dir === nothing ? deg2rad(s.set.upwind_dir) : upwind_dir
        s.set.wind_vec = wind_vec_from_angles(vw, ud, 0.0)
    end

    # KCU actuator dynamics: command → finite-speed tape motion → geometry.
    # The KCU getters must be qualified: V3Kite's calibration.jl defines
    # get_depower/get_steering(sys, config), which shadow the KitePodModels
    # (kcu) methods of the same name.
    set_depower_steering(s.kcu, rel_depower, rel_steering)
    KitePodModels.on_timer(s.kcu, dt)
    dp = KitePodModels.get_depower(s.kcu)
    st = KitePodModels.get_steering(s.kcu)
    set_depower!(s.sys, dp, st, s.gc)
    set_steering!(s.sys, st, s.gc)

    # Winch: exactly one of position / torque / hold.
    torque = if set_length !== nothing
        winch_position_torque!(s, set_length, speed_limit, acceleration_limit)
    elseif set_torque !== nothing
        set_torque
    else
        force_to_torque(winch_force(s), s.sys)
    end

    if !sim_step!(s.sam; set_values = [torque], dt, vsm_interval = 1)
        error("next_step! failed at t=$(round(t, digits=3)) s")
    end

    update_sys_state!(s.sys_state, s)
    s.sys_state.time = t
    # Both channels log the command and the KCU's tape-lagged actual value.
    # `steering`/`set_steering` are the standard SysState pair; depower has no
    # `set_depower` field, so its command goes to the spare slot `var_14`.
    s.sys_state.steering = st
    s.sys_state.set_steering = rel_steering
    s.sys_state.var_14 = rel_depower
    lift, wing_drag = lift_drag(s)
    _, _, total_d = total_drag(s)
    s.sys_state.var_15 = wing_drag > 1e-6 ? lift / wing_drag : 0.0
    s.sys_state.var_16 = total_d > 1e-6 ? lift / total_d : 0.0
    log!(s.logger, s.sys_state)

    if prn
        @info "t=$(round(t, digits=2)) s" lift=round(lift, digits=1) wing_drag=round(wing_drag, digits=1) F_winch=round(winch_force(s), digits=1) v_ro=round(reel_out_speed(s), digits=3)
    end

    i = Int(round(t / dt))
    if i % 100 == 0
        now = time()
        rtf_str = isnan(s.last_step_time) ? "----" :
            @sprintf("%.2f", (100 * dt) / (now - s.last_step_time))
        @info @sprintf("step %04d / %04d, %s times realtime, lift/drag [N]: %7.2f/%7.2f",
                       i, s.steps, rtf_str, lift, wing_drag)
        s.last_step_time = now
    end
    return nothing
end