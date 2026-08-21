import SymbolicAWEModels: winch_force, tether_length
using Printf

"""
    V3KITE <: AbstractKiteModel

The plant. Winch controllers live in WinchControllers.jl; `step!` here takes a
torque. What remains is the plant side they read: `drum_params`, `winch_force`,
`reel_out_speed`, `unstretched_length` and `force_to_torque`.

The KCU tape rate limits (`v_depower`/`v_steering`) live in the `kcu:` section
of `data/sim_settings_default.yaml`, loaded like any other `Settings` field.
"""
@with_kw mutable struct V3KITE <: AbstractKiteModel
    "Reference to the settings struct"
    set::Settings
    "Reference to the KCU model (Kite Control Unit) as implemented in the package KitePodModels"
    kcu::KCU
    sam::SymbolicAWEModel
    """
    Reference to the atmospheric model as implemented in the package AtmosphericModels.

    Shared with the DAE: this is `sam.sys_struct.am`, the same instance the generated
    equations evaluate via `calc_wind_factor`/`calc_rho`. Constructing a second one here
    would load a second copy of the turbulent wind field (~1.2 GB) when
    `set.use_turbulence > 0`.
    """
    am::AtmosphericModel = sam.am
    "Geometry config used by the depower/steering tape conversions"
    gc::V3GeomAdjustConfig = V3GeomAdjustConfig()
    """
    Commanded mean ground wind vector [m/s]; `set.wind_vec` equals this except across the
    solver call, where [`apply_turbulence!`](@ref) replaces it by the turbulent wind.
    """
    wind_vec_mean::KiteUtils.SVec3 = zeros(KiteUtils.SVec3)
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

"""
    calc_orient_quat(s::V3KITE; viewer=false) -> Vector{Float64}

Return the kite orientation as a quaternion `[w, x, y, z]`. With `viewer=true`,
build it from a camera-style triad (kite position looking towards `z`, up
`-x`) instead of the body axes rotated into NED.
"""
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

Determine the heading angle of the kite in radian. Adds a `π` correction
because `SymbolicAWEModels`' body x-axis points opposite to the Xsens-sensor
convention `KiteUtils.calc_heading` assumes — the same correction
`calc_csv_heading` applies to EKF-derived orientation.
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
    drum_params(s::V3KITE) -> (drum_radius, gear_ratio, friction)

The first winch's drum geometry and its CURRENT friction torque, the three plant
numbers WinchControllers.jl's torque controllers need. `friction` is a runtime
value, not a setting: it flips sign with the reel-out direction.
"""
function drum_params(s::V3KITE)
    winch = s.sys.winches[1]
    return winch.drum_radius, winch.gear_ratio, winch.friction
end

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

"""
    init(v_wind_gnd, l_tether; elevation=nothing, upwind_dir=-π/2,
         depower_setpoint=0.25, dt=nothing, sim_time=nothing,
         gc=nothing, body_start_damping=[0.0, 0.0, 40.0],
         body_sim_damping=nothing, damping_per_stiffness=nothing,
         aero_mode=nothing, data_path=v3_data_path(), cache_path=nothing,
         use_turbulence=nothing, warmup_time=0.0, warmup_torque=nothing,
         remake_model=nothing, remake_settled_state=nothing) -> V3KITE

Build and return a ready `V3KITE`, settled at a fixed depower equilibrium,
for a `step!` simulation loop (see `examples/simple_parking.jl`).

`v_wind_gnd` is the ground wind speed at the reference height [m/s] and
`l_tether` the initial tether length [m]. `elevation` [deg] falls back to the
settings value; `upwind_dir` [rad] sets the wind direction. `depower_setpoint`
is the initial `rel_depower` in `[0, 1]` (not meters). `dt` [s] and `sim_time`
[s] size the logger and step count, falling back to `1/set.sample_freq` and
`set.sim_time`. `gc` holds the geometry adjustments;
`remake_settled_state=true` forces re-settling (ignoring the
`data/settled_*.arrow` state) and `remake_model=true` rebuilds the serialized
equations, both defaulting to the project's own flags. There is no winch-gain
argument: V3Kite owns no winch controller (see [`step!`](@ref)).
`system_yaml` names the project file (default `"system_psm.yaml"`) that points at
the settings, geometry and kite-settings files; pass e.g.
`"system_beam.yaml"` to fly another kite. `aero_mode` and `gc` override
the aerodynamics and the geometry adjustments of that project's
`kite_settings:` file; `nothing` (the default) takes both from it.

`data_path` is the directory the geometry/settings YAMLs are READ from,
defaulting to the bundled [`v3_data_path`](@ref); `cache_path` is where
everything generated is WRITTEN — the settled-geometry cache
(`settled_*.arrow`), the settling log, and the serialized model binary
(`model_*.bin`).

Redirecting `data_path` as well means that directory must hold the source
geometry too (`struc_geometry.yaml`, `cfd_aero_geometry.yaml`, `vsm_settings.yaml`,
the system YAML and the settings file it names), since the settling stage reads
all of them from it.

The turbulence level comes from `data/gui.yaml` (see
[`get_default_turbulence`](@ref)), a per-checkout preference applied before the
wind-field decision above; its `"default"` keyword leaves the settings YAML in
charge. `use_turbulence` overrides that file: pass a level in `[0.0, 1.0]`, or
the `"default"` keyword to keep the settings YAML. It is what a package driving
V3Kite uses to keep the preference in its OWN `data/`, since `data_path` must
be the directory holding the model geometry and is read-only for an installed
V3Kite.

`body_start_damping` is the per-axis body-frame damping `[x, y, z]` in the wing
frame, applied to every point during settling. It acts on point velocity
*relative to the wing*, so it damps bridle vibration without slowing the kite's
global motion (unlike world-frame damping). It is the value settling *starts*
from: over `decay_steps` it decays linearly to `body_sim_damping`. It is baked
into the settled geometry, so it forms part of the settling cache key — changing
it produces a different `data/settled_*.arrow` rather than silently reusing the
old one.

`body_sim_damping` overrides the `body_sim_damping:` of the project's
`kite_settings:` file, and is the damping the RETURNED MODEL FLIES WITH —
`body_start_damping` only shapes the settling transient above it. Two
consequences: any `body_start_damping` above it gives the same flown damping and
differs only in the settled geometry it converges to, and an in-plane
`body_start_damping` decays to zero in flight unless `body_sim_damping` carries
the in-plane terms too. It is part of the cache key as well. Runs identified
before 2026-08-08 saw no floor at all (`[0, 0, 0]`) and settled 400 steps of a
2000-step ramp, ending at `0.8 * body_start_damping`; reproducing such a run
means passing `body_sim_damping = 0.8 .* body_start_damping`.

The default damps only normal to the wing surface. The in-plane (x, y) terms
trade accuracy for solver cost: `[10, 10, ...]` cuts parked AoA ripple and solver
steps several-fold at unchanged settled trim, but it also resists the deformation
that produces steering and so **reduces the turn rate**. Use in-plane damping for
parked/quasi-static runs, not for validating turning maneuvers, and note that
settling goes unstable somewhere above 15. See
PlanSuppressOscillations.md for the sweep.

`damping_per_stiffness` [s] sets the structural damping of the tether and bridle
segments as a ratio of their stiffness, `unit_damping = ratio * unit_stiffness`
(see [`set_damping_per_stiffness!`](@ref)); it overrides the material value of
`data/struc_geometry.yaml`, while the wing frame keeps the damping given there.
`nothing`, the default, leaves the segments as loaded: the bridles at the
material value (0.002 for `dyneema`) and the main tether undamped.

`warmup_time` [s] runs the returned model forward that long with the controls
held at the settled values and then discards those steps, so the run does not
start on the settling-to-dynamics transient; `0.0`, the default, skips it.
`warmup_torque` is the winch-torque callback it relaxes against and must match
what the run will command. Both are documented under [`warmup!`](@ref).

The settling stage mirrors `examples/parking.jl`. The KCU actuator model, the
logger, `sys_state`, and `steps` are stored on the returned instance, and the
winch is left un-braked.
"""
function init(v_wind_gnd, l_tether;
              elevation = nothing,
              upwind_dir = -π/2,
              depower_setpoint = 0.25,
              dt = nothing,
              sim_time = nothing,
              gc = nothing,
              system_yaml = "system_psm.yaml",
              body_start_damping = [0.0, 0.0, 40.0],
              body_sim_damping = nothing,
              damping_per_stiffness = nothing,
              aero_mode = nothing,
              data_path = v3_data_path(),
              cache_path = nothing,
              use_turbulence = nothing,
              warmup_time = 0.0,
              warmup_torque = nothing,
              remake_model = nothing,
              remake_settled_state = nothing)
    system_path = joinpath(data_path, system_yaml)
    if isnothing(elevation)
        elevation = Settings(system_path).elevation
    end
    el_rad = deg2rad(elevation)
    wind_vec = wind_vec_from_angles(v_wind_gnd, upwind_dir, 0.0)

    position = [cos(el_rad) * l_tether, 0.0, sin(el_rad) * l_tether]
    kite_set = load_kite(system_yaml; data_path)
    isnothing(aero_mode) || (kite_set.aero_mode = aero_mode)
    isnothing(gc) || (kite_set.geom = gc)
    isnothing(body_sim_damping) ||
        (kite_set.body_sim_damping = collect(Float64, body_sim_damping))
    isnothing(remake_model) && (remake_model = kite_set.remake_model)
    isnothing(remake_settled_state) &&
        (remake_settled_state = kite_set.remake_settled_state)
    settle_config = V3SettleConfig(
        project = system_yaml,
        kite_set = kite_set,
        v_wind = v_wind_gnd,
        tether_length = l_tether,
        dt = 0.05,
        num_steps = 40,
        num_substeps = 1,
        decay_steps = 30,
        body_start_damping = body_start_damping,
        damping_per_stiffness = damping_per_stiffness,
        start_depower = depower_setpoint * 100.0 + 10.0,
        course_correction_mode = :heading,
        course_correction_gain = 0.05,
    )
    @info "init: settling V3 model at rel_depower = $depower_setpoint..."
    sam, _, settle_failed = settle_wing(settle_config;
        position, velocity = [0.0, 0.0, 0.0], heading = 0.0,
        steering = 0.0, depower = depower_setpoint, wind_vec,
        data_path, cache_path, remake_model, remake_settled_state)
    settle_failed && error("Settling failed")
    sys = sam.sys_struct

    # Settling ran at `MIN_SETTLE_DAMPING_PER_STIFFNESS` if the flown ratio is
    # below it; set the flown one here, on a structure already at rest. Also
    # covers a cache hit, whose `.bin` is keyed by the floored value alone.
    if !isnothing(damping_per_stiffness)
        set_damping_per_stiffness!(sys, tether_bridle_segments(sys),
                                   damping_per_stiffness)
    end

    # The serialized `set_value` is stale: releasing the brake alone steps at t = 0.
    sys.winches[1].brake = false
    init_winch_torque!(sys)

    set = sam.set
    set.wind_vec = wind_vec
    set.cs_4p = 1.0

    turb = something(use_turbulence, get_default_turbulence(data_path), Some(nothing))
    turb isa Real && (set.use_turbulence = turb)

    sam.am.set = set
    clear(sam.am)
    sam.am.wf = set.use_turbulence > 0 ? WindField(sam.am, set.v_wind) : nothing

    isnothing(dt) && (dt = 1 / set.sample_freq)
    isnothing(sim_time) && (sim_time = set.sim_time)

    # Start at the settled depower/steering so the first step introduces no geometry jump.
    kcu = KCU(set)
    kcu.set_depower = depower_setpoint
    kcu.depower = depower_setpoint
    kcu.set_steering = 0.0
    kcu.steering = 0.0

    steps = Int(round(sim_time / dt))
    logger, sys_state = create_logger(sam, steps)

    s = V3KITE(set = set, kcu = kcu, sam = sam, gc = kite_set.geom, dt = dt,
        sys_state = sys_state, logger = logger, steps = steps,
        wind_vec_mean = wind_vec)

    warmup_time > 0 && warmup!(s, warmup_time;
                               depower = depower_setpoint,
                               winch_torque = warmup_torque)
    return s
end

"""
    apply_turbulence!(s::V3KITE) -> Bool

Overwrite `s.set.wind_vec` with the turbulent wind sampled at the first wing, and return
`true` if it did; [`step!`](@ref) calls this immediately before the integration step and
must restore `s.wind_vec_mean` afterwards. Returns `false` without touching anything when
turbulence is off or the mean wind has no horizontal component.

`set.wind_vec` is the per-step wind hook that reaches the whole system: it becomes an MTK
parameter `next_step!` re-syncs every step, and the point apparent wind is built from it.

The sampled wind already contains the mean at the kite height, so it is divided by
`calc_wind_factor` at that height before going back into the ground vector — the DAE
multiplies it in again. With `profile_law: 0` the factor is 1 and the division is a no-op.

The value is held constant over the step on purpose: `AtmosphericModels.get_wind` is a
nearest-grid-point lookup into the Mann field, a discontinuous term if it were evaluated
inside the implicit solve.
"""
function apply_turbulence!(s::V3KITE)
    s.set.use_turbulence > 0 || return false
    ud = upwind_dir(s.wind_vec_mean)
    isfinite(ud) || return false
    pos = s.sys.wings[1].pos_w
    v_turb, _ = calc_turbulent_wind(s.am, pos, s.sys_state.time; upwind_dir = ud)
    # 10 m clamp mirrors the one calc_turbulent_wind/get_wind apply internally.
    s.set.wind_vec = v_turb / calc_wind_factor(s.am, max(pos[3], 10.0))
    return true
end

"""
    step!(s::V3KITE; rel_depower=0.0, rel_steering=0.0,
          v_wind_gnd=nothing, upwind_dir=nothing,
          set_torque=nothing, vsm_interval=1, prn=false)

Advance the simulation by `s.dt`, update `s.sys_state` (including
`heading_rate`), and log it.

`rel_depower` (`0..1`) and `rel_steering` (`-1..1`) are routed through the KCU
actuator model (finite tape speed) before being applied to the geometry.
`v_wind_gnd` [m/s] and/or `upwind_dir` [rad], if given, update the live wind
vector `set.wind_vec` (read by the DAE via `get_wind_vec`) before the step;
either may be omitted since KiteUtils keeps `set.v_wind`/`set.upwind_dir` in
sync with `set.wind_vec` (`use_wind_vec: true`), filling in whichever is
missing. That mean is also kept on `s.wind_vec_mean`, because
[`apply_turbulence!`](@ref) borrows `set.wind_vec` for the duration of the
solver call and a `finally` puts the mean back — including when the step
fails, so no gust is ever latched into the settings. The restore happens
*after* `update_sys_state!`, so the logged `v_wind_gnd` is the instantaneous
wind the kite saw.

The winch takes a torque: `set_torque` [N·m] is applied as given, and omitting
it applies the measured holding torque, i.e. the drum holds whatever force the
tether currently pulls. There is no `set_length` — turning a length setpoint
into a torque is a controller, which belongs to the caller; use
WinchControllers.jl's `winch_position_torque!` fed with `drum_params(s)`, as
`examples/winch_adapter.jl` does. `prn` logs per-step lift/drag diagnostics;
progress is reported every 200 steps.

`vsm_interval` is forwarded to `sim_step!`: the VSM aero load is recomputed
every `vsm_interval` steps and held frozen inside the DAE in between (`0`
disables the VSM update entirely). The default `1` is the tightest coupling
available, so raising it only increases the aero lag — it is exposed for
sweeps/speed trade-offs, not as a way to stabilize the explicit coupling.

Each logged row carries both the command and the KCU's actual (tape-lagged)
value for both channels, so plot scripts never need to re-declare a setpoint:

| quantity | commanded          | actual     |
|:---------|:-------------------|:-----------|
| steering | `set_steering`     | `steering` |
| depower  | `var_14`           | `depower`  |

Depower uses a spare slot because `SysState` has no `set_depower` field; the
actual values are filled by `update_sys_state!`. The other spare slots filled
here are `var_15` (L/D_wing) and `var_16` (L/D_eff), both `NaN` while the wing
is unloaded, i.e. below [`drag_floor`](@ref).
"""
function step!(s::V3KITE; rel_depower = 0.0, rel_steering = 0.0,
               v_wind_gnd = nothing, upwind_dir = nothing,
               set_torque = nothing, vsm_interval = 1, prn = false)
    dt = s.dt
    t = s.sys_state.time + dt

    if v_wind_gnd !== nothing || upwind_dir !== nothing
        vw = something(v_wind_gnd, s.set.v_wind)
        ud = upwind_dir === nothing ? deg2rad(s.set.upwind_dir) : upwind_dir
        s.set.wind_vec = wind_vec_from_angles(vw, ud, 0.0)
        s.wind_vec_mean = s.set.wind_vec
    end

    set_depower_steering(s.kcu, rel_depower, rel_steering)
    KitePodModels.on_timer(s.kcu, dt)
    # Qualified: calibration.jl's get_depower/get_steering(sys, config) would shadow these.
    dp = KitePodModels.get_depower(s.kcu)
    st = KitePodModels.get_steering(s.kcu)
    set_depower!(s.sys, dp, st, s.gc)
    set_steering!(s.sys, st, s.gc)

    # Winch: a torque, or hold the measured force. A LENGTH setpoint is not a
    # plant input — converting one into a torque needs a controller, which this
    # package deliberately does not own; see the docstring.
    torque = isnothing(set_torque) ? force_to_torque(winch_force(s), s.sys) :
                                     set_torque

    turbulent = apply_turbulence!(s)
    try
        if !sim_step!(s.sam; set_values = [torque], dt, vsm_interval)
            error("next_step! failed at t=$(round(t, digits=3)) s")
        end
        update_sys_state!(s.sys_state, s)
    finally
        turbulent && (s.set.wind_vec = s.wind_vec_mean)
    end
    s.sys_state.time = t
    s.sys_state.steering = st
    s.sys_state.set_steering = rel_steering
    s.sys_state.var_14 = rel_depower
    lift, wing_drag = lift_drag(s)
    _, _, total_d = total_drag(s)
    # Below the floor the wing is unloaded and the ratio is a spike; see `drag_floor`.
    d_min = drag_floor(s.sam)
    s.sys_state.var_15 = wing_drag > d_min ? lift / wing_drag : NaN
    s.sys_state.var_16 = total_d > d_min ? lift / total_d : NaN
    log!(s.logger, s.sys_state)

    if prn
        @info "t=$(round(t, digits=2)) s" lift=round(lift, digits=1) wing_drag=round(wing_drag, digits=1) F_winch=round(winch_force(s), digits=1) v_ro=round(reel_out_speed(s), digits=3)
    end

    i = Int(round(t / dt))
    if i % 200 == 0
        now = time()
        # Widths fixed so the column does not jog: step counts reach 5 digits and
        # the realtime factor 3 before the point.
        rtf_str = isnan(s.last_step_time) ? "  ----" :
            @sprintf("%6.2f", (200 * dt) / (now - s.last_step_time))
        @info @sprintf("step %5d / %5d, %s times realtime, lift/drag [N]: %7.2f/%7.2f",
                       i, s.steps, rtf_str, lift, wing_drag)
        s.last_step_time = now
    end
    return nothing
end

"""
    warmup!(s::V3KITE, warmup_time; depower=0.0, winch_torque=nothing) -> nothing

Relax `s` into an equilibrium of ITS OWN model, then throw the relaxation away:
step the model forward `warmup_time` seconds with zero steering, the depower
held at `depower` and the winch in the mode the run will use, and afterwards
replace the logger and `sys_state` so the run's first logged row is again
`t = 0`. Called by [`init`](@ref) when `warmup_time > 0`.

`settle_wing` returns an equilibrium of the SETTLING model (heavily
damped, winch braked), which is not a fixed point of the model the run
integrates. Without a warm-up that difference is a decaying transient over the
first second of every log, sharpest in the logged L/D. This is real integration,
not a re-settle: it costs `warmup_time / dt` full steps, and a diverging model
diverges here too.

`winch_torque` MUST drive the drum the same way the run will afterwards, or the
warm-up reintroduces the discontinuity it exists to remove. It is a callback
`(s, l_hold) -> torque [N·m]`, called once per warm-up step with the length the
warm-up is holding; `nothing` (the default) holds the drum at the measured
tether force, which is what a run that commands no torque of its own gets.

A caller running a winch controller passes a closure over it, e.g.
`(s, l) -> winch_torque!(wpc, s, l)` with `examples/winch_adapter.jl`.

The integrator clock keeps running; only the logged time restarts. Progress lines
printed during the warm-up count against `s.steps` and can be ignored.
"""
function warmup!(s::V3KITE, warmup_time; depower = 0.0, winch_torque = nothing)
    n = round(Int, warmup_time / s.dt)
    n < 1 && return nothing
    if n > s.steps
        # The warm-up logs through the run's logger, which holds `steps + 1` rows.
        @warn "warmup_time is longer than the whole run; clamping to sim_time."
        n = s.steps
    end
    @info @sprintf("warmup: %.2f s (%d steps) with the winch %s...", warmup_time, n,
                   isnothing(winch_torque) ? "holding the measured force" :
                                             "under the caller's controller")
    # The run re-references to wherever the warm-up leaves the length.
    l_hold = unstretched_length(s)
    for _ in 1:n
        set_torque = isnothing(winch_torque) ? nothing : winch_torque(s, l_hold)
        step!(s; rel_depower = depower, rel_steering = 0.0, set_torque)
    end
    logger, sys_state = create_logger(s.sam, s.steps)
    s.logger = logger
    s.sys_state = sys_state
    s.sys_state.time = 0.0
    s.last_step_time = NaN
    return nothing
end
