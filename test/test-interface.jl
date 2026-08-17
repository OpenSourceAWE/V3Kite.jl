# Copyright (c) 2025 Jelle Poland, Bart van de Lint, Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

using Pkg
if ! ("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test
using LinearAlgebra
using V3Kite
using KitePodModels: KCU
# V3Kite is torque-only; the winch length loop is the caller's.
isdefined(@__MODULE__, :hold_torque!) ||
    include(joinpath(@__DIR__, "winch_hold_stub.jl"))

@testset "Interface Functions" begin
    data_path = v3_data_path()
    config = V3SimConfig(
        struc_yaml_path   = "struc_geometry.yaml",
        aero_yaml_path    = "aero_geometry.yaml",
        vsm_settings_path = "vsm_settings.yaml",
        v_wind            = 10.0,
        tether_length     = 150.0,
    )
    sam, sys = create_v3_model(config; data_path)
    sam.set.wind_vec = [10.0, 0.0, 0.0]
    init!(sam; remake=false, remake_vsm=true)
    sys.winches[1].brake = true
    kcu = KCU(sam.set)
    v3kite = V3KITE(set=sam.set, kcu=kcu, sam=sam)

    @testset "lift_drag" begin
        @test sim_step!(sam; set_values=[0.0], dt=1/60, vsm_interval=1)
        lift, drag = lift_drag(v3kite)
        @test isfinite(lift)
        @test isfinite(drag)
        @test lift > 0.0
        @test drag > 0.0
        @test lift > drag  # kites typically have L/D > 1
    end

    @testset "unstretched_length" begin
        len = unstretched_length(v3kite)
        @test isfinite(len)
        @test len > 0.0
        @test len ≈ config.tether_length  # should match configured tether length
    end

    @testset "v_wind_kite" begin
        v = v_wind_kite(v3kite)
        @test length(v) == 3
        @test all(isfinite, v)
        @test norm(v) > 0.0          # non-zero wind
        @test norm(v) ≈ config.v_wind  # profile_law=0: constant wind, factor=1
    end

    @testset "pos_kite" begin
        pos = pos_kite(v3kite)
        @test length(pos) == 3
        @test all(isfinite, pos)
        @test pos[3] > 0.0           # kite is above ground
    end

    @testset "calc_height" begin
        h = calc_height(v3kite)
        @test isfinite(h)
        @test h > 0.0                # kite is above ground
        @test h ≈ pos_kite(v3kite)[3]  # consistent with pos_kite z-component
    end

    @testset "calc_elevation" begin
        el = calc_elevation(v3kite)
        @test isfinite(el)
        @test el > 0.0               # kite is above the horizon
        @test el < π/2              # elevation below 90°
    end

    @testset "upwind_dir" begin
        dir = upwind_dir(v3kite)
        @test isfinite(dir)
        @test dir ≈ -π/2  # wind_vec=[10,0,0] → wind_dir=0 → upwind_dir=-(0+π/2)
        # zero wind returns NaN
        @test isnan(upwind_dir([0.0, 0.0, 0.0]))
    end

    @testset "tether_length" begin
        len = tether_length(v3kite)
        @test isfinite(len)
        @test len > 0.0
        @test len ≈ unstretched_length(v3kite) rtol=0.01  # close to unstretched at rest
    end

    @testset "calc_azimuth" begin
        az = calc_azimuth(v3kite)
        @test isfinite(az)
        @test -π <= az <= π
    end

    @testset "calc_azimuth_east" begin
        az = calc_azimuth_east(v3kite)
        @test isfinite(az)
        @test -π <= az <= π
    end

    @testset "calc_azimuth_north" begin
        az = calc_azimuth_north(v3kite)
        @test isfinite(az)
        @test -π <= az <= π
    end

    @testset "kite_ref_frame" begin
        x, y, z = kite_ref_frame(v3kite)
        @test length(x) == 3 && length(y) == 3 && length(z) == 3
        @test all(isfinite, x) && all(isfinite, y) && all(isfinite, z)
        @test norm(x) ≈ 1.0 atol=1e-10  # unit vectors
        @test norm(y) ≈ 1.0 atol=1e-10
        @test norm(z) ≈ 1.0 atol=1e-10
        @test dot(x, y) ≈ 0.0 atol=1e-10  # orthogonal
        @test dot(x, z) ≈ 0.0 atol=1e-10
        @test dot(y, z) ≈ 0.0 atol=1e-10
    end

    @testset "calc_orient_quat" begin
        q = calc_orient_quat(v3kite)
        @test length(q) == 4
        @test all(isfinite, q)
        @test norm(q) ≈ 1.0 atol=1e-10  # unit quaternion
    end

    @testset "orient_euler" begin
        rpy = orient_euler(v3kite)
        @test length(rpy) == 3
        @test all(isfinite, rpy)
    end

    @testset "calc_heading" begin
        heading = calc_heading(v3kite)
        @test isfinite(heading)
        @test abs(heading) <= π + 1e-3
    end

    @testset "calc_course" begin
        course = calc_course(v3kite)
        @test isfinite(course) || isnan(course)  # NaN allowed if velocity ≈ 0
    end

    @testset "cl_cd" begin
        cl, cd = cl_cd(v3kite)
        @test isfinite(cl) && isfinite(cd)
        @test cl > 0.0 && cd > 0.0
        @test cl > cd  # kites typically have CL/CD > 1
    end

    @testset "span_mean_aoa" begin
        # NaN would mean the VSM engine was not FOUND, not an unloaded wing: it
        # hangs off `wing.aero` and is only reached through `getproperty`
        # forwarding, so any feature test on the wing itself silently disables this.
        alpha = span_mean_aoa(sys)
        @test isfinite(alpha)
        @test abs(alpha) < deg2rad(45)
        # The VSM's own span mean, over effective instead of geometric AoA:
        # the two differ by the induced angle, a degree or so here.
        @test abs(alpha - sys.wings[1].vsm_aoa) < deg2rad(10)
    end

    @testset "winch_force" begin
        f = winch_force(v3kite)
        @test isfinite(f)
        @test f >= 0.0
    end

    @testset "reel_out_speed" begin
        v = reel_out_speed(v3kite)
        @test isfinite(v)
        @test v ≈ 0.0  # brake is on
    end

    @testset "states" begin
        n = states(v3kite)
        @test n > 0
    end

    @testset "spring_forces" begin
        sf = spring_forces(v3kite)
        @test length(sf) > 0
        @test all(isfinite, sf)
    end
end

@testset "init / step! Interface" begin
    # Mirrors examples/simple_parking.jl's init() call so it hits the
    # data/settled_*.bin cache (see stabilization.jl) instead of resettling.
    PROJECT          = "system_cabauw.yaml"
    V_WIND           = 10.0
    TETHER_LENGTH    = 150.0
    DEPOWER_SETPOINT = 0.25
    SIM_TIME         = 1.0

    s = init(V_WIND, TETHER_LENGTH;
        depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, system_yaml = PROJECT)

    @testset "init" begin
        @test s isa V3KITE
        @test s.dt > 0.0
        @test s.steps == round(Int, SIM_TIME / s.dt)
        @test s.sys.winches[1].brake == false  # un-braked, ready for step!
        @test s.kcu.depower ≈ DEPOWER_SETPOINT
        @test s.kcu.steering ≈ 0.0
        @test s.sys_state.time == 0.0
        # No winch controller on the model any more: V3Kite is the plant, and
        # `step!` takes a torque. The loop is the caller's (winch_hold_stub.jl).
        @test !hasproperty(s, :winch_ctrl)
        # `winch: max_acc:` of data/settings.yaml, read by the caller's rate limiter.
        @test s.set.max_acc > 0.0
    end

    @testset "init default min_damping" begin
        # Settling decays `body_damping` linearly to the `min_damping` floor and
        # the returned model runs with that floor, so the default `min_damping`
        # is readable off the settled points: 0.8 .* the default body_damping.
        expected = 0.8 .* [0.0, 0.0, 40.0]
        tether_pts = Set(tether_point_idxs(s.sys))
        wing_pts = [i for i in eachindex(s.sys.points) if !(i in tether_pts)]
        @test !isempty(wing_pts)
        @test all(i -> s.sys.points[i].body_frame_damping ≈ expected, wing_pts)
        # Tether points are skipped by `set_body_frame_damping!`, floor or not.
        @test all(i -> all(iszero, s.sys.points[i].body_frame_damping), tether_pts)
    end

    l0 = unstretched_length(s)
    # The winch length loop is the caller's now; one per model.
    lhc = length_hold_controller(s)

    @testset "step! holding torque (no set_torque)" begin
        t0 = s.sys_state.time
        step!(s; rel_depower = DEPOWER_SETPOINT)
        @test s.sys_state.time ≈ t0 + s.dt
        @test isfinite(winch_force(s))
    end

    @testset "step! torque mode (set_torque)" begin
        t0 = s.sys_state.time
        step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = 50.0)
        @test s.sys_state.time ≈ t0 + s.dt
        @test isfinite(reel_out_speed(s))
    end

    @testset "step! position mode (caller's length loop)" begin
        t0 = s.sys_state.time
        # Three steps: the previous torque-mode step left residual winch speed,
        # and the position PI loop (Ti = 2s) needs more than one 0.05s step to
        # bring it back down. The decay from that transient is smooth and
        # monotone (1.31, 1.06, 0.89, 0.73, ... m/s), so two steps sit right on
        # the 1.0 bound below — the third gives it margin.
        step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = hold_torque!(lhc, s, l0))
        step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = hold_torque!(lhc, s, l0))
        step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = hold_torque!(lhc, s, l0))
        @test s.sys_state.time ≈ t0 + 3 * s.dt
        @test isfinite(unstretched_length(s))
        @test abs(reel_out_speed(s)) < 1.0  # holding l0: speed setpoint ≈ 0
    end

    @testset "step! logged L/D" begin
        # The parked wing is loaded, so a NaN would mean the floor sits too high.
        @test isfinite(s.sys_state.var_15)
        @test isfinite(s.sys_state.var_16)
        @test s.sys_state.var_15 > 0.0
        # Effective L/D counts tether drag too, so it is the smaller of the two.
        @test s.sys_state.var_16 < s.sys_state.var_15
        @test drag_floor(s.sam) > 0.0
    end

    @testset "step! vsm_interval" begin
        t0 = s.sys_state.time
        # Not the default: the aero load is held frozen between VSM solves.
        step!(s; rel_depower = DEPOWER_SETPOINT,
              set_torque = hold_torque!(lhc, s, l0), vsm_interval = 2)
        @test s.sys_state.time ≈ t0 + s.dt
        @test isfinite(winch_force(s))
        @test isfinite(unstretched_length(s))
    end

    @testset "step! live wind update" begin
        t0 = s.sys_state.time
        step!(s; rel_depower = DEPOWER_SETPOINT,
              set_torque = hold_torque!(lhc, s, l0), v_wind_gnd = 12.0)
        @test s.sys_state.time ≈ t0 + s.dt
        # The mean, not `set.wind_vec`: turbulence borrows the latter across the solve.
        @test norm(s.wind_vec_mean) ≈ 12.0
    end

    @testset "init damping_per_stiffness" begin
        # Deliberately below MIN_SETTLE_DAMPING_PER_STIFFNESS: settling runs at
        # the floor and `init` applies the flown ratio to the settled structure
        # afterwards, so the returned model must carry the flown one, not the
        # floor. Settles once (the ratio enters the cache key) and is not
        # stepped; only the structural damping is under test here.
        RATIO = 0.001
        @test RATIO < V3Kite.MIN_SETTLE_DAMPING_PER_STIFFNESS
        s_dps = init(V_WIND, TETHER_LENGTH;
            depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME,
            system_yaml = PROJECT, damping_per_stiffness = RATIO)

        seg_idxs = tether_bridle_segments(s_dps.sys)
        @test !isempty(seg_idxs)
        # Segments with a callable force law have no stiffness to scale.
        damped = [i for i in seg_idxs
                  if s_dps.sys.segments[i].unit_stiffness isa Real]
        @test !isempty(damped)
        @test all(i -> s_dps.sys.segments[i].unit_damping ≈
                       RATIO * s_dps.sys.segments[i].unit_stiffness, damped)
        # `s` was built with the default `nothing`, which leaves the material
        # values of struc_geometry.yaml: 0.002 on the bridles and none on the
        # main tether, so every one of these segments changed.
        @test all(i -> s_dps.sys.segments[i].unit_damping !=
                       s.sys.segments[i].unit_damping, damped)
        # The wing frame keeps the damping given in struc_geometry.yaml.
        wing_segs = setdiff(collect(eachindex(s_dps.sys.segments)), seg_idxs)
        @test !isempty(wing_segs)
        @test all(i -> s_dps.sys.segments[i].unit_damping ==
                       s.sys.segments[i].unit_damping, wing_segs)
    end
end
nothing
