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
isdefined(@__MODULE__, :winch_torque!) ||
    include(joinpath(@__DIR__, "..", "examples", "winch_adapter.jl"))

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

@testset "WCSettings" begin
    # The struct and its controllers live in WinchControllers.jl now; V3Kite
    # reads no winch gains at all. What this checks is that the file V3Kite
    # SHIPS still loads into that struct and still carries the V3 tuning.
    set_data_path(v3_data_path())
    default = WCSettings()
    loaded = load_wc_settings("wc_settings.yaml"; dt = 0.05)

    @test loaded.dt == 0.05                      # dt always wins over the file
    @test loaded.winch_pos_kp == 0.5
    @test loaded.winch_speed_k == 30.0
    @test loaded.winch_speed_ti == 2.0
    @test loaded.winch_torque_limit == 500.0
    # Default feed-forward is exact, i.e. a perfectly stiff drum.
    @test loaded.winch_ff_scale == 1.0
    # Force mode; unused while the drum holds a length.
    @test loaded.winch_force_tau == 10.0
    @test loaded.winch_len_kp == 100.0
    @test loaded.winch_damp == 500.0
    @test loaded.winch_force_min == 100.0
    # The shipped file spells the torque half out, so it doubles as the list.
    @test loaded.winch_pos_kp == default.winch_pos_kp

    # The MERGED half: WinchControllers' own speed-controller fields ride along
    # in the same struct and keep their defaults, since V3Kite's file omits them.
    @test loaded.kv == default.kv
    @test loaded.f_low == default.f_low
    @test loaded.mode == default.mode

    # Type-aware loading, and an unknown key is a typo rather than a default.
    mktempdir() do dir
        good = joinpath(dir, "good.yaml")
        write(good, "wc_settings:\n  winch_pos_kp: 0.75\n  mode: \"reelout\"\n")
        wcs = load_wc_settings(good; dt = 0.02)
        @test wcs.winch_pos_kp == 0.75
        @test wcs.mode == "reelout"                  # a String field survives
        @test wcs.winch_damp == default.winch_damp   # omitted key keeps default
        bad = joinpath(dir, "bad.yaml")
        write(bad, "wc_settings:\n  winch_pos_kpp: 0.75\n")
        @test_throws ErrorException load_wc_settings(bad; dt = 0.02)
    end

    # Starts with no reference force, so the first step produces no torque jump.
    wfc = WinchForceController(loaded)
    @test wfc.force_tau == loaded.winch_force_tau
    @test wfc.len_kp == loaded.winch_len_kp
    @test wfc.damp == loaded.winch_damp
    @test wfc.force_min == loaded.winch_force_min
    @test isnan(wfc.f_lpf)
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
        # `step!` takes a torque. The loop is the caller's (winch_adapter.jl).
        @test !hasproperty(s, :winch_ctrl)
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
    wpc = WinchPosController(load_wc_settings("wc_settings.yaml"; dt = s.dt); dt = s.dt)

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
        step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = winch_torque!(wpc, s, l0))
        step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = winch_torque!(wpc, s, l0))
        step!(s; rel_depower = DEPOWER_SETPOINT, set_torque = winch_torque!(wpc, s, l0))
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
              set_torque = winch_torque!(wpc, s, l0), vsm_interval = 2)
        @test s.sys_state.time ≈ t0 + s.dt
        @test isfinite(winch_force(s))
        @test isfinite(unstretched_length(s))
    end

    @testset "step! live wind update" begin
        t0 = s.sys_state.time
        step!(s; rel_depower = DEPOWER_SETPOINT,
              set_torque = winch_torque!(wpc, s, l0), v_wind_gnd = 12.0)
        @test s.sys_state.time ≈ t0 + s.dt
        # The mean, not `set.wind_vec`: turbulence borrows the latter across the solve.
        @test norm(s.wind_vec_mean) ≈ 12.0
    end

    @testset "winch length loop speed feed-forward" begin
        # The loop under test is WinchControllers.jl's, reached through the
        # adapter; `wpc.v_sp_prev` reads back the resulting speed setpoint.
        ctrl = wpc
        v_sp_saved = ctrl.v_sp_prev
        # The model is not stepped in here, so the length error is exactly what
        # each call is given, and `v_sp_prev` reads back the resulting setpoint.
        l_now = unstretched_length(s)
        # A huge acceleration_limit keeps the rate limiter out of the way; it is
        # `v_sp_prev = 0` before each call that makes the readback meaningful.
        acc = 1e6

        # Default: pure feedback, identical to before the kwarg existed.
        ctrl.v_sp_prev = 0.0
        winch_torque!(wpc, s, l_now; speed_limit = 8.0, acceleration_limit = acc)
        @test ctrl.v_sp_prev ≈ 0.0 atol = 1e-12

        # No length error: the setpoint IS the feed-forward.
        ctrl.v_sp_prev = 0.0
        winch_torque!(wpc, s, l_now; v_ff = 1.5, speed_limit = 8.0,
                      acceleration_limit = acc)
        @test ctrl.v_sp_prev ≈ 1.5

        # The outer P loop ADDS to the feed-forward, it does not replace it.
        ctrl.v_sp_prev = 0.0
        winch_torque!(wpc, s, l_now + 2.0; v_ff = 1.5, speed_limit = 8.0,
                      acceleration_limit = acc)
        @test ctrl.v_sp_prev ≈ 1.5 + ctrl.kp_pos * 2.0

        # speed_limit clamps the TOTAL setpoint, feed-forward included.
        ctrl.v_sp_prev = 0.0
        winch_torque!(wpc, s, l_now; v_ff = 5.0, speed_limit = 1.0,
                      acceleration_limit = acc)
        @test ctrl.v_sp_prev ≈ 1.0

        ctrl.v_sp_prev = v_sp_saved
    end

    @testset "acceleration limit from the settings" begin
        ctrl = wpc
        v_sp_saved = ctrl.v_sp_prev
        l_now = unstretched_length(s)

        # `winch: max_acc:` of data/settings.yaml, taken as given when positive.
        @test s.set.max_acc > 0.0
        @test V3Kite.winch_acc_limit(s.set) == s.set.max_acc
        # A settings file that names no limit means unlimited, not a dead winch.
        set_no_acc = deepcopy(s.set)
        set_no_acc.max_acc = 0.0
        @test V3Kite.winch_acc_limit(set_no_acc) == Inf

        # The ADAPTER defaults to it (as `step!` used to): a 100 m length error
        # would ask for `kp_pos * 100` m/s, and the rate limiter allows
        # `max_acc * dt` per step.
        ctrl.v_sp_prev = 0.0
        step!(s; rel_depower = DEPOWER_SETPOINT,
              set_torque = winch_torque!(wpc, s, l_now + 100.0))
        @test ctrl.v_sp_prev ≈ s.set.max_acc * s.dt

        # Passing `Inf` opts out (checked without stepping: uncapped here is
        # `kp_pos * 100` m/s, which is not a torque worth putting into the DAE).
        ctrl.v_sp_prev = 0.0
        # Re-read the length: the step above moved it.
        winch_torque!(wpc, s, unstretched_length(s) + 100.0;
                      speed_limit = Inf, acceleration_limit = Inf)
        @test ctrl.v_sp_prev ≈ ctrl.kp_pos * 100.0

        ctrl.v_sp_prev = v_sp_saved
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
