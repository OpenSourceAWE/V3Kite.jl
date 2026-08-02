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

@testset "WC_Settings" begin
    default = WC_Settings()
    @test default.winch_pos_kp == 0.5
    @test default.winch_speed_k == 30.0
    @test default.winch_speed_ti == 2.0
    @test default.winch_torque_limit == 500.0
    # Default feed-forward is exact, i.e. a perfectly stiff drum; the active
    # config deliberately differs (see below).
    @test default.winch_ff_scale == 1.0

    # Loaded from data/wc_settings.yaml (see wc_settings: field of system.yaml).
    set_data_path(v3_data_path())
    loaded = WC_Settings("wc_settings.yaml")
    @test loaded.winch_pos_kp == default.winch_pos_kp
    @test loaded.winch_speed_k == default.winch_speed_k
    @test loaded.winch_torque_limit == default.winch_torque_limit
    # Weak integral action, so the yield produced by winch_ff_scale below is not
    # integrated away again within a lap.
    @test loaded.winch_speed_ti == 20.0
    # Compliant winch since 2026-08-01: the drum holds only 70 % of the tether
    # force, so it pays out under load (header of data/wc_settings.yaml).
    @test loaded.winch_ff_scale == 0.7
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
        @test s.winch_ctrl !== nothing
        @test s.winch_ctrl.kp_pos ≈ 0.5  # from data/wc_settings.yaml
    end

    l0 = unstretched_length(s)

    @testset "step! holding torque (no set_length/set_torque)" begin
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

    @testset "step! position mode (set_length)" begin
        t0 = s.sys_state.time
        # Three steps: the previous torque-mode step left residual winch speed
        # that the position loop needs more than one 0.05 s step to work off
        # (1.34, 1.28, 1.23 m/s here). It does NOT come back to ~0: since
        # 2026-08-01 the drum is deliberately compliant (winch_ff_scale = 0.7,
        # data/wc_settings.yaml), so the holding torque falls 30 % short of the
        # load and the winch keeps creeping out at ~1 m/s until the growing
        # length error turns the outer loop around (peak drift ≈ 3.2 m at
        # t ≈ 5 s, then hauled back in). What position mode guarantees on this
        # timescale is only that the speed stays bounded and the length stays
        # near l0 — see the wc_settings testset above for the compliance knob.
        step!(s; rel_depower = DEPOWER_SETPOINT, set_length = l0)
        step!(s; rel_depower = DEPOWER_SETPOINT, set_length = l0)
        step!(s; rel_depower = DEPOWER_SETPOINT, set_length = l0)
        @test s.sys_state.time ≈ t0 + 3 * s.dt
        @test isfinite(unstretched_length(s))
        @test abs(reel_out_speed(s)) < 2.0                # bounded, no runaway
        @test abs(unstretched_length(s) - l0) < 0.5       # ≈ 0.19 m of creep
    end

    @testset "step! live wind update" begin
        t0 = s.sys_state.time
        step!(s; rel_depower = DEPOWER_SETPOINT, set_length = l0, v_wind_gnd = 12.0)
        @test s.sys_state.time ≈ t0 + s.dt
        @test norm(s.set.wind_vec) ≈ 12.0
    end
end
nothing
