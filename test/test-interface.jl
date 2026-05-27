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
        sim_step!(sam; set_values=[0.0], dt=1/60, vsm_interval=1)
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
nothing
