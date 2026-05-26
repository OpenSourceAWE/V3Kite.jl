# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using Test
using LinearAlgebra
using V3Kite
using KitePodModels: KCU

@testset "V3Kite.jl" begin

    @testset "Calibration Constants" begin
        @test V3_STEERING_L0_BASE == 1.6
        @test V3_DEPOWER_L0_BASE == 0.2
        @test V3_STEERING_GAIN == 1.4
        @test V3_DEPOWER_GAIN == 5.0
    end

    @testset "Steering Conversion" begin
        # Zero steering — base values
        L_left, L_right =
            steering_percentage_to_lengths(0.0)
        @test L_left ≈ V3_STEERING_L0_BASE
        @test L_right ≈ V3_STEERING_L0_BASE

        # Full positive (left turn): left tape longer
        L_left, L_right =
            steering_percentage_to_lengths(100.0)
        @test_broken L_left > L_right
        @test_broken L_left ≈ V3_STEERING_L0_BASE + V3_STEERING_GAIN
        @test_broken L_right ≈ V3_STEERING_L0_BASE - V3_STEERING_GAIN

        # Full negative (right turn): right tape longer
        L_left, L_right =
            steering_percentage_to_lengths(-100.0)
        @test_broken L_right > L_left
        @test_broken L_left ≈ V3_STEERING_L0_BASE - V3_STEERING_GAIN
        @test_broken L_right ≈ V3_STEERING_L0_BASE + V3_STEERING_GAIN

        # Symmetry
        L_left_neg, L_right_neg =
            steering_percentage_to_lengths(-50.0)
        L_left_pos, L_right_pos =
            steering_percentage_to_lengths(50.0)
        @test L_left_neg ≈ L_right_pos
        @test L_right_neg ≈ L_left_pos
    end

    @testset "Steering Round-Trip" begin
        for pct in [-100.0, -50.0, -25.0, 0.0,
                     25.0, 50.0, 100.0]
            L_left, L_right =
                steering_percentage_to_lengths(pct)
            pct_recovered =
                steering_length_to_percentage(
                    L_left, L_right)
            @test pct_recovered ≈ pct
        end
    end

    @testset "Depower Conversion" begin
        L_depower = depower_percentage_to_length(0.0)
        @test L_depower ≈ V3_DEPOWER_L0_BASE

        L_depower = depower_percentage_to_length(100.0)
        @test L_depower ≈ V3_DEPOWER_L0_BASE + V3_DEPOWER_GAIN

        L_depower = depower_percentage_to_length(50.0)
        @test L_depower ≈ V3_DEPOWER_L0_BASE +
            V3_DEPOWER_GAIN / 2
    end

    @testset "Depower Round-Trip" begin
        for pct in [0.0, 25.0, 50.0, 75.0, 100.0]
            L_depower = depower_percentage_to_length(pct)
            pct_recovered =
                depower_length_to_percentage(L_depower)
            @test pct_recovered ≈ pct
        end
    end

    @testset "Custom l0_base Parameter" begin
        custom_base = V3_STEERING_L0_BASE - 0.2
        L_left, L_right = steering_percentage_to_lengths(
            0.0; l0_base=custom_base)
        @test L_left ≈ custom_base
        @test L_right ≈ custom_base

        custom_base = V3_DEPOWER_L0_BASE - 0.2
        L_depower = depower_percentage_to_length(
            0.0; l0_base=custom_base)
        @test L_depower ≈ custom_base
    end

    @testset "Geometry Suffix" begin
        suffix = build_geom_suffix(0.0, 1.6, 1.6, 0.4, 0.95)
        @test suffix == "dp0.0_sl1.6_sr1.6_tip0.4_te0.95"

        suffix = build_geom_suffix(0.2, 1.4, 1.8, 0.5, 1.0)
        @test suffix == "dp0.2_sl1.4_sr1.8_tip0.5_te1.0"
    end

    @testset "V3GeomAdjustConfig Defaults" begin
        gc = V3GeomAdjustConfig()
        @test gc.reduce_steering == false
        @test gc.steering_reduction == 0.2
        @test gc.reduce_depower == false
        @test gc.depower_reduction == 0.2
    end

    @testset "Coordinate Utilities" begin
        @test wrap_to_pi(0.0) ≈ 0.0
        @test wrap_to_pi(π) ≈ -π atol=1e-10
        @test wrap_to_pi(-π) ≈ -π atol=1e-10
        @test wrap_to_pi(2π) ≈ 0.0 atol=1e-10
        @test wrap_to_pi(3π) ≈ -π atol=1e-10
        @test wrap_to_pi(-3π) ≈ -π atol=1e-10
    end

    @testset "V3 Data Path" begin
        path = v3_data_path()
        @test isdir(path)
        @test isfile(joinpath(path, "system.yaml"))
    end

    @testset "V3SimConfig Defaults" begin
        config = V3SimConfig()
        @test config.sim_time == 60.0
        @test config.fps == 60
        @test config.v_wind == 10.0
        @test config.up == 40.0
        @test config.us == 0.0
        @test config.tether_length == 250.0
        @test config.brake == true
    end

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
        v3kite = V3KITE(set=sam.set, kcu=kcu, sam=sam, sys=sys)

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

end
