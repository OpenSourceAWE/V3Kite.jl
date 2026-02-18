# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using Test
using V3Kite

@testset "V3Kite.jl" begin

    @testset "Calibration Constants" begin
        # Test base values (official KCU measurements)
        @test V3_STEERING_L0_BASE == 1.6
        @test V3_DEPOWER_L0_BASE == 0.2
        @test V3_STEERING_GAIN == 1.4
        @test V3_DEPOWER_GAIN == 5.0

        # Test default delta
        @test V3_DEFAULT_DELTA == -0.2

        # Test effective values (base + delta)
        @test V3_STEERING_L0 == V3_STEERING_L0_BASE + V3_DEFAULT_DELTA
        @test V3_STEERING_L0 == 1.4  # 1.6 + (-0.2)
        @test V3_DEPOWER_L0 == V3_DEPOWER_L0_BASE + V3_DEFAULT_DELTA
        @test V3_DEPOWER_L0 == 0.0   # 0.2 + (-0.2)
    end

    @testset "Steering Conversion" begin
        # Test zero steering
        L_left, L_right = steering_percentage_to_lengths(0.0)
        @test L_left ≈ V3_STEERING_L0
        @test L_right ≈ V3_STEERING_L0
        @test L_left ≈ L_right

        # Test full left turn (negative percentage)
        L_left, L_right = steering_percentage_to_lengths(-100.0)
        @test L_left > L_right  # Left tape longer for left turn
        @test L_left ≈ V3_STEERING_L0 + V3_STEERING_GAIN / 2
        @test L_right ≈ V3_STEERING_L0 - V3_STEERING_GAIN / 2

        # Test full right turn (positive percentage)
        L_left, L_right = steering_percentage_to_lengths(100.0)
        @test L_right > L_left  # Right tape longer for right turn
        @test L_left ≈ V3_STEERING_L0 - V3_STEERING_GAIN / 2
        @test L_right ≈ V3_STEERING_L0 + V3_STEERING_GAIN / 2

        # Test symmetry
        L_left_neg, L_right_neg = steering_percentage_to_lengths(-50.0)
        L_left_pos, L_right_pos = steering_percentage_to_lengths(50.0)
        @test L_left_neg ≈ L_right_pos
        @test L_right_neg ≈ L_left_pos
    end

    @testset "Steering Round-Trip" begin
        # Test that conversion is invertible
        for pct in [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0]
            L_left, L_right = steering_percentage_to_lengths(pct)
            pct_recovered = steering_length_to_percentage(L_left, L_right)
            @test pct_recovered ≈ pct
        end
    end

    @testset "Depower Conversion" begin
        # Test zero depower
        L_depower = depower_percentage_to_length(0.0)
        @test L_depower ≈ V3_DEPOWER_L0

        # Test full depower
        L_depower = depower_percentage_to_length(100.0)
        @test L_depower ≈ V3_DEPOWER_L0 + V3_DEPOWER_GAIN

        # Test 50% depower
        L_depower = depower_percentage_to_length(50.0)
        @test L_depower ≈ V3_DEPOWER_L0 + V3_DEPOWER_GAIN / 2
    end

    @testset "Depower Round-Trip" begin
        # Test that conversion is invertible
        for pct in [0.0, 25.0, 50.0, 75.0, 100.0]
            L_depower = depower_percentage_to_length(pct)
            pct_recovered = depower_length_to_percentage(L_depower)
            @test pct_recovered ≈ pct
        end
    end

    @testset "Delta Parameter Effect" begin
        # Test that delta parameter affects output correctly
        delta_test = 0.0  # No delta

        # With delta=0, should use base values directly
        L_left, L_right = steering_percentage_to_lengths(0.0; delta=delta_test)
        @test L_left ≈ V3_STEERING_L0_BASE
        @test L_right ≈ V3_STEERING_L0_BASE

        L_depower = depower_percentage_to_length(0.0; delta=delta_test)
        @test L_depower ≈ V3_DEPOWER_L0_BASE

        # Test with custom delta
        delta_custom = 0.5
        L_left, L_right = steering_percentage_to_lengths(0.0; delta=delta_custom)
        @test L_left ≈ V3_STEERING_L0_BASE + delta_custom
        @test L_right ≈ V3_STEERING_L0_BASE + delta_custom
    end

    @testset "CSV Steering Conversion" begin
        # CSV uses opposite sign convention and full gain
        L_left, L_right = csv_steering_percentage_to_lengths(0.0)
        @test L_left ≈ V3_STEERING_L0
        @test L_right ≈ V3_STEERING_L0

        # Positive CSV percentage should result in L_left > L_right
        # (opposite of regular steering_percentage_to_lengths)
        L_left, L_right = csv_steering_percentage_to_lengths(100.0)
        @test L_left > L_right
    end

    @testset "Geometry Suffix" begin
        suffix = build_geom_suffix(0.0, 0.4, 0.95)
        @test suffix == "depower0.0_tip0.4_te0.95"

        suffix = build_geom_suffix(0.2, 0.5, 1.0)
        @test suffix == "depower0.2_tip0.5_te1.0"
    end

    @testset "Coordinate Utilities" begin
        # Test wrap_to_pi
        @test wrap_to_pi(0.0) ≈ 0.0
        @test wrap_to_pi(π) ≈ π atol=1e-10
        @test wrap_to_pi(-π) ≈ -π atol=1e-10
        @test wrap_to_pi(2π) ≈ 0.0 atol=1e-10
        @test wrap_to_pi(3π) ≈ π atol=1e-10
        @test wrap_to_pi(-3π) ≈ -π atol=1e-10
    end

    @testset "V3 Data Path" begin
        # Test that v3_data_path returns a valid directory
        path = v3_data_path()
        @test isdir(path)
        @test isfile(joinpath(path, "system.yaml"))
    end

    @testset "V3SimConfig Defaults" begin
        config = V3SimConfig()

        # Test default values
        @test config.sim_time == 60.0
        @test config.fps == 60
        @test config.v_wind == 10.0
        @test config.up == 40.0
        @test config.us == 0.0
        @test config.tether_length == 250.0
        @test config.brake == true
    end

end
