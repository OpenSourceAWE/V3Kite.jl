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

        # Full positive: left tape shorter, right tape longer
        L_left, L_right =
            steering_percentage_to_lengths(100.0)
        @test L_left < L_right
        @test L_left ≈ V3_STEERING_L0_BASE - V3_STEERING_GAIN
        @test L_right ≈ V3_STEERING_L0_BASE + V3_STEERING_GAIN

        # Full negative: right tape shorter, left tape longer
        L_left, L_right =
            steering_percentage_to_lengths(-100.0)
        @test L_right < L_left
        @test L_left ≈ V3_STEERING_L0_BASE + V3_STEERING_GAIN
        @test L_right ≈ V3_STEERING_L0_BASE - V3_STEERING_GAIN

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

    @testset "Cache-Key Number Tag" begin
        # num_tag feeds a file name: no trailing ".0", no dots from vector separators.
        @test V3Kite.num_tag(40.0) == "40"
        @test V3Kite.num_tag(0.0) == "0"
        @test V3Kite.num_tag([0.0, 0.0, 40.0]) == "0-0-40"
        @test V3Kite.num_tag(69.5) == "69.5"
        # Distinct elevations must produce distinct tags.
        @test V3Kite.num_tag(70.0) != V3Kite.num_tag(69.5)
    end

    @testset "Settled Cache Aero Mode" begin
        # An aero mode reaches the settled geometry through the wing's aero
        # object, so it has to reach the cache name too.
        init_row = (x=100.0, y=0.0, z=180.0, vx=0.0, vy=0.0, vz=0.0,
                    heading=0.0, steering=0.0, depower=0.25,
                    wind_vec=[10.0, 0.0, 0.0])
        cfg_dir = V3Kite.V3SettleConfig()
        cfg_cont = V3Kite.V3SettleConfig(
            aero_mode=V3Kite.SymbolicAWEModels.ContinuousAero())
        path_dir = V3Kite.settled_struct_path(cfg_dir, init_row)
        path_cont = V3Kite.settled_struct_path(cfg_cont, init_row)
        @test path_dir != path_cont
        @test occursin("_aerocont", path_cont)
        # The default adds nothing, so caches predating the aero tag stay in use.
        @test !occursin("_aero", path_dir)

        # Guards a file written before the tag existed, or renamed by hand.
        cont_wing = (aero=V3Kite.SymbolicAWEModels.ContinuousAero(),)
        @test V3Kite.aero_mode_matches((wings=[cont_wing],),
            V3Kite.SymbolicAWEModels.ContinuousAero())
        @test !V3Kite.aero_mode_matches((wings=[cont_wing],),
            V3Kite.SymbolicAWEModels.AeroDirect())
        # A wing without aerodynamics cannot disagree.
        @test V3Kite.aero_mode_matches((wings=[(aero=nothing,)],),
            V3Kite.SymbolicAWEModels.AeroDirect())
    end

    @testset "Default Cache Path" begin
        # A development checkout caches in place, keeping the existing data/*.bin.
        dev = v3_data_path()
        @test V3Kite.default_cache_path(dev) == dev

        # A Pkg-installed copy must not be written to: Pkg.gc deletes that tree.
        installed = joinpath(DEPOT_PATH[1], "packages", "V3Kite", "abc123", "data")
        redirected = V3Kite.default_cache_path(installed)
        @test redirected != installed
        @test !startswith(redirected, joinpath(DEPOT_PATH[1], "packages"))
        @test occursin("scratchspaces", redirected)
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

    include("test_ripple_metrics.jl")

    include("test_turn_rate_id.jl")

    include("test-default_turbulence.jl")

    include("test-interface.jl")

    include("test-turbulence-injection.jl")

    # Slowest of the lot: a 600-step parking run (see the file header).
    include("test_parking_ripple.jl")

end
nothing
