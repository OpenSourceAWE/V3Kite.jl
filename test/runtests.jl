# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using Test
using LinearAlgebra
using V3Kite
using KitePodModels: KCU

@testset "V3Kite.jl" begin

    # Runs first: wipes the shared cache scratchspace, so every other test
    # below it rebuilds from a clean cache rather than reusing a stale one.
    include("test_delete_cache_parking.jl")

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
        cfg_cont = V3Kite.V3SettleConfig(kite_set=V3KiteConfig(
            aero_mode=V3Kite.SymbolicAWEModels.ContinuousAero()))
        path_dir = V3Kite.settled_state_path(cfg_dir, init_row)
        path_cont = V3Kite.settled_state_path(cfg_cont, init_row)
        @test path_dir != path_cont
        @test occursin("_aerocont", path_cont)
        @test !occursin("_aero", path_dir)
        @test endswith(path_dir, ".arrow")

        # A state logged for one geometry has the wrong point count for another.
        cfg_beam = V3Kite.V3SettleConfig(project="system_beam.yaml")
        path_beam = V3Kite.settled_state_path(cfg_beam, init_row)
        @test path_beam != path_dir
        @test occursin("_struc_geometry_beam", path_beam)
    end

    @testset "Default Cache Path" begin
        # Every install mode — Pkg-installed or a development checkout — caches
        # to the same scratchspace, regardless of the `data_path` argument, so
        # `precompile.jl`'s warm-up artifacts and every runtime caller agree on
        # where to look.
        dev = v3_data_path()
        foreign = mktempdir()
        redirected = V3Kite.default_cache_path(dev)
        @test redirected != dev
        @test !startswith(redirected, joinpath(DEPOT_PATH[1], "packages"))
        @test occursin("scratchspaces", redirected)
        @test V3Kite.default_cache_path(foreign) == redirected
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
    end

    @testset "Project Settings Files" begin
        # Every project file has to name what a run needs, and every settings
        # file it names has to parse into its struct: a key renamed on one side
        # only surfaces here rather than mid-run.
        set_data_path(v3_data_path())
        for project in ("system_reelout.yaml",
                        "system_cabauw.yaml", "system_psm.yaml",
                        "system_beam.yaml", "system_psm_replay.yaml",
                        "system_beam_replay.yaml")
            kite_set = load_kite(project)
            @test kite_set isa V3KiteConfig
            @test kite_set.init_mode in (:settle, :relaxed_state)
            @test isfile(struc_geometry_path(project))
            settle = load_settle(project; kite_set)
            @test settle.project == project
            @test settle.num_steps > 0
        end

        # The beam is flown from a state relaxed at one depower, so the
        # project's depower is not free to disagree with it.
        beam_set = Settings("system_beam.yaml")
        @test occursin("dp$(Int(beam_set.depower))",
            load_kite("system_beam.yaml").init_state)

        beam = load_kite("system_beam.yaml")
        @test beam.backend isa KernelBackend
        @test !beam.geom.reduce_tip && !beam.geom.reduce_te
        @test beam.bridle.compression_frac == 0.0
        @test beam.init_mode == :settle
        @test beam.aero_mode isa AeroPressure

        # The settling schedule sets the transient, the kite file what flies.
        beam_settle = load_settle("system_beam.yaml"; kite_set=beam)
        @test beam_settle.body_start_damping == [0.0, 0.0, 40.0]
        @test beam_settle.kite_set.body_sim_damping == beam.body_sim_damping

        # A geometry carrying polars alone cannot fly `pressure`, and says so
        # when the project loads rather than inside the model build.
        @test_throws ErrorException aero_geometry_path("system_psm.yaml";
            aero_mode=AeroPressure())

        # A typo in a settings file is caught when it loads, not silently
        # defaulted.
        @test_throws ErrorException V3Kite.fill_struct(
            V3BridleConfig, Dict("compresion_frac" => 0.5))
    end

    @testset "Project Outside The Package" begin
        # A project kept elsewhere reads its own settings files and falls back
        # to the ones V3Kite ships for what it does not carry.
        mktempdir() do dir
            cp(joinpath(v3_data_path(), "system_beam.yaml"),
               joinpath(dir, "system_beam.yaml"))
            text = read(joinpath(v3_data_path(), "kite_settings_beam.yaml"), String)
            write(joinpath(dir, "kite_settings_beam.yaml"),
                  replace(text, "wing_mass: 0.0" => "wing_mass: 3.0"))
            project = joinpath(dir, "system_beam.yaml")
            @test load_kite(project).wing_mass == 3.0
            @test dirname(struc_geometry_path(project)) == v3_data_path()
            @test project_data_path(project, nothing) == dir
            @test project_data_path("system_beam.yaml", nothing) == v3_data_path()
            @test load_kite("system_beam.yaml").wing_mass == 0.0
        end
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
