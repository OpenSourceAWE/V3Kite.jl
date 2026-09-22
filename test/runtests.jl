# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using Test
using LinearAlgebra
using V3Kite
using KitePodModels: KCU
using SymbolicAWEModels: quaternion_to_rotation_matrix, segment_world_length,
    set_unstretched_length!, update_from_sysstate!

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

        # A beam wing's nodes are BODY_STATIC points, which the point damping
        # in every other settling schedule cannot reach.
        for project in ("system_beam.yaml", "system_beam_replay.yaml")
            settle = load_settle(project; kite_set=load_kite(project))
            @test any(!iszero, settle.beam_body_start_damping)
            @test any(!iszero, settle.beam_angular_start_damping)
        end

        # A replay settles onto a recorded row that carries a velocity, so its
        # course is defined and is what the flight it feeds starts on.
        for project in ("system_psm_replay.yaml", "system_beam_replay.yaml")
            settle = load_settle(project; kite_set=load_kite(project))
            @test settle.course_correction_mode === :course
        end

        # A node body's y axis is the in-plane span, and spanwise runs -y to
        # +y, which is VortexStepMethod's `spanwise_direction`.
        for name in ("struc_geometry_beam.yaml", "struc_geometry_beam_wing.yaml")
            table = V3Kite.YAML.load_file(
                joinpath(v3_data_path(), name))["bodies"]
            quat_col = findfirst(==("Q_b_to_w"), table["headers"])
            spanwise = count(table["data"]) do row
                quaternion_to_rotation_matrix(Float64.(row[quat_col]))[2, 2] > 0
            end
            @test spanwise == length(table["data"])
        end

        # The hinge is the chord receiver nearest the crease the aero tables
        # were deflected about; the other two are the chord's own ends.
        topo = V3BeamTopology()
        nodes = V3Kite.SurfplanAdapter.flap_delta_nodes(
            topo.chord_control_fractions, topo.crease_frac)
        for name in ("struc_geometry_beam.yaml", "struc_geometry_beam_wing.yaml")
            table = V3Kite.YAML.load_file(
                joinpath(v3_data_path(), name))["stations"]
            flap_col = findfirst(==("flap_points"), table["headers"])
            point_col = findfirst(==("points"), table["headers"])
            for (i, row) in enumerate(table["data"])
                @test row[flap_col] == ["wing_ctrl_$(i)_$j" for j in nodes]
                @test row[flap_col] ⊆ row[point_col]
            end
        end

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

    function settling_struct(config)
        data_path = V3Kite.project_data_path(config.project, nothing)
        sys, _ = V3Kite.build_settling_struct(config; data_path,
            source_struc = V3Kite.struc_geometry_path(config.project; data_path),
            source_aero = V3Kite.aero_geometry_path(config.project; data_path,
                aero_mode = V3Kite.resolve_aero_mode(config.kite_set)))
        return sys
    end

    @testset "Wing Stations" begin
        sys = settling_struct(V3Kite.V3SettleConfig())
        @test length(sys.stations) == 10
        @test length(sys.wings[1].station_idxs) == 10
        @test length(wing_station_chords(sys)) == 10
        span_y, twist = wing_twist_dist(sys)
        @test length(span_y) == 10
        @test length(twist) == 10
    end

    @testset "Relaxed state is placed unstrained at the tether's own length" begin
        config = load_settle("system_beam_replay.yaml")
        state_path = project_file(config.project, config.kite_set.init_state)
        state = V3Kite.read_state_log(state_path)

        sys = settling_struct(config)
        tether = sys.tethers[1]
        tether.init_stretched_len = 247.557
        set_unstretched_length!(sys, tether, 247.557)
        @test V3Kite.apply_relaxed_state!(sys, state_path)
        segments = [sys.segments[idx] for idx in tether.segment_idxs]
        placed = sum(segment_world_length(segment, sys.points) for segment in segments)
        @test placed ≈ 247.557 atol=0.1
        @test tether.len ≈ placed
        @test sum(segment.l0 for segment in segments) ≈ tether.len

        # At the length it was relaxed at, the state is restored as logged.
        sys = settling_struct(config)
        logged = settling_struct(config)
        update_from_sysstate!(logged, state)
        @test V3Kite.apply_relaxed_state!(sys, state_path)
        @test sys.tethers[1].len == logged.tethers[1].len
        @test all(point.pos_w == logged_point.pos_w
                  for (point, logged_point) in zip(sys.points, logged.points))
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
