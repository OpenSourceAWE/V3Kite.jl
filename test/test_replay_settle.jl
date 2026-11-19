# Copyright (c) 2026 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using Test
using LinearAlgebra
using V3Kite
using V3Kite: SymbolicAWEModels

@testset "Replay settling releases the kite at the recorded course, speed and force" begin
    # First 10 Hz row of the 2025-10-09 15:36:29.0 replay window, as flight_replay.jl
    # builds it from the EKF log.
    row = (x = 209.43393640596855, y = -49.30041881791425, z = 142.51935435441496,
           vx = 0.33861928937337515, vy = 7.066431666697887, vz = 2.052624316852965,
           heading = -0.8339269657590052, steering = 0.0204, depower = 0.3937,
           wind_vec = [6.264257299389055, -0.5509124442815528, -0.28590902790135375],
           omega_b = [-0.009251420479632733, 0.007880394831869867, -0.03000543804028756])
    recorded_tether_force = 890.0 # [N] winch force logged in the same row
    project = "system_psm_replay.yaml"
    kite_set = load_kite(project)
    config = load_settle(project; kite_set)
    config.v_wind = 11.017043776974722
    config.tether_length = 247.55713106086483
    config.start_depower = row.depower * 100.0 + 10.0

    sam, _, failed = settle_wing(config, row; show_progress = false,
                                 remake_settled_state = true)
    @test !failed

    data_pos = [row.x, row.y, row.z]
    data_vel = [row.vx, row.vy, row.vz]
    R_t_to_w = SymbolicAWEModels.calc_R_t_to_w(data_pos)
    tangential(vel) = vel - (vel ⋅ R_t_to_w[:, 3]) * R_t_to_w[:, 3]
    course(vel) = atan(vel ⋅ R_t_to_w[:, 2], vel ⋅ R_t_to_w[:, 1])
    wing_vel = collect(sam.sys_struct.wings[1].vel_w)
    @test abs(rad2deg(wrap_to_pi(course(wing_vel) - course(data_vel)))) < 2.0
    @test norm(tangential(wing_vel)) ≈ norm(tangential(data_vel)) atol = 0.2
    @test norm(sam.sys_struct.winches[1].force) ≈ recorded_tether_force rtol = 0.15
end
