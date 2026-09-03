# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Regression test for `bin/delete_cache_files`: deletes V3Kite's cached
# model/settled-geometry files and then flies the run of
# `examples/simple_parking.jl` from a clean cache, checking that the rebuild
# does not throw and that the resulting log has the expected number of rows.
# This is the exact scenario the script's header warns about — a cache that
# outlives the versions it was written against — so clearing it here and
# re-running must always succeed.
#
# Every simulation parameter mirrors `examples/simple_parking.jl` (see
# `test_parking_ripple.jl` for the same convention). Runs first in
# `runtests.jl`, before anything else has a chance to rely on a warm cache:
# it forces a fresh model compile plus a fresh settling run on top of the
# ~600-step parking run itself, so every later test rebuilds from a clean
# cache too.

using Pkg
if !("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test
using V3Kite
isdefined(@__MODULE__, :hold_torque!) ||
    include(joinpath(@__DIR__, "winch_hold_stub.jl"))
using KiteUtils: set_data_path, get_data_path

_old_data_path = get_data_path()
try
    set_data_path(v3_data_path())

    @testset "Delete Cache Files Then Parking" begin
        script = joinpath(@__DIR__, "..", "bin", "delete_cache_files")
        @test success(`$script --yes`)

        PROJECT               = "system_reelout.yaml"
        SIM_TIME               = 10.0
        DT                     = 0.05/3
        V_WIND                 = 9.51
        TETHER_LENGTH          = 150.0
        USE_BRAKE              = true
        DEPOWER_SETPOINT       = 0.25
        REL_STEERING           = 0.0040
        AERO_MODE              = ContinuousAero()
        VSM_INTERVAL           = 5
        BODY_START_DAMPING     = [0.0, 0.0, 40.0]
        BODY_SIM_DAMPING       = [0.0, 0.0, 32.0]
        DAMPING_PER_STIFFNESS  = 0.001

        s = init(V_WIND, TETHER_LENGTH; body_start_damping = BODY_START_DAMPING,
            body_sim_damping = BODY_SIM_DAMPING,
            damping_per_stiffness = DAMPING_PER_STIFFNESS,
            depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
            system_yaml = PROJECT, aero_mode = AERO_MODE, remake_model = false)
        s.sys.winches[1].brake = USE_BRAKE

        # Constant-length setpoint: the tether length just after settling.
        l0 = s.sys_state.l_tether[1]
        lhc = length_hold_controller(s)

        steps_done = 0
        no_exception = true
        try
            for _ in 1:s.steps
                step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering = REL_STEERING,
                      set_torque = hold_torque!(lhc, s, l0), vsm_interval = VSM_INTERVAL)
                steps_done += 1
            end
        catch e
            no_exception = false
            @error "Parking run threw after cache deletion" exception=(e, catch_backtrace())
        end
        @test no_exception
        @test steps_done == s.steps

        # `init` sizes the logger to `steps + 1` rows: one at t=0 plus one per
        # completed step (see `warmup!`'s docstring in `src/interface.jl`).
        # `output/` (gitignored), not the package data path, mirroring
        # `examples/simple_parking.jl`.
        OUTPUT_DIR = joinpath(@__DIR__, "..", "output")
        mkpath(OUTPUT_DIR)
        syslog = save_and_load_log(s.logger, "tmp_delete_cache_parking"; path=OUTPUT_DIR)
        @test length(syslog.syslog) == s.steps + 1
    end
finally
    set_data_path(_old_data_path)
end

nothing
