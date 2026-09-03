# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Settles the wing from scratch: forces both a model rebuild and a re-settle, so
# neither the serialized model nor the settled state can come from a cache
# written under an older dependency. This is the path a machine with no cache
# takes, and the one that fails as `ERROR: Settling failed` when settling
# diverges. The parameters mirror `examples/simple_sinus.jl`; the assertions are
# on the physics of the settled state, since `test_parking_ripple.jl` carries
# the numeric baseline.

using Pkg
if !("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test
using LinearAlgebra
using V3Kite
isdefined(@__MODULE__, :hold_torque!) ||
    include(joinpath(@__DIR__, "winch_hold_stub.jl"))
using KiteUtils: set_data_path, get_data_path

old_data_path = get_data_path()
try
    set_data_path(v3_data_path())

    @testset "Settling From Scratch" begin
        PROJECT               = "system_reelout.yaml"
        V_WIND                = 9.51
        TETHER_LENGTH         = 150.0
        DEPOWER_SETPOINT      = 0.25
        DT                    = 0.05/3
        BODY_START_DAMPING    = [0.0, 0.0, 40.0]
        DAMPING_PER_STIFFNESS = 0.002

        s = init(V_WIND, TETHER_LENGTH; body_start_damping = BODY_START_DAMPING,
            damping_per_stiffness = DAMPING_PER_STIFFNESS,
            depower_setpoint = DEPOWER_SETPOINT, sim_time = 1.0, dt = DT,
            system_yaml = PROJECT, aero_mode = ContinuousAero(),
            remake_model = true, remake_settled_state = true)

        pos = pos_kite(s)
        @test all(isfinite, pos)
        @test pos[3] > 0.0
        elevation = rad2deg(calc_elevation(s))
        @test 20.0 < elevation < 85.0
        @test unstretched_length(s) ≈ TETHER_LENGTH

        lift, drag = lift_drag(s)
        @test isfinite(lift) && isfinite(drag)
        @test lift > drag > 0.0

        # The settled state is a resting one: nothing is left flying apart.
        @test norm(s.sys_state.vel_kite) < 5.0
        @test all(isfinite, s.sys_state.X) && all(isfinite, s.sys_state.Z)

        # It also has to be a state the dynamics can be started from.
        lhc = length_hold_controller(s)
        l0 = s.sys_state.l_tether[1]
        for _ in 1:10
            step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering = 0.0,
                  set_torque = hold_torque!(lhc, s, l0), vsm_interval = 1)
        end
        @test all(isfinite, pos_kite(s))
        @test s.sys_state.time > 0.0
    end
finally
    set_data_path(old_data_path)
end
