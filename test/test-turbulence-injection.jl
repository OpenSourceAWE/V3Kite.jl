# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Turbulence reaches this plant only through `set.wind_vec` (see `apply_turbulence!`): the
# wing's `wind_disturb` parameter, used before, is invisible to a PARTICLE_DYNAMICS wing,
# whose aero force is a per-point VSM solve. The first testset pins that injection point
# down without needing a wind field; the second flies the real thing and is skipped when the
# 1.24 GB Mann field is not on disk.

using Pkg
if !("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test
using LinearAlgebra
using V3Kite
isdefined(@__MODULE__, :winch_torque!) ||
    include(joinpath(@__DIR__, "..", "examples", "winch_adapter.jl"))
using AtmosphericModels
using KiteUtils: set_data_path, get_data_path

_old_data_path = get_data_path()
try
    set_data_path(v3_data_path())

    @testset "turbulence injection" begin
        PROJECT   = "system_cabauw.yaml"
        V_WIND    = 10.0
        L_TETHER  = 150.0
        DEPOWER   = 0.25
        N_STEPS   = 20
        # Two runs of the same settled model are bitwise identical; measured margins 4.7/20 cm.
        MIN_DIVERGENCE = 1e-3

        # `sim_time` only sizes the logger here; the loops below count the steps themselves.
        fresh_model() = init(V_WIND, L_TETHER; depower_setpoint = DEPOWER,
                             sim_time = 5.0, system_yaml = PROJECT)

        function fly(perturb!)
            s = fresh_model()
            l0 = unstretched_length(s)
            wpc = WinchPosController(load_wc_settings("wc_settings.yaml"; dt = s.dt); dt = s.dt)
            for i in 1:N_STEPS
                perturb!(s, i)
                step!(s; rel_depower = DEPOWER, set_torque = winch_torque!(wpc, s, l0))
            end
            return s
        end

        pos_ref = pos_kite(fly((s, i) -> nothing))

        @testset "set.wind_vec reaches the plant" begin
            s = fly((s, i) -> (s.set.wind_vec = (1 + 0.05 * i / N_STEPS) * s.wind_vec_mean))
            @test norm(pos_kite(s) - pos_ref) > MIN_DIVERGENCE
        end

        @testset "end-to-end turbulence" begin
            s = fresh_model()
            s.set.use_turbulence = 0.5
            clear(s.am)
            idx = findmin(abs.(s.set.v_wind_gnds .- s.set.v_wind))[2]
            field = AtmosphericModels.find_windfield(s.set, s.set.v_wind_gnds[idx])
            if isnothing(field)
                @info "Skipping end-to-end turbulence test: no wind field for $(s.set.v_wind_gnds[idx]) m/s"
            else
                s.am.wf = WindField(s.am, s.set.v_wind)
                l0 = unstretched_length(s)
                wpc = WinchPosController(load_wc_settings("wc_settings.yaml"; dt = s.dt); dt = s.dt)
                for _ in 1:N_STEPS
                    step!(s; rel_depower = DEPOWER, set_torque = winch_torque!(wpc, s, l0))
                end
                @test norm(pos_kite(s) - pos_ref) > MIN_DIVERGENCE
                # The gust is borrowed for the solve only, never latched into the settings.
                @test s.set.wind_vec ≈ s.wind_vec_mean
                @test norm(s.wind_vec_mean) ≈ V_WIND
            end
        end
    end
finally
    set_data_path(_old_data_path)
end

nothing
