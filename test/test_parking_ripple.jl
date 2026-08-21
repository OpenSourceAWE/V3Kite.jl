# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Regression test on the parked AoA oscillation: flies the run of
# `examples/simple_parking.jl` and checks its ripple RMS against the baseline
# below. Unlike `test_ripple_metrics.jl`, which measures synthetic signals, this
# one measures the MODEL, so it catches a damping/solver change that lets the
# ~5.5 Hz mode grow again. It is the slowest test in the suite (600 steps).
#
# Every simulation parameter here mirrors `examples/simple_parking.jl` and must
# keep doing so: the metric is only comparable across runs that fix PROJECT,
# V_WIND, TETHER_LENGTH, DEPOWER_SETPOINT, REL_STEERING, SIM_TIME, DT and the
# damping constants (see the docstring of `src/ripple_metrics.jl`). Retake the
# baseline whenever one of them, or `data/ripple_settings.yaml`, changes.

using Pkg
if !("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test
using V3Kite
isdefined(@__MODULE__, :hold_torque!) ||
    include(joinpath(@__DIR__, "winch_hold_stub.jl"))
using KiteUtils: set_data_path, get_data_path
import KiteUtils   # `syslog` is called qualified: the plots scripts bind the bare
                   # name to a SysLog value, which would shadow the import here

_old_data_path = get_data_path()
try
    set_data_path(v3_data_path())

    @testset "Parking AoA Ripple" begin
        PROJECT               = "system_reelout.yaml"
        SIM_TIME              = 10.0
        DT                    = 0.05/3
        V_WIND                = 9.51
        TETHER_LENGTH         = 150.0
        USE_BRAKE             = true
        DEPOWER_SETPOINT      = 0.25
        REL_STEERING          = 0.0040
        AERO_MODE             = ContinuousAero()
        VSM_INTERVAL          = 1
        BODY_START_DAMPING    = [0.0, 0.0, 40.0]
        DAMPING_PER_STIFFNESS = 0.001

        # Measured with the parameters above; 1.5x leaves room for the run-to-run
        # spread of the solver without letting the oscillation come back.
        RIPPLE_RMS_BASELINE = 0.0064   # [deg]
        RIPPLE_RMS_LIMIT    = 1.5 * RIPPLE_RMS_BASELINE

        s = init(V_WIND, TETHER_LENGTH; body_start_damping = BODY_START_DAMPING,
            damping_per_stiffness = DAMPING_PER_STIFFNESS,
            depower_setpoint = DEPOWER_SETPOINT, sim_time = SIM_TIME, dt = DT,
            system_yaml = PROJECT, aero_mode = AERO_MODE)
        s.sys.winches[1].brake = USE_BRAKE

        # Constant-length setpoint: the tether length just after settling.
        l0 = s.sys_state.l_tether[1]
        lhc = length_hold_controller(s)
        for _ in 1:s.steps
            step!(s; rel_depower = DEPOWER_SETPOINT, rel_steering = REL_STEERING,
                  set_torque = hold_torque!(lhc, s, l0), vsm_interval = VSM_INTERVAL)
        end
        @test s.sys_state.time ≈ SIM_TIME rtol=1e-6   # the run reached the end

        r = aoa_ripple(KiteUtils.syslog(s.logger))
        # The analysis window must be the full post-transient one, or the RMS
        # below is measured over something other than the baseline's window.
        @test r.t_start ≈ RippleSettings("ripple_settings.yaml").t_start atol=DT
        @test r.t_end ≈ SIM_TIME rtol=1e-6
        @test isfinite(r.rms)
        # Logged, not just compared: a passing run still shows how much margin is
        # left, which is what says whether the baseline needs retaking.
        @info "Parking AoA ripple RMS: $(round(r.rms, digits=5))° " *
              "(baseline $(RIPPLE_RMS_BASELINE)°, limit $(round(RIPPLE_RMS_LIMIT, digits=5))°)"
        @test r.rms < RIPPLE_RMS_LIMIT
    end
finally
    set_data_path(_old_data_path)
end

nothing
