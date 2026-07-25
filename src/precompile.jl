# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using PrecompileTools: @compile_workload

# Warm up the `init`/`step!` path so a fresh `julia -J <sysimage>` process does
# not pay the JIT cost on the first run of an example script.
#
# The sysimage built by bin/create_sys_image deliberately excludes V3Kite.jl
# itself (so Revise keeps working on it), and its precompile_execution_file
# (test/test_for_precompile.jl -> examples/v3kite.jl) drives `settle_wing` and
# `sim_step!` *directly* — it never calls `init` or `step!(::V3KITE, ...)`. So
# the whole high-level interface path (V3KITE wrapper, KCU actuator dynamics,
# winch_position_torque!, update_sys_state!) is otherwise not precompiled at
# all. Mirrors examples/simple_sinus.jl minus the control law and logging.
#
# `remake=false` (the `init` default) means the serialized settled geometry and
# model in data/ are consumed, not rebuilt. Wrapped in try/catch: a workload
# failure must never break `using V3Kite`.
#
# If either data/*.bin cache is missing (e.g. right after a SymbolicAWEModels
# or Julia version bump changes the model filename), `init` rebuilds it from
# scratch and writes it to data/ — turning a few-second workload into a
# multi-minute one and making precompilation non-hermetic. Set
# V3KITE_SKIP_PRECOMPILE_WORKLOAD=1 to skip the workload entirely in that case
# (e.g. from bin/create_sys_image / bin/install right after `--update`).
if get(ENV, "V3KITE_SKIP_PRECOMPILE_WORKLOAD", "0") != "1"
    @compile_workload begin
        try
            # sim_time only sizes the logger; keep it small.
            s = init(10.0, 150.0;
                depower_setpoint = 0.25, sim_time = 1.0,
                system_yaml = "system_cabauw.yaml")

            l0 = s.sys_state.l_tether[1]
            for _ in 1:3
                step!(s; rel_depower = 0.25, rel_steering = 0.0, set_length = l0)
            end

            # Query functions used by the examples.
            lift_drag(s)
            total_drag(s)
            pos_kite(s)
            winch_force(s)
            reel_out_speed(s)
            cl_cd(s)
            calc_elevation(s)
            calc_azimuth(s)
            calc_heading(s)
        catch e
            @warn "PrecompileTools workload failed; continuing without it" exception=(e, catch_backtrace())
        end
    end
end
