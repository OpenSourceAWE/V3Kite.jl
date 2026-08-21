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
# update_sys_state!) is otherwise not precompiled at all. Mirrors
# examples/simple_parking.jl minus the compression logging, the ripple report,
# the saved log, and the winch length loop — `src/` must not depend on
# WinchControllers.jl, so `step!` here just applies the holding torque.
#
# Every `init`/`step!` keyword below is passed explicitly, even where it equals
# the current default: `aero_mode` and `system_yaml` select which model binary
# the SciML specializations are compiled against, so a workload that drifts from
# the example it mirrors warms a code path no example flies.
#
# Both remake flags are pinned false here rather than left to the project file:
# precompilation must consume the serialized geometry and model in data/, never
# rebuild them. Wrapped in try/catch: a workload failure must never break
# `using V3Kite`.
#
# If either data/*.bin cache is missing (e.g. right after a SymbolicAWEModels
# or Julia version bump changes the model filename), `init` rebuilds it from
# scratch and writes it to data/ — turning a few-second workload into a
# multi-minute one and making precompilation non-hermetic. Set
# `precompile_workload = false` under `[V3Kite]` in the project's
# LocalPreferences.toml to skip the workload entirely in that case.
@compile_workload begin
    try
        body_start_damping = [0.0, 0.0, 40.0]
        # sim_time only sizes the logger; keep it below the example's 10 s.
        s = init(9.51, 150.0;
            depower_setpoint = 0.25, sim_time = 1.0, dt = 0.05/3,
            system_yaml = "system_reelout.yaml",
            aero_mode = ContinuousAero(),
            body_start_damping,
            body_sim_damping = 0.8 .* body_start_damping,
            damping_per_stiffness = 0.001,
            remake_model = false, remake_settled_state = false)

        s.sys.winches[1].brake = true

        for _ in 1:3
            step!(s; rel_depower = 0.25, rel_steering = 0.0040,
                  vsm_interval = 1)
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
