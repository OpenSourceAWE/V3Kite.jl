# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using PrecompileTools

@setup_workload begin
    # local _can_precompile = true
    # local _sam = nothing
    # local _sam_q = nothing
    # try
    #     config = V3SimConfig(
    #         wing_type=PARTICLE_DYNAMICS, remake_cache=true)
    #     _sam, _ = create_v3_model(config)

    #     config_rigid_dynamics = V3SimConfig(
    #         wing_type=RIGID_DYNAMICS, remake_cache=true)
    #     _sam_q, _ = create_v3_model(config_rigid_dynamics)
    # catch e
    #     _can_precompile = false
    #     @info "V3Kite: skipping precompile workload" reason=e
    # end

    @compile_workload begin
        # if _can_precompile
        #     # PARTICLE_DYNAMICS wing — build + serialize, then deserialize
        #     init!(_sam; remake=true,
        #         ignore_l0=false, remake_vsm=true)
        #     init!(_sam; remake=false,
        #         ignore_l0=false, remake_vsm=true)

        #     # RIGID_DYNAMICS wing — build + serialize, then
        #     # deserialize
        #     init!(_sam_q; remake=true,
        #         ignore_l0=false, remake_vsm=true)
        #     init!(_sam_q; remake=false,
        #         ignore_l0=false, remake_vsm=true)
        # end
    end
end
