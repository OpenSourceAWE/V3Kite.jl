# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
V3Kite HDF5 Extension

Provides `load_flight_data` for the measured V3 flight logs when HDF5 is
available. Kept out of the package proper because `HDF5.__init__` costs a
multi-second recompile at every `using V3Kite`, and no simulation path needs it.
"""
module V3KiteHDF5Ext

using V3Kite
using HDF5

function V3Kite.load_flight_data(h5_path::String)
    @info "Loading flight data from: $h5_path"
    data = Dict{Symbol, Vector{Float64}}()

    h5open(h5_path, "r") do fid
        # ekf_output datasets with "ekf_" prefix
        if haskey(fid, "ekf_output")
            grp = fid["ekf_output"]::HDF5.Group
            for name in keys(grp)
                obj = grp[name]
                obj isa HDF5.Dataset || continue
                ds = read(obj)
                if eltype(ds) <: Real
                    data[Symbol("ekf_", name)] =
                        convert(Vector{Float64}, ds)
                end
            end
        end

        # flight_data datasets without prefix
        if haskey(fid, "flight_data")
            grp = fid["flight_data"]::HDF5.Group
            for name in keys(grp)
                obj = grp[name]
                obj isa HDF5.Dataset || continue
                ds = read(obj)
                if eltype(ds) <: Real
                    data[Symbol(name)] =
                        convert(Vector{Float64}, ds)
                end
            end
        end
    end

    col_names = Tuple(keys(data))
    return NamedTuple{col_names}(
        Tuple(data[k] for k in col_names))
end

end
