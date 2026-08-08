# SPDX-FileCopyrightText: 2025 Jelle Poland, Bart van de Lint, Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# --- Standard Library & General Utilities ---
using Pkg, LinearAlgebra, Statistics, Serialization, Printf, Dates

# --- Numerical & Scientific Computing ---
using StaticArrays, Rotations, NonlinearSolve, OrdinaryDiffEqBDF,
      OrdinaryDiffEqCore, OrdinaryDiffEqNonlinearSolve, SteadyStateDiffEq,
      ModelingToolkit, ControlSystemsBase, SymbolicIndexingInterface

# --- Data & I/O ---
using CSV, DataFrames, HDF5, Parameters, UnPack, DiscretePIDs

# --- Visualization ---
using GLMakie, Colors, LaTeXStrings

# --- Open Source AWE Packages ---
using AtmosphericModels, KiteUtils, KitePodModels, VortexStepMethod,
      SymbolicAWEModels

GC.gc(true)
let mem = Sys.free_memory() / 1024^2
    @info "Free memory: $(round(mem; digits=1)) MB"
    swap_gb = 0.0
    if Sys.islinux()
        swapon_cmd = Sys.which("swapon")
        if swapon_cmd === nothing
            @info "swapon command not found; skipping swap size detection"
        else
            try
                swap_info = read(`$swapon_cmd --show --bytes --noheadings`, String)
                if !isempty(strip(swap_info))
                    swap_size = sum(parse(Int, split(line)[3]) for line in split(strip(swap_info), '\n') if !isempty(line))
                    swap_gb = swap_size / 1024^3
                    @info "Swap size: $(round(swap_gb; digits=1)) GB"
                else
                    @info "No swap configured"
                end
            catch e
                @warn "Failed to query swap size via swapon; proceeding without swap information" exception = e
            end
        end
    end
    if haskey(ENV, "JULIA_IMAGE_THREADS")
        @info "JULIA_IMAGE_THREADS: $(ENV["JULIA_IMAGE_THREADS"])"
    else
        free_gb = mem / 1024
        if free_gb + swap_gb < 36.0
            @error "JULIA_IMAGE_THREADS is not defined and total available memory ($(round(free_gb + swap_gb; digits=1)) GB free RAM + swap) is less than 36 GB. System image creation may fail!"
        else
            @info "JULIA_IMAGE_THREADS not defined!"
        end
    end
end

# PackageCompiler is deliberately not a dependency of the test project: it is only needed to build
# the sysimage, so install it into a temporary environment and keep building against the test
# project, which is where all the packages listed below live.
const TEST_PROJECT = dirname(Base.active_project())
Pkg.activate(; temp=true)
Pkg.add("PackageCompiler")
using PackageCompiler

@info "Creating sysimage ..."
PackageCompiler.create_sysimage(
    [:Pkg, :LinearAlgebra, :Statistics, :Serialization, :Printf, :Dates,
     :StaticArrays, :Rotations, :NonlinearSolve, :OrdinaryDiffEqBDF,
     :OrdinaryDiffEqCore, :OrdinaryDiffEqNonlinearSolve, :SteadyStateDiffEq,
     :ModelingToolkit, :ControlSystemsBase, :SymbolicIndexingInterface,
     :CSV, :DataFrames, :HDF5, :Parameters, :UnPack, :DiscretePIDs,
     :GLMakie, :Colors, :LaTeXStrings,
     :AtmosphericModels, :KiteUtils, :KitePodModels, :VortexStepMethod,
     :SymbolicAWEModels];
    sysimage_path="kps-image_tmp.so",
    project=TEST_PROJECT,
    precompile_execution_file=joinpath("test", "test_for_precompile.jl")
)
