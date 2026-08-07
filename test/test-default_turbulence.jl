# Copyright (c) 2026 Uwe Fechner
# SPDX-License-Identifier: MPL-2.0

# Ported from KiteModels.jl (test/test-default_turbulence.jl), which persists the same setting
# the same way. Needs no wind field, so unlike the rest of the turbulence feature it runs in the
# regular suite.

using Pkg
if !("Test" ∈ keys(Pkg.project().dependencies))
    Pkg.activate("test")
end

using Test, V3Kite
using KiteUtils: set_data_path, get_data_path
using KiteUtils.YAML   # not a direct test dependency

function write_gui_default(path::AbstractString; turbulence=0.0)
    open(path, "w") do io
        println(io, "gui:")
        println(io, "    default_turbulence: $(turbulence)   # a comment that must survive")
    end
end

function write_gui_without_turbulence(path::AbstractString)
    open(path, "w") do io
        println(io, "gui:")
        println(io, "    project: system_reelout.yaml")
    end
end

_old_data_path = get_data_path()
try
    @testset "default_turbulence config" begin
        mktempdir() do tmpdir
            set_data_path(tmpdir)

            gui_yaml = joinpath(tmpdir, "gui.yaml")
            write_gui_default(gui_yaml * ".default"; turbulence=0.15)

            # Created on demand from the .default
            @test !isfile(gui_yaml)
            @test get_default_turbulence() ≈ 0.15
            @test isfile(gui_yaml)

            # Round trip, and the trailing comment is kept
            @test set_default_turbulence(0.35) ≈ 0.35
            @test Float64(YAML.load_file(gui_yaml)["gui"]["default_turbulence"]) ≈ 0.35
            @test get_default_turbulence() ≈ 0.35
            @test any(contains("a comment that must survive"), readlines(gui_yaml))

            # Out of range is rejected and changes nothing
            @test isnothing(set_default_turbulence(1.5))
            @test isnothing(set_default_turbulence(-0.1))
            @test get_default_turbulence() ≈ 0.35

            # Bounds are inclusive
            @test set_default_turbulence(0.0) ≈ 0.0
            @test set_default_turbulence(1.0) ≈ 1.0

            # The key is inserted when the section exists without it
            rm(gui_yaml)
            write_gui_without_turbulence(gui_yaml)
            @test set_default_turbulence(0.45) ≈ 0.45
            inserted = YAML.load_file(gui_yaml)
            @test inserted["gui"]["project"] == "system_reelout.yaml"
            @test Float64(inserted["gui"]["default_turbulence"]) ≈ 0.45
            @test get_default_turbulence() ≈ 0.45

            # "default" means "no opinion": stored as the keyword, read back as `nothing`, which
            # is what makes `init` keep `use_turbulence` from the settings YAML.
            @test set_default_turbulence("default") == "default"
            @test YAML.load_file(gui_yaml)["gui"]["default_turbulence"] == "default"
            @test isnothing(get_default_turbulence())
            # Casing is irrelevant, and a number takes over again afterwards
            @test set_default_turbulence("DeFaUlT") == "default"
            @test isnothing(get_default_turbulence())
            @test set_default_turbulence(0.2) ≈ 0.2
            @test get_default_turbulence() ≈ 0.2

            # Any other string is rejected and changes nothing
            @test isnothing(set_default_turbulence("high"))
            @test get_default_turbulence() ≈ 0.2
        end
    end
finally
    set_data_path(_old_data_path)
end

nothing
