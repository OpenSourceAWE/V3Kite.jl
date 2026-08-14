# Copyright (c) 2025 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

"""
Report how much spanwise resolution a built model actually has.

Three counts that are easy to confuse: how many sections the aero geometry ships,
how many unrefined sections survive the structural remesh — every refined
section's polar and surface table is blended from those — and how many
structural stations drive their geometry. Run it on a `sam` built by any
example, or let it build one.
"""

using Pkg
if !Base.generating_output() &&
        Base.active_project() != joinpath(@__DIR__, "Project.toml")
    Pkg.activate(joinpath(@__DIR__))
end

using V3Kite
using VortexStepMethod
using YAML

if !@isdefined(sam)
    PROJECT = select_project(
        ["particle lattice" => "system_v3kite_psm.yaml",
         "Timoshenko-beam wing" => "system_v3kite_beam.yaml"];
        prompt = "Which wing model should be measured?")
    sam, sys = build_v3_model(PROJECT)
end

wing = sam.sys_struct.wings[1]
vsm_wing = wing.vsm_wing
refined = vsm_wing.refined_sections
unrefined = Int(vsm_wing.n_unrefined_sections)

if @isdefined(PROJECT)
    geometry = YAML.load_file(aero_geometry_path(PROJECT))
    @info "Sections the aero geometry ships" n=length(geometry["wing_sections"]["data"]) airfoils=length(geometry["wing_airfoils"]["data"])
end

@info "Sections after the structural remesh" unrefined refined=length(refined) panels=vsm_wing.n_panels
@info "Refined sections carrying a surface table" n=count(s -> !isnothing(s.section_aero), refined)
@info "Structural stations driving the geometry" twist_surfaces=length(wing.twist_surface_idxs)

span_y, twist = wing_twist_dist(sam.sys_struct)
@info "Twist per station [deg]" y=round.(span_y, digits=2) twist=round.(rad2deg.(twist), digits=2)
@info "Steering" differential_twist_deg=round(rad2deg(differential_twist(sam.sys_struct)), digits=3) aero_moment_z=round(aero_moment_z(sam.sys_struct), digits=1)

alpha = wing.vsm_solver.sol.alpha_dist
@info "Panel alpha [deg]" min=round(rad2deg(minimum(alpha)), digits=2) max=round(rad2deg(maximum(alpha)), digits=2)
nothing
