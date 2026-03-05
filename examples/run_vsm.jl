#!/usr/bin/env julia
# Copyright (c) 2025 Jelle Poland, Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

using VortexStepMethod
using GLMakie

const VW_MAG = 8.4
const N_PANELS = 36
const ALPHAS_DEG = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

function build_solver_and_body(; data_dir, aero_geometry_file, n_panels)
    settings_path = joinpath(data_dir, "vsm_settings.yaml")
    settings_data = VortexStepMethod.YAML.load_file(settings_path)

    wing_cfg = settings_data["wings"][1]
    solver_cfg = settings_data["solver_settings"]

    wing = Wing(
        aero_geometry_file;
        n_panels=n_panels,
        spanwise_distribution=getproperty(
            VortexStepMethod,
            Symbol(wing_cfg["spanwise_panel_distribution"])),
        spanwise_direction=Float64.(wing_cfg["spanwise_direction"]),
        remove_nan=wing_cfg["remove_nan"],
    )
    refine!(wing)
    body_aero = BodyAerodynamics([wing])
    VortexStepMethod.reinit!(body_aero)

    solver = Solver(body_aero;
        solver_type=(solver_cfg["solver_type"] == "NONLIN" ? NONLIN : LOOP),
        aerodynamic_model_type=getproperty(
            VortexStepMethod,
            Symbol(solver_cfg["aerodynamic_model_type"])),
        density=solver_cfg["density"],
        max_iterations=solver_cfg["max_iterations"],
        rtol=solver_cfg["rtol"],
        tol_reference_error=solver_cfg["tol_reference_error"],
        relaxation_factor=solver_cfg["relaxation_factor"],
        is_with_artificial_damping=solver_cfg["artificial_damping"],
        artificial_damping=(k2=solver_cfg["k2"], k4=solver_cfg["k4"]),
        type_initial_gamma_distribution=getproperty(
            VortexStepMethod,
            Symbol(solver_cfg["type_initial_gamma_distribution"])),
        use_gamma_prev=get(
            solver_cfg, "use_gamma_prev",
            get(solver_cfg, "use_gamme_prev", true)),
        core_radius_fraction=solver_cfg["core_radius_fraction"],
        mu=solver_cfg["mu"],
        is_only_f_and_gamma_output=get(
            solver_cfg, "calc_only_f_and_gamma", false),
        correct_aoa=get(solver_cfg, "correct_aoa", false),
        reference_point=get(
            solver_cfg, "reference_point",
            [0.422646, 0.0, 9.3667]),
    )

    return solver, body_aero
end

function sweep_cmy!(solver, body_aero, alphas_deg; vw_mag)
    cmy = similar(alphas_deg, Float64)
    gamma_prev = isnothing(solver.sol.gamma_distribution) ?
                 nothing : copy(solver.sol.gamma_distribution)

    for (i, alpha_deg) in enumerate(alphas_deg)
        alpha_rad = deg2rad(alpha_deg)
        va = vw_mag .* [cos(alpha_rad), 0.0, sin(alpha_rad)]
        set_va!(body_aero, va)

        if isnothing(gamma_prev)
            solve!(solver, body_aero; log=false)
        else
            solve!(solver, body_aero, gamma_prev; log=false)
        end

        gamma_prev = isnothing(solver.sol.gamma_distribution) ?
                     nothing : copy(solver.sol.gamma_distribution)
        cmy[i] = solver.sol.moment_coeffs[2]
    end

    return cmy
end

function plot_cmy_vs_alpha(alphas_deg, cmy)
    fig = Figure(size=(800, 500))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha [deg]",
        ylabel="CMy [-]",
        title="VSM CMy vs Angle of Attack",
    )
    lines!(ax, alphas_deg, cmy; linewidth=2)
    scatter!(ax, alphas_deg, cmy; markersize=10)
    return fig
end

function main()
    root_dir = dirname(@__DIR__)
    data_dir = joinpath(root_dir, "data")
    aero_geometry_file = joinpath(data_dir, "aero_geometry.yaml")

    solver, body_aero = build_solver_and_body(;
        data_dir,
        aero_geometry_file,
        n_panels=N_PANELS,
    )
    cmy = sweep_cmy!(solver, body_aero, ALPHAS_DEG; vw_mag=VW_MAG)

    println("alpha_deg,CMy")
    for (a, cm) in zip(ALPHAS_DEG, cmy)
        println("$(a),$(cm)")
    end

    fig = plot_cmy_vs_alpha(ALPHAS_DEG, cmy)
    scr = display(fig)
    isinteractive() && wait(scr)

    return nothing
end

main()
