# Copyright (c) 2026 Bart van de Lint
# SPDX-License-Identifier: MPL-2.0

@testset "Beam rigidities list every element of the beam wing" begin
    project = "system_beam.yaml"
    config = load_settle(project; kite_set = load_kite(project))
    data_path = V3Kite.project_data_path(project, nothing)
    sys, _ = V3Kite.build_settling_struct(config; data_path,
        source_struc = struc_geometry_path(project; data_path),
        source_aero = aero_geometry_path(project; data_path,
            aero_mode = V3Kite.resolve_aero_mode(config.kite_set)))
    elements = beam_rigidities(sys)

    @test length(elements) == length(sys.timoshenko_joints) == 21
    @test count(element -> element.member === :leading_edge, elements) == 11
    @test count(element -> element.member === :strut, elements) == 10

    # The centre leading-edge bay is the thickest tube, so the stiffest element.
    stiffest = elements[argmax([element.EI for element in elements])]
    @test stiffest.name === :le_beam_5
    @test stiffest.EI ≈ 605.0797
    @test abs(stiffest.span) < 1e-9
    @test stiffest.radius ≈ 0.1008

    # The wing is symmetric about its centre plane.
    leading_edge = sort(filter(element -> element.member === :leading_edge, elements);
        by = element -> element.span)
    @test [element.EI for element in leading_edge] ≈
        reverse([element.EI for element in leading_edge])
    @test all(element -> element.length > 0, elements)
end
