using Gaugefields
using Test

import Wilsonloop: get_direction, get_position

function _make_coloring_action(U; rectangle=false, complex_coefficient=false)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, adjoint(plaquettes))
    coefficient = complex_coefficient ? 1 + 0.2im : 1.0
    push!(action, coefficient, plaquettes)
    if rectangle
        rectangles = make_loops_fromname("rectangular", Dim=4)
        append!(rectangles, adjoint(rectangles))
        push!(action, -0.1, rectangles)
    end
    return action
end

function _same_direction_displacements(action, direction, global_size)
    dependencies = Set{NTuple{4,Int}}()
    for dataset in action.dataset
        iszero(dataset.β) && continue
        for staple in dataset.staples[direction]
            for link_index in 1:length(staple)
                link = staple[link_index]
                get_direction(link) == direction || continue
                displacement = get_position(link)
                push!(dependencies, ntuple(
                    d -> mod(displacement[d], global_size[d]), 4
                ))
            end
        end
    end
    return dependencies
end

function _validate_coloring_by_enumeration(
    action,
    global_size,
    colorings,
)
    for direction in 1:4
        coloring = colorings[direction]
        dependencies = _same_direction_displacements(
            action, direction, global_size
        )
        for site in CartesianIndices(global_size)
            indices = Tuple(site)
            source_color = heatbath_site_color(coloring, indices)
            for displacement in dependencies
                neighbor = ntuple(
                    d -> mod(indices[d] - 1 + displacement[d], global_size[d]) + 1,
                    4,
                )
                @test heatbath_site_color(coloring, neighbor) != source_color
            end
        end
    end
end

@testset "general-action heatbath coloring" begin
    global_size = (8, 8, 8, 8)
    U = Initialize_Gaugefields(
        2, 0, global_size...; condition="cold", verbose_level=0
    )

    plaquette_action = _make_coloring_action(U)
    plaquette_colorings = heatbath_colorings(
        plaquette_action, global_size
    )
    @test all(coloring -> coloring.ncolors == 2, plaquette_colorings)
    _validate_coloring_by_enumeration(
        plaquette_action, global_size, plaquette_colorings
    )

    rectangle_action = _make_coloring_action(U; rectangle=true)
    rectangle_colorings = heatbath_colorings(
        rectangle_action, global_size
    )
    @test all(coloring -> coloring.ncolors == 4, rectangle_colorings)
    @test rectangle_colorings[1].coefficients == (2, 1, 1, 1)
    @test rectangle_colorings[2].coefficients == (1, 2, 1, 1)
    @test rectangle_colorings[3].coefficients == (1, 1, 2, 1)
    @test rectangle_colorings[4].coefficients == (1, 1, 1, 2)
    _validate_coloring_by_enumeration(
        rectangle_action, global_size, rectangle_colorings
    )

    six_colorings = heatbath_colorings(
        rectangle_action, (6, 6, 6, 6)
    )
    @test all(coloring -> coloring.ncolors == 6, six_colorings)
    @test_throws ArgumentError heatbath_colorings(
        rectangle_action, global_size; max_colors=3
    )

    complex_action = _make_coloring_action(U; complex_coefficient=true)
    @test_throws ArgumentError heatbath_colorings(complex_action, global_size)

    # A distance-two rectangle aliases the target link on a length-two
    # periodic dimension, so no single-link heatbath conditional exists.
    @test_throws ArgumentError heatbath_colorings(
        rectangle_action, (2, 2, 2, 2)
    )
end
