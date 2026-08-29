using Gaugefields
using LinearAlgebra
using Test

const BFIELD_PLANES_4D = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

function _site_matrix(field, site)
    nc = field.NC
    return [field[row, col, site...] for row in 1:nc, col in 1:nc]
end

function _set_site_matrix!(field, matrix, site)
    for col in axes(matrix, 2), row in axes(matrix, 1)
        field[row, col, site...] = matrix[row, col]
    end
    return field
end

function _plaquette_loops_4d()
    loops = make_loops_fromname("plaquette"; Dim=4)
    append!(loops, loops')
    return loops
end

function _plaquette_action(U, B=nothing; coefficient=1.0)
    action = isnothing(B) ? GaugeAction(U) : GaugeAction(U, B)
    push!(action, coefficient, _plaquette_loops_4d())
    return action
end

function _diagonal_su2_transformation(site)
    angle = 0.173 * site[1] - 0.119 * site[2] +
            0.071 * site[3] + 0.137 * site[4]
    phase = cis(angle)
    return ComplexF64[phase 0; 0 conj(phase)]
end

function _gauge_transform_su2(U, lattice_size)
    transformed = similar(U)
    for site_index in CartesianIndices(lattice_size)
        site = Tuple(site_index)
        gauge_at_site = _diagonal_su2_transformation(site)
        for direction in 1:4
            forward_site = ntuple(4) do dimension
                dimension == direction ?
                    mod1(site[dimension] + 1, lattice_size[dimension]) :
                    site[dimension]
            end
            gauge_at_forward_site = _diagonal_su2_transformation(forward_site)
            link = _site_matrix(U[direction], site)
            transformed_link =
                gauge_at_site * link * adjoint(gauge_at_forward_site)
            _set_site_matrix!(transformed[direction], transformed_link, site)
        end
    end
    foreach(set_wing_U!, transformed)
    return transformed
end

@testset "B-field physical invariants" begin
    @testset "center phases and global 't Hooft flux" begin
        nc = 3
        lattice_size = (2, 3, 2, 2)
        flux = [1, 2, 0, 2, 1, 0]
        B = Initialize_Bfields(
            nc,
            flux,
            0,
            lattice_size...;
            condition="tflux",
            verbose_level=0,
        )
        B_periodic_flux = Initialize_Bfields(
            nc,
            flux .+ nc,
            0,
            lattice_size...;
            condition="tflux",
            verbose_level=0,
        )
        identity_matrix = Matrix{ComplexF64}(I, nc, nc)

        for (flux_index, (mu, nu)) in enumerate(BFIELD_PLANES_4D)
            center_phase = exp(-2pi * im * flux[flux_index] / nc)
            @test center_phase^nc ≈ 1

            for site_index in CartesianIndices(lattice_size)
                site = Tuple(site_index)
                crosses_both_boundaries =
                    site[mu] == lattice_size[mu] &&
                    site[nu] == lattice_size[nu]
                expected_phase = crosses_both_boundaries ? center_phase : 1
                expected = expected_phase * identity_matrix
                matrix = _site_matrix(B[mu, nu], site)

                @test matrix ≈ expected atol=2e-14 rtol=2e-14
                @test matrix' * matrix ≈ identity_matrix atol=2e-14 rtol=2e-14
                @test det(matrix) ≈ 1 atol=5e-14 rtol=5e-14
                periodic_flux_matrix =
                    _site_matrix(B_periodic_flux[mu, nu], site)
                @test matrix ≈ periodic_flux_matrix atol=2e-14 rtol=2e-14
            end

            transverse_site = ones(Int, 4)
            plane_product = one(ComplexF64)
            for coordinate_mu in 1:lattice_size[mu]
                for coordinate_nu in 1:lattice_size[nu]
                    site = ntuple(4) do dimension
                        if dimension == mu
                            coordinate_mu
                        elseif dimension == nu
                            coordinate_nu
                        else
                            transverse_site[dimension]
                        end
                    end
                    plane_product *= B[mu, nu][1, 1, site...]
                end
            end
            @test plane_product ≈ center_phase atol=2e-14 rtol=2e-14
        end

        cold_U = Initialize_Gaugefields(
            nc,
            0,
            lattice_size...;
            condition="cold",
            verbose_level=0,
        )
        action = _plaquette_action(cold_U, B)
        measured_action = real(evaluate_GaugeAction(action, cold_U, B))
        volume = prod(lattice_size)
        expected_action = 2nc * (
            6volume - sum(
                volume ÷ (lattice_size[mu] * lattice_size[nu]) *
                (1 - real(exp(-2pi * im * flux[flux_index] / nc)))
                for (flux_index, (mu, nu)) in enumerate(BFIELD_PLANES_4D)
            )
        )
        @test measured_action ≈ expected_action atol=2e-12 rtol=2e-14
    end

    @testset "zero flux agrees with the ordinary Wilson action" begin
        nc = 2
        lattice_size = (2, 2, 2, 2)
        U = Initialize_Gaugefields(
            nc,
            0,
            lattice_size...;
            condition="hot",
            randomnumber="Reproducible",
            verbose_level=0,
        )
        B_zero = Initialize_Bfields(
            nc,
            zeros(Int, 6),
            0,
            lattice_size...;
            condition="tflux",
            verbose_level=0,
        )
        ordinary_action = _plaquette_action(U)
        B_action = _plaquette_action(U, B_zero)

        ordinary_value = evaluate_GaugeAction(ordinary_action, U)
        B_value = evaluate_GaugeAction(B_action, U, B_zero)
        @test B_value ≈ ordinary_value atol=2e-12 rtol=2e-12
    end

    @testset "gauge invariance with nonzero flux" begin
        nc = 2
        lattice_size = (2, 2, 2, 2)
        U = Initialize_Gaugefields(
            nc,
            0,
            lattice_size...;
            condition="hot",
            randomnumber="Reproducible",
            verbose_level=0,
        )
        B = Initialize_Bfields(
            nc,
            [1, 0, 1, 0, 1, 0],
            0,
            lattice_size...;
            condition="tflux",
            verbose_level=0,
        )
        transformed_U = _gauge_transform_su2(U, lattice_size)
        action = _plaquette_action(U, B; coefficient=0.7)

        before = evaluate_GaugeAction(action, U, B)
        after = evaluate_GaugeAction(action, transformed_U, B)
        @test after ≈ before atol=2e-11 rtol=2e-12
    end

    @testset "B-field force agrees with a finite difference" begin
        nc = 2
        lattice_size = (2, 2, 2, 2)
        U = Initialize_Gaugefields(
            nc,
            0,
            lattice_size...;
            condition="hot",
            randomnumber="Reproducible",
            verbose_level=0,
        )
        B = Initialize_Bfields(
            nc,
            [1, 0, 0, 0, 0, 0],
            0,
            lattice_size...;
            condition="tflux",
            verbose_level=0,
        )
        action = _plaquette_action(U, B; coefficient=0.7)
        direction = 1
        site = (2, 2, 1, 1)
        generator = im * ComplexF64[1 0; 0 -1]
        derivative = calc_dSdUμ(action, direction, U, B)
        link = _site_matrix(U[direction], site)
        raw_derivative = _site_matrix(derivative, site)
        analytic_derivative =
            2 * real(tr(generator * link * raw_derivative))

        zero_B = Initialize_Bfields(
            nc,
            zeros(Int, 6),
            0,
            lattice_size...;
            condition="tflux",
            verbose_level=0,
        )
        zero_B_action = _plaquette_action(U, zero_B; coefficient=0.7)
        zero_B_derivative = calc_dSdUμ(zero_B_action, direction, U, zero_B)
        zero_B_raw_derivative = _site_matrix(zero_B_derivative, site)
        @test norm(raw_derivative - zero_B_raw_derivative) > 1e-6

        function varied_action(epsilon)
            varied_U = similar(U)
            substitute_U!(varied_U, U)
            varied_link = exp(epsilon * generator) * link
            _set_site_matrix!(varied_U[direction], varied_link, site)
            set_wing_U!(varied_U[direction])
            return real(evaluate_GaugeAction(action, varied_U, B))
        end

        epsilon = 1e-6
        finite_difference =
            (varied_action(epsilon) - varied_action(-epsilon)) / (2epsilon)
        @test finite_difference ≈ analytic_derivative atol=2e-8 rtol=2e-7
    end
end
