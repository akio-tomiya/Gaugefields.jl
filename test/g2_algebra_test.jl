using Gaugefields
using LinearAlgebra
using Test

@testset "G2 algebra basis" begin
    basis = g2_basis()

    @test length(basis) == G2_ALGEBRA_DIM

    gram = zeros(Float64, G2_ALGEBRA_DIM, G2_ALGEBRA_DIM)
    for a in 1:G2_ALGEBRA_DIM
        A = basis[a]
        @test size(A) == (G2_FUNDAMENTAL_DIM, G2_FUNDAMENTAL_DIM)
        @test eltype(A) == Float64
        @test norm(A + transpose(A)) < 1.0e-12
        for b in 1:G2_ALGEBRA_DIM
            gram[a, b] = -2 * tr(basis[a] * basis[b])
        end
    end
    @test norm(gram - I) < 1.0e-12
end

@testset "G2 projection" begin
    basis = g2_basis()
    coeffs = [sin(0.37 * a) for a in 1:G2_ALGEBRA_DIM]
    X = g2_matrix(coeffs; basis = basis)

    @test is_g2_algebra_matrix(X; basis = basis, atol = 1.0e-12)
    @test norm(g2_coefficients(X; basis = basis, antisymmetrize = false) - coeffs) < 1.0e-12
    @test norm(project_to_g2(X; basis = basis) - X) < 1.0e-12

    raw = reshape([sin(0.11 * i) + cos(0.07 * j) for i in 1:7, j in 1:7], 7, 7)
    projected_once = project_to_g2(raw; basis = basis)
    projected_twice = project_to_g2(projected_once; basis = basis)

    @test is_g2_algebra_matrix(projected_once; basis = basis, atol = 1.0e-12)
    @test norm(projected_once - projected_twice) < 1.0e-12
end

@testset "G2 link diagnostics" begin
    basis = g2_basis()
    coeffs = [0.04 * cos(0.29 * a) for a in 1:G2_ALGEBRA_DIM]
    U = exp(g2_matrix(coeffs; basis = basis))
    defects = g2_link_defects(U; basis = basis)

    @test defects.imaginary < 1.0e-12
    @test defects.orthogonal < 1.0e-12
    @test defects.determinant < 1.0e-12
    @test defects.algebra < 1.0e-11
    @test is_g2_link(U; basis = basis, atol = 1.0e-11)

    complex_U = ComplexF64.(U)
    complex_U[1, 1] += 1.0e-4im
    complex_defects = g2_link_defects(complex_U; basis = basis)
    @test complex_defects.imaginary > 0
    @test !is_g2_link(complex_U; basis = basis, atol = 1.0e-8)
end
