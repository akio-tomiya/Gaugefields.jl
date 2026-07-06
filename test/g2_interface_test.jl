using Gaugefields
using LinearAlgebra
using Test

function _fill_diagonal_field!(U, values)
    clear_U!(U)
    for it in 1:U.NT
        for iz in 1:U.NZ
            for iy in 1:U.NY
                for ix in 1:U.NX
                    for k in 1:7
                        U[k, k, ix, iy, iz, it] = values[k]
                    end
                end
            end
        end
    end
    set_wing_U!(U)
    return U
end

function _site_matrix(U, ix, iy, iz, it)
    return [U[i, j, ix, iy, iz, it] for i in 1:7, j in 1:7]
end

@testset "G2 link tr and add" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    V = similar(U)
    _fill_diagonal_field!(V, collect(1:7))

    @test tr(U) == 7 * U.NV
    @test tr(V) == sum(1:7) * V.NV
    @test tr(U, V) == tr(V)

    W = similar(U)
    clear_U!(W)
    add_U!(W, V)
    @test W.U == V.U
    add_U!(W, -1, V)
    @test all(iszero, W.U)
end

@testset "G2 link mul" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    V = similar(U)
    W = similar(U)
    _fill_diagonal_field!(V, collect(1:7))

    mul!(W, U, V)
    @test W.U == V.U

    mul!(W, V, U)
    @test W.U == V.U

    mul!(W, 2.0, V)
    @test _site_matrix(W, 1, 1, 1, 1) == 2.0 .* _site_matrix(V, 1, 1, 1, 1)

    mul!(W, U, V, 3.0, 0.0)
    @test _site_matrix(W, 1, 1, 1, 1) == 3.0 .* _site_matrix(V, 1, 1, 1, 1)

    mul!(W, U, V, 2.0, -1.0)
    @test _site_matrix(W, 1, 1, 1, 1) == -1.0 .* _site_matrix(V, 1, 1, 1, 1)
end

@testset "G2 shifted and adjoint mul" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    V = similar(U)
    W = similar(U)

    V[1, 1, 1, 1, 1, 1] = 2
    V[1, 1, 2, 1, 1, 1] = 3
    set_wing_U!(V)

    mul!(W, shift_U(U, 1), V)
    @test W[1, 1, 1, 1, 1, 1] == V[1, 1, 1, 1, 1, 1]

    mul!(W, V', U)
    @test W[1, 1, 1, 1, 1, 1] == conj(V[1, 1, 1, 1, 1, 1])
end
