using Gaugefields
using LinearAlgebra
using Test

function _site_matrix(U, ix, iy, iz, it)
    return [U[i, j, ix, iy, iz, it] for i in 1:7, j in 1:7]
end

@testset "G2 link cold field" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)

    @test U isa G2Gaugefields_4D_wing
    @test size(U) == (7, 7, 2, 2, 2, 2)
    @test U.NC == 7
    @test U.NDW == 1

    identity7 = Matrix{ComplexF64}(I, 7, 7)
    for it in 1:2
        for iz in 1:2
            for iy in 1:2
                for ix in 1:2
                    @test _site_matrix(U, ix, iy, iz, it) == identity7
                end
            end
        end
    end
end

@testset "G2 link similar and substitute" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    V = similar(U)

    @test V isa G2Gaugefields_4D_wing
    @test size(V) == size(U)
    @test V !== U

    clear_U!(V)
    @test all(iszero, V.U)
    substitute_U!(V, U)
    @test V.U == U.U

    U4 = [identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0) for _ in 1:4]
    V4 = similar(U4)
    @test length(V4) == 4
    @test all(v -> v isa G2Gaugefields_4D_wing, V4)
end

@testset "G2 link wing and shift" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    U[1, 1, 1, 1, 1, 1] = 2
    U[1, 1, 2, 1, 1, 1] = 3
    set_wing_U!(U)

    shifted = shift_U(U, 1)
    @test shifted[1, 1, 1, 1, 1, 1] == U[1, 1, 2, 1, 1, 1]
    @test shifted[1, 1, 2, 1, 1, 1] == U[1, 1, 1, 1, 1, 1]

    shifted_back = shift_U(U, (-1, 0, 0, 0))
    @test shifted_back[1, 1, 1, 1, 1, 1] == U[1, 1, 2, 1, 1, 1]
end

@testset "G2 random link field" begin
    U = randomG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0, randomnumber = "Reproducible", scale = 0.05)
    @test maximum(abs.(imag.(U.U))) < 1.0e-12

    for it in 1:2
        for iz in 1:2
            for iy in 1:2
                for ix in 1:2
                    @test is_g2_link(_site_matrix(U, ix, iy, iz, it); atol = 1.0e-10)
                end
            end
        end
    end
end
