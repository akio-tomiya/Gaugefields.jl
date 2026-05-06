using Gaugefields
using Test
using LinearAlgebra

const AG = Gaugefields.AbstractGaugefields_module

function sample_su2_matrix()
    c = cos(0.37)
    s = sin(0.37)
    return ComplexF64[
        c im*s
        im*s c
    ]
end

function check_su2_embedding(NC, block)
    u2 = sample_su2_matrix()
    U = fill(99.0 + 99.0im, NC, NC)

    @test AG._embed_su2_matrix_in_sun!(U, u2, block) === U
    i, j = block
    spectator = setdiff(1:NC, block)

    @test U[i, i] == u2[1, 1]
    @test U[i, j] == u2[1, 2]
    @test U[j, i] == u2[2, 1]
    @test U[j, j] == u2[2, 2]

    for c in spectator
        @test U[c, c] == 1
        for d = 1:NC
            if d != c
                @test U[c, d] == 0
                @test U[d, c] == 0
            end
        end
    end

    @test U' * U ≈ Matrix{ComplexF64}(I, NC, NC)
    @test det(U) ≈ 1
end

@testset "SU(2) matrix embedding into SU(Nc)" begin
    @test AG._validate_su2_embedding_block(3, (1, 2)) == (1, 2)
    @test AG._validate_su2_embedding_block(4, [2, 4]) == (2, 4)

    check_su2_embedding(3, (1, 2))
    check_su2_embedding(3, (1, 3))
    check_su2_embedding(3, (2, 3))
    check_su2_embedding(4, (2, 4))
    check_su2_embedding(5, (2, 5))

    @test_throws ArgumentError AG._validate_su2_embedding_block(1, (1, 2))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (1, 1))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (0, 2))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (1, 4))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (2, 1))
    @test_throws ArgumentError AG._validate_su2_embedding_block(3, (1, 2, 3))
    @test_throws ArgumentError AG._embed_su2_matrix_in_sun!(zeros(3, 3), zeros(3, 3), (1, 2))
    @test_throws ArgumentError AG._embed_su2_matrix_in_sun!(zeros(3, 2), sample_su2_matrix(), (1, 2))
end
