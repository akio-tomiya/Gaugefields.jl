using Gaugefields
using Test
using LinearAlgebra
import Wilsonloop: make_plaq

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

function check_su2_instanton_links(U)
    L = (U[1].NX, U[1].NY, U[1].NZ, U[1].NT)
    center = (L[1] / 2 + 0.5, L[2] / 2 + 0.5, L[3] / 2 + 0.5, L[4] / 2 + 0.5)
    radius = div(L[1], 2)

    for μ = 1:4
        for it = 1:L[4], iz = 1:L[3], iy = 1:L[2], ix = 1:L[1]
            link = AG._su2_instanton_link(μ, ix, iy, iz, it, L; center, radius)
            measured = ComplexF64[
                U[μ][1, 1, ix, iy, iz, it] U[μ][1, 2, ix, iy, iz, it]
                U[μ][2, 1, ix, iy, iz, it] U[μ][2, 2, ix, iy, iz, it]
            ]

            @test measured ≈ link
            @test link' * link ≈ Matrix{ComplexF64}(I, 2, 2)
            @test det(link) ≈ 1
        end
    end
end

@testset "SU(2) instanton link helper" begin
    check_su2_instanton_links(Oneinstanton(2, 0, 4, 4, 4, 4))
    check_su2_instanton_links(Oneinstanton(2, 1, 4, 4, 4, 4))

    L = (4, 4, 4, 4)
    anti_link = AG._su2_instanton_link(1, 1, 1, 1, 1, L; sign=-1)
    @test anti_link' * anti_link ≈ Matrix{ComplexF64}(I, 2, 2)
    @test det(anti_link) ≈ 1

    @test_throws ArgumentError AG._su2_instanton_link(0, 1, 1, 1, 1, L)
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, (4, 4, 4))
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, L; radius=0)
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, L; sign=0)
    @test_throws ArgumentError AG._su2_instanton_link(1, 1, 1, 1, 1, L; center=(1, 2, 3))
end

function normalized_plaquette(U)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    return calculate_Plaquette(U, temp1, temp2) / (6 * U[1].NV * U[1].NC)
end

function check_sun_embedded_instanton(NC, block; NDW=0)
    L = (4, 4, 4, 4)
    U2 = Oneinstanton(2, NDW, L...)
    U = Oneinstanton_SUN_embedded(NC, L...; NDW, block)
    spectator = setdiff(1:NC, block)

    for μ = 1:4
        for it = 1:L[4], iz = 1:L[3], iy = 1:L[2], ix = 1:L[1]
            link = Matrix(U[μ][:, :, ix, iy, iz, it])
            u2 = ComplexF64[
                U2[μ][1, 1, ix, iy, iz, it] U2[μ][1, 2, ix, iy, iz, it]
                U2[μ][2, 1, ix, iy, iz, it] U2[μ][2, 2, ix, iy, iz, it]
            ]

            for b = 1:2, a = 1:2
                @test link[block[a], block[b]] ≈ u2[a, b]
            end
            for c in spectator
                @test link[c, c] == 1
                for d = 1:NC
                    if d != c
                        @test link[c, d] == 0
                        @test link[d, c] == 0
                    end
                end
            end

            @test link' * link ≈ Matrix{ComplexF64}(I, NC, NC)
            @test det(link) ≈ 1
        end
    end

    @test normalized_plaquette(U) ≈ (2 * normalized_plaquette(U2) + (NC - 2)) / NC
end

@testset "SUN embedded instanton public API" begin
    check_sun_embedded_instanton(3, (1, 2))
    check_sun_embedded_instanton(3, (1, 3))
    check_sun_embedded_instanton(3, (2, 3))
    check_sun_embedded_instanton(4, (2, 4))
    check_sun_embedded_instanton(5, (2, 5))
    check_sun_embedded_instanton(3, (1, 3); NDW=1)

    @test_throws ArgumentError Oneinstanton_SUN_embedded(3, 4, 4, 4, 4; block=(1, 1))
    @test_throws ArgumentError Oneinstanton_SUN_embedded(3, 4, 4, 4, 4; block=(1, 4))
    @test_throws ArgumentError Oneinstanton_SUN_embedded(3, 4, 4, 4, 4; NDW=-1)
end

function epsilon4(μ, ν, ρ, σ)
    p = (μ, ν, ρ, σ)
    length(unique(p)) == 4 || return 0

    inversions = 0
    for i = 1:4
        for j = i+1:4
            inversions += p[i] > p[j]
        end
    end
    return iseven(inversions) ? 1 : -1
end

function plaquette_topological_charge(U)
    Dim = 4
    temps = [similar(U[1]), similar(U[1]), similar(U[1]), similar(U[1])]
    F = Matrix{typeof(U[1])}(undef, Dim, Dim)
    for μ = 1:Dim
        for ν = 1:Dim
            F[μ, ν] = similar(U[1])
        end
    end

    for μ = 1:Dim
        for ν = 1:Dim
            if μ != ν
                evaluate_gaugelinks!(temps[1], [make_plaq(μ, ν, Dim=Dim)], U, temps)
                Traceless_antihermitian!(F[μ, ν], temps[1])
            end
        end
    end

    Q = 0.0
    for μ = 1:Dim
        for ν = 1:Dim
            for ρ = 1:Dim
                for σ = 1:Dim
                    if μ != ν && ρ != σ
                        Q += epsilon4(μ, ν, ρ, σ) * tr(F[μ, ν], F[ρ, σ])
                    end
                end
            end
        end
    end

    return -real(Q) / (32 * pi^2)
end

@testset "SUN embedded instanton topological charge" begin
    L = (4, 4, 4, 4)
    cold = Initialize_Gaugefields(3, 0, L..., condition="cold")
    @test isapprox(plaquette_topological_charge(cold), 0; atol=1e-12)

    U2 = Oneinstanton(2, 0, L...)
    Q2 = plaquette_topological_charge(U2)
    @test abs(Q2) > 1

    U3 = Oneinstanton_SUN_embedded(3, L...; block=(1, 2))
    U3_alt = Oneinstanton_SUN_embedded(3, L...; block=(2, 3))
    U4 = Oneinstanton_SUN_embedded(4, L...; block=(2, 4))
    U5 = Oneinstanton_SUN_embedded(5, L...; block=(2, 5))

    @test plaquette_topological_charge(U3) ≈ Q2
    @test plaquette_topological_charge(U3_alt) ≈ Q2
    @test plaquette_topological_charge(U4) ≈ Q2
    @test plaquette_topological_charge(U5) ≈ Q2

    U3_anti = Oneinstanton_SUN_embedded(3, L...; block=(1, 2), sign=-1)
    @test plaquette_topological_charge(U3_anti) ≈ -Q2
end
