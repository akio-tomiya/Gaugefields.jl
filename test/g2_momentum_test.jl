using Gaugefields
using Test

@testset "G2 momentum initialization" begin
    U = identityG2Gaugefields_4D_wing(2, 2, 2, 2, 1; verbose_level = 0)
    p = initialize_TA_Gaugefields(U)

    @test p isa G2TA_Gaugefields_4D_serial
    @test p.NC == 7
    @test p.NumofBasis == 14
    @test size(p.a) == (14, 2, 2, 2, 2)
    @test length(p.basis) == 14
end

@testset "G2 momentum indexing and similar" begin
    p = G2TA_Gaugefields_4D_serial(2, 2, 2, 2)
    p[3, 1, 1, 1, 1] = 1.25
    @test p[3, 1, 1, 1, 1] == 1.25

    q = similar(p)
    @test q isa G2TA_Gaugefields_4D_serial
    @test size(q.a) == size(p.a)
    @test all(iszero, q.a)
end

@testset "G2 momentum algebra" begin
    p = G2TA_Gaugefields_4D_serial(2, 2, 2, 2)
    q = G2TA_Gaugefields_4D_serial(2, 2, 2, 2)

    p[1, 1, 1, 1, 1] = 2.0
    p[14, 2, 2, 2, 2] = -3.0
    q[1, 1, 1, 1, 1] = 5.0
    q[14, 2, 2, 2, 2] = 7.0

    @test p * q == -11.0

    add_U!(p, 0.5, q)
    @test p[1, 1, 1, 1, 1] == 4.5
    @test p[14, 2, 2, 2, 2] == 0.5

    clear_U!(p)
    @test all(iszero, p.a)
end

@testset "G2 momentum distribution and substitution" begin
    p = G2TA_Gaugefields_4D_serial(2, 2, 2, 2)
    gauss_distribution!(p; σ = 0.1)
    @test size(p.a, 1) == 14
    @test all(isfinite, p.a)

    work = collect(1.0:length(p.a))
    substitute_U!(p, work)
    @test p[1, 1, 1, 1, 1] == 1.0
    @test p[14, 2, 2, 2, 2] == length(p.a)
end
