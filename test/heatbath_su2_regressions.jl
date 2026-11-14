module HeatbathSU2Regressions
using Gaugefields, LinearAlgebra, Random, Test
const HB = Gaugefields.heatbath_module

@testset "SU(2) heatbath normalization" begin
    for seed in 1:1000
        a = randn(MersenneTwister(seed), 4)
        V = ComplexF64[a[1]+im*a[4] a[3]+im*a[2]; -a[3]+im*a[2] a[1]-im*a[4]]
        U = zeros(ComplexF64, 2, 2)
        result = HB._su2_update_kp_core!(
            MersenneTwister(seed + 1000), U, V, 1.9, 2,
            (similar(U), similar(U)), 100_000)
        @test result[2]
        @test U' * U ≈ I atol=2e-15
        @test det(U) ≈ 1 atol=2e-15
    end
    # A nonsingular input with a non-unit projected norm distinguishes division
    # by the quaternion norm from division by its square.
    U = zeros(ComplexF64, 2, 2)
    result = HB._su2_update_kp_core!(
        MersenneTwister(42), U, ComplexF64[2 0; 0 1], 1.9, 2,
        (similar(U), similar(U)), 100_000)
    @test result[2]
    @test U' * U ≈ I atol=2e-15
    @test det(U) ≈ 1 atol=2e-15
end

@testset "Invalid staples fail without modifying links" begin
    for T in (ComplexF32, ComplexF64), V in (
        zeros(T, 2, 2), fill(T(NaN), 2, 2), fill(T(Inf), 2, 2), T[1 0; 0 0])
        U = Matrix{T}(I, 2, 2)
        temps = (zeros(T, 2, 2), zeros(T, 2, 2))
        rng = MersenneTwister(7)
        unchanged_rng = copy(rng)
        result = HB._su2_update_kp_core!(rng, U, V, 1.9, 2, temps, 100_000)
        @test !result[2] && result[3] == 0
        @test U == I
        @test all(isfinite, temps[1])
        @test rand(rng) == rand(unchanged_rng)
        @test_throws ArgumentError HB.SU2update_KP!(U, V, 1.9, 2, temps)
        @test U == I
    end
    # Rejection exhaustion remains a distinct failure, also preserving U.
    U = Matrix{ComplexF64}(I, 2, 2)
    @test_throws ErrorException HB.SU2update_KP!(
        U, copy(U), 1.9, 2, (similar(U), similar(U)), 0)
    @test U == I
end
end
