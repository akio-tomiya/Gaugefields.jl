using Gaugefields
using LatticeMatrices
using LinearAlgebra
using Random
using Test

import Gaugefields.heatbath_module: heatbath_log_uniform, heatbath_uniform

mutable struct ReplaySU3SiteRNG <: LatticeMatrices.SiteRNG
    values::Vector{Float64}
    index::Int
end

function _replay_su3_uniform(rng::ReplaySU3SiteRNG, ::Type{T}) where T
    value = T(rng.values[rng.index])
    rng.index += 1
    return rng, value
end

LatticeMatrices.rand_uniform(rng::ReplaySU3SiteRNG, ::Type{Float32}) =
    _replay_su3_uniform(rng, Float32)
LatticeMatrices.rand_uniform(rng::ReplaySU3SiteRNG, ::Type{Float64}) =
    _replay_su3_uniform(rng, Float64)
LatticeMatrices.rand_uniform_open(rng::ReplaySU3SiteRNG, ::Type{Float32}) =
    _replay_su3_uniform(rng, Float32)
LatticeMatrices.rand_uniform_open(rng::ReplaySU3SiteRNG, ::Type{Float64}) =
    _replay_su3_uniform(rng, Float64)

mutable struct ReplaySU3PortableRNG
    values::Vector{Float64}
    index::Int
end

function _replay_su3_portable_uniform(
    rng::ReplaySU3PortableRNG,
    ::Type{T},
) where T
    value = T(rng.values[rng.index])
    rng.index += 1
    return rng, value
end

heatbath_uniform(rng::ReplaySU3PortableRNG, ::Type{T}) where T =
    _replay_su3_portable_uniform(rng, T)
heatbath_log_uniform(rng::ReplaySU3PortableRNG, ::Type{T}) where T =
    _replay_su3_portable_uniform(rng, T)

su3_matrix_bits(A::AbstractMatrix{ComplexF64}) =
    collect(reinterpret(UInt64, vec(Matrix(A))))

function _allocated_su3_rng_update(u, V, temps2, temps3, rngs)
    return @allocated SU3update_matrix_rng!(
        u, V, 5.7, temps2, temps3, rngs, 100_000
    )
end

function _allocated_su3_legacy_update(u, V, temps2, temps3)
    return @allocated SU3update_matrix_allocationfree!(
        u, V, 5.7, 3, temps2, temps3, 100_000
    )
end

@testset "allocation-free SU(3) update matches SU3update_matrix!" begin
    staples = [
        ComplexF64[
            4 0.2+0.1im -0.3im
            -0.1+0.2im 3.5 0.4-0.2im
            0.3+0.1im -0.2-0.3im 3.8
        ],
        ComplexF64[
            1.7+0.3im -0.4+0.2im 0.1-0.5im
            0.2+0.6im 2.1-0.1im -0.3+0.4im
            -0.2+0.1im 0.5+0.2im 1.9+0.4im
        ],
        Matrix{ComplexF64}(I, 3, 3) * 6,
    ]
    betas = (0.5, 5.7, 20.0)
    iteration_max = 100_000

    for V in staples, beta in betas, seed in 1:40
        reference = Matrix{ComplexF64}(I, 3, 3)
        candidate = copy(reference)
        reference_temps2 = [zeros(ComplexF64, 2, 2) for _ in 1:5]
        reference_temps3 = [zeros(ComplexF64, 3, 3) for _ in 1:5]
        candidate_temps2 = ntuple(_ -> zeros(ComplexF64, 2, 2), 4)
        candidate_temps3 = ntuple(_ -> zeros(ComplexF64, 3, 3), 3)

        Random.seed!(seed)
        SU3update_matrix!(
            reference,
            V,
            beta,
            3,
            reference_temps2,
            reference_temps3,
            iteration_max,
        )
        reference_continuation = ntuple(_ -> rand(UInt64), 4)

        Random.seed!(seed)
        SU3update_matrix_allocationfree!(
            candidate,
            V,
            beta,
            3,
            candidate_temps2,
            candidate_temps3,
            iteration_max,
        )
        candidate_continuation = ntuple(_ -> rand(UInt64), 4)

        @test su3_matrix_bits(candidate) == su3_matrix_bits(reference)
        for i in 1:4
            @test su3_matrix_bits(candidate_temps2[i]) ==
                  su3_matrix_bits(reference_temps2[i])
        end
        for i in 1:3
            @test su3_matrix_bits(candidate_temps3[i]) ==
                  su3_matrix_bits(reference_temps3[i])
        end
        @test candidate_continuation == reference_continuation
    end

    V = staples[1]
    u = Matrix{ComplexF64}(I, 3, 3)
    temps2 = ntuple(_ -> zeros(ComplexF64, 2, 2), 4)
    temps3 = ntuple(_ -> zeros(ComplexF64, 3, 3), 3)
    Random.seed!(1234)
    SU3update_matrix_allocationfree!(u, V, 5.7, 3, temps2, temps3, iteration_max)
    Random.seed!(1234)
    @test _allocated_su3_legacy_update(u, V, temps2, temps3) == 0
end

@testset "clean portable and site-RNG SU(3) updates agree" begin
    V = ComplexF64[
        4 0.2+0.1im -0.3im
        -0.1+0.2im 3.5 0.4-0.2im
        0.3+0.1im -0.2-0.3im 3.8
    ]
    iteration_max = 100_000

    for beta in (0.5, 5.7, 20.0), seed in 1:40
        Random.seed!(seed)
        replay_values = [rand() for _ in 1:20_000]

        reference = Matrix{ComplexF64}(I, 3, 3)
        reference_temps2 = ntuple(_ -> zeros(ComplexF64, 2, 2), 4)
        reference_temps3 = ntuple(_ -> zeros(ComplexF64, 3, 3), 3)
        portable_rng = ReplaySU3PortableRNG(replay_values, 1)
        portable_rng, reference_accepted, reference_failed_subgroup =
            SU3update_matrix!(
                portable_rng,
                reference,
                V,
                beta,
                3,
                reference_temps2,
                reference_temps3,
                iteration_max,
            )

        candidate = Matrix{ComplexF64}(I, 3, 3)
        candidate_temps2 = ntuple(_ -> zeros(ComplexF64, 2, 2), 4)
        candidate_temps3 = ntuple(_ -> zeros(ComplexF64, 3, 3), 3)
        replay_rng = ReplaySU3SiteRNG(replay_values, 1)
        _, accepted, failed_subgroup = SU3update_matrix_rng!(
            candidate,
            V,
            beta,
            candidate_temps2,
            candidate_temps3,
            (replay_rng, replay_rng, replay_rng),
            iteration_max,
        )

        @test accepted == reference_accepted
        @test failed_subgroup == reference_failed_subgroup
        @test portable_rng.index == replay_rng.index
        @test su3_matrix_bits(candidate) == su3_matrix_bits(reference)
        for i in 1:4
            @test su3_matrix_bits(candidate_temps2[i]) ==
                  su3_matrix_bits(reference_temps2[i])
        end
        for i in 1:3
            @test su3_matrix_bits(candidate_temps3[i]) ==
                  su3_matrix_bits(reference_temps3[i])
        end
    end

    @testset "fixed clean-stream reference" begin
        values = repeat([0.5, 0.5, 0.25, 0.1, 0.3, 0.7], 3)
        expected = UInt64[
            0x3fee788dd4c820c6, 0x3fb363976de294e6,
            0xbfc8fe6a6480b5de, 0xbfb916a8321d5a8c,
            0xbfc8d461c0a1aac2, 0xbfa7f1a3d6d12dca,
            0x3fc3239f8dcdb200, 0xbfb311762e6d1967,
            0x3fee573625ccbed9, 0x3fa87c1964de86a4,
            0xbfd0a72b14e7525d, 0xbfac9a7b6dbb225c,
            0x3fcd0efa7b282682, 0xbfb70bf2d4264899,
            0x3fcb03b05b16f31c, 0xbfb4927b33d81487,
            0x3fedf15f35ee9414, 0xbfbe11ac2a19f14b,
        ]
        u = Matrix{ComplexF64}(I, 3, 3)
        Vfixed = Matrix{ComplexF64}(I, 3, 3) * 6
        temps2 = ntuple(_ -> zeros(ComplexF64, 2, 2), 4)
        temps3 = ntuple(_ -> zeros(ComplexF64, 3, 3), 3)
        rng, accepted, failed_subgroup = SU3update_matrix!(
            ReplaySU3PortableRNG(values, 1),
            u,
            Vfixed,
            5.7,
            3,
            temps2,
            temps3,
            100,
        )

        @test accepted
        @test failed_subgroup == 0
        @test rng.index == 19
        @test su3_matrix_bits(u) == expected
    end

    @testset "released Julia-RNG reference" begin
        # Gaugefields v0.7.3 reference on the supported Julia 1.11 series.
        expected = UInt64[
            0x3fec519c4522aeda, 0x3fbb7137b68004fc,
            0x3fd0a4b62e07f8f8, 0xbfd346e0b61fa745,
            0x3fb69483b9abdbd2, 0x3fc9597d204033ff,
            0xbfcc0572b9821cdf, 0xbfd5f63ece535e15,
            0x3fed18642c29d80d, 0xbf988d8ea5ab199f,
            0xbfb3e5552817c7bb, 0xbfa047934b0f69a1,
            0xbfbd5f2eb1a555ce, 0x3fc4d7026f0ad45e,
            0x3fbd30614acd214d, 0x3fa305b82e312577,
            0x3feef5aeee1ac65e, 0xbfb97beea0262afe,
        ]
        u = Matrix{ComplexF64}(I, 3, 3)
        temps2 = ntuple(_ -> zeros(ComplexF64, 2, 2), 4)
        temps3 = ntuple(_ -> zeros(ComplexF64, 3, 3), 3)

        Random.seed!(1)
        result = SU3update_matrix!(
            u,
            V,
            5.7,
            3,
            temps2,
            temps3,
            iteration_max,
        )
        @test result === nothing
        @test su3_matrix_bits(u) == expected
        @test rand(UInt64) == 0x6e1e10f05537e4ce

        u = Matrix{ComplexF64}(I, 3, 3)
        Random.seed!(1)
        allocation_free_result = SU3update_matrix_allocationfree!(
            u, V, 5.7, 3, temps2, temps3, iteration_max
        )
        @test allocation_free_result === nothing
        @test su3_matrix_bits(u) == expected
        @test rand(UInt64) == 0x6e1e10f05537e4ce
    end

    for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
        for global_site in UInt64(0):UInt64(39)
            u = LatticeMatrices.MMatrix{3,3,ComplexF64}(
                Matrix{ComplexF64}(I, 3, 3)
            )
            Vstatic = LatticeMatrices.MMatrix{3,3,ComplexF64}(
                Matrix{ComplexF64}(I, 3, 3) * 6
            )
            temps2 = ntuple(
                _ -> LatticeMatrices.MMatrix{2,2,ComplexF64}(undef), 4
            )
            temps3 = ntuple(
                _ -> LatticeMatrices.MMatrix{3,3,ComplexF64}(undef), 3
            )
            rngs = ntuple(3) do subgroup
                key = RNGStreamKey(1234, 5, 2, 0, subgroup)
                site_rng(key, global_site, algorithm)
            end

            _, accepted, failed_subgroup = SU3update_matrix_rng!(
                u, Vstatic, 5.7, temps2, temps3, rngs, iteration_max
            )
            @test accepted
            @test failed_subgroup == 0
            @test Matrix(u)' * Matrix(u) ≈ Matrix{ComplexF64}(I, 3, 3) atol = 2e-13
            @test det(Matrix(u)) ≈ 1 atol = 2e-13
        end
    end

    algorithm = Philox4x32()
    u = LatticeMatrices.MMatrix{3,3,ComplexF64}(Matrix{ComplexF64}(I, 3, 3))
    Vstatic = LatticeMatrices.MMatrix{3,3,ComplexF64}(
        Matrix{ComplexF64}(I, 3, 3) * 6
    )
    temps2 = ntuple(_ -> LatticeMatrices.MMatrix{2,2,ComplexF64}(undef), 4)
    temps3 = ntuple(_ -> LatticeMatrices.MMatrix{3,3,ComplexF64}(undef), 3)
    make_rngs() = ntuple(3) do subgroup
        site_rng(RNGStreamKey(1234, 5, 2, 0, subgroup), UInt64(0), algorithm)
    end
    SU3update_matrix_rng!(u, Vstatic, 5.7, temps2, temps3, make_rngs(), iteration_max)
    @test _allocated_su3_rng_update(
        u, Vstatic, temps2, temps3, make_rngs()
    ) == 0
end
