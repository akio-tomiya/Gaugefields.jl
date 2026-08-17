using Gaugefields
using LatticeMatrices
using LinearAlgebra
using Random
using Test

import Gaugefields.heatbath_module: heatbath_log_uniform, heatbath_uniform

mutable struct ReplaySiteRNG <: LatticeMatrices.SiteRNG
    values::Vector{Float64}
    index::Int
end

function _replay_uniform(rng::ReplaySiteRNG, ::Type{T}) where T
    value = T(rng.values[rng.index])
    rng.index += 1
    return rng, value
end

LatticeMatrices.rand_uniform(rng::ReplaySiteRNG, ::Type{Float32}) =
    _replay_uniform(rng, Float32)
LatticeMatrices.rand_uniform(rng::ReplaySiteRNG, ::Type{Float64}) =
    _replay_uniform(rng, Float64)
LatticeMatrices.rand_uniform_open(rng::ReplaySiteRNG, ::Type{Float32}) =
    _replay_uniform(rng, Float32)
LatticeMatrices.rand_uniform_open(rng::ReplaySiteRNG, ::Type{Float64}) =
    _replay_uniform(rng, Float64)

mutable struct ReplayPortableRNG
    values::Vector{Float64}
    index::Int
end

function _replay_portable_uniform(rng::ReplayPortableRNG, ::Type{T}) where T
    value = T(rng.values[rng.index])
    rng.index += 1
    return rng, value
end

heatbath_uniform(rng::ReplayPortableRNG, ::Type{T}) where T =
    _replay_portable_uniform(rng, T)
heatbath_log_uniform(rng::ReplayPortableRNG, ::Type{T}) where T =
    _replay_portable_uniform(rng, T)

function matrix_bits(A::AbstractMatrix{ComplexF64})
    return collect(reinterpret(UInt64, vec(Matrix(A))))
end

function _site_rng_su2_allocated(U, V, temps, rng)
    return @allocated SU2update_KP_rng!(U, V, 5.7, 2, temps, rng, 100_000)
end


@testset "clean portable and site-RNG SU(2) updates agree" begin
    staples = [
        ComplexF64[2 + 0.1im 0.2 - 0.3im; -0.1 + 0.4im 1.7 - 0.2im],
        ComplexF64[0.7 + 0.9im -0.8 + 0.2im; 0.3 - 0.6im 1.1 + 0.5im],
        ComplexF64[4 0; 0 4],
        ComplexF64[1.2 - 0.7im 0.5 + 0.8im; -0.9 - 0.4im 2.3 + 0.6im],
    ]
    betas = (0.1, 1.0, 5.7, 20.0)
    iteration_max = 100_000

    for V in staples, beta in betas, seed in 1:20
        Random.seed!(seed)
        replay_values = [rand() for _ in 1:2000]

        reference = fill(ComplexF64(NaN, NaN), 2, 2)
        reference_temps = [similar(reference), similar(reference)]
        portable_rng = ReplayPortableRNG(replay_values, 1)
        portable_rng, reference_accepted, reference_tries = SU2update_KP!(
            portable_rng,
            reference,
            V,
            beta,
            2,
            reference_temps,
            iteration_max,
        )

        candidate = fill(ComplexF64(NaN, NaN), 2, 2)
        candidate_temps = [similar(candidate), similar(candidate)]
        site_rng_state = ReplaySiteRNG(replay_values, 1)
        site_rng_state, accepted, tries = SU2update_KP_rng!(
            candidate,
            V,
            beta,
            2,
            candidate_temps,
            site_rng_state,
            iteration_max,
        )

        @test accepted == reference_accepted
        @test tries == reference_tries
        @test portable_rng.index == site_rng_state.index
        @test matrix_bits(candidate) == matrix_bits(reference)
        @test matrix_bits(candidate_temps[1]) == matrix_bits(reference_temps[1])
        @test matrix_bits(candidate_temps[2]) == matrix_bits(reference_temps[2])
    end

    # The clean portable and LM paths report rejection exhaustion without
    # consuming the four legacy compatibility draws.
    V = staples[1]
    Random.seed!(1)
    replay_values = [rand() for _ in 1:100]
    sentinel = ComplexF64[1 2; 3 4]

    reference = copy(sentinel)
    reference_temps = [similar(reference), similar(reference)]
    portable_rng = ReplayPortableRNG(replay_values, 1)
    portable_rng, reference_accepted, reference_tries = SU2update_KP!(
        portable_rng, reference, V, 0.1, 2, reference_temps, 1
    )

    candidate = copy(sentinel)
    candidate_temps = [similar(candidate), similar(candidate)]
    site_rng_state = ReplaySiteRNG(replay_values, 1)
    site_rng_state, accepted, tries = SU2update_KP_rng!(
        candidate, V, 0.1, 2, candidate_temps, site_rng_state, 1
    )
    @test !accepted
    @test accepted == reference_accepted
    @test tries == reference_tries == 1
    @test candidate == reference == sentinel
    @test portable_rng.index == site_rng_state.index == 5

    @testset "fixed clean-stream reference" begin
        values = [0.5, 0.5, 0.25, 0.1, 0.3, 0.7]
        expected = UInt64[
            0x3fef06f414416045, 0x3fb90eaabfcb087b,
            0xbfcb4d3e07a3f166, 0xbfb1bde28d91d658,
            0x3fcb4d3e07a3f166, 0xbfb1bde28d91d658,
            0x3fef06f414416045, 0xbfb90eaabfcb087b,
        ]
        U = fill(ComplexF64(NaN, NaN), 2, 2)
        temps = [similar(U), similar(U)]
        rng, accepted, tries = SU2update_KP!(
            ReplayPortableRNG(values, 1),
            U,
            ComplexF64[4 0; 0 4],
            5.7,
            2,
            temps,
            100,
        )

        @test accepted
        @test tries == 1
        @test rng.index == 7
        @test matrix_bits(U) == expected
    end

    @testset "released Julia-RNG reference" begin
        # Gaugefields v0.7.3 reference on the supported Julia 1.11 series.
        expected = UInt64[
            0xbfd576989dabe0f7, 0x3fe7257c0c05caa6,
            0xbfe0c1fd9b3fe149, 0xbfbf763656942642,
            0x3fe0c1fd9b3fe149, 0xbfbf763656942642,
            0xbfd576989dabe0f7, 0xbfe7257c0c05caa6,
        ]
        U = fill(ComplexF64(NaN, NaN), 2, 2)
        temps = [similar(U), similar(U)]

        Random.seed!(1)
        result = SU2update_KP!(U, staples[1], 0.1, 2, temps, iteration_max)
        @test result === nothing
        @test matrix_bits(U) == expected
        @test rand(UInt64) == 0xfdb6d3cef1d2a283

        fill!(U, ComplexF64(NaN, NaN))
        Random.seed!(1)
        allocation_free_result = SU2update_KP_allocationfree!(
            U, staples[1], 0.1, 2, temps, iteration_max
        )
        @test allocation_free_result === nothing
        @test matrix_bits(U) == expected
        @test rand(UInt64) == 0xfdb6d3cef1d2a283
    end

    @testset "all site RNG algorithms produce SU(2) matrices" begin
        for algorithm in (
            LatticeMatrices.PCG32(),
            LatticeMatrices.Xoshiro256PlusPlus(),
            LatticeMatrices.Philox4x32(),
        )
            for global_site in UInt64(0):UInt64(99)
                key = LatticeMatrices.RNGStreamKey(1234, 5, 2, 0, 0)
                rng = LatticeMatrices.site_rng(key, global_site, algorithm)
                U = LatticeMatrices.MMatrix{2,2,ComplexF64}(undef)
                Vstatic = LatticeMatrices.MMatrix{2,2,ComplexF64}(staples[3])
                temps = (
                    LatticeMatrices.MMatrix{2,2,ComplexF64}(undef),
                    LatticeMatrices.MMatrix{2,2,ComplexF64}(undef),
                )
                rng, accepted, tries = SU2update_KP_rng!(
                    U, Vstatic, 5.7, 2, temps, rng, iteration_max
                )
                @test accepted
                @test tries >= 1
                @test Matrix(U)' * Matrix(U) ≈
                      Matrix{ComplexF64}(I, 2, 2) atol = 1e-13
                @test det(Matrix(U)) ≈ 1 atol = 1e-13
            end

            key = LatticeMatrices.RNGStreamKey(1234, 5, 2, 0, 0)
            U = LatticeMatrices.MMatrix{2,2,ComplexF64}(undef)
            Vstatic = LatticeMatrices.MMatrix{2,2,ComplexF64}(staples[3])
            temps = (
                LatticeMatrices.MMatrix{2,2,ComplexF64}(undef),
                LatticeMatrices.MMatrix{2,2,ComplexF64}(undef),
            )
            rng = LatticeMatrices.site_rng(key, UInt64(0), algorithm)
            SU2update_KP_rng!(U, Vstatic, 5.7, 2, temps, rng, iteration_max)
            rng = LatticeMatrices.site_rng(key, UInt64(0), algorithm)
            @test _site_rng_su2_allocated(U, Vstatic, temps, rng) == 0
        end
    end
end
@testset "allocation-free SU(2) update matches SU2update_KP!" begin
    staples = [
        ComplexF64[2 + 0.1im 0.2 - 0.3im; -0.1 + 0.4im 1.7 - 0.2im],
        ComplexF64[0.7 + 0.9im -0.8 + 0.2im; 0.3 - 0.6im 1.1 + 0.5im],
        ComplexF64[4 0; 0 4],
        ComplexF64[1.2 - 0.7im 0.5 + 0.8im; -0.9 - 0.4im 2.3 + 0.6im],
    ]
    betas = (0.1, 1.0, 5.7, 20.0)
    seeds = 1:100
    iteration_max = 100_000

    reference = zeros(ComplexF64, 2, 2)
    allocation_free = similar(reference)
    temps = [similar(reference), similar(reference)]
    allocation_free_temps = [similar(reference), similar(reference)]

    for V in staples, beta in betas, seed in seeds
        fill!(reference, ComplexF64(NaN, NaN))
        fill!(allocation_free, ComplexF64(NaN, NaN))
        V_before = copy(V)

        Random.seed!(seed)
        SU2update_KP!(reference, V, beta, 2, temps, iteration_max)
        reference_rng_continuation = ntuple(_ -> rand(UInt64), 4)

        Random.seed!(seed)
        SU2update_KP_allocationfree!(
            allocation_free,
            V,
            beta,
            2,
            allocation_free_temps,
            iteration_max,
        )
        allocation_free_rng_continuation = ntuple(_ -> rand(UInt64), 4)

        # Compare the storage bits, rather than using an approximate numerical
        # comparison.  The RNG continuation also detects any difference in the
        # number or order of draws inside the rejection loop.
        @test matrix_bits(allocation_free) == matrix_bits(reference)
        @test matrix_bits(allocation_free_temps[1]) == matrix_bits(temps[1])
        @test matrix_bits(allocation_free_temps[2]) == matrix_bits(temps[2])
        @test allocation_free_rng_continuation == reference_rng_continuation
        @test V == V_before
    end

    # Warm up the exact method before measuring allocations.
    V = staples[1]
    Random.seed!(1234)
    SU2update_KP_allocationfree!(
        allocation_free,
        V,
        5.7,
        2,
        allocation_free_temps,
        iteration_max,
    )
    Random.seed!(1234)
    @test @allocated(
        SU2update_KP_allocationfree!(
            allocation_free,
            V,
            5.7,
            2,
            allocation_free_temps,
            iteration_max,
        )
    ) == 0

    # With no rejection the implementation consumes ten Float64 draws: four
    # overwritten draws, four draws for the first proposal, and two angles.
    # This low-beta case must advance farther, so the comparison above covers
    # repeated iterations of the rejection loop rather than only acceptance on
    # the first proposal.
    Random.seed!(1)
    SU2update_KP!(reference, staples[1], 0.1, 2, temps, iteration_max)
    continuation_after_rejection = rand(UInt64)
    Random.seed!(1)
    for _ in 1:10
        rand()
    end
    @test continuation_after_rejection != rand(UInt64)

    Random.seed!(1)
    reference_error = try
        SU2update_KP!(reference, staples[1], 0.1, 2, temps, 1)
        nothing
    catch err
        err
    end
    reference_rng_after_error = rand(UInt64)

    Random.seed!(1)
    allocation_free_error = try
        SU2update_KP_allocationfree!(
            allocation_free,
            staples[1],
            0.1,
            2,
            allocation_free_temps,
            1,
        )
        nothing
    catch err
        err
    end
    allocation_free_rng_after_error = rand(UInt64)

    @test reference_error isa ErrorException
    @test typeof(allocation_free_error) === typeof(reference_error)
    @test sprint(showerror, allocation_free_error) == sprint(showerror, reference_error)
    @test allocation_free_rng_after_error == reference_rng_after_error

    @testset "MMatrix site-local inputs" begin
        MMatrix = LatticeMatrices.MMatrix
        V_static = MMatrix{2,2,ComplexF64}(staples[1])
        reference_static = MMatrix{2,2,ComplexF64}(undef)
        allocation_free_static = MMatrix{2,2,ComplexF64}(undef)
        static_temps = (
            MMatrix{2,2,ComplexF64}(undef),
            MMatrix{2,2,ComplexF64}(undef),
        )

        # These Matrix temporaries reproduce the operand types used by the
        # current MPILattice heatbath closure.  Its MMatrix destination selects
        # LinearAlgebra's generic muladd-based multiplication path.
        for beta in betas, seed in seeds
            Random.seed!(seed)
            SU2update_KP!(
                reference_static,
                V_static,
                beta,
                2,
                temps,
                iteration_max,
            )
            reference_rng_continuation = ntuple(_ -> rand(UInt64), 4)

            Random.seed!(seed)
            SU2update_KP_allocationfree!(
                allocation_free_static,
                V_static,
                beta,
                2,
                static_temps,
                iteration_max,
            )
            allocation_free_rng_continuation = ntuple(_ -> rand(UInt64), 4)

            @test matrix_bits(allocation_free_static) == matrix_bits(reference_static)
            @test matrix_bits(static_temps[1]) == matrix_bits(temps[1])
            @test matrix_bits(static_temps[2]) == matrix_bits(temps[2])
            @test allocation_free_rng_continuation == reference_rng_continuation
        end

        Random.seed!(1234)
        SU2update_KP_allocationfree!(
            allocation_free_static,
            V_static,
            5.7,
            2,
            static_temps,
            iteration_max,
        )
        Random.seed!(1234)
        @test @allocated(
            SU2update_KP_allocationfree!(
                allocation_free_static,
                V_static,
                5.7,
                2,
                static_temps,
                iteration_max,
            )
        ) == 0
    end
end
