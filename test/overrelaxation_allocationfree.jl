using Gaugefields
using LatticeMatrices
using LinearAlgebra
using Random
using Test

mutable struct ReplayOverrelaxationRNG <: LatticeMatrices.SiteRNG
    values::Vector{UInt32}
    index::Int
end

function LatticeMatrices.rand_u32(rng::ReplayOverrelaxationRNG)
    value = rng.values[rng.index]
    rng.index += 1
    return rng, value
end

function _legacy_overrelaxation_pairs(nc, seed)
    Random.seed!(seed)
    return ntuple(nc) do _
        n = rand(1:(nc-1))
        m = rand(n:nc)
        while n == m
            m = rand(n:nc)
        end
        (n, m)
    end
end

function _pair_replay_values(pairs)
    values = UInt32[]
    for (n, m) in pairs
        push!(values, UInt32(n - 1))
        push!(values, UInt32(m - n - 1))
    end
    return values
end

function _overrelaxation_test_link(nc)
    if nc == 2
        return ComplexF64[
            0.5+0.5im 0.5+0.5im
            -0.5+0.5im 0.5-0.5im
        ]
    end
    c1, s1 = 3 / 5, 4 / 5
    c2, s2 = 5 / 13, 12 / 13
    return ComplexF64[
        c1 s1*c2 s1*s2
        -s1 c1*c2 c1*s2
        0 -s2 c2
    ]
end

function _overrelaxation_test_staple(nc)
    if nc == 2
        return ComplexF64[
            2.7+0.2im -0.4+0.7im
            0.3-0.5im 2.2-0.1im
        ]
    end
    return ComplexF64[
        3.1+0.2im -0.4+0.1im 0.2-0.3im
        0.3+0.5im 2.7-0.1im -0.2+0.4im
        -0.1+0.2im 0.4-0.3im 2.9+0.3im
    ]
end

function _run_local_overrelaxation(u, V, rng, nc)
    temps2 = ntuple(_ -> zeros(eltype(u), 2, 2), 2)
    tempsN = ntuple(_ -> zeros(eltype(u), nc, nc), 3)
    return SUN_overrelaxation_rng!(
        u, V, temps2, tempsN, rng, Val(nc)
    )
end

@testset "allocation-free overrelaxation matches legacy subgroup updates" begin
    for nc in (2, 3), seed in 1:40
        initial = _overrelaxation_test_link(nc)
        staple = _overrelaxation_test_staple(nc)
        pairs = _legacy_overrelaxation_pairs(nc, seed)

        reference = copy(initial)
        reference_temps2 = [zeros(ComplexF64, 2, 2) for _ in 1:5]
        reference_tempsN = [zeros(ComplexF64, nc, nc) for _ in 1:5]
        Random.seed!(seed)
        Gaugefields.heatbath_module.SUN_overrelaxation!(
            reference,
            staple,
            1.0,
            nc,
            reference_temps2,
            reference_tempsN,
            100_000,
        )

        candidate = copy(initial)
        replay_rng = ReplayOverrelaxationRNG(
            _pair_replay_values(pairs), 1
        )
        _, success = _run_local_overrelaxation(
            candidate, staple, replay_rng, nc
        )

        @test success
        @test candidate ≈ reference atol = 2e-14 rtol = 2e-14
        @test candidate' * candidate ≈ Matrix{ComplexF64}(I, nc, nc) atol = 3e-14
        @test det(candidate) ≈ det(reference) atol = 3e-14
        @test real(tr(candidate * staple)) ≈
              real(tr(initial * staple)) atol = 5e-14 rtol = 5e-14
    end
end

function _static_overrelaxation(::Val{NC}, ::Type{T}, algorithm) where {NC,T}
    u = LatticeMatrices.MMatrix{NC,NC,T}(
        T.(_overrelaxation_test_link(NC))
    )
    staple = LatticeMatrices.MMatrix{NC,NC,T}(
        T.(_overrelaxation_test_staple(NC))
    )
    temps2 = ntuple(_ -> LatticeMatrices.MMatrix{2,2,T}(undef), 2)
    tempsN = ntuple(
        _ -> LatticeMatrices.MMatrix{NC,NC,T}(undef), 3
    )
    rng = site_rng(
        RNGStreamKey(1234, 7, 2, 1, typemax(UInt32)),
        UInt64(19),
        algorithm,
    )
    return u, staple, temps2, tempsN, rng
end

function _allocated_static_overrelaxation(
    u,
    staple,
    temps2,
    tempsN,
    rng,
    nc,
)
    return @allocated SUN_overrelaxation_rng!(
        u, staple, temps2, tempsN, rng, nc
    )
end

@testset "site-RNG overrelaxation is allocation-free" begin
    for nc in (2, 3)
        for T in (ComplexF32, ComplexF64)
            tolerance = T == ComplexF32 ? 3f-5 : 5e-14
            for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
                u, staple, temps2, tempsN, rng =
                    _static_overrelaxation(Val(nc), T, algorithm)
                _, success = SUN_overrelaxation_rng!(
                    u, staple, temps2, tempsN, rng, Val(nc)
                )
                @test success
                @test Matrix(u)' * Matrix(u) ≈
                      Matrix{T}(I, nc, nc) atol = tolerance rtol = tolerance

                u, staple, temps2, tempsN, rng =
                    _static_overrelaxation(Val(nc), T, algorithm)
                SUN_overrelaxation_rng!(
                    u, staple, temps2, tempsN, rng, Val(nc)
                )
                @test _allocated_static_overrelaxation(
                    u, staple, temps2, tempsN, rng, Val(nc)
                ) == 0
            end
        end
    end
end
