import LatticeMatrices

function _nhyp_global(link)
    return LatticeMatrices.gather_and_bcast_matrix(link.U)
end

@testset "Gaugefields nHYP wrapper" begin
    lattice = (2, 2, 2, 2)
    process_grid = (1, 1, 1, 1)
    U = gauge_configuration(
        lattice;
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(8675309),
        process_grid,
        comm=SerialCommunicator(),
        verbose=0,
    )
    left = gauge_configuration(
        lattice;
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(314159),
        process_grid,
        comm=SerialCommunicator(),
        verbose=0,
    )

    specification = nhyp_smearing(
        U;
        alpha_outer=0.5,
        alpha_middle=0.5,
        alpha_inner=0.4,
    )
    @test specification isa NHYPSmearing
    @test specification.parameters.alpha_outer == 0.5
    @test specification.parameters.alpha_middle == 0.5
    @test specification.parameters.alpha_inner == 0.4

    recorded = smear(U, specification; record=true)
    @test length(recorded.configuration) == 4
    @test recorded.history isa NHYPSmearingCache
    @test recorded.derivative === nothing
    @test gauge_backend(recorded.configuration) isa LatticeMatricesBackend

    direct, _ = LatticeMatrices.nhyp_smear(
        [link.U for link in U],
        specification.parameters,
    )
    for mu in 1:4
        @test _nhyp_global(recorded.configuration[mu]) ≈
              LatticeMatrices.gather_and_bcast_matrix(direct[mu]) rtol=2e-12 atol=2e-12
    end

    smeared = similar(U)
    cache = NHYPSmearingCache(U, specification)
    @test nhyp_smear!(smeared, U, cache) === smeared
    for mu in 1:4
        @test _nhyp_global(smeared[mu]) ≈ _nhyp_global(recorded.configuration[mu])
    end

    dU = similar(U)
    @test nhyp_pullback!(dU, left, U, cache) === dU
    allocating_dU = nhyp_pullback(left, U, cache)
    for mu in 1:4
        @test _nhyp_global(allocating_dU[mu]) ≈ _nhyp_global(dU[mu])
    end

    direct_dU = [similar(link.U) for link in U]
    LatticeMatrices.nhyp_pullback!(
        direct_dU,
        [link.U for link in left],
        [link.U for link in U],
        cache.lattice_cache,
    )
    for mu in 1:4
        @test _nhyp_global(dU[mu]) ≈
              LatticeMatrices.gather_and_bcast_matrix(direct_dU[mu]) rtol=2e-12 atol=2e-12
    end

    @test length(smear(U, specification)) == 4
    @test_throws ArgumentError smear(
        U, specification; calcdSdU=true)
    @test_throws ArgumentError smear(
        U, specification; temps=similar(U))

    U32 = gauge_configuration(
        lattice;
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(271828),
        process_grid,
        comm=SerialCommunicator(),
        eltype=ComplexF32,
        verbose=0,
    )
    cache32 = NHYPSmearingCache(U32, specification)
    @test cache32.lattice_cache.parameters isa
          LatticeMatrices.NHYPParameters{Float32}

    no_halo = gauge_configuration(
        lattice;
        colors=3,
        halo=0,
        start=:cold,
        process_grid,
        comm=SerialCommunicator(),
        verbose=0,
    )
    @test_throws ArgumentError NHYPSmearingCache(no_halo, specification)

    two_dimensional = gauge_configuration(
        (2, 2);
        colors=3,
        halo=1,
        start=:cold,
        process_grid=(1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )
    @test_throws ArgumentError nhyp_smearing(two_dimensional)

    legacy = gauge_configuration(
        lattice;
        backend=LegacyBackend(),
        colors=3,
        halo=1,
        start=:cold,
        verbose=0,
    )
    @test_throws ArgumentError nhyp_smearing(legacy)

    LatticeMatrices.add_matrix!(U[1].U, left[1].U, 1e-7)
    @test_throws ArgumentError nhyp_pullback!(dU, left, U, cache)
end
