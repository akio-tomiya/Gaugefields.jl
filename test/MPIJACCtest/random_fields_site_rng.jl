import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

function _maximum_unitarity_error(field)
    identity_matrix = Matrix{eltype(field)}(I, size(field, 1), size(field, 2))
    maximum_error = 0.0
    for site in CartesianIndices(size(field)[3:end])
        matrix = @view field[:, :, Tuple(site)...]
        maximum_error = max(maximum_error, norm(matrix' * matrix - identity_matrix))
    end
    return maximum_error
end

function _momentum_reference(global_size, num_basis, seed, sweep, direction, algorithm, sigma, ::Type{T}) where {T}
    tag = Gaugefields.AbstractGaugefields_module._GAUSSIAN_MOMENTUM_STREAM_TAG
    key = RNGStreamKey(seed, sweep, direction, 0, tag)
    reference = Array{T}(undef, num_basis, 1, global_size...)
    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        stream = site_rng(key, global_site_id(global_indices, global_size), algorithm)
        use_spare = false
        spare = zero(T)
        for component in 1:num_basis
            if use_spare
                value = spare
            else
                stream, value, spare = rand_normal_pair(stream, T)
            end
            reference[component, 1, global_indices...] = T(sigma) * value
            use_spare = !use_spare
        end
    end
    return reference
end

if !("gaussian-only" in ARGS)
@testset "MPILattice hot start global-site RNG" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    seed = 0x6a09e667f3bcc909

    for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
        global_size = (8, 6)
        process_grid = (nprocs, 1)
        fields = Initialize_Gaugefields(
            3,
            1,
            global_size...;
            condition="hot",
            isMPILattice=true,
            PEs=process_grid,
            seed,
            rng_algorithm=algorithm,
            verbose_level=0,
        )
        repeated = Initialize_Gaugefields(
            3,
            1,
            global_size...;
            condition="hot",
            isMPILattice=true,
            PEs=process_grid,
            seed,
            rng_algorithm=algorithm,
            verbose_level=0,
        )

        global_fields = gather_and_bcast_matrix.(getproperty.(fields, :U))
        global_repeated = gather_and_bcast_matrix.(getproperty.(repeated, :U))
        @test global_fields == global_repeated
        @test global_fields[1] != global_fields[2]
        @test all(field -> _maximum_unitarity_error(field) < 2e-12, global_fields)
    end

    reproducible_a = Initialize_Gaugefields(
        2,
        1,
        8,
        4;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(nprocs, 1),
        verbose_level=0,
    )
    reproducible_b = Initialize_Gaugefields(
        2,
        1,
        8,
        4;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(nprocs, 1),
        verbose_level=0,
    )
    global_reproducible_a =
        gather_and_bcast_matrix.(getproperty.(reproducible_a, :U))
    global_reproducible_b =
        gather_and_bcast_matrix.(getproperty.(reproducible_b, :U))
    @test global_reproducible_a == global_reproducible_b
    @test global_reproducible_a[1] != global_reproducible_a[2]

    reproducible_4d_a = Initialize_Gaugefields(
        3,
        1,
        4,
        2,
        2,
        2;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        verbose_level=0,
    )
    reproducible_4d_b = Initialize_Gaugefields(
        3,
        1,
        4,
        2,
        2,
        2;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        verbose_level=0,
    )
    global_reproducible_4d_a =
        gather_and_bcast_matrix.(getproperty.(reproducible_4d_a, :U))
    global_reproducible_4d_b =
        gather_and_bcast_matrix.(getproperty.(reproducible_4d_b, :U))
    @test global_reproducible_4d_a == global_reproducible_4d_b
    @test all(
        global_reproducible_4d_a[i] != global_reproducible_4d_a[j]
        for i in eachindex(global_reproducible_4d_a)
        for j in (i+1):length(global_reproducible_4d_a)
    )

    if nprocs > 1
        decomposition_size = (2 * nprocs, 2 * nprocs)
        decomposition_x = Initialize_Gaugefields(
            2,
            1,
            decomposition_size...;
            condition="hot",
            randomnumber="Reproducible",
            isMPILattice=true,
            PEs=(nprocs, 1),
            verbose_level=0,
        )
        decomposition_y = Initialize_Gaugefields(
            2,
            1,
            decomposition_size...;
            condition="hot",
            randomnumber="Reproducible",
            isMPILattice=true,
            PEs=(1, nprocs),
            verbose_level=0,
        )
        @test gather_and_bcast_matrix.(getproperty.(decomposition_x, :U)) ==
              gather_and_bcast_matrix.(getproperty.(decomposition_y, :U))
    end

    fields_4d = Initialize_Gaugefields(
        3,
        1,
        4,
        2,
        2,
        2;
        condition="hot",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        seed,
        rng_algorithm=Philox4x32(),
        verbose_level=0,
    )
    @test all(
        field -> _maximum_unitarity_error(gather_and_bcast_matrix(field.U)) < 2e-12,
        fields_4d,
    )
end
end

if !("hot-only" in ARGS)
@testset "MPILattice Gaussian momentum global-site RNG" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (32, 32)
    process_grid = (nprocs, 1)
    seed = 0xbb67ae8584caa73b
    sweep = 23
    sigma = 1.7

    cold = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )

    for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
        momentum = initialize_TA_Gaugefields(cold)
        repeated = initialize_TA_Gaugefields(cold)
        gauss_distribution!(momentum; σ=sigma, seed, sweep, rng_algorithm=algorithm)
        gauss_distribution!(repeated; σ=sigma, seed, sweep, rng_algorithm=algorithm)

        global_momentum = gather_and_bcast_matrix.(getproperty.(momentum, :a))
        global_repeated = gather_and_bcast_matrix.(getproperty.(repeated, :a))
        @test global_momentum == global_repeated
        @test global_momentum[1] != global_momentum[2]

        reference = _momentum_reference(
            global_size,
            8,
            seed,
            sweep,
            1,
            algorithm,
            sigma,
            Float64,
        )
        @test global_momentum[1] ≈ reference rtol = 2e-12 atol = 2e-12

        values = global_momentum[1]
        sample_mean = sum(values) / length(values)
        sample_std = sqrt(sum(abs2, values) / length(values) - sample_mean^2)
        @test abs(sample_mean) < 0.04 * sigma
        @test abs(sample_std - sigma) < 0.04 * sigma
    end

    cold_4d = Initialize_Gaugefields(
        3,
        1,
        4,
        2,
        2,
        2;
        condition="cold",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        verbose_level=0,
    )
    momentum_4d = initialize_TA_Gaugefields(cold_4d)
    gauss_distribution!(
        momentum_4d;
        σ=sigma,
        seed,
        sweep,
        rng_algorithm=Philox4x32(),
    )
    expected_4d = _momentum_reference(
        (4, 2, 2, 2),
        8,
        seed,
        sweep,
        4,
        Philox4x32(),
        sigma,
        Float64,
    )
    @test gather_and_bcast_matrix(momentum_4d[4].a) ≈ expected_4d rtol = 2e-12 atol = 2e-12
end
end
