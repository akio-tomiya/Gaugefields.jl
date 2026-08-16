import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

host_storage(U) = JACC.to_host(U.U.A)

@inline function host_field_value(storage, U, ic, jc, local_indices)
    array_indices = ntuple(d -> local_indices[d] + U.U.nw, length(local_indices))
    return storage[ic, jc, array_indices...]
end

@inline function kernel_set_constant_staple!(i, data, dindexer, ::Val{nw}, values) where nw
    indices = LatticeMatrices.delinearize(dindexer, i, nw)
    @inbounds begin
        data[1, 1, indices...] = values[1]
        data[2, 1, indices...] = values[2]
        data[1, 2, indices...] = values[3]
        data[2, 2, indices...] = values[4]
    end
    return nothing
end

function set_constant_staple!(staple, values)
    JACC.parallel_for(
        prod(staple.U.PN),
        kernel_set_constant_staple!,
        staple.U.A,
        staple.U.indexer,
        Val(staple.U.nw),
        values,
    )
    return nothing
end

function reference_site_update(
    global_indices,
    global_size,
    key,
    algorithm,
    beta,
    staple_values,
)
    global_site = LatticeMatrices.global_site_id(global_indices, global_size)
    rng = LatticeMatrices.site_rng(key, global_site, algorithm)
    U = LatticeMatrices.MMatrix{2,2,ComplexF64}(Matrix{ComplexF64}(I, 2, 2))
    V = LatticeMatrices.MMatrix{2,2,ComplexF64}(
        staple_values[1], staple_values[2], staple_values[3], staple_values[4]
    )
    temps = (
        LatticeMatrices.MMatrix{2,2,ComplexF64}(undef),
        LatticeMatrices.MMatrix{2,2,ComplexF64}(undef),
    )
    rng, accepted, _ = Gaugefields.SU2update_KP_rng!(
        U, V, beta, 2, temps, rng, 100_000
    )
    @test accepted
    return U
end

function make_mpi_and_reference_fields(global_size, process_grid)
    mpi_fields = Initialize_Gaugefields(
        2,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    reference_fields = Initialize_Gaugefields(
        2,
        0,
        global_size...;
        condition="cold",
        verbose_level=0,
    )
    return mpi_fields[1], mpi_fields[2], reference_fields[1]
end

@testset "site-RNG MPILattice SU(2) heatbath kernel" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    shift = (1, 0, 0, 0)
    beta = 5.7
    seed = 0x123456789abcdef0
    sweep = 11
    direction = 3
    staple_values = (
        2.0 + 0.3im,
        -0.4 + 0.2im,
        0.4 + 0.2im,
        2.0 - 0.3im,
    )

    for algorithm in (
        LatticeMatrices.PCG32(),
        LatticeMatrices.Xoshiro256PlusPlus(),
        LatticeMatrices.Philox4x32(),
    )
        for target_even in (true, false)
            color = target_even ? 0 : 1
            U, staple, reference = make_mpi_and_reference_fields(
                global_size, process_grid
            )
            set_constant_staple!(staple, staple_values)
            key = LatticeMatrices.RNGStreamKey(
                seed, sweep, direction, color, 0
            )

            for site in CartesianIndices(global_size)
                global_indices = Tuple(site)
                if iseven(sum(global_indices)) == target_even
                    expected = reference_site_update(
                        global_indices,
                        global_size,
                        key,
                        algorithm,
                        beta,
                        staple_values,
                    )
                    for jc in 1:2, ic in 1:2
                        reference[ic, jc, global_indices...] = expected[ic, jc]
                    end
                end
            end

            Gaugefields.heatbath_su2_sites!(
                U,
                staple,
                beta,
                target_even;
                seed,
                sweep,
                direction,
                color,
                rng_algorithm=algorithm,
            )

            shifted = similar(U)
            Gaugefields.substitute_U!(shifted, Gaugefields.shift_U(U, shift))
            shifted_reference = Gaugefields.shift_U(reference, shift)
            U_host = host_storage(U)
            shifted_host = host_storage(shifted)

            for site in CartesianIndices(U.U.PN)
                local_indices = Tuple(site)
                global_indices = LatticeMatrices.global_site_coordinates(
                    U.U, local_indices
                )
                for jc in 1:2, ic in 1:2
                    # The no-wing Gaugefields field is the global, non-
                    # LatticeMatrices reference implementation.
                    @test host_field_value(U_host, U, ic, jc, local_indices) ≈
                          reference[ic, jc, global_indices...] atol = 5e-14 rtol = 5e-14
                    @test host_field_value(
                        shifted_host, shifted, ic, jc, local_indices
                    ) ≈ shifted_reference[ic, jc, global_indices...] atol = 5e-14 rtol = 5e-14
                end
            end
        end
    end
end

@testset "Float32 site-RNG heatbath kernel" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    fields = Initialize_Gaugefields(
        2,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        singleprecision=true,
        verbose_level=0,
    )
    U = fields[1]
    staple = fields[2]
    Gaugefields.heatbath_su2_sites!(
        U,
        staple,
        5.7f0,
        true;
        seed=123,
        sweep=2,
        direction=1,
        color=0,
    )
    U_host = host_storage(U)

    for site in CartesianIndices(U.U.PN)
        local_indices = Tuple(site)
        global_indices = LatticeMatrices.global_site_coordinates(U.U, local_indices)
        matrix = [
            host_field_value(U_host, U, ic, jc, local_indices)
            for ic in 1:2, jc in 1:2
        ]
        @test eltype(matrix) == ComplexF32
        if iseven(sum(global_indices))
            @test matrix' * matrix ≈ Matrix{ComplexF32}(I, 2, 2) atol = 2f-6
            @test det(matrix) ≈ ComplexF32(1) atol = 2f-6
        else
            @test matrix == Matrix{ComplexF32}(I, 2, 2)
        end
    end
end

@testset "heatbath kernel reproducibility and stream tags" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)

    U1, staple1, _ = make_mpi_and_reference_fields(global_size, process_grid)
    U2, staple2, _ = make_mpi_and_reference_fields(global_size, process_grid)
    U3, staple3, _ = make_mpi_and_reference_fields(global_size, process_grid)

    kwargs = (; seed=9876, sweep=4, direction=2, color=0)
    Gaugefields.heatbath_su2_sites!(U1, staple1, 5.7, true; kwargs...)
    Gaugefields.heatbath_su2_sites!(U2, staple2, 5.7, true; kwargs...)
    Gaugefields.heatbath_su2_sites!(
        U3, staple3, 5.7, true; kwargs..., subgroup=1
    )
    U1_host = host_storage(U1)
    U2_host = host_storage(U2)
    U3_host = host_storage(U3)

    found_different_stream = false
    for site in CartesianIndices(U1.U.PN)
        local_indices = Tuple(site)
        global_indices = LatticeMatrices.global_site_coordinates(U1.U, local_indices)
        target = iseven(sum(global_indices))
        for jc in 1:2, ic in 1:2
            value1 = host_field_value(U1_host, U1, ic, jc, local_indices)
            value2 = host_field_value(U2_host, U2, ic, jc, local_indices)
            value3 = host_field_value(U3_host, U3, ic, jc, local_indices)
            @test value1 == value2
            if target && value1 != value3
                found_different_stream = true
            end
        end
    end
    @test MPI.Allreduce(found_different_stream ? 1 : 0, MPI.SUM, MPI.COMM_WORLD) > 0
end

@testset "heatbath kernel reports rejection on the host" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (4, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    U, staple, _ = make_mpi_and_reference_fields(global_size, process_grid)

    @test_throws ErrorException Gaugefields.heatbath_su2_sites!(
        U,
        staple,
        eps(Float64),
        true;
        seed=1,
        sweep=0,
        direction=1,
        color=0,
        iteration_max=1,
    )
    U_host = host_storage(U)

    for site in CartesianIndices(U.U.PN)
        local_indices = Tuple(site)
        @test [
            host_field_value(U_host, U, ic, jc, local_indices)
            for ic in 1:2, jc in 1:2
        ] ==
              Matrix{ComplexF64}(I, 2, 2)
    end
end
