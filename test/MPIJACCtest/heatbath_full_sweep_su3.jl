import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

import Gaugefields.Temporalfields_module: get_temp, unused!

MPI.Initialized() || MPI.Init()

@inline function _su3_host_field_value(storage, field, ic, jc, local_indices)
    array_indices = ntuple(
        d -> local_indices[d] + field.U.nw,
        length(local_indices),
    )
    return storage[ic, jc, array_indices...]
end

function _reference_su3_heatbath_color!(
    link,
    staple,
    beta,
    target_even;
    seed,
    sweep,
    direction,
    algorithm,
    iteration_max,
)
    global_size = (link.NX, link.NY, link.NZ, link.NT)
    color = target_even ? 0 : 1
    element_type = eltype(link.U)

    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        iseven(sum(global_indices)) == target_even || continue

        link_local = LatticeMatrices.MMatrix{3,3,element_type}(undef)
        staple_local = LatticeMatrices.MMatrix{3,3,element_type}(undef)
        temps2 = ntuple(
            _ -> LatticeMatrices.MMatrix{2,2,element_type}(undef), 4
        )
        temps3 = ntuple(
            _ -> LatticeMatrices.MMatrix{3,3,element_type}(undef), 3
        )
        for jc in 1:3, ic in 1:3
            link_local[ic, jc] = link[ic, jc, global_indices...]
            staple_local[ic, jc] = staple[ic, jc, global_indices...]
        end

        global_site = global_site_id(global_indices, global_size)
        rngs = ntuple(3) do subgroup
            key = RNGStreamKey(seed, sweep, direction, color, subgroup)
            site_rng(key, global_site, algorithm)
        end
        _, accepted, failed_subgroup = SU3update_matrix_rng!(
            link_local,
            staple_local,
            beta,
            temps2,
            temps3,
            rngs,
            iteration_max,
        )
        @test accepted
        @test failed_subgroup == 0

        for jc in 1:3, ic in 1:3
            link[ic, jc, global_indices...] = link_local[ic, jc]
        end
    end
    return nothing
end

"""
Serial SU(3) full-sweep reference.  Staple construction, shifts, checkerboard
operations, and field storage are provided by `Gaugefields_4D_nowing`, so the
lattice backend under comparison does not use LatticeMatrices.  The public
site-RNG/local-SU(3) APIs are deliberately shared to make every random stream
identical to the accelerator sweep.
"""
function _reference_su3_heatbath_sweep!(
    U,
    beta;
    seed,
    sweep,
    algorithm,
    iteration_max=100_000,
)
    temps_g = Temporalfields(U[1], num=5)
    staple, staple_index = get_temp(temps_g)
    temps, temp_indices = get_temp(temps_g, 4)
    try
        for direction in eachindex(U)
            loops = loops_staple[(4, direction)]
            for target_even in (true, false)
                evaluate_gaugelinks_evenodd!(
                    staple, loops, U, temps, target_even
                )
                _reference_su3_heatbath_color!(
                    U[direction],
                    staple,
                    beta,
                    target_even;
                    seed,
                    sweep,
                    direction,
                    algorithm,
                    iteration_max,
                )
            end
        end
    finally
        unused!(temps_g, staple_index)
        unused!(temps_g, temp_indices)
    end
    return nothing
end

function _compare_su3_with_reference(U, reference; atol=1e-11, rtol=1e-11)
    host_fields = map(field -> JACC.to_host(field.U.A), U)
    for direction in eachindex(U)
        field = U[direction]
        storage = host_fields[direction]
        for site in CartesianIndices(field.U.PN)
            local_indices = Tuple(site)
            global_indices = global_site_coordinates(field.U, local_indices)
            for jc in 1:3, ic in 1:3
                @test _su3_host_field_value(
                    storage, field, ic, jc, local_indices
                ) ≈ reference[direction][ic, jc, global_indices...] atol = atol rtol = rtol
            end
        end
    end
    return nothing
end

@testset "MPILattice complete SU(3) plaquette heatbath sweep" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    beta = 5.7
    seed = 0xbb67ae8584caa73b
    initial_sweep = 29

    for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
        U = Initialize_Gaugefields(
            3,
            1,
            global_size...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )
        reference = Initialize_Gaugefields(
            3,
            0,
            global_size...;
            condition="cold",
            verbose_level=0,
        )
        h = Heatbath(
            U,
            beta;
            seed,
            sweep=initial_sweep,
            rng_algorithm=algorithm,
        )

        _reference_su3_heatbath_sweep!(
            reference,
            beta;
            seed,
            sweep=initial_sweep,
            algorithm,
        )
        heatbath!(U, h)

        @test h.seed == UInt64(seed)
        @test h.sweep == UInt64(initial_sweep + 1)
        @test typeof(h.rng_algorithm) == typeof(algorithm)
        _compare_su3_with_reference(U, reference)

        if algorithm isa Philox4x32
            _reference_su3_heatbath_sweep!(
                reference,
                beta;
                seed,
                sweep=initial_sweep + 1,
                algorithm,
            )
            heatbath!(U, h)
            @test h.sweep == UInt64(initial_sweep + 2)
            _compare_su3_with_reference(
                U, reference; atol=3e-11, rtol=3e-11
            )
        end
    end
end

@testset "SU(3) full sweep failure preserves counter" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    U = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        verbose_level=0,
    )
    initial_sweep = 31
    h = Heatbath(
        U,
        1e-12;
        ITERATION_MAX=1,
        seed=123,
        sweep=initial_sweep,
        rng_algorithm=Philox4x32(),
    )

    @test_throws ErrorException heatbath!(U, h)
    @test h.sweep == UInt64(initial_sweep)
end

@testset "Float32 SU(3) full heatbath sweep" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    U = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        singleprecision=true,
        verbose_level=0,
    )
    h = Heatbath(
        U,
        5.7f0;
        seed=0x3c6ef372fe94f82b,
        sweep=4,
        rng_algorithm=Philox4x32(),
    )
    heatbath!(U, h)
    @test h.sweep == UInt64(5)

    for field in U
        storage = JACC.to_host(field.U.A)
        for site in CartesianIndices(field.U.PN)
            local_indices = Tuple(site)
            matrix = [
                _su3_host_field_value(storage, field, ic, jc, local_indices)
                for ic in 1:3, jc in 1:3
            ]
            @test eltype(matrix) == ComplexF32
            @test all(isfinite, matrix)
            @test matrix' * matrix ≈ Matrix{ComplexF32}(I, 3, 3) atol = 8f-6
            @test det(matrix) ≈ ComplexF32(1) atol = 8f-6
        end
    end
end
