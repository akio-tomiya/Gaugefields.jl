import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using MPI
using Test

import Gaugefields.Temporalfields_module: get_temp, unused!

MPI.Initialized() || MPI.Init()

@inline function _host_field_value(storage, field, ic, jc, local_indices)
    array_indices = ntuple(
        d -> local_indices[d] + field.U.nw,
        length(local_indices),
    )
    return storage[ic, jc, array_indices...]
end

function _reference_heatbath_color!(
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
    key = RNGStreamKey(seed, sweep, direction, color, 0)
    element_type = eltype(link.U)

    for site in CartesianIndices(global_size)
        global_indices = Tuple(site)
        iseven(sum(global_indices)) == target_even || continue

        link_local = LatticeMatrices.MMatrix{2,2,element_type}(undef)
        staple_local = LatticeMatrices.MMatrix{2,2,element_type}(undef)
        temp1 = LatticeMatrices.MMatrix{2,2,element_type}(undef)
        temp2 = LatticeMatrices.MMatrix{2,2,element_type}(undef)
        for jc in 1:2, ic in 1:2
            link_local[ic, jc] = link[ic, jc, global_indices...]
            staple_local[ic, jc] = staple[ic, jc, global_indices...]
        end

        global_site = global_site_id(global_indices, global_size)
        rng = site_rng(key, global_site, algorithm)
        _, accepted, _ = SU2update_KP_rng!(
            link_local,
            staple_local,
            beta,
            2,
            (temp1, temp2),
            rng,
            iteration_max,
        )
        @test accepted

        for jc in 1:2, ic in 1:2
            link[ic, jc, global_indices...] = link_local[ic, jc]
        end
    end
    return nothing
end

"""
Serial full-sweep reference using Gaugefields_4D_nowing storage.  Its staple,
shift, checkerboard arithmetic, and field access do not use LatticeMatrices;
only the deliberately shared public site-RNG/local-SU(2) APIs are used so the
parallel and serial sweeps consume identical random streams.
"""
function _reference_heatbath_sweep!(
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
                _reference_heatbath_color!(
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

function _compare_with_reference(U, reference; atol=2e-12, rtol=2e-12)
    host_fields = map(field -> JACC.to_host(field.U.A), U)
    for direction in eachindex(U)
        field = U[direction]
        storage = host_fields[direction]
        for site in CartesianIndices(field.U.PN)
            local_indices = Tuple(site)
            global_indices = global_site_coordinates(field.U, local_indices)
            for jc in 1:2, ic in 1:2
                @test _host_field_value(
                    storage, field, ic, jc, local_indices
                ) ≈ reference[direction][ic, jc, global_indices...] atol = atol rtol = rtol
            end
        end
    end
    return nothing
end

@testset "MPILattice complete SU(2) plaquette heatbath sweep" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    beta = 2.3
    seed = 0x6a09e667f3bcc909
    initial_sweep = 17

    algorithms = (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
    for algorithm in algorithms
        U = Initialize_Gaugefields(
            2,
            1,
            global_size...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )
        reference = Initialize_Gaugefields(
            2,
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

        _reference_heatbath_sweep!(
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
        _compare_with_reference(U, reference)

        # A second evolved sweep checks that the public state advances and
        # that later staples see halos from the preceding color/direction.
        if algorithm isa Philox4x32
            _reference_heatbath_sweep!(
                reference,
                beta;
                seed,
                sweep=initial_sweep + 1,
                algorithm,
            )
            heatbath!(U, h)
            @test h.sweep == UInt64(initial_sweep + 2)
            _compare_with_reference(U, reference; atol=5e-12, rtol=5e-12)
        end
    end
end

@testset "full heatbath sweep counter advances only on success" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size = (2 * nprocs, 2, 2, 2)
    U = Initialize_Gaugefields(
        2,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        verbose_level=0,
    )
    initial_sweep = 23
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
