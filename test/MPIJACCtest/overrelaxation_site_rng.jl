import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

@inline function _overrelaxation_site_host_value(
    storage,
    field,
    ic,
    jc,
    local_indices,
)
    indices = ntuple(
        d -> local_indices[d] + field.U.nw,
        length(local_indices),
    )
    return storage[ic, jc, indices...]
end
@testset "MPILattice overrelaxation site RNG algorithms" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1, 1, 1)

    for singleprecision in (false, true), nc in (2, 3)
        U = Initialize_Gaugefields(
            nc,
            1,
            4,
            4,
            4,
            4;
            condition="hot",
            isMPILattice=true,
            PEs=process_grid,
            singleprecision,
            verbose_level=0,
        )
        staple = similar(U[1])
        substitute_U!(staple, U[2])

        for algorithm in (PCG32(), Xoshiro256PlusPlus(), Philox4x32())
            if nc == 2
                overrelaxation_su2_sites!(
                    U[1],
                    staple,
                    true;
                    seed=1234,
                    sweep=2,
                    direction=1,
                    rng_algorithm=algorithm,
                )
            else
                overrelaxation_su3_sites!(
                    U[1],
                    staple,
                    true;
                    seed=1234,
                    sweep=2,
                    direction=1,
                    rng_algorithm=algorithm,
                )
            end
        end

        tolerance = singleprecision ? 4f-5 : 8e-13
        storage = JACC.to_host(U[1].U.A)
        for site in CartesianIndices(U[1].U.PN)
            local_indices = Tuple(site)
            matrix = [
                _overrelaxation_site_host_value(
                    storage, U[1], ic, jc, local_indices
                )
                for ic in 1:nc, jc in 1:nc
            ]
            @test matrix' * matrix ≈
                  Matrix{eltype(matrix)}(I, nc, nc) atol = tolerance rtol = tolerance
        end
    end
end
