import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "Gaugefields MPILattice halo epochs" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    global_size = (4 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    U = Initialize_Gaugefields(
        2,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    A = U[1]

    @test !halo_is_dirty(A.U)
    for site in CartesianIndices(A.U.PN)
        local_indices = Tuple(site)
        global_indices = global_site_coordinates(A.U, local_indices)
        value = ComplexF64(global_indices[1])
        for jc in 1:A.NC, ic in 1:A.NC
            A[ic, jc, local_indices...] = value
        end
    end
    @test halo_is_dirty(A.U)

    shifted = shift_U(A, (1, 0, 0, 0))
    @test !halo_is_dirty(A.U)
    result_field = similar(A)
    substitute_U!(result_field, shifted)
    result = gather_matrix(result_field.U)

    if rank == 0
        original = Array{ComplexF64}(
            undef, A.NC, A.NC, global_size...
        )
        for site in CartesianIndices(global_size)
            original[:, :, Tuple(site)...] .= ComplexF64(site[1])
        end
        @test result == circshift(original, (0, 0, -1, 0, 0, 0))
    end
end
