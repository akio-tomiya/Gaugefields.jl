import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices Bridge text round-trip" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    global_size = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    filename = MPI.bcast(tempname(), 0, comm)

    U = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    restored = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )

    save_textdata(U, filename)
    load_BridgeText!(filename, restored, global_size, 3)

    for direction in eachindex(U)
        original = gather_and_bcast_matrix(U[direction].U)
        loaded = gather_and_bcast_matrix(restored[direction].U)
        @test maximum(abs, loaded .- original) == 0
    end

    MPI.Barrier(comm)
    rank == 0 && Base.Filesystem.rm(filename; force=true)
end
