import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix, halo_epochs
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "MPI LatticeMatrices ILDG round-trip" begin
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    global_size = (2 * nprocs, 2, 2, 2 * nprocs)
    process_grids = nprocs == 1 ? ((1, 1, 1, 1),) :
        ((nprocs, 1, 1, 1), (1, 1, 1, nprocs))

    for process_grid in process_grids, precision in (64, 32)
        prefix = MPI.bcast(
            rank == 0 ? tempname(pwd()) : "",
            0,
            comm,
        )
        filename = prefix * ".ildg"
        payload = prefix * ".payload"
        filelist = prefix * ".list"

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
        epochs_before = halo_epochs(restored[1].U)

        save_binarydata(
            U,
            filename;
            precision,
            tempfile1=payload,
            tempfile2=filelist,
        )
        ildg = ILDG(filename)
        @test ildg[1]["L"] == global_size
        @test ildg[1]["precision"] == precision
        load_gaugefield!(restored, 1, ildg, global_size, 3)

        epochs_after = halo_epochs(restored[1].U)
        @test epochs_after.core > epochs_before.core
        @test epochs_after.halo == epochs_after.core

        for direction in eachindex(U)
            original = gather_and_bcast_matrix(U[direction].U)
            loaded = gather_and_bcast_matrix(restored[direction].U)
            expected = precision == 64 ? original :
                ComplexF64.(ComplexF32.(original))
            @test maximum(abs, loaded .- expected) <= eps(Float32)
        end

        MPI.Barrier(comm)
        if rank == 0
            rm(filename; force=true)
            rm(payload; force=true)
            rm(filelist; force=true)
        end
        MPI.Barrier(comm)
    end
end
