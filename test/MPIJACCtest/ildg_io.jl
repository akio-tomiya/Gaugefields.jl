import JACC
JACC.@init_backend

using Gaugefields
using JLD2
using LatticeMatrices: gather_and_bcast_matrix, halo_epochs
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "MPI portable JLD2 and ILDG round-trip" begin
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


    automatic_prefix = MPI.bcast(
        rank == 0 ? tempname(pwd()) : "",
        0,
        comm,
    )
    automatic_filename = automatic_prefix * ".ildg"
    automatic_grid = first(process_grids)
    automatic_source = gauge_configuration(
        global_size;
        colors=3,
        start=:hot,
        seed=UInt64(0x1234),
        process_grid=automatic_grid,
        comm,
        verbose=0,
    )
    automatic_target = gauge_configuration(
        global_size;
        colors=3,
        start=:cold,
        process_grid=automatic_grid,
        comm,
        verbose=0,
    )

    portable_filename = automatic_prefix * ".jld2"
    @test_throws ErrorException save_configuration(
        joinpath(automatic_prefix * ".missing", "configuration.jld2"),
        automatic_source,
    )
    save_configuration(portable_filename, automatic_source)
    MPI.Barrier(comm)
    if rank == 0
        portable_data = JLD2.load(portable_filename)
        @test portable_data["gaugefields_format"] ==
              "Gaugefields.jl portable gauge configuration"
        @test all(link isa Array for link in portable_data["links"])
        @test portable_data["lattice_size"] == collect(global_size)
    end
    MPI.Barrier(comm)

    portable_target = gauge_configuration(
        global_size;
        colors=3,
        start=:cold,
        process_grid=last(process_grids),
        comm,
        verbose=0,
    )
    load_configuration!(portable_target, portable_filename)
    allocated_target = load_configuration(
        portable_filename;
        process_grid=last(process_grids),
        comm,
        verbose=0,
    )
    one_rank_target = load_configuration(
        portable_filename;
        process_grid=(1, 1, 1, 1),
        comm=MPI.COMM_SELF,
        verbose=0,
    )
    for direction in eachindex(automatic_source)
        original = gather_and_bcast_matrix(automatic_source[direction].U)
        @test gather_and_bcast_matrix(portable_target[direction].U) == original
        @test gather_and_bcast_matrix(allocated_target[direction].U) == original
        @test gather_and_bcast_matrix(one_rank_target[direction].U) == original
    end
    MPI.Barrier(comm)
    rank == 0 && rm(portable_filename; force=true)
    MPI.Barrier(comm)

    failure_prefix = MPI.bcast(
        rank == 0 ? tempname(pwd()) : "",
        0,
        comm,
    )
    failure_payload = failure_prefix * ".payload"
    failure_filelist = failure_prefix * ".list"
    failure_filename = joinpath(
        failure_prefix * ".missing",
        "configuration.ildg",
    )
    @test_throws ErrorException save_binarydata(
        automatic_source,
        failure_filename;
        precision=32,
        tempfile1=failure_payload,
        tempfile2=failure_filelist,
    )
    MPI.Barrier(comm)
    if rank == 0
        rm(failure_payload; force=true)
        rm(failure_filelist; force=true)
    end
    MPI.Barrier(comm)

    save_configuration(
        automatic_filename,
        automatic_source;
        format=:ildg,
        precision=32,
    )
    load_configuration!(automatic_target, automatic_filename; format=:ildg)
    for direction in eachindex(automatic_source)
        original = gather_and_bcast_matrix(automatic_source[direction].U)
        loaded = gather_and_bcast_matrix(automatic_target[direction].U)
        @test maximum(abs, loaded .- ComplexF64.(ComplexF32.(original))) <=
              eps(Float32)
    end
    MPI.Barrier(comm)
    rank == 0 && rm(automatic_filename; force=true)
    MPI.Barrier(comm)
end
