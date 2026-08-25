import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using MPI
using Test

@testset "High-level API with MPI" begin
    @test !MPI.Initialized()
    dimensions = (4, 2, 2)
    U = gauge_configuration(
        dimensions;
        colors=2,
        start=:cold,
        process_grid=:auto,
        verbose=0,
    )
    @test MPI.Initialized()

    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs <= 2 || error("this test supports at most two MPI ranks")
    grid = (nprocs, 1, 1)

    @test length(U) == 3
    @test gauge_backend(U) isa LatticeMatricesBackend
    @test gauge_lattice_size(U) == dimensions
    @test gauge_process_grid(U) == grid
    @test gauge_communicator(U) == MPI.COMM_WORLD
    @test measure_plaquette(U) ≈ 1
    @test measure_polyakov_loop(U) ≈ 1

    if nprocs == 1
        serial = gauge_configuration(
            dimensions;
            colors=2,
            start=:cold,
            process_grid=(1, 1, 1),
            comm=SerialCommunicator(),
            verbose=0,
        )
        @test gauge_communicator(serial) isa SerialCommunicator
        @test measure_plaquette(serial) ≈ 1
    end

    hot1 = gauge_configuration(
        dimensions;
        colors=2,
        start=:hot,
        seed=UInt64(0x1234),
        process_grid=grid,
        verbose=0,
    )
    hot2 = gauge_configuration(
        dimensions;
        colors=2,
        start=:hot,
        seed=UInt64(0x1234),
        process_grid=grid,
        verbose=0,
    )
    @test gather_and_bcast_matrix.(getproperty.(hot1, :U)) ==
          gather_and_bcast_matrix.(getproperty.(hot2, :U))

    momenta = gaussian_momenta(hot1; seed=UInt64(0x5678))
    @test isfinite(momenta * momenta)
end
