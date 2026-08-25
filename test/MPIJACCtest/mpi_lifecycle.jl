import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: SerialCommunicator
using Test

@test Base.get_extension(Gaugefields, :GaugefieldsMPIExt) === nothing

using MPI

@testset "MPI extension lifecycle" begin
    @test Base.get_extension(Gaugefields, :GaugefieldsMPIExt) !== nothing
    @test !MPI.Initialized()

    serial = gauge_configuration(
        (2, 2);
        colors=2,
        process_grid=(1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )
    @test gauge_communicator(serial) isa SerialCommunicator
    @test !MPI.Initialized()

    distributed = gauge_configuration(
        (2, 2);
        colors=2,
        process_grid=(1, 1),
        verbose=0,
    )
    @test MPI.Initialized()
    @test gauge_communicator(distributed) == MPI.COMM_WORLD

    MPI.Finalize()
    @test MPI.Finalized()
    @test_throws ArgumentError gauge_configuration(
        (2, 2);
        colors=2,
        process_grid=(1, 1),
        verbose=0,
    )

    serial_after_finalize = gauge_configuration(
        (2, 2);
        colors=2,
        process_grid=(1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )
    @test gauge_communicator(serial_after_finalize) isa SerialCommunicator
end
