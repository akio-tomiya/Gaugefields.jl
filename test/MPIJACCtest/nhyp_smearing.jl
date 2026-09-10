import JACC
JACC.@init_backend

using MPI
using Gaugefields
using Test
import LatticeMatrices

MPI.Initialized() || MPI.Init()

@testset "distributed Gaugefields nHYP" begin
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    lattice = (2nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    keywords = (
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(8675309),
        verbose=0,
    )
    distributed = gauge_configuration(
        lattice;
        keywords...,
        process_grid,
        comm,
    )
    serial = gauge_configuration(
        lattice;
        keywords...,
        process_grid=(1, 1, 1, 1),
        comm=SerialCommunicator(),
    )
    distributed_left = gauge_configuration(
        lattice;
        keywords...,
        seed=UInt64(314159),
        process_grid,
        comm,
    )
    serial_left = gauge_configuration(
        lattice;
        keywords...,
        seed=UInt64(314159),
        process_grid=(1, 1, 1, 1),
        comm=SerialCommunicator(),
    )

    distributed_specification = nhyp_smearing(distributed)
    serial_specification = nhyp_smearing(serial)
    distributed_smeared, distributed_cache =
        nhyp_smear(distributed, distributed_specification)
    serial_smeared, serial_cache = nhyp_smear(serial, serial_specification)

    for mu in 1:4
        distributed_host = LatticeMatrices.gather_and_bcast_matrix(
            distributed_smeared[mu].U,
        )
        serial_host = LatticeMatrices.gather_and_bcast_matrix(
            serial_smeared[mu].U,
        )
        @test distributed_host ≈ serial_host rtol=3e-11 atol=3e-11
    end

    distributed_force = nhyp_pullback(
        distributed_left,
        distributed,
        distributed_cache,
    )
    serial_force = nhyp_pullback(serial_left, serial, serial_cache)
    for mu in 1:4
        distributed_host = LatticeMatrices.gather_and_bcast_matrix(
            distributed_force[mu].U,
        )
        serial_host = LatticeMatrices.gather_and_bcast_matrix(
            serial_force[mu].U,
        )
        @test distributed_host ≈ serial_host rtol=4e-11 atol=4e-11
    end
end
