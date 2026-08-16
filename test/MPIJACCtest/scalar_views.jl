import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices shifted and staggered scalar views" begin
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)

    global_size_4d = (2 * nprocs, 2, 2, 2)
    U4 = Initialize_Gaugefields(
        3,
        1,
        global_size_4d...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(nprocs, 1, 1, 1),
        verbose_level=0,
    )[1]
    global_U4 = gather_and_bcast_matrix(U4.U)
    local_x4 = U4.U.PN[1]
    global_x4 = U4.U.coords[1] * U4.U.PN[1] + local_x4
    source_x4 = mod(global_x4, global_size_4d[1]) + 1

    shifted4 = shift_U(U4, (1, 0, 0, 0))
    @test shifted4[1, 1, local_x4, 1, 1, 1] ==
          global_U4[1, 1, source_x4, 1, 1, 1]

    staggered4 = Gaugefields.staggered_U(U4, 4)
    phase4 = iseven(global_x4 - 1) ? 1 : -1
    @test staggered4[1, 1, local_x4, 1, 1, 1] ==
          phase4 * global_U4[1, 1, global_x4, 1, 1, 1]

    shifted_staggered4 = Gaugefields.staggered_U(shifted4, 4)
    shifted_phase4 = iseven(source_x4 - 1) ? 1 : -1
    @test shifted_staggered4[1, 1, local_x4, 1, 1, 1] ==
          shifted_phase4 * global_U4[1, 1, source_x4, 1, 1, 1]
    close(shifted4)

    momentum4 = initialize_TA_Gaugefields(U4)
    momentum4[1, local_x4, 1, 1, 1] = 0.125
    @test momentum4[1, local_x4, 1, 1, 1] == 0.125
    projected4 = similar(momentum4)
    added4 = similar(momentum4)
    Traceless_antihermitian!(projected4, U4)
    clear_U!(added4)
    Traceless_antihermitian_add!(added4, 1, U4)
    @test gather_and_bcast_matrix(projected4.a) ==
          gather_and_bcast_matrix(added4.a)

    global_size_2d = (2 * nprocs, 2)
    U2 = Initialize_Gaugefields(
        3,
        1,
        global_size_2d...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(nprocs, 1),
        verbose_level=0,
    )[1]
    global_U2 = gather_and_bcast_matrix(U2.U)
    local_x2 = U2.U.PN[1]
    global_x2 = U2.U.coords[1] * U2.U.PN[1] + local_x2
    source_x2 = mod(global_x2, global_size_2d[1]) + 1

    shifted2 = shift_U(U2, (1, 0))
    @test shifted2[1, 1, local_x2, 1] == global_U2[1, 1, source_x2, 1]

    staggered2 = Gaugefields.staggered_U(U2, 2)
    phase2 = iseven(global_x2 - 1) ? 1 : -1
    @test staggered2[1, 1, local_x2, 1] ==
          phase2 * global_U2[1, 1, global_x2, 1]
    close(shifted2)

    momentum2 = initialize_TA_Gaugefields(U2)
    momentum2[1, local_x2, 1] = 0.25
    @test momentum2[1, local_x2, 1] == 0.25
    projected2 = similar(momentum2)
    added2 = similar(momentum2)
    Traceless_antihermitian!(projected2, U2)
    clear_U!(added2)
    Traceless_antihermitian_add!(added2, 1, U2)
    @test gather_and_bcast_matrix(projected2.a) ==
          gather_and_bcast_matrix(added2.a)
end
