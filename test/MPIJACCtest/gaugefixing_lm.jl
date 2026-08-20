import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices gauge fixing MPI decomposition" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs in (1, 2) || error("this test supports one or two MPI ranks")

    U = gauge_configuration(
        (4, 4, 4, 4);
        colors=3,
        start=:hot,
        seed=UInt64(0x155),
        process_grid=(nprocs, 1, 1, 1),
        verbose=0,
    )
    U_initial = similar(U)
    substitute_U!(U_initial, U)
    g = similar(U[1])
    temps = [similar(U[1]) for _ in 1:6]
    initial_plaquette = measure_plaquette(U)

    gaugefixing!(U, g, 1.0, 2, 1.0, 2, 0.0, 155, temps; D_fix=4)

    trace, residual = Gaugefields.AbstractGaugefixing_module.validate_training(
        U, temps; D_fix=4,
    )
    @test trace ≈ 0.49391874874198577 rtol=2e-12 atol=2e-12
    @test residual ≈ 0.10569226126573777 rtol=2e-12 atol=2e-12
    @test measure_plaquette(U) ≈ initial_plaquette rtol=2e-12 atol=2e-12

    global_checksum = (
        sum(sum(real, gather_and_bcast_matrix(field.U)) for field in U),
        sum(sum(imag, gather_and_bcast_matrix(field.U)) for field in U),
    )
    @test global_checksum[1] ≈ 1524.5782040190834 rtol=2e-12 atol=2e-12
    @test global_checksum[2] ≈ 78.55450733033379 rtol=2e-12 atol=2e-12

    U_from_g = similar(U)
    gUgshift!(U_from_g, U_initial, g, similar(g))
    for μ in eachindex(U)
        @test gather_and_bcast_matrix(U_from_g[μ].U) ≈
              gather_and_bcast_matrix(U[μ].U) rtol=2e-12 atol=2e-12
    end
end
