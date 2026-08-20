import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using MPI
using Test

MPI.Initialized() || MPI.Init()

@testset "LM and legacy MPI gauge fixing across rank boundaries" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs in (1, 2) || error("this test supports one or two MPI ranks")

    dims = (4, 4, 4, 4)
    process_grid = (nprocs, 1, 1, 1)
    local_x = dims[1] ÷ nprocs
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    x_range = (rank * local_x + 1):((rank + 1) * local_x)

    for storage in (:nowing_mpi, :wing_mpi)
        @testset "$storage" begin
            U_lm = gauge_configuration(
                dims;
                colors=3,
                start=:hot,
                seed=UInt64(0x155),
                process_grid,
                verbose=0,
            )
            constructor = storage == :nowing_mpi ?
                (() -> Gaugefields.identityGaugefields_4D_nowing_mpi(
                    3, dims..., process_grid; verbose_level=0,
                )) :
                (() -> Gaugefields.identityGaugefields_4D_wing_mpi(
                    3, dims..., 1, process_grid; verbose_level=0,
                ))
            U_legacy = [constructor() for _ in 1:4]
            for μ in eachindex(U_lm)
                substitute_U!(U_legacy[μ], U_lm[μ])
            end

            g_lm = similar(U_lm[1])
            g_legacy = similar(U_legacy[1])
            temps_lm = [similar(U_lm[1]) for _ in 1:6]
            temps_legacy = [similar(U_legacy[1]) for _ in 1:6]
            gaugefixing!(
                U_lm, g_lm, 1.0, 2, 1.0, 2, 0.0, 155, temps_lm;
                D_fix=4,
            )
            gaugefixing!(
                U_legacy,
                g_legacy,
                1.0,
                2,
                1.0,
                2,
                0.0,
                155,
                temps_legacy;
                D_fix=4,
            )

            lm_diagnostics =
                Gaugefields.AbstractGaugefixing_module.validate_training(
                    U_lm, temps_lm; D_fix=4,
                )
            legacy_diagnostics =
                Gaugefields.AbstractGaugefixing_module.validate_training(
                    U_legacy, temps_legacy; D_fix=4,
                )
            @test lm_diagnostics[1] ≈ legacy_diagnostics[1] rtol=2e-12 atol=2e-12
            @test lm_diagnostics[2] ≈ legacy_diagnostics[2] rtol=2e-12 atol=2e-12

            legacy_interior(field) = storage == :nowing_mpi ? field.U :
                @view field.U[
                    :, :, 2:(local_x + 1), 2:5, 2:5, 2:5,
                ]
            for μ in eachindex(U_lm)
                lm_global = gather_and_bcast_matrix(U_lm[μ].U)
                @test @view(lm_global[:, :, x_range, :, :, :]) ≈
                      legacy_interior(U_legacy[μ]) rtol=2e-12 atol=2e-12
            end
            g_global = gather_and_bcast_matrix(g_lm.U)
            @test @view(g_global[:, :, x_range, :, :, :]) ≈
                  legacy_interior(g_legacy) rtol=2e-12 atol=2e-12
        end
    end
end
