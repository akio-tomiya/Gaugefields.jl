using LatticeMatrices: gather_and_bcast_matrix
using MPI

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices and serial nowing agree" begin
    U_lm = gauge_configuration(
        (4, 4, 4, 4);
        colors=3,
        start=:hot,
        seed=UInt64(0x155),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    U_serial = Initialize_Gaugefields(
        3, 0, 4, 4, 4, 4;
        condition="cold",
        isMPILattice=false,
        verbose_level=0,
    )
    substitute_U!(U_serial, U_lm)

    g_lm = similar(U_lm[1])
    g_serial = similar(U_serial[1])
    temps_lm = [similar(U_lm[1]) for _ in 1:6]
    temps_serial = [similar(U_serial[1]) for _ in 1:6]

    gaugefixing!(
        U_lm, g_lm, 1.0, 2, 1.0, 2, 0.0, 155, temps_lm; D_fix=4,
    )
    gaugefixing!(
        U_serial,
        g_serial,
        1.0,
        2,
        1.0,
        2,
        0.0,
        155,
        temps_serial;
        D_fix=4,
    )

    lm_trace, lm_residual =
        Gaugefields.AbstractGaugefixing_module.validate_training(
            U_lm, temps_lm; D_fix=4,
        )
    serial_trace, serial_residual =
        Gaugefields.AbstractGaugefixing_module.validate_training(
            U_serial, temps_serial; D_fix=4,
        )

    @test lm_trace ≈ serial_trace rtol=2e-13 atol=2e-13
    @test lm_residual ≈ serial_residual rtol=2e-13 atol=2e-13
    @test isapprox(
        measure_plaquette(U_lm), measure_plaquette(U_serial);
        rtol=2e-13, atol=2e-13,
    )

    for μ in eachindex(U_lm)
        @test isapprox(
            gather_and_bcast_matrix(U_lm[μ].U), U_serial[μ].U;
            rtol=2e-13, atol=2e-13,
        )
    end
    @test isapprox(
        gather_and_bcast_matrix(g_lm.U), g_serial.U;
        rtol=2e-13, atol=2e-13,
    )
end

@testset "LatticeMatrices and portable accelerator storage agree" begin
    dims = (4, 4, 4, 4)
    U_lm = gauge_configuration(
        dims;
        colors=3,
        start=:hot,
        seed=UInt64(0x155),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    U_serial = Initialize_Gaugefields(
        3,
        0,
        dims...;
        condition="cold",
        isMPILattice=false,
        verbose_level=0,
    )
    substitute_U!(U_serial, U_lm)
    constructor = () ->
        Gaugefields.AbstractGaugefields_module.identityGaugefields_4D_accelerator(
            3,
            dims...,
            (2, 2, 2, 2);
            verbose_level=0,
            accelerator="JACC",
        )
    U_accelerator = [constructor() for _ in 1:4]
    for μ in eachindex(U_lm)
        substitute_U!(U_accelerator[μ], U_serial[μ])
    end
    g_lm = similar(U_lm[1])
    g_accelerator = similar(U_accelerator[1])
    temps_lm = [similar(U_lm[1]) for _ in 1:6]
    temps_accelerator = [similar(U_accelerator[1]) for _ in 1:6]
    gaugefixing!(
        U_lm, g_lm, 1.0, 2, 1.0, 2, 0.0, 155, temps_lm; D_fix=4,
    )
    gaugefixing!(
        U_accelerator,
        g_accelerator,
        1.0,
        2,
        1.0,
        2,
        0.0,
        155,
        temps_accelerator;
        D_fix=4,
    )

    lm_diagnostics = Gaugefields.AbstractGaugefixing_module.validate_training(
        U_lm, temps_lm; D_fix=4,
    )
    accelerator_diagnostics =
        Gaugefields.AbstractGaugefixing_module.validate_training(
            U_accelerator, temps_accelerator; D_fix=4,
        )
    @test lm_diagnostics[1] ≈ accelerator_diagnostics[1] rtol=3e-13 atol=3e-13
    @test lm_diagnostics[2] ≈ accelerator_diagnostics[2] rtol=3e-13 atol=3e-13

    for μ in eachindex(U_lm)
        @test reshape(gather_and_bcast_matrix(U_lm[μ].U), 3, 3, :) ≈
              Array(U_accelerator[μ].U) rtol=3e-13 atol=3e-13
    end
    @test reshape(gather_and_bcast_matrix(g_lm.U), 3, 3, :) ≈
          Array(g_accelerator.U) rtol=3e-13 atol=3e-13

end

@testset "Portable accelerator Float32 Coulomb path" begin
    dims = (2, 2, 2, 2)
    U_lm = gauge_configuration(
        dims;
        colors=3,
        start=:hot,
        seed=UInt64(0x3155),
        process_grid=(1, 1, 1, 1),
        eltype=ComplexF32,
        verbose=0,
    )
    U_serial = Initialize_Gaugefields(
        3,
        0,
        dims...;
        condition="cold",
        isMPILattice=false,
        verbose_level=0,
    )
    substitute_U!(U_serial, U_lm)
    constructor = () ->
        Gaugefields.AbstractGaugefields_module.identityGaugefields_4D_accelerator(
            3,
            dims...,
            (1, 1, 1, 1);
            verbose_level=0,
            accelerator="JACC",
            singleprecision=true,
        )
    U_accelerator = [constructor() for _ in 1:4]
    for μ in eachindex(U_lm)
        substitute_U!(U_accelerator[μ], U_serial[μ])
    end

    g_lm = similar(U_lm[1])
    g_accelerator = similar(U_accelerator[1])
    temps_lm = [similar(U_lm[1]) for _ in 1:6]
    temps_accelerator = [similar(U_accelerator[1]) for _ in 1:6]
    gaugefixing!(
        U_lm, g_lm, 1.5, 1, 1.5, 1, 0.0, 3155, temps_lm; D_fix=3,
    )
    gaugefixing!(
        U_accelerator,
        g_accelerator,
        1.5,
        1,
        1.5,
        1,
        0.0,
        3155,
        temps_accelerator;
        D_fix=3,
    )

    lm_diagnostics = Gaugefields.AbstractGaugefixing_module.validate_training(
        U_lm, temps_lm; D_fix=3,
    )
    accelerator_diagnostics =
        Gaugefields.AbstractGaugefixing_module.validate_training(
            U_accelerator, temps_accelerator; D_fix=3,
        )
    @test lm_diagnostics[1] ≈ accelerator_diagnostics[1] rtol=3e-5 atol=3e-5
    @test lm_diagnostics[2] ≈ accelerator_diagnostics[2] rtol=3e-5 atol=3e-5
    for μ in eachindex(U_lm)
        @test reshape(gather_and_bcast_matrix(U_lm[μ].U), 3, 3, :) ≈
              Array(U_accelerator[μ].U) rtol=3e-5 atol=3e-5
    end
    @test reshape(gather_and_bcast_matrix(g_lm.U), 3, 3, :) ≈
          Array(g_accelerator.U) rtol=3e-5 atol=3e-5
end

@testset "LatticeMatrices and legacy MPI storage agree (one rank)" begin
    MPI.Comm_size(MPI.COMM_WORLD) == 1 || error(
        "the package test for legacy MPI storage must run on one MPI rank",
    )

    for storage in (:nowing_mpi, :wing_mpi)
        @testset "$storage" begin
            dims = (4, 4, 4, 4)
            process_grid = (1, 1, 1, 1)
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
            @test lm_diagnostics[1] ≈ legacy_diagnostics[1] rtol=2e-13 atol=2e-13
            @test lm_diagnostics[2] ≈ legacy_diagnostics[2] rtol=2e-13 atol=2e-13

            legacy_interior(field) = storage == :nowing_mpi ? field.U :
                @view field.U[:, :, 2:5, 2:5, 2:5, 2:5]
            for μ in eachindex(U_lm)
                @test gather_and_bcast_matrix(U_lm[μ].U) ≈
                      legacy_interior(U_legacy[μ]) rtol=2e-13 atol=2e-13
            end
            @test gather_and_bcast_matrix(g_lm.U) ≈
                  legacy_interior(g_legacy) rtol=2e-13 atol=2e-13
        end
    end
end
