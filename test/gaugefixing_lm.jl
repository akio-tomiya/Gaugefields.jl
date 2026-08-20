using LinearAlgebra
using MPI
using LatticeMatrices: gather_and_bcast_matrix

MPI.Initialized() || MPI.Init()

@testset "LatticeMatrices gauge fixing" begin
    U = gauge_configuration(
        (4, 4, 4, 4);
        colors=3,
        start=:hot,
        seed=UInt64(0x155),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    U_initial = similar(U)
    substitute_U!(U_initial, U)
    g = similar(U[1])
    temps = [similar(U[1]) for _ in 1:6]

    diagnostics() = Gaugefields.AbstractGaugefixing_module.validate_training(
        U, temps; D_fix=4,
    )

    initial_trace, initial_residual = diagnostics()
    initial_plaquette = measure_plaquette(U)

    # These two values characterize the exact input used for comparison with
    # the original PR #155 implementation.
    @test initial_trace ≈ -0.0016495155923378993 rtol=5e-13 atol=5e-13
    @test initial_residual ≈ 0.8969653524627318 rtol=5e-13 atol=5e-13

    @test gaugefixing!(
        U,
        g,
        1.0,
        2,
        1.0,
        2,
        0.0,
        155,
        temps;
        D_fix=4,
    ) === U

    final_trace, final_residual = diagnostics()
    @test final_trace > initial_trace
    @test final_residual < initial_residual
    @test measure_plaquette(U) ≈ initial_plaquette rtol=5e-13 atol=5e-13

    # g is the complete transformation, not only the final checkerboard step.
    U_from_g = similar(U)
    gUgshift!(U_from_g, U_initial, g, similar(g))
    for μ in eachindex(U)
        reconstructed = gather_and_bcast_matrix(U_from_g[μ].U)
        transformed = gather_and_bcast_matrix(U[μ].U)
        @test reconstructed ≈ transformed rtol=5e-13 atol=5e-13
    end

    # Every site of the returned transformation must lie in SU(3).
    g_global = gather_and_bcast_matrix(g.U)
    for site in CartesianIndices(size(g_global)[3:end])
        matrix = @view g_global[:, :, Tuple(site)...]
        @test matrix * matrix' ≈ I rtol=5e-13 atol=5e-13
        @test det(matrix) ≈ 1 rtol=5e-13 atol=5e-13
    end

    @test_throws ArgumentError gaugefixing!(
        U, g, 1.0, 0, 1.0, 0, 0.0, 155, temps; D_fix=5,
    )
    @test_throws ArgumentError gaugefixing!(
        U, g, 1.0, -1, 1.0, 0, 0.0, 155, temps; D_fix=4,
    )
end

@testset "Generic SU(4) LatticeMatrices gauge fixing" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=4,
        start=:hot,
        seed=UInt64(0x4155),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    U_initial = similar(U)
    substitute_U!(U_initial, U)
    g = similar(U[1])
    temps = [similar(U[1]) for _ in 1:6]
    initial_plaquette = measure_plaquette(U)

    gaugefixing!(U, g, 1.0, 1, 1.0, 1, 0.0, 4155, temps; D_fix=4)

    @test measure_plaquette(U) ≈ initial_plaquette rtol=2e-12 atol=2e-12
    U_from_g = similar(U)
    gUgshift!(U_from_g, U_initial, g, similar(g))
    for μ in eachindex(U)
        reconstructed = gather_and_bcast_matrix(U_from_g[μ].U)
        transformed = gather_and_bcast_matrix(U[μ].U)
        @test reconstructed ≈ transformed rtol=2e-12 atol=2e-12
    end

    g_global = gather_and_bcast_matrix(g.U)
    for site in CartesianIndices(size(g_global)[3:end])
        matrix = @view g_global[:, :, Tuple(site)...]
        @test matrix * matrix' ≈ I rtol=2e-12 atol=2e-12
        @test det(matrix) ≈ 1 rtol=2e-12 atol=2e-12
    end
end

@testset "Float32 Coulomb gauge path" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=3,
        start=:hot,
        seed=UInt64(0x3155),
        process_grid=(1, 1, 1, 1),
        eltype=ComplexF32,
        verbose=0,
    )
    U_initial = similar(U)
    substitute_U!(U_initial, U)
    g = similar(U[1])
    temps = [similar(U[1]) for _ in 1:6]
    initial_plaquette = measure_plaquette(U)
    _, initial_residual =
        Gaugefields.AbstractGaugefixing_module.validate_training(
            U, temps; D_fix=3,
        )

    gaugefixing!(U, g, 1.5, 1, 1.5, 1, 0.0, 3155, temps; D_fix=3)

    _, final_residual = Gaugefields.AbstractGaugefixing_module.validate_training(
        U, temps; D_fix=3,
    )
    @test isfinite(final_residual)
    @test final_residual < initial_residual
    @test measure_plaquette(U) ≈ initial_plaquette rtol=3e-5 atol=3e-5

    U_from_g = similar(U)
    gUgshift!(U_from_g, U_initial, g, similar(g))
    for μ in eachindex(U)
        reconstructed = gather_and_bcast_matrix(U_from_g[μ].U)
        transformed = gather_and_bcast_matrix(U[μ].U)
        @test reconstructed ≈ transformed rtol=3e-5 atol=3e-5
    end

    g_global = gather_and_bcast_matrix(g.U)
    for site in CartesianIndices(size(g_global)[3:end])
        matrix = @view g_global[:, :, Tuple(site)...]
        @test matrix * matrix' ≈ I rtol=3e-5 atol=3e-5
        @test det(matrix) ≈ 1 rtol=3e-5 atol=3e-5
    end
end
