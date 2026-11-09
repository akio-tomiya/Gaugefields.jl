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
    @test final_trace ≈ 0.4939187487419858 rtol=5e-13 atol=5e-13
    @test final_residual ≈ 0.10569226126573773 rtol=5e-13 atol=5e-13
    @test measure_plaquette(U) ≈ initial_plaquette rtol=5e-13 atol=5e-13

    # Match the corrected SU(3) implementation from the original PR.
    expected_link = ComplexF64[
        0.8130563115405814 + 0.1520581033956765im 0.29737306613275055 - 0.24993359414804583im 0.35784201312556774 + 0.1920138514613559im
        -0.3138079774657333 - 0.3435443998338487im 0.8571206121364563 - 0.15347058392611604im -0.04269028801087267 + 0.15320043750400808im
        -0.31058724911345886 + 0.053433386244339756im 0.011676581043848888 + 0.3012592764221165im 0.8994050964037139 - 0.02928160361685113im
    ]
    U_global = gather_and_bcast_matrix(U[1].U)
    @test U_global[:, :, 1, 1, 1, 1] ≈ expected_link rtol=5e-13 atol=5e-13

    # g contains the most recent checkerboard transformation. The inactive
    # parity is the identity and tr[G] therefore approaches one at convergence.
    g_global = gather_and_bcast_matrix(g.U)
    @test real(tr(g) / (3 * g.NV)) ≈ 0.978210374858813 rtol=5e-13 atol=5e-13
    @test g_global[:, :, 1, 1, 1, 1] ≈ I rtol=5e-13 atol=5e-13
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
    @test_throws ArgumentError gaugefixing!(
        U, g, 1.0, 0, 1.0, 0, 0.0, 155, temps;
        D_fix=4, min_iterations=-1,
    )
end

@testset "Corrected PR #155 reference trajectory" begin
    U = gauge_configuration(
        (4, 4, 4, 4);
        colors=3,
        start=:hot,
        seed=UInt64(0x155),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    g = similar(U[1])
    temps = [similar(U[1]) for _ in 1:6]

    gaugefixing!(U, g, 1.0, 20, 1.99, 20, 1e-14, 155, temps; D_fix=3)
    trace, residual = Gaugefields.AbstractGaugefixing_module.validate_training(
        U, temps; D_fix=3,
    )

    @test trace ≈ 0.6610997098198766 rtol=5e-13 atol=5e-13
    @test residual ≈ 0.021836712597243808 rtol=5e-13 atol=5e-13
    @test real(tr(g) / (3 * g.NV)) ≈ 0.9943505600620449 rtol=5e-13 atol=5e-13

    expected_link = ComplexF64[
        0.8200640482119832 + 0.08783683660724424im 0.4082890078669665 + 0.3082712241927962im 0.015260813604176166 - 0.24044894036536352im
        -0.5028786195186308 + 0.0943651851592511im 0.7726006419469793 + 0.11831242150709993im 0.2966658979851915 - 0.1982121838176903im
        0.14198992694108123 - 0.19449116554015317im -0.24567969657081654 - 0.2588452125552973im 0.8575997050406842 - 0.28138129951776225im
    ]
    U_global = gather_and_bcast_matrix(U[1].U)
    @test U_global[:, :, 1, 1, 1, 1] ≈ expected_link rtol=5e-13 atol=5e-13
end

@testset "Coulomb gauge-fixing regression at convergence" begin
    # This is an implementation-specific regression value, not a unique
    # gauge-fixing result: another update algorithm may reach a different
    # Gribov copy. The fixed seed and parameters reproduce the comparison
    # made for PR #155 across the supported Gaugefields backends.
    U = gauge_configuration(
        (8, 8, 8, 8);
        colors=3,
        start=:hot,
        seed=UInt64(1234),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    g = similar(U[1])
    temps = [similar(U[1]) for _ in 1:6]
    initial_plaquette = measure_plaquette(U)
    initial_trace, _ = Gaugefields.AbstractGaugefixing_module.validate_training(
        U, temps; D_fix=3,
    )

    @test initial_trace ≈ 0.002085448143734312 rtol=5e-13 atol=5e-13

    gaugefixing!(
        U,
        g,
        1.0,
        200,
        1.99,
        10_000,
        1e-14,
        1234,
        temps;
        D_fix=3,
    )

    final_trace, final_residual =
        Gaugefields.AbstractGaugefixing_module.validate_training(
            U, temps; D_fix=3,
        )

    @test final_trace ≈ 0.6755459192810215 rtol=5e-13 atol=5e-13
    @test final_residual < 1e-14
    @test measure_plaquette(U) ≈ initial_plaquette rtol=5e-12 atol=5e-12
end

@testset "Configurable minimum gauge-fixing iterations" begin
    U_seed = gauge_configuration(
        (2, 2, 2, 2);
        colors=3,
        start=:hot,
        seed=UInt64(0x50155),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    U_default = similar(U_seed)
    U_full = similar(U_seed)
    U_early = similar(U_seed)
    U_one = similar(U_seed)
    for field in (U_default, U_full, U_early, U_one)
        substitute_U!(field, U_seed)
    end

    g_default = similar(U_seed[1])
    g_full = similar(U_seed[1])
    g_early = similar(U_seed[1])
    g_one = similar(U_seed[1])
    temps_default = [similar(U_seed[1]) for _ in 1:6]
    temps_full = [similar(U_seed[1]) for _ in 1:6]
    temps_early = [similar(U_seed[1]) for _ in 1:6]
    temps_one = [similar(U_seed[1]) for _ in 1:6]

    # The default does all three requested iterations even when the tolerance
    # is already satisfied. Setting min_iterations=0 permits an immediate stop.
    gaugefixing!(
        U_default, g_default, 1.0, 3, 1.0, 0, Inf, 155, temps_default;
        D_fix=4,
    )
    gaugefixing!(
        U_full, g_full, 1.0, 3, 1.0, 0, 0.0, 155, temps_full;
        D_fix=4, min_iterations=0,
    )
    gaugefixing!(
        U_early, g_early, 1.0, 3, 1.0, 0, Inf, 155, temps_early;
        D_fix=4, min_iterations=0,
    )
    gaugefixing!(
        U_one, g_one, 1.0, 1, 1.0, 0, 0.0, 155, temps_one;
        D_fix=4, min_iterations=0,
    )

    for μ in eachindex(U_seed)
        @test gather_and_bcast_matrix(U_default[μ].U) ≈
              gather_and_bcast_matrix(U_full[μ].U) rtol=5e-13 atol=5e-13
        @test gather_and_bcast_matrix(U_early[μ].U) ≈
              gather_and_bcast_matrix(U_one[μ].U) rtol=5e-13 atol=5e-13
    end
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
    g = similar(U[1])
    temps = [similar(U[1]) for _ in 1:6]
    initial_plaquette = measure_plaquette(U)

    gaugefixing!(U, g, 1.5, 1, 1.0, 1, 0.0, 4155, temps; D_fix=4)

    @test measure_plaquette(U) ≈ initial_plaquette rtol=2e-12 atol=2e-12
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

    g_global = gather_and_bcast_matrix(g.U)
    for site in CartesianIndices(size(g_global)[3:end])
        matrix = @view g_global[:, :, Tuple(site)...]
        @test matrix * matrix' ≈ I rtol=3e-5 atol=3e-5
        @test det(matrix) ≈ 1 rtol=3e-5 atol=3e-5
    end
end

@testset "Fused MPILattice gauge-fixing steps equal the field-level operations" begin
    # gaugefixing! builds g and transforms the links in fused kernels on
    # MPILattice fields; each site repeats the arithmetic of the field-level
    # functions, so the links must agree bit for bit with those functions.
    M = Gaugefields.AbstractGaugefixing_module
    start = gauge_configuration(
        (4, 4, 4, 8);
        colors=3,
        start=:hot,
        seed=UInt64(0x2026),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    function copy_links(V)
        W = [similar(v) for v in V]
        for μ in eachindex(V)
            Gaugefields.substitute_U!(W[μ], V[μ])
            Gaugefields.set_wing_U!(W[μ])
        end
        return W
    end
    core(x) = x.U.A[:, :, 2:(end - 1), 2:(end - 1), 2:(end - 1), 2:(end - 1)]
    temps = [similar(start[1]) for _ in 1:6]
    D_fix = 3

    reference = copy_links(start)
    fused = copy_links(start)
    Δ_reference = similar(start[1])
    Δ_fused = similar(start[1])
    M.get_Δ!(Δ_reference, reference, temps[2:6], D_fix)
    M._gaugefixing_delta!(Δ_fused, fused, temps[2:6], D_fix)
    @test core(Δ_fused) == core(Δ_reference)

    # Los Alamos and steepest-descent sweeps over both checkerboards
    g_reference = similar(start[1])
    g_fused = similar(start[1])
    for parity in (0, 1)
        M.make_g_los_alamos!(reference, g_reference, temps[2], parity,
            1.7, 0.595, -0.0595, D_fix)
        M.gUgshift!(reference, g_reference, temps[2])
    end
    M.gaugefixing_step_la!(fused, g_fused, 1.7, 0.595, -0.0595, temps[2]; D_fix)
    @test core(g_fused) == core(g_reference)
    @test all(core(fused[μ]) == core(reference[μ]) for μ in 1:4)

    for parity in (0, 1)
        M.make_g_steepest_descent!(reference, g_reference, Δ_reference,
            parity, 1.99, temps[2:6], D_fix)
        M.gUgshift!(reference, g_reference, temps[2])
    end
    M.gaugefixing_step_sd!(fused, g_fused, Δ_fused, 1.99, temps[2:6]; D_fix)
    @test core(g_fused) == core(g_reference)
    @test all(core(fused[μ]) == core(reference[μ]) for μ in 1:4)

    # gaugefixing! defers the link halos during the iterations and sets them
    # before it returns.
    gaugefixing!(fused, g_fused, 1.0, 3, 1.99, 3, 0.0, 0, temps; D_fix)
    for μ in 1:4
        with_halo = copy(fused[μ].U.A)
        Gaugefields.set_wing_U!(fused[μ])
        @test fused[μ].U.A == with_halo
    end
end
