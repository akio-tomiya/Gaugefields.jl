import LatticeMatrices

const LMCompat = Gaugefields.LatticeMatricesCompat

@test LMCompat.HAS_SHIFT_RELEASE == isdefined(LatticeMatrices, :release!)
@test LMCompat.HAS_HALO_EPOCHS ==
      isdefined(LatticeMatrices, :mark_halo_dirty!)

U = Initialize_Gaugefields(
    3,
    1,
    4,
    4,
    4,
    4;
    condition="cold",
    isMPILattice=true,
    PEs=(1, 1, 1, 1),
    verbose_level=0,
)

if LMCompat.HAS_HALO_EPOCHS
    @test !LatticeMatrices.halo_is_dirty(U[1].U)
end

U[1][1, 1, 1, 1, 1, 1] = 2 + 0im

if LMCompat.HAS_HALO_EPOCHS
    @test LatticeMatrices.halo_is_dirty(U[1].U)
end

# This is wider than NDW=1. LatticeMatrices 1.x borrows storage for the
# materialized shift; 0.3 shifted lattices are non-owning.
shifted = shift_U(U[1], (2, 0, 0, 0))
@test isopen(shifted)
close(shifted)
@test isopen(shifted) == !LMCompat.HAS_SHIFT_RELEASE

adjoint_shifted = shift_U(U[1], (2, 0, 0, 0))'
@test isopen(adjoint_shifted)
close(adjoint_shifted)
@test isopen(adjoint_shifted) == !LMCompat.HAS_SHIFT_RELEASE

temp1 = similar(U[1])
temp2 = similar(U[1])
plaquette = calculate_Plaquette(U, temp1, temp2)
polyakov = calculate_Polyakov_loop(U, temp1, temp2)
@test isfinite(real(plaquette))
@test isfinite(real(polyakov))

U4_wrapper = Initialize_4DGaugefields(
    3,
    1,
    2,
    2,
    2,
    2;
    condition="hot",
    randomnumber="Reproducible",
    isMPILattice=true,
    PEs=(1, 1, 1, 1),
    elementtype=ComplexF32,
    verbose_level=0,
)
@test eltype(U4_wrapper[1]) == ComplexF32

mktempdir() do directory
    filename = joinpath(directory, "gaugefields.jld2")
    saveU(filename, U4_wrapper)
    loaded = loadU(filename)
    @test typeof(loaded[1]) == typeof(U4_wrapper[1])

    target = Initialize_4DGaugefields(
        3,
        1,
        2,
        2,
        2,
        2;
        condition="cold",
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        elementtype=ComplexF32,
        verbose_level=0,
    )
    Gaugefields.IOmodule.loadU!(filename, target)
    @test target[1][1, 1, 1, 1, 1, 1] == loaded[1][1, 1, 1, 1, 1, 1]
end

@testset "2D legacy-compatible API" begin
    U2 = Initialize_Gaugefields(
        3,
        0,
        4,
        4;
        condition="cold",
        isMPILattice=true,
        PEs=(1, 1),
        verbose_level=0,
    )

    @test U2[1].NT == 4
    @test hasproperty(U2[1], :NT)
    @test get_myrank(U2) == 0
    @test get_nprocs(U2) == 1

    U2[1][1, 1, 1, 1] = 2 + 0im
    @test U2[1][1, 1, 1] == 2 + 0im

    U2copy = similar(U2)
    @test length(U2copy) == 2
    @test all(isassigned(U2copy, μ) for μ = 1:2)
    set_wing_U!(U2copy)

    map_U_sequential!(
        U2copy[1],
        (matrix, _, ix, it) -> (matrix .= ix + it),
        nothing,
    )
    @test U2copy[1][1, 1, 2, 3] == 5

    Gaugefields.AbstractGaugefields_module.barrier(U2[1])

    mktempdir() do directory
        filename = joinpath(directory, "gaugefield_2d.npz")
        write_to_numpyarray(U2[1], filename)
        stored = Gaugefields.AbstractGaugefields_module.npzread(filename)
        @test size(stored["U"]) == (3, 3, 4, 4)
        @test stored["NT"] == 4
    end
end

@testset "2D LatticeMatrices algorithms" begin
    U2 = Initialize_Gaugefields(
        2,
        1,
        4,
        4;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(1, 1),
        verbose_level=0,
    )

    momentum = initialize_TA_Gaugefields(U2[1])
    momentum_copy = similar(momentum)
    @test typeof(momentum_copy) == typeof(momentum)

    flow = Gradientflow(U2; eps=0.01)
    flow!(U2, flow)

    network = CovNeuralnet(U2)
    @test network isa Gaugefields.Abstractsmearing_module.CovNeuralnet{2}
    push!(network, STOUT_Layer(["plaquette"], [0.1], U2))
    smeared, history, derivative = calc_smearedU(U2, network)
    @test length(history) == 1
    @test derivative === nothing

    updater = Heatbath(U2, 2.0; seed=11)
    heatbath!(U2, updater)
    overrelaxation!(U2, updater)
    @test updater.sweep == 1
    @test updater.overrelaxation_sweep == 1

    action = GaugeAction(U2)
    plaquette_loops = make_loops_fromname("plaquette"; Dim=2)
    append!(plaquette_loops, plaquette_loops')
    push!(action, 1.0, plaquette_loops)
    general_updater = Heatbath_update(U2, action; seed=12)
    heatbath!(U2, general_updater)
    overrelaxation!(U2, general_updater)
    @test general_updater.sweep == 1
    @test general_updater.overrelaxation_sweep == 1

    U3 = Initialize_Gaugefields(
        3,
        1,
        4,
        4;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(1, 1),
        verbose_level=0,
    )
    updater3 = Heatbath(U3, 2.0; seed=13)
    heatbath!(U3, updater3)
    overrelaxation!(U3, updater3)
    @test updater3.sweep == 1
    @test updater3.overrelaxation_sweep == 1

    U4 = Initialize_Gaugefields(
        4,
        1,
        4,
        4;
        condition="cold",
        isMPILattice=true,
        PEs=(1, 1),
        verbose_level=0,
    )
    updater4 = Heatbath(U4, 2.0; seed=14)
    heatbath!(U4, updater4)
    overrelaxation!(U4, updater4)
    @test updater4.sweep == 1
    @test updater4.overrelaxation_sweep == 1

    action4 = GaugeAction(U4)
    loops4 = make_loops_fromname("plaquette"; Dim=2)
    append!(loops4, adjoint(loops4))
    push!(action4, 1.0, loops4)
    general_updater4 = Heatbath_update(U4, action4; seed=15)
    heatbath!(U4, general_updater4)
    overrelaxation!(U4, general_updater4)
    @test general_updater4.sweep == 1
    @test general_updater4.overrelaxation_sweep == 1

    plaquette = calculate_Plaquette(
        smeared,
        similar(smeared[1]),
        similar(smeared[1]),
    )
    @test isfinite(real(plaquette))
end

@testset "special configurations on LatticeMatrices" begin
    legacy_2d = Oneinstanton(2, 0, 4, 4; verbose_level=0)
    lm_2d = Oneinstanton(
        2,
        1,
        4,
        4;
        isMPILattice=true,
        PEs=(1, 1),
        verbose_level=0,
    )
    lm_2d_host = similar(legacy_2d)
    substitute_U!(lm_2d_host, lm_2d)
    @test maximum(
        maximum(abs.(lm_2d_host[mu].U .- legacy_2d[mu].U)) for mu = 1:2
    ) < 1e-12

    legacy_4d = Oneinstanton(2, 0, 2, 2, 2, 2; verbose_level=0)
    lm_4d = Oneinstanton(
        2,
        1,
        2,
        2,
        2,
        2;
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        verbose_level=0,
    )
    lm_4d_host = similar(legacy_4d)
    substitute_U!(lm_4d_host, lm_4d)
    @test maximum(
        maximum(abs.(lm_4d_host[mu].U .- legacy_4d[mu].U)) for mu = 1:4
    ) < 1e-12

    legacy_sun = Oneinstanton_SUN_embedded(
        3,
        2,
        2,
        2,
        2;
        verbose_level=0,
    )
    lm_sun = Oneinstanton_SUN_embedded(
        3,
        2,
        2,
        2,
        2;
        NDW=1,
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        verbose_level=0,
    )
    lm_sun_host = similar(legacy_sun)
    substitute_U!(lm_sun_host, lm_sun)
    @test maximum(
        maximum(abs.(lm_sun_host[mu].U .- legacy_sun[mu].U)) for mu = 1:4
    ) < 1e-12

    lm_sun_f32 = Oneinstanton_SUN_embedded(
        3,
        2,
        2,
        2,
        2;
        isMPILattice=true,
        elementtype=ComplexF32,
        PEs=(1, 1, 1, 1),
        verbose_level=0,
    )
    @test eltype(lm_sun_f32[1]) == ComplexF32
    @test_throws ArgumentError Oneinstanton(
        2,
        0,
        2,
        2;
        isMPILattice=true,
        elementtype=Float64,
        verbose_level=0,
    )
end

@testset "B-fields on LatticeMatrices" begin
    dimensions = (2, 2, 2, 2)
    flux = [1, 0, 1, 0, 1, 0]
    B = Initialize_Bfields(
        2,
        flux,
        1,
        dimensions...;
        condition="tflux",
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        verbose_level=0,
    )
    Bcopy = similar(B)
    substitute_U!(Bcopy, B)
    substitute_U!(Bcopy, B, true)
    @test typeof(Bcopy[1, 2]) == typeof(B[1, 2])

    U = Initialize_Gaugefields(
        2,
        1,
        dimensions...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        verbose_level=0,
    )
    plaquettes = make_loops_fromname("plaquette"; Dim=4)
    flow = Gradientflow_general_Bfields(
        U,
        B,
        [plaquettes],
        [1 + 0im];
        eps=0.01,
    )
    flow!(U, B, flow)
    @test isfinite(real(calculate_Plaquette(
        U, similar(U[1]), similar(U[1])
    )))

    Bloop = Initialize_Bfields(
        2,
        flux,
        1,
        dimensions...;
        condition="tloop",
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        elementtype=ComplexF32,
        verbose_level=0,
    )
    @test eltype(Bloop[1, 2]) == ComplexF32
end

@testset "general-NC TA exponential" begin
    for dimensions in ((2, 2), (2, 2, 2, 2))
        process_grid = ntuple(_ -> 1, length(dimensions))
        UNC4 = Initialize_Gaugefields(
            4,
            0,
            dimensions...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid,
            verbose_level=0,
        )
        momentum = initialize_TA_Gaugefields(UNC4)
        gauss_distribution!(momentum; seed=UInt64(123))

        matrix_field = similar(UNC4[1])
        substitute_U!(matrix_field, momentum[1])
        site = (1, 1, ntuple(_ -> 1, length(dimensions))...)
        @test isfinite(real(getindex(matrix_field, site...)))

        exponential = similar(UNC4[1])
        exptU!(exponential, 0.01, momentum[1], [similar(UNC4[1])])
        @test isfinite(real(getindex(exponential, site...)))
    end
end

@testset "4D LatticeMatrices stout forward" begin
    Ulm = Initialize_Gaugefields(
        3,
        0,
        2,
        2,
        2,
        2;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        verbose_level=0,
    )

    momentum = initialize_TA_Gaugefields(Ulm[1])
    momentum_copy = similar(momentum)
    @test typeof(momentum_copy) == typeof(momentum)
    @test momentum_copy.a.dims == momentum.a.dims
    @test momentum_copy.a.phases == momentum.a.phases
    @test momentum_copy.a.comm == momentum.a.comm

    Ulegacy = Initialize_Gaugefields(
        3,
        0,
        2,
        2,
        2,
        2;
        condition="cold",
        verbose_level=0,
    )
    substitute_U!(Ulegacy, Ulm)

    nn_lm = CovNeuralnet(Ulm)
    push!(nn_lm, STOUT_Layer(["plaquette"], [0.1], Ulm))
    smeared_lm, history_lm, derivative_lm = calc_smearedU(Ulm, nn_lm)

    nn_legacy = CovNeuralnet(Ulegacy)
    push!(nn_legacy, STOUT_Layer(["plaquette"], [0.1], Ulegacy))
    smeared_legacy, _, _ = calc_smearedU(Ulegacy, nn_legacy)

    smeared_lm_host = similar(Ulegacy)
    substitute_U!(smeared_lm_host, smeared_lm)
    max_error = maximum(
        maximum(abs.(smeared_lm_host[mu].U .- smeared_legacy[mu].U)) for mu = 1:4
    )

    @test length(history_lm) == 1
    @test derivative_lm === nothing
    @test max_error < 1e-11

    Q_legacy = similar(Ulegacy[1])
    Traceless_antihermitian!(Q_legacy, Ulegacy[1])
    pullback_legacy = similar(Ulegacy[1])
    Gaugefields.Abstractsmearing_module.CdexpQdQ!(
        pullback_legacy,
        Ulegacy[2],
        Q_legacy,
    )

    Q_lm = similar(Ulm[1])
    cotangent_lm = similar(Ulm[1])
    pullback_lm = similar(Ulm[1])
    substitute_U!(Q_lm, Q_legacy)
    substitute_U!(cotangent_lm, Ulegacy[2])
    Gaugefields.Abstractsmearing_module.CdexpQdQ!(
        pullback_lm,
        cotangent_lm,
        Q_lm,
    )
    pullback_lm_host = similar(Ulegacy[1])
    substitute_U!(pullback_lm_host, pullback_lm)
    @test maximum(abs.(pullback_lm_host.U .- pullback_legacy.U)) < 1e-11
end

@testset "4D legacy-compatible API" begin
    @test get_myrank(U) == 0
    @test get_nprocs(U) == 1

    mapped = similar(U[1])
    Gaugefields.AbstractGaugefields_module.unit_U!(mapped)
    map_U_sequential!(
        mapped,
        (matrix, _, ix, iy, iz, it) -> (matrix .*= 2),
        nothing,
    )
    @test mapped[1, 1, 1, 1, 1, 1] == 2 + 0im

    for method in (:plaquette, :clover, :improved)
        density = Gaugefields.AbstractGaugefields_module.topological_charge_density(U; method)
        charge = Gaugefields.AbstractGaugefields_module.topological_charge(U; method)
        @test size(density) == (4, 4, 4, 4)
        @test isapprox(sum(density), charge; rtol=1e-12, atol=1e-12)
    end

    adjoint_U = construct_Adjoint_rep_Gaugefields(U)
    @test length(adjoint_U) == 4
    @test size(adjoint_U[1]) == (8, 8, 4, 4, 4, 4)

    UNC4 = Initialize_Gaugefields(
        4,
        0,
        2,
        2,
        2,
        2;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=(1, 1, 1, 1),
        verbose_level=0,
    )
    exponential = similar(UNC4[1])
    Gaugefields.AbstractGaugefields_module.exptU!(
        exponential,
        0.01,
        UNC4[1],
        [similar(UNC4[1])],
    )
    @test isfinite(real(exponential[1, 1, 1, 1, 1, 1]))

    mktempdir() do directory
        filename = joinpath(directory, "gaugefield_4d.npz")
        write_to_numpyarray(mapped, filename)
        stored = Gaugefields.AbstractGaugefields_module.npzread(filename)
        @test size(stored["U"]) == (3, 3, 4, 4, 4, 4)
        @test stored["NT"] == 4
    end
end
