import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

function local_storage(field)
    return JACC.to_host(field.U.A)
end

function plaquette_action(U)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette"; Dim=3)
    append!(plaquettes, adjoint(plaquettes))
    push!(action, 1.0, plaquettes)
    return action
end

@testset "3D LatticeMatrices compatibility" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs <= 2 || error("this regression test supports at most two MPI ranks")
    global_size = (2 * nprocs, 2, 2)
    process_grid = (nprocs, 1, 1)

    cold = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        elementtype=ComplexF32,
        verbose_level=0,
    )
    @test length(cold) == 3
    @test size(cold[1]) == (3, 3, global_size...)
    @test eltype(cold[1]) == ComplexF32
    @test get_nprocs(cold) == nprocs

    plaquette = calculate_Plaquette(
        cold,
        similar(cold[1]),
        similar(cold[1]),
    )
    @test plaquette ≈ binomial(3, 2) * 3 * prod(global_size)
    polyakov = calculate_Polyakov_loop(
        cold,
        similar(cold[1]),
        similar(cold[1]),
    )
    @test polyakov ≈ 3

    shifted = shift_U(cold[1], 3)
    @test shifted[1, 1, 1, 1, 1] == 1
    close(shifted)

    staggered = Gaugefields.AbstractGaugefields_module.staggered_U(cold[1], 3)
    global_x = cold[1].U.coords[1] * cold[1].U.PN[1]
    @test staggered[1, 1, 1, 1, 1] == (iseven(global_x) ? 1 : -1)

    hot1 = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    hot2 = Initialize_Gaugefields(
        3,
        1,
        global_size...;
        condition="hot",
        randomnumber="Reproducible",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    @test all(
        local_storage(hot1[mu]) == local_storage(hot2[mu]) for mu in 1:3
    )

    legacy = Initialize_Gaugefields(
        3,
        0,
        global_size...;
        condition="cold",
        verbose_level=0,
    )
    substitute_U!(legacy, hot1)
    roundtrip = similar(hot1)
    substitute_U!(roundtrip, legacy)
    roundtrip_legacy = similar(legacy)
    substitute_U!(roundtrip_legacy, roundtrip)
    @test all(
        roundtrip_legacy[mu].U ≈ legacy[mu].U for mu in 1:3
    )

    momenta1 = initialize_TA_Gaugefields(hot1)
    momenta2 = initialize_TA_Gaugefields(hot1)
    gauss_distribution!(momenta1; seed=0x12345678)
    gauss_distribution!(momenta2; seed=0x12345678)
    @test all(
        JACC.to_host(momenta1[mu].a.A) == JACC.to_host(momenta2[mu].a.A)
        for mu in 1:3
    )
    @test isfinite(momenta1 * momenta1)

    exponential = similar(hot1[1])
    exptU!(exponential, 0.01, momenta1[1], [similar(hot1[1])])
    @test isfinite(real(tr(exponential)))

    flow = Gradientflow(hot1; eps=0.01)
    flow!(hot1, flow)
    network = CovNeuralnet(hot1)
    push!(network, STOUT_Layer(["plaquette"], [0.1], hot1))
    smeared, history, derivative = calc_smearedU(hot1, network)
    @test length(history) == 1
    @test derivative === nothing
    @test isfinite(real(calculate_Plaquette(
        smeared,
        similar(smeared[1]),
        similar(smeared[1]),
    )))

    su2 = Initialize_Gaugefields(
        2,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    updater = Heatbath(su2, 2.0; seed=17)
    heatbath!(su2, updater)
    overrelaxation!(su2, updater)
    @test updater.sweep == 1
    @test updater.overrelaxation_sweep == 1

    su4 = Initialize_Gaugefields(
        4,
        1,
        global_size...;
        condition="cold",
        isMPILattice=true,
        PEs=process_grid,
        verbose_level=0,
    )
    su4_action = plaquette_action(su4)
    general_updater = Gaugefields.heatbath_module.Heatbath_update(
        su4,
        su4_action;
        seed=19,
    )
    heatbath!(su4, general_updater)
    overrelaxation!(su4, general_updater)
    @test general_updater.sweep == 1
    @test general_updater.overrelaxation_sweep == 1

    su4_momenta = initialize_TA_Gaugefields(su4)
    gauss_distribution!(su4_momenta; seed=23)
    derivative = similar(su4[1])
    force = similar(su4[1])
    calc_dSdUμ!(derivative, su4_action, 1, su4)
    mul!(force, su4[1], derivative)

    projected_lm = initialize_TA_Gaugefields(force)
    Traceless_antihermitian!(projected_lm, force)
    force_legacy = Initialize_Gaugefields(
        4,
        0,
        global_size...;
        condition="cold",
        verbose_level=0,
    )[1]
    substitute_U!(force_legacy, force)
    projected_legacy = initialize_TA_Gaugefields(force_legacy)
    Traceless_antihermitian!(projected_legacy, force_legacy)
    projected_global = LatticeMatrices.gather_and_bcast_matrix(projected_lm.a)
    @test dropdims(projected_global; dims=2) ≈ projected_legacy.a

    Traceless_antihermitian_add!(su4_momenta[1], -0.01, force)
    update = similar(su4[1])
    exptU!(update, 0.01, su4_momenta[1], [similar(su4[1])])
    @test isfinite(su4_momenta * su4_momenta)
    @test isfinite(real(tr(update)))
end
