using LatticeMatrices: gather_and_bcast_matrix

function _uv_gauge_maximum_difference(left, right)
    return maximum(
        maximum(abs, gather_and_bcast_matrix(left[mu].U) .-
                     gather_and_bcast_matrix(right[mu].U))
        for mu in eachindex(left)
    )
end

function _uv_gauge_test_action(U)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=length(U))
    append!(plaquettes, plaquettes')
    push!(action, 5.7, plaquettes)
    return action
end

@testset "Native UV-smearing API" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(0x5556534d),
        process_grid=(1, 1, 1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )

    @test StoutSmearing().parameters.rho == 0.1
    @test EXPSmearing === StoutSmearing
    @test HEXSmearing().parameters ==
          LatticeMatrices.HEXParameters(0.125, 0.15, 0.15)
    @test APESmearing().parameters.alpha == 0.6
    @test HYPSmearing().parameters ==
          LatticeMatrices.HYPParameters(0.75, 0.6, 0.3)
    @test_throws ArgumentError StoutSmearing(iterations=0)
    @test has_smearing_pullback(NHYPSmearing())
    @test has_smearing_pullback(StoutSmearing())
    @test has_smearing_pullback(HEXSmearing())
    @test !has_smearing_pullback(APESmearing())
    @test !has_smearing_pullback(HYPSmearing())
    @test has_smearing_pullback(APESmearing(projection=:polar))
    @test has_smearing_pullback(HYPSmearing(projection=:polar))
    exception = try
        smear(U, APESmearing(); calcdSdU=true)
        nothing
    catch caught
        caught
    end
    @test exception isa ArgumentError
    @test occursin("projection=:polar", sprint(showerror, exception))

    action_exception = try
        SmearedGaugeAction(_uv_gauge_test_action(U), HYPSmearing())
        nothing
    catch caught
        caught
    end
    @test action_exception isa ArgumentError
    @test occursin(
        "projection=:polar", sprint(showerror, action_exception))

    specifications = (
        StoutSmearing(rho=0, iterations=2),
        HEXSmearing(alpha_outer=0, alpha_middle=0, alpha_inner=0),
        APESmearing(alpha=0, projection=:polar),
        HYPSmearing(
            alpha_outer=0, alpha_middle=0, alpha_inner=0,
            projection=:polar),
    )
    for specification in specifications
        recorded = smear(U, specification; record=true)
        @test recorded.history isa LinkSmearingCache
        @test _uv_gauge_maximum_difference(recorded.configuration, U) < 8e-11
    end

    left = gauge_configuration(
        (2, 2, 2, 2);
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(0x434f5441),
        process_grid=(1, 1, 1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )
    for specification in (
        StoutSmearing(), HEXSmearing(),
        APESmearing(projection=:polar),
        HYPSmearing(projection=:polar),
    )
        smeared, cache = link_smear(U, specification)
        dU = link_smear_pullback(left, U, cache)
        @test length(smeared) == 4
        @test length(dU) == 4
    end

    @test stout_link_smearing(U; iterations=2) isa StoutSmearing
    @test exp_smearing(U) isa EXPSmearing
    @test hex_smearing(U) isa HEXSmearing
    @test ape_smearing(U) isa APESmearing
    @test hyp_smearing(U) isa HYPSmearing
    @test stout_smearing(U) isa CovNeuralnet
end

@testset "Analytically smeared GaugeAction" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(0x534d4741),
        process_grid=(1, 1, 1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )
    thin_action = _uv_gauge_test_action(U)

    direction = gaussian_momenta(
        U;
        sigma=0.2,
        seed=UInt64(0x534d4449),
    )
    epsilon = 1e-6
    for specification in (
        StoutSmearing(rho=0.07), HEXSmearing(),
        APESmearing(alpha=0.5, projection=:polar),
        HYPSmearing(projection=:polar),
    )
        action = SmearedGaugeAction(thin_action, specification)
        workspace = md_action_workspace(action, U)
        force = gauge_momenta(U)
        md_force!(force, action, U, workspace)

        driver = md_driver(U, action; steps=1)
        plus_U = copy_configuration(U)
        minus_U = copy_configuration(U)
        update_gaugefields!(plus_U, direction, epsilon, driver)
        update_gaugefields!(minus_U, direction, -epsilon, driver)
        finite_difference = (
            md_potential(action, plus_U, workspace) -
            md_potential(action, minus_U, workspace)
        ) / (2epsilon)
        analytic = -real(direction * force)
        @test analytic ≈ finite_difference atol=3e-6 rtol=3e-6
    end
end
