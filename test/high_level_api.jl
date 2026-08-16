@testset "High-level API" begin
    for dimensions in ((2, 2), (2, 2, 2), (2, 2, 2, 2))
        configuration = gauge_configuration(
            dimensions;
            colors=2,
            start=:cold,
            process_grid=ntuple(_ -> 1, length(dimensions)),
            verbose=0,
        )
        @test length(configuration) == length(dimensions)
        @test gauge_backend(configuration) isa LatticeMatricesBackend
        @test all(
            gauge_backend(link) isa LatticeMatricesBackend
            for link in configuration
        )
    end

    historical_default = Initialize_Gaugefields(
        2,
        0,
        2,
        2;
        condition="cold",
        verbose_level=0,
    )
    @test gauge_backend(historical_default) isa LegacyBackend

    legacy = gauge_configuration(
        (4, 4);
        backend=LegacyBackend(),
        colors=2,
        start=:cold,
        verbose=0,
    )
    @test length(legacy) == 2
    @test gauge_backend(legacy) isa LegacyBackend
    @test gauge_lattice_size(legacy) == (4, 4)
    @test gauge_num_colors(legacy) == 2
    @test gauge_halo_width(legacy) == 1
    @test gauge_process_grid(legacy) == (1, 1)
    @test gauge_communicator(legacy) === nothing
    @test measure_plaquette(legacy) ≈ 1
    @test measure_polyakov_loop(legacy) ≈ 1

    @test_throws ArgumentError gauge_configuration(
        (4, 4, 4);
        backend=LegacyBackend(),
        halo=1,
    )
    @test_throws ArgumentError gauge_configuration(
        (4, 4);
        backend=LegacyBackend(),
        seed=1,
        start=:hot,
    )
    @test_throws ArgumentError gauge_configuration((4,); colors=2)
    @test_throws ArgumentError gauge_configuration((4, 4); start=:warm)

    lm = gauge_configuration(
        (4, 4);
        colors=2,
        start=:cold,
        process_grid=(1, 1),
        eltype=ComplexF32,
        verbose=0,
    )
    @test length(lm) == 2
    @test gauge_backend(lm) isa LatticeMatricesBackend
    @test gauge_lattice_size(lm) == (4, 4)
    @test gauge_num_colors(lm) == 2
    @test gauge_halo_width(lm) == 1
    @test gauge_process_grid(lm) == (1, 1)
    @test gauge_communicator(lm) !== nothing
    @test eltype(lm[1]) == ComplexF32
    @test measure_plaquette(lm) ≈ 1 atol=2f-6
    @test measure_polyakov_loop(lm) ≈ 1 atol=2f-6

    hot1 = gauge_configuration(
        (4, 4);
        colors=2,
        start=:hot,
        seed=UInt64(1234),
        process_grid=(1, 1),
        verbose=0,
    )
    hot2 = gauge_configuration(
        [4, 4];
        colors=2,
        start=:hot,
        seed=UInt64(1234),
        process_grid=[1, 1],
        verbose=0,
    )
    @test all(hot1[mu].U.A == hot2[mu].U.A for mu in 1:2)

    zero_momenta = gauge_momenta(hot1)
    @test length(zero_momenta) == 2
    momenta = gaussian_momenta(hot1; seed=UInt64(5678), sigma=0.5)
    @test isfinite(momenta * momenta)

    flow = gradient_flow(hot1; steps=1, step_size=0.01)
    flow!(hot1, flow)
    @test isfinite(measure_plaquette(hot1))

    updater = heatbath_updater(lm; beta=2.0, seed=17)
    heatbath!(lm, updater)
    @test updater.sweep == 1

    smearing = stout_smearing(hot2; loops=:plaquette, rho=0.1)
    result = smear(hot2, smearing; record=true)
    @test length(result.configuration) == 2
    @test length(result.history) == 1
    @test result.derivative === nothing
    @test isfinite(measure_plaquette(result.configuration))

    mktempdir() do directory
        filename = joinpath(directory, "configuration.jld2")
        save_configuration(filename, hot2)
        loaded = load_configuration(filename)
        @test typeof(loaded) == typeof(hot2)

        target = gauge_configuration(
            (4, 4);
            colors=2,
            start=:cold,
            process_grid=(1, 1),
            verbose=0,
        )
        returned = load_configuration!(target, filename)
        @test returned === target
        @test all(target[mu].U.A == hot2[mu].U.A for mu in 1:2)
    end
end
