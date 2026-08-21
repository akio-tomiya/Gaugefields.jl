using MPI
using JLD2
using LatticeMatrices: gather_and_bcast_matrix

@testset "High-level API" begin
    @test similar(Union{}[]) == Union{}[]

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

    automatic = gauge_configuration(
        (4, 4);
        colors=2,
        start=:cold,
        process_grid=:auto,
        comm=MPI.COMM_SELF,
        verbose=0,
    )
    @test gauge_process_grid(automatic) == (1, 1)
    @test MPI.Comm_compare(
        gauge_communicator(automatic),
        MPI.COMM_SELF,
    ) in (MPI.IDENT, MPI.CONGRUENT)

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
    refreshed = gauge_momenta(hot1)
    @test gaussian_momenta!(
        refreshed;
        seed=UInt64(5678),
        sigma=0.5,
    ) === refreshed
    @test all(
        refreshed[mu].a.A == momenta[mu].a.A
        for mu in eachindex(refreshed)
    )
    @test_throws ArgumentError gaussian_momenta!(refreshed; sweep=-1)

    snapshot = copy_configuration(hot1)
    @test snapshot !== hot1
    @test all(
        gather_and_bcast_matrix(snapshot[mu].U) ==
        gather_and_bcast_matrix(hot1[mu].U)
        for mu in eachindex(hot1)
    )

    flow = gradient_flow(hot1; steps=1, step_size=0.01)
    @test flow.eps isa Float64
    flow!(hot1, flow)
    @test isfinite(measure_plaquette(hot1))
    @test any(
        gather_and_bcast_matrix(snapshot[mu].U) !=
        gather_and_bcast_matrix(hot1[mu].U)
        for mu in eachindex(hot1)
    )
    @test copy_configuration!(hot1, snapshot) === hot1
    @test all(
        gather_and_bcast_matrix(snapshot[mu].U) ==
        gather_and_bcast_matrix(hot1[mu].U)
        for mu in eachindex(hot1)
    )

    flow32 = gradient_flow(lm; steps=1, step_size=0.01)
    @test flow32.eps isa Float32
    flow!(lm, flow32)
    @test isfinite(measure_plaquette(lm))

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
        stored = JLD2.load(filename)
        @test stored["gaugefields_format"] ==
              "Gaugefields.jl portable gauge configuration"
        @test stored["gaugefields_format_version"] == 1
        @test stored["lattice_size"] == [4, 4]
        @test stored["num_colors"] == 2
        @test all(link isa Array for link in stored["links"])
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

        target32 = gauge_configuration(
            (4, 4);
            colors=2,
            start=:cold,
            process_grid=(1, 1),
            eltype=ComplexF32,
            verbose=0,
        )
        load_configuration!(target32, filename)
        @test all(
            gather_and_bcast_matrix(target32[mu].U) ==
            ComplexF32.(gather_and_bcast_matrix(hot2[mu].U))
            for mu in 1:2
        )

        legacy_filename = joinpath(directory, "legacy-configuration.jld2")
        save_configuration(legacy_filename, legacy)
        legacy_loaded = load_configuration(
            legacy_filename;
            process_grid=(1, 1),
        )
        @test gauge_backend(legacy_loaded) isa LatticeMatricesBackend
        @test measure_plaquette(legacy_loaded) ≈ 1

        object_filename = joinpath(directory, "legacy-object-format.jld2")
        saveU(object_filename, hot2)
        object_loaded = load_configuration(object_filename)
        @test typeof(object_loaded) == typeof(hot2)
        @test all(object_loaded[mu].U.A == hot2[mu].U.A for mu in 1:2)
    end
end
