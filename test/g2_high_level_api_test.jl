using Gaugefields
using Test

function _g2_api_site_matrix(U, ix, iy, iz, it)
    return [U[i, j, ix, iy, iz, it] for i in 1:7, j in 1:7]
end

@testset "G2Backend construction and metadata" begin
    cold = gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        halo=1,
        start=:cold,
        process_grid=(1, 1, 1, 1),
        comm=SerialCommunicator(),
        verbose=0,
    )

    @test length(cold) == 4
    @test all(link isa G2Gaugefields_4D_wing for link in cold)
    @test gauge_backend(cold) isa G2Backend
    @test all(gauge_backend(link) isa G2Backend for link in cold)
    @test gauge_lattice_size(cold) == (2, 2, 2, 2)
    @test gauge_num_colors(cold) == 7
    @test gauge_halo_width(cold) == 1
    @test gauge_process_grid(cold) == (1, 1, 1, 1)
    @test gauge_communicator(cold) === nothing
    @test measure_plaquette(cold) ≈ 1
    @test all(
        is_g2_link(_g2_api_site_matrix(link, 1, 1, 1, 1))
        for link in cold
    )

    momenta = gauge_momenta(cold)
    @test length(momenta) == 4
    @test all(momentum isa G2TA_Gaugefields_4D_serial for momentum in momenta)
    @test all(iszero, (momentum.a for momentum in momenta))

    snapshot = copy_configuration(cold)
    @test snapshot !== cold
    @test all(snapshot[direction].U == cold[direction].U for direction in 1:4)
    clear_U!(cold[1])
    @test snapshot[1].U != cold[1].U
    @test copy_configuration!(cold, snapshot) === cold
    @test all(snapshot[direction].U == cold[direction].U for direction in 1:4)
end

@testset "G2Backend seeded hot start" begin
    hot1 = gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        halo=0,
        start=:hot,
        seed=UInt64(20260903),
        verbose=0,
    )
    hot2 = gauge_configuration(
        [2, 2, 2, 2];
        backend=G2Backend(),
        colors=7,
        halo=0,
        start=:hot,
        seed=UInt64(20260903),
        verbose=0,
    )

    @test all(hot1[direction].U == hot2[direction].U for direction in 1:4)
    @test hot1[1].U != hot1[2].U
    @test isfinite(measure_plaquette(hot1))
    @test all(
        is_g2_link(_g2_api_site_matrix(link, 1, 1, 1, 1); atol=2e-12)
        for link in hot1
    )

    momenta1 = gaussian_momenta(hot1; seed=17, sweep=3, sigma=0.25)
    momenta2 = gaussian_momenta(hot1; seed=17, sweep=3, sigma=0.25)
    @test all(momenta1[direction].a == momenta2[direction].a for direction in 1:4)
    @test momenta1[1].a != momenta1[2].a
    @test all(isfinite, (value for momentum in momenta1 for value in momentum.a))
    @test_throws ArgumentError gaussian_momenta(hot1; sweep=1)
end

@testset "G2Backend rejects ambiguous SU(7) and unsupported execution" begin
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2);
        backend=G2Backend(),
        colors=7,
    )
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=3,
    )
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        eltype=ComplexF32,
    )
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        boundary=(1, 1, 1, -1),
    )
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        process_grid=(1, 1, 1, 2),
    )
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        comm=:not_a_serial_communicator,
    )
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        start=:hot,
        seed=-1,
    )
    @test_throws ArgumentError gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        start=:hot,
        seed=big(typemax(UInt64)) + 1,
    )

    su7 = gauge_configuration(
        (2, 2);
        backend=LegacyBackend(),
        colors=7,
        verbose=0,
    )
    @test gauge_backend(su7) isa LegacyBackend
end

@testset "G2Backend current MD driver" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        backend=G2Backend(),
        colors=7,
        halo=1,
        start=:hot,
        seed=31415,
        verbose=0,
    )
    momenta = gaussian_momenta(U; seed=27182, sigma=0.02)

    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette")
    append!(plaquettes, plaquettes')
    push!(action, 0.2, plaquettes)

    initial_links = [copy(link.U) for link in U]
    initial_momenta = [copy(momentum.a) for momentum in momenta]
    forward = md_driver(
        U,
        action;
        steps=2,
        trajectory_length=0.01,
        integrator=QPQ(),
    )
    diagnostics = md_trajectory!(U, momenta, forward)
    @test isfinite(diagnostics.initial_hamiltonian)
    @test isfinite(diagnostics.final_hamiltonian)
    @test abs(diagnostics.delta_hamiltonian) < 1e-5

    backward = md_driver(
        U,
        action;
        steps=2,
        trajectory_length=-0.01,
        integrator=QPQ(),
    )
    md_trajectory!(U, momenta, backward; diagnostics=false)
    @test maximum(
        maximum(abs, U[direction].U .- initial_links[direction])
        for direction in 1:4
    ) < 2e-12
    @test maximum(
        maximum(abs, momenta[direction].a .- initial_momenta[direction])
        for direction in 1:4
    ) < 2e-12
    @test all(
        is_g2_link(_g2_api_site_matrix(link, 1, 1, 1, 1); atol=2e-11)
        for link in U
    )
end
