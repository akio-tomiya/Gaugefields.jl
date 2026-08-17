import JACC
JACC.@init_backend

using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using MPI
using Test

MPI.Initialized() || MPI.Init()

struct _MPITestCustomPQP end

function Gaugefields.md_step!(::_MPITestCustomPQP, U, P, step_size, driver)
    update_momenta!(P, U, 0.5 * step_size, driver)
    update_gaugefields!(U, P, step_size, driver)
    update_momenta!(P, U, 0.5 * step_size, driver)
    return nothing
end

function _mpi_md_action(U)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=length(U))
    append!(plaquettes, plaquettes')
    push!(action, 1.0, plaquettes)
    return action
end

function _mpi_md_difference(left, right)
    return maximum(
        maximum(abs, left[direction] .- right[direction])
        for direction in eachindex(left)
    )
end

@testset "Molecular dynamics with MPI" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    lattice = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)

    cold = gauge_configuration(
        lattice;
        colors=2,
        start=:cold,
        process_grid,
        verbose=0,
    )
    zero_momenta = gauge_momenta(cold)
    cold_action = _mpi_md_action(cold)
    custom_driver = md_driver(
        cold,
        cold_action;
        steps=2,
        trajectory_length=0.1,
        integrator=_MPITestCustomPQP(),
    )
    cold_result = md_trajectory!(cold, zero_momenta, custom_driver)
    @test cold_result.delta_hamiltonian == 0
    @test measure_plaquette(cold) ≈ 1

    hot = gauge_configuration(
        lattice;
        colors=2,
        start=:hot,
        seed=UInt64(0x1234),
        process_grid,
        verbose=0,
    )
    momenta = gaussian_momenta(hot; seed=UInt64(0x5678))
    initial_links = gather_and_bcast_matrix.(getproperty.(hot, :U))
    initial_momenta = gather_and_bcast_matrix.(getproperty.(momenta, :a))
    action = _mpi_md_action(hot)

    forward = md_driver(
        hot,
        action;
        steps=3,
        trajectory_length=0.1,
        integrator=QPQ(),
    )
    result = md_trajectory!(hot, momenta, forward)
    @test isfinite(result.delta_hamiltonian)

    backward = md_driver(
        hot,
        action;
        steps=3,
        trajectory_length=-0.1,
        integrator=QPQ(),
    )
    md_trajectory!(hot, momenta, backward; diagnostics=false)
    final_links = gather_and_bcast_matrix.(getproperty.(hot, :U))
    final_momenta = gather_and_bcast_matrix.(getproperty.(momenta, :a))
    @test _mpi_md_difference(final_links, initial_links) < 2e-12
    @test _mpi_md_difference(final_momenta, initial_momenta) < 2e-12
end
