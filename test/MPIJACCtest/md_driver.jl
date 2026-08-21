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

function _mpi_test_gauge_action(U; coupling=1.0)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=length(U))
    append!(plaquettes, plaquettes')
    push!(action, coupling, plaquettes)
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

    automatic = gauge_configuration(
        lattice;
        colors=2,
        start=:cold,
        process_grid=:auto,
        comm=MPI.COMM_WORLD,
        verbose=0,
    )
    @test prod(gauge_process_grid(automatic)) == nprocs
    @test MPI.Comm_compare(
        gauge_communicator(automatic),
        MPI.COMM_WORLD,
    ) in (MPI.IDENT, MPI.CONGRUENT)

    cold = gauge_configuration(
        lattice;
        colors=2,
        start=:cold,
        process_grid,
        verbose=0,
    )
    zero_momenta = gauge_momenta(cold)
    cold_action = _mpi_test_gauge_action(cold)
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
    refreshed_momenta = gauge_momenta(hot)
    gaussian_momenta!(refreshed_momenta; seed=UInt64(0x5678))
    @test _mpi_md_difference(
        gather_and_bcast_matrix.(getproperty.(refreshed_momenta, :a)),
        gather_and_bcast_matrix.(getproperty.(momenta, :a)),
    ) == 0
    hot_snapshot = copy_configuration(hot)
    @test _mpi_md_difference(
        gather_and_bcast_matrix.(getproperty.(hot_snapshot, :U)),
        gather_and_bcast_matrix.(getproperty.(hot, :U)),
    ) == 0
    initial_links = gather_and_bcast_matrix.(getproperty.(hot, :U))
    initial_momenta = gather_and_bcast_matrix.(getproperty.(momenta, :a))
    action = _mpi_test_gauge_action(hot)

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

    split_hot = gauge_configuration(
        lattice;
        colors=2,
        start=:hot,
        seed=UInt64(0x9abc),
        process_grid,
        verbose=0,
    )
    split_momenta = gaussian_momenta(split_hot; seed=UInt64(0xdef0))
    split_initial_links = gather_and_bcast_matrix.(
        getproperty.(split_hot, :U),
    )
    split_initial_momenta = gather_and_bcast_matrix.(
        getproperty.(split_momenta, :a),
    )
    split_actions = MDActionSet(;
        slow=_mpi_test_gauge_action(split_hot; coupling=0.75),
        fast=_mpi_test_gauge_action(split_hot; coupling=0.25),
    )
    split_integrator = SextonWeingarten(;
        slow=:slow,
        fast=:fast,
        n_fast=2,
    )
    split_forward = md_driver(
        split_hot,
        split_actions;
        steps=2,
        trajectory_length=0.1,
        integrator=split_integrator,
    )
    split_result = md_trajectory!(split_hot, split_momenta, split_forward)
    @test isfinite(split_result.delta_hamiltonian)

    split_backward = md_driver(
        split_hot,
        split_actions;
        steps=2,
        trajectory_length=-0.1,
        integrator=split_integrator,
    )
    md_trajectory!(
        split_hot,
        split_momenta,
        split_backward;
        diagnostics=false,
    )
    @test _mpi_md_difference(
        gather_and_bcast_matrix.(getproperty.(split_hot, :U)),
        split_initial_links,
    ) < 3e-12
    @test _mpi_md_difference(
        gather_and_bcast_matrix.(getproperty.(split_momenta, :a)),
        split_initial_momenta,
    ) < 3e-12
end
