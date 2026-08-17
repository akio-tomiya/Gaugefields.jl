import JACC
JACC.@init_backend

using Enzyme
using Gaugefields
using LatticeMatrices: gather_and_bcast_matrix
using LinearAlgebra
using MPI
using Test

MPI.Initialized() || MPI.Init()

function _mpi_enzyme_plaquette_step!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    mul_shifted!(C, Uμ, Uν, shift_μ)
    mul_shifted_adjoint!(D, C, Uμ, shift_ν)
    mul_adjoint!(E, D, Uν)
    value = realtrace(E)

    mul_shifted!(C, Uν, Uμ, shift_ν)
    mul_shifted_adjoint!(D, C, Uν, shift_μ)
    mul_adjoint!(E, D, Uμ)
    return value + realtrace(E)
end

function _mpi_enzyme_potential(U1, U2, U3, U4, coefficient, colors, temps)
    C = temps[1]
    D = temps[2]
    E = temps[3]
    shift_1 = (1, 0, 0, 0)
    shift_2 = (0, 1, 0, 0)
    shift_3 = (0, 0, 1, 0)
    shift_4 = (0, 0, 0, 1)
    value = _mpi_enzyme_plaquette_step!(C, D, E, U1, U2, shift_1, shift_2)
    value += _mpi_enzyme_plaquette_step!(C, D, E, U1, U3, shift_1, shift_3)
    value += _mpi_enzyme_plaquette_step!(C, D, E, U1, U4, shift_1, shift_4)
    value += _mpi_enzyme_plaquette_step!(C, D, E, U2, U3, shift_2, shift_3)
    value += _mpi_enzyme_plaquette_step!(C, D, E, U2, U4, shift_2, shift_4)
    value += _mpi_enzyme_plaquette_step!(C, D, E, U3, U4, shift_3, shift_4)
    return -coefficient * value / colors
end

function _mpi_enzyme_analytic_action(U, coefficient)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, coefficient, plaquettes)
    return action
end

function _mpi_enzyme_difference(left, right)
    return maximum(
        maximum(abs, left[direction] .- right[direction])
        for direction in eachindex(left)
    )
end

struct _MPIEnzymeQPQ end

function Gaugefields.md_step!(::_MPIEnzymeQPQ, U, P, step_size, driver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    update_momenta!(P, U, step_size, driver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    return nothing
end

@testset "Enzyme molecular dynamics with MPI" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    nprocs <= 2 || error("this test supports at most two MPI ranks")
    lattice = (2 * nprocs, 2, 2, 2)
    process_grid = (nprocs, 1, 1, 1)
    coefficient = 1.75

    U = gauge_configuration(
        lattice;
        colors=2,
        start=:hot,
        seed=UInt64(0x1234),
        halo=1,
        process_grid,
        verbose=0,
    )
    analytic_action = _mpi_enzyme_analytic_action(U, coefficient)
    automatic_action = enzyme_md_action(
        _mpi_enzyme_potential,
        coefficient,
        2;
        num_temps=3,
    )
    analytic_workspace = md_action_workspace(analytic_action, U)
    automatic_workspace = md_action_workspace(automatic_action, U)
    @test md_potential(analytic_action, U, analytic_workspace) ≈
          md_potential(automatic_action, U, automatic_workspace) rtol = 2e-12

    analytic_force = gauge_momenta(U)
    automatic_force = gauge_momenta(U)
    md_force!(analytic_force, analytic_action, U, analytic_workspace)
    md_force!(automatic_force, automatic_action, U, automatic_workspace)
    @test _mpi_enzyme_difference(
        gather_and_bcast_matrix.(getproperty.(analytic_force, :a)),
        gather_and_bcast_matrix.(getproperty.(automatic_force, :a)),
    ) < 2e-10

    momenta = gaussian_momenta(U; seed=UInt64(0x5678))
    initial_links = gather_and_bcast_matrix.(getproperty.(U, :U))
    initial_momenta = gather_and_bcast_matrix.(getproperty.(momenta, :a))
    forward = md_driver(
        U,
        automatic_action;
        steps=2,
        trajectory_length=0.02,
        integrator=_MPIEnzymeQPQ(),
    )
    backward = md_driver(
        U,
        automatic_action;
        steps=2,
        trajectory_length=-0.02,
        integrator=_MPIEnzymeQPQ(),
    )
    result = md_trajectory!(U, momenta, forward)
    @test isfinite(result.delta_hamiltonian)
    md_trajectory!(U, momenta, backward; diagnostics=false)
    @test _mpi_enzyme_difference(
        gather_and_bcast_matrix.(getproperty.(U, :U)),
        initial_links,
    ) < 2e-10
    @test _mpi_enzyme_difference(
        gather_and_bcast_matrix.(getproperty.(momenta, :a)),
        initial_momenta,
    ) < 2e-10
end
