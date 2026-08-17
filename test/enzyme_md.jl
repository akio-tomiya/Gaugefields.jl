using Enzyme
using LatticeMatrices: gather_and_bcast_matrix
using LinearAlgebra

function _enzyme_md_plaquette_step!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    mul_shifted!(C, Uμ, Uν, shift_μ)
    mul_shifted_adjoint!(D, C, Uμ, shift_ν)
    mul_adjoint!(E, D, Uν)
    value = realtrace(E)

    mul_shifted!(C, Uν, Uμ, shift_ν)
    mul_shifted_adjoint!(D, C, Uν, shift_μ)
    mul_adjoint!(E, D, Uμ)
    return value + realtrace(E)
end

function _enzyme_md_plaquette_potential(
    U1,
    U2,
    U3,
    U4,
    coefficient,
    colors,
    temps,
)
    C = temps[1]
    D = temps[2]
    E = temps[3]
    shift_1 = (1, 0, 0, 0)
    shift_2 = (0, 1, 0, 0)
    shift_3 = (0, 0, 1, 0)
    shift_4 = (0, 0, 0, 1)
    value = _enzyme_md_plaquette_step!(C, D, E, U1, U2, shift_1, shift_2)
    value += _enzyme_md_plaquette_step!(C, D, E, U1, U3, shift_1, shift_3)
    value += _enzyme_md_plaquette_step!(C, D, E, U1, U4, shift_1, shift_4)
    value += _enzyme_md_plaquette_step!(C, D, E, U2, U3, shift_2, shift_3)
    value += _enzyme_md_plaquette_step!(C, D, E, U2, U4, shift_2, shift_4)
    value += _enzyme_md_plaquette_step!(C, D, E, U3, U4, shift_3, shift_4)
    return -coefficient * value / colors
end

struct _EnzymeTestCustomQPQ end

function Gaugefields.md_step!(::_EnzymeTestCustomQPQ, U, P, step_size, driver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    update_momenta!(P, U, step_size, driver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    return nothing
end

function _enzyme_md_action(U, coefficient)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, coefficient, plaquettes)
    return action
end

function _enzyme_md_global_links(U)
    return gather_and_bcast_matrix.(getproperty.(U, :U))
end

function _enzyme_md_global_momenta(p)
    return gather_and_bcast_matrix.(getproperty.(p, :a))
end

function _enzyme_md_maximum_difference(left, right)
    return maximum(
        maximum(abs, left[direction] .- right[direction])
        for direction in eachindex(left)
    )
end

@testset "Enzyme MD action provider" begin
    lattice = (2, 2, 2, 2)
    configuration_arguments = (
        colors=2,
        start=:hot,
        seed=UInt64(0x1234),
        halo=1,
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    coefficient = 2.75
    U = gauge_configuration(lattice; configuration_arguments...)
    analytic_action = _enzyme_md_action(U, coefficient)
    automatic_action = enzyme_md_action(
        _enzyme_md_plaquette_potential,
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
    @test _enzyme_md_maximum_difference(
        _enzyme_md_global_momenta(analytic_force),
        _enzyme_md_global_momenta(automatic_force),
    ) < 2e-10

    U_analytic = gauge_configuration(lattice; configuration_arguments...)
    U_automatic = gauge_configuration(lattice; configuration_arguments...)
    p_analytic = gaussian_momenta(U_analytic; seed=UInt64(0x5678))
    p_automatic = gaussian_momenta(U_automatic; seed=UInt64(0x5678))
    analytic_driver = md_driver(
        U_analytic,
        _enzyme_md_action(U_analytic, coefficient);
        steps=2,
        trajectory_length=0.04,
        integrator=QPQ(),
    )
    automatic_driver = md_driver(
        U_automatic,
        automatic_action;
        steps=2,
        trajectory_length=0.04,
        integrator=_EnzymeTestCustomQPQ(),
    )
    analytic_result = md_trajectory!(U_analytic, p_analytic, analytic_driver)
    automatic_result = md_trajectory!(
        U_automatic,
        p_automatic,
        automatic_driver,
    )

    @test analytic_result.initial_hamiltonian ≈
          automatic_result.initial_hamiltonian rtol = 2e-12
    @test analytic_result.final_hamiltonian ≈
          automatic_result.final_hamiltonian rtol = 2e-10
    @test _enzyme_md_maximum_difference(
        _enzyme_md_global_links(U_analytic),
        _enzyme_md_global_links(U_automatic),
    ) < 2e-10
    @test _enzyme_md_maximum_difference(
        _enzyme_md_global_momenta(p_analytic),
        _enzyme_md_global_momenta(p_automatic),
    ) < 2e-10

    initial_links = _enzyme_md_global_links(U_automatic)
    initial_momenta = _enzyme_md_global_momenta(p_automatic)
    forward = md_driver(
        U_automatic,
        automatic_action;
        steps=2,
        trajectory_length=0.03,
        integrator=PQP(),
    )
    backward = md_driver(
        U_automatic,
        automatic_action;
        steps=2,
        trajectory_length=-0.03,
        integrator=PQP(),
    )
    md_trajectory!(U_automatic, p_automatic, forward; diagnostics=false)
    md_trajectory!(U_automatic, p_automatic, backward; diagnostics=false)
    @test _enzyme_md_maximum_difference(
        _enzyme_md_global_links(U_automatic),
        initial_links,
    ) < 2e-10
    @test _enzyme_md_maximum_difference(
        _enzyme_md_global_momenta(p_automatic),
        initial_momenta,
    ) < 2e-10

    @test_throws ArgumentError enzyme_md_action(
        _enzyme_md_plaquette_potential;
        num_temps=-1,
    )
    legacy = Initialize_Gaugefields(
        2,
        1,
        lattice...;
        condition="cold",
        verbose_level=0,
    )
    @test_throws ArgumentError md_driver(
        legacy,
        automatic_action;
        steps=1,
    )
end
