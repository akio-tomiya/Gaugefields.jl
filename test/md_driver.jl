using LatticeMatrices: gather_and_bcast_matrix
using Random

function _test_custom_qpq!(U, P, step_size, driver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    update_momenta!(P, U, step_size, driver)
    update_gaugefields!(U, P, 0.5 * step_size, driver)
    return nothing
end

struct _TestInvalidIntegrator end

struct _CountingMDAction{A}
    action::A
    force_calls::Base.RefValue{Int}
end

Gaugefields.md_action_workspace(action::_CountingMDAction, U) =
    md_action_workspace(action.action, U)

Gaugefields.md_potential(action::_CountingMDAction, U, workspace) =
    md_potential(action.action, U, workspace)

function Gaugefields.md_force!(
    force,
    action::_CountingMDAction,
    U,
    workspace,
)
    action.force_calls[] += 1
    return md_force!(force, action.action, U, workspace)
end

function _md_test_action(U; coupling=1.0)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=length(U))
    append!(plaquettes, plaquettes')
    push!(action, coupling, plaquettes)
    return action
end

function _md_maximum_difference(left, right)
    return maximum(
        maximum(abs, left[direction] .- right[direction])
        for direction in eachindex(left)
    )
end

function _md_global_links(U)
    return gather_and_bcast_matrix.(getproperty.(U, :U))
end

function _md_global_momenta(p)
    return gather_and_bcast_matrix.(getproperty.(p, :a))
end

@testset "Explicit integrator interface" begin
    @test PQP() isa AbstractMDIntegrator
    @test QPQ() isa AbstractMDIntegrator
    @test SextonWeingarten(
        slow=:fermion,
        fast=:gauge,
        n_fast=2,
    ) isa AbstractMDIntegrator
    @test _test_custom_qpq! isa Function
end

@testset "Multiple MD actions and time scales" begin
    lattice = (2, 2, 2, 2)
    configuration_arguments = (
        colors=2,
        start=:hot,
        seed=UInt64(0x1234),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    U = gauge_configuration(lattice; configuration_arguments...)
    first_action = _md_test_action(U; coupling=0.25)
    second_action = _md_test_action(U; coupling=0.75)
    total_action = _md_test_action(U; coupling=1.0)
    actions = MDActionSet(; first=first_action, second=second_action)

    actions_workspace = md_action_workspace(actions, U)
    total_workspace = md_action_workspace(total_action, U)
    @test md_potential(actions, U, actions_workspace) ≈
          md_potential(total_action, U, total_workspace)

    actions_force = initialize_TA_Gaugefields(U)
    total_force = initialize_TA_Gaugefields(U)
    md_force!(actions_force, actions, U, actions_workspace)
    md_force!(total_force, total_action, U, total_workspace)
    @test _md_maximum_difference(
        _md_global_momenta(actions_force),
        _md_global_momenta(total_force),
    ) < 2e-12

    all_momenta = gauge_momenta(U)
    grouped_momenta = gauge_momenta(U)
    actions_driver = md_driver(U, actions; steps=1)
    update_momenta!(all_momenta, U, 0.02, actions_driver)
    update_momenta!(
        grouped_momenta,
        U,
        0.02,
        actions_driver,
        MDForceGroup(:first),
    )
    update_momenta!(
        grouped_momenta,
        U,
        0.02,
        actions_driver,
        MDForceGroup((:second,)),
    )
    @test _md_maximum_difference(
        _md_global_momenta(all_momenta),
        _md_global_momenta(grouped_momenta),
    ) < 2e-12

    three_actions = MDActionSet(;
        first=first_action,
        second=second_action,
        third=_md_test_action(U; coupling=0.5),
    )
    grouped_integrator = SextonWeingarten(;
        slow=(:second, :third),
        fast=:first,
        n_fast=2,
    )
    @test md_driver(
        U,
        three_actions;
        steps=1,
        integrator=grouped_integrator,
    ) isa MDDriver
    @test typeof(grouped_integrator) === typeof(SextonWeingarten(;
        slow=(:second, :third),
        fast=:first,
        n_fast=7,
    ))

    @test_throws ArgumentError MDActionSet()
    @test_throws ArgumentError MDForceGroup()
    @test_throws ArgumentError MDForceGroup(:first, :first)
    @test_throws ArgumentError SextonWeingarten(
        slow=:second,
        fast=:first,
        n_fast=0,
    )
    @test_throws ArgumentError SextonWeingarten(
        slow=:second,
        fast=:first,
        n_fast=1,
        ordering=_TestInvalidIntegrator(),
    )
    @test_throws ArgumentError md_driver(
        U,
        actions;
        steps=1,
        integrator=SextonWeingarten(
            slow=:first,
            fast=:first,
            n_fast=1,
        ),
    )
    @test_throws ArgumentError md_driver(
        U,
        three_actions;
        steps=1,
        integrator=SextonWeingarten(
            slow=:second,
            fast=:first,
            n_fast=1,
        ),
    )
    @test_throws ArgumentError md_driver(
        U,
        actions;
        steps=1,
        integrator=SextonWeingarten(
            slow=:missing,
            fast=:first,
            n_fast=1,
        ),
    )
    @test_throws ArgumentError md_driver(
        U,
        first_action;
        steps=1,
        integrator=SextonWeingarten(
            slow=:second,
            fast=:first,
            n_fast=1,
        ),
    )

    for ordering in (QPQ(), PQP())
        U = gauge_configuration(lattice; configuration_arguments...)
        momenta = gaussian_momenta(U; seed=UInt64(0x5678))
        initial_links = _md_global_links(U)
        initial_momenta = _md_global_momenta(momenta)
        slow_calls = Ref(0)
        fast_calls = Ref(0)
        slow_action = _CountingMDAction(
            _md_test_action(U; coupling=0.75),
            slow_calls,
        )
        fast_action = _CountingMDAction(
            _md_test_action(U; coupling=0.25),
            fast_calls,
        )
        split_actions = MDActionSet(;
            slow=slow_action,
            fast=fast_action,
        )
        integrator = SextonWeingarten(;
            slow=:slow,
            fast=:fast,
            n_fast=3,
            ordering,
        )
        forward = md_driver(
            U,
            split_actions;
            steps=1,
            trajectory_length=0.1,
            integrator,
        )
        md_trajectory!(U, momenta, forward; diagnostics=false)

        if ordering isa QPQ
            @test slow_calls[] == 1
            @test fast_calls[] == 6
        else
            @test slow_calls[] == 2
            @test fast_calls[] == 3
        end

        backward = md_driver(
            U,
            split_actions;
            steps=1,
            trajectory_length=-0.1,
            integrator,
        )
        md_trajectory!(U, momenta, backward; diagnostics=false)
        @test _md_maximum_difference(
            _md_global_links(U),
            initial_links,
        ) < 3e-12
        @test _md_maximum_difference(
            _md_global_momenta(momenta),
            initial_momenta,
        ) < 3e-12
    end
end

@testset "Legacy MD" begin
    for lattice in ((2, 2), (2, 2, 2), (2, 2, 2, 2))
        U = Initialize_Gaugefields(
            2,
            0,
            lattice...;
            condition="cold",
            verbose_level=0,
        )
        p = initialize_TA_Gaugefields(U)
        action = _md_test_action(U)
        driver = md_driver(
            U,
            action;
            steps=2,
            trajectory_length=0.1,
            integrator=PQP(),
        )
        result = md_trajectory!(U, p, driver)

        @test md_step_size(driver) == 0.05
        @test result.delta_hamiltonian == 0
        @test measure_plaquette(U) ≈ 1
        @test p * p == 0
    end

    U = Initialize_Gaugefields(
        2,
        0,
        2,
        2,
        2,
        2;
        condition="cold",
        verbose_level=0,
    )
    p = initialize_TA_Gaugefields(U)
    action = _md_test_action(U)
    @test_throws ArgumentError md_driver(U, action; steps=0)
    @test_throws ArgumentError md_driver(
        U,
        action;
        steps=1,
        trajectory_length=0,
    )
    @test_throws ArgumentError md_driver(
        U,
        action;
        steps=1,
        trajectory_length=Inf,
    )
    invalid_driver = md_driver(
        U,
        action;
        steps=1,
        integrator=_TestInvalidIntegrator(),
    )
    @test_throws ArgumentError md_trajectory!(
        U,
        p,
        invalid_driver;
        diagnostics=false,
    )

    elementary_driver = md_driver(U, action; steps=1)
    @test update_momenta!(p, U, 0.01, elementary_driver) === p
    @test update_gaugefields!(U, p, 0.01, elementary_driver) === U
    @test_throws ArgumentError update_momenta!(p, U, Inf, elementary_driver)
    @test_throws ArgumentError update_gaugefields!(U, p, NaN, elementary_driver)

    for integrator in (PQP(), QPQ())
        U = Initialize_Gaugefields(
            2,
            0,
            2,
            2,
            2,
            2;
            condition="hot",
            randomnumber="Reproducible",
            verbose_level=0,
        )
        p = initialize_TA_Gaugefields(U)
        Random.seed!(123)
        gauss_distribution!(p)
        initial_links = [copy(link.U) for link in U]
        initial_momenta = [copy(momentum.a) for momentum in p]
        action = _md_test_action(U)

        forward = md_driver(
            U,
            action;
            steps=4,
            trajectory_length=0.2,
            integrator,
        )
        @test md_trajectory!(U, p, forward; diagnostics=false) === nothing
        backward = md_driver(
            U,
            action;
            steps=4,
            trajectory_length=-0.2,
            integrator,
        )
        md_trajectory!(U, p, backward; diagnostics=false)

        final_links = [link.U for link in U]
        final_momenta = [momentum.a for momentum in p]
        @test _md_maximum_difference(final_links, initial_links) < 2e-12
        @test _md_maximum_difference(final_momenta, initial_momenta) < 2e-12
    end
end

@testset "LatticeMatrices MD" begin
    for lattice in ((2, 2), (2, 2, 2), (2, 2, 2, 2))
        process_grid = ntuple(_ -> 1, length(lattice))
        U = gauge_configuration(
            lattice;
            colors=2,
            start=:cold,
            process_grid,
            verbose=0,
        )
        p = gauge_momenta(U)
        action = _md_test_action(U)
        driver = md_driver(
            U,
            action;
            steps=2,
            trajectory_length=0.1,
            integrator=_test_custom_qpq!,
        )
        result = md_trajectory!(U, p, driver)

        @test result.delta_hamiltonian == 0
        @test measure_plaquette(U) ≈ 1
        @test p * p == 0
    end

    U32 = gauge_configuration(
        (2, 2, 2, 2);
        colors=2,
        start=:hot,
        seed=UInt64(0x1234),
        process_grid=(1, 1, 1, 1),
        eltype=ComplexF32,
        verbose=0,
    )
    p32 = gaussian_momenta(U32; seed=UInt64(0x5678))
    driver32 = md_driver(
        U32,
        _md_test_action(U32);
        steps=4,
        trajectory_length=0.2,
    )
    @test driver32.trajectory_length isa Float32
    @test md_step_size(driver32) isa Float32
    @test isfinite(md_trajectory!(U32, p32, driver32).delta_hamiltonian)

    lattice = (2, 2, 2, 2)
    for colors in (2, 3, 4)
        configuration_arguments = (
            colors,
            start=:hot,
            seed=UInt64(0x1234),
            process_grid=(1, 1, 1, 1),
            verbose=0,
        )
        U_builtin = gauge_configuration(lattice; configuration_arguments...)
        U_custom = gauge_configuration(lattice; configuration_arguments...)
        p_builtin = gaussian_momenta(U_builtin; seed=UInt64(0x5678))
        p_custom = gaussian_momenta(U_custom; seed=UInt64(0x5678))
        initial_links = _md_global_links(U_builtin)
        initial_momenta = _md_global_momenta(p_builtin)
        builtin_action = _md_test_action(U_builtin)
        custom_action = _md_test_action(U_custom)

        builtin = md_driver(
            U_builtin,
            builtin_action;
            steps=3,
            trajectory_length=0.1,
            integrator=QPQ(),
        )
        custom = md_driver(
            U_custom,
            custom_action;
            steps=3,
            trajectory_length=0.1,
            integrator=_test_custom_qpq!,
        )
        builtin_result = md_trajectory!(U_builtin, p_builtin, builtin)
        custom_result = md_trajectory!(U_custom, p_custom, custom)

        @test isfinite(builtin_result.delta_hamiltonian)
        @test builtin_result.initial_hamiltonian ≈
              custom_result.initial_hamiltonian
        @test builtin_result.final_hamiltonian ≈ custom_result.final_hamiltonian
        @test builtin_result.delta_hamiltonian ≈ custom_result.delta_hamiltonian
        @test _md_maximum_difference(
            _md_global_links(U_builtin),
            _md_global_links(U_custom),
        ) < 2e-12
        @test _md_maximum_difference(
            _md_global_momenta(p_builtin),
            _md_global_momenta(p_custom),
        ) < 2e-12

        backward = md_driver(
            U_builtin,
            builtin_action;
            steps=3,
            trajectory_length=-0.1,
            integrator=QPQ(),
        )
        md_trajectory!(U_builtin, p_builtin, backward; diagnostics=false)
        @test _md_maximum_difference(
            _md_global_links(U_builtin),
            initial_links,
        ) < 2e-12
        @test _md_maximum_difference(
            _md_global_momenta(p_builtin),
            initial_momenta,
        ) < 2e-12
    end
end
