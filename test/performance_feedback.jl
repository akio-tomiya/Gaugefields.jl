using Gaugefields
import JACC
using LinearAlgebra
using Test

JACC.@init_backend

struct _FeedbackCountingAction{A}
    action::A
    force_calls::Base.RefValue{Int}
end

Gaugefields.md_action_workspace(action::_FeedbackCountingAction, U) =
    md_action_workspace(action.action, U)
Gaugefields.md_potential(action::_FeedbackCountingAction, U, workspace) =
    md_potential(action.action, U, workspace)
function Gaugefields.md_force!(force, action::_FeedbackCountingAction, U, workspace)
    action.force_calls[] += 1
    return md_force!(force, action.action, U, workspace)
end

function _feedback_gauge_action(U)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, 1.0, plaquettes)
    return action
end

function _feedback_old_pqp!(U, P, step_size, driver)
    update_momenta!(P, U, step_size / 2, driver)
    update_gaugefields!(U, P, step_size, driver)
    update_momenta!(P, U, step_size / 2, driver)
    return nothing
end

@testset "HMC performance feedback" begin
    U = gauge_configuration(
        (2, 2, 2, 2);
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(0x70657266),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )

    smearing = CovNeuralnet(U; numtemps=20)
    @test length(smearing._temp_U) == 20
    push!(smearing, STOUT_Layer(["plaquette"], [0.1], U))
    smeared, history, _ = calc_smearedU(U, smearing)
    cotangent = map(similar, smeared)
    substitute_U!(cotangent, smeared)
    full = map(similar, U)
    links_only = map(similar, U)
    back_prop!(full, cotangent, smearing, history, U)
    back_prop!(
        links_only,
        cotangent,
        smearing,
        history,
        U;
        calculate_parameter_derivatives=false,
    )
    @test maximum(
        maximum(abs, full[direction].U.A .- links_only[direction].U.A)
        for direction in eachindex(U)
    ) < 5e-12

    product_temporary = similar(U[1])
    for (left, right) in (
        (U[1], U[2]),
        (U[1]', U[2]),
        (U[1], U[2]'),
        (U[1]', U[2]'),
    )
        separate_projection = initialize_TA_Gaugefields(U[1])
        fused_projection = initialize_TA_Gaugefields(U[1])
        clear_U!(separate_projection)
        clear_U!(fused_projection)
        mul!(product_temporary, left, right)
        Traceless_antihermitian_add!(
            separate_projection, -0.375, product_temporary)
        Traceless_antihermitian_product_add!(
            fused_projection, -0.375, left, right)
        @test maximum(abs,
            separate_projection.a.A .- fused_projection.a.A) < 5e-12
    end
    for colors in (2, 4)
        group_U = gauge_configuration(
            (2, 2, 2, 2);
            colors,
            halo=1,
            start=:hot,
            seed=UInt64(0x53554e00 + colors),
            process_grid=(1, 1, 1, 1),
            verbose=0,
        )
        group_temporary = similar(group_U[1])
        separate_projection = initialize_TA_Gaugefields(group_U[1])
        fused_projection = initialize_TA_Gaugefields(group_U[1])
        clear_U!(separate_projection)
        clear_U!(fused_projection)
        mul!(group_temporary, group_U[1], group_U[2]')
        Traceless_antihermitian_add!(
            separate_projection, 0.625, group_temporary)
        Traceless_antihermitian_product_add!(
            fused_projection, 0.625, group_U[1], group_U[2]')
        @test maximum(abs,
            separate_projection.a.A .- fused_projection.a.A) < 5e-12
    end

    optimized_U = copy_configuration(U)
    reference_U = copy_configuration(U)
    optimized_P = gaussian_momenta(U; seed=UInt64(0x6d6f6d65))
    reference_P = gaussian_momenta(U; seed=UInt64(0x6d6f6d65))
    optimized_calls = Ref(0)
    reference_calls = Ref(0)
    optimized_action = _FeedbackCountingAction(
        _feedback_gauge_action(optimized_U), optimized_calls)
    reference_action = _FeedbackCountingAction(
        _feedback_gauge_action(reference_U), reference_calls)
    steps = 4
    optimized_driver = md_driver(
        optimized_U,
        optimized_action;
        steps,
        trajectory_length=0.05,
        integrator=PQP(),
    )
    reference_driver = md_driver(
        reference_U,
        reference_action;
        steps,
        trajectory_length=0.05,
        integrator=_feedback_old_pqp!,
    )
    md_trajectory!(optimized_U, optimized_P, optimized_driver; diagnostics=false)
    md_trajectory!(reference_U, reference_P, reference_driver; diagnostics=false)
    @test optimized_calls[] == steps + 1
    @test reference_calls[] == 2steps
    @test maximum(
        maximum(abs, optimized_U[d].U.A .- reference_U[d].U.A)
        for d in eachindex(U)
    ) < 5e-12
    @test maximum(
        maximum(abs, optimized_P[d].a.A .- reference_P[d].a.A)
        for d in eachindex(U)
    ) < 5e-12
end
