using LinearAlgebra
using Random

function _or_wilson_action(U; beta=6.0)
    gauge_action = GaugeAction(U)
    plaquette_loops = make_loops_fromname("plaquette", Dim=4)
    append!(plaquette_loops, adjoint(plaquette_loops))
    push!(gauge_action, beta / 2, plaquette_loops)
    return gauge_action
end

function _or_action_value(gauge_action, U)
    return -real(evaluate_GaugeAction(gauge_action, U)) / U[1].NC
end

function _or_link_distance(U, V)
    maximum_distance = 0.0
    for μ in eachindex(U), t in 1:U[μ].NT, z in 1:U[μ].NZ,
        y in 1:U[μ].NY, x in 1:U[μ].NX
        difference =
            Matrix(U[μ][:, :, x, y, z, t]) -
            Matrix(V[μ][:, :, x, y, z, t])
        maximum_distance = max(maximum_distance, norm(difference))
    end
    return maximum_distance
end

function _or_group_errors(U)
    identity_matrix = Matrix{ComplexF64}(I, U[1].NC, U[1].NC)
    maximum_unitarity_error = 0.0
    maximum_determinant_error = 0.0
    for μ in eachindex(U), t in 1:U[μ].NT, z in 1:U[μ].NZ,
        y in 1:U[μ].NY, x in 1:U[μ].NX
        link = Matrix(U[μ][:, :, x, y, z, t])
        maximum_unitarity_error = max(
            maximum_unitarity_error,
            norm(link' * link - identity_matrix),
        )
        maximum_determinant_error = max(
            maximum_determinant_error,
            abs(det(link) - 1),
        )
    end
    return maximum_unitarity_error, maximum_determinant_error
end

@testset "Heatbath_update overrelaxation is microcanonical" begin
    U = Initialize_Gaugefields(
        3,
        0,
        2,
        2,
        2,
        2;
        condition="hot",
        randomnumber="Reproducible",
        verbose_level=0,
    )
    U_overrelaxation = similar(U)
    U_heatbath = similar(U)
    substitute_U!(U_overrelaxation, U)
    substitute_U!(U_heatbath, U)

    overrelaxation_action = _or_wilson_action(U_overrelaxation)
    heatbath_action = _or_wilson_action(U_heatbath)
    action_before =
        _or_action_value(overrelaxation_action, U_overrelaxation)

    Random.seed!(314159)
    overrelaxation!(
        U_overrelaxation,
        Heatbath_update(U_overrelaxation, overrelaxation_action),
    )
    Random.seed!(314159)
    heatbath!(U_heatbath, Heatbath_update(U_heatbath, heatbath_action))

    action_after =
        _or_action_value(overrelaxation_action, U_overrelaxation)
    relative_action_drift =
        abs(action_after - action_before) / max(1, abs(action_before))
    maximum_unitarity_error, maximum_determinant_error =
        _or_group_errors(U_overrelaxation)

    @test relative_action_drift < 1.0e-10
    @test maximum_unitarity_error < 1.0e-12
    @test maximum_determinant_error < 1.0e-12
    @test _or_link_distance(U_overrelaxation, U) > 1.0e-8
    @test _or_link_distance(U_overrelaxation, U_heatbath) > 1.0e-8
end
