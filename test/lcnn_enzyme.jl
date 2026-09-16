using Enzyme
using LinearAlgebra
using Random

const EnzymeLCNN = Gaugefields.LCNN

function _lcnn_parameter_objective(parameters, action, links, workspace)
    return action(links, parameters, workspace)
end

function _lcnn_central_difference!(evaluate, values, index; step=2e-6)
    original = values[index]
    try
        values[index] = original + step
        plus = evaluate()
        values[index] = original - step
        minus = evaluate()
        return (plus - minus) / (2step)
    finally
        values[index] = original
    end
end

function _lcnn_sparse_link_direction(links, lattice)
    direction = [similar(first(links)) for _ in eachindex(links)]
    clear_U!.(direction)
    colors = first(links).NC
    for mu in eachindex(direction)
        site = ntuple(axis -> mod1(mu + axis, lattice[axis]), length(lattice))
        row = mod1(mu, colors)
        column = mod1(mu + 1, colors)
        direction[mu][row, column, site...] =
            (0.13 + 0.07mu) + (0.11 - 0.03mu)im
    end
    set_wing_U!.(direction)
    return direction
end

function _lcnn_perturbed_links(links, direction, coefficient)
    perturbed = [similar(first(links)) for _ in eachindex(links)]
    for mu in eachindex(links)
        substitute_U!(perturbed[mu], links[mu])
        add_U!(perturbed[mu], coefficient, direction[mu])
        set_wing_U!(perturbed[mu])
    end
    return perturbed
end

function _lcnn_link_central_difference(
    action,
    links,
    parameters,
    workspace,
    direction;
    step=2e-6,
)
    plus = _lcnn_perturbed_links(links, direction, step)
    minus = _lcnn_perturbed_links(links, direction, -step)
    return (
        action(plus, parameters, workspace) -
        action(minus, parameters, workspace)
    ) / (2step)
end

function _lcnn_gradient_contraction(gradient, direction)
    return sum(
        real(dot(getproperty(gradient[mu], :U), getproperty(direction[mu], :U)))
        for mu in eachindex(gradient)
    )
end

function _lcnn_link_output_contraction(
    model, links, parameters, workspace, output_cotangent,
)
    output = EnzymeLCNN.forward_links!(model, links, parameters, workspace)
    return sum(
        real(dot(output_cotangent[mu].U, output[mu].U))
        for mu in eachindex(output)
    )
end

function _lcnn_split_link_forward!(
    links,
    feature_parameters,
    weights,
    feature_model,
    output,
    feature_workspace,
    combinations,
    generators,
    exponentials,
    exponential_temps,
)
    channels = EnzymeLCNN._forward_feature_channels!(
        feature_model, links, feature_parameters, feature_workspace,
    )
    EnzymeLCNN._lexp_only!(
        output,
        links,
        channels,
        weights,
        combinations,
        generators,
        exponentials,
        exponential_temps,
    )
    return nothing
end

@testset "LCNN Enzyme parameter gradients" begin
    @test Base.get_extension(Gaugefields, :GaugefieldsEnzymeExt) !== nothing

    links = gauge_configuration(
        (2, 2);
        colors=2,
        halo=1,
        start=:hot,
        seed=0x454e5a4c,
        process_grid=(1, 1),
        verbose=0,
    )
    # Depth and widths are ordinary constructor data. The resulting concrete
    # Tuple is type stable, while callers remain free to choose another stack.
    feature_model = EnzymeLCNN.LCNNFeatureModel(
        links, 2, 1;
        shifts=:both,
        identity=true,
        adjoints=true,
    )
    action = EnzymeLCNN.LCNNAction(feature_model; reduction=:mean)
    parameters = EnzymeLCNN.initial_parameters(
        MersenneTwister(0x454e5a), action, Float64,
    )
    gradient = Enzyme.make_zero(parameters)
    workspace = EnzymeLCNN.ModelWorkspace(action, links)
    dworkspace = Enzyme.make_zero(workspace)

    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(_lcnn_parameter_objective),
        Enzyme.Active,
        Enzyme.Duplicated(parameters, gradient),
        Enzyme.Const(action),
        Enzyme.Const(links),
        Enzyme.Duplicated(workspace, dworkspace),
    )

    evaluate() = action(links, parameters, workspace)
    checks = (
        (parameters.layers[1].weight, gradient.layers[1].weight,
         CartesianIndex(2, 2, 8)),
        (parameters.layers[2].weight, gradient.layers[2].weight,
         CartesianIndex(1, 4, 12)),
        (parameters.readout.weight, gradient.readout.weight,
         CartesianIndex(1)),
        (parameters.readout.bias, gradient.readout.bias,
         CartesianIndex(1)),
    )
    for (values, derivatives, index) in checks
        finite_difference = _lcnn_central_difference!(evaluate, values, index)
        @test derivatives[index] ≈ finite_difference rtol=3e-5 atol=3e-6
    end

    public_parameter_gradient = EnzymeLCNN.parameter_gradient(
        action, links, parameters; workspace,
    )
    for (expected, actual) in (
        (gradient.layers[1].weight,
         public_parameter_gradient.layers[1].weight),
        (gradient.layers[2].weight,
         public_parameter_gradient.layers[2].weight),
        (gradient.readout.weight,
         public_parameter_gradient.readout.weight),
        (gradient.readout.bias,
         public_parameter_gradient.readout.bias),
    )
        @test actual ≈ expected rtol=2e-12 atol=2e-12
    end

    @testset "LExp link and parameter pullback" begin
        link_model = EnzymeLCNN.LCNNLinkModel(action.feature_model)
        link_parameters = (
            layers=parameters.layers,
            lexp=(weight=reshape(Float64[0.025, 0.01], 2, 1),),
        )
        specification = lcnn_smearing(link_model, link_parameters)
        recorded = smear(
            links, specification; record=true, calcdSdU=true,
        )
        output_cotangent = gauge_configuration(
            (2, 2);
            colors=2,
            halo=1,
            start=:hot,
            seed=0x4c455850,
            process_grid=(1, 1),
            verbose=0,
        )

        input_cotangent = recorded.derivative(output_cotangent)
        full_pullback = EnzymeLCNN.link_model_pullback(
            output_cotangent, links, recorded.history,
        )
        for direction_index in eachindex(input_cotangent)
            @test input_cotangent[direction_index].U.A ≈
                full_pullback.links[direction_index].U.A rtol=2e-12 atol=2e-12
        end

        # Exercise the registered Enzyme rules independently of the explicit
        # public VJP composition. Splitting the immutable workspace into its
        # mutable leaves avoids a mixed-activity aggregate at the AD boundary.
        enzyme_link_cotangent = [
            similar(first(links)) for _ in eachindex(links)
        ]
        clear_U!.(enzyme_link_cotangent)
        enzyme_parameter_cotangent = Enzyme.make_zero(link_parameters)
        shadow_link_workspace = Enzyme.make_zero(recorded.history.workspace)
        Enzyme.remake_zero!(shadow_link_workspace)
        for direction_index in eachindex(output_cotangent)
            add_U!(
                shadow_link_workspace.output[direction_index],
                output_cotangent[direction_index],
            )
            set_wing_U!(shadow_link_workspace.output[direction_index])
        end
        feature_parameters = (layers=link_parameters.layers,)
        feature_parameter_cotangent = (
            layers=enzyme_parameter_cotangent.layers,
        )
        primal_link_workspace = recorded.history.workspace
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(_lcnn_split_link_forward!),
            Enzyme.Const,
            Enzyme.Duplicated(links, enzyme_link_cotangent),
            Enzyme.Duplicated(
                feature_parameters, feature_parameter_cotangent,
            ),
            Enzyme.Duplicated(
                link_parameters.lexp.weight,
                enzyme_parameter_cotangent.lexp.weight,
            ),
            Enzyme.Const(link_model.feature_model),
            Enzyme.Duplicated(
                primal_link_workspace.output,
                shadow_link_workspace.output,
            ),
            Enzyme.Duplicated(
                primal_link_workspace.feature,
                shadow_link_workspace.feature,
            ),
            Enzyme.Duplicated(
                primal_link_workspace.combinations,
                shadow_link_workspace.combinations,
            ),
            Enzyme.Duplicated(
                primal_link_workspace.generators,
                shadow_link_workspace.generators,
            ),
            Enzyme.Duplicated(
                primal_link_workspace.exponentials,
                shadow_link_workspace.exponentials,
            ),
            Enzyme.Duplicated(
                primal_link_workspace.exponential_temps,
                shadow_link_workspace.exponential_temps,
            ),
        )
        set_wing_U!.(enzyme_link_cotangent)
        for direction_index in eachindex(input_cotangent)
            @test enzyme_link_cotangent[direction_index].U.A ≈
                full_pullback.links[direction_index].U.A rtol=2e-12 atol=2e-12
        end
        @test enzyme_parameter_cotangent.layers[1].weight ≈
            full_pullback.parameters.layers[1].weight rtol=2e-12 atol=2e-12
        @test enzyme_parameter_cotangent.layers[2].weight ≈
            full_pullback.parameters.layers[2].weight rtol=2e-12 atol=2e-12
        @test enzyme_parameter_cotangent.lexp.weight ≈
            full_pullback.parameters.lexp.weight rtol=2e-12 atol=2e-12

        direction = _lcnn_sparse_link_direction(links, (2, 2))
        epsilon = 2e-6
        plus = _lcnn_perturbed_links(links, direction, epsilon)
        minus = _lcnn_perturbed_links(links, direction, -epsilon)
        link_workspace = recorded.history.workspace
        finite_difference = (
            _lcnn_link_output_contraction(
                link_model, plus, link_parameters, link_workspace,
                output_cotangent,
            ) -
            _lcnn_link_output_contraction(
                link_model, minus, link_parameters, link_workspace,
                output_cotangent,
            )
        ) / (2epsilon)
        analytic = _lcnn_gradient_contraction(input_cotangent, direction)
        @test analytic ≈ finite_difference rtol=2e-4 atol=2e-5

        evaluate_link() = _lcnn_link_output_contraction(
            link_model,
            links,
            link_parameters,
            link_workspace,
            output_cotangent,
        )
        for (values, derivatives, index) in (
            (
                link_parameters.layers[1].weight,
                full_pullback.parameters.layers[1].weight,
                CartesianIndex(1, 2, 3),
            ),
            (
                link_parameters.lexp.weight,
                full_pullback.parameters.lexp.weight,
                CartesianIndex(2, 1),
            ),
        )
            finite_parameter = _lcnn_central_difference!(
                evaluate_link, values, index,
            )
            @test derivatives[index] ≈ finite_parameter rtol=2e-4 atol=2e-5
        end
    end


    @testset "dS/dU through a flexible two-layer LCNN" begin
        direction = _lcnn_sparse_link_direction(links, (2, 2))
        link_gradient = EnzymeLCNN.dSdu(
            action, links, parameters; workspace,
        )
        enzyme_directional = _lcnn_gradient_contraction(
            link_gradient, direction,
        )
        finite_difference = _lcnn_link_central_difference(
            action, links, parameters, workspace, direction,
        )
        @test enzyme_directional ≈ finite_difference rtol=8e-5 atol=8e-6

        reused_gradient = [similar(first(links)) for _ in eachindex(links)]
        reused_shadow_workspace = Enzyme.make_zero(workspace)
        @test EnzymeLCNN.dSdu!(
            reused_gradient,
            action,
            links,
            parameters,
            workspace,
            reused_shadow_workspace,
        ) === reused_gradient
        @test _lcnn_gradient_contraction(reused_gradient, direction) ≈
            enzyme_directional rtol=2e-12 atol=2e-12
    end

    @testset "convention-specific multi-step transport pullbacks" begin
        direction = _lcnn_sparse_link_direction(links, (2, 2))
        for convention in (:favoni_arxiv, :favoni_prl)
            layer = EnzymeLCNN.LCB(
                1 => 1;
                convention,
                kernel_size=3,
                shifts=:positive,
            )
            model = EnzymeLCNN.LCNNFeatureModel(links, (layer,))
            convention_action = EnzymeLCNN.LCNNAction(
                model; component=:both, reduction=:mean,
            )
            convention_parameters = EnzymeLCNN.initial_parameters(
                MersenneTwister(
                    convention === :favoni_arxiv ? 0xa11ce : 0xb11ce,
                ),
                convention_action,
                Float64,
            )
            convention_workspace = EnzymeLCNN.ModelWorkspace(
                convention_action, links,
            )
            gradient = EnzymeLCNN.dSdu(
                convention_action,
                links,
                convention_parameters;
                workspace=convention_workspace,
            )
            enzyme_directional = _lcnn_gradient_contraction(
                gradient, direction,
            )
            finite_difference = _lcnn_link_central_difference(
                convention_action,
                links,
                convention_parameters,
                convention_workspace,
                direction,
            )
            @test enzyme_directional ≈ finite_difference rtol=2e-4 atol=2e-5
        end
    end

    @testset "4D SU(3) specialization" begin
        links4 = gauge_configuration(
            (2, 2, 2, 2);
            colors=3,
            halo=1,
            start=:hot,
            seed=0x4c3444,
            process_grid=(1, 1, 1, 1),
            verbose=0,
        )
        model4 = EnzymeLCNN.LCNNFeatureModel(
            links4, 1;
            shifts=:positive,
            identity=false,
            adjoints=false,
        )
        action4 = EnzymeLCNN.LCNNAction(model4; reduction=:mean)
        parameters4 = EnzymeLCNN.initial_parameters(
            MersenneTwister(4), action4, Float64,
        )
        gradient4 = Enzyme.make_zero(parameters4)
        workspace4 = EnzymeLCNN.ModelWorkspace(action4, links4)
        dworkspace4 = Enzyme.make_zero(workspace4)

        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(_lcnn_parameter_objective),
            Enzyme.Active,
            Enzyme.Duplicated(parameters4, gradient4),
            Enzyme.Const(action4),
            Enzyme.Const(links4),
            Enzyme.Duplicated(workspace4, dworkspace4),
        )

        index = CartesianIndex(1, 2, 25)
        evaluate4() = action4(links4, parameters4, workspace4)
        finite_difference = _lcnn_central_difference!(
            evaluate4, parameters4.layers[1].weight, index,
        )
        @test isapprox(
            gradient4.layers[1].weight[index], finite_difference;
            rtol=3e-5,
            atol=3e-6,
        )

        direction4 = _lcnn_sparse_link_direction(links4, (2, 2, 2, 2))
        link_gradient4 = EnzymeLCNN.link_gradient(
            action4, links4, parameters4; workspace=workspace4,
        )
        enzyme_directional4 = _lcnn_gradient_contraction(
            link_gradient4, direction4,
        )
        finite_difference4 = _lcnn_link_central_difference(
            action4, links4, parameters4, workspace4, direction4,
        )
        @test enzyme_directional4 ≈ finite_difference4 rtol=2e-4 atol=2e-5

        link_model4 = EnzymeLCNN.LCNNLinkModel(model4)
        link_parameters4 = (
            layers=parameters4.layers,
            lexp=(weight=fill(0.004, 4, 1),),
        )
        link_workspace4 = EnzymeLCNN.LinkModelWorkspace(link_model4, links4)
        output_cotangent4 = _lcnn_sparse_link_direction(
            links4, (2, 2, 2, 2),
        )
        pullback4 = EnzymeLCNN.link_model_pullback(
            output_cotangent4,
            link_model4,
            links4,
            link_parameters4;
            workspace=link_workspace4,
        )
        epsilon4 = 2e-6
        plus4 = _lcnn_perturbed_links(links4, direction4, epsilon4)
        minus4 = _lcnn_perturbed_links(links4, direction4, -epsilon4)
        finite_link_model4 = (
            _lcnn_link_output_contraction(
                link_model4, plus4, link_parameters4, link_workspace4,
                output_cotangent4,
            ) -
            _lcnn_link_output_contraction(
                link_model4, minus4, link_parameters4, link_workspace4,
                output_cotangent4,
            )
        ) / (2epsilon4)
        analytic_link_model4 = _lcnn_gradient_contraction(
            pullback4.links, direction4,
        )
        @test analytic_link_model4 ≈ finite_link_model4 rtol=4e-4 atol=4e-5
    end
end
