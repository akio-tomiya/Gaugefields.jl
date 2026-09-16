using LinearAlgebra
using NPZ
using Random
import Wilsonloop: Wilsonline

const ReferenceLCNN = Gaugefields.LCNN

function _lcnn_site_matrix(field, site)
    colors = field.NC
    return [field[row, column, site...] for row in 1:colors, column in 1:colors]
end

function _lcnn_set_site_matrix!(field, matrix, site)
    for column in axes(matrix, 2), row in axes(matrix, 1)
        field[row, column, site...] = matrix[row, column]
    end
    return field
end

function _lcnn_gauge_matrix(site, colors)
    angles = [
        0.037 * color * sum(
            (-1)^axis * (axis + color) * site[axis] for axis in eachindex(site)
        ) for color in 1:(colors - 1)
    ]
    phases = cis.(vcat(angles, -sum(angles)))
    return Matrix(Diagonal(phases))
end

function _lcnn_transform_links(U, lattice)
    transformed = similar(U)
    colors = first(U).NC
    for site_index in CartesianIndices(lattice)
        site = Tuple(site_index)
        omega = _lcnn_gauge_matrix(site, colors)
        for direction in eachindex(U)
            forward = ntuple(length(lattice)) do axis
                axis == direction ? mod1(site[axis] + 1, lattice[axis]) : site[axis]
            end
            link = _lcnn_site_matrix(U[direction], site)
            transformed_link = omega * link * _lcnn_gauge_matrix(forward, colors)'
            _lcnn_set_site_matrix!(transformed[direction], transformed_link, site)
        end
    end
    set_wing_U!.(transformed)
    return transformed
end

function _lcnn_transform_feature(feature, lattice)
    transformed = similar(feature)
    colors = feature.NC
    for site_index in CartesianIndices(lattice)
        site = Tuple(site_index)
        omega = _lcnn_gauge_matrix(site, colors)
        value = _lcnn_site_matrix(feature, site)
        _lcnn_set_site_matrix!(transformed, omega * value * omega', site)
    end
    set_wing_U!(transformed)
    return transformed
end

function _lcnn_test_field_isapprox(actual, expected, lattice; atol=2e-11, rtol=2e-11)
    for site_index in CartesianIndices(lattice)
        site = Tuple(site_index)
        isapprox(
            _lcnn_site_matrix(actual, site),
            _lcnn_site_matrix(expected, site);
            atol,
            rtol,
        ) || return false
    end
    return true
end

function _lcnn_dense_field(field, lattice)
    output = Array{Matrix{ComplexF64}}(undef, lattice...)
    for site_index in CartesianIndices(lattice)
        output[site_index] = _lcnn_site_matrix(field, Tuple(site_index))
    end
    return output
end

function _lcnn_dense_channels(fields, lattice)
    colors = first(fields).NC
    output = Array{ComplexF64}(undef, length(fields), colors, colors, lattice...)
    for channel in eachindex(fields), site_index in CartesianIndices(lattice)
        site = Tuple(site_index)
        for column in 1:colors, row in 1:colors
            output[channel, row, column, site...] =
                fields[channel][row, column, site...]
        end
    end
    return output
end

# Independent dense translation of `transport` in the upstream Python layer.
# It deliberately does not call any LCNN transport or descriptor helper.
function _favoni_dense_transport(links, values, direction, step, lattice)
    current = values
    orientation = sign(step)
    for _ in 1:abs(step)
        transported = similar(current)
        for site_index in CartesianIndices(lattice)
            site = Tuple(site_index)
            source = ntuple(length(lattice)) do axis
                axis == direction ?
                    mod1(site[axis] + orientation, lattice[axis]) : site[axis]
            end
            if orientation > 0
                link = links[direction][site_index]
                transported[site_index] =
                    link * current[source...] * link'
            else
                link = links[direction][source...]
                transported[site_index] =
                    link' * current[source...] * link
            end
        end
        current = transported
    end
    return current
end

function _favoni_dense_lcb(links, input, weight, convention, lattice)
    raw = if convention === :favoni_arxiv
        [_favoni_dense_transport(links, input, axis, 1, lattice)
         for axis in 1:length(lattice)]
    else
        [input,
         (_favoni_dense_transport(links, input, axis, 1, lattice)
          for axis in 1:length(lattice))...]
    end
    colors = size(first(input), 1)
    identity_field = fill(Matrix{ComplexF64}(I, colors, colors), lattice...)
    local_basis = [input, map(adjoint, input), identity_field]
    transported_basis = [raw..., (map(adjoint, term) for term in raw)...,
                         identity_field]
    outputs = [similar(input) for _ in axes(weight, 1)]
    for output_index in axes(weight, 1), site_index in CartesianIndices(lattice)
        value = zeros(ComplexF64, colors, colors)
        for local_index in axes(weight, 2), transported_index in axes(weight, 3)
            value .+= weight[output_index, local_index, transported_index] .*
                (local_basis[local_index][site_index] *
                 transported_basis[transported_index][site_index])
        end
        outputs[output_index][site_index] = value
    end
    return outputs
end

@testset "LCNN radius-one reference transport" begin
    lattice = (4, 6)
    U = gauge_configuration(
        lattice;
        colors=2,
        halo=1,
        start=:hot,
        seed=0x4c434e4e,
        process_grid=(1, 1),
        verbose=0,
    )
    transformed_U = _lcnn_transform_links(U, lattice)

    workspace = ReferenceLCNN.ReferenceWorkspace(U)
    transformed_workspace = ReferenceLCNN.ReferenceWorkspace(transformed_U)
    features = ReferenceLCNN.plaquette_features(
        ReferenceLCNN.Plaq(), U, workspace,
    )
    transformed_features = ReferenceLCNN.plaquette_features(
        ReferenceLCNN.Plaq(), transformed_U, transformed_workspace,
    )

    @test length(features.channels) == 1
    expected_plaquette = _lcnn_transform_feature(features.channels[1], lattice)
    @test _lcnn_test_field_isapprox(
        transformed_features.channels[1], expected_plaquette, lattice,
    )

    for direction in 1:2, step in (-2, -1, 0, 1, 2)
        transported = ReferenceLCNN.parallel_transport(
            features, 1, direction, step, workspace,
        )
        transformed_transported = ReferenceLCNN.parallel_transport(
            transformed_features, 1, direction, step, transformed_workspace,
        )
        expected = _lcnn_transform_feature(transported, lattice)
        @test _lcnn_test_field_isapprox(
            transformed_transported, expected, lattice,
        )
    end

    lcb_layer = ReferenceLCNN.LCB(1 => 2; radius=1, shifts=:both)
    lcb_shape = ReferenceLCNN.parameter_shape(lcb_layer, 2)
    lcb_parameters = (
        weight=0.05 .* randn(MersenneTwister(0x1cbb), Float64, lcb_shape),
    )
    lcb_features = ReferenceLCNN.lcb(
        lcb_layer, features, lcb_parameters, workspace,
    )
    transformed_lcb_features = ReferenceLCNN.lcb(
        lcb_layer,
        transformed_features,
        lcb_parameters,
        transformed_workspace,
    )
    for channel in eachindex(lcb_features.channels)
        expected = _lcnn_transform_feature(lcb_features.channels[channel], lattice)
        @test _lcnn_test_field_isapprox(
            transformed_lcb_features.channels[channel], expected, lattice;
            atol=8e-11,
            rtol=8e-11,
        )
    end

    readout = ReferenceLCNN.TraceReadout(component=:both, reduction=:mean)
    @test readout(transformed_lcb_features) ≈ readout(lcb_features) atol=2e-10 rtol=2e-10

    rectangle_layer = ReferenceLCNN.LCB(1 => 1; radius=1, shifts=:both)
    rectangle_weights = zeros(
        Float64,
        ReferenceLCNN.parameter_shape(rectangle_layer, 2),
    )
    transported_index = findfirst(
        index -> ReferenceLCNN._transported_descriptor(
            rectangle_layer, Val(2), index,
        ) == (:ordinary, 1, (2, 1)),
        axes(rectangle_weights, 3),
    )
    rectangle_weights[1, 1, transported_index] = 1
    rectangle_features = ReferenceLCNN.lcb(
        rectangle_layer,
        features,
        (weight=rectangle_weights,),
        workspace,
    )

    expected_rectangle = similar(U[1])
    rectangle_path = Wilsonline(
        [(1, 1), (2, 2), (1, -1), (2, -2)];
        Dim=2,
    )
    Gaugefields.evaluate_gaugelinks!(
        expected_rectangle,
        rectangle_path,
        U,
        workspace.path_temps,
    )
    @test _lcnn_test_field_isapprox(
        rectangle_features.channels[1], expected_rectangle, lattice;
        atol=8e-11,
        rtol=8e-11,
    )

    feature_model = ReferenceLCNN.LCNNFeatureModel(U, 2; shifts=:both)
    action = ReferenceLCNN.LCNNAction(
        feature_model; component=:both, reduction=:mean,
    )
    parameters = ReferenceLCNN.initial_parameters(
        MersenneTwister(0x1caa), action, Float64,
    )
    @test ReferenceLCNN.parameter_shapes(action) == (
        layers=((weight=(2, 3, 11),),),
        readout=(weight=(4,), bias=(1,)),
    )
    @test parameters.layers isa Tuple
    @test eltype(parameters.layers[1].weight) === Float64
    @test eltype(parameters.readout.bias) === Float64
    @test length(parameters.readout.bias) == 1
    parameters32 = ReferenceLCNN.initial_parameters(
        MersenneTwister(0x1ca32), action, Float32,
    )
    @test eltype(parameters32.layers[1].weight) === Float32
    @test eltype(parameters32.readout.weight) === Float32
    @test action(transformed_U, parameters, transformed_workspace) ≈
        action(U, parameters, workspace) atol=2e-10 rtol=2e-10

    flexible_model = ReferenceLCNN.LCNNFeatureModel(
        2,
        (
            ReferenceLCNN.LCB(1 => 3; shifts=:positive),
            ReferenceLCNN.LCB(
                3 => 2;
                shifts=:both,
                identity=false,
                adjoints=false,
            ),
        ),
    )
    @test length(flexible_model.layers) == 2
    @test ReferenceLCNN.parameter_shapes(flexible_model) == (
        layers=(
            (weight=(3, 3, 7),),
            (weight=(2, 3, 15),),
        ),
    )
    @test_throws DimensionMismatch ReferenceLCNN.LCNNFeatureModel(
        2, (ReferenceLCNN.LCB(2 => 1),),
    )

    @testset "Favoni et al. 1x2 compatibility conventions" begin
        paper_action = ReferenceLCNN.favoni2022_wilson_1x2_small(
            U; convention=:favoni_arxiv,
        )
        checkpoint_action = ReferenceLCNN.favoni2022_wilson_1x2_small(
            U; convention=:favoni_prl,
        )
        paper_layer = only(paper_action.feature_model.layers)
        checkpoint_layer = only(checkpoint_action.feature_model.layers)

        @test ReferenceLCNN.displacements(paper_layer, Val(2)) ==
            ((1, 1), (2, 1))
        @test ReferenceLCNN.displacements(checkpoint_layer, Val(2)) ==
            ((0, 0), (1, 1), (2, 1))
        @test ReferenceLCNN.parameter_shapes(paper_action) == (
            layers=((weight=(2, 3, 5),),),
            readout=(weight=(4,), bias=(1,)),
        )
        @test ReferenceLCNN.parameter_shapes(checkpoint_action) == (
            layers=((weight=(2, 3, 7),),),
            readout=(weight=(4,), bias=(1,)),
        )
        @test ReferenceLCNN.parameter_count(paper_action) == 35
        @test ReferenceLCNN.parameter_count(checkpoint_action) == 47
        @test paper_layer.dilation == 0
        @test checkpoint_layer.dilation == 1
        @test ReferenceLCNN.LCB(1 => 2; convention=:favoni_arxiv).shifts ==
            :positive
        @test_throws ArgumentError ReferenceLCNN.LCB(
            1 => 2; convention=:favoni_arxiv, shifts=:both,
        )
        paper_dilated = ReferenceLCNN.LCB(
            1 => 1; convention=:favoni_arxiv, kernel_size=3, dilation=1,
        )
        checkpoint_dilated = ReferenceLCNN.LCB(
            1 => 1; convention=:favoni_prl, kernel_size=3, dilation=1,
            shifts=:positive,
        )
        @test ReferenceLCNN.displacements(paper_dilated, Val(2)) ==
            ((1, 2), (1, 4), (2, 2), (2, 4))
        @test ReferenceLCNN.displacements(checkpoint_dilated, Val(2)) ==
            ((0, 0), (1, 1), (1, 2), (2, 1), (2, 2))

        # Checkpoint basis order: W, W†, I locally; raw transports, their
        # adjoints, I on the transported side.
        @test ReferenceLCNN._local_descriptor(checkpoint_layer, 1) == (:ordinary, 1)
        @test ReferenceLCNN._local_descriptor(checkpoint_layer, 2) == (:adjoint, 1)
        @test ReferenceLCNN._local_descriptor(checkpoint_layer, 3) == (:identity, 0)
        @test ReferenceLCNN._transported_descriptor(
            checkpoint_layer, Val(2), 1,
        ) == (:ordinary, 1, (0, 0))
        @test ReferenceLCNN._transported_descriptor(
            checkpoint_layer, Val(2), 4,
        ) == (:adjoint, 1, (0, 0))
        @test ReferenceLCNN._transported_descriptor(
            checkpoint_layer, Val(2), 7,
        ) == (:identity, 0, (0, 0))

        # Same U, W, and weights through a dense translation of the released
        # Python formulas and through the Gaugefields implementation.
        dense_links = [_lcnn_dense_field(link, lattice) for link in U]
        dense_input = _lcnn_dense_field(features.channels[1], lattice)
        for (convention, upstream_action) in (
            (:favoni_arxiv, paper_action), (:favoni_prl, checkpoint_action),
        )
            shape = ReferenceLCNN.parameter_shapes(upstream_action).layers[1].weight
            weight = reshape(
                collect(range(-0.09, 0.07; length=prod(shape))), shape,
            )
            readout_weight = [0.4, -0.3, 0.2, 0.1]
            parameters = (
                layers=((; weight),),
                readout=(weight=readout_weight, bias=[-0.05]),
            )
            julia_features = ReferenceLCNN.forward_features(
                upstream_action.feature_model,
                U,
                (layers=parameters.layers,),
                workspace,
            )
            dense_outputs = _favoni_dense_lcb(
                dense_links, dense_input, weight, convention, lattice,
            )
            for output_index in eachindex(dense_outputs)
                for site_index in CartesianIndices(lattice)
                    @test _lcnn_site_matrix(
                        julia_features.channels[output_index], Tuple(site_index),
                    ) ≈ dense_outputs[output_index][site_index] atol=8e-11 rtol=8e-11
                end
            end
            dense_invariants = Float64[]
            for output in dense_outputs
                mean_trace = sum(tr, output) / prod(lattice)
                append!(dense_invariants, (real(mean_trace), imag(mean_trace)))
            end
            dense_scalar = dot(readout_weight, dense_invariants) - 0.05
            @test upstream_action(U, parameters, workspace) ≈
                dense_scalar atol=2e-10 rtol=2e-10
        end

        result_action = checkpoint_action
        @test length(result_action.feature_model.layers) == 1
        @test (
            checkpoint_layer.in_channels,
            checkpoint_layer.out_channels,
            checkpoint_layer.radius,
            checkpoint_layer.shifts,
        ) == (1, 2, 1, :positive)

        # Every trainable scalar can be supplied without an ML framework.
        explicit_parameters = (
            layers=((weight=reshape(
                collect(range(-0.08, 0.08; length=42)), 2, 3, 7,
            ),),),
            readout=(weight=[0.4, -0.3, 0.2, 0.1], bias=[-0.05]),
        )
        @test ReferenceLCNN.validate_parameters(
            result_action, explicit_parameters,
        )
        @test isfinite(result_action(U, explicit_parameters, workspace))
        malformed_parameters = merge(
            explicit_parameters,
            (readout=(weight=zeros(3), bias=zeros(1)),),
        )
        @test_throws DimensionMismatch ReferenceLCNN.validate_parameters(
            result_action, malformed_parameters,
        )

        @testset "portable PyTorch checkpoint loader" begin
            @test ReferenceLCNN._checkpoint_convention(:paper) === :favoni_arxiv
            @test ReferenceLCNN._checkpoint_convention(:checkpoint) === :favoni_prl
            @test ReferenceLCNN._checkpoint_convention(:arxiv_v1) === :favoni_arxiv
            @test ReferenceLCNN._checkpoint_convention(:prl_2022) === :favoni_prl
            mktempdir() do directory
                path = joinpath(directory, "lcnn.npz")
                text(value) = collect(codeunits(value))
                npzwrite(path, Dict(
                    "meta.format_version" => Int64(1),
                    "meta.source" => text("openpixi/lge-cnn"),
                    "meta.source_commit" => text("prl_2022"),
                    "meta.convention" => text("favoni_prl"),
                    "meta.dimension" => Int64(2),
                    "meta.colors" => Int64(2),
                    "meta.layer_count" => Int64(1),
                    "meta.conv_channels" => Int64[2],
                    "meta.kernel_size" => Int64[2],
                    "meta.dilation" => Int64[1],
                    "meta.symmetric" => false,
                    "meta.use_unit_elements" => true,
                    "meta.global_average" => false,
                    "meta.complex_component_order" => text("channel_re_im"),
                    "layers.1.weight" => explicit_parameters.layers[1].weight,
                    "readout.weight" => explicit_parameters.readout.weight,
                    "readout.bias" => explicit_parameters.readout.bias,
                ))
                checkpoint = ReferenceLCNN.read_pytorch_checkpoint(path)
                @test checkpoint.metadata.convention === :favoni_prl
                @test checkpoint.metadata.linear_sizes == ()
                loaded_action = ReferenceLCNN.model_from_pytorch_checkpoint(
                    U, checkpoint,
                )
                loaded_parameters = ReferenceLCNN.load_pytorch_parameters(
                    loaded_action,
                    checkpoint;
                    parameter_type=Float64,
                )
                @test loaded_parameters == explicit_parameters
                @test loaded_action(U, loaded_parameters, workspace) ≈
                    result_action(U, explicit_parameters, workspace)
                model2, parameters2, metadata2 =
                    ReferenceLCNN.load_pytorch_model(
                        U, path; parameter_type=Float64,
                    )
                @test metadata2.convention === :favoni_prl
                @test ReferenceLCNN.parameter_count(model2) == 47
                @test parameters2 == explicit_parameters
                @test_throws ArgumentError ReferenceLCNN.load_pytorch_parameters(
                    paper_action, checkpoint,
                )
            end
        end
    end
end

@testset "LCNN frozen official PyTorch outputs" begin
    golden = npzread(joinpath(
        @__DIR__, "data", "lcnn", "favoni2022_pytorch_outputs.npz",
    ))
    lattice = (4, 4)
    for convention in (:favoni_arxiv, :favoni_prl)
        for seed in (0x4c434e4e, 0x554c434e)
            U = gauge_configuration(
                lattice;
                colors=2,
                halo=1,
                start=:hot,
                seed,
                process_grid=(1, 1),
                verbose=0,
            )
            action = ReferenceLCNN.favoni2022_wilson_1x2_small(U; convention)
            shape = ReferenceLCNN.parameter_shapes(action).layers[1].weight
            parameters = (
                layers=((weight=reshape(
                    collect(range(-0.09f0, 0.07f0; length=prod(shape))), shape,
                ),),),
                readout=(
                    weight=Float32[0.4, -0.3, 0.2, 0.1],
                    bias=Float32[-0.05],
                ),
            )
            workspace = ReferenceLCNN.ReferenceWorkspace(U)
            output = ReferenceLCNN.forward_features(
                action.feature_model,
                U,
                (layers=parameters.layers,),
                workspace,
            )
            prefix = "$(convention).$(string(seed; base=16))"
            @test _lcnn_dense_channels(output.channels, lattice) ≈
                golden["$prefix.features"] atol=2e-12 rtol=2e-12
            @test action(U, parameters, workspace) ≈
                only(golden["$prefix.scalar"]) atol=2e-12 rtol=2e-12
        end
    end
end

@testset "LCNN Gaugefields smearing API" begin
    lattice = (4, 4)
    U = gauge_configuration(
        lattice;
        colors=2,
        halo=1,
        start=:hot,
        seed=0x534d4541,
        process_grid=(1, 1),
        verbose=0,
    )
    action = ReferenceLCNN.favoni2022_wilson_1x2_small(
        U; convention=:favoni_prl,
    )
    action_parameters = ReferenceLCNN.initial_parameters(
        MersenneTwister(0x4c434e4e), action, Float64,
    )
    link_model = ReferenceLCNN.LCNNLinkModel(
        action.feature_model,
        ReferenceLCNN.LExp(2),
    )
    parameters = (
        layers=action_parameters.layers,
        lexp=(weight=Float64[0.03 -0.02; 0.01 0.04],),
    )
    specification = lcnn_smearing(link_model, parameters)

    @test specification isa LCNNLinkSmearing
    @test has_smearing_pullback(specification)
    @test ReferenceLCNN.parameter_shapes(link_model) == (
        layers=((weight=(2, 3, 7),),),
        lexp=(weight=(2, 2),),
    )
    @test ReferenceLCNN.parameter_count(link_model) == 46
    @test ReferenceLCNN.validate_parameters(link_model, parameters)
    initialized = ReferenceLCNN.initial_parameters(
        MersenneTwister(0x4c455850), link_model, Float32,
    )
    @test size(initialized.lexp.weight) == (2, 2)
    @test eltype(initialized.lexp.weight) === Float32
    @test ReferenceLCNN.validate_parameters(link_model, initialized)
    @test_throws MethodError lcnn_smearing(action, action_parameters)

    ordinary = smear(U, specification)
    @test ordinary isa Vector
    @test length(ordinary) == length(U)
    for direction in eachindex(ordinary), site_index in CartesianIndices(lattice)
        matrix = _lcnn_site_matrix(ordinary[direction], Tuple(site_index))
        @test matrix' * matrix ≈ Matrix{ComplexF64}(I, 2, 2) atol=2e-10 rtol=2e-10
        @test det(matrix) ≈ 1 atol=2e-10 rtol=2e-10
    end

    recorded = smear(U, specification; record=true)
    @test recorded.configuration isa Vector
    @test recorded.history isa ReferenceLCNN.LCNNLinkSmearingCache
    @test recorded.derivative === nothing
    for direction in eachindex(ordinary)
        @test _lcnn_test_field_isapprox(
            recorded.configuration[direction],
            ordinary[direction],
            lattice,
        )
    end

    reused = smear(U, specification; temps=recorded.history)
    @test reused === recorded.configuration
    differentiable = smear(
        U, specification;
        record=true,
        calcdSdU=true,
        temps=recorded.history,
    )
    @test differentiable.derivative isa ReferenceLCNN.LCNNLinkPullback
    @test differentiable.history isa ReferenceLCNN.LCNNLinkSmearingCache
    @test_throws ArgumentError smear(
        U, specification; temps=ReferenceLCNN.ReferenceWorkspace(U),
    )

    zero_parameters = merge(
        parameters, (lexp=(weight=zeros(Float64, 2, 2),),),
    )
    unchanged = smear(U, lcnn_smearing(link_model, zero_parameters))
    for direction in eachindex(U)
        @test _lcnn_test_field_isapprox(
            unchanged[direction], U[direction],
            lattice,
        )
    end

    transformed_U = _lcnn_transform_links(U, lattice)
    transformed_output = smear(transformed_U, specification)
    expected_output = _lcnn_transform_links(ordinary, lattice)
    for direction in eachindex(ordinary)
        @test _lcnn_test_field_isapprox(
            transformed_output[direction], expected_output[direction], lattice;
            atol=2e-10,
            rtol=2e-10,
        )
    end
end

@testset "LCNN 4D SU(3) reference model" begin
    lattice = (2, 2, 2, 2)
    U = gauge_configuration(
        lattice;
        colors=3,
        halo=1,
        start=:hot,
        seed=0x4c434e4e43,
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    transformed_U = _lcnn_transform_links(U, lattice)
    workspace = ReferenceLCNN.ReferenceWorkspace(U)
    transformed_workspace = ReferenceLCNN.ReferenceWorkspace(transformed_U)

    plaquettes = ReferenceLCNN.plaquette_features(
        ReferenceLCNN.Plaq(), U, workspace,
    )
    transformed_plaquettes = ReferenceLCNN.plaquette_features(
        ReferenceLCNN.Plaq(), transformed_U, transformed_workspace,
    )
    @test length(plaquettes.channels) == binomial(4, 2) == 6
    for channel in eachindex(plaquettes.channels)
        expected = _lcnn_transform_feature(plaquettes.channels[channel], lattice)
        @test _lcnn_test_field_isapprox(
            transformed_plaquettes.channels[channel], expected, lattice;
            atol=8e-11,
            rtol=8e-11,
        )
    end

    # Use all six 4D plaquette channels. Positive shifts keep this correctness
    # test compact; the separate 2D test exercises negative transports.
    feature_model = ReferenceLCNN.LCNNFeatureModel(
        U, 1;
        radius=1,
        shifts=:positive,
        identity=false,
        adjoints=false,
    )
    action = ReferenceLCNN.LCNNAction(feature_model; reduction=:mean)
    parameters = ReferenceLCNN.initial_parameters(
        MersenneTwister(0x4d535533), action, Float64,
    )
    @test ReferenceLCNN.parameter_shapes(action) == (
        layers=((weight=(1, 6, 30),),),
        readout=(weight=(1,), bias=(1,)),
    )
    @test action(transformed_U, parameters, transformed_workspace) ≈
        action(U, parameters, workspace) atol=5e-10 rtol=5e-10

    link_model = ReferenceLCNN.LCNNLinkModel(feature_model)
    link_parameters = (
        layers=parameters.layers,
        lexp=(weight=fill(0.005, 4, 1),),
    )
    @test ReferenceLCNN.parameter_shapes(link_model).lexp.weight == (4, 1)
    @test ReferenceLCNN.parameter_count(link_model) == 184
    smeared = smear(U, lcnn_smearing(link_model, link_parameters))
    transformed_smeared = smear(
        transformed_U, lcnn_smearing(link_model, link_parameters),
    )
    expected_smeared = _lcnn_transform_links(smeared, lattice)
    @test length(smeared) == 4
    for direction in eachindex(smeared)
        @test _lcnn_test_field_isapprox(
            transformed_smeared[direction], expected_smeared[direction], lattice;
            atol=2e-9,
            rtol=2e-9,
        )
        for site_index in CartesianIndices(lattice)
            matrix = _lcnn_site_matrix(smeared[direction], Tuple(site_index))
            @test matrix' * matrix ≈ Matrix{ComplexF64}(I, 3, 3) atol=2e-9 rtol=2e-9
            @test det(matrix) ≈ 1 atol=2e-9 rtol=2e-9
        end
    end
end
