using Enzyme
using HDF5
using Optimisers
using Random

const TrainingLCNN = Gaugefields.LCNN

function _training_dense_dataset(action, teacher, lattice, seeds)
    dimension = length(lattice)
    colors = 2
    links = Array{ComplexF64}(
        undef, colors, colors, lattice..., dimension, length(seeds),
    )
    targets = Array{Float64}(undef, lattice..., length(seeds))
    for (sample, seed) in pairs(seeds)
        U = gauge_configuration(
            lattice; colors, halo=1, start=:hot, seed,
            process_grid=ntuple(_ -> 1, dimension), verbose=0,
        )
        workspace = TrainingLCNN.ModelWorkspace(action, U)
        selectors = (ntuple(_ -> Colon(), dimension)..., sample)
        targets[selectors...] .= TrainingLCNN.site_predictions(
            action, U, teacher, workspace,
        )
        for site_index in CartesianIndices(lattice)
            site = Tuple(site_index)
            for direction in 1:dimension, column in 1:colors, row in 1:colors
                links[row, column, site..., direction, sample] =
                    U[direction][row, column, site...]
            end
        end
    end
    return TrainingLCNN.DenseLCNNDataset(
        links, targets, lattice; beta=collect(range(0.1, 0.2; length=length(seeds))),
    )
end

function _training_dataset_gradient(action, parameters, dataset, indices)
    dimension = length(dataset.lattice)
    U = gauge_configuration(
        dataset.lattice;
        colors=dataset.colors,
        halo=1,
        start=:cold,
        process_grid=ntuple(_ -> 1, dimension),
        eltype=eltype(dataset.links),
        verbose=0,
    )
    workspace = TrainingLCNN.ModelWorkspace(action, U)
    shadow_workspace = Enzyme.make_zero(workspace)
    gradient = Enzyme.make_zero(parameters)
    for index in indices
        TrainingLCNN.load_sample!(U, dataset, index)
        Enzyme.remake_zero!(shadow_workspace)
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(TrainingLCNN.site_mse_loss),
            Enzyme.Active,
            Enzyme.Duplicated(parameters, gradient),
            Enzyme.Const(action),
            Enzyme.Const(U),
            Enzyme.Duplicated(workspace, shadow_workspace),
            Enzyme.Const(TrainingLCNN.sample_target(dataset, index)),
        )
    end
    return gradient
end

@testset "LCNN site-local loss and optional training stack" begin
    lattice = (2, 2)
    U = gauge_configuration(
        lattice; colors=2, halo=1, start=:hot, seed=0x515445,
        process_grid=(1, 1), verbose=0,
    )
    action = TrainingLCNN.favoni2022_wilson_1x2_small(
        U; convention=:favoni_prl,
    )
    parameters = TrainingLCNN.initial_parameters(
        MersenneTwister(91), action, Float64,
    )
    workspace = TrainingLCNN.ModelWorkspace(action, U)
    predictions = TrainingLCNN.site_predictions(
        action, U, parameters, workspace,
    )
    target = predictions .+ reshape([0.1, -0.2, 0.3, -0.4], lattice)
    expected_loss = sum(abs2, predictions .- target) / length(target)
    @test TrainingLCNN.site_mse_loss(
        parameters, action, U, workspace, target,
    ) ≈ expected_loss
    @test TrainingLCNN.global_average_mse(
        action, U, parameters, target, workspace,
    ) ≈ (sum(predictions - target) / length(target))^2

    gradient = Enzyme.make_zero(parameters)
    shadow_workspace = Enzyme.make_zero(workspace)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(TrainingLCNN.site_mse_loss),
        Enzyme.Active,
        Enzyme.Duplicated(parameters, gradient),
        Enzyme.Const(action),
        Enzyme.Const(U),
        Enzyme.Duplicated(workspace, shadow_workspace),
        Enzyme.Const(target),
    )
    step = 1.0e-6
    original_bias = only(parameters.readout.bias)
    parameters.readout.bias[1] = original_bias + step
    plus = TrainingLCNN.site_mse_loss(parameters, action, U, workspace, target)
    parameters.readout.bias[1] = original_bias - step
    minus = TrainingLCNN.site_mse_loss(parameters, action, U, workspace, target)
    parameters.readout.bias[1] = original_bias
    @test gradient.readout.bias[1] ≈ (plus - minus) / (2step) rtol=2e-6 atol=2e-8

    weight_index = first(eachindex(parameters.layers[1].weight))
    original_weight = parameters.layers[1].weight[weight_index]
    parameters.layers[1].weight[weight_index] = original_weight + step
    plus = TrainingLCNN.site_mse_loss(parameters, action, U, workspace, target)
    parameters.layers[1].weight[weight_index] = original_weight - step
    minus = TrainingLCNN.site_mse_loss(parameters, action, U, workspace, target)
    parameters.layers[1].weight[weight_index] = original_weight
    @test gradient.layers[1].weight[weight_index] ≈
        (plus - minus) / (2step) rtol=2e-5 atol=2e-7

    @testset "4D SU(3) site-local readout" begin
        lattice4 = (2, 2, 2, 2)
        U4 = gauge_configuration(
            lattice4; colors=3, halo=1, start=:hot, seed=0x53495445,
            process_grid=(1, 1, 1, 1), verbose=0,
        )
        layer4 = TrainingLCNN.LCB(
            6 => 1;
            convention=:favoni_arxiv,
            kernel_size=2,
            identity=false,
            adjoints=false,
        )
        action4 = TrainingLCNN.LCNNAction(
            TrainingLCNN.LCNNFeatureModel(U4, (layer4,));
            component=:real,
            reduction=:mean,
        )
        parameters4 = TrainingLCNN.initial_parameters(
            MersenneTwister(92), action4, Float64,
        )
        workspace4 = TrainingLCNN.ModelWorkspace(action4, U4)
        prediction4 = TrainingLCNN.site_predictions(
            action4, U4, parameters4, workspace4,
        )
        target4 = zeros(Float64, lattice4)
        @test size(prediction4) == lattice4
        @test TrainingLCNN.site_mse_loss(
            parameters4, action4, U4, workspace4, target4,
        ) ≈ sum(abs2, prediction4) / prod(lattice4)
    end

    @testset "PyTorch AdamW AMSGrad equation" begin
        rule = TrainingLCNN.adamw_amsgrad(
            ; eta=0.03, beta=(0.8, 0.9), lambda=0.04, epsilon=1e-7,
        )
        values = [0.7, -1.2]
        reference = copy(values)
        state = Optimisers.setup(rule, (values=values,))
        first_moment = zeros(2)
        second_moment = zeros(2)
        maximum_second_moment = zeros(2)
        beta_power = (0.8, 0.9)
        for current_gradient in ([0.2, -0.4], [-0.5, 0.1], [0.3, 0.6])
            first_moment .= 0.8 .* first_moment .+ 0.2 .* current_gradient
            second_moment .= 0.9 .* second_moment .+ 0.1 .* abs2.(current_gradient)
            maximum_second_moment .= max.(maximum_second_moment, second_moment)
            adaptive = first_moment ./ (1 - beta_power[1]) ./
                (sqrt.(maximum_second_moment ./ (1 - beta_power[2])) .+ 1e-7)
            reference .-= 0.03 .* (adaptive .+ 0.04 .* reference)
            state, updated = Optimisers.update!(
                state, (values=values,), (values=current_gradient,),
            )
            values = updated.values
            beta_power = beta_power .* (0.8, 0.9)
            @test values ≈ reference rtol=2e-15 atol=2e-15
        end
    end

    @testset "Favoni HDF5 axis and site conversion" begin
        teacher = deepcopy(parameters)
        teacher.readout.bias[1] += 0.17
        teacher.layers[1].weight[1] -= 0.09
        dataset = _training_dense_dataset(
            action, teacher, lattice, (0xD001, 0xD002),
        )
        gradient1 = _training_dataset_gradient(
            action, parameters, dataset, (1,),
        )
        gradient2 = _training_dataset_gradient(
            action, parameters, dataset, (2,),
        )
        accumulated_gradient = _training_dataset_gradient(
            action, parameters, dataset, (1, 2),
        )
        for layer_index in eachindex(parameters.layers)
            @test accumulated_gradient.layers[layer_index].weight ≈
                gradient1.layers[layer_index].weight .+
                gradient2.layers[layer_index].weight
        end
        @test accumulated_gradient.readout.weight ≈
            gradient1.readout.weight .+ gradient2.readout.weight
        @test accumulated_gradient.readout.bias ≈
            gradient1.readout.bias .+ gradient2.readout.bias
        mktempdir() do directory
            path = joinpath(directory, "favoni.hdf5")
            volume = prod(lattice)
            raw_links = Array{ComplexF64}(
                undef, 2, 2, 2, volume, length(dataset),
            )
            raw_targets = Array{ComplexF64}(undef, volume, length(dataset))
            for sample in 1:length(dataset), flat_site in 1:volume
                zero_based = flat_site - 1
                site = (
                    div(zero_based, lattice[2]) + 1,
                    mod(zero_based, lattice[2]) + 1,
                )
                for direction in 1:2, column in 1:2, row in 1:2
                    raw_links[row, column, direction, flat_site, sample] =
                        dataset.links[row, column, site..., direction, sample]
                end
                raw_targets[flat_site, sample] = dataset.targets[site..., sample]
            end
            h5open(path, "w") do file
                file["u"] = raw_links
                file["trW_1x2"] = raw_targets
                file["dims"] = collect(lattice)
                file["beta"] = dataset.beta
                file["w"] = zeros(ComplexF64, 1)
            end
            loaded = TrainingLCNN.read_favoni2022_dataset(path)
            @test loaded.lattice == lattice
            @test loaded.colors == 2
            @test loaded.links == dataset.links
            @test loaded.targets == dataset.targets
            @test loaded.beta == dataset.beta
            subset = TrainingLCNN.read_favoni2022_dataset(path; samples=2:2)
            @test length(subset) == 1
            @test subset.links == dataset.links[:, :, :, :, :, 2:2]
        end

        initial = deepcopy(parameters)
        config = TrainingLCNN.TrainingConfig(
            ; max_epochs=4, batch_size=2, learning_rate=0.0,
            patience=1, shuffle=false,
        )
        result = TrainingLCNN.fit!(
            action, initial, dataset, dataset; config,
        )
        @test result.stopped_early
        @test result.best_epoch == 1
        @test length(result.history) == 2
        @test result.parameters.layers[1].weight == parameters.layers[1].weight
        @test TrainingLCNN.evaluate_dataset(
            action, result.parameters, dataset,
        ) ≈ result.best_validation_loss
        @test TrainingLCNN.evaluate_dataset(
            action, result.parameters, dataset; global_average=true,
        ) ≤ result.best_validation_loss + 10eps(Float64)

        dataset32 = TrainingLCNN.DenseLCNNDataset(
            ComplexF32.(dataset.links),
            Float32.(dataset.targets),
            dataset.lattice;
            beta=Float32.(dataset.beta),
        )
        student32 = TrainingLCNN.initial_parameters(
            MersenneTwister(93), action, Float32,
        )
        loss_before = TrainingLCNN.evaluate_dataset(
            action, student32, dataset32,
        )
        trained32 = TrainingLCNN.fit!(
            action,
            student32,
            dataset32,
            dataset32;
            config=TrainingLCNN.TrainingConfig(
                ; max_epochs=1, batch_size=2, learning_rate=3.0f-3,
                patience=1, shuffle=false,
            ),
        )
        loss_after = TrainingLCNN.evaluate_dataset(
            action, trained32.parameters, dataset32,
        )
        @test isfinite(loss_after)
        @test loss_after < loss_before
    end
end
