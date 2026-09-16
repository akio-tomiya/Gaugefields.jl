module GaugefieldsOptimisersExt

using Enzyme
using Gaugefields
using Optimisers
using Random: MersenneTwister, shuffle!

const LCNN = Gaugefields.LCNN

"""PyTorch-compatible AdamW with AMSGrad's maximum second moment."""
struct AdamWAMSGrad{E,B,L,P} <: Optimisers.AbstractRule
    eta::E
    beta::B
    lambda::L
    epsilon::P
end

function AdamWAMSGrad(
    eta::Real=0.001,
    beta::Tuple{<:Real,<:Real}=(0.9, 0.999),
    lambda::Real=0.0,
    epsilon::Real=1.0e-8,
)
    eta >= 0 || throw(DomainError(eta, "learning rate must be nonnegative"))
    lambda >= 0 || throw(DomainError(lambda, "weight decay must be nonnegative"))
    length(beta) == 2 && all(value -> 0 <= value < 1, beta) ||
        throw(ArgumentError("beta must contain two values in [0, 1)"))
    epsilon >= 0 || throw(DomainError(epsilon, "epsilon must be nonnegative"))
    converted = (float(eta), Tuple(beta), float(lambda), float(epsilon))
    return AdamWAMSGrad{
        typeof(converted[1]),
        typeof(converted[2]),
        typeof(converted[3]),
        typeof(converted[4]),
    }(converted...)
end

function Optimisers.init(rule::AdamWAMSGrad, parameter::AbstractArray{T}) where {T}
    return (zero(parameter), zero(parameter), zero(parameter), T.(rule.beta))
end

function Optimisers.apply!(
    rule::AdamWAMSGrad,
    state,
    parameter::AbstractArray{T},
    gradient,
) where {T}
    eta = T(rule.eta)
    beta = T.(rule.beta)
    lambda = T(rule.lambda)
    epsilon = T(rule.epsilon)
    first_moment, second_moment, maximum_second_moment, beta_power = state

    @. first_moment = beta[1] * first_moment + (1 - beta[1]) * gradient
    @. second_moment = beta[2] * second_moment +
        (1 - beta[2]) * abs2(gradient)
    @. maximum_second_moment = max(maximum_second_moment, second_moment)
    update = @. eta * (
        first_moment / (1 - beta_power[1]) /
        (sqrt(maximum_second_moment / (1 - beta_power[2])) + epsilon) +
        lambda * parameter
    )
    return (
        first_moment,
        second_moment,
        maximum_second_moment,
        beta_power .* beta,
    ), update
end

function Base.show(io::IO, rule::AdamWAMSGrad)
    print(
        io,
        "AdamWAMSGrad(eta=$(rule.eta), beta=$(rule.beta), " *
        "lambda=$(rule.lambda), epsilon=$(rule.epsilon))",
    )
end

"""Construct the exact optimizer variant used by PyTorch AdamW(amsgrad=true)."""
LCNN.adamw_amsgrad(; eta=0.001, beta=(0.9, 0.999), lambda=0.0, epsilon=1.0e-8) =
    AdamWAMSGrad(eta, beta, lambda, epsilon)

function _optimizer(config::LCNN.TrainingConfig)
    if config.amsgrad
        return LCNN.adamw_amsgrad(
            ; eta=config.learning_rate,
            beta=config.beta,
            lambda=config.weight_decay,
            epsilon=config.epsilon,
        )
    end
    return Optimisers.AdamW(
        config.learning_rate,
        config.beta,
        config.weight_decay,
        config.epsilon;
        couple=true,
    )
end

function _clear_gradient!(gradient)
    for group in gradient.layers
        fill!(group.weight, zero(eltype(group.weight)))
    end
    fill!(gradient.readout.weight, zero(eltype(gradient.readout.weight)))
    fill!(gradient.readout.bias, zero(eltype(gradient.readout.bias)))
    return gradient
end

function _scale_gradient!(gradient, scale)
    for group in gradient.layers
        group.weight .*= scale
    end
    gradient.readout.weight .*= scale
    gradient.readout.bias .*= scale
    return gradient
end

function _copy_parameters(parameters)
    layers = map(parameters.layers) do group
        (weight=copy(group.weight),)
    end
    return (
        layers=layers,
        readout=(
            weight=copy(parameters.readout.weight),
            bias=copy(parameters.readout.bias),
        ),
    )
end

function _copy_parameters!(destination, source)
    for index in eachindex(destination.layers)
        copyto!(destination.layers[index].weight, source.layers[index].weight)
    end
    copyto!(destination.readout.weight, source.readout.weight)
    copyto!(destination.readout.bias, source.readout.bias)
    return destination
end

function _configuration(dataset::LCNN.DenseLCNNDataset{Dim}) where {Dim}
    return Gaugefields.gauge_configuration(
        dataset.lattice;
        colors=dataset.colors,
        halo=1,
        start=:cold,
        process_grid=ntuple(_ -> 1, Dim),
        eltype=eltype(dataset.links),
        verbose=0,
    )
end

function _require_reference_training_backend(U)
    field = first(U)
    hasproperty(field, :U) || throw(ArgumentError(
        "the LCNN dataset trainer currently requires LatticeMatrices-backed fields",
    ))
    lattice = getproperty(field, :U)
    hasproperty(lattice, :A) && lattice.A isa Array || throw(ArgumentError(
        "site-local reverse training currently requires the CPU JACC backend; " *
        "GPU/MPI forward and scalar-action gradients remain supported",
    ))
    hasproperty(lattice, :dims) && all(==(1), lattice.dims) ||
        throw(ArgumentError(
            "site-local dataset training currently requires one MPI rank",
        ))
    return nothing
end

function _dataset_loss!(
    action,
    parameters,
    dataset,
    U,
    workspace;
    global_average::Bool=false,
)
    iszero(length(dataset)) && throw(ArgumentError(
        "dataset must contain at least one sample",
    ))
    total = zero(promote_type(
        eltype(parameters.readout.weight), eltype(dataset.targets),
    ))
    for index in 1:length(dataset)
        LCNN.load_sample!(U, dataset, index)
        target = LCNN.sample_target(dataset, index)
        total += global_average ?
            LCNN.global_average_mse(action, U, parameters, target, workspace) :
            LCNN.site_mse_loss(parameters, action, U, workspace, target)
    end
    return total / length(dataset)
end

"""
    LCNN.evaluate_dataset(action, parameters, dataset; global_average=false)

Evaluate either the site-local MSE used for training or, with
`global_average=true`, the final test metric used by the released code.
"""
function LCNN.evaluate_dataset(
    action::LCNN.LCNNAction,
    parameters,
    dataset::LCNN.DenseLCNNDataset;
    global_average::Bool=false,
)
    U = _configuration(dataset)
    _require_reference_training_backend(U)
    workspace = LCNN.ModelWorkspace(action, U)
    return _dataset_loss!(
        action, parameters, dataset, U, workspace; global_average,
    )
end

"""
    LCNN.fit!(action, parameters, train, validation;
              config=TrainingConfig(), optimiser=nothing, callback=nothing)

Train with site-local MSE and sample-wise gradient accumulation. `batch_size`
therefore has the same mathematical meaning as the PyTorch batch axis without
adding a batch dimension to `LatticeMatrix`. The validation loss is monitored
once per epoch; the best parameter copy is restored before returning.
"""
function LCNN.fit!(
    action::LCNN.LCNNAction,
    parameters,
    train::LCNN.DenseLCNNDataset,
    validation::LCNN.DenseLCNNDataset;
    config::LCNN.TrainingConfig=LCNN.TrainingConfig(),
    optimiser=nothing,
    callback=nothing,
)
    train.lattice == validation.lattice || throw(DimensionMismatch(
        "training and validation lattice sizes differ",
    ))
    train.colors == validation.colors || throw(DimensionMismatch(
        "training and validation color counts differ",
    ))
    iszero(length(train)) && throw(ArgumentError("training dataset must not be empty"))
    iszero(length(validation)) && throw(ArgumentError(
        "validation dataset must not be empty",
    ))
    LCNN.validate_parameters(action, parameters)

    U = _configuration(train)
    _require_reference_training_backend(U)
    workspace = LCNN.ModelWorkspace(action, U)
    shadow_workspace = Enzyme.make_zero(workspace)
    gradient = Enzyme.make_zero(parameters)
    rule = optimiser === nothing ? _optimizer(config) : optimiser
    optimiser_state = Optimisers.setup(rule, parameters)
    best_optimiser_state = deepcopy(optimiser_state)
    rng = MersenneTwister(config.seed)
    order = collect(1:length(train))

    loss_type = promote_type(
        eltype(parameters.readout.weight), eltype(validation.targets),
    )
    validation_loss = zero(loss_type)
    # The released Lightning run starts checkpoint/early-stopping monitoring
    # after epoch one.  Do not treat the untrained epoch-zero parameters as a
    # checkpoint candidate.
    best_validation_loss = oftype(validation_loss, Inf)
    best_parameters = _copy_parameters(parameters)
    best_epoch = 0
    epochs_without_improvement = 0
    history = NamedTuple[]
    stopped_early = false

    for epoch in 1:config.max_epochs
        config.shuffle && shuffle!(rng, order)
        accumulated_training_loss = zero(validation_loss)
        samples_seen = 0
        for batch_start in 1:config.batch_size:length(order)
            batch_stop = min(batch_start + config.batch_size - 1, length(order))
            batch = @view order[batch_start:batch_stop]
            _clear_gradient!(gradient)
            for sample_index in batch
                LCNN.load_sample!(U, train, sample_index)
                target = LCNN.sample_target(train, sample_index)
                accumulated_training_loss += LCNN.site_mse_loss(
                    parameters, action, U, workspace, target,
                )
                Enzyme.remake_zero!(shadow_workspace)
                Enzyme.autodiff(
                    Enzyme.set_runtime_activity(Enzyme.Reverse),
                    Enzyme.Const(LCNN.site_mse_loss),
                    Enzyme.Active,
                    Enzyme.Duplicated(parameters, gradient),
                    Enzyme.Const(action),
                    Enzyme.Const(U),
                    Enzyme.Duplicated(workspace, shadow_workspace),
                    Enzyme.Const(target),
                )
                samples_seen += 1
            end
            _scale_gradient!(gradient, inv(length(batch)))
            optimiser_state, parameters = Optimisers.update!(
                optimiser_state, parameters, gradient,
            )
        end

        training_loss = accumulated_training_loss / samples_seen
        validation_loss = _dataset_loss!(
            action, parameters, validation, U, workspace,
        )
        improved = validation_loss < best_validation_loss - config.min_delta
        if improved
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_parameters = _copy_parameters(parameters)
            best_optimiser_state = deepcopy(optimiser_state)
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end
        record = (
            epoch=epoch,
            training_loss=training_loss,
            validation_loss=validation_loss,
            best_validation_loss=best_validation_loss,
        )
        push!(history, record)
        callback === nothing || callback(record)

        if !improved && epochs_without_improvement >= config.patience
            stopped_early = epoch < config.max_epochs
            break
        end
    end

    _copy_parameters!(parameters, best_parameters)
    return LCNN.TrainingResult(
        parameters,
        best_optimiser_state,
        history,
        best_epoch,
        best_validation_loss,
        stopped_early,
    )
end

end
