raw"""
    LExp(channels; init_weight_factor=0.01)

Gauge-equivariant exponentiation head from Favoni et al.,
Eqs. (8)--(9). For each link direction `mu`, real weights form

```math
A_{x,\mu}=\sum_i \beta_{\mu i}W_{x,i},\qquad
U'_{x,\mu}=\exp([A_{x,\mu}]_{\mathrm{TA}})U_{x,\mu},
```

where `TA` is the traceless anti-Hermitian projection used by Gaugefields.
The weight shape is `(Dim, channels)`. This turns site-covariant LCNN
features into a genuine SU(N) link smearing.
"""
struct LExp <: AbstractGaugeEquivariantMap
    channels::Int
    init_weight_factor::Float64

    function LExp(
        channels::Integer;
        init_weight_factor::Real=0.01,
    )
        channels > 0 || throw(ArgumentError(
            "LExp channel count must be positive; got $channels",
        ))
        isfinite(init_weight_factor) || throw(ArgumentError(
            "LExp init_weight_factor must be finite",
        ))
        return new(Int(channels), Float64(init_weight_factor))
    end
end

parameter_shape(layer::LExp, dimension::Integer) =
    (Int(dimension), layer.channels)
parameter_shape(layer::LExp, ::Val{Dim}) where {Dim} =
    parameter_shape(layer, Dim)

"""
    LCNNLinkModel(feature_model[, update=LExp(output_channels(feature_model))])

Compose an LCNN feature model with an `LExp` head. Unlike
[`LCNNFeatureModel`](@ref), this model maps gauge links to updated gauge links.
"""
struct LCNNLinkModel{Dim,M,E} <: AbstractGaugeEquivariantMap
    feature_model::M
    update::E
end

function LCNNLinkModel(
    feature_model::LCNNFeatureModel{Dim},
    update::LExp=LExp(output_channels(feature_model)),
) where {Dim}
    update.channels == output_channels(feature_model) || throw(DimensionMismatch(
        "LExp expects $(update.channels) channels, but the feature model " *
        "produces $(output_channels(feature_model))",
    ))
    return LCNNLinkModel{Dim,typeof(feature_model),typeof(update)}(
        feature_model, update,
    )
end

function parameter_shapes(model::LCNNLinkModel{Dim}) where {Dim}
    feature_shapes = parameter_shapes(model.feature_model)
    return (
        layers=feature_shapes.layers,
        lexp=(weight=parameter_shape(model.update, Dim),),
    )
end

function parameter_count(model::LCNNLinkModel)
    shapes = parameter_shapes(model)
    return sum(prod(group.weight) for group in shapes.layers; init=0) +
        prod(shapes.lexp.weight)
end

function validate_parameters(model::LCNNLinkModel{Dim}, parameters) where {Dim}
    validate_parameters(model.feature_model, parameters)
    hasproperty(parameters, :lexp) || throw(ArgumentError(
        "LCNN link parameters must provide a `lexp` group",
    ))
    hasproperty(parameters.lexp, :weight) || throw(ArgumentError(
        "LCNN LExp parameters must provide `weight`",
    ))
    weight = parameters.lexp.weight
    eltype(weight) <: Real || throw(ArgumentError(
        "LCNN LExp weights must be real; got $(eltype(weight))",
    ))
    expected = parameter_shape(model.update, Dim)
    size(weight) == expected || throw(DimensionMismatch(
        "LCNN LExp weight shape must be $expected, got $(size(weight))",
    ))
    return true
end

function initial_parameters(
    rng::AbstractRNG,
    model::LCNNLinkModel{Dim},
    ::Type{T}=Float32,
) where {Dim,T<:Real}
    feature_parameters = initial_parameters(rng, model.feature_model, T)
    scale = T(model.update.init_weight_factor) / sqrt(T(model.update.channels))
    lexp = (
        weight=scale .* randn(
            rng, T, parameter_shape(model.update, Dim)...,
        ),
    )
    return (; layers=feature_parameters.layers, lexp)
end

"""Reusable feature, Lie-algebra, exponential, and output fields for `LExp`."""
struct LinkModelWorkspace{T,F,V}
    feature::F
    combinations::V
    generators::V
    exponentials::V
    output::V
    exponential_temps::V
end

function LinkModelWorkspace(
    model::LCNNLinkModel{Dim},
    U::Vector{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(U) == Dim || throw(ArgumentError(
        "a $Dim-dimensional LCNN link model expects $Dim links",
    ))
    fields() = [similar(first(U)) for _ in 1:Dim]
    return LinkModelWorkspace{T,ModelWorkspace{T},Vector{T}}(
        ModelWorkspace(model.feature_model, U),
        fields(),
        fields(),
        fields(),
        fields(),
        [similar(first(U)) for _ in 1:3],
    )
end

ModelWorkspace(model::LCNNLinkModel, U) = LinkModelWorkspace(model, U)

function _lexp_only!(
    output::Vector{T},
    links,
    channels,
    weights,
    combinations::Vector{T},
    generators::Vector{T},
    exponentials::Vector{T},
    exponential_temps::Vector{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    for direction in 1:Dim
        clear_U!(combinations[direction])
        for channel in eachindex(channels)
            add_U!(
                combinations[direction],
                weights[direction, channel],
                channels[channel],
            )
        end
        Traceless_antihermitian!(
            generators[direction], combinations[direction],
        )
        exptU!(
            exponentials[direction], 1, generators[direction],
            exponential_temps,
        )
        mul!(output[direction], exponentials[direction], links[direction])
    end
    set_wing_U!.(output)
    return output
end

"""Apply `LExp` into preallocated link fields."""
function lexp!(
    output::Vector{T},
    layer::LExp,
    features::GaugeFeatures,
    parameters,
    combinations::Vector{T},
    generators::Vector{T},
    exponentials::Vector{T},
    exponential_temps::Vector{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(features.channels) == layer.channels || throw(DimensionMismatch(
        "LExp expects $(layer.channels) channels, got " *
        "$(length(features.channels))",
    ))
    length(output) == Dim || throw(DimensionMismatch(
        "LExp needs $Dim output links, got $(length(output))",
    ))
    expected = parameter_shape(layer, Dim)
    hasproperty(parameters, :weight) || throw(ArgumentError(
        "LExp parameters must provide `weight`",
    ))
    size(parameters.weight) == expected || throw(DimensionMismatch(
        "LExp weight shape must be $expected, got $(size(parameters.weight))",
    ))
    eltype(parameters.weight) <: Real || throw(ArgumentError(
        "LExp weights must be real; got $(eltype(parameters.weight))",
    ))
    return _lexp_only!(
        output,
        features.links,
        features.channels,
        parameters.weight,
        combinations,
        generators,
        exponentials,
        exponential_temps,
    )
end

function forward_links!(
    model::LCNNLinkModel{Dim},
    U::Vector{T},
    parameters,
    workspace::LinkModelWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    validate_parameters(model, parameters)
    channels = _forward_feature_channels!(
        model.feature_model,
        U,
        (layers=parameters.layers,),
        workspace.feature,
    )
    features = GaugeFeatures(U, channels)
    return lexp!(
        workspace.output,
        model.update,
        features,
        parameters.lexp,
        workspace.combinations,
        workspace.generators,
        workspace.exponentials,
        workspace.exponential_temps,
    )
end

function forward_links(
    model::LCNNLinkModel,
    U,
    parameters,
    workspace::LinkModelWorkspace=LinkModelWorkspace(model, U),
)
    return forward_links!(model, U, parameters, workspace)
end

(model::LCNNLinkModel)(U, parameters, workspace=LinkModelWorkspace(model, U)) =
    forward_links!(model, U, parameters, workspace)
