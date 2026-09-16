module LCNN

using LinearAlgebra
using Random: AbstractRNG, randn
import LuxCore
import NPZ: npzread

import Wilsonloop: make_plaq
import ..AbstractGaugefields_module:
    AbstractGaugefields,
    _release_shifted_U!,
    add_U!,
    clear_U!,
    exptU!,
    evaluate_gaugelinks!,
    set_wing_U!,
    shift_U,
    substitute_U!,
    Traceless_antihermitian!,
    unit_U!

abstract type AbstractGaugeEquivariantMap end
abstract type AbstractLCNNFeatureMap <: AbstractGaugeEquivariantMap end
abstract type AbstractLCNNReadout end

# Implemented by optional package extensions.  Keeping these entry points in
# the framework-independent module lets dataset inspection work without
# loading an optimiser and keeps HDF5 out of Gaugefields' mandatory deps.
function read_favoni2022_dataset end
function adamw_amsgrad end
function fit! end
function evaluate_dataset end

# Implemented by the Enzyme package extension.  The declarations keep Enzyme
# optional while giving the framework-independent model a stable dS/dU API.
function link_gradient end
function link_gradient! end
function parameter_gradient end
function parameter_gradient! end
function link_model_pullback end
function link_model_pullback! end

"""Alias for [`link_gradient`](@ref), spelling the result as `dS/dU`."""
dSdu(args...; kwargs...) = link_gradient(args...; kwargs...)

"""In-place alias for [`link_gradient!`](@ref)."""
dSdu!(args...) = link_gradient!(args...)

"""
    GaugeFeatures(links, channels)

Framework-independent LCNN data: gauge links together with matrix-valued
site features. This is an internal reference representation; the channel
container will be revisited before the LCNN API is exported.
"""
struct GaugeFeatures{L,C}
    links::L
    channels::C
end

"""
    DenseLCNNDataset(links, targets, lattice; beta=nothing)

In-memory LCNN regression data in a Julia-native layout. `links` has shape
`(NC, NC, lattice..., Dim, samples)` and real `targets` has shape
`(lattice..., samples)`.  The HDF5 extension converts the flattened C-order
site axis used by the Favoni et al. datasets to this coordinate layout once
when loading.
"""
struct DenseLCNNDataset{Dim,L,T,B}
    links::L
    targets::T
    lattice::NTuple{Dim,Int}
    colors::Int
    beta::B
end

"""
    TrainingConfig(; max_epochs=20, batch_size=50, learning_rate=3f-3,
                     weight_decay=0, beta=(0.9, 0.999), epsilon=1e-8,
                     amsgrad=true, patience=max_epochs ÷ 4, min_delta=0,
                     shuffle=true, seed=0)

Settings for the optional Enzyme + Optimisers LCNN trainer. Defaults reproduce
the optimizer and early-stopping choices of the released 2D Wilson-loop
training scripts.
"""
struct TrainingConfig{LR,WD,B,E,D}
    max_epochs::Int
    batch_size::Int
    learning_rate::LR
    weight_decay::WD
    beta::B
    epsilon::E
    amsgrad::Bool
    patience::Int
    min_delta::D
    shuffle::Bool
    seed::UInt64
end

function TrainingConfig(;
    max_epochs::Integer=20,
    batch_size::Integer=50,
    learning_rate=3.0f-3,
    weight_decay=0.0f0,
    beta=(0.9f0, 0.999f0),
    epsilon=1.0f-8,
    amsgrad::Bool=true,
    patience::Union{Nothing,Integer}=nothing,
    min_delta=zero(learning_rate),
    shuffle::Bool=true,
    seed::Integer=0,
)
    max_epochs > 0 || throw(ArgumentError("max_epochs must be positive"))
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    learning_rate >= 0 || throw(ArgumentError("learning_rate must be nonnegative"))
    weight_decay >= 0 || throw(ArgumentError("weight_decay must be nonnegative"))
    length(beta) == 2 && all(value -> 0 <= value < 1, beta) ||
        throw(ArgumentError("beta must contain two values in [0, 1)"))
    epsilon >= 0 || throw(ArgumentError("epsilon must be nonnegative"))
    selected_patience = patience === nothing ?
        max(1, Int(max_epochs) ÷ 4) : Int(patience)
    selected_patience >= 0 || throw(ArgumentError("patience must be nonnegative"))
    min_delta >= 0 || throw(ArgumentError("min_delta must be nonnegative"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    return TrainingConfig(
        Int(max_epochs), Int(batch_size), learning_rate, weight_decay,
        Tuple(beta), epsilon, amsgrad, selected_patience, min_delta,
        shuffle, UInt64(seed),
    )
end

"""Summary returned by [`fit!`](@ref), including the restored best parameters."""
struct TrainingResult{P,O,H,T}
    parameters::P
    optimiser_state::O
    history::H
    best_epoch::Int
    best_validation_loss::T
    stopped_early::Bool
end

function DenseLCNNDataset(
    links::AbstractArray,
    targets::AbstractArray,
    lattice::NTuple{Dim,<:Integer};
    beta=nothing,
) where {Dim}
    lattice_tuple = ntuple(index -> Int(lattice[index]), Dim)
    all(>(0), lattice_tuple) || throw(ArgumentError(
        "all lattice extents must be positive; got $lattice_tuple",
    ))
    ndims(links) == Dim + 4 || throw(DimensionMismatch(
        "links must have shape (NC, NC, lattice..., Dim, samples)",
    ))
    size(links, 1) == size(links, 2) || throw(DimensionMismatch(
        "link matrices must be square; got $(size(links, 1)) by $(size(links, 2))",
    ))
    Tuple(size(links)[3:(Dim + 2)]) == lattice_tuple ||
        throw(DimensionMismatch(
            "link lattice axes $(Tuple(size(links)[3:(Dim + 2)])) do not " *
            "match $lattice_tuple",
        ))
    size(links, Dim + 3) == Dim || throw(DimensionMismatch(
        "links contain $(size(links, Dim + 3)) directions, expected $Dim",
    ))
    ndims(targets) == Dim + 1 || throw(DimensionMismatch(
        "targets must have shape (lattice..., samples)",
    ))
    Tuple(size(targets)[1:Dim]) == lattice_tuple ||
        throw(DimensionMismatch(
            "target lattice axes $(Tuple(size(targets)[1:Dim])) do not " *
            "match $lattice_tuple",
        ))
    size(targets, Dim + 1) == size(links, Dim + 4) ||
        throw(DimensionMismatch(
            "links contain $(size(links, Dim + 4)) samples but targets " *
            "contain $(size(targets, Dim + 1))",
        ))
    eltype(targets) <: Real || throw(ArgumentError(
        "LCNN regression targets must be real; got $(eltype(targets))",
    ))
    samples = size(targets, Dim + 1)
    beta === nothing || length(beta) == samples || throw(DimensionMismatch(
        "beta must contain one value per sample ($samples)",
    ))
    return DenseLCNNDataset{Dim,typeof(links),typeof(targets),typeof(beta)}(
        links, targets, lattice_tuple, size(links, 1), beta,
    )
end

Base.length(dataset::DenseLCNNDataset{Dim}) where {Dim} =
    size(dataset.targets, Dim + 1)
Base.isempty(dataset::DenseLCNNDataset) = iszero(length(dataset))

function sample_target(dataset::DenseLCNNDataset{Dim}, index::Integer) where {Dim}
    checkbounds(1:length(dataset), index)
    selectors = (ntuple(_ -> Colon(), Dim)..., Int(index))
    return @view dataset.targets[selectors...]
end

"""Copy one dense dataset sample into a reusable gauge configuration."""
function load_sample!(
    U::Vector{T},
    dataset::DenseLCNNDataset{Dim},
    index::Integer,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(U) == Dim || throw(ArgumentError(
        "expected $Dim link directions, got $(length(U))",
    ))
    NC == dataset.colors || throw(DimensionMismatch(
        "configuration has SU($NC) links but dataset has SU($(dataset.colors))",
    ))
    Tuple(size(first(U))[3:end]) == dataset.lattice ||
        throw(DimensionMismatch(
            "configuration lattice $(Tuple(size(first(U))[3:end])) does not " *
            "match dataset lattice $(dataset.lattice)",
        ))
    checkbounds(1:length(dataset), index)
    for site_index in CartesianIndices(dataset.lattice)
        site = Tuple(site_index)
        for direction in 1:Dim, column in 1:NC, row in 1:NC
            U[direction][row, column, site...] = dataset.links[
                row, column, site..., direction, index,
            ]
        end
    end
    set_wing_U!.(U)
    return U
end

"""
    Plaq(; orientations=:positive)

Configuration for the plaquette feature generator. The first reference
implementation supports the nonredundant positive orientations `mu < nu`.
"""
struct Plaq <: AbstractLCNNFeatureMap
    orientations::Symbol

    function Plaq(; orientations::Symbol=:positive)
        orientations === :positive || throw(ArgumentError(
            "the reference Plaq implementation currently supports " *
            "orientations=:positive only",
        ))
        return new(orientations)
    end
end

"""Reusable fields for the allocation-heavy LCNN reference implementation."""
struct ReferenceWorkspace{T}
    path_temps::Vector{T}
    transport_product::T
    transported::T
    bilinear_product::T
    product_cotangent::T
    local_cotangent::T
    transported_cotangent::T
    transport_input_cotangent::T
    inverse_transport::T
    identity::T
end

function ReferenceWorkspace(U::Vector{T}) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(U) == Dim || throw(ArgumentError(
        "expected $Dim gauge-link directions, got $(length(U))",
    ))
    isempty(U) && throw(ArgumentError("a gauge configuration must not be empty"))
    identity = similar(first(U))
    unit_U!(identity)
    return ReferenceWorkspace{T}(
        [similar(first(U)), similar(first(U))],
        similar(first(U)),
        similar(first(U)),
        similar(first(U)),
        similar(first(U)),
        similar(first(U)),
        similar(first(U)),
        similar(first(U)),
        similar(first(U)),
        identity,
    )
end

"""
    plaquette_features(Plaq(), U[, workspace])

Construct one untraced, matrix-valued plaquette channel for every `mu < nu`.
The links are preserved in the returned `GaugeFeatures` object so subsequent
layers can perform gauge-covariant parallel transport.
"""
function plaquette_features(
    layer::Plaq,
    U::Vector{T},
    workspace::ReferenceWorkspace{T}=ReferenceWorkspace(U),
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    layer.orientations === :positive || error("unreachable orientation")
    length(U) == Dim || throw(ArgumentError(
        "expected $Dim gauge-link directions, got $(length(U))",
    ))

    channels = Vector{T}(undef, binomial(Dim, 2))
    channel = 1
    for mu in 1:Dim
        for nu in (mu + 1):Dim
            output = similar(first(U))
            path = make_plaq(mu, nu; Dim)
            evaluate_gaugelinks!(output, path, U, workspace.path_temps)
            channels[channel] = output
            channel += 1
        end
    end
    return GaugeFeatures(Tuple(U), channels)
end

function _plaquette_features_only!(
    channels::Vector{T},
    layer::Plaq,
    U::Vector{T},
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    expected = binomial(Dim, 2)
    length(channels) == expected || throw(DimensionMismatch(
        "Plaq expects $expected output channels, got $(length(channels))",
    ))
    layer.orientations === :positive || error("unreachable orientation")

    channel = 1
    for mu in 1:Dim
        for nu in (mu + 1):Dim
            path = make_plaq(mu, nu; Dim)
            evaluate_gaugelinks!(
                channels[channel], path, U, workspace.path_temps,
            )
            channel += 1
        end
    end
    return nothing
end

function plaquette_features!(
    channels::Vector{T},
    layer::Plaq,
    U::Vector{T},
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    _plaquette_features_only!(channels, layer, U, workspace)
    return GaugeFeatures(Tuple(U), channels)
end

@inline function _axis_shift(::Val{Dim}, direction::Int, step::Int) where {Dim}
    return ntuple(axis -> axis == direction ? step : 0, Dim)
end

"""
    parallel_transport!(output, links, feature, direction, step, workspace)

Transport a local feature from `x + step * e_direction` back to `x`.
The implementation accepts an arbitrary integer `step`.  Longer Wilson lines
are composed from the same radius-one primitive used by the Enzyme rule:

```
 step =  1: U(x) W(x+mu) U(x)'
 step =  0: W(x)
 step = -1: U(x-mu)' W(x-mu) U(x-mu)
```

"""
function _parallel_transport_one_step_only!(
    output::T,
    links,
    feature::T,
    direction::Integer,
    step::Integer,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(links) == Dim || throw(ArgumentError(
        "expected $Dim gauge-link directions, got $(length(links))",
    ))
    1 <= direction <= Dim || throw(ArgumentError(
        "direction must lie in 1:$Dim; got $direction",
    ))
    step in (-1, 1) || throw(ArgumentError(
        "the one-step transport primitive expects -1 or 1; got $step",
    ))

    shift = _axis_shift(Val(Dim), Int(direction), Int(step))
    shifted_feature = shift_U(feature, shift)
    if step > 0
        try
            mul!(workspace.transport_product, links[direction], shifted_feature)
            mul!(output, workspace.transport_product, links[direction]')
        finally
            _release_shifted_U!(shifted_feature)
        end
    else
        shifted_link = shift_U(links[direction], shift)
        try
            mul!(workspace.transport_product, shifted_link', shifted_feature)
            mul!(output, workspace.transport_product, shifted_link)
        finally
            _release_shifted_U!(shifted_link)
            _release_shifted_U!(shifted_feature)
        end
    end
    set_wing_U!(output)
    return nothing
end

function _parallel_transport_only!(
    output::T,
    links,
    feature::T,
    direction::Integer,
    step::Integer,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(links) == Dim || throw(ArgumentError(
        "expected $Dim gauge-link directions, got $(length(links))",
    ))
    1 <= direction <= Dim || throw(ArgumentError(
        "direction must lie in 1:$Dim; got $direction",
    ))
    if iszero(step)
        substitute_U!(output, feature)
        return nothing
    end

    orientation = sign(Int(step))
    source = feature
    distance = abs(Int(step))
    for segment in 1:distance
        destination = segment == distance ? output :
            workspace.path_temps[isodd(segment) ? 1 : 2]
        _parallel_transport_one_step_only!(
            destination,
            links,
            source,
            direction,
            orientation,
            workspace,
        )
        source = destination
    end
    return nothing
end

function parallel_transport!(
    output::T,
    links,
    feature::T,
    direction::Integer,
    step::Integer,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    _parallel_transport_only!(
        output, links, feature, direction, step, workspace,
    )
    return output
end

function parallel_transport(
    features::GaugeFeatures{<:NTuple{Dim,T}},
    channel::Integer,
    direction::Integer,
    step::Integer,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    checkbounds(features.channels, channel)
    output = similar(features.channels[channel])
    parallel_transport!(
        output,
        features.links,
        features.channels[channel],
        direction,
        step,
        workspace,
    )
    return output
end

"""
    LCB(in_channels => out_channels; convention=:favoni_prl, kernel_size=2,
        dilation=nothing, shifts=nothing, identity=true, adjoints=true)

Configuration for a combined gauge-equivariant convolution and bilinear
layer. `convention=:favoni_arxiv` reproduces the 35-parameter architecture
counted in the Favoni et al. arXiv table (no centered transported term and the
original dilation convention), while `convention=:favoni_prl` reproduces the 47-parameter
architecture stored in the released result checkpoints (one centered term and
the revised dilation convention). The basis order is ordinary channels, their
Hermitian conjugates, then the identity.
"""
struct LCB <: AbstractLCNNFeatureMap
    in_channels::Int
    out_channels::Int
    radius::Int
    kernel_size::Int
    dilation::Int
    shifts::Symbol
    include_identity::Bool
    include_adjoints::Bool
    convention::Symbol
    init_weight_factor::Float64

    function LCB(
        channels::Pair{<:Integer,<:Integer};
        convention::Symbol=:favoni_prl,
        kernel_size::Integer=2,
        dilation::Union{Nothing,Integer}=nothing,
        radius::Union{Nothing,Integer}=nothing,
        shifts::Union{Nothing,Symbol}=nothing,
        identity::Bool=true,
        adjoints::Bool=true,
        init_weight_factor::Real=1.0,
    )
        channels.first > 0 || throw(ArgumentError(
            "LCB input channel count must be positive; got $(channels.first)",
        ))
        channels.second > 0 || throw(ArgumentError(
            "LCB output channel count must be positive; got $(channels.second)",
        ))
        convention in (:favoni_arxiv, :favoni_prl) || throw(ArgumentError(
            "convention must be :favoni_arxiv or :favoni_prl; got $convention",
        ))
        if radius !== nothing
            radius >= 1 || throw(ArgumentError(
                "radius must be positive; got $radius",
            ))
            requested_kernel_size = Int(radius) + 1
            kernel_size == 2 || kernel_size == requested_kernel_size ||
                throw(ArgumentError(
                    "radius=$radius conflicts with kernel_size=$kernel_size",
                ))
            kernel_size = requested_kernel_size
        end
        kernel_size >= 2 || throw(ArgumentError(
            "kernel_size must be at least 2; got $kernel_size",
        ))
        actual_dilation = dilation === nothing ?
            (convention === :favoni_arxiv ? 0 : 1) : Int(dilation)
        actual_shifts = shifts === nothing ?
            (convention === :favoni_arxiv ? :positive : :both) : shifts
        if convention === :favoni_arxiv
            actual_dilation >= 0 || throw(ArgumentError(
                "favoni_arxiv convention dilation must be nonnegative; got $actual_dilation",
            ))
            actual_shifts === :positive || throw(ArgumentError(
                "the favoni_arxiv convention uses shifts=:positive",
            ))
        else
            actual_dilation >= 1 || throw(ArgumentError(
                "favoni_prl convention dilation must be positive; got $actual_dilation",
            ))
        end
        actual_shifts in (:positive, :both) || throw(ArgumentError(
            "shifts must be :positive or :both; got $actual_shifts",
        ))
        isfinite(init_weight_factor) || throw(ArgumentError(
            "init_weight_factor must be finite; got $init_weight_factor",
        ))
        return new(
            Int(channels.first),
            Int(channels.second),
            Int(kernel_size) - 1,
            Int(kernel_size),
            actual_dilation,
            actual_shifts,
            identity,
            adjoints,
            convention,
            Float64(init_weight_factor),
        )
    end
end

@inline function augmented_channel_count(layer::LCB)
    count = layer.in_channels
    layer.include_adjoints && (count += layer.in_channels)
    layer.include_identity && (count += 1)
    return count
end

function displacements(layer::LCB, ::Val{Dim}) where {Dim}
    spacing = layer.convention === :favoni_arxiv ?
        layer.dilation + 1 : layer.dilation
    paths = Tuple{Int,Int}[]
    layer.convention === :favoni_prl && push!(paths, (0, 0))
    for direction in 1:Dim
        if layer.shifts === :both
            for distance in 1:layer.radius
                push!(paths, (direction, -distance * spacing))
            end
        end
        for distance in 1:layer.radius
            push!(paths, (direction, distance * spacing))
        end
    end
    return Tuple(paths)
end

displacements(layer::LCB, dimension::Integer) =
    displacements(layer, Val(Int(dimension)))

function parameter_shape(layer::LCB, ::Val{Dim}) where {Dim}
    local_count = augmented_channel_count(layer)
    raw_transported = layer.in_channels * length(displacements(layer, Val(Dim)))
    transported_count = raw_transported
    layer.include_adjoints && (transported_count += raw_transported)
    layer.include_identity && (transported_count += 1)
    return (
        layer.out_channels,
        local_count,
        transported_count,
    )
end

parameter_shape(layer::LCB, dimension::Integer) =
    parameter_shape(layer, Val(Int(dimension)))

@inline function _local_descriptor(layer::LCB, index::Int)
    1 <= index <= augmented_channel_count(layer) ||
        throw(BoundsError(1:augmented_channel_count(layer), index))
    index <= layer.in_channels && return (:ordinary, index)
    offset = index - layer.in_channels
    if layer.include_adjoints && offset <= layer.in_channels
        return (:adjoint, offset)
    end
    return (:identity, 0)
end

function _transported_descriptor(layer::LCB, ::Val{Dim}, index::Int) where {Dim}
    paths = displacements(layer, Val(Dim))
    raw_count = layer.in_channels * length(paths)
    total = raw_count * (layer.include_adjoints ? 2 : 1) +
        (layer.include_identity ? 1 : 0)
    1 <= index <= total || throw(BoundsError(1:total, index))
    if index > raw_count
        if layer.include_adjoints && index <= 2 * raw_count
            raw_index = index - raw_count
            kind = :adjoint
        else
            return (:identity, 0, (0, 0))
        end
    else
        raw_index = index
        kind = :ordinary
    end
    path_index = (raw_index - 1) ÷ layer.in_channels + 1
    channel = (raw_index - 1) % layer.in_channels + 1
    return (kind, channel, paths[path_index])
end

function _augmented_channel(layer::LCB, channels, identity, index::Int)
    kind, channel = _local_descriptor(layer, index)
    kind === :identity && return identity
    return kind === :adjoint ? channels[channel]' : channels[channel]
end

function _transported_channel!(
    layer::LCB,
    links,
    channels::Vector{T},
    transported_index::Int,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    kind, channel, displacement = _transported_descriptor(
        layer, Val(Dim), transported_index,
    )
    kind === :identity && return workspace.identity
    direction, step = displacement
    value = if step == 0
        channels[channel]
    else
        _parallel_transport_only!(
            workspace.transported,
            links,
            channels[channel],
            direction,
            step,
            workspace,
        )
        workspace.transported
    end
    return kind === :adjoint ? value' : value
end

"""
    lcb!(outputs, layer, features, parameters, workspace)

Apply the allocation-heavy reference LCB. `parameters.weight` must be a real
array with shape returned by `parameter_shape(layer, Dim)` and indexing
`(output, local_basis, transported_basis)`.  This is the nonredundant layout
used by both upstream implementations.
"""
function _lcb_only!(
    outputs::Vector{T},
    layer::LCB,
    links,
    channels::Vector{T},
    parameters,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(channels) == layer.in_channels || throw(DimensionMismatch(
        "LCB expects $(layer.in_channels) input channels, got " *
        "$(length(channels))",
    ))
    length(outputs) == layer.out_channels || throw(DimensionMismatch(
        "LCB expects $(layer.out_channels) output fields, got $(length(outputs))",
    ))
    hasproperty(parameters, :weight) || throw(ArgumentError(
        "LCB parameters must provide a real `weight` array",
    ))
    weights = parameters.weight
    eltype(weights) <: Real || throw(ArgumentError(
        "LCB reference weights must be real; got $(eltype(weights))",
    ))
    expected_shape = parameter_shape(layer, Val(Dim))
    size(weights) == expected_shape || throw(DimensionMismatch(
        "LCB weight shape must be $expected_shape, got $(size(weights))",
    ))

    clear_U!.(outputs)
    local_count = augmented_channel_count(layer)
    transported_count = size(weights, 3)
    for transported_index in 1:transported_count
        transported = _transported_channel!(
            layer, links, channels, transported_index, workspace,
        )
        for local_index in 1:local_count
            local_feature = _augmented_channel(
                layer, channels, workspace.identity, local_index,
            )
            mul!(workspace.bilinear_product, local_feature, transported)
            for output_index in eachindex(outputs)
                add_U!(
                    outputs[output_index],
                    weights[output_index, local_index, transported_index],
                    workspace.bilinear_product,
                )
            end
        end
    end
    set_wing_U!.(outputs)
    return nothing
end

function lcb!(
    outputs::Vector{T},
    layer::LCB,
    features::GaugeFeatures{<:NTuple{Dim,T},<:Vector{T}},
    parameters,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    _lcb_only!(
        outputs,
        layer,
        features.links,
        features.channels,
        parameters,
        workspace,
    )
    return outputs
end

function lcb(
    layer::LCB,
    features::GaugeFeatures{<:NTuple{Dim,T},<:Vector{T}},
    parameters,
    workspace::ReferenceWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    outputs = [similar(first(features.channels)) for _ in 1:layer.out_channels]
    lcb!(outputs, layer, features, parameters, workspace)
    return GaugeFeatures(features.links, outputs)
end

"""
    TraceReadout(; component=:real, reduction=:sum)

Gauge-invariant reference readout. It fuses the site trace with a global
sum or mean and returns one scalar per feature channel. A future local
`Trace` layer will use a generic site-trace primitive from LatticeMatrices.
"""
struct TraceReadout{Component,Reduction} <: AbstractLCNNReadout
end

function TraceReadout(; component::Symbol=:real, reduction::Symbol=:sum)
    component in (:real, :imag, :both, :complex) || throw(ArgumentError(
        "component must be :real, :imag, :both, or :complex; got $component",
    ))
    reduction in (:sum, :mean) || throw(ArgumentError(
        "reduction must be :sum or :mean; got $reduction",
    ))
    return TraceReadout{component,reduction}()
end

@inline trace_component(::TraceReadout{Component}) where {Component} = Component
@inline trace_reduction(::TraceReadout{Component,Reduction}) where {Component,Reduction} =
    Reduction

function _trace_readout(
    ::TraceReadout{Component,Reduction},
    channels,
    _identity=nothing,
) where {Component,Reduction}
    traces = tr.(channels)
    if Reduction === :mean
        volume = prod(size(first(channels))[3:end])
        traces ./= volume
    end
    Component === :complex && return traces
    Component === :real && return real.(traces)
    Component === :imag && return imag.(traces)
    # PyTorch flattens `(channel, complex_component)` with the component axis
    # innermost: re(W1), im(W1), re(W2), im(W2), ... .  Preserve that order so
    # the final Linear weights can be copied without permutation.
    return [part for value in traces for part in (real(value), imag(value))]
end

(readout::TraceReadout)(features::GaugeFeatures) =
    _trace_readout(readout, features.channels)

"""
    LCNNFeatureModel(Val(Dim), widths...; input=Plaq(), lcb_options...)
    LCNNFeatureModel(Val(Dim), layers::Tuple; input=Plaq())
    LCNNFeatureModel(Dim, widths...; input=Plaq(), lcb_options...)
    LCNNFeatureModel(U, widths...; input=Plaq(), lcb_options...)

A framework-independent stack

```
U -> Plaq -> LCB -> ... -> GaugeFeatures
```

that deliberately stops before taking traces. `widths` gives the output
channel count of each LCB and applies common options to the whole stack. Pass a
concrete tuple of individually configured `LCB`s when layer options differ.
The plaquette input has `binomial(Dim, 2)` channels, so both constructors fix
and validate every channel boundary up front.
"""
struct LCNNFeatureModel{Dim,I,L} <: AbstractLCNNFeatureMap
    input::I
    layers::L
end

# `Val` remains an internal specialization detail.  Users may provide either
# an ordinary dimension or an existing gauge configuration from which the
# dimension is inferred.
LCNNFeatureModel(dimension::Integer, layers::Tuple; input::Plaq=Plaq()) =
    LCNNFeatureModel(Val(Int(dimension)), layers; input)

LCNNFeatureModel(
    U::Vector{T},
    layers::Tuple;
    input::Plaq=Plaq(),
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}} =
    LCNNFeatureModel(Val(Dim), layers; input)

function LCNNFeatureModel(
    ::Val{Dim},
    layers::Tuple;
    input::Plaq=Plaq(),
) where {Dim}
    Dim >= 2 || throw(ArgumentError("LCNN requires Dim >= 2; got $Dim"))
    expected_channels = binomial(Dim, 2)
    for (index, layer) in pairs(layers)
        layer isa LCB || throw(ArgumentError(
            "LCNN layer $index must currently be an LCB; got $(typeof(layer))",
        ))
        layer.in_channels == expected_channels || throw(DimensionMismatch(
            "LCNN layer $index expects $(layer.in_channels) channels, but " *
            "the preceding stage produces $expected_channels",
        ))
        expected_channels = layer.out_channels
    end
    return LCNNFeatureModel{Dim,typeof(input),typeof(layers)}(input, layers)
end

LCNNFeatureModel(
    dimension::Integer,
    widths::Vararg{Integer,N};
    kwargs...,
) where {N} = LCNNFeatureModel(Val(Int(dimension)), widths...; kwargs...)

LCNNFeatureModel(
    U::Vector{T},
    widths::Vararg{Integer,N};
    kwargs...,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim},N} =
    LCNNFeatureModel(Val(Dim), widths...; kwargs...)

"""
Preallocated fields for a complete feature model. This is the workspace used
by Enzyme paths: all lattice storage is allocated before differentiation.
"""
struct ModelWorkspace{T,R,B}
    reference::R
    feature_buffers::B
end

function ModelWorkspace(
    model::LCNNFeatureModel{Dim},
    U::Vector{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    length(U) == Dim || throw(ArgumentError(
        "a $Dim-dimensional LCNN expects $Dim link fields, got $(length(U))",
    ))
    channel_counts = (
        binomial(Dim, 2),
        (layer.out_channels for layer in model.layers)...,
    )
    buffers = map(channel_counts) do count
        [similar(first(U)) for _ in 1:count]
    end
    return ModelWorkspace{T,ReferenceWorkspace{T},typeof(buffers)}(
        ReferenceWorkspace(U), buffers,
    )
end

function LCNNFeatureModel(
    ::Val{Dim},
    widths::Vararg{Integer,N};
    input::Plaq=Plaq(),
    lcb_options...,
) where {Dim,N}
    Dim >= 2 || throw(ArgumentError("LCNN requires Dim >= 2; got $Dim"))
    all(>(0), widths) || throw(ArgumentError(
        "all LCNN channel widths must be positive; got $widths",
    ))

    input_channels = binomial(Dim, 2)
    layers = ntuple(N) do index
        output_channels = Int(widths[index])
        layer = LCB(input_channels => output_channels; lcb_options...)
        input_channels = output_channels
        return layer
    end
    return LCNNFeatureModel(Val(Dim), layers; input)
end

@inline input_channels(::LCNNFeatureModel{Dim}) where {Dim} = binomial(Dim, 2)

@inline function output_channels(model::LCNNFeatureModel{Dim}) where {Dim}
    return isempty(model.layers) ? input_channels(model) : last(model.layers).out_channels
end

function parameter_shapes(model::LCNNFeatureModel{Dim}) where {Dim}
    return (
        layers=map(model.layers) do layer
            (weight=parameter_shape(layer, Val(Dim)),)
        end,
    )
end

"""
    validate_parameters(model_or_action, parameters)

Check that every trainable leaf required by an LCNN is present, real-valued,
and has exactly the shape reported by [`parameter_shapes`](@ref).  Parameters
remain ordinary nested `NamedTuple`s, so callers may initialize, load, or edit
every value without depending on an ML framework.
"""
function validate_parameters(model::LCNNFeatureModel{Dim}, parameters) where {Dim}
    hasproperty(parameters, :layers) || throw(ArgumentError(
        "LCNN parameters must provide a `layers` tuple",
    ))
    length(parameters.layers) == length(model.layers) || throw(DimensionMismatch(
        "LCNN expects $(length(model.layers)) layer parameter groups, got " *
        "$(length(parameters.layers))",
    ))
    for (index, layer) in pairs(model.layers)
        group = parameters.layers[index]
        hasproperty(group, :weight) || throw(ArgumentError(
            "LCNN layer $index parameters must provide `weight`",
        ))
        eltype(group.weight) <: Real || throw(ArgumentError(
            "LCNN layer $index weights must be real; got $(eltype(group.weight))",
        ))
        expected = parameter_shape(layer, Val(Dim))
        size(group.weight) == expected || throw(DimensionMismatch(
            "LCNN layer $index weight shape must be $expected, got " *
            "$(size(group.weight))",
        ))
    end
    return true
end

"""Return the number of independently trainable real scalars in an LCNN."""
parameter_count(model::LCNNFeatureModel) = sum(
    prod(group.weight) for group in parameter_shapes(model).layers;
    init=0,
)

function initial_parameters(
    rng::AbstractRNG,
    model::LCNNFeatureModel{Dim},
    ::Type{T}=Float32,
) where {Dim,T<:Real}
    layers = map(model.layers) do layer
        shape = parameter_shape(layer, Val(Dim))
        # Both upstream implementations use Normal(0, init_w/sqrt(v*w)),
        # where v and w are the two bilinear basis sizes.
        fan_in = prod(shape[2:end])
        scale = T(layer.init_weight_factor) / sqrt(T(fan_in))
        (weight=scale .* randn(rng, T, shape...),)
    end
    return (; layers)
end

"""Apply only the gauge-equivariant, matrix-valued part of an LCNN."""
function forward_features(
    model::LCNNFeatureModel{Dim},
    U::Vector{T},
    parameters,
    workspace::ReferenceWorkspace{T}=ReferenceWorkspace(U),
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    hasproperty(parameters, :layers) || throw(ArgumentError(
        "LCNN feature parameters must provide a `layers` tuple",
    ))
    length(parameters.layers) == length(model.layers) || throw(DimensionMismatch(
        "LCNN expects $(length(model.layers)) parameter groups, got " *
        "$(length(parameters.layers))",
    ))

    features = plaquette_features(model.input, U, workspace)
    for index in eachindex(model.layers)
        features = lcb(
            model.layers[index],
            features,
            parameters.layers[index],
            workspace,
        )
    end
    return features
end

function _forward_feature_channels!(
    model::LCNNFeatureModel{Dim},
    U::Vector{T},
    parameters,
    workspace::ModelWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    hasproperty(parameters, :layers) || throw(ArgumentError(
        "LCNN feature parameters must provide a `layers` tuple",
    ))
    length(parameters.layers) == length(model.layers) || throw(DimensionMismatch(
        "LCNN expects $(length(model.layers)) parameter groups, got " *
        "$(length(parameters.layers))",
    ))
    length(workspace.feature_buffers) == length(model.layers) + 1 ||
        throw(DimensionMismatch("LCNN workspace does not match the model depth"))

    _plaquette_features_only!(
        workspace.feature_buffers[1], model.input, U, workspace.reference,
    )
    channels = workspace.feature_buffers[1]
    for index in eachindex(model.layers)
        next_channels = workspace.feature_buffers[index + 1]
        _lcb_only!(
            next_channels,
            model.layers[index],
            U,
            channels,
            parameters.layers[index],
            workspace.reference,
        )
        channels = next_channels
    end
    return channels
end

"""Allocation-free-in-lattice-storage feature forward pass."""
function forward_features!(
    model::LCNNFeatureModel{Dim},
    U::Vector{T},
    parameters,
    workspace::ModelWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    channels = _forward_feature_channels!(model, U, parameters, workspace)
    return GaugeFeatures(Tuple(U), channels)
end

(model::LCNNFeatureModel)(U, parameters, workspace=ReferenceWorkspace(U)) =
    forward_features(model, U, parameters, workspace)

(model::LCNNFeatureModel)(U, parameters, workspace::ModelWorkspace) =
    forward_features!(model, U, parameters, workspace)

raw"""
    LCNNAction(feature_model; component=:real, reduction=:sum)

Turn the feature part of an LCNN into a real, gauge-invariant scalar:

```math
S_\theta(U) = b + \sum_i a_i R_x\,c\!\left(\operatorname{Tr}W_{x,i}\right),
```

where `c` selects real, imaginary, or both components and `R_x` is a global
sum or mean. This is an action/readout, not a gauge-link smearing: its output
is a scalar rather than a new set of links.
"""
struct LCNNAction{M,R}
    feature_model::M
    readout::R
end

ModelWorkspace(action::LCNNAction, U) = ModelWorkspace(action.feature_model, U)

function LCNNAction(
    feature_model::LCNNFeatureModel;
    component::Symbol=:real,
    reduction::Symbol=:sum,
)
    component === :complex && throw(ArgumentError(
        "LCNNAction must be real; use component=:real, :imag, or :both",
    ))
    return LCNNAction(feature_model, TraceReadout(; component, reduction))
end

@inline function readout_channels(action::LCNNAction)
    channels = output_channels(action.feature_model)
    return trace_component(action.readout) === :both ? 2 * channels : channels
end

function parameter_shapes(action::LCNNAction)
    feature_shapes = parameter_shapes(action.feature_model)
    return (
        layers=feature_shapes.layers,
        readout=(weight=(readout_channels(action),), bias=(1,)),
    )
end

function validate_parameters(action::LCNNAction, parameters)
    validate_parameters(action.feature_model, parameters)
    hasproperty(parameters, :readout) || throw(ArgumentError(
        "LCNN action parameters must provide a `readout` group",
    ))
    readout = parameters.readout
    for name in (:weight, :bias)
        hasproperty(readout, name) || throw(ArgumentError(
            "LCNN readout parameters must provide `$name`",
        ))
        value = getproperty(readout, name)
        eltype(value) <: Real || throw(ArgumentError(
            "LCNN readout $name must be real; got $(eltype(value))",
        ))
    end
    expected_weight = (readout_channels(action),)
    size(readout.weight) == expected_weight || throw(DimensionMismatch(
        "LCNN readout weight shape must be $expected_weight, got " *
        "$(size(readout.weight))",
    ))
    size(readout.bias) == (1,) || throw(DimensionMismatch(
        "LCNN readout bias shape must be (1,), got $(size(readout.bias))",
    ))
    return true
end

parameter_count(action::LCNNAction) =
    parameter_count(action.feature_model) + readout_channels(action) + 1

"""Portable parameters and architecture metadata exported from PyTorch."""
struct PyTorchLCNNCheckpoint{M,L,R}
    metadata::M
    layer_weights::L
    readout::R
end

@inline function _checkpoint_required(arrays, key::String)
    haskey(arrays, key) || throw(ArgumentError(
        "LCNN checkpoint is missing required entry `$key`",
    ))
    return arrays[key]
end

@inline _checkpoint_integer(value) = Int(value isa AbstractArray ? only(value) : value)
@inline _checkpoint_boolean(value) = Bool(value isa AbstractArray ? only(value) : value)
@inline _checkpoint_tuple(value) = Tuple(Int.(vec(value isa AbstractArray ? value : [value])))
@inline _checkpoint_scalar(value) = value isa AbstractArray ? only(value) : value

function _checkpoint_string(value, key)
    value isa AbstractVector{UInt8} || throw(ArgumentError(
        "LCNN checkpoint entry `$key` must be UTF-8 bytes, got $(typeof(value))",
    ))
    return String(copy(value))
end

@inline function _checkpoint_convention(value::Symbol)
    value === :favoni_arxiv && return :favoni_arxiv
    value === :favoni_prl && return :favoni_prl
    # Read NPZ files written by the prototype before the public names became
    # semantic. These aliases are intentionally confined to file I/O.
    value === :paper && return :favoni_arxiv
    value === :checkpoint && return :favoni_prl
    value === :arxiv_v1 && return :favoni_arxiv
    value === :prl_2022 && return :favoni_prl
    throw(ArgumentError(
        "checkpoint convention must be `favoni_arxiv` or `favoni_prl`; got $value",
    ))
end

"""
    read_pytorch_checkpoint(path)

Read the portable NPZ representation produced by
`examples/lcnn/export_lge_checkpoint.py`. This function never evaluates a
Python pickle and does not require PyTorch.
"""
function read_pytorch_checkpoint(path::AbstractString)
    arrays = npzread(path)
    format_version = _checkpoint_integer(_checkpoint_required(
        arrays, "meta.format_version",
    ))
    format_version == 1 || throw(ArgumentError(
        "unsupported LCNN checkpoint format version $format_version",
    ))
    layer_count = _checkpoint_integer(_checkpoint_required(
        arrays, "meta.layer_count",
    ))
    layer_count > 0 || throw(ArgumentError(
        "LCNN checkpoint layer count must be positive; got $layer_count",
    ))
    layer_weights = ntuple(layer_count) do index
        Array(_checkpoint_required(arrays, "layers.$index.weight"))
    end
    readout = (
        weight=vec(Array(_checkpoint_required(arrays, "readout.weight"))),
        bias=vec(Array(_checkpoint_required(arrays, "readout.bias"))),
    )
    metadata = (
        format_version,
        source=_checkpoint_string(
            _checkpoint_required(arrays, "meta.source"), "meta.source",
        ),
        source_commit=_checkpoint_string(
            _checkpoint_required(arrays, "meta.source_commit"),
            "meta.source_commit",
        ),
        convention=_checkpoint_convention(Symbol(_checkpoint_string(
            haskey(arrays, "meta.convention") ? arrays["meta.convention"] :
                _checkpoint_required(arrays, "meta.revision"),
            haskey(arrays, "meta.convention") ?
                "meta.convention" : "meta.revision",
        ))),
        dimension=_checkpoint_integer(_checkpoint_required(
            arrays, "meta.dimension",
        )),
        colors=_checkpoint_integer(_checkpoint_required(
            arrays, "meta.colors",
        )),
        layer_count,
        conv_channels=_checkpoint_tuple(_checkpoint_required(
            arrays, "meta.conv_channels",
        )),
        kernel_size=_checkpoint_tuple(_checkpoint_required(
            arrays, "meta.kernel_size",
        )),
        dilation=_checkpoint_tuple(_checkpoint_required(
            arrays, "meta.dilation",
        )),
        symmetric=_checkpoint_boolean(_checkpoint_required(
            arrays, "meta.symmetric",
        )),
        use_unit_elements=_checkpoint_boolean(_checkpoint_required(
            arrays, "meta.use_unit_elements",
        )),
        global_average=_checkpoint_boolean(_checkpoint_required(
            arrays, "meta.global_average",
        )),
        linear_sizes=haskey(arrays, "meta.linear_sizes") ?
            _checkpoint_tuple(arrays["meta.linear_sizes"]) : (),
        complex_component_order=_checkpoint_string(
            _checkpoint_required(arrays, "meta.complex_component_order"),
            "meta.complex_component_order",
        ),
        test_mse=haskey(arrays, "meta.test_mse") ?
            _checkpoint_scalar(arrays["meta.test_mse"]) : nothing,
    )
    return PyTorchLCNNCheckpoint(metadata, layer_weights, readout)
end

function _validate_checkpoint_model(
    action::LCNNAction{<:LCNNFeatureModel{Dim}},
    checkpoint::PyTorchLCNNCheckpoint;
    strict::Bool,
) where {Dim}
    metadata = checkpoint.metadata
    metadata.dimension == Dim || throw(DimensionMismatch(
        "checkpoint dimension $(metadata.dimension) does not match model dimension $Dim",
    ))
    length(action.feature_model.layers) == metadata.layer_count ||
        throw(DimensionMismatch(
            "checkpoint has $(metadata.layer_count) LCB layers, model has " *
            "$(length(action.feature_model.layers))",
        ))
    length(metadata.conv_channels) == metadata.layer_count ||
        throw(DimensionMismatch("checkpoint conv_channels length is inconsistent"))
    length(metadata.kernel_size) == metadata.layer_count ||
        throw(DimensionMismatch("checkpoint kernel_size length is inconsistent"))
    length(metadata.dilation) == metadata.layer_count ||
        throw(DimensionMismatch("checkpoint dilation length is inconsistent"))
    isempty(metadata.linear_sizes) || throw(ArgumentError(
        "checkpoint contains hidden linear layers, which LCNNAction does not yet support",
    ))
    metadata.complex_component_order == "channel_re_im" ||
        throw(ArgumentError(
            "unsupported complex component order " *
            "$(repr(metadata.complex_component_order))",
        ))
    trace_component(action.readout) === :both || throw(ArgumentError(
        "PyTorch LGE-CNN checkpoints require component=:both",
    ))

    strict || return true
    expected_shifts = metadata.symmetric ? :both : :positive
    for (index, layer) in pairs(action.feature_model.layers)
        layer.convention === metadata.convention || throw(ArgumentError(
            "checkpoint convention $(metadata.convention) does not match layer " *
            "$index convention $(layer.convention)",
        ))
        layer.out_channels == metadata.conv_channels[index] ||
            throw(DimensionMismatch(
                "checkpoint layer $index has $(metadata.conv_channels[index]) " *
                "outputs, model expects $(layer.out_channels)",
            ))
        layer.kernel_size == metadata.kernel_size[index] ||
            throw(ArgumentError(
                "checkpoint layer $index kernel_size=$(metadata.kernel_size[index]), " *
                "model uses $(layer.kernel_size)",
            ))
        layer.dilation == metadata.dilation[index] || throw(ArgumentError(
            "checkpoint layer $index dilation=$(metadata.dilation[index]), " *
            "model uses $(layer.dilation)",
        ))
        layer.shifts === expected_shifts || throw(ArgumentError(
            "checkpoint layer $index shifts=$expected_shifts, model uses $(layer.shifts)",
        ))
        layer.include_identity == metadata.use_unit_elements ||
            throw(ArgumentError(
                "checkpoint and model disagree about unit elements in layer $index",
            ))
        layer.include_adjoints || throw(ArgumentError(
            "PyTorch LGE-CNN checkpoint requires adjoint channels in layer $index",
        ))
    end
    return true
end

"""
    load_pytorch_parameters(action, checkpoint_or_path;
                            parameter_type=nothing, strict=true)

Convert a portable PyTorch checkpoint to the ordinary nested `NamedTuple`
used by LCNN, checking architecture and basis conventions before returning.
"""
function load_pytorch_parameters(
    action::LCNNAction,
    checkpoint::PyTorchLCNNCheckpoint;
    parameter_type::Union{Nothing,Type{<:Real}}=nothing,
    strict::Bool=true,
)
    _validate_checkpoint_model(action, checkpoint; strict)
    T = parameter_type === nothing ?
        eltype(first(checkpoint.layer_weights)) : parameter_type
    T <: Real || throw(ArgumentError("checkpoint parameter type must be real"))
    layers = map(checkpoint.layer_weights) do weight
        (weight=T.(weight),)
    end
    parameters = (
        layers,
        readout=(
            weight=T.(checkpoint.readout.weight),
            bias=T.(checkpoint.readout.bias),
        ),
    )
    validate_parameters(action, parameters)
    all(isfinite, Iterators.flatten((
        (group.weight for group in parameters.layers)...,
        parameters.readout.weight,
        parameters.readout.bias,
    ))) || throw(ArgumentError("checkpoint parameters contain NaN or Inf"))
    return parameters
end

function load_pytorch_parameters(
    action::LCNNAction,
    path::AbstractString;
    kwargs...,
)
    return load_pytorch_parameters(
        action, read_pytorch_checkpoint(path); kwargs...,
    )
end

"""Build the framework-independent LCNN architecture recorded in a checkpoint."""
function model_from_pytorch_checkpoint(
    U::Vector{T},
    checkpoint::PyTorchLCNNCheckpoint;
    reduction::Symbol=:mean,
    strict::Bool=true,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    metadata = checkpoint.metadata
    metadata.dimension == Dim || throw(DimensionMismatch(
        "checkpoint dimension $(metadata.dimension) does not match U dimension $Dim",
    ))
    metadata.colors == NC || throw(DimensionMismatch(
        "checkpoint has SU($(metadata.colors)), while U has SU($NC)",
    ))
    isempty(metadata.linear_sizes) || throw(ArgumentError(
        "checkpoint contains hidden linear layers, which are not yet supported",
    ))
    input_count = binomial(Dim, 2)
    layers = ntuple(metadata.layer_count) do index
        layer = LCB(
            input_count => metadata.conv_channels[index];
            convention=metadata.convention,
            kernel_size=metadata.kernel_size[index],
            dilation=metadata.dilation[index],
            shifts=metadata.symmetric ? :both : :positive,
            identity=metadata.use_unit_elements,
            adjoints=true,
        )
        input_count = layer.out_channels
        return layer
    end
    action = LCNNAction(
        LCNNFeatureModel(U, layers); component=:both, reduction,
    )
    _validate_checkpoint_model(action, checkpoint; strict)
    return action
end

function model_from_pytorch_checkpoint(
    U,
    path::AbstractString;
    kwargs...,
)
    return model_from_pytorch_checkpoint(
        U, read_pytorch_checkpoint(path); kwargs...,
    )
end

"""
    load_pytorch_model(U, path; parameter_type=nothing,
                       reduction=:mean, strict=true)

Construct the recorded LCNN and return `(model, parameters, metadata)`.
"""
function load_pytorch_model(
    U,
    path::AbstractString;
    parameter_type::Union{Nothing,Type{<:Real}}=nothing,
    reduction::Symbol=:mean,
    strict::Bool=true,
)
    checkpoint = read_pytorch_checkpoint(path)
    model = model_from_pytorch_checkpoint(
        U, checkpoint; reduction, strict,
    )
    parameters = load_pytorch_parameters(
        model, checkpoint; parameter_type, strict,
    )
    return model, parameters, checkpoint.metadata
end

"""
    favoni2022_wilson_1x2_small(; convention=:favoni_prl, reduction=:mean)

Build the 1+1-dimensional small L-CNN topology used for the `1 x 2` Wilson
loop regression in Favoni et al., *Phys. Rev. Lett.* **128**, 032003 (2022):
one positive-shift L-CB with one plaquette input and two outputs, followed by
real/imaginary trace channels and a scalar linear readout.

Use `convention=:favoni_arxiv` for the 35-parameter model reported in
supplemental Table V. Use `convention=:favoni_prl` for the 47-parameter model
found in the released result checkpoints. The two conventions intentionally have distinct
transported bases and dilation semantics.
"""
function favoni2022_wilson_1x2_small(
    U=nothing;
    convention::Symbol=:favoni_prl,
    reduction::Symbol=:mean,
)
    convention in (:favoni_arxiv, :favoni_prl) || throw(ArgumentError(
        "convention must be :favoni_arxiv or :favoni_prl; got $convention",
    ))
    layer = LCB(
        1 => 2;
        convention,
        kernel_size=2,
        dilation=convention === :favoni_arxiv ? 0 : 1,
        shifts=:positive,
        identity=true,
        adjoints=true,
    )
    model = LCNNFeatureModel(
        U === nothing ? 2 : U,
        (layer,),
    )
    return LCNNAction(model; component=:both, reduction)
end

function initial_parameters(
    rng::AbstractRNG,
    action::LCNNAction,
    ::Type{T}=Float32,
) where {T<:Real}
    feature_parameters = initial_parameters(rng, action.feature_model, T)
    count = readout_channels(action)
    readout = (
        weight=inv(sqrt(T(count))) .* randn(rng, T, count),
        bias=zeros(T, 1),
    )
    parameters = (; feature_parameters.layers, readout)
    validate_parameters(action, parameters)
    return parameters
end

function (action::LCNNAction)(
    U::Vector{T},
    parameters,
    workspace::ReferenceWorkspace{T}=ReferenceWorkspace(U),
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    hasproperty(parameters, :readout) || throw(ArgumentError(
        "LCNN action parameters must provide a `readout` group",
    ))
    readout_parameters = parameters.readout
    hasproperty(readout_parameters, :weight) || throw(ArgumentError(
        "LCNN readout parameters must provide `weight`",
    ))
    hasproperty(readout_parameters, :bias) || throw(ArgumentError(
        "LCNN readout parameters must provide `bias`",
    ))
    eltype(readout_parameters.weight) <: Real || throw(ArgumentError(
        "LCNN readout weights must be real",
    ))
    eltype(readout_parameters.bias) <: Real || throw(ArgumentError(
        "LCNN readout bias must be real",
    ))
    length(readout_parameters.bias) == 1 || throw(DimensionMismatch(
        "LCNN readout bias must contain one value, got " *
        "$(length(readout_parameters.bias))",
    ))

    features = forward_features(
        action.feature_model,
        U,
        (layers=parameters.layers,),
        workspace,
    )
    invariants = action.readout(features)
    length(readout_parameters.weight) == length(invariants) ||
        throw(DimensionMismatch(
            "LCNN readout expects $(length(invariants)) weights, got " *
            "$(length(readout_parameters.weight))",
        ))
    return dot(readout_parameters.weight, invariants) + only(readout_parameters.bias)
end


function (action::LCNNAction)(
    U::Vector{T},
    parameters,
    workspace::ModelWorkspace{T},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    hasproperty(parameters, :readout) || throw(ArgumentError(
        "LCNN action parameters must provide a `readout` group",
    ))
    channels = _forward_feature_channels!(
        action.feature_model,
        U,
        (layers=parameters.layers,),
        workspace,
    )
    # Pass the preallocated identity through the Enzyme boundary.  The primal
    # readout does not need it, but its analytic reverse rule uses it to add a
    # diagonal cotangent without differentiating a backend reduction kernel.
    invariants = _trace_readout(
        action.readout, channels, workspace.reference.identity,
    )
    readout_parameters = parameters.readout
    length(readout_parameters.weight) == length(invariants) ||
        throw(DimensionMismatch(
            "LCNN readout expects $(length(invariants)) weights, got " *
            "$(length(readout_parameters.weight))",
        ))
    length(readout_parameters.bias) == 1 || throw(DimensionMismatch(
        "LCNN readout bias must contain one value, got " *
        "$(length(readout_parameters.bias))",
    ))
    return dot(readout_parameters.weight, invariants) + only(readout_parameters.bias)
end

@inline function _site_trace(channel, site::NTuple, ::Val{NC}) where {NC}
    value = zero(eltype(channel))
    for color in 1:NC
        value += channel[color, color, site...]
    end
    return value
end

@inline function _site_prediction(
    ::TraceReadout{Component},
    channels,
    readout_parameters,
    site::NTuple,
    colors::Val,
) where {Component}
    value = only(readout_parameters.bias)
    weight_index = 1
    for channel in channels
        trace_value = _site_trace(channel, site, colors)
        if Component === :real
            value += readout_parameters.weight[weight_index] * real(trace_value)
            weight_index += 1
        elseif Component === :imag
            value += readout_parameters.weight[weight_index] * imag(trace_value)
            weight_index += 1
        elseif Component === :both
            value += readout_parameters.weight[weight_index] * real(trace_value)
            value += readout_parameters.weight[weight_index + 1] * imag(trace_value)
            weight_index += 2
        else
            error("site-local LCNN regression requires a real readout")
        end
    end
    return value
end

function _validate_site_target(target, U, ::Val{Dim}) where {Dim}
    ndims(target) == Dim || throw(DimensionMismatch(
        "a site-local target for a $Dim-dimensional model must have $Dim axes",
    ))
    lattice = Tuple(size(first(U))[3:end])
    size(target) == lattice || throw(DimensionMismatch(
        "target shape $(size(target)) does not match lattice $lattice",
    ))
    return lattice
end

"""
    site_predictions(action, U, parameters[, workspace])

Return the scalar prediction at every site before global averaging. This is
the output used by the Favoni et al. training loss. The trace is deliberately
not divided by `NC`; the learned final linear layer acts on the same
unnormalized `LTrace` features as the reference PyTorch model.
"""
function site_predictions(
    action::LCNNAction{<:LCNNFeatureModel{Dim}},
    U::Vector{T},
    parameters,
    workspace::ModelWorkspace{T}=ModelWorkspace(action, U),
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    validate_parameters(action, parameters)
    channels = _forward_feature_channels!(
        action.feature_model,
        U,
        (layers=parameters.layers,),
        workspace,
    )
    lattice = Tuple(size(first(U))[3:end])
    value_type = promote_type(
        eltype(parameters.readout.weight),
        typeof(real(zero(eltype(first(channels))))),
    )
    output = Array{value_type}(undef, lattice)
    colors = Val(NC)
    for site_index in CartesianIndices(output)
        site = Tuple(site_index)
        output[site_index] = _site_prediction(
            action.readout, channels, parameters.readout, site, colors,
        )
    end
    return output
end

"""
    site_mse_loss(parameters, action, U, workspace, target)

Mean squared error over all lattice sites, matching the training loss of the
released LGE-CNN implementation when `global_average=false`. The argument
order keeps the parameter tree first for Enzyme's reverse pass.
"""
function site_mse_loss(
    parameters,
    action::LCNNAction{<:LCNNFeatureModel{Dim}},
    U::Vector{T},
    workspace::ModelWorkspace{T},
    target::AbstractArray{<:Real},
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    _validate_site_target(target, U, Val(Dim))
    channels = _forward_feature_channels!(
        action.feature_model,
        U,
        (layers=parameters.layers,),
        workspace,
    )
    loss = zero(promote_type(
        eltype(parameters.readout.weight), eltype(target),
    ))
    colors = Val(NC)
    for site_index in CartesianIndices(target)
        site = Tuple(site_index)
        prediction = _site_prediction(
            action.readout, channels, parameters.readout, site, colors,
        )
        residual = prediction - target[site_index]
        loss += residual * residual
    end
    return loss / length(target)
end

site_mse_loss(action::LCNNAction, U, parameters, target; workspace=ModelWorkspace(action, U)) =
    site_mse_loss(parameters, action, U, workspace, target)

"""
    global_average_mse(action, U, parameters, target[, workspace])

MSE after separately averaging the prediction and target over sites. This is
the metric used by the released code for its final test report; it is not the
site-local optimization loss.
"""
function global_average_mse(
    action::LCNNAction,
    U,
    parameters,
    target::AbstractArray{<:Real},
    workspace::ModelWorkspace=ModelWorkspace(action, U),
)
    prediction = sum(site_predictions(action, U, parameters, workspace)) /
        length(target)
    target_mean = sum(target) / length(target)
    residual = prediction - target_mean
    return residual * residual
end

include("lexp.jl")
include("smearing.jl")
include("luxcore.jl")

end
