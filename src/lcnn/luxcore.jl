"""
    LuxLCNN(model; parameter_type=Float32)

LuxCore adapter around a complete framework-independent LCNN model or action.
Calling the layer with `U` allocates its lattice workspace on each call. Pass
`(U, workspace)` during Enzyme training to keep all lattice allocation outside
the differentiated region.
"""
struct LuxLCNN{M,T<:Real} <: LuxCore.AbstractLuxLayer
    model::M
    parameter_type::Type{T}
end

"""
    lux_layer(model; parameter_type=Float32)

Wrap an LCNN feature model or scalar action as a LuxCore layer. LuxCore is a
normal Gaugefields dependency, so this adapter is available immediately after
`using Gaugefields`; loading `LuxCore` separately is not an activation step.
"""
function lux_layer(
    model::Union{LCNNFeatureModel,LCNNAction};
    parameter_type::Type{T}=Float32,
) where {T<:Real}
    return LuxLCNN{typeof(model),T}(model, parameter_type)
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::LuxLCNN)
    return initial_parameters(rng, layer.model, layer.parameter_type)
end

LuxCore.initialstates(::AbstractRNG, ::LuxLCNN) = NamedTuple()

function (layer::LuxLCNN)(U, parameters, state)
    workspace = ModelWorkspace(layer.model, U)
    return layer.model(U, parameters, workspace), state
end

function (layer::LuxLCNN)(
    input::Tuple{U,W},
    parameters,
    state,
) where {U,W<:ModelWorkspace}
    links, workspace = input
    return layer.model(links, parameters, workspace), state
end
