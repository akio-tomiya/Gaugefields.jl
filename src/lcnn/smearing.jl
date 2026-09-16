import ..Gaugefields:
    has_smearing_pullback,
    link_smear_pullback,
    link_smear_pullback!,
    smear
import ..Abstractsmearing_module: Abstractsmearing

"""
    LCNNLinkSmearing(model::LCNNLinkModel, parameters)
    lcnn_smearing(model::LCNNLinkModel, parameters)

Gaugefields smearing specification for an LCNN ending in `LExp`.
The model implements a genuine link map `U -> Unew`; feature-only LCNN models
are intentionally not accepted by the `smear` API.
"""
struct LCNNLinkSmearing{M<:LCNNLinkModel,P} <: Abstractsmearing
    model::M
    parameters::P
    function LCNNLinkSmearing(model::M, parameters::P) where {
        M<:LCNNLinkModel,P
    }
        validate_parameters(model, parameters)
        return new{M,P}(model, parameters)
    end
end

"""Construct an `LCNNLinkSmearing` for use with `Gaugefields.smear`."""
lcnn_smearing(model::LCNNLinkModel, parameters) =
    LCNNLinkSmearing(model, parameters)

has_smearing_pullback(::LCNNLinkSmearing) = true

"""Forward workspace and bound model/parameters needed by the LCNN VJP."""
struct LCNNLinkSmearingCache{S,W}
    smearing::S
    workspace::W
end

LCNNLinkSmearingCache(U, smearing::LCNNLinkSmearing) =
    LCNNLinkSmearingCache(smearing, LinkModelWorkspace(smearing.model, U))

"""
    LCNNLinkPullback(source, cache)

Callable reverse map returned in `smear(...; record=true, calcdSdU=true)`.
Calling it with an output-link cotangent returns the corresponding input-link
cotangent. Loading Enzyme activates the implementation.
"""
struct LCNNLinkPullback{U,C}
    source::U
    cache::C
end

(pullback::LCNNLinkPullback)(output_cotangent) = link_smear_pullback(
    output_cotangent, pullback.source, pullback.cache,
)

function _require_lcnn_enzyme_extension()
    Base.get_extension(parentmodule(@__MODULE__), :GaugefieldsEnzymeExt) !==
        nothing && return nothing
    throw(ArgumentError(
        "LCNN link pullbacks require Enzyme; load it with `using Enzyme` " *
        "before applying the recorded derivative",
    ))
end

function link_smear_pullback!(
    input_cotangent::Vector{T},
    output_cotangent::Vector{T},
    U::Vector{T},
    cache::LCNNLinkSmearingCache,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    _require_lcnn_enzyme_extension()
    return link_model_pullback!(
        input_cotangent,
        nothing,
        output_cotangent,
        cache.smearing.model,
        U,
        cache.smearing.parameters,
        cache.workspace,
    )
end

function link_smear_pullback(
    output_cotangent::Vector{T},
    U::Vector{T},
    cache::LCNNLinkSmearingCache,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    input_cotangent = [similar(first(U)) for _ in 1:Dim]
    return link_smear_pullback!(
        input_cotangent, output_cotangent, U, cache,
    )
end

"""
    smear(U, smearing::LCNNLinkSmearing;
          record=false, calcdSdU=false, temps=nothing)

Apply `Plaq -> LCNN feature layers -> LExp` through the ordinary Gaugefields
link-smearing API. The default return value is a vector of updated gauge links.
With `record=true`, return `(configuration, history, derivative)`, where
`history` is a reusable `LCNNLinkSmearingCache`. If `calcdSdU=true`,
`derivative` is a callable vector--Jacobian product accepting an output-link
cotangent; otherwise it is `nothing`.

Passing `history` back as `temps` reuses all lattice-sized storage and
overwrites the earlier returned configuration owned by that workspace.
"""
function smear(
    U::Vector{T},
    smearing::LCNNLinkSmearing;
    record::Bool=false,
    calcdSdU::Bool=false,
    temps=nothing,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    smearing.model isa LCNNLinkModel{Dim} || throw(DimensionMismatch(
        "LCNN link-smearing dimension does not match the " *
        "$Dim-dimensional input",
    ))
    cache = if temps === nothing
        LCNNLinkSmearingCache(U, smearing)
    elseif temps isa LCNNLinkSmearingCache
        typeof(temps.smearing.model) === typeof(smearing.model) ||
            throw(ArgumentError(
                "the LCNN smearing cache belongs to a different model",
            ))
        LCNNLinkSmearingCache(smearing, temps.workspace)
    elseif temps isa LinkModelWorkspace{T}
        LCNNLinkSmearingCache(smearing, temps)
    else
        throw(ArgumentError(
            "temps must be nothing, an LCNNLinkSmearingCache, or an " *
            "LCNN.LinkModelWorkspace compatible with the input gauge fields",
        ))
    end
    configuration = forward_links!(
        smearing.model,
        U,
        smearing.parameters,
        cache.workspace,
    )
    derivative = calcdSdU ? LCNNLinkPullback(U, cache) : nothing
    return record ?
        (; configuration, history=cache, derivative) :
        configuration
end
