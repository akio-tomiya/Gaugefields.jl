import LatticeMatrices
import .AbstractGaugefields_module:
    AbstractGaugefields,
    Gaugefields_4D_MPILattice

"""
    NHYPSmearing(; alpha_outer=0.5, alpha_middle=0.5, alpha_inner=0.4)

Parameters for QEX-compatible normalized HYP smearing. The names describe the
geometric nesting levels; QEX's `(alpha1, alpha2, alpha3)` corresponds to
`(alpha_inner, alpha_middle, alpha_outer)`.
"""
struct NHYPSmearing{P} <: Abstractsmearing_module.Abstractsmearing
    parameters::P
end

NHYPSmearing(parameters::LatticeMatrices.NHYPParameters) =
    NHYPSmearing{typeof(parameters)}(parameters)

function NHYPSmearing(;
    alpha_outer::Real=0.5,
    alpha_middle::Real=0.5,
    alpha_inner::Real=0.4,
)
    return NHYPSmearing(LatticeMatrices.NHYPParameters(;
        alpha_outer,
        alpha_middle,
        alpha_inner,
    ))
end

"""
    NHYPSmearingCache(U, smearing=NHYPSmearing())

Reusable LatticeMatrices nHYP forward and pullback workspace for a
Gaugefields configuration. One cache must not be used concurrently by
multiple tasks.
"""
struct NHYPSmearingCache{C}
    lattice_cache::C
end

function _validate_nhyp_configuration(
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    label::AbstractString,
)
    length(U) == 4 || throw(ArgumentError(
        "$label must contain four gauge-link directions",
    ))
    return U
end

function _validate_nhyp_configuration(
    U::AbstractVector{<:AbstractGaugefields},
    label::AbstractString,
)
    throw(ArgumentError(
        "$label must use the four-dimensional LatticeMatrices backend; " *
        "legacy and non-4D gauge fields are not supported by nHYP smearing",
    ))
end

function _nhyp_lattice_links(
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    label::AbstractString,
)
    _validate_nhyp_configuration(U, label)
    return [link.U for link in U]
end

function NHYPSmearingCache(
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    smearing::NHYPSmearing=NHYPSmearing(),
)
    links = _nhyp_lattice_links(U, "U")
    return NHYPSmearingCache(
        LatticeMatrices.NHYPSmearingCache4D(links, smearing.parameters),
    )
end

function NHYPSmearingCache(
    U::AbstractVector{<:AbstractGaugefields},
    smearing::NHYPSmearing=NHYPSmearing(),
)
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end

"""
    nhyp_smearing(U; alpha_outer=0.5, alpha_middle=0.5, alpha_inner=0.4)

Construct a normalized HYP smearing specification for a four-dimensional
LatticeMatrices-backed Gaugefields configuration.
"""
function nhyp_smearing(
    U::AbstractVector{<:AbstractGaugefields};
    alpha_outer::Real=0.5,
    alpha_middle::Real=0.5,
    alpha_inner::Real=0.4,
)
    _validate_nhyp_configuration(U, "U")
    return NHYPSmearing(; alpha_outer, alpha_middle, alpha_inner)
end

"""
    nhyp_smear!(smeared, U, cache)

Apply nHYP smearing into a preallocated Gaugefields configuration and retain
the forward intermediates in `cache` for [`nhyp_pullback!`](@ref).
"""
function nhyp_smear!(
    smeared::AbstractVector{<:Gaugefields_4D_MPILattice},
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    cache::NHYPSmearingCache,
)
    output_links = _nhyp_lattice_links(smeared, "smeared")
    input_links = _nhyp_lattice_links(U, "U")
    LatticeMatrices.nhyp_smear!(
        output_links,
        input_links,
        cache.lattice_cache,
    )
    return smeared
end

function nhyp_smear!(
    smeared::AbstractVector{<:AbstractGaugefields},
    U::AbstractVector{<:AbstractGaugefields},
    cache::NHYPSmearingCache,
)
    _validate_nhyp_configuration(smeared, "smeared")
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end

"""
    nhyp_smear(U, smearing=NHYPSmearing())

Allocate and return `(smeared, cache)`. The cache belongs to this forward
pass and can be passed to [`nhyp_pullback!`](@ref).
"""
function nhyp_smear(
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    smearing::NHYPSmearing=NHYPSmearing(),
)
    smeared = similar(U)
    cache = NHYPSmearingCache(U, smearing)
    nhyp_smear!(smeared, U, cache)
    return smeared, cache
end

function nhyp_smear(
    U::AbstractVector{<:AbstractGaugefields},
    smearing::NHYPSmearing=NHYPSmearing(),
)
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end

"""
    nhyp_pullback!(dU, dsmeared, U, cache)

Apply the analytic nHYP reverse pass. `dsmeared` is the cotangent of the
smeared links, and `dU` is overwritten with the corresponding unconstrained
thin-link cotangent. Projection to the Lie algebra, if required by an HMC
integrator, remains the caller's responsibility.
"""
function nhyp_pullback!(
    dU::AbstractVector{<:Gaugefields_4D_MPILattice},
    dsmeared::AbstractVector{<:Gaugefields_4D_MPILattice},
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    cache::NHYPSmearingCache,
)
    dthin_links = _nhyp_lattice_links(dU, "dU")
    dsmeared_links = _nhyp_lattice_links(dsmeared, "dsmeared")
    thin_links = _nhyp_lattice_links(U, "U")
    LatticeMatrices.nhyp_pullback!(
        dthin_links,
        dsmeared_links,
        thin_links,
        cache.lattice_cache,
    )
    return dU
end

function nhyp_pullback!(
    dU::AbstractVector{<:AbstractGaugefields},
    dsmeared::AbstractVector{<:AbstractGaugefields},
    U::AbstractVector{<:AbstractGaugefields},
    cache::NHYPSmearingCache,
)
    _validate_nhyp_configuration(dU, "dU")
    _validate_nhyp_configuration(dsmeared, "dsmeared")
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end

"""Allocate the thin-link cotangent for a cached nHYP forward pass."""
function nhyp_pullback(
    dsmeared::AbstractVector{<:Gaugefields_4D_MPILattice},
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    cache::NHYPSmearingCache,
)
    dU = similar(U)
    nhyp_pullback!(dU, dsmeared, U, cache)
    return dU
end


function nhyp_pullback(
    dsmeared::AbstractVector{<:AbstractGaugefields},
    U::AbstractVector{<:AbstractGaugefields},
    cache::NHYPSmearingCache,
)
    _validate_nhyp_configuration(dsmeared, "dsmeared")
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end
