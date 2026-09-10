import LatticeMatrices
import .AbstractGaugefields_module:
    AbstractGaugefields,
    Gaugefields_4D_MPILattice

abstract type NativeLinkSmearing <: Abstractsmearing_module.Abstractsmearing end

function _validate_smearing_iterations(iterations::Integer)
    iterations >= 1 || throw(ArgumentError(
        "smearing iterations must be positive; got $iterations",
    ))
    return Int(iterations)
end

"""
    NHYPSmearing(; alpha_outer=0.5, alpha_middle=0.5, alpha_inner=0.4,
                  iterations=1)

Parameters for QEX-compatible normalized HYP smearing. The names describe the
geometric nesting levels; QEX's `(alpha1, alpha2, alpha3)` corresponds to
`(alpha_inner, alpha_middle, alpha_outer)`.
"""
struct NHYPSmearing{P} <: NativeLinkSmearing
    parameters::P
    iterations::Int
end

NHYPSmearing{P}(parameters::P) where P = NHYPSmearing{P}(parameters, 1)

NHYPSmearing(parameters::LatticeMatrices.NHYPParameters) =
    NHYPSmearing(parameters, 1)

NHYPSmearing(parameters::LatticeMatrices.NHYPParameters, iterations::Int) =
    NHYPSmearing{typeof(parameters)}(
        parameters,
        _validate_smearing_iterations(iterations),
    )

NHYPSmearing(parameters::LatticeMatrices.NHYPParameters, iterations::Integer) =
    NHYPSmearing(parameters, _validate_smearing_iterations(iterations))

function NHYPSmearing(;
    alpha_outer::Real=0.5,
    alpha_middle::Real=0.5,
    alpha_inner::Real=0.4,
    iterations::Integer=1,
)
    return NHYPSmearing(
        LatticeMatrices.NHYPParameters(;
            alpha_outer,
            alpha_middle,
            alpha_inner,
        ),
        iterations,
    )
end

"""One or more analytic stout/EXP smearing steps."""
struct StoutSmearing{P} <: NativeLinkSmearing
    parameters::P
    iterations::Int
end

StoutSmearing(parameters::LatticeMatrices.StoutParameters, iterations::Int=1) =
    StoutSmearing{typeof(parameters)}(
        parameters,
        _validate_smearing_iterations(iterations),
    )

StoutSmearing(parameters::LatticeMatrices.StoutParameters, iterations::Integer) =
    StoutSmearing(parameters, _validate_smearing_iterations(iterations))

StoutSmearing(; rho::Real=0.1, iterations::Integer=1) =
    StoutSmearing(LatticeMatrices.StoutParameters(rho), iterations)

const EXPSmearing = StoutSmearing

"""One or more analytic HEX smearing steps."""
struct HEXSmearing{P} <: NativeLinkSmearing
    parameters::P
    iterations::Int
end

HEXSmearing(parameters::LatticeMatrices.HEXParameters, iterations::Int=1) =
    HEXSmearing{typeof(parameters)}(
        parameters,
        _validate_smearing_iterations(iterations),
    )

HEXSmearing(parameters::LatticeMatrices.HEXParameters, iterations::Integer) =
    HEXSmearing(parameters, _validate_smearing_iterations(iterations))

function HEXSmearing(;
    alpha_outer::Real=0.125,
    alpha_middle::Real=0.15,
    alpha_inner::Real=0.15,
    iterations::Integer=1,
)
    return HEXSmearing(
        LatticeMatrices.HEXParameters(;
            alpha_outer,
            alpha_middle,
            alpha_inner,
        ),
        iterations,
    )
end

"""One or more APE steps with MaxReTr or differentiable polar projection."""
struct APESmearing{P} <: NativeLinkSmearing
    parameters::P
    iterations::Int
end

APESmearing(parameters::LatticeMatrices.APEParameters, iterations::Int=1) =
    APESmearing{typeof(parameters)}(
        parameters,
        _validate_smearing_iterations(iterations),
    )

APESmearing(parameters::LatticeMatrices.APEParameters, iterations::Integer) =
    APESmearing(parameters, _validate_smearing_iterations(iterations))

function APESmearing(;
    alpha::Real=0.6,
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
    iterations::Integer=1,
)
    return APESmearing(
        LatticeMatrices.APEParameters(
            alpha;
            projection,
            max_retr_iterations,
            max_retr_tolerance,
        ),
        iterations,
    )
end

"""One or more HYP steps with MaxReTr or differentiable polar projection."""
struct HYPSmearing{P} <: NativeLinkSmearing
    parameters::P
    iterations::Int
end

HYPSmearing(parameters::LatticeMatrices.HYPParameters, iterations::Int=1) =
    HYPSmearing{typeof(parameters)}(
        parameters,
        _validate_smearing_iterations(iterations),
    )

HYPSmearing(parameters::LatticeMatrices.HYPParameters, iterations::Integer) =
    HYPSmearing(parameters, _validate_smearing_iterations(iterations))

function HYPSmearing(;
    alpha_outer::Real=0.75,
    alpha_middle::Real=0.6,
    alpha_inner::Real=0.3,
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
    iterations::Integer=1,
)
    return HYPSmearing(
        LatticeMatrices.HYPParameters(;
            alpha_outer,
            alpha_middle,
            alpha_inner,
            projection,
            max_retr_iterations,
            max_retr_tolerance,
        ),
        iterations,
    )
end

function _lm_smearing_specification(smearing::NativeLinkSmearing)
    return smearing.iterations == 1 ?
        smearing.parameters :
        LatticeMatrices.IteratedSmearing(
            smearing.parameters,
            smearing.iterations,
        )
end

has_smearing_pullback(::NativeLinkSmearing) = false
has_smearing_pullback(
    ::Union{NHYPSmearing,StoutSmearing,HEXSmearing},
) = true
has_smearing_pullback(smearing::Union{APESmearing,HYPSmearing}) =
    smearing.parameters.projection === :polar

function _smearing_pullback_error(
    smearing::Union{APESmearing,HYPSmearing},
)
    name = nameof(typeof(smearing))
    projection = repr(smearing.parameters.projection)
    return "$name pullback is unavailable for projection=$projection; " *
           "use $name(...; projection=:polar) to enable the analytic pullback"
end

function _smearing_pullback_error(smearing::NativeLinkSmearing)
    name = nameof(typeof(smearing))
    return "$name does not provide a link-smearing pullback"
end

"""
    LinkSmearingCache(U, smearing=NHYPSmearing())

Reusable LatticeMatrices workspace for native nHYP, stout/EXP, HEX, APE, and
HYP smearing. One cache must not be used concurrently by multiple tasks.
`NHYPSmearingCache` remains as a compatibility name for the same type.
"""
struct NHYPSmearingCache{C}
    lattice_cache::C
end

const LinkSmearingCache = NHYPSmearingCache

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
        "legacy and non-4D gauge fields are not supported by native link smearing",
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
    smearing::NativeLinkSmearing=NHYPSmearing(),
)
    links = _nhyp_lattice_links(U, "U")
    return NHYPSmearingCache(
        LatticeMatrices.smearing_cache(
            links,
            _lm_smearing_specification(smearing),
        ),
    )
end

function NHYPSmearingCache(
    U::AbstractVector{<:AbstractGaugefields},
    smearing::NativeLinkSmearing=NHYPSmearing(),
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
    iterations::Integer=1,
)
    _validate_nhyp_configuration(U, "U")
    return NHYPSmearing(;
        alpha_outer,
        alpha_middle,
        alpha_inner,
        iterations,
    )
end

function stout_link_smearing(
    U::AbstractVector{<:AbstractGaugefields};
    rho::Real=0.1,
    iterations::Integer=1,
)
    _validate_nhyp_configuration(U, "U")
    return StoutSmearing(; rho, iterations)
end

exp_smearing(U::AbstractVector{<:AbstractGaugefields}; kwargs...) =
    stout_link_smearing(U; kwargs...)

function hex_smearing(
    U::AbstractVector{<:AbstractGaugefields};
    alpha_outer::Real=0.125,
    alpha_middle::Real=0.15,
    alpha_inner::Real=0.15,
    iterations::Integer=1,
)
    _validate_nhyp_configuration(U, "U")
    return HEXSmearing(;
        alpha_outer,
        alpha_middle,
        alpha_inner,
        iterations,
    )
end

function ape_smearing(
    U::AbstractVector{<:AbstractGaugefields};
    alpha::Real=0.6,
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
    iterations::Integer=1,
)
    _validate_nhyp_configuration(U, "U")
    return APESmearing(;
        alpha,
        projection,
        max_retr_iterations,
        max_retr_tolerance,
        iterations,
    )
end

function hyp_smearing(
    U::AbstractVector{<:AbstractGaugefields};
    alpha_outer::Real=0.75,
    alpha_middle::Real=0.6,
    alpha_inner::Real=0.3,
    projection::Symbol=:max_retr,
    max_retr_iterations::Integer=1000,
    max_retr_tolerance::Real=1e-14,
    iterations::Integer=1,
)
    _validate_nhyp_configuration(U, "U")
    return HYPSmearing(;
        alpha_outer,
        alpha_middle,
        alpha_inner,
        projection,
        max_retr_iterations,
        max_retr_tolerance,
        iterations,
    )
end

"""
    link_smear!(smeared, U, cache)

Apply a native link-smearing specification into a preallocated Gaugefields
configuration and retain the forward intermediates for the pullback.
"""
function link_smear!(
    smeared::AbstractVector{<:Gaugefields_4D_MPILattice},
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    cache::NHYPSmearingCache,
)
    output_links = _nhyp_lattice_links(smeared, "smeared")
    input_links = _nhyp_lattice_links(U, "U")
    LatticeMatrices.smear_links!(
        output_links,
        input_links,
        cache.lattice_cache,
    )
    return smeared
end

function link_smear!(
    smeared::AbstractVector{<:AbstractGaugefields},
    U::AbstractVector{<:AbstractGaugefields},
    cache::NHYPSmearingCache,
)
    _validate_nhyp_configuration(smeared, "smeared")
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end

nhyp_smear!(smeared, U, cache::NHYPSmearingCache) =
    link_smear!(smeared, U, cache)

"""
    nhyp_smear(U, smearing=NHYPSmearing())

Allocate and return `(smeared, cache)`. The cache belongs to this forward
pass and can be passed to [`nhyp_pullback!`](@ref).
"""
function nhyp_smear(
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    smearing::NHYPSmearing=NHYPSmearing(),
)
    return link_smear(U, smearing)
end

function link_smear(
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    smearing::NativeLinkSmearing,
)
    smeared = similar(U)
    cache = NHYPSmearingCache(U, smearing)
    link_smear!(smeared, U, cache)
    return smeared, cache
end

function link_smear(
    U::AbstractVector{<:AbstractGaugefields},
    smearing::NativeLinkSmearing,
)
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end

function nhyp_smear(
    U::AbstractVector{<:AbstractGaugefields},
    smearing::NHYPSmearing=NHYPSmearing(),
)
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
end

"""
    link_smear_pullback!(dU, dsmeared, U, cache)

Apply an analytic native-smearing reverse pass. `dsmeared` is the cotangent of
the smeared links, and `dU` is overwritten with the corresponding
unconstrained thin-link cotangent. Projection to the Lie algebra, if required
by an HMC integrator, remains the caller's responsibility. For APE and HYP,
the principal determinant-phase derivative is defined away from its branch
cut.
"""
function link_smear_pullback!(
    dU::AbstractVector{<:Gaugefields_4D_MPILattice},
    dsmeared::AbstractVector{<:Gaugefields_4D_MPILattice},
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    cache::NHYPSmearingCache,
)
    dthin_links = _nhyp_lattice_links(dU, "dU")
    dsmeared_links = _nhyp_lattice_links(dsmeared, "dsmeared")
    thin_links = _nhyp_lattice_links(U, "U")
    LatticeMatrices.smear_links_pullback!(
        dthin_links,
        dsmeared_links,
        thin_links,
        cache.lattice_cache,
    )
    return dU
end

function link_smear_pullback!(
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

nhyp_pullback!(dU, dsmeared, U, cache::NHYPSmearingCache) =
    link_smear_pullback!(dU, dsmeared, U, cache)

"""Allocate the thin-link cotangent for a cached nHYP forward pass."""
function nhyp_pullback(
    dsmeared::AbstractVector{<:Gaugefields_4D_MPILattice},
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    cache::NHYPSmearingCache,
)
    return link_smear_pullback(dsmeared, U, cache)
end

function link_smear_pullback(
    dsmeared::AbstractVector{<:Gaugefields_4D_MPILattice},
    U::AbstractVector{<:Gaugefields_4D_MPILattice},
    cache::NHYPSmearingCache,
)
    dU = similar(U)
    link_smear_pullback!(dU, dsmeared, U, cache)
    return dU
end

function link_smear_pullback(
    dsmeared::AbstractVector{<:AbstractGaugefields},
    U::AbstractVector{<:AbstractGaugefields},
    cache::NHYPSmearingCache,
)
    _validate_nhyp_configuration(dsmeared, "dsmeared")
    _validate_nhyp_configuration(U, "U")
    error("unreachable")
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
