"""
    AbstractGaugeBackend

Backend selector for the high-level Gaugefields API.
"""
abstract type AbstractGaugeBackend end

"""
    LatticeMatricesBackend()

Select the portable LatticeMatrices/JACC implementation. This is the default
backend of [`gauge_configuration`](@ref).
"""
struct LatticeMatricesBackend <: AbstractGaugeBackend end

"""
    LegacyBackend()

Select the serial compatibility implementation used by the historical API.
The historical [`Initialize_Gaugefields`](@ref) entry point keeps its existing
default independently of the new API.
"""
struct LegacyBackend <: AbstractGaugeBackend end

import .AbstractGaugefields_module:
    Gaugefields_2D_MPILattice,
    Gaugefields_3D_MPILattice,
    Gaugefields_4D_MPILattice

const _LatticeMatricesGaugefield = Union{
    Gaugefields_2D_MPILattice,
    Gaugefields_3D_MPILattice,
    Gaugefields_4D_MPILattice,
}

function _validate_lattice_size(lattice, dim)
    dim in (2, 3, 4) || throw(ArgumentError(
        "gauge configurations support 2, 3, or 4 dimensions; got $dim",
    ))
    all(>(0), lattice) || throw(ArgumentError(
        "all lattice extents must be positive; got $lattice",
    ))
    return ntuple(i -> Int(lattice[i]), dim)
end

function _resolve_boundary(boundary, dim)
    boundary === :periodic && return ones(Float64, dim)
    boundary isa AbstractVector || boundary isa Tuple || throw(ArgumentError(
        "boundary must be :periodic or a collection with one phase per dimension",
    ))
    length(boundary) == dim || throw(ArgumentError(
        "boundary must have length $dim; got $(length(boundary))",
    ))
    return collect(boundary)
end

function _resolve_process_grid(process_grid, dim)
    process_grid === nothing && return nothing
    process_grid isa AbstractVector || process_grid isa Tuple || throw(ArgumentError(
        "process_grid must be a collection with one entry per dimension",
    ))
    length(process_grid) == dim || throw(ArgumentError(
        "process_grid must have length $dim; got $(length(process_grid))",
    ))
    all(>(0), process_grid) || throw(ArgumentError(
        "all process-grid entries must be positive; got $process_grid",
    ))
    return ntuple(i -> Int(process_grid[i]), dim)
end

"""
    gauge_configuration(lattice; kwargs...)

Create one gauge link field for every direction of `lattice`. The return value
is the existing `Vector` representation and therefore has length
`length(lattice)`.

The new API defaults to [`LatticeMatricesBackend`](@ref), while the historical
`Initialize_Gaugefields` default is unchanged. Use `backend=LegacyBackend()`
to request the serial compatibility implementation explicitly.

# Keywords

- `colors=3`: number of colors.
- `halo=1`: halo width.
- `start=:cold`: either `:cold` or `:hot`.
- `seed=nothing`: reproducible global-site seed for a LatticeMatrices hot start.
- `process_grid=nothing`: MPI process grid for LatticeMatrices.
- `boundary=:periodic`: boundary phases or `:periodic`.
- `eltype=ComplexF64`: element type.
- `rng=Philox4x32()`: site-local RNG algorithm for LatticeMatrices.
- `verbose=0`: verbosity level.
"""
function gauge_configuration(
    lattice::NTuple{Dim,T};
    backend::AbstractGaugeBackend=LatticeMatricesBackend(),
    colors::Integer=3,
    halo::Integer=1,
    start::Symbol=:cold,
    seed=nothing,
    process_grid=nothing,
    boundary=:periodic,
    eltype::Type=ComplexF64,
    rng::SiteRNGAlgorithm=Philox4x32(),
    verbose::Integer=0,
) where {Dim,T<:Integer}
    dimensions = _validate_lattice_size(lattice, Dim)
    colors > 0 || throw(ArgumentError("colors must be positive; got $colors"))
    start in (:cold, :hot) || throw(ArgumentError(
        "start must be :cold or :hot; got $start",
    ))
    halo_width = Int(halo)
    halo_width >= 0 || throw(ArgumentError("halo must be nonnegative; got $halo_width"))
    boundary_phases = _resolve_boundary(boundary, Dim)
    grid = _resolve_process_grid(process_grid, Dim)
    condition = String(start)

    if backend isa LatticeMatricesBackend
        return Initialize_Gaugefields(
            Int(colors),
            halo_width,
            dimensions...;
            condition,
            PEs=grid,
            verbose_level=Int(verbose),
            randomnumber=seed === nothing ? "Random" : "Reproducible",
            isMPILattice=true,
            boundarycondition=boundary_phases,
            seed,
            rng_algorithm=rng,
            elementtype=eltype,
        )
    end

    eltype == ComplexF64 || throw(ArgumentError(
        "LegacyBackend currently supports eltype=ComplexF64 in the new API; " *
        "use Initialize_Gaugefields for legacy accelerator-specific element types",
    ))
    seed === nothing || throw(ArgumentError(
        "LegacyBackend cannot honor an explicit seed; use LatticeMatricesBackend",
    ))
    grid === nothing || throw(ArgumentError(
        "process_grid belongs to LatticeMatricesBackend in the new API; " *
        "use Initialize_Gaugefields for the historical legacy MPI interface",
    ))
    all(==(1), boundary_phases) || throw(ArgumentError(
        "LegacyBackend does not expose boundary phases through the new API",
    ))
    Dim == 3 && halo_width != 0 && throw(ArgumentError(
        "LegacyBackend supports 3D configurations only with halo=0",
    ))

    return Initialize_Gaugefields(
        Int(colors),
        halo_width,
        dimensions...;
        condition,
        verbose_level=Int(verbose),
        isMPILattice=false,
    )
end

function gauge_configuration(lattice::AbstractVector{<:Integer}; kwargs...)
    return gauge_configuration(Tuple(lattice); kwargs...)
end

@inline function _first_gauge_link(U::AbstractVector{<:AbstractGaugefields})
    isempty(U) && throw(ArgumentError("a gauge configuration must not be empty"))
    return first(U)
end

"""Return the backend used by a gauge link or gauge configuration."""
gauge_backend(::AbstractGaugefields) = LegacyBackend()
gauge_backend(::_LatticeMatricesGaugefield) = LatticeMatricesBackend()
gauge_backend(U::AbstractVector{<:AbstractGaugefields}) = gauge_backend(_first_gauge_link(U))

"""Return the global lattice size as a tuple."""
gauge_lattice_size(U::AbstractGaugefields) = Tuple(size(U)[3:end])
gauge_lattice_size(U::AbstractVector{<:AbstractGaugefields}) =
    gauge_lattice_size(_first_gauge_link(U))

"""Return the number of colors."""
gauge_num_colors(U::AbstractGaugefields) = U.NC
gauge_num_colors(U::AbstractVector{<:AbstractGaugefields}) =
    gauge_num_colors(_first_gauge_link(U))

"""Return the halo width."""
gauge_halo_width(U::AbstractGaugefields) = U.NDW
gauge_halo_width(U::AbstractVector{<:AbstractGaugefields}) =
    gauge_halo_width(_first_gauge_link(U))

"""Return the MPI process grid as a tuple."""
gauge_process_grid(U::_LatticeMatricesGaugefield) = Tuple(U.U.dims)
function gauge_process_grid(U::AbstractGaugefields)
    hasproperty(U, :PEs) && return Tuple(getproperty(U, :PEs))
    return ntuple(_ -> 1, length(gauge_lattice_size(U)))
end
gauge_process_grid(U::AbstractVector{<:AbstractGaugefields}) =
    gauge_process_grid(_first_gauge_link(U))

"""Return the MPI communicator, or `nothing` for a serial legacy field."""
gauge_communicator(U::_LatticeMatricesGaugefield) = U.U.comm
function gauge_communicator(U::AbstractGaugefields)
    hasproperty(U, :comm) && return getproperty(U, :comm)
    return nothing
end
gauge_communicator(U::AbstractVector{<:AbstractGaugefields}) =
    gauge_communicator(_first_gauge_link(U))

"""Allocate zero-valued conjugate momentum fields compatible with `U`."""
gauge_momenta(U) = initialize_TA_Gaugefields(U)

"""
    gaussian_momenta(U; sigma=1, seed=nothing, sweep=0, rng=Philox4x32())

Allocate conjugate momenta compatible with `U` and fill them from a Gaussian
distribution. Explicit seeds and site-local RNG algorithms require the
LatticeMatrices backend.
"""
function gaussian_momenta(
    U;
    sigma=1.0,
    seed=nothing,
    sweep::Integer=0,
    rng::SiteRNGAlgorithm=Philox4x32(),
)
    momenta = gauge_momenta(U)
    if gauge_backend(U) isa LatticeMatricesBackend
        gauss_distribution!(
            momenta;
            σ=sigma,
            seed,
            sweep,
            rng_algorithm=rng,
        )
    else
        seed === nothing || throw(ArgumentError(
            "LegacyBackend cannot honor an explicit momentum seed",
        ))
        sweep == 0 || throw(ArgumentError(
            "LegacyBackend does not support a momentum sweep counter",
        ))
        gauss_distribution!(momenta; σ=sigma)
    end
    return momenta
end

"""
    measure_plaquette(U; normalize=true)

Calculate the plaquette. The default is normalized so that a cold
configuration returns one. Set `normalize=false` for the historical summed
value returned by `calculate_Plaquette`.
"""
function measure_plaquette(U; normalize::Bool=true)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    value = calculate_Plaquette(U, temp1, temp2)
    normalize || return value
    dim = length(U)
    return value / (
        binomial(dim, 2) * prod(gauge_lattice_size(U)) * gauge_num_colors(U)
    )
end

"""
    measure_polyakov_loop(U; normalize=true)

Calculate the Polyakov loop in the final lattice direction. The default
divides the color trace by the number of colors.
"""
function measure_polyakov_loop(U; normalize::Bool=true)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    value = calculate_Polyakov_loop(U, temp1, temp2)
    return normalize ? value / gauge_num_colors(U) : value
end

"""Construct the standard Wilson gradient-flow integrator."""
function gradient_flow(U; steps::Integer=1, step_size::Real=0.01)
    steps > 0 || throw(ArgumentError("steps must be positive; got $steps"))
    step_size > 0 || throw(ArgumentError("step_size must be positive; got $step_size"))
    return Gradientflow(U; Nflow=Int(steps), eps=Float64(step_size))
end

"""Construct a general-action gradient-flow integrator."""
function gradient_flow(U, loops, coefficients; steps::Integer=1, step_size::Real=0.01)
    steps > 0 || throw(ArgumentError("steps must be positive; got $steps"))
    step_size > 0 || throw(ArgumentError("step_size must be positive; got $step_size"))
    return Gradientflow_general(
        U,
        loops,
        coefficients;
        Nflow=Int(steps),
        eps=Float64(step_size),
    )
end

"""Construct a plaquette heatbath updater."""
heatbath_updater(U; beta, kwargs...) = Heatbath(U, beta; kwargs...)
heatbath_updater(U, beta::Real; kwargs...) = Heatbath(U, beta; kwargs...)

"""Construct a general-action heatbath updater."""
heatbath_updater(U, action; kwargs...) = Heatbath_update(U, action; kwargs...)

"""
    stout_smearing(U; loops=:plaquette, rho=0.1)

Construct a one-layer stout-smearing pipeline without exposing the internal
`CovNeuralnet` terminology.
"""
function stout_smearing(U; loops=:plaquette, rho=0.1)
    loop_names = loops isa Union{Symbol,AbstractString} ?
        [String(loops)] : String.(collect(loops))
    coefficients = rho isa Number ?
        fill(rho, length(loop_names)) : collect(rho)
    length(coefficients) == length(loop_names) || throw(ArgumentError(
        "rho must be a scalar or have one entry per loop",
    ))
    network = CovNeuralnet(U)
    push!(network, STOUT_Layer(loop_names, coefficients, U))
    return network
end

"""
    smear(U, smearing; record=false)

Apply a smearing pipeline. Return only the smeared configuration by default.
With `record=true`, return a named tuple containing `configuration`, `history`,
and `derivative`.
"""
function smear(U, smearing; record::Bool=false, calcdSdU::Bool=false, temps=nothing)
    configuration, history, derivative = calc_smearedU(
        U,
        smearing;
        calcdSdU,
        temps,
    )
    return record ? (; configuration, history, derivative) : configuration
end

"""Save a gauge configuration in JLD2, Bridge, or ILDG format."""
function save_configuration(filename, U; format::Symbol=:jld2, kwargs...)
    if format === :jld2
        isempty(kwargs) || throw(ArgumentError("JLD2 output does not accept format keywords"))
        return saveU(filename, U)
    elseif format === :bridge
        isempty(kwargs) || throw(ArgumentError("Bridge output does not accept format keywords"))
        return save_textdata(U, filename)
    elseif format === :ildg
        return save_binarydata(U, filename; kwargs...)
    end
    throw(ArgumentError("format must be :jld2, :bridge, or :ildg; got $format"))
end

"""Load a JLD2 gauge configuration and preserve its stored backend type."""
function load_configuration(filename; format::Symbol=:jld2)
    format === :jld2 || throw(ArgumentError(
        "allocating load currently supports only format=:jld2; " *
        "use load_configuration! for Bridge or ILDG",
    ))
    return loadU(filename)
end

"""Load a gauge configuration into an existing target."""
function load_configuration!(U, filename; format::Symbol=:jld2)
    if format === :jld2
        loadU!(filename, U)
    elseif format === :bridge
        load_BridgeText!(
            filename,
            U,
            collect(gauge_lattice_size(U)),
            gauge_num_colors(U),
        )
    elseif format === :ildg
        load_binarydata!(U, filename)
    else
        throw(ArgumentError("format must be :jld2, :bridge, or :ildg; got $format"))
    end
    return U
end

export AbstractGaugeBackend,
    LatticeMatricesBackend,
    LegacyBackend,
    gauge_configuration,
    gauge_backend,
    gauge_lattice_size,
    gauge_num_colors,
    gauge_halo_width,
    gauge_process_grid,
    gauge_communicator,
    gauge_momenta,
    gaussian_momenta,
    measure_plaquette,
    measure_polyakov_loop,
    gradient_flow,
    heatbath_updater,
    stout_smearing,
    smear,
    save_configuration,
    load_configuration,
    load_configuration!
