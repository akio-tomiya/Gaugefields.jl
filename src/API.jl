"""
    AbstractGaugeBackend

Backend selector for the high-level Gaugefields API.
"""
abstract type AbstractGaugeBackend end

import JLD2
import MPI
import LatticeMatrices: LatticeMatrix, gather_matrix, set_halo!, substitute!

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
    Gaugefields_4D_MPILattice,
    TA_Gaugefields_2D_MPILattice,
    TA_Gaugefields_3D_MPILattice,
    TA_Gaugefields_4D_MPILattice

const _LatticeMatricesGaugefield = Union{
    Gaugefields_2D_MPILattice,
    Gaugefields_3D_MPILattice,
    Gaugefields_4D_MPILattice,
}

const _LatticeMatricesMomentum = Union{
    TA_Gaugefields_2D_MPILattice,
    TA_Gaugefields_3D_MPILattice,
    TA_Gaugefields_4D_MPILattice,
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

function _initialize_gauge_communicator(comm)
    MPI.Finalized() && throw(ArgumentError(
        "MPI has already been finalized; restart Julia before creating a " *
        "LatticeMatrices gauge configuration",
    ))
    MPI.Initialized() || MPI.Init()
    return comm === nothing ? MPI.COMM_WORLD : comm
end

function _automatic_process_grid(lattice, nprocs)
    candidates = Tuple[]
    dim = length(lattice)

    function visit(prefix, direction, remaining)
        if direction == dim
            process_count = remaining
            if lattice[direction] % process_count == 0 &&
               lattice[direction] > process_count
                push!(candidates, (prefix..., process_count))
            end
            return nothing
        end

        for process_count in 1:remaining
            remaining % process_count == 0 || continue
            lattice[direction] % process_count == 0 || continue
            lattice[direction] > process_count || continue
            visit(
                (prefix..., process_count),
                direction + 1,
                remaining ÷ process_count,
            )
        end
        return nothing
    end

    visit((), 1, nprocs)
    isempty(candidates) && throw(ArgumentError(
        "cannot decompose lattice $lattice over $nprocs MPI processes; " *
        "pass an explicit process_grid or use fewer processes",
    ))

    function score(grid)
        local_sizes = ntuple(i -> lattice[i] ÷ grid[i], dim)
        surface_to_volume = sum(grid[i] // lattice[i] for i in 1:dim)
        imbalance = maximum(local_sizes) // minimum(local_sizes)
        return (surface_to_volume, imbalance, grid)
    end
    sort!(candidates; by=score)
    return first(candidates)
end

function _resolve_process_grid(process_grid, lattice, comm)
    dim = length(lattice)
    nprocs = MPI.Comm_size(comm)
    if process_grid === nothing || process_grid === :auto
        return _automatic_process_grid(lattice, nprocs)
    end
    process_grid isa AbstractVector || process_grid isa Tuple || throw(ArgumentError(
        "process_grid must be :auto or a collection with one entry per dimension",
    ))
    length(process_grid) == dim || throw(ArgumentError(
        "process_grid must have length $dim; got $(length(process_grid))",
    ))
    all(>(0), process_grid) || throw(ArgumentError(
        "all process-grid entries must be positive; got $process_grid",
    ))
    grid = ntuple(i -> Int(process_grid[i]), dim)
    prod(grid) == nprocs || throw(ArgumentError(
        "product(process_grid) must equal communicator size $nprocs; got $grid",
    ))
    for direction in 1:dim
        lattice[direction] % grid[direction] == 0 || throw(ArgumentError(
            "lattice extent $(lattice[direction]) in direction $direction " *
            "is not divisible by process_grid[$direction]=$(grid[direction])",
        ))
        lattice[direction] > grid[direction] || throw(ArgumentError(
            "lattice extent $(lattice[direction]) in direction $direction " *
            "must exceed process_grid[$direction]=$(grid[direction])",
        ))
    end
    return grid
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
- `comm=nothing`: MPI communicator; `nothing` selects `MPI.COMM_WORLD`.
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
    comm=nothing,
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
    condition = String(start)

    if backend isa LatticeMatricesBackend
        communicator = _initialize_gauge_communicator(comm)
        grid = _resolve_process_grid(process_grid, dimensions, communicator)
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
            comm=communicator,
        )
    end

    eltype == ComplexF64 || throw(ArgumentError(
        "LegacyBackend currently supports eltype=ComplexF64 in the new API; " *
        "use Initialize_Gaugefields for legacy accelerator-specific element types",
    ))
    seed === nothing || throw(ArgumentError(
        "LegacyBackend cannot honor an explicit seed; use LatticeMatricesBackend",
    ))
    (process_grid === nothing || process_grid === :auto) || throw(ArgumentError(
        "process_grid belongs to LatticeMatricesBackend in the new API; " *
        "use Initialize_Gaugefields for the historical legacy MPI interface",
    ))
    comm === nothing || throw(ArgumentError(
        "comm belongs to LatticeMatricesBackend in the new API",
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

"""
    copy_configuration!(destination, source)

Copy a complete gauge configuration into preallocated compatible storage.
Backend-specific copy methods are used so device synchronization and halo
state are handled consistently. The destination is returned.
"""
function copy_configuration!(
    destination::AbstractVector{<:AbstractGaugefields},
    source::AbstractVector{<:AbstractGaugefields},
)
    length(destination) == length(source) || throw(DimensionMismatch(
        "destination and source have different numbers of directions",
    ))
    gauge_lattice_size(destination) == gauge_lattice_size(source) ||
        throw(DimensionMismatch("destination and source lattice sizes differ"))
    gauge_num_colors(destination) == gauge_num_colors(source) ||
        throw(DimensionMismatch("destination and source color counts differ"))
    gauge_halo_width(destination) == gauge_halo_width(source) ||
        throw(DimensionMismatch("destination and source halo widths differ"))
    substitute_U!(destination, source)
    return destination
end

"""Allocate and return an independent copy of a gauge configuration."""
function copy_configuration(source::AbstractVector{<:AbstractGaugefields})
    destination = similar(source)
    return copy_configuration!(destination, source)
end

"""Allocate zero-valued conjugate momentum fields compatible with `U`."""
gauge_momenta(U) = initialize_TA_Gaugefields(U)

"""
    gaussian_momenta!(momenta; sigma=1, seed=nothing, sweep=0,
                      rng=Philox4x32())

Fill preallocated conjugate momenta in place. Explicit seeds and site-local
RNG algorithms require LatticeMatrices-backed momenta. The momentum array is
returned.
"""
function gaussian_momenta!(
    momenta::AbstractVector;
    sigma=1.0,
    seed=nothing,
    sweep::Integer=0,
    rng::SiteRNGAlgorithm=Philox4x32(),
)
    sweep >= 0 || throw(ArgumentError("sweep must be nonnegative; got $sweep"))
    isempty(momenta) && return momenta
    if first(momenta) isa _LatticeMatricesMomentum
        gauss_distribution!(
            momenta;
            σ=sigma,
            seed,
            sweep,
            rng_algorithm=rng,
        )
    else
        seed === nothing || throw(ArgumentError(
            "legacy momenta cannot honor an explicit momentum seed",
        ))
        sweep == 0 || throw(ArgumentError(
            "legacy momenta do not support a momentum sweep counter",
        ))
        gauss_distribution!(momenta; σ=sigma)
    end
    return momenta
end

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
    return gaussian_momenta!(momenta; sigma, seed, sweep, rng)
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
    return Gradientflow(U; Nflow=Int(steps), eps=step_size)
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
        eps=step_size,
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

const _PORTABLE_JLD2_FORMAT = "Gaugefields.jl portable gauge configuration"
const _PORTABLE_JLD2_VERSION = 1

function _portable_jld2_element_type(name)
    name = String(name)
    name in ("Float32", "Core.Float32") && return Float32
    name in ("Float64", "Core.Float64") && return Float64
    name in ("ComplexF32", "Complex{Float32}") && return ComplexF32
    name in ("ComplexF64", "Complex{Float64}") && return ComplexF64
    throw(ArgumentError(
        "unsupported element type $name in portable Gaugefields JLD2 file",
    ))
end

function _portable_configuration_element_type(link)
    element_type = eltype(link)
    if element_type === Any && hasproperty(link, :U)
        element_type = eltype(getproperty(link, :U))
    end
    element_type <: Number || throw(ArgumentError(
        "cannot determine the numeric storage type of $(typeof(link))",
    ))
    return element_type
end

function _portable_jld2_boundary(U)
    first_link = _first_gauge_link(U)
    if first_link isa _LatticeMatricesGaugefield
        return collect(first_link.U.phases)
    end
    element_type = _portable_configuration_element_type(first_link)
    return ones(element_type, length(gauge_lattice_size(U)))
end

function _portable_host_link(link::AbstractGaugefields)
    lattice = gauge_lattice_size(link)
    colors = gauge_num_colors(link)
    element_type = _portable_configuration_element_type(link)
    values = Array{element_type}(undef, colors, colors, lattice...)
    for site in CartesianIndices(lattice)
        position = Tuple(site)
        for column in 1:colors, row in 1:colors
            values[row, column, position...] =
                link[row, column, position...]
        end
    end
    return values
end

function _portable_jld2_links(U)
    if gauge_backend(U) isa LatticeMatricesBackend
        return [gather_matrix(link.U; root=0) for link in U]
    end
    communicator = gauge_communicator(U)
    if communicator !== nothing && MPI.Comm_size(communicator) > 1
        throw(ArgumentError(
            "portable JLD2 for distributed fields requires the " *
            "LatticeMatrices backend",
        ))
    end
    return [_portable_host_link(link) for link in U]
end

function _portable_jld2_root_operation(operation, communicator)
    error_message = nothing
    result = nothing
    rank = isnothing(communicator) ? 0 : MPI.Comm_rank(communicator)
    if rank == 0
        try
            result = operation()
        catch err
            error_message = sprint(showerror, err, catch_backtrace())
        end
    end
    if !isnothing(communicator)
        error_message = MPI.bcast(error_message, 0, communicator)
    end
    error_message === nothing || error(error_message)
    return result
end

function _write_portable_jld2(filename, U, links)
    JLD2.jldsave(
        filename;
        gaugefields_format=_PORTABLE_JLD2_FORMAT,
        gaugefields_format_version=_PORTABLE_JLD2_VERSION,
        dimension=length(U),
        lattice_size=collect(gauge_lattice_size(U)),
        num_colors=gauge_num_colors(U),
        halo_width=gauge_halo_width(U),
        boundary=_portable_jld2_boundary(U),
        element_type=string(
            _portable_configuration_element_type(_first_gauge_link(U)),
        ),
        links,
    )
    return nothing
end

function _read_portable_jld2_metadata(filename)
    return JLD2.jldopen(filename, "r") do file
        haskey(file, "gaugefields_format") || return nothing
        file["gaugefields_format"] == _PORTABLE_JLD2_FORMAT || throw(
            ArgumentError("unrecognized Gaugefields JLD2 format in $filename"),
        )
        version = Int(file["gaugefields_format_version"])
        version == _PORTABLE_JLD2_VERSION || throw(ArgumentError(
            "unsupported portable Gaugefields JLD2 version $version",
        ))
        dimension = Int(file["dimension"])
        lattice = Tuple(Int.(file["lattice_size"]))
        length(lattice) == dimension || throw(ArgumentError(
            "JLD2 dimension $dimension does not match lattice $lattice",
        ))
        return (
            dimension,
            lattice,
            colors=Int(file["num_colors"]),
            halo=Int(file["halo_width"]),
            boundary=collect(file["boundary"]),
            element_type=_portable_jld2_element_type(file["element_type"]),
        )
    end
end

function _collective_portable_jld2_metadata(filename, communicator)
    metadata = _portable_jld2_root_operation(communicator) do
        _read_portable_jld2_metadata(filename)
    end
    if !isnothing(communicator)
        metadata = MPI.bcast(metadata, 0, communicator)
    end
    return metadata
end

function _read_portable_jld2_links(filename, metadata, communicator)
    links = _portable_jld2_root_operation(communicator) do
        JLD2.jldopen(filename, "r") do file
            values = file["links"]
            length(values) == metadata.dimension || throw(ArgumentError(
                "JLD2 file contains $(length(values)) links; expected " *
                "$(metadata.dimension)",
            ))
            expected_size = (
                metadata.colors,
                metadata.colors,
                metadata.lattice...,
            )
            for (direction, values_direction) in pairs(values)
                size(values_direction) == expected_size || throw(ArgumentError(
                    "JLD2 link $direction has size $(size(values_direction)); " *
                    "expected $expected_size",
                ))
                eltype(values_direction) == metadata.element_type || throw(
                    ArgumentError(
                        "JLD2 link $direction has element type " *
                        "$(eltype(values_direction)); expected " *
                        "$(metadata.element_type)",
                    ),
                )
            end
            return values
        end
    end
    return links
end

function _validate_portable_jld2_target(U, metadata)
    length(U) == metadata.dimension || throw(DimensionMismatch(
        "target has $(length(U)) directions; file has $(metadata.dimension)",
    ))
    gauge_lattice_size(U) == metadata.lattice || throw(DimensionMismatch(
        "target lattice $(gauge_lattice_size(U)) differs from file lattice " *
        "$(metadata.lattice)",
    ))
    gauge_num_colors(U) == metadata.colors || throw(DimensionMismatch(
        "target has $(gauge_num_colors(U)) colors; file has $(metadata.colors)",
    ))
    return nothing
end

function _load_portable_jld2_lm!(U, links, metadata, communicator)
    rank = MPI.Comm_rank(communicator)
    expected_size = (
        metadata.colors,
        metadata.colors,
        metadata.lattice...,
    )
    first_link = first(U)
    for direction in eachindex(U)
        global_link = rank == 0 ? links[direction] :
            Array{metadata.element_type}(undef, expected_size)
        temporary = LatticeMatrix(
            global_link,
            metadata.dimension,
            Tuple(first_link.U.dims);
            nw=first_link.U.nw,
            phases=first_link.U.phases,
            comm0=first_link.U.comm,
            device_mapping=:current,
        )
        substitute!(U[direction].U, temporary)
        set_halo!(U[direction].U)
    end
    return U
end

function _load_portable_jld2_legacy!(U, links, metadata)
    lattice = metadata.lattice
    colors = metadata.colors
    for direction in eachindex(U)
        values = links[direction]
        for site in CartesianIndices(lattice)
            position = Tuple(site)
            for column in 1:colors, row in 1:colors
                U[direction][row, column, position...] =
                    values[row, column, position...]
            end
        end
    end
    set_wing_U!(U)
    return U
end

"""
Save a gauge configuration in JLD2, Bridge, or ILDG format. Portable JLD2
output gathers physical links to rank 0 and stores backend-independent host
arrays; all ranks owning a distributed configuration must participate.
"""
function save_configuration(filename, U; format::Symbol=:jld2, kwargs...)
    if format === :jld2
        isempty(kwargs) || throw(ArgumentError("JLD2 output does not accept format keywords"))
        communicator = gauge_communicator(U)
        links = _portable_jld2_links(U)
        _portable_jld2_root_operation(communicator) do
            _write_portable_jld2(filename, U, links)
        end
        return nothing
    elseif format === :bridge
        isempty(kwargs) || throw(ArgumentError("Bridge output does not accept format keywords"))
        return save_textdata(U, filename)
    elseif format === :ildg
        return save_binarydata(U, filename; kwargs...)
    end
    throw(ArgumentError("format must be :jld2, :bridge, or :ildg; got $format"))
end

"""
Load a portable JLD2 gauge configuration. The destination is allocated on the
current JACC backend and communicator; `process_grid` and `comm` may differ
from those used when the file was written.
"""
function load_configuration(
    filename;
    format::Symbol=:jld2,
    backend::AbstractGaugeBackend=LatticeMatricesBackend(),
    process_grid=nothing,
    comm=nothing,
    halo=nothing,
    boundary=nothing,
    eltype=nothing,
    verbose::Integer=0,
)
    format === :jld2 || throw(ArgumentError(
        "allocating load currently supports only format=:jld2; " *
        "use load_configuration! for Bridge or ILDG",
    ))
    communicator = backend isa LatticeMatricesBackend ?
        _initialize_gauge_communicator(comm) : nothing
    metadata = _collective_portable_jld2_metadata(filename, communicator)
    if metadata === nothing
        isnothing(communicator) || MPI.Comm_size(communicator) == 1 || throw(
            ArgumentError(
                "legacy object-serialized JLD2 files can only be loaded on " *
                "one rank; rewrite the file with save_configuration",
            ),
        )
        return loadU(filename)
    end
    target_halo = isnothing(halo) ? metadata.halo : Int(halo)
    target_boundary = isnothing(boundary) ? metadata.boundary : boundary
    target_element_type = isnothing(eltype) ? metadata.element_type : eltype
    U = gauge_configuration(
        metadata.lattice;
        backend,
        colors=metadata.colors,
        halo=target_halo,
        start=:cold,
        process_grid,
        comm=communicator,
        boundary=target_boundary,
        eltype=target_element_type,
        verbose,
    )
    load_configuration!(U, filename; format=:jld2)
    return U
end

"""Load a gauge configuration into an existing target."""
function load_configuration!(U, filename; format::Symbol=:jld2)
    if format === :jld2
        communicator = gauge_communicator(U)
        metadata = _collective_portable_jld2_metadata(filename, communicator)
        if metadata === nothing
            isnothing(communicator) || MPI.Comm_size(communicator) == 1 || throw(
                ArgumentError(
                    "legacy object-serialized JLD2 files can only be loaded " *
                    "on one rank",
                ),
            )
            loadU!(filename, U)
            return U
        end
        _validate_portable_jld2_target(U, metadata)
        if !(gauge_backend(U) isa LatticeMatricesBackend) &&
           !isnothing(communicator) && MPI.Comm_size(communicator) > 1
            throw(ArgumentError(
                "distributed portable JLD2 input requires the " *
                "LatticeMatrices backend",
            ))
        end
        links = _read_portable_jld2_links(
            filename,
            metadata,
            communicator,
        )
        if gauge_backend(U) isa LatticeMatricesBackend
            _load_portable_jld2_lm!(U, links, metadata, communicator)
        else
            _load_portable_jld2_legacy!(U, links, metadata)
        end
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
    copy_configuration,
    copy_configuration!,
    gauge_momenta,
    gaussian_momenta,
    gaussian_momenta!,
    measure_plaquette,
    measure_polyakov_loop,
    gradient_flow,
    heatbath_updater,
    stout_smearing,
    smear,
    save_configuration,
    load_configuration,
    load_configuration!
