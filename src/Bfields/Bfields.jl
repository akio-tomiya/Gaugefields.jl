module Bfield_module
import ..AbstractGaugefields_module: AbstractGaugefields, TA_Gaugefields, evaluate_gaugelinks!,
    Gaugefields_4D_nowing,
    Gaugefields_4D_MPILattice,
    Shifted_Gaugefields_4D_MPILattice,
    thooftFlux_4D_B_at_bndry,
    thooftFlux_4D_B_at_bndry_wing,
    thooftLoop_4D_B_temporal_wing,
    Initialize_Gaugefields,
    set_wing_U!,
    calculate_Plaquette,
    shift_U,
    substitute_U!,
    clear_U!,
    add_force!,
    Traceless_antihermitian_product_add!,
    unit_U!,
    multiply_12!,
    add_U!, thooftFlux_4D_B_at_bndry_nowing_mpi,
    thooftLoop_4D_B_temporal
import Wilsonloop: loops_staple_prime, Wilsonline, get_position, get_direction, GLink, isdag, make_cloverloops
import ..Wilsonloops_module: Wilson_loop_set
import ..Temporalfields_module: Temporalfields, get_temp, unused!
using LinearAlgebra
import LatticeMatrices




export Bfield, clear_Bpath_cache!

struct BPathStep{Dim}
    direction::Int8
    isdag::Bool
    coordinate::NTuple{Dim,Int64}
end

struct BPathPlan{Dim}
    origin::NTuple{Dim,Int64}
    steps::Vector{BPathStep{Dim}}
    active::Bool
end

"""
    Bfield(u; center_valued=false, use_path_cache=true)

Wrap the antisymmetric matrix of two-form fields in `u`. Set
`center_valued=true` only when every field has the form `z(x) I`; this enables
the scalar-phase fast path on supported backends. `Initialize_Bfields` sets
this automatically. The default preserves full-matrix semantics for fields
constructed directly by users.
"""
struct Bfield{T,Dim,S<:AbstractMatrix{T}}
    u::S
    center_valued::Bool
    use_path_cache::Bool
    pathplans::IdDict{Wilsonline{Dim},BPathPlan{Dim}}
    pathplans_lock::ReentrantLock

    function Bfield(
        u::AbstractMatrix{T};
        center_valued=false,
        use_path_cache=true,
    ) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
        plans = IdDict{Wilsonline{Dim},BPathPlan{Dim}}()
        return new{eltype(u),Dim,typeof(u)}(
            u, center_valued, use_path_cache, plans, ReentrantLock())
    end
end

include("ScalarBfields.jl")

@inline function Base.getindex(B::Bfield, μ, ν)
    @inbounds return B.u[μ, ν]
end

"""
    clear_Bpath_cache!(B)

Discard the cached geometric plans used to insert `B` along Wilson lines.
The plans contain no B-field values, so changing `B` does not require
invalidation. Call this function only after mutating a previously evaluated
`Wilsonline` object itself.
"""
function clear_Bpath_cache!(B::Bfield)
    lock(B.pathplans_lock) do
        empty!(B.pathplans)
    end
    return B
end

function Base.similar(B::Bfield{T,Dim}) where {T,Dim}
    output = Matrix{T}(undef, Dim, Dim)
    for μ in 1:Dim
        for ν in 1:Dim
            μ == ν && continue
            output[μ, ν] = similar(B[μ, ν])
        end
    end
    return Bfield(
        output;
        center_valued=B.center_valued,
        use_path_cache=B.use_path_cache,
    )
end

function substitute_U!(a::Bfield, b::Bfield)
    _substitute_Bstorage!(a.u, b.u)
end

function substitute_U!(a::Bfield, b::Bfield, iseven::Bool)
    _substitute_Bstorage!(a.u, b.u, iseven)
end

include("GaugeActions_Bfields.jl")



function substitute_U!(
    a::Array{<:AbstractGaugefields{NC,Dim},2},
    b::Array{<:AbstractGaugefields{NC,Dim},2},
) where {NC,Dim}
    size(a) == size(b) || throw(DimensionMismatch(
        "B-field matrices must have the same size"
    ))
    for μ in axes(a, 1)
        for ν in axes(a, 2)
            μ == ν && continue
            substitute_U!(a[μ, ν], b[μ, ν])
        end
    end
    return nothing
end

function substitute_U!(
    a::Array{T1,2},
    b::Array{T2,2},
    iseven::Bool,
) where {T1<:AbstractGaugefields,T2<:AbstractGaugefields}
    size(a) == size(b) || throw(DimensionMismatch(
        "B-field matrices must have the same size"
    ))
    for μ in axes(a, 1)
        for ν in axes(a, 2)
            μ == ν && continue
            substitute_U!(a[μ, ν], b[μ, ν], iseven)
        end
    end
    return nothing
end

"""
    Initialize_Bfields(NC, Flux, NDW, NN...;
        bfield_evaluation=:optimized, use_center_fastpath=true,
        bfield_storage=:auto, ...)

Initialize the lattice two-form B field. The default uses scalar-phase
multiplication for center-valued B fields on supported backends. Set
`use_center_fastpath=false` to force the original full-matrix implementation;
this retains path caching. Set `bfield_evaluation=:legacy` to reproduce the
pre-optimization evaluation: no path cache and full-matrix B multiplication.
These options do not change the initialized B values. On a supporting
LatticeMatrices backend, `bfield_storage=:auto` uses six scalar-backed
identity planes. Use `:matrix` for the old representation or `:scalar` to
require compact storage. Legacy/full-matrix evaluation selects matrix
storage automatically. Matrix access through `B[mu,nu]` materializes a
mutable compatibility plane; `get_Bplane` accesses compact storage instead.
"""
function Initialize_Bfields(
    NC,
    Flux,
    NDW,
    NN...;
    condition="tflux",
    mpi=false,
    PEs=nothing,
    mpiinit=nothing,
    verbose_level=2,
    randomnumber="Random",
    tloop_pos=[1, 1, 1, 1],
    tloop_dir=[1, 4],
    tloop_dis=1,
    singleprecision=false,
    isMPILattice=false,
    boundarycondition=ones(length(NN)),
    elementtype=nothing,
    bfield_evaluation::Symbol=:optimized,
    use_center_fastpath::Bool=true,
    bfield_storage::Symbol=:auto,
)

    bfield_evaluation in (:optimized, :legacy) || throw(ArgumentError(
        "bfield_evaluation must be :optimized or :legacy; " *
        "got $bfield_evaluation"))
    legacy_evaluation = bfield_evaluation === :legacy
    bfield_storage in (:auto, :matrix, :scalar) || throw(ArgumentError(
        "bfield_storage must be :auto, :matrix, or :scalar"))
    scalar_supported = isMPILattice
    want_scalar = bfield_storage === :scalar ||
        (bfield_storage === :auto && scalar_supported && use_center_fastpath && !legacy_evaluation &&
         condition in ("tflux", "tloop"))
    if want_scalar
        scalar_supported || throw(ArgumentError(
            "scalar B storage requires the LatticeMatrices backend and ScaledIdentityLattice"))
        (use_center_fastpath && !legacy_evaluation) || throw(ArgumentError(
            "scalar B storage requires optimized center-phase evaluation; choose bfield_storage=:matrix for legacy evaluation"))
        return _initialize_scalar_Bfields(NC, Flux, NDW, NN...;
            condition, PEs, verbose_level, tloop_pos, tloop_dir, tloop_dis,
            singleprecision, boundarycondition, elementtype)
    end

    Dim = length(NN)
    fluxnum = 1
    if condition == "tflux"
        u1 = B_TfluxGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus=false,
            mpi=mpi,
            PEs=PEs,
            mpiinit=mpiinit,
            verbose_level=verbose_level,
            singleprecision=singleprecision,
            isMPILattice=isMPILattice,
            boundarycondition=boundarycondition,
            elementtype=elementtype,
        )
        u2 = B_TfluxGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus=true,
            mpi=mpi,
            PEs=PEs,
            mpiinit=mpiinit,
            verbose_level=verbose_level,
            singleprecision=singleprecision,
            isMPILattice=isMPILattice,
            boundarycondition=boundarycondition,
            elementtype=elementtype,
        )
    elseif condition == "tloop"
        u1 = B_TloopGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus=false,
            mpi=mpi,
            PEs=PEs,
            mpiinit=mpiinit,
            verbose_level=verbose_level,
            tloop_pos=tloop_pos,
            tloop_dir=tloop_dir,
            tloop_dis=tloop_dis,
            singleprecision=singleprecision,
            isMPILattice=isMPILattice,
            boundarycondition=boundarycondition,
            elementtype=elementtype,
        )
        u2 = B_TloopGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus=true,
            mpi=mpi,
            PEs=PEs,
            mpiinit=mpiinit,
            verbose_level=verbose_level,
            tloop_pos=tloop_pos,
            tloop_dir=tloop_dir,
            tloop_dis=tloop_dis,
            singleprecision=singleprecision,
            isMPILattice=isMPILattice,
            boundarycondition=boundarycondition,
            elementtype=elementtype,
        )
    elseif condition == "random"
        u1 = B_RandomGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus=false,
            mpi=mpi,
            PEs=PEs,
            mpiinit=mpiinit,
            verbose_level=verbose_level,
            randomnumber=randomnumber,
            singleprecision=singleprecision,
            isMPILattice=isMPILattice,
            boundarycondition=boundarycondition,
            elementtype=elementtype,
        )
        u2 = B_RandomGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus=true,
            mpi=mpi,
            PEs=PEs,
            mpiinit=mpiinit,
            verbose_level=verbose_level,
            randomnumber=randomnumber,
            singleprecision=singleprecision,
            isMPILattice=isMPILattice,
            boundarycondition=boundarycondition,
            elementtype=elementtype,
        )
        # elseif condition == "hot"
        #     u1 = RandomGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level,randomnumber = "Random")
        # elseif condition == "identity"
        #     u1 = IdentityGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level)
    else
        error("not supported")
    end

    U = Array{typeof(u1),2}(undef, Dim, Dim)

    U[1, 2] = u1
    U[2, 1] = u2

    for μ = 1:Dim
        for ν = μ+1:Dim
            if (μ, ν) != (1, 2)
                fluxnum += 1
                if condition == "tflux"
                    U[μ, ν] = B_TfluxGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus=false,
                        mpi=mpi,
                        PEs=PEs,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        singleprecision=singleprecision,
                        isMPILattice=isMPILattice,
                        boundarycondition=boundarycondition,
                        elementtype=elementtype,
                    )
                    U[ν, μ] = B_TfluxGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus=true,
                        mpi=mpi,
                        PEs=PEs,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        singleprecision=singleprecision,
                        isMPILattice=isMPILattice,
                        boundarycondition=boundarycondition,
                        elementtype=elementtype,
                    )
                elseif condition == "tloop"
                    U[μ, ν] = B_TloopGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus=false,
                        mpi=mpi,
                        PEs=PEs,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        tloop_pos=tloop_pos,
                        tloop_dir=tloop_dir,
                        tloop_dis=tloop_dis,
                        singleprecision=singleprecision,
                        isMPILattice=isMPILattice,
                        boundarycondition=boundarycondition,
                        elementtype=elementtype,
                    )
                    U[ν, μ] = B_TloopGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus=true,
                        mpi=mpi,
                        PEs=PEs,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        tloop_pos=tloop_pos,
                        tloop_dir=tloop_dir,
                        tloop_dis=tloop_dis,
                        singleprecision=singleprecision,
                        isMPILattice=isMPILattice,
                        boundarycondition=boundarycondition,
                        elementtype=elementtype,
                    )
                elseif condition == "random"
                    U[μ, ν] = B_RandomGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus=false,
                        mpi=mpi,
                        PEs=PEs,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        randomnumber=randomnumber,
                        singleprecision=singleprecision,
                        isMPILattice=isMPILattice,
                        boundarycondition=boundarycondition,
                        elementtype=elementtype,
                    )
                    U[ν, μ] = B_RandomGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus=true,
                        mpi=mpi,
                        PEs=PEs,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        randomnumber=randomnumber,
                        singleprecision=singleprecision,
                        isMPILattice=isMPILattice,
                        boundarycondition=boundarycondition,
                        elementtype=elementtype,
                    )
                    # elseif condition == "hot"
                    #     U[μ,ν] = RandomGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level,randomnumber = "Random")
                    # elseif condition == "identity"
                    #     U[μ,ν] = IdentityGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level)
                else
                    error("not supported")
                end
            end
        end
    end
    return Bfield(
        U;
        center_valued=use_center_fastpath && !legacy_evaluation,
        use_path_cache=!legacy_evaluation,
    )
    #return U
end

function B_RandomGauges(
    NC,
    Flux,
    FluxNum,
    NDW,
    NN...;
    overallminus=false,
    mpi=false,
    PEs=nothing,
    mpiinit=nothing,
    verbose_level=2,
    randomnumber="Random",
    singleprecision=false,
    isMPILattice=false,
    boundarycondition=ones(length(NN)),
    elementtype=nothing,
)
    dim = length(NN)
    println("Not implemented yet! In what follows, let us use B_TfluxGauges.")
    U = B_TfluxGauges(
        NC,
        Flux,
        FluxNum,
        NDW,
        NN...;
        overallminus,
        mpi,
        PEs,
        mpiinit,
        verbose_level,
        singleprecision,
        isMPILattice,
        boundarycondition,
        elementtype,
    )
    return U
end

function B_TfluxGauges(
    NC,
    Flux,
    FluxNum,
    NDW,
    NN...;
    overallminus=false,
    mpi=false,
    PEs=nothing,
    mpiinit=nothing,
    verbose_level=2,
    singleprecision=false,
    isMPILattice=false,
    boundarycondition=ones(length(NN)),
    elementtype=nothing,
)
    dim = length(NN)
    if isMPILattice
        dim == 4 || error("$dim dimension is not implemented yet!")
        legacy = B_TfluxGauges(
            NC,
            Flux,
            FluxNum,
            0,
            NN...;
            overallminus,
            verbose_level,
        )
        U = Initialize_Gaugefields(
            NC,
            NDW,
            NN...;
            condition="cold",
            PEs,
            verbose_level,
            singleprecision,
            isMPILattice=true,
            boundarycondition,
            elementtype,
        )[1]
        substitute_U!(U, legacy)
    elseif mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4
                if NDW == 0
                    U = thooftFlux_4D_B_at_bndry_nowing_mpi(
                        NC,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus=overallminus,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                    )
                else
                    U = thooftFlux_4D_B_at_bndry_wing_mpi(
                        NC,
                        NDW,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus=overallminus,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                    )
                end
            else
                error("$dim dimension is not implemented yet!")
            end
        end
    else
        if dim == 4
            if NDW == 0
                U = thooftFlux_4D_B_at_bndry(
                    NC,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus=overallminus,
                    verbose_level=2,
                )
            else
                U = thooftFlux_4D_B_at_bndry_wing(
                    NC,
                    NDW,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus=overallminus,
                    verbose_level=2,
                )
            end
        else
            error("$dim dimension is not implemented yet!")
        end
    end
    set_wing_U!(U)
    return U
end

function B_TloopGauges(
    NC,
    Flux,
    FluxNum,
    NDW,
    NN...;
    overallminus=false,
    mpi=false,
    PEs=nothing,
    mpiinit=nothing,
    verbose_level=2,
    tloop_pos=[1, 1, 1, 1],
    tloop_dir=[1, 4],
    tloop_dis=1,
    singleprecision=false,
    isMPILattice=false,
    boundarycondition=ones(length(NN)),
    elementtype=nothing,
)
    # pos = position of Polyakov loop
    # dir = [1-dir shift of anti-Polyakov loop,temporal 4-dir]
    # dis = distance between two loops in 1-dir with sign
    #
    # Polyakov loop at [ix,iy+1/2,iz+1/2,:]
    # anti-Polyakov loop at [ix+dis,iy+1/2,iz+1/2,end:1]
    #
    #           NT |     |
    #              |     |
    #              |     |
    # Polyakovloop |     | antiPolyakovloop
    #              |     |
    #              |     |
    #           1  |     |
    #              x     x+dis
    # and
    #         ----
    #        /   /
    #   ----/  ----- t
    #      /   /
    #     ----  y-z plaquette
    #
    dim = length(NN)
    if isMPILattice
        dim == 4 || error("$dim dimension is not implemented yet!")
        legacy = B_TloopGauges(
            NC,
            Flux,
            FluxNum,
            0,
            NN...;
            overallminus,
            verbose_level,
            tloop_pos,
            tloop_dir,
            tloop_dis,
        )
        U = Initialize_Gaugefields(
            NC,
            NDW,
            NN...;
            condition="cold",
            PEs,
            verbose_level,
            singleprecision,
            isMPILattice=true,
            boundarycondition,
            elementtype,
        )[1]
        substitute_U!(U, legacy)
    elseif mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4
                if NDW == 0
                    U = thooftLoop_4D_B_temporal_nowing_mpi(
                        NC,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus=overallminus,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        tloop_pos=tloop_pos,
                        tloop_dir=tloop_dir,
                        tloop_dis=tloop_dis,
                    )
                else
                    U = thooftLoop_4D_B_temporal_wing_mpi(
                        NC,
                        NDW,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus=overallminus,
                        mpiinit=mpiinit,
                        verbose_level=verbose_level,
                        tloop_pos=tloop_pos,
                        tloop_dir=tloop_dir,
                        tloop_dis=tloop_dis,
                    )
                end
            else
                error("$dim dimension is not implemented yet!")
            end
        end
    else
        if dim == 4
            if NDW == 0
                U = thooftLoop_4D_B_temporal(
                    NC,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus=overallminus,
                    verbose_level=2,
                    tloop_pos=tloop_pos,
                    tloop_dir=tloop_dir,
                    tloop_dis=tloop_dis,
                )
            else
                U = thooftLoop_4D_B_temporal_wing(
                    NC,
                    NDW,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus=overallminus,
                    verbose_level=2,
                    tloop_pos=tloop_pos,
                    tloop_dir=tloop_dir,
                    tloop_dis=tloop_dis,
                )
            end
        else
            error("$dim dimension is not implemented yet!")
        end
    end
    set_wing_U!(U)
    return U
end



function evaluate_gaugelinks!(
    uout::T,
    w::Wilsonline{Dim},
    U::Array{T,1},
    B::Bfield{T,Dim},
    temps::Array{T,1}, # length >= 4 for the matrix B fallback
) where {T<:AbstractGaugefields,Dim}
    # Reuse the B-free evaluator, including its shifted-buffer lifecycle.
    # The B surface only multiplies this product; it must not translate it.
    evaluate_gaugelinks!(uout, w, U, temps)
    multiply_Bplaquettes!(uout, w, B, temps)
    return nothing
end

function evaluate_Bplaquettes!(
    uout::T,
    w::Wilsonline{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1},
) where {T<:AbstractGaugefields,Dim}
    multiply_Bplaquettes!(uout, w, B, temps, true)
end

function _build_Bpath_plan(
    w::Wilsonline{Dim};
    validate_path=true,
) where {Dim}
    numlinks = length(w)
    inactive_plan() = BPathPlan{Dim}(
        ntuple(_ -> Int64(0), Dim), BPathStep{Dim}[], false)
    numlinks == 0 && return inactive_plan()

    if validate_path
        numlinks < 3 && return inactive_plan()
        displacement = zeros(Int64, Dim)
        for index = 1:numlinks
            link = w[index]
            direction = get_direction(link)
            displacement[direction] += isdag(link) ? -1 : 1
        end
        isloop = numlinks >= 4 && all(iszero, displacement)
        isstaple = sum(abs, displacement) == 1
        (isloop || isstaple) || return inactive_plan()
    end

    firstlink = w[1]
    origin = collect(get_position(firstlink))
    if isdag(firstlink)
        origin[get_direction(firstlink)] += 1
    end

    coordinate = copy(origin)
    steps = Vector{BPathStep{Dim}}(undef, numlinks)
    for index = 1:numlinks
        link = w[index]
        direction = get_direction(link)
        link_isdag = isdag(link)
        if link_isdag
            coordinate[direction] -= 1
        end
        steps[index] = BPathStep{Dim}(
            Int8(direction), link_isdag, Tuple(coordinate))
        if !link_isdag
            coordinate[direction] += 1
        end
    end
    return BPathPlan{Dim}(Tuple(origin), steps, true)
end

function _get_Bpath_plan!(B::Bfield{T,Dim}, w::Wilsonline{Dim}) where {T,Dim}
    lock(B.pathplans_lock) do
        return get!(B.pathplans, w) do
            _build_Bpath_plan(w)
        end
    end
end

function _build_legacy_Bpath_step(
    w::Wilsonline{Dim},
    linknum,
) where {Dim}
    1 <= linknum <= length(w) || return nothing

    firstlink = w[1]
    origin = get_position(firstlink)
    if isdag(firstlink)
        origin_shift = [0, 0, 0, 0]
        origin_shift[get_direction(firstlink)] += 1
        origin = Tuple(origin_shift .+ collect(origin))
    end

    coordinate = [0, 0, 0, 0] .+ collect(origin)
    for index in 1:(linknum - 1)
        link = w[index]
        direction = get_direction(link)
        coordinate[direction] += isdag(link) ? -1 : 1
    end

    link = w[linknum]
    direction = get_direction(link)
    link_isdag = isdag(link)
    link_isdag && (coordinate[direction] -= 1)
    step = BPathStep{Dim}(Int8(direction), link_isdag, Tuple(coordinate))
    return Tuple(origin), step
end

function _multiply_Bplaquettes_legacy!(
    uout::T,
    w::Wilsonline{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1},
    unity,
) where {T<:AbstractGaugefields,Dim}
    unity && unit_U!(uout)

    numlinks = length(w)
    numlinks < 3 && return
    (isLoopwithB(w) || isStaplewithB(w)) || return

    for linknum in 1:numlinks
        origin, step = _build_legacy_Bpath_step(w, linknum)
        _sweepaway_4D_Bplaquettes!(uout, origin, step, B, temps)
    end
    return nothing
end

function multiply_Bplaquettes!(
    uout::T,
    w::Wilsonline{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1},
    unity=false,
) where {T<:AbstractGaugefields,Dim}
    B.use_path_cache || return _multiply_Bplaquettes_legacy!(
        uout, w, B, temps, unity)

    plan = _get_Bpath_plan!(B, w)
    return multiply_Bplaquettes!(uout, plan, B, temps, unity)
end

function multiply_Bplaquettes!(
    uout::T,
    plan::BPathPlan{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1},
    unity=false,
) where {T<:AbstractGaugefields,Dim}
    if unity
        unit_U!(uout)
    end

    plan.active || return

    for step in plan.steps
        _sweepaway_4D_Bplaquettes!(uout, plan.origin, step, B, temps)
    end

end

function sweepaway_4D_Bplaquettes!(
    uout::T,
    w::Wilsonline{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1}, # length(temps) >= 4
    linknum,
) where {T<:AbstractGaugefields,Dim}
    if !B.use_path_cache
        path_step = _build_legacy_Bpath_step(w, linknum)
        isnothing(path_step) && return
        origin, step = path_step
        return _sweepaway_4D_Bplaquettes!(uout, origin, step, B, temps)
    end

    plan = _build_Bpath_plan(w; validate_path=false)
    plan.active || return
    1 <= linknum <= length(plan.steps) || return
    return _sweepaway_4D_Bplaquettes!(
        uout, plan.origin, plan.steps[linknum], B, temps)
end

@inline function _multiply_center_phase!(
    uout::Gaugefields_4D_nowing{NC},
    Uin,
    Bplane::Gaugefields_4D_nowing{NC},
    shift::NTuple{4,Int64},
    isdag::Bool,
) where {NC}
    NX, NY, NZ, NT = Bplane.NX, Bplane.NY, Bplane.NZ, Bplane.NT
    sx, sy, sz, st = shift

    @inbounds for it in 1:NT
        bt = mod1(it + st, NT)
        for iz in 1:NZ
            bz = mod1(iz + sz, NZ)
            for iy in 1:NY
                by = mod1(iy + sy, NY)
                for ix in 1:NX
                    bx = mod1(ix + sx, NX)
                    phase = Bplane[1, 1, bx, by, bz, bt]
                    isdag && (phase = conj(phase))
                    for j in 1:NC
                        @simd for i in 1:NC
                            uout[i, j, ix, iy, iz, it] =
                                Uin[i, j, ix, iy, iz, it] * phase
                        end
                    end
                end
            end
        end
    end
    return nothing
end

@inline _supports_center_phase(::AbstractGaugefields) = false
@inline _supports_center_phase(::Gaugefields_4D_nowing) = true
@inline _supports_center_phase(::Gaugefields_4D_MPILattice) = true

@inline _release_center_input!(input) = nothing
@inline _release_center_input!(input::Shifted_Gaugefields_4D_MPILattice) =
    LatticeMatrices.release!(input.U)

function _multiply_center_phase!(
    uout::Gaugefields_4D_MPILattice, Uin,
    Bplane, shift::NTuple{4,Int64}, isdag::Bool,
)
    LatticeMatrices.with_shifted_lattice(Bplane, shift) do shifted_B
        mul!(uout.U, Uin.U, isdag ? shifted_B' : shifted_B)
    end
    return nothing
end

"""
Fast path for supported center-valued fields. A lattice two-form B field is
center valued, `B_{μν}(x) = z_{μν}(x) I`, so multiplying by a shifted B
matrix is exactly scalar multiplication by its `(1, 1)` element. The B value
is read at every call; only the path geometry is cached, so dynamical B fields
remain valid.

Other gauge-field backends continue to use the generic full-matrix fallback
below.
"""
function _sweepaway_4D_Bplaquettes_center!(
    uout::T,
    origin::NTuple{4,Int64},
    step::BPathStep{4},
    B::Bfield{T,4},
    temps::Array{T,1},
) where {T<:AbstractGaugefields}
    direction = Int(step.direction)
    direction == 4 && return nothing

    # step.coordinate already includes the path origin. Shift only B,
    # never the accumulated Wilson product, even for a displaced staple.
    coordinate = step.coordinate
    for transverse in (direction + 1):4
        displacement = coordinate[transverse]
        displacement == 0 && continue
        offsets = displacement > 0 ?
                  (0:(displacement - 1)) : (-1:-1:displacement)
        Bdag = displacement > 0 ? !step.isdag : step.isdag
        Bplane = _center_Bplane(B, direction, transverse)
        for offset in offsets
            Bshift = ntuple(4) do axis
                axis < transverse ? coordinate[axis] :
                axis == transverse ? offset : 0
            end
            _multiply_center_phase!(uout, uout, Bplane, Bshift, Bdag)
        end
    end
    return nothing
end

function _sweepaway_4D_Bplaquettes!(
    uout::T,
    origin::NTuple{Dim,Int64},
    step::BPathStep{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1},
) where {T<:AbstractGaugefields,Dim}
    if B.center_valued && _supports_center_phase(uout) &&
       (uout isa Gaugefields_4D_nowing || _uses_scalar_storage(B.u))
        return _sweepaway_4D_Bplaquettes_center!(uout, origin, step, B, temps)
    end
    return _sweepaway_4D_Bplaquettes_fullmatrix!(
        uout, origin, step, B, temps)
end

function _sweepaway_4D_Bplaquettes_fullmatrix!(
    uout::T,
    origin::NTuple{Dim,Int64},
    step::BPathStep{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1},
) where {T<:AbstractGaugefields,Dim}
    direction = Int(step.direction)
    coordinate = step.coordinate
    Unew = temps[1]

    for transverse in (direction + 1):Dim
        displacement = coordinate[transverse]
        displacement == 0 && continue
        offsets = displacement > 0 ?
                  (0:(displacement - 1)) : (-1:-1:displacement)
        Bdag = displacement > 0 ? !step.isdag : step.isdag
        for offset in offsets
            Bshift = ntuple(Dim) do axis
                axis < transverse ? coordinate[axis] :
                axis == transverse ? offset : 0
            end
            # Shift the original plane directly. Repeatedly copying a
            # shifted view back onto its parent aliases on wing backends.
            shifted_B = shift_U(B[direction, transverse], Bshift)
            try
                substitute_U!(Unew, uout)
                multiply_12!(uout, Unew, shifted_B, 0, Bdag, false)
            finally
                _release_center_input!(shifted_B)
            end
        end
    end
    return nothing
end


function isLoopwithB(
    w::Wilsonline{Dim},
) where {Dim}
    glinks = w
    numlinks = length(glinks)
    if numlinks < 4
        return false
    end

    coordinate = [0, 0, 0, 0]
    for j = 1:numlinks
        Ujlink = glinks[j]
        direction = get_direction(Ujlink)
        isU1dag = isdag(Ujlink)
        if isU1dag
            coordinate[direction] += -1
        else
            coordinate[direction] += +1
        end
    end

    if coordinate == [0, 0, 0, 0]
        return true
    else
        return false
    end

end

function isStaplewithB(
    w::Wilsonline{Dim},
) where {Dim}
    glinks = w
    numlinks = length(glinks)
    if numlinks < 3
        return false
    end

    coordinate = [0, 0, 0, 0]
    for j = 1:numlinks
        Ujlink = glinks[j]
        direction = get_direction(Ujlink)
        isU1dag = isdag(Ujlink)
        if isU1dag
            coordinate[direction] += -1
        else
            coordinate[direction] += +1
        end
    end

    if norm(coordinate, 1) == 1.0
        return true
    else
        return false
    end

end

function evaluate_gaugelinks!(
    xout::T,
    w::Array{WL,1},
    U::Array{T,1},
    B::Bfield{T,Dim},
    temps::Array{T,1}, # length >= 5
) where {Dim,WL<:Wilsonline{Dim},T<:AbstractGaugefields}
    num = length(w)
    temp1 = temps[5]

    clear_U!(xout)
    for i = 1:num
        glinks = w[i]
        evaluate_gaugelinks!(temp1, glinks, U, B, temps[1:4]) # length >= 4
        add_U!(xout, temp1)
    end

    return
end

function evaluate_wilson_loops!(
    xout::T,
    w::Wilson_loop_set,
    U::Array{T,1},
    B::Bfield{T,Dim},
    temps::Array{T,1},
) where {T<:AbstractGaugefields,Dim}
    num = length(w)
    clear_U!(xout)
    Uold = temps[1]
    Unew = temps[2]

    for i = 1:num
        wi = w[i]
        numloops = length(wi)
        shifts = calc_shift(wi)

        loopk = wi[1]
        k = 1
        substitute_U!(Uold, U[loopk[1]])
        Ushift1 = shift_U(Uold, shifts[1])

        loopk1_2 = loopk[2]
        evaluate_wilson_loops_inside!(
            U,
            B,
            shifts,
            wi,
            Ushift1,
            Uold,
            Unew,
            numloops,
            loopk,
            loopk1_2,
            temps,
        )
        add_U!(xout, Uold)
    end
end

function evaluate_wilson_loops_inside!(
    U,
    B,
    shifts,
    wi,
    Ushift1,
    Uold,
    Unew,
    numloops,
    loopk,
    loopk1_2,
    temps,
)
    for k = 2:numloops
        loopk = wi[k]
        Ushift2 = shift_U(U[loopk[1]], shifts[k])

        multiply_12!(Unew, Ushift1, Ushift2, k, loopk, loopk1_2)

        Unew, Uold = Uold, Unew
        Ushift1 = shift_U(Uold, (0, 0, 0, 0))
    end
    multiply_Bplaquettes!(Unew, wi, B, temps)
end

function calculate_Plaquette(
    U::Array{T,1},
    B::Bfield{T,Dim},
) where {T<:AbstractGaugefields,Dim}
    error("calculate_Plaquette is not implemented in type $(typeof(U)) ")
end

function calculate_Plaquette(
    U::Array{T,1},
    B::Bfield{T,Dim},
    temps::Array{T1,1},
) where {T<:AbstractGaugefields,T1<:AbstractGaugefields,Dim}
    return calculate_Plaquette(U, B, temps[1], temps[2])
end

function calculate_Plaquette(
    U::Array{T,1},
    B::Bfield{T,Dim},
    temp::AbstractGaugefields{NC,Dim},
    staple::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    if _uses_scalar_storage(B.u)
        plaq = zero(real(zero(eltype(U[1]))))
        for mu in 1:(Dim-1), nu in (mu+1):Dim
            shifted_nu = shift_U(U[nu], mu)
            try
                mul!(temp, U[mu], shifted_nu)
            finally
                _release_center_input!(shifted_nu)
            end
            shifted_mu = shift_U(U[mu], nu)
            try
                mul!(staple, temp, shifted_mu')
            finally
                _release_center_input!(shifted_mu)
            end
            mul!(temp, staple, U[nu]')
            mul!(staple.U, temp.U, B.u.phases[mu, nu])
            plaq += real(tr(staple))
        end
        return plaq
    end
    plaq = 0
    V = staple
    b_link = similar(temp)
    for μ = 1:Dim
        construct_staple!(V, U, B, μ, temp, b_link)
        mul!(temp, U[μ], V')
        plaq += tr(temp)

    end
    return real(plaq * 0.5)
end

function construct_staple!(staple::T, U, B, μ) where {T<:AbstractGaugefields}
    error("construct_staple! is not implemented in type $(typeof(U)) ")
end

function add_force!(
    F::Array{T1,1},
    U::Array{T2,1},
    B::Bfield{T2,Dim},
    temps::Temporalfields{<:AbstractGaugefields{NC,Dim}};
    #temps::Array{<:AbstractGaugefields{NC,Dim},1};
    plaqonly=false,
    staplefactors::Union{Array{<:Number,1},Nothing}=nothing,
    factor=1,
) where {NC,Dim,T1<:TA_Gaugefields,T2<:AbstractGaugefields}
    plaqonly || throw(ArgumentError(
        "B-field add_force! supports plaqonly=true; use Gradientflow_general_Bfields for other actions",
    ))
    work, indices = get_temp(temps, 7)
    try
        add_force!(F, U, B, work; plaqonly, staplefactors, factor)
    finally
        unused!(temps, indices)
    end
    return nothing
end
function add_force!(
    F::Array{T1,1},
    U::Array{T2,1},
    B::Bfield{T2,Dim},
    temps::Array{<:AbstractGaugefields{NC,Dim},1};
    plaqonly=false,
    staplefactors::Union{Array{<:Number,1},Nothing}=nothing,
    factor=1,
) where {NC,Dim,T1<:TA_Gaugefields,T2<:AbstractGaugefields}
    plaqonly || throw(ArgumentError(
        "B-field add_force! supports plaqonly=true; use Gradientflow_general_Bfields for other actions",
    ))
    length(temps) >= 7 || throw(ArgumentError("B-field add_force! needs seven work fields"))
    V, product = temps[1], temps[2]
    path_temps = temps[3:7]
    for μ = 1:Dim
        construct_double_staple!(V, U, B, μ, path_temps)
        Traceless_antihermitian_product_add!(F[μ], factor, U[μ], V', product)
    end
    return nothing
end

function construct_double_staple!(
    staple::AbstractGaugefields{NC,Dim},
    U::Array{T,1},
    B::Bfield{T,Dim},
    μ,
    temps::Array{<:AbstractGaugefields{NC,Dim},1},
) where {NC,Dim,T<:AbstractGaugefields}
    loops = loops_staple_prime[(Dim, μ)]
    evaluate_gaugelinks!(staple, loops, U, B, temps)
end

function construct_staple!(
    staple::AbstractGaugefields{NC,Dim},
    U::Array{T,1},
    B::Bfield{T,Dim},
    μ,
    temp::AbstractGaugefields{NC,Dim},
    b_link::AbstractGaugefields{NC,Dim}=similar(temp),
) where {NC,Dim,T<:AbstractGaugefields}
    U1U2 = temp
    # Never use a physical link as scratch: measurement must preserve U.
    U1 = b_link
    firstterm = true

    for ν = 1:Dim
        if ν == μ
            continue
        end

        if μ < ν
            mul!(U1, U[ν], B[μ, ν]')
        else
            # The lower triangle stores -B[ν, μ], not the reversed
            # group-valued plaquette factor. Use the upper triangle.
            mul!(U1, U[ν], B[ν, μ])
        end
        U2 = shift_U(U[μ], ν)
        try
            mul!(U1U2, U1, U2)
        finally
            _release_center_input!(U2)
        end

        U3 = shift_U(U[ν], μ)
        if firstterm
            β = 0
            firstterm = false
        else
            β = 1
        end
        try
            mul!(staple, U1U2, U3', 1, β)
        finally
            _release_center_input!(U3)
        end
    end
    set_wing_U!(staple)
end



end
