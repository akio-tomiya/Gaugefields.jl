module Bfield_module
import ..AbstractGaugefields_module: AbstractGaugefields, TA_Gaugefields, evaluate_gaugelinks!,
    Gaugefields_4D_nowing,
    thooftFlux_4D_B_at_bndry,
    Initialize_Gaugefields,
    set_wing_U!,
    calculate_Plaquette,
    shift_U,
    substitute_U!,
    clear_U!,
    unit_U!,
    multiply_12!,
    add_U!, thooftFlux_4D_B_at_bndry_nowing_mpi,
    thooftLoop_4D_B_temporal
import Wilsonloop: loops_staple_prime, Wilsonline, get_position, get_direction, GLink, isdag, make_cloverloops
import ..Wilsonloops_module: Wilson_loop_set
import ..Temporalfields_module: Temporalfields, get_temp, unused!
using LinearAlgebra




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
struct Bfield{T,Dim}
    u::Matrix{T}
    center_valued::Bool
    use_path_cache::Bool
    pathplans::IdDict{Wilsonline{Dim},BPathPlan{Dim}}
    pathplans_lock::ReentrantLock

    function Bfield(
        u::Matrix{<:AbstractGaugefields{NC,Dim}};
        center_valued=false,
        use_path_cache=true,
    ) where {NC,Dim}
        plans = IdDict{Wilsonline{Dim},BPathPlan{Dim}}()
        return new{eltype(u),Dim}(
            u, center_valued, use_path_cache, plans, ReentrantLock())
    end
end

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
    substitute_U!(a.u, b.u)
end

function substitute_U!(a::Bfield, b::Bfield, iseven::Bool)
    substitute_U!(a.u, b.u, iseven)
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
        bfield_evaluation=:optimized, use_center_fastpath=true, ...)

Initialize the lattice two-form B field. The default uses scalar-phase
multiplication for center-valued B fields on supported backends. Set
`use_center_fastpath=false` to force the original full-matrix implementation;
this retains path caching. Set `bfield_evaluation=:legacy` to reproduce the
pre-optimization evaluation: no path cache and full-matrix B multiplication.
These options change only the evaluation algorithm, not the initialized B
values.
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
)

    bfield_evaluation in (:optimized, :legacy) || throw(ArgumentError(
        "bfield_evaluation must be :optimized or :legacy; " *
        "got $bfield_evaluation"))
    legacy_evaluation = bfield_evaluation === :legacy

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
    temps::Array{T,1}, # length >= 3
) where {T<:AbstractGaugefields,Dim}
    Unew = temps[1]
    origin = Tuple(zeros(Int64, Dim))

    Ushift1 = temps[2]
    Ushift2 = temps[3]

    glinks = w
    numlinks = length(glinks)
    if numlinks == 0
        unit_U!(uout)
        return
    end

    j = 1
    U1link = glinks[1]
    direction = get_direction(U1link)
    position = get_position(U1link)
    isU1dag = isdag(U1link)

    if numlinks == 1
        substitute_U!(Unew, U[direction])
        Ushift1 = shift_U(Unew, position)
        if isU1dag
            substitute_U!(uout, Ushift1')
        else
            substitute_U!(uout, Ushift1)
        end

        return
    end

    substitute_U!(Unew, U[direction])
    Ushift1 = shift_U(Unew, position)

    for j = 2:numlinks
        Ujlink = glinks[j]
        isUkdag = isdag(Ujlink)
        position = get_position(Ujlink)
        direction = get_direction(Ujlink)
        Ushift2 = shift_U(U[direction], position)

        multiply_12!(uout, Ushift1, Ushift2, j, isUkdag, isU1dag)

        substitute_U!(Unew, uout)
        Ushift1 = shift_U(Unew, origin)
    end

    multiply_Bplaquettes!(uout, w, B, temps)

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

"""
Fast path for the non-MPI, no-wing CPU field. A lattice two-form B field is
center valued, `B_{μν}(x) = z_{μν}(x) I`, so multiplying by a shifted B
matrix is exactly scalar multiplication by its `(1, 1)` element. The B value
is read at every call; only the path geometry is cached, so dynamical B fields
remain valid.

Other gauge-field backends continue to use the generic full-matrix fallback
below.
"""
function _sweepaway_4D_Bplaquettes!(
    uout::Gaugefields_4D_nowing{NC},
    origin::NTuple{4,Int64},
    step::BPathStep{4},
    B::Bfield{Gaugefields_4D_nowing{NC},4},
    temps::Array{Gaugefields_4D_nowing{NC},1},
) where {NC}
    B.center_valued || return _sweepaway_4D_Bplaquettes_fullmatrix!(
        uout, origin, step, B, temps)

    direction = Int(step.direction)
    direction == 4 && return nothing

    coordinate = step.coordinate
    origin_iszero = all(iszero, origin)
    Uin = uout
    Unew = temps[1]

    for transverse in (direction + 1):4
        displacement = coordinate[transverse]
        displacement == 0 && continue

        offsets = displacement > 0 ?
                  (0:(displacement - 1)) : (-1:-1:displacement)
        Bdag = displacement > 0 ? !step.isdag : step.isdag
        Bplane = B[direction, transverse]

        for offset in offsets
            Bshift = ntuple(4) do axis
                axis < transverse ? coordinate[axis] :
                axis == transverse ? offset : 0
            end
            _multiply_center_phase!(uout, Uin, Bplane, Bshift, Bdag)

            if origin_iszero
                Uin = uout
            else
                substitute_U!(Unew, uout)
                Uin = shift_U(Unew, origin)
            end
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
    return _sweepaway_4D_Bplaquettes_fullmatrix!(
        uout, origin, step, B, temps)
end

function _sweepaway_4D_Bplaquettes_fullmatrix!(
    uout::T,
    origin::NTuple{Dim,Int64},
    step::BPathStep{Dim},
    B::Bfield{T,Dim},
    temps::Array{T,1}, # length(temps) >= 4
) where {T<:AbstractGaugefields,Dim}
    Unew = temps[1]
    direction = Int(step.direction)
    isU1dag = step.isdag
    coordinate = step.coordinate

    substitute_U!(Unew, uout)
    Ushift = shift_U(Unew, (0, 0, 0, 0))

    if direction == 1
        if isU1dag
            Bshift12 = shift_U(B[1, 2], (0, 0, 0, 0))
            Bshift13 = shift_U(B[1, 3], (0, 0, 0, 0))
            Bshift14 = shift_U(B[1, 4], (0, 0, 0, 0))
        else
            Bshift12 = shift_U(B[1, 2], (0, 0, 0, 0))'
            Bshift13 = shift_U(B[1, 3], (0, 0, 0, 0))'
            Bshift14 = shift_U(B[1, 4], (0, 0, 0, 0))'
        end

        Bshift12new = temps[2]
        Bshift13new = temps[3]
        Bshift14new = temps[4]

        for ix = 1:abs(coordinate[1])
            if coordinate[1] > 0
                substitute_U!(Bshift12new, Bshift12)
                Bshift12 = shift_U(Bshift12new, (1, 0, 0, 0))
                substitute_U!(Bshift13new, Bshift13)
                Bshift13 = shift_U(Bshift13new, (1, 0, 0, 0))
                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (1, 0, 0, 0))
            else # coordinate[1] < 0
                substitute_U!(Bshift12new, Bshift12)
                Bshift12 = shift_U(Bshift12new, (-1, 0, 0, 0))
                substitute_U!(Bshift13new, Bshift13)
                Bshift13 = shift_U(Bshift13new, (-1, 0, 0, 0))
                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (-1, 0, 0, 0))
            end
        end

        for iy = 1:abs(coordinate[2])
            if coordinate[2] > 0
                multiply_12!(uout, Ushift, Bshift12, 0, false, false)

                substitute_U!(Bshift12new, Bshift12)
                Bshift12 = shift_U(Bshift12new, (0, 1, 0, 0))
                substitute_U!(Bshift13new, Bshift13)
                Bshift13 = shift_U(Bshift13new, (0, 1, 0, 0))
                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (0, 1, 0, 0))
            else # coordinate[2] < 0
                substitute_U!(Bshift12new, Bshift12)
                Bshift12 = shift_U(Bshift12new, (0, -1, 0, 0))
                substitute_U!(Bshift13new, Bshift13)
                Bshift13 = shift_U(Bshift13new, (0, -1, 0, 0))
                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (0, -1, 0, 0))

                multiply_12!(uout, Ushift, Bshift12, 0, true, false)
            end

            substitute_U!(Unew, uout)
            Ushift = shift_U(Unew, origin)

        end

        for iz = 1:abs(coordinate[3])
            if coordinate[3] > 0
                multiply_12!(uout, Ushift, Bshift13, 0, false, false)

                substitute_U!(Bshift13new, Bshift13)
                Bshift13 = shift_U(Bshift13new, (0, 0, 1, 0))
                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (0, 0, 1, 0))
            else # coordinate[3] < 0
                substitute_U!(Bshift13new, Bshift13)
                Bshift13 = shift_U(Bshift13new, (0, 0, -1, 0))
                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (0, 0, -1, 0))

                multiply_12!(uout, Ushift, Bshift13, 0, true, false)
            end

            substitute_U!(Unew, uout)
            Ushift = shift_U(Unew, origin)

        end

        for it = 1:abs(coordinate[4])
            if coordinate[4] > 0
                multiply_12!(uout, Ushift, Bshift14, 0, false, false)

                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (0, 0, 0, 1))
            else # coordinate[4] < 0
                substitute_U!(Bshift14new, Bshift14)
                Bshift14 = shift_U(Bshift14new, (0, 0, 0, -1))

                multiply_12!(uout, Ushift, Bshift14, 0, true, false)
            end

            substitute_U!(Unew, uout)
            Ushift = shift_U(Unew, origin)

        end
    elseif direction == 2
        if isU1dag
            Bshift23 = shift_U(B[2, 3], (0, 0, 0, 0))
            Bshift24 = shift_U(B[2, 4], (0, 0, 0, 0))
        else
            Bshift23 = shift_U(B[2, 3], (0, 0, 0, 0))'
            Bshift24 = shift_U(B[2, 4], (0, 0, 0, 0))'
        end

        Bshift23new = temps[2]
        Bshift24new = temps[3]

        for ix = 1:abs(coordinate[1])
            if coordinate[1] > 0
                substitute_U!(Bshift23new, Bshift23)
                Bshift23 = shift_U(Bshift23new, (1, 0, 0, 0))
                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (1, 0, 0, 0))
            else # coordinate[1] < 0
                substitute_U!(Bshift23new, Bshift23)
                Bshift23 = shift_U(Bshift23new, (-1, 0, 0, 0))
                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (-1, 0, 0, 0))
            end
        end

        for iy = 1:abs(coordinate[2])
            if coordinate[2] > 0
                substitute_U!(Bshift23new, Bshift23)
                Bshift23 = shift_U(Bshift23new, (0, 1, 0, 0))
                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (0, 1, 0, 0))
            else # coordinate[2] < 0
                substitute_U!(Bshift23new, Bshift23)
                Bshift23 = shift_U(Bshift23new, (0, -1, 0, 0))
                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (0, -1, 0, 0))
            end
        end

        for iz = 1:abs(coordinate[3])
            if coordinate[3] > 0
                multiply_12!(uout, Ushift, Bshift23, 0, false, false)

                substitute_U!(Bshift23new, Bshift23)
                Bshift23 = shift_U(Bshift23new, (0, 0, 1, 0))
                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (0, 0, 1, 0))
            else # coordinate[3] < 0
                substitute_U!(Bshift23new, Bshift23)
                Bshift23 = shift_U(Bshift23new, (0, 0, -1, 0))
                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (0, 0, -1, 0))

                multiply_12!(uout, Ushift, Bshift23, 0, true, false)
            end

            substitute_U!(Unew, uout)
            Ushift = shift_U(Unew, origin)

        end

        for it = 1:abs(coordinate[4])
            if coordinate[4] > 0
                multiply_12!(uout, Ushift, Bshift24, 0, false, false)

                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (0, 0, 0, 1))
            else # coordinate[4] < 0
                substitute_U!(Bshift24new, Bshift24)
                Bshift24 = shift_U(Bshift24new, (0, 0, 0, -1))

                multiply_12!(uout, Ushift, Bshift24, 0, true, false)
            end

            substitute_U!(Unew, uout)
            Ushift = shift_U(Unew, origin)
        end
    elseif direction == 3
        if isU1dag
            Bshift34 = shift_U(B[3, 4], (0, 0, 0, 0))
        else
            Bshift34 = shift_U(B[3, 4], (0, 0, 0, 0))'
        end

        Bshift34new = temps[2]

        for ix = 1:abs(coordinate[1])
            if coordinate[1] > 0
                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (1, 0, 0, 0))
            else # coordinate[1] < 0
                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (-1, 0, 0, 0))
            end
        end

        for iy = 1:abs(coordinate[2])
            if coordinate[2] > 0
                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (0, 1, 0, 0))
            else # coordinate[2] < 0
                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (0, -1, 0, 0))
            end
        end

        for iz = 1:abs(coordinate[3])
            if coordinate[3] > 0
                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (0, 0, 1, 0))
            else # coordinate[3] < 0
                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (0, 0, -1, 0))
            end
        end

        for it = 1:abs(coordinate[4])
            if coordinate[4] > 0
                multiply_12!(uout, Ushift, Bshift34, 0, false, false)

                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (0, 0, 0, 1))
            else # coordinate[4] < 0
                substitute_U!(Bshift34new, Bshift34)
                Bshift34 = shift_U(Bshift34new, (0, 0, 0, -1))

                multiply_12!(uout, Ushift, Bshift34, 0, true, false)
            end

            substitute_U!(Unew, uout)
            Ushift = shift_U(Unew, origin)

        end
    else
        # direction==4: no multiplications
    end
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
    plaq = 0
    V = staple
    for μ = 1:Dim
        construct_staple!(V, U, B, μ, temp)
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
) where {NC,Dim,T1<:AbstractGaugefields,T2<:AbstractGaugefields}
    error("add_force! is not implemented in type $(typeof(F)) ")
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
    @assert length(temps) >= 3 "length(temps) should be >= 3. But $(length(temps))"

    V = temps[3]
    temp1 = temps[1]
    temp2 = temps[2]

    for μ = 1:Dim
        if plaqonly
            construct_double_staple!(V, U, μ, temps[1:2])
            mul!(temp1, U[μ], V') #U U*V
        else
            clear_U!(V)
            for i = 1:gparam.numactions
                loops = gparam.staples[i][μ]
                evaluate_wilson_loops!(temp3, loops, U, B, [temp1, temp2])
                add_U!(V, staplefactors[i], temp3)
            end
            mul!(temp1, U[μ], V) #U U*V
        end

        Traceless_antihermitian_add!(F[μ], factor, temp1)
    end
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
) where {NC,Dim,T<:AbstractGaugefields}
    U1U2 = temp
    firstterm = true

    for ν = 1:Dim
        if ν == μ
            continue
        end

        U1 = U[ν]
        # mul!(U1, U[ν], B[μ,ν]')
        if μ < ν
            mul!(U1, U[ν], B[μ, ν]')
        else
            mul!(U1, U[ν], B[μ, ν])
        end
        U2 = shift_U(U[μ], ν)
        mul!(U1U2, U1, U2)

        U3 = shift_U(U[ν], μ)
        if firstterm
            β = 0
            firstterm = false
        else
            β = 1
        end
        mul!(staple, U1U2, U3', 1, β)
    end
    set_wing_U!(staple)
end



end
