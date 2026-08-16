import ..MPILattice:
    TALattice,
    makeidentity_matrix!,
    set_halo!,
    substitute!,
    partial_trace,
    get_PEs,
    clear_matrix!,
    add_matrix!,
    add_matrix_evenodd!,
    map_matrix_evenodd!,
    expt!,
    get_4Dindex,
    traceless_antihermitian_add!,
    normalize_matrix!,
    randomize_matrix!,
    randomize_gaussian_matrix!,
    RNGStreamKey,
    SiteRNGAlgorithm,
    Philox4x32,
    get_shift,
    gather_and_bcast_matrix,
    traceless_antihermitian!
import LatticeMatrices: LatticeMatrix,
    Shifted_Lattice,
    Adjoint_Lattice, delinearize, shift_L, Traceless_AntiHermitian


abstract type Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,DI} <: Gaugefields_4D{NC} end

Base.eltype(::Type{<:Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T}}) where {NC,NX,NY,NZ,NT,T} = T
Base.eltype(U::Fields_4D_MPILattice) = eltype(typeof(U))

struct Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,DI,TU<:LatticeMatrix{4,T,AT,NC,NC}} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,DI}
    U::TU
    mpi::Bool
    verbose_print::Verbose_print
    singleprecision::Bool
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64


    function Gaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
        NDW=1, singleprecision=false,
        elementtype=nothing,
        boundarycondition=ones(4),
        PEs=nothing,
        comm=MPI.COMM_WORLD,
        #mpiinit=false,
        verbose_level=2
    )
        if MPI.Initialized() == false
            MPI.Init()
            mpiinit = true
        end

        #if mpiinit == false
        #    MPI.Init()
        #    mpiinit = true
        #end
        comm0 = comm

        gsize = (NX, NY, NZ, NT)
        dim = 4
        nw = NDW
        @assert NDW >= 0 "NDW should be non-negative."
        elementtype, singleprecision =
            _resolve_mpialattice_elementtype(elementtype, singleprecision)
        phases = boundarycondition
        nprocs = MPI.Comm_size(comm)
        if isnothing(PEs)
            PEs_in = (1, 1, 1, nprocs)
        else
            PEs_in = deepcopy(PEs)
        end

        @assert NX > PEs_in[1] "PEs[1] is larger than NX. Now NX = $NX and PEs = $PEs_in"
        @assert NY > PEs_in[2] "PEs[2] is larger than NY. Now NX = $NY and PEs = $PEs_in"
        @assert NZ > PEs_in[3] "PEs[3] is larger than NZ. Now NX = $NZ and PEs = $PEs_in"
        @assert NT > PEs_in[4] "PEs[4] is larger than NT. Now NX = $NT and PEs = $PEs_in"

        @assert NX % PEs_in[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs_in"
        @assert NY % PEs_in[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs_in"
        @assert NZ % PEs_in[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs_in"
        @assert NT % PEs_in[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs_in"

        @assert prod(PEs_in) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)

        verbose_print = Verbose_print(verbose_level, myid=myrank)


        U = LatticeMatrix(NC, NC, dim, gsize, PEs_in;
            nw, elementtype, phases, comm0)
        T = elementtype
        AT = typeof(U.A)

        mpi = true

        NV = NX * NY * NZ * NT
        DI = typeof(U.indexer)
        TU = typeof(U)

        return new{NC,NX,NY,NZ,NT,T,AT,NDW,DI,TU}(
            U, mpi, verbose_print, singleprecision,
            NX,
            NY,
            NZ,
            NT,
            NDW,
            NV,
            NC)

        #LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)
    end
end

get_myrank(U::Gaugefields_4D_MPILattice) = MPI.Comm_rank(U.U.comm)
get_myrank(U::Array{T,1}) where {T<:Gaugefields_4D_MPILattice} = get_myrank(U[1])
get_nprocs(U::Gaugefields_4D_MPILattice) = MPI.Comm_size(U.U.comm)
get_nprocs(U::Array{T,1}) where {T<:Gaugefields_4D_MPILattice} = get_nprocs(U[1])

function barrier(U::Gaugefields_4D_MPILattice)
    MPI.Barrier(U.U.comm)
    return nothing
end

function write_to_numpyarray(U::Gaugefields_4D_MPILattice, filename)
    global_U = gather_and_bcast_matrix(U.U)
    if get_myrank(U) == 0
        data = Dict{String,Any}(
            "U" => global_U,
            "NX" => U.NX,
            "NY" => U.NY,
            "NZ" => U.NZ,
            "NT" => U.NT,
            "NV" => U.NV,
            "NDW" => U.NDW,
            "NC" => U.NC,
        )
        npzwrite(filename, data)
    end
    barrier(U)
    return nothing
end

@inline function kernel_getindex_4D_MPILattice!(
    _, output, input, i1, i2, i3, i4, i5, i6,
)
    @inbounds output[1] = input[i1, i2, i3, i4, i5, i6]
    return nothing
end

@inline function kernel_setindex_4D_MPILattice!(
    _, output, value, i1, i2, i3, i4, i5, i6,
)
    @inbounds output[i1, i2, i3, i4, i5, i6] = value
    return nothing
end

function Base.getindex(x::Gaugefields_4D_MPILattice, i1, i2, i3, i4, i5, i6)
    indices = (i1, i2, i3 + x.NDW, i4 + x.NDW, i5 + x.NDW, i6 + x.NDW)
    if x.U.A isa Array
        @inbounds return x.U.A[indices...]
    end
    output = JACC.zeros(eltype(x), 1)
    JACC.parallel_for(1, kernel_getindex_4D_MPILattice!, output, x.U.A, indices...)
    return JACC.to_host(output)[1]
end

function Base.setindex!(x::Gaugefields_4D_MPILattice, v, i1, i2, i3, i4, i5, i6)
    indices = (i1, i2, i3 + x.NDW, i4 + x.NDW, i5 + x.NDW, i6 + x.NDW)
    if x.U.A isa Array
        @inbounds x.U.A[indices...] = v
    else
        JACC.parallel_for(
            1, kernel_setindex_4D_MPILattice!, x.U.A, convert(eltype(x), v), indices...,
        )
    end
    mark_lattice_dirty!(x.U)
    return v
end

@inline function getvalue(x::Gaugefields_4D_MPILattice, i1, i2, i3, i4, i5, i6)
    @inbounds return x.U.A[
        i1, i2, i3+x.NDW, i4+x.NDW, i5+x.NDW, i6+x.NDW]
end

@inline function setvalue!(x::Gaugefields_4D_MPILattice, v, i1, i2, i3, i4, i5, i6)
    @inbounds x.U.A[
        i1, i2, i3+x.NDW, i4+x.NDW, i5+x.NDW, i6+x.NDW] = v
    mark_lattice_dirty!(x.U)
    return v
end

function map_U_sequential!(U::Gaugefields_4D_MPILattice{NC}, f!::Function, Uin) where {NC}
    get_nprocs(U) == 1 ||
        error("The function map_U_sequential! can not be used with MPI")

    host_U = gather_and_bcast_matrix(U.U)
    B = Matrix{eltype(U)}(undef, NC, NC)
    for it = 1:U.NT
        for iz = 1:U.NZ
            for iy = 1:U.NY
                for ix = 1:U.NX
                    @views B .= host_U[:, :, ix, iy, iz, it]
                    f!(B, Uin, ix, iy, iz, it)
                    @views host_U[:, :, ix, iy, iz, it] .= B
                end
            end
        end
    end
    substitute!(U.U, host_U)
    set_wing_U!(U)
    return nothing
end



#struct TA_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,DI} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW}
#    U::TALattice{4,T,AT,NC}
#end

struct Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI,L} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI}
    U::Shifted_Lattice{L,4}

    @inline function Shifted_Gaugefields_4D_MPILattice(U::TU, shift) where {NC,NX,NY,NZ,NT,T,AT,nw,DI,TM,TU<:Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI,TM}}
        #sU = Shifted_Lattice{typeof(U.U),shift}(U.U)
        #sU = Shifted_Lattice(U.U, shift)
        sU = shift_L(U.U, shift)
        s = new{NC,NX,NY,NZ,NT,T,AT,nw,DI,TM}(sU)
        return s
    end
end



struct Adjoint_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI,L} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI}
    U::Adjoint_Lattice{L} #LatticeMatrix{4,T,AT,NC,NC,nw,DI}
end

struct Adjoint_Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI,L} <: Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI}
    U::Adjoint_Lattice{Shifted_Lattice{L,4}} #LatticeMatrix{4,T,AT,NC,NC,nw,DI}
end

function Base.getindex(
    x::Shifted_Gaugefields_4D_MPILattice,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
)
    shift = get_shift(x.U)
    data = x.U.data
    indices = (
        i1,
        i2,
        i3 + data.nw + shift[1],
        i4 + data.nw + shift[2],
        i5 + data.nw + shift[3],
        i6 + data.nw + shift[4],
    )
    if data.A isa Array
        @inbounds return data.A[indices...]
    end
    output = JACC.zeros(eltype(x), 1)
    JACC.parallel_for(
        1, kernel_getindex_4D_MPILattice!, output, data.A, indices...,
    )
    return JACC.to_host(output)[1]
end

@inline function _staggered_phase_4d(direction, coordinates)
    1 <= direction <= 4 || throw(ArgumentError(
        "staggered direction must be in 1:4",
    ))
    coordinate_sum = zero(eltype(coordinates))
    @inbounds for d in 1:(direction-1)
        coordinate_sum += coordinates[d]
    end
    return iseven(coordinate_sum) ? 1 : -1
end

function Base.getindex(
    u::Staggered_Gaugefields{T,direction},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:Gaugefields_4D_MPILattice,direction}
    data = u.parent.U
    local_coordinates = (i3, i4, i5, i6)
    global_zero_based = ntuple(
        d -> data.coords[d] * data.PN[d] + local_coordinates[d] - 1,
        4,
    )
    phase = _staggered_phase_4d(direction, global_zero_based)
    return phase * u.parent[i1, i2, i3, i4, i5, i6]
end

function Base.getindex(
    u::Staggered_Gaugefields{T,direction},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:Shifted_Gaugefields_4D_MPILattice,direction}
    shifted = u.parent.U
    shift = get_shift(shifted)
    data = shifted.data
    local_coordinates = (i3, i4, i5, i6)
    global_zero_based = ntuple(
        d -> mod(
            data.coords[d] * data.PN[d] + local_coordinates[d] - 1 + shift[d],
            data.gsize[d],
        ),
        4,
    )
    phase = _staggered_phase_4d(direction, global_zero_based)
    return phase * u.parent[i1, i2, i3, i4, i5, i6]
end

@inline _release_shifted_U!(shifted::Shifted_Gaugefields_4D_MPILattice) =
    release_lattice!(getfield(shifted, :U))
@inline _release_shifted_U!(shifted::Adjoint_Shifted_Gaugefields_4D_MPILattice) =
    release_lattice!(getfield(shifted, :U))

Base.close(shifted::Shifted_Gaugefields_4D_MPILattice) =
    _release_shifted_U!(shifted)
Base.close(shifted::Adjoint_Shifted_Gaugefields_4D_MPILattice) =
    _release_shifted_U!(shifted)
Base.isopen(shifted::Shifted_Gaugefields_4D_MPILattice) =
    lattice_isopen(getfield(shifted, :U))
Base.isopen(shifted::Adjoint_Shifted_Gaugefields_4D_MPILattice) =
    lattice_isopen(getfield(shifted, :U))

function evaluate_gaugelinks!(
    uout::T,
    w::Wilsonline{Dim},
    U::Vector{T},
    temps::Vector{T},
) where {T<:Gaugefields_4D_MPILattice,Dim}
    glinks = w
    numlinks = length(glinks)

    if numlinks == 0
        unit_U!(uout)
        return
    end

    Acc = temps[1]
    Tmp = temps[2]

    link1 = glinks[1]
    isU1dag = isdag(link1)
    U_initial = shift_U(U[get_direction(link1)], get_position(link1))

    if numlinks == 1
        if isU1dag
            substitute_U!(Acc, U_initial')
        else
            substitute_U!(Acc, U_initial)
        end
        _release_shifted_U!(U_initial)
        substitute_U!(uout, Acc)
        return
    end

    # Leave the shifted field with LatticeMatrices instead of copying it to Acc.
    link2 = glinks[2]
    Ushift_2 = shift_U(U[get_direction(link2)], get_position(link2))
    multiply_12!(Acc, U_initial, Ushift_2, 2, isdag(link2), isU1dag)
    _release_shifted_U!(Ushift_2)
    _release_shifted_U!(U_initial)
    Uaccumulated = Acc

    for k = 3:numlinks
        linkk = glinks[k]
        Ushift_k = shift_U(U[get_direction(linkk)], get_position(linkk))
        multiply_12!(Tmp, Uaccumulated, Ushift_k, k, isdag(linkk), false)
        _release_shifted_U!(Ushift_k)

        Acc, Tmp = Tmp, Acc
        Uaccumulated = Acc
    end

    substitute_U!(uout, Acc)
    return
end


function identityGaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
    NDW=1,
    verbose_level=2,
    singleprecision=false,
    elementtype=nothing,
    boundarycondition=ones(4),
    PEs=nothing,
    comm=MPI.COMM_WORLD,
    #mpiinit=false
)


    U = Gaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
        NDW,
        singleprecision,
        elementtype,
        boundarycondition,
        PEs,
        comm,
        verbose_level
    )

    makeidentity_matrix!(U.U)
    return U

end

function randomGaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
    NDW=1,
    verbose_level=2,
    singleprecision=false,
    elementtype=nothing,
    boundarycondition=ones(4),
    PEs=nothing,
    comm=MPI.COMM_WORLD,
    randomnumber="Random",
    seed=nothing,
    rng_algorithm::SiteRNGAlgorithm=Philox4x32(),
    direction::Integer=0,
    #mpiinit=false
)


    U = Gaugefields_4D_MPILattice(NC, NX, NY, NZ, NT;
        NDW,
        singleprecision,
        elementtype,
        boundarycondition,
        PEs,
        comm,
        verbose_level
    )

    if randomnumber != "Random" && randomnumber != "Reproducible"
        error(
            "randomnumber should be \"Random\" or \"Reproducible\". Now randomnumber = $randomnumber",
        )
    end

    shared_seed = _shared_mpialattice_seed(
        seed,
        U.U.comm;
        reproducible=randomnumber == "Reproducible",
    )
    key = RNGStreamKey(shared_seed, 0, direction, 0, _HOT_START_STREAM_TAG)
    randomize_matrix!(U.U, key; rng_algorithm)
    normalize_matrix!(U.U)
    return U

end

function mul_skiplastindex!(
    c::Gaugefields_4D_MPILattice{NC},
    a::T1,
    b::T2,
) where {NC,T1<:Abstractfields,T2<:Abstractfields}
    #@assert NC != 2 && NC != 3 "This function is for NC != 2,3"

    mul!(c, a, b)

end

import LatticeMatrices
Base.@noinline function LatticeMatrices.realtrace(C::T) where {NC,T<:Gaugefields_4D_MPILattice{NC}}
    return LatticeMatrices.realtrace(C.U)
end


function partial_tr(a::Gaugefields_4D_MPILattice{NC}, μ) where {NC}
    s = partial_trace(a.U, μ)
    return s
end


@inline function set_wing_U!(u::Array{Gaugefields_4D_MPILattice{NC},1}) where {NC}
    for i = 1:length(u)
        set_halo!(u[i].U)
    end
    return
end

@inline function set_wing_U!(u::Gaugefields_4D_MPILattice{NC}) where {NC}
    set_halo!(u.U)
    return
end

function substitute_U!(a::Gaugefields_4D_MPILattice, b::Gaugefields_4D_MPILattice)
    substitute!(a.U, b.U)
    set_wing_U!(a)
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_MPILattice,T2<:Gaugefields_4D_MPILattice}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end


function substitute_U!(a::Gaugefields_4D_MPILattice, b::Shifted_Gaugefields_4D_MPILattice)
    substitute!(a.U, b.U)
    set_wing_U!(a)
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_MPILattice,T2<:Shifted_Gaugefields_4D_MPILattice}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end


function substitute_U!(a::Gaugefields_4D_MPILattice, b::Adjoint_Shifted_Gaugefields_4D_MPILattice)
    substitute!(a.U, b.U)
    set_wing_U!(a)
end

function substitute_U!(
    a::Gaugefields_4D_MPILattice,
    b::T,
    target_even::Bool,
) where {T<:Fields_4D_MPILattice}
    clear_U!(a, target_even)
    add_U!(a, b, target_even)
    set_wing_U!(a)
    return nothing
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_MPILattice,T2<:Adjoint_Shifted_Gaugefields_4D_MPILattice}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_nowing,T2<:Gaugefields_4D_MPILattice}

    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end



function substitute_U!(A::Gaugefields_4D_nowing, B::Gaugefields_4D_MPILattice)
    tempmatrix = gather_and_bcast_matrix(B.U)
    A.U .= tempmatrix
end


function substitute_U!(A::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT},
    B::Gaugefields_4D_nowing{NC}) where {NC,NX,NY,NZ,NT,T,AT}
    dim = 4
    PEs = A.U.dims
    phases = A.U.phases
    nw = A.U.nw
    comm0 = A.U.comm

    tempU = LatticeMatrix(B.U, dim, PEs;
        nw,
        phases,
        comm0)
    substitute!(A.U, tempU)
    set_halo!(A.U)
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_MPILattice,T2<:Gaugefields_4D_nowing}

    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function ges_PEs(U::Gaugefields_4D_MPILattice)
    return U.U.dims
end

function Base.similar(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT}) where {NC,NX,NY,NZ,NT,T,AT}
    NDW = U.U.nw
    boundarycondition = U.U.phases
    PEs = ges_PEs(U)
    comm = U.U.comm

    Uout = Gaugefields_4D_MPILattice(
        NC, NX, NY, NZ, NT;
        NDW,
        U.singleprecision,
        elementtype=T,
        boundarycondition,
        PEs,
        comm,
        verbose_level=U.verbose_print.level
    )
    #identityGaugefields_4D_nowing(U.NC,U.NX,U.NY,U.NZ,U.NT,U.NDW)
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Gaugefields_4D_MPILattice}
    Uout = Array{T,1}(undef, 4)
    for μ = 1:4
        Uout[μ] = similar(U[μ])
    end
    return Uout
end



@inline function shift_U(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI}, ν::Ts) where {Ts<:Integer,T,NC,NX,NY,NZ,NT,AT,nw,DI}
    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    s = shift_U(U, shift)

    return s
end


@inline function shift_U(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI}, shift::NTuple{4,Ts}) where {NC,NX,NY,NZ,NT,T,AT,Ts<:Int,nw,DI}
    s = Shifted_Gaugefields_4D_MPILattice(U, shift)
    return s
end




function Base.adjoint(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI}) where {NC,NX,NY,NZ,NT,T,AT,nw,DI}
    Adjoint_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI,typeof(U.U)}(U.U')
end




function Base.adjoint(U::Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI,L}) where {L,NC,NX,NY,NZ,NT,T,AT,nw,DI}
    Adjoint_Shifted_Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,nw,DI,L}(U.U')
end

import LatticeMatrices: mul_AtransB!
@inline function mul_AtransB!(
    c::T,
    a::T1,
    b::T2
) where {T<:Gaugefields_4D_MPILattice,T1<:Fields_4D_MPILattice,T2<:Fields_4D_MPILattice}
    mul_AtransB!(c.U, a.U, b.U)
end


@inline function LinearAlgebra.mul!(
    c::T,
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {T<:Gaugefields_4D_MPILattice,T1<:Fields_4D_MPILattice,T2<:Fields_4D_MPILattice,Ta<:Number,Tb<:Number}
    mul!(c.U, a.U, b.U, α, β)

end

@inline function LinearAlgebra.mul!(
    c::T,
    a::T1,
    b::T2,
    target_even::Bool,
) where {T<:Gaugefields_4D_MPILattice,T1<:Fields_4D_MPILattice,T2<:Fields_4D_MPILattice}
    mul!(c.U, a.U, b.U, target_even)
end

@inline function LinearAlgebra.tr(a::Gaugefields_4D_MPILattice)
    tr(a.U)
    #set_halo!(a.U)
end

@inline function LinearAlgebra.tr(a::Gaugefields_4D_MPILattice, b::Gaugefields_4D_MPILattice)
    tr(a.U, b.U)
end


@inline function clear_U!(c::Gaugefields_4D_MPILattice)
    clear_matrix!(c.U)
end

@inline function clear_U!(c::Gaugefields_4D_MPILattice, iseven::Bool)
    clear_matrix!(c.U, iseven)
end

@inline function add_U!(c::Gaugefields_4D_MPILattice, t::T, a::T1) where {T1<:Fields_4D_MPILattice,T<:Number}
    add_matrix!(c.U, a.U, t)
end

@inline function add_U!(c::Gaugefields_4D_MPILattice, a::T1) where {T1<:Fields_4D_MPILattice}
    add_matrix!(c.U, a.U)
end

@inline function add_U!(
    c::Gaugefields_4D_MPILattice,
    a::T1,
    target_even::Bool,
) where {T1<:Fields_4D_MPILattice}
    add_matrix_evenodd!(c.U, a.U, target_even)
end

@inline function add_U!(
    c::Gaugefields_4D_MPILattice,
    α::T,
    a::T1,
    target_even::Bool,
) where {T<:Number,T1<:Fields_4D_MPILattice}
    add_matrix_evenodd!(c.U, a.U, target_even, α)
end

@inline function map_U!(
    U::Gaugefields_4D_MPILattice,
    f!::Function,
    V::Gaugefields_4D_MPILattice,
    target_even::Bool,
)
    map_matrix_evenodd!(U.U, V.U, f!, target_even)
    return nothing
end

@inline function Traceless_antihermitian!(
    vout::Gaugefields_4D_MPILattice,
    vin::Gaugefields_4D_MPILattice,
)

    traceless_antihermitian!(vout.U, vin.U)

end

@inline function Traceless_AntiHermitian(C::Gaugefields_4D_MPILattice)
    return Traceless_AntiHermitian(C.U)
end
export Traceless_AntiHermitian

@inline function exptU!(C::TC, A::Traceless_AntiHermitian{L}, t=1) where {
    L,TC<:Gaugefields_4D_MPILattice}
    # Always use TA-specialized path (forward + Enzyme custom reverse).
    LatticeMatrices.expt_TA!(C.U, A.data, t)
    return
    #set_halo!(C)
end



function exptU!(
    uout::Tg,
    t::N,
    v::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis},
    temps::Array{Tg,1},
) where {NC,NX,NY,NZ,NT,T,AT,NumofBasis,Tg<:Gaugefields_4D_MPILattice,N<:Number} #uout = exp(t*u)

    expt!(uout.U, v.U, t)
    set_wing_U!(uout)
    return nothing
end


function unit_U!(U::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NumofBasis}
) where {NC,NX,NY,NZ,NT,T,AT,NumofBasis}
    makeidentity_matrix!(U.U)
    set_wing_U!(U)
end

function lambda_k_mul!(
    b::Gaugefields_4D_MPILattice{NC},
    a::Gaugefields_4D_MPILattice{NC},
    k,
    generator,
) where NC
    1 <= k <= length(generator) || throw(BoundsError(generator, k))
    mul!(b.U, generator.generator[k] / 2, a.U)
    set_wing_U!(b)
    return nothing
end

function Antihermitian!(
    vout::Gaugefields_4D_MPILattice{NC},
    vin::Gaugefields_4D_MPILattice{NC};
    factor=1,
) where NC
    vout === vin && throw(ArgumentError(
        "in-place Antihermitian! requires distinct output and input fields",
    ))
    clear_matrix!(vout.U)
    add_matrix!(vout.U, vin.U, factor)
    add_matrix!(vout.U, vin.U', -factor)
    set_wing_U!(vout)
    return nothing
end
