import ..MPILattice: LatticeMatrix,
    Shifted_Lattice,
    Adjoint_Lattice,
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
    get_2Dindex,
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
import LatticeMatrices: shift_L
abstract type Fields_2D_MPILattice{NC,NX,NY,T,AT,NDW} <: Gaugefields_2D{NC} end

Base.eltype(::Type{<:Fields_2D_MPILattice{NC,NX,NY,T}}) where {NC,NX,NY,T} = T
Base.eltype(U::Fields_2D_MPILattice) = eltype(typeof(U))

struct Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NDW} <: Fields_2D_MPILattice{NC,NX,NY,T,AT,NDW}
    U::LatticeMatrix{2,T,AT,NC,NC}
    mpi::Bool
    verbose_print::Verbose_print
    singleprecision::Bool
    NX::Int64
    NY::Int64
    #    NZ::Int64
    #    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64


    function Gaugefields_2D_MPILattice(NC, NX, NY;
        NDW=1, singleprecision=false,
        elementtype=nothing,
        boundarycondition=ones(2),
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

        gsize = (NX, NY)
        dim = 2
        nw = NDW
        @assert NDW >= 0 "NDW should be non-negative."
        elementtype, singleprecision =
            _resolve_mpialattice_elementtype(elementtype, singleprecision)
        phases = boundarycondition
        nprocs = MPI.Comm_size(comm)
        if isnothing(PEs)
            PEs_in = (1, nprocs)
        else
            PEs_in = deepcopy(PEs)
        end

        @assert NX > PEs_in[1] "PEs[1] is larger than NX. Now NX = $NX and PEs = $PEs_in"
        @assert NY > PEs_in[2] "PEs[2] is larger than NY. Now NX = $NY and PEs = $PEs_in"
        #@assert NZ > PEs_in[3] "PEs[3] is larger than NZ. Now NX = $NZ and PEs = $PEs_in"
        #@assert NT > PEs_in[4] "PEs[4] is larger than NT. Now NX = $NT and PEs = $PEs_in"

        @assert NX % PEs_in[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs_in"
        @assert NY % PEs_in[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs_in"
        #@assert NZ % PEs_in[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs_in"
        #@assert NT % PEs_in[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs_in"

        @assert prod(PEs_in) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)

        verbose_print = Verbose_print(verbose_level, myid=myrank)


        U = LatticeMatrix(NC, NC, dim, gsize, PEs_in;
            nw, elementtype, phases, comm0)
        T = elementtype
        AT = typeof(U.A)

        mpi = true

        NV = NX * NY# * NZ * NT

        return new{NC,NX,NY,T,AT,NDW}(
            U, mpi, verbose_print, singleprecision,
            NX,
            NY,
            #NZ,
            #NT,
            NDW,
            NV,
            NC)

        #LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)
    end
end

function Base.getproperty(U::Gaugefields_2D_MPILattice, name::Symbol)
    name === :NT && return getfield(U, :NY)
    return getfield(U, name)
end

function Base.propertynames(U::Gaugefields_2D_MPILattice, private::Bool=false)
    names = fieldnames(typeof(U))
    return private ? (names..., :NT) : (names..., :NT)
end

function Base.size(U::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NDW}) where {NC,NX,NY,T,AT,NDW}
    return NC, NC, NX, NY
end

@inline function kernel_getindex_2D_MPILattice!(_, output, input, i1, i2, i3, i4)
    @inbounds output[1] = input[i1, i2, i3, i4]
    return nothing
end

@inline function kernel_setindex_2D_MPILattice!(_, output, value, i1, i2, i3, i4)
    @inbounds output[i1, i2, i3, i4] = value
    return nothing
end

function Base.getindex(x::Gaugefields_2D_MPILattice, i1, i2, i3, i4)
    indices = (i1, i2, i3 + x.NDW, i4 + x.NDW)
    if x.U.A isa Array
        @inbounds return x.U.A[indices...]
    end
    output = JACC.zeros(eltype(x), 1)
    JACC.parallel_for(1, kernel_getindex_2D_MPILattice!, output, x.U.A, indices...)
    return JACC.to_host(output)[1]
end

function Base.setindex!(x::Gaugefields_2D_MPILattice, v, i1, i2, i3, i4)
    indices = (i1, i2, i3 + x.NDW, i4 + x.NDW)
    if x.U.A isa Array
        @inbounds x.U.A[indices...] = v
    else
        JACC.parallel_for(
            1, kernel_setindex_2D_MPILattice!, x.U.A, convert(eltype(x), v), indices...,
        )
    end
    mark_lattice_dirty!(x.U)
    return v
end

@inline function Base.getindex(x::Gaugefields_2D_MPILattice, i1, i2, ii)
    ix, it = get_latticeindex(ii, x.NX, x.NT)
    return x[i1, i2, ix, it]
end

function Base.setindex!(x::Gaugefields_2D_MPILattice, v, i1, i2, ii)
    ix, it = get_latticeindex(ii, x.NX, x.NT)
    x[i1, i2, ix, it] = v
    return v
end

@inline function getvalue(x::Gaugefields_2D_MPILattice, i1, i2, i3, i4)
    @inbounds return x.U.A[i1, i2, i3+x.NDW, i4+x.NDW]
end

@inline function setvalue!(x::Gaugefields_2D_MPILattice, v, i1, i2, i3, i4)
    @inbounds x.U.A[i1, i2, i3+x.NDW, i4+x.NDW] = v
    mark_lattice_dirty!(x.U)
    return v
end

get_myrank(U::Gaugefields_2D_MPILattice) = MPI.Comm_rank(U.U.comm)
get_myrank(U::Array{T,1}) where {T<:Gaugefields_2D_MPILattice} = get_myrank(U[1])
get_nprocs(U::Gaugefields_2D_MPILattice) = MPI.Comm_size(U.U.comm)
get_nprocs(U::Array{T,1}) where {T<:Gaugefields_2D_MPILattice} = get_nprocs(U[1])

function barrier(U::Gaugefields_2D_MPILattice)
    MPI.Barrier(U.U.comm)
    return nothing
end

function write_to_numpyarray(U::Gaugefields_2D_MPILattice, filename)
    global_U = gather_and_bcast_matrix(U.U)
    if get_myrank(U) == 0
        data = Dict{String,Any}(
            "U" => global_U,
            "NX" => U.NX,
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

#struct TA_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,NDW} <: Fields_2D_MPILattice{NC,NX,NY,T,AT,NDW}
#    U::TALattice{4,T,AT,NC}
#end

struct Shifted_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,shift,nw,L} <: Fields_2D_MPILattice{NC,NX,NY,T,AT,nw}
    U::Shifted_Lattice{L,2} #L<: LatticeMatrix{2,T,AT,NC,NC,nw}

    function Shifted_Gaugefields_2D_MPILattice(U::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,nw}, shift) where {NC,NX,NY,T,AT,nw}
        sU = shift_L(U.U, shift)
        shiftin = get_shift(sU)
        return new{NC,NX,NY,T,AT,shiftin,nw,typeof(U.U)}(sU)
    end
end


struct Adjoint_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,nw,L} <: Fields_2D_MPILattice{NC,NX,NY,T,AT,nw}
    U::Adjoint_Lattice{L} #LatticeMatrix{2,T,AT,NC,NC,nw}
end

struct Adjoint_Shifted_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,shift,nw,L} <: Fields_2D_MPILattice{NC,NX,NY,T,AT,nw}
    U::Adjoint_Lattice{Shifted_Lattice{L,2}} #LatticeMatrix{2,T,AT,NC,NC,nw}
end

function Base.getindex(
    x::Shifted_Gaugefields_2D_MPILattice,
    i1,
    i2,
    i3,
    i4,
)
    shift = get_shift(x.U)
    data = x.U.data
    indices = (
        i1,
        i2,
        i3 + data.nw + shift[1],
        i4 + data.nw + shift[2],
    )
    if data.A isa Array
        @inbounds return data.A[indices...]
    end
    output = JACC.zeros(eltype(x), 1)
    JACC.parallel_for(
        1, kernel_getindex_2D_MPILattice!, output, data.A, indices...,
    )
    return JACC.to_host(output)[1]
end

function Base.getindex(
    u::Staggered_Gaugefields{T,direction},
    i1,
    i2,
    i3,
    i4,
) where {T<:Gaugefields_2D_MPILattice,direction}
    1 <= direction <= 2 || throw(ArgumentError(
        "staggered direction must be in 1:2",
    ))
    data = u.parent.U
    global_x_zero_based = data.coords[1] * data.PN[1] + i3 - 1
    phase = direction == 1 || iseven(global_x_zero_based) ? 1 : -1
    return phase * u.parent[i1, i2, i3, i4]
end

@inline _release_shifted_U!(shifted::Shifted_Gaugefields_2D_MPILattice) =
    release_lattice!(getfield(shifted, :U))
@inline _release_shifted_U!(shifted::Adjoint_Shifted_Gaugefields_2D_MPILattice) =
    release_lattice!(getfield(shifted, :U))

Base.close(shifted::Shifted_Gaugefields_2D_MPILattice) =
    _release_shifted_U!(shifted)
Base.close(shifted::Adjoint_Shifted_Gaugefields_2D_MPILattice) =
    _release_shifted_U!(shifted)
Base.isopen(shifted::Shifted_Gaugefields_2D_MPILattice) =
    lattice_isopen(getfield(shifted, :U))
Base.isopen(shifted::Adjoint_Shifted_Gaugefields_2D_MPILattice) =
    lattice_isopen(getfield(shifted, :U))






function identityGaugefields_2D_MPILattice(NC, NX, NY;
    NDW=1,
    verbose_level=2,
    singleprecision=false,
    elementtype=nothing,
    boundarycondition=ones(4),
    PEs=nothing,
    comm=MPI.COMM_WORLD,
    #mpiinit=false
)


    U = Gaugefields_2D_MPILattice(NC, NX, NY;
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

function randomGaugefields_2D_MPILattice(NC, NX, NY;
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


    U = Gaugefields_2D_MPILattice(NC, NX, NY;
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
    c::Gaugefields_2D_MPILattice{NC},
    a::T1,
    b::T2,
) where {NC,T1<:Abstractfields,T2<:Abstractfields}
    #@assert NC != 2 && NC != 3 "This function is for NC != 2,3"

    mul!(c, a, b)

end

function partial_tr(a::Gaugefields_2D_MPILattice{NC}, μ) where {NC}
    s = partial_trace(a.U, μ)
    return s
end


function set_wing_U!(u::Array{Gaugefields_2D_MPILattice{NC},1}) where {NC}
    for i = 1:length(u)
        set_halo!(u[i].U)
    end
    return
end

function set_wing_U!(u::Gaugefields_2D_MPILattice{NC}) where {NC}
    set_halo!(u.U)
    return
end

function substitute_U!(a::Gaugefields_2D_MPILattice, b::Gaugefields_2D_MPILattice)
    substitute!(a.U, b.U)
    set_wing_U!(a)
end

function substitute_U!(a::Gaugefields_2D_MPILattice, b::Shifted_Gaugefields_2D_MPILattice)
    substitute!(a.U, b.U)
    set_wing_U!(a)
end

function substitute_U!(
    a::Gaugefields_2D_MPILattice,
    b::Adjoint_Shifted_Gaugefields_2D_MPILattice,
)
    substitute!(a.U, b.U)
    set_wing_U!(a)
end

function substitute_U!(
    a::Gaugefields_2D_MPILattice,
    b::T,
    target_even::Bool,
) where {T<:Fields_2D_MPILattice}
    clear_U!(a, target_even)
    add_U!(a, b, target_even)
    set_wing_U!(a)
    return nothing
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_2D_MPILattice,T2<:Gaugefields_2D_MPILattice}
    for μ = 1:2
        substitute_U!(a[μ], b[μ])
    end
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_2D_nowing,T2<:Gaugefields_2D_MPILattice}

    for μ = 1:2
        substitute_U!(a[μ], b[μ])
    end
end



function substitute_U!(A::Gaugefields_2D_nowing, B::Gaugefields_2D_MPILattice)
    tempmatrix = gather_and_bcast_matrix(B.U)
    A.U .= tempmatrix
end


function substitute_U!(A::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT},
    B::Gaugefields_2D_nowing{NC}) where {NC,NX,NY,T,AT}

    dim = 2
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
) where {T1<:Gaugefields_2D_MPILattice,T2<:Gaugefields_2D_nowing}

    for μ = 1:2
        substitute_U!(a[μ], b[μ])
    end
end

function ges_PEs(U::Gaugefields_2D_MPILattice)
    return U.U.dims
end

function Base.similar(U::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT}) where {NC,NX,NY,T,AT}
    NDW = U.U.nw
    boundarycondition = U.U.phases
    PEs = ges_PEs(U)
    comm = U.U.comm

    Uout = Gaugefields_2D_MPILattice(
        NC, NX, NY;
        NDW,
        U.singleprecision,
        elementtype=T,
        boundarycondition,
        PEs,
        comm,
        verbose_level=U.verbose_print.level
    )
    #identityGaugefields_2D_nowing(U.NC,U.NX,U.NY,U.NZ,U.NT,U.NDW)
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Gaugefields_2D_MPILattice}
    Uout = Array{T,1}(undef, 2)
    for μ = 1:2
        Uout[μ] = similar(U[μ])
    end
    return Uout
end

function map_U_sequential!(U::Gaugefields_2D_MPILattice{NC}, f!::Function, Uin) where {NC}
    get_nprocs(U) == 1 ||
        error("The function map_U_sequential! can not be used with MPI")

    host_U = gather_and_bcast_matrix(U.U)
    B = Matrix{eltype(U)}(undef, NC, NC)
    for it = 1:U.NT
        for ix = 1:U.NX
            @views B .= host_U[:, :, ix, it]
            f!(B, Uin, ix, it)
            @views host_U[:, :, ix, it] .= B
        end
    end
    substitute!(U.U, host_U)
    set_wing_U!(U)
    return nothing
end


function shift_U(U::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,nw}, ν::Ts) where {Ts<:Integer,T,NC,NX,NY,AT,nw}
    if ν == 1
        shift = (1, 0)
    elseif ν == 2
        shift = (0, 1)
        ##elseif ν == 3
        #    shift = (0, 0, 1, 0)
        #elseif ν == 4
        #    shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0)
    elseif ν == -2
        shift = (0, -1)
        #elseif ν == -3
        #    shift = (0, 0, -1, 0)
        #elseif ν == -4
        #    shift = (0, 0, 0, -1)
    end

    return shift_U(U, shift)
end


function shift_U(U::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,nw}, shift::NTuple{2,Ts}) where {NC,NX,NY,T,AT,Ts<:Int,nw}
    return Shifted_Gaugefields_2D_MPILattice(U, shift)
end




function Base.adjoint(U::Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,nw}) where {NC,NX,NY,T,AT,nw}
    Adjoint_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,nw,typeof(U.U)}(U.U')
end




function Base.adjoint(U::Shifted_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,shift,nw,L}) where {L,NC,NX,NY,T,AT,shift,nw}
    Adjoint_Shifted_Gaugefields_2D_MPILattice{NC,NX,NY,T,AT,shift,nw,L}(U.U')
end

function LinearAlgebra.mul!(
    c::T,
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {T<:Gaugefields_2D_MPILattice,T1<:Fields_2D_MPILattice,T2<:Fields_2D_MPILattice,Ta<:Number,Tb<:Number}
    mul!(c.U, a.U, b.U, α, β)
end

function LinearAlgebra.mul!(
    c::T,
    a::T1,
    b::T2,
    target_even::Bool,
) where {T<:Gaugefields_2D_MPILattice,T1<:Fields_2D_MPILattice,T2<:Fields_2D_MPILattice}
    mul!(c.U, a.U, b.U, target_even)
end

function LinearAlgebra.tr(a::Gaugefields_2D_MPILattice)
    tr(a.U)
end

function LinearAlgebra.tr(a::Gaugefields_2D_MPILattice, b::Gaugefields_2D_MPILattice)
    tr(a.U, b.U)
end


function clear_U!(c::Gaugefields_2D_MPILattice)
    clear_matrix!(c.U)
end

function clear_U!(c::Gaugefields_2D_MPILattice, target_even::Bool)
    clear_matrix!(c.U, target_even)
end

function add_U!(c::Gaugefields_2D_MPILattice, t::T, a::T1) where {T1<:Fields_2D_MPILattice,T<:Number}
    add_matrix!(c.U, a.U, t)
end

function add_U!(c::Gaugefields_2D_MPILattice, a::T1) where {T1<:Fields_2D_MPILattice}
    add_matrix!(c.U, a.U)
end

function add_U!(
    c::Gaugefields_2D_MPILattice,
    a::T1,
    target_even::Bool,
) where {T1<:Fields_2D_MPILattice}
    add_matrix_evenodd!(c.U, a.U, target_even)
end

function add_U!(
    c::Gaugefields_2D_MPILattice,
    α::T,
    a::T1,
    target_even::Bool,
) where {T<:Number,T1<:Fields_2D_MPILattice}
    add_matrix_evenodd!(c.U, a.U, target_even, α)
end

function map_U!(
    U::Gaugefields_2D_MPILattice,
    f!::Function,
    V::Gaugefields_2D_MPILattice,
    target_even::Bool,
)
    map_matrix_evenodd!(U.U, V.U, f!, target_even)
    return nothing
end

function Traceless_antihermitian!(
    vout::Gaugefields_2D_MPILattice,
    vin::Gaugefields_2D_MPILattice,
)

    traceless_antihermitian!(vout.U, vin.U)

end

function exptU!(
    uout::Tg,
    t::N,
    v::Gaugefields_2D_MPILattice,
    temps::Array{Tg,1},
) where {Tg<:Gaugefields_2D_MPILattice,N<:Number}
    expt!(uout.U, v.U, t)
    set_wing_U!(uout)
    return nothing
end

function unit_U!(U::Gaugefields_2D_MPILattice)
    makeidentity_matrix!(U.U)
    set_wing_U!(U)
    return nothing
end
