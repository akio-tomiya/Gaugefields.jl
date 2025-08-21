##############################################################################
#  LatticeVector (no derived datatypes version)
#  --------------------------------------
#  * column-major layout   :  (NC , X , Y , …)
#  * halo width            :  nw
#  * per–direction phases  :  φ
#  * internal DoF          :  NC  (fastest dim)
#  * ALWAYS packs faces into contiguous buffers and sends them as
#    plain arrays (no MPI_Type_create_subarray, no commit/free hustle).
#
#  Back-end: CPU threads / CUDA / ROCm via JACC.
##############################################################################

using MPI, StaticArrays, JACC

# ---------------------------------------------------------------------------
# container  (faces / derived datatypes are GONE)
# ---------------------------------------------------------------------------
struct LatticeVector{D,T,AT} <: Lattice{D,T,AT}
    nw::Int                          # ghost width
    phases::SVector{D,T}                 # phases
    NC::Int                          # internal DoF
    gsize::NTuple{D,Int}                # global size

    cart::MPI.Comm
    coords::NTuple{D,Int}
    dims::NTuple{D,Int}
    nbr::NTuple{D,NTuple{2,Int}}

    A::AT                           # main array (NC first)
    buf::Vector{AT}                   # 2D work buffers (minus/plus)
    myrank::Int
    PN::NTuple{D,Int}
end

# ---------------------------------------------------------------------------
# constructor + heavy init (still cheap to call)
# ---------------------------------------------------------------------------
function LatticeVector(NC, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)

    # Cartesian grid
    D = dim
    T = elementtype
    dims = PEs #MPI.dims_create(MPI.Comm_size(MPI.COMM_WORLD), D)
    cart = MPI.Cart_create(comm0, dims; periodic=ntuple(_ -> true, D))
    coords = MPI.Cart_coords(cart, MPI.Comm_rank(cart))

    #comm  = MPI.Cart_create(MPI.COMM_WORLD, dims; periods=ntuple(_->true,D))
    #coords= MPI.Cart_coords(cart, MPI.Comm_rank(cart))
    nbr = ntuple(d -> ntuple(s -> MPI.Cart_shift(cart, d - 1, ifelse(s == 1, -1, 1))[2], 2), D)
    # local array (NC first)
    locS = ntuple(i -> gsize[i] ÷ dims[i] + 2nw, D)
    loc = (NC, locS...)
    A = JACC.zeros(T, loc...)

    # contiguous buffers for each face
    buf = Vector{typeof(A)}(undef, 4D)
    for d in 1:D
        shp = ntuple(i -> i == d ? nw : locS[i], D)   # halo slab shape
        buf[4d-3] = JACC.zeros(T, (NC, shp...)...)  # minus side
        buf[4d-2] = JACC.zeros(T, (NC, shp...)...)  # plus  side
        buf[4d-1] = JACC.zeros(T, (NC, shp...)...)  # minus side
        buf[4d] = JACC.zeros(T, (NC, shp...)...)  # plus  side
    end


    PN = ntuple(i -> gsize[i] ÷ dims[i], D)

    return LatticeVector{D,T,typeof(A)}(nw, phases, NC, gsize,
        cart, Tuple(coords), dims, nbr,
        A, buf, MPI.Comm_rank(cart), PN)
end

function LatticeVector(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)

    NC, NN... = size(A)
    elementtype = eltype(A)
    #@assert dim == length(NN)
    if dim == 1
        gsize = (NN,)
    else
        gsize = NN
    end

    ls = LatticeVector(NC, dim, gsize, PEs; elementtype, nw, phases, comm0)
    MPI.Bcast!(A, ls.cart)
    Acpu = Array(ls.A)

    idx = ntuple(i -> i == 1 ? Colon() : (ls.nw+1):(size(ls.A, i)-ls.nw), dim .+ 1)



    idx_global = ntuple(i -> i == 1 ? Colon() : get_globalrange(ls, i - 1), dim .+ 1)

    #println(idx)
    #println(idx_global)
    Acpu[idx...] = A[idx_global...]
    #println(Acpu)


    Agpu = JACC.array(Acpu)
    ls.A .= Agpu

    set_halo!(ls)
    #println(ls.A)

    return ls

    #coords_r = MPI.Cart_coords(ls.cart, ls.myrank)
    # 0-based coords
    #println(coords_r)

end

function get_globalrange(ls::LatticeVector{D,T,TA}, dim) where {D,T,TA}
    coords_r = MPI.Cart_coords(ls.cart, ls.myrank)
    istart = get_globalindex(ls, 1, dim, coords_r[dim])
    iend = get_globalindex(ls, ls.PN[dim], dim, coords_r[dim])
    return istart:iend
end

function get_globalindex(ls::LatticeVector{D,T,TA}, i, dim, myrank_dim) where {D,T,TA}
    ix = i + ls.PN[dim] * myrank_dim
    return ix
end



function set_halo!(ls::LatticeVector{D,T,TA}) where {D,T,TA}
    for id = 1:D
        exchange_dim!(ls, id)
    end
end
export set_halo!

# ---------------------------------------------------------------------------
# helpers that build proper “view tuples” without parsing errors
# ---------------------------------------------------------------------------
"""
    _face(A, nw, d, side)

Return a view of the halo–1 slab (width = `nw`) in spatial
dimension `d` on `side = :minus | :plus`.

* Array ordering is `(NC, X, Y, Z, …)` so the spatial
  dimension maps to index `d + 1`.
"""
function _face(A, nw, d, side::Symbol)
    # (1) decide the range WITHOUT the ternary-inside-range trick
    face_rng = if side === :minus
        (nw+1):(2*nw)
    else
        sz = size(A, d + 1)
        (sz-2*nw+1):(sz-nw)
    end

    # (2) build an indexing tuple, replacing only index d+1
    idx = ntuple(i -> i == d + 1 ? face_rng : Colon(), ndims(A))
    @views return A[idx...]            # a view, no copy
end

"""
    _ghost(A, nw, d, side)

Return a `@view` of the *internal* ghost layer (width `nw`) for
dimension `d` on the requested `side`.
"""
function _ghost(A, nw, d, side::Symbol)
    ghost_rng = if side === :minus
        1:nw
    else
        sz = size(A, d + 1)
        (sz-nw+1):sz
    end

    idx = ntuple(i -> i == d + 1 ? ghost_rng : Colon(), ndims(A))
    @views return A[idx...]
end
@inline _mul_phase!(buf, ϕ) =
    JACC.parallel_for(length(buf)) do i
        buf[i] *= ϕ
    end

##############################################################################
# exchange_dim!  –  no-derived-datatype version that never aliases buffers
#                   (works with MPI.jl v0.20.x)
#
#  * four contiguous buffers per spatial dimension:
#        bufSM (send minus), bufRM (recv minus),
#        bufSP (send plus) , bufRP (recv plus)
#  * send-buffers are filled with `_face`, optionally phase-multiplied,
#    then passed to MPI.Isend
#  * recv-buffers are passed to MPI.Irecv!  and finally copied into `_ghost`
##############################################################################
function exchange_dim!(ls::LatticeVector{D}, d::Int) where D
    # buffer indices
    iSM, iRM = 4d - 3, 4d - 2
    iSP, iRP = 4d - 1, 4d

    bufSM, bufRM = ls.buf[iSM], ls.buf[iRM]      # minus side: send / recv
    bufSP, bufRP = ls.buf[iSP], ls.buf[iRP]      # plus  side: send / recv

    rankM, rankP = ls.nbr[d]                     # neighbour ranks
    me = ls.myrank
    reqs = MPI.Request[]

    # --- self-neighbor on BOTH sides (happens iff dims[d] == 1) -------------
    if rankM == me && rankP == me
        # minus ghost <= plus face
        copy!(_ghost(ls.A, ls.nw, d, :minus),
            _face(ls.A, ls.nw, d, :plus))
        _mul_phase!(_ghost(ls.A, ls.nw, d, :minus), ls.phases[d])

        # plus  ghost <= minus face
        copy!(_ghost(ls.A, ls.nw, d, :plus),
            _face(ls.A, ls.nw, d, :minus))
        _mul_phase!(_ghost(ls.A, ls.nw, d, :plus), ls.phases[d])

        # no MPI in the self-case; just compute the interior and return
        compute_interior!(ls)
        return
    end


    baseT = MPI.Datatype(eltype(ls.A))           # elementary datatype
    #println("M ", rankM, "\t $me")
    MPI.Barrier(ls.cart)
    #println("P ", rankP, "\t $me")
    # ---------------- minus direction -------------------
    if rankM == me
        copy!(_ghost(ls.A, ls.nw, d, :minus),
            _face(ls.A, ls.nw, d, :minus))
        if ls.coords[d] == 0                     # wrap ⇒ phase
            _mul_phase!(_ghost(ls.A, ls.nw, d, :minus), ls.phases[d])
        end
    else
        copy!(bufSM, _face(ls.A, ls.nw, d, :minus))
        if ls.coords[d] == 0
            _mul_phase!(bufSM, ls.phases[d])
        end

        cnt = length(bufSM)

        push!(reqs, MPI.Isend(bufSM, rankM, d, ls.cart))#;
        #count=cnt, datatype=baseT))

        push!(reqs, MPI.Irecv!(bufRM, rankM, d + D, ls.cart))#;
        #count=cnt, datatype=baseT))
    end

    # ---------------- plus direction --------------------
    if rankP == me
        copy!(_ghost(ls.A, ls.nw, d, :plus),
            _face(ls.A, ls.nw, d, :plus))
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(_ghost(ls.A, ls.nw, d, :plus), ls.phases[d])
        end
    else
        copy!(bufSP, _face(ls.A, ls.nw, d, :plus))
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(bufSP, ls.phases[d])
        end

        cnt = length(bufSP)

        push!(reqs, MPI.Isend(bufSP, rankP, d + D, ls.cart))#;
        #count=cnt, datatype=baseT))
        push!(reqs, MPI.Irecv!(bufRP, rankP, d, ls.cart))
        #count=cnt, datatype=baseT))
    end

    # -------- overlap bulk computation -----------------
    compute_interior!(ls)
    isempty(reqs) || MPI.Waitall!(reqs)

    # -------- copy received data into ghosts -----------
    if rankM != me
        copy!(_ghost(ls.A, ls.nw, d, :minus), bufRM)
    end
    if rankP != me
        copy!(_ghost(ls.A, ls.nw, d, :plus), bufRP)
    end
end

# ---------------------------------------------------------------------------
# hooks (user overrides)
# ---------------------------------------------------------------------------
compute_interior!(ls::LatticeVector) = nothing
compute_boundary!(ls::LatticeVector) = nothing

export LatticeVector