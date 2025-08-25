##############################################################################
#  LatticeMatrix (no derived datatypes version)
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
struct LatticeMatrix{D,T,AT,NC1,NC2,nw} <: Lattice{D,T,AT}
    nw::Int                          # ghost width
    phases::SVector{D,T}                 # phases
    NC1::Int
    NC2::Int                        # internal DoF
    gsize::NTuple{D,Int}                # global size

    cart::MPI.Comm
    coords::NTuple{D,Int}
    dims::NTuple{D,Int}
    nbr::NTuple{D,NTuple{2,Int}}

    A::AT                           # main array (NC first)
    buf::Vector{AT}                   # 2D work buffers (minus/plus)
    myrank::Int
    PN::NTuple{D,Int}
    comm::MPI.Comm
end



# ---------------------------------------------------------------------------
# constructor + heavy init (still cheap to call)
# ---------------------------------------------------------------------------
function LatticeMatrix(NC1, NC2, dim, gsize, PEs; nw=1, elementtype=ComplexF64, phases=ones(dim), comm0=MPI.COMM_WORLD)

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
    #println(gsize)
    locS = ntuple(i -> gsize[i] ÷ dims[i] + 2nw, D)
    loc = (NC1, NC2, locS...)
    A = JACC.zeros(T, loc...)

    # contiguous buffers for each face
    buf = Vector{typeof(A)}(undef, 4D)
    for d in 1:D
        shp = ntuple(i -> i == d ? nw : locS[i], D)   # halo slab shape
        buf[4d-3] = JACC.zeros(T, (NC1, NC2, shp...)...)  # minus side
        buf[4d-2] = JACC.zeros(T, (NC1, NC2, shp...)...)  # plus  side
        buf[4d-1] = JACC.zeros(T, (NC1, NC2, shp...)...)  # minus side
        buf[4d] = JACC.zeros(T, (NC1, NC2, shp...)...)  # plus  side
    end


    PN = ntuple(i -> gsize[i] ÷ dims[i], D)
    #println("LatticeMatrix: $dims, $gsize, $PN, $nw")

    return LatticeMatrix{D,T,typeof(A),NC1,NC2,nw}(nw, phases, NC1, NC2, gsize,
        cart, Tuple(coords), dims, nbr,
        A, buf, MPI.Comm_rank(cart), PN, comm0)
end

function LatticeMatrix(A, dim, PEs; nw=1, phases=ones(dim), comm0=MPI.COMM_WORLD)

    NC1, NC2, NN... = size(A)
    #println(NN)
    elementtype = eltype(A)

    @assert dim == length(NN) "Dimension mismatch: expected $dim, got $(length(NN))"
    #if dim == 1
    #    gsize = (NN,)
    #else
    #    gsize = NN
    #end
    gsize = NN

    ls = LatticeMatrix(NC1, NC2, dim, gsize, PEs; elementtype, nw, phases, comm0)
    MPI.Bcast!(A, ls.cart)
    Acpu = Array(ls.A)

    idx = ntuple(i -> (i == 1 || i == 2) ? Colon() : (ls.nw+1):(size(ls.A, i)-ls.nw), dim .+ 2)



    idx_global = ntuple(i -> (i == 1 || i == 2) ? Colon() : get_globalrange(ls, i - 2), dim .+ 2)

    #println(idx)
    #=
    for i = 1:MPI.Comm_size(ls.cart)
        if ls.myrank == i
            println(get_globalrange(ls, 1))
        end
        MPI.Barrier(ls.cart)
    end
    =#



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

function Base.similar(ls::LatticeMatrix{D,T,AT,NC1,NC2}) where {D,T,AT,NC1,NC2}
    return LatticeMatrix(NC1, NC2, D, ls.gsize, ls.dims; nw=ls.nw, elementtype=T, phases=ls.phases, comm0=ls.comm)
end

function get_PEs(ls::LatticeMatrix{D,T,AT,NC1,NC2}) where {D,T,AT,NC1,NC2}
    return ls.dims
end
export get_PEs

function Base.display(ls::LatticeMatrix{4,T,AT,NC1,NC2}) where {T,AT,NC1,NC2}

    NN = size(ls.A)
    for rank = 0:MPI.Comm_size(ls.cart)-1
        if ls.myrank == rank
            println("LatticeMatrix (rank $rank):")
            indices = map(d -> get_globalrange(ls, d), 1:4)
            println("Global indices: ", indices)
            #println(ls.nw+1:NN[4]-ls.nw)
            for it in 1:ls.PN[4]
                for iz in 1:ls.PN[3]
                    for iy in 1:ls.PN[2]
                        for ix in 1:ls.PN[1]
                            println((indices[1][ix], indices[2][iy], indices[3][iz], indices[4][it]))
                            display(ls.A[:, :, ls.nw+ix, ls.nw+iy, ls.nw+iz, ls.nw+it])
                            #print("$(ls.A[:, :, ix, iy, iz, it]) ")
                        end
                    end
                end
            end
            #display(ls.A[:, :, ls.nw+1:end-ls.nw, ls.nw+1:end-ls.nw, ls.nw+1:end-ls.nw, ls.nw+1:end-ls.nw])
        end
        MPI.Barrier(ls.cart)
    end
end



function allsum(ls::LatticeMatrix{D,T,AT,NC1,NC2}) where {D,T,AT,NC1,NC2}
    NN = ls.PN
    indices = ntuple(i -> (i == 1 || i == 2) ? Colon() : (ls.nw+1):(ls.nw+NN[i-2]), D + 2)
    # sum all elements in the local array
    local_sum = sum(ls.A[indices...])
    #local_sum = sum(ls.A[:, :, ls.nw+1:ls.nw+NN[1], ls.nw+1:ls.nw+NN[2], ls.nw+1:ls.nw+NN[3], ls.nw+1:ls.nw+NN[4]])
    # reduce to all processes
    global_sum = MPI.Reduce(local_sum, MPI.SUM, 0, ls.cart)
    return global_sum
end

export allsum

function get_globalrange(ls::LatticeMatrix{D,T,TA}, dim) where {D,T,TA}
    coords_r = MPI.Cart_coords(ls.cart, ls.myrank)
    istart = get_globalindex(ls, 1, dim, coords_r[dim])
    #if dim == 1
    #    println(" $( ls.PN[dim])")
    # end
    iend = get_globalindex(ls, ls.PN[dim], dim, coords_r[dim])
    return istart:iend
end

function get_globalindex(ls::LatticeMatrix{D,T,TA}, i, dim, myrank_dim) where {D,T,TA}
    ix = i + ls.PN[dim] * myrank_dim
    return ix
end



function set_halo!(ls::LatticeMatrix{D,T,TA}) where {D,T,TA}
    for id = 1:D
        exchange_dim!(ls, id)
    end
end
export set_halo!

# ---------------------------------------------------------------------------
# helpers that build proper “view tuples” without parsing errors
# ---------------------------------------------------------------------------
"""
    _faceMatrix(A, nw, d, side)

Return a view of the halo–1 slab (width = `nw`) in spatial
dimension `d` on `side = :minus | :plus`.

* Array ordering is `(NC1, NC2, X, Y, Z, …)` so the spatial
  dimension maps to index `d + 2`.
"""
function _faceMatrix(A, nw, d, side::Symbol)
    # (1) decide the range WITHOUT the ternary-inside-range trick
    face_rng = if side === :minus
        (nw+1):(2*nw)
    else
        sz = size(A, d + 2)
        (sz-2*nw+1):(sz-nw)
    end

    # (2) build an indexing tuple, replacing only index d+2
    idx = ntuple(i -> i == d + 2 ? face_rng : Colon(), ndims(A))
    @views return A[idx...]            # a view, no copy
end

"""
    _ghostMatrix(A, nw, d, side)

Return a `@view` of the *internal* ghost layer (width `nw`) for
dimension `d` on the requested `side`.
"""
function _ghostMatrix(A, nw, d, side::Symbol)
    ghost_rng = if side === :minus
        1:nw
    else
        sz = size(A, d + 2)
        (sz-nw+1):sz
    end

    idx = ntuple(i -> i == d + 2 ? ghost_rng : Colon(), ndims(A))
    @views return A[idx...]
end


##############################################################################
# exchange_dim!  –  no-derived-datatype version that never aliases buffers
#                   (works with MPI.jl v0.20.x)
#
#  * four contiguous buffers per spatial dimension:
#        bufSM (send minus), bufRM (recv minus),
#        bufSP (send plus) , bufRP (recv plus)
#  * send-buffers are filled with `_faceMatrix`, optionally phase-multiplied,
#    then passed to MPI.Isend
#  * recv-buffers are passed to MPI.Irecv!  and finally copied into `_ghostMatrix`
##############################################################################
function exchange_dim!(ls::LatticeMatrix{D}, d::Int) where D
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
        copy!(_ghostMatrix(ls.A, ls.nw, d, :minus),
            _faceMatrix(ls.A, ls.nw, d, :plus))
        #    println( ls.phases[d])
        #println(ls.nw)
        #println(length(_ghostMatrix(ls.A, ls.nw, d, :minus)))
        _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :minus), ls.phases[d])

        # plus  ghost <= minus face
        copy!(_ghostMatrix(ls.A, ls.nw, d, :plus),
            _faceMatrix(ls.A, ls.nw, d, :minus))
        _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :plus), ls.phases[d])

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
        copy!(_ghostMatrix(ls.A, ls.nw, d, :minus),
            _faceMatrix(ls.A, ls.nw, d, :minus))
        if ls.coords[d] == 0                     # wrap ⇒ phase
            _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :minus), ls.phases[d])
        end
    else
        copy!(bufSM, _faceMatrix(ls.A, ls.nw, d, :minus))
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
        copy!(_ghostMatrix(ls.A, ls.nw, d, :plus),
            _faceMatrix(ls.A, ls.nw, d, :plus))
        if ls.coords[d] == ls.dims[d] - 1
            _mul_phase!(_ghostMatrix(ls.A, ls.nw, d, :plus), ls.phases[d])
        end
    else
        copy!(bufSP, _faceMatrix(ls.A, ls.nw, d, :plus))
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
        copy!(_ghostMatrix(ls.A, ls.nw, d, :minus), bufRM)
    end
    if rankP != me
        copy!(_ghostMatrix(ls.A, ls.nw, d, :plus), bufRP)
    end
end

# ---------------------------------------------------------------------------
# hooks (user overrides)
# ---------------------------------------------------------------------------
compute_interior!(ls::LatticeMatrix) = nothing
compute_boundary!(ls::LatticeMatrix) = nothing

export LatticeMatrix

# ---------------------------------------------------------------------------
# gather_matrix: collect local (halo-stripped) blocks to rank=0
# Reconstruct a global array of shape (NC1, NC2, gsize...)
# Communication is done on host memory for portability (CPU/GPU back-ends).
# ---------------------------------------------------------------------------
function gather_matrix(ls::LatticeMatrix{D,T,AT,NC1,NC2};
    root::Int=0) where {D,T,AT,NC1,NC2}
    comm = ls.cart
    me = ls.myrank
    nprocs = MPI.Comm_size(comm)

    # 1) Build view of the interior block (without halos)
    #    Spatial dims are shifted by +2 because array layout = (NC1, NC2, X, Y, Z, ...)
    interior_idx = ntuple(i -> (i <= 2 ? Colon() : (ls.nw+1):(ls.nw+ls.PN[i-2])), D + 2)
    @views local_view = ls.A[interior_idx...]   # a view on device/host
    local_block_cpu = Array(local_view)        # ensure host memory for MPI

    # Flatten to 1D send buffer for simple point-to-point
    sendbuf = reshape(local_block_cpu, :)
    count = length(sendbuf)

    # Helper: place a received block into the correct global offsets
    # coords are 0-based along each cart dimension
    function _place_block!(G, block, coords::NTuple{D,Int})
        # Compute global spatial ranges for this coords
        ranges = ntuple(d -> begin
                start = coords[d] * ls.PN[d] + 1
                stop = start + ls.PN[d] - 1
                start:stop
            end, D)
        # Build indexing tuple = (Colon, Colon, ranges...)
        idx = (Colon(), Colon(), ranges...)
        @views G[idx...] = block
        return nothing
    end

    if me == root
        # 2) Allocate the global array on root
        gshape = (ls.NC1, ls.NC2, ls.gsize...)
        G = Array{T}(undef, gshape)

        # 2a) Place root's own block
        _place_block!(G, reshape(sendbuf, size(local_block_cpu)), ls.coords)

        # 2b) Receive all other ranks and place
        #     For simplicity use a fixed tag per direction.
        tag = 900
        recvbuf = similar(sendbuf)  # reuse buffer
        for r in 0:nprocs-1
            r == root && continue
            MPI.Recv!(recvbuf, r, tag, comm)
            coords_r = Tuple(MPI.Cart_coords(comm, r))  # 0-based coords
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
        return G
    else
        # Non-root: send and return nothing
        tag = 900
        MPI.Send(sendbuf, root, tag, comm)
        return nothing
    end
end

export gather_matrix

# ---------------------------------------------------------------------------
# gather_and_bcast_matrix:
#   Collect halo-stripped blocks to root, reconstruct global matrix,
#   then broadcast it so all ranks receive the same array.
#   Returns Array{T}(NC1, NC2, gsize...)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# gather_and_bcast_matrix:
#   Collect local halo-free blocks to root, reconstruct global matrix on root,
#   then broadcast the global matrix so that ALL ranks return the same Array.
# ---------------------------------------------------------------------------
function gather_and_bcast_matrix(ls::LatticeMatrix{D,T,AT,NC1,NC2};
    root::Int=0) where {D,T,AT,NC1,NC2}
    comm = ls.cart
    me = ls.myrank
    nprocs = MPI.Comm_size(comm)

    # --- 1) local interior (no halo) on HOST ---
    interior_idx = ntuple(i -> (i <= 2 ? Colon() : (ls.nw+1):(ls.nw+ls.PN[i-2])), D + 2)
    @views local_view = ls.A[interior_idx...]
    local_block_cpu = Array(local_view)              # host buffer
    sendbuf = reshape(local_block_cpu, :)

    # helper to place a block at correct global offsets
    function _place_block!(G, block, coords::NTuple{D,Int})
        ranges = ntuple(d -> begin
                s = coords[d] * ls.PN[d] + 1
                e = s + ls.PN[d] - 1
                s:e
            end, D)
        idx = (Colon(), Colon(), ranges...)
        @views G[idx...] = block
        return nothing
    end

    G = nothing
    if me == root
        # --- 2) reconstruct on root ---
        gshape = (ls.NC1, ls.NC2, ls.gsize...)
        G = Array{T}(undef, gshape)

        # root’s own block
        _place_block!(G, reshape(sendbuf, size(local_block_cpu)), ls.coords)

        # receive others
        recvbuf = similar(sendbuf)
        for r in 0:nprocs-1
            r == root && continue
            MPI.Recv!(recvbuf, r, 900, comm)
            coords_r = Tuple(MPI.Cart_coords(comm, r))
            blk = reshape(recvbuf, size(local_block_cpu))
            _place_block!(G, blk, coords_r)
        end
    else
        # non-root: send local block
        MPI.Send(sendbuf, root, 900, comm)
    end

    # --- 3) broadcast ONLY the data (shape is deterministic) ---
    gshape = (ls.NC1, ls.NC2, ls.gsize...)   # same on all ranks
    if me != root
        G = Array{T}(undef, gshape)          # allocate receive buffer
    end
    MPI.Bcast!(G, root, comm)                # broadcast the global array

    return G
end
export gather_and_bcast_matrix