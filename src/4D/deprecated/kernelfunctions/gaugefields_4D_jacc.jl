function index_to_coords(i, NX, NY, NZ, NT)
    #return inv_evenodd_index(i, NX, NY, NZ, NT)

    ix = mod1(i, NX)
    iy = mod1(div(i - 1, NX) + 1, NY)
    iz = mod1(div(i - 1, NX * NY) + 1, NZ)
    it = div(i - 1, NX * NY * NZ) + 1
    return ix, iy, iz, it
end

function coords_to_index(ix, iy, iz, it, NX, NY, NZ, NT)
    #return evenodd_index(ix, iy, iz, it, NX, NY, NZ, NT)

    i = (it - 1) * NX * NY * NZ + (iz - 1) * NX * NY + (iy - 1) * NX + ix
    return i
end

"""
    evenodd_index(ix::Int, iy::Int, iz::Int, it::Int,
                  NX::Int, NY::Int, NZ::Int, NT::Int) -> Int

Compute a 1-based “evenodd” index by splitting the lexicographic range [1…V]
into two halves (even-parity first, odd-parity second), where
V = NX*NY*NZ*NT must be even.
"""
function evenodd_index(ix::Int, iy::Int, iz::Int, it::Int,
                       NX::Int, NY::Int, NZ::Int, NT::Int)
    # Total number of sites
    V = NX * NY * NZ * NT
    #@assert V % 2 == 0 "Total number of sites V must be even"

    # Compute 0-based lexicographic index
    site4 = (it - 1) * NZ * NY * NX +
            (iz - 1) * NY * NX +
            (iy - 1) * NX +
            (ix - 1)         # in [0, V-1]

    # Determine parity (0 = even, 1 = odd)
    parity = mod(ix + iy + iz + it, 2)

    # Position within its parity block
    idx0 = div(site4, 2) + parity * div(V, 2)

    # Convert to 1-based and return
    return idx0 + 1
end

#=
function evenodd_index(ix::Int, iy::Int, iz::Int, it::Int,
                       NX::Int, NY::Int, NZ::Int, NT::Int)
    V = NX * NY * NZ * NT
    site4 = (it - 1) * NZ * NY * NX +
            (iz - 1) * NY * NX +
            (iy - 1) * NX +
            (ix - 1)              # 0 … V-1
    parity = mod(ix + iy + iz + it, 2)

    idx0 = div(site4, 2) + parity * div(V, 2)
    return idx0 + 1      
end
=#

function decode(site4,NX,NY,NZ,NT)
    sites_per_time  = NZ * NY * NX
    sites_per_plane = NY * NX
    it   = fld(site4, sites_per_time) + 1
    rem1 = site4 % sites_per_time
    iz   = fld(rem1, sites_per_plane) + 1
    rem2 = rem1 % sites_per_plane
    iy   = fld(rem2, NX) + 1
    ix   = (rem2 % NX) + 1
    return ix, iy, iz, it
end

function coord_parity(ix, iy, iz, it )
    #ix, iy, iz, it = coords
    return mod(ix + iy + iz + it, 2)
end

"""
    inv_evenodd_index(idx::Int, NX::Int, NY::Int, NZ::Int, NT::Int) -> (ix, iy, iz, it)

Given a 1-based “evenodd” index `idx`, returns the original 4D coordinates
`(ix, iy, iz, it)` in the range `1:NX`, `1:NY`, `1:NZ`, `1:NT`.
"""
function inv_evenodd_index(idx::Int, NX::Int, NY::Int, NZ::Int, NT::Int)
    # Total number of sites
    V = NX * NY * NZ * NT
    halfV = div(V, 2)
    #@assert V % 2 == 0 "Total number of sites V must be even"
    #@assert 1 ≤ idx ≤ V "Index out of range"

    # Convert to 0-based index
    idx0 = idx - 1

    # Determine parity (0 = even, 1 = odd) based on which half of the range we're in
    parity = idx0 < halfV ? 0 : 1

    # Recover the “raw” half-index (i.e., div(site4, 2))
    idx0raw = idx0 - parity * halfV

    # Compute both candidate linear indices:
    site4_even = 2 * idx0raw         # candidate for even parity
    site4_odd  = 2 * idx0raw + 1     # candidate for odd parity

    # Decode both candidates
    cand1  = decode(site4_even,NX,NY,NZ,NT)
    cand2 = decode(site4_odd,NX,NY,NZ,NT)
    ix,iy,iz,it = cand1

    ix, iy, iz, it = ifelse(coord_parity(ix, iy, iz, it) == parity,cand1,cand2)# ? ix, iy, iz, it  : cand2

    # Select the candidate matching the stored parity
    # parity == 0 → even candidate, parity == 1 → odd candidate
    #site4 = parity == 0 ? site4_even : site4_odd

    # Reconstruct the original linear index (0-based)
    # site4 = 2 * div(site4,2) + parity
    #site4 = 2 * idx0raw + parity

    #=
    # Precompute block sizes
    sites_per_time = NZ * NY * NX   # sites in one it-slice
    sites_per_plane = NY * NX       # sites in one iz-plane

    # Compute time coordinate (it)
    it = fld(site4, sites_per_time) + 1
    rem1 = site4 % sites_per_time

    # Compute z-coordinate (iz)
    iz = fld(rem1, sites_per_plane) + 1
    rem2 = rem1 % sites_per_plane

    # Compute y-coordinate (iy)
    iy = fld(rem2, NX) + 1

    # Compute x-coordinate (ix)
    ix = (rem2 % NX) + 1
    =#

    i = evenodd_index(ix, iy, iz, it,NX,NY,NZ,NT)
    #@assert i == idx "$i $idx ix,iy,iz,it $(coords...)"

    return ix,iy,iz,it
end

function inverse_evenodd_index(idx::Int,
                               NX::Int, NY::Int, NZ::Int, NT::Int)
    V = NX * NY * NZ * NT
    halfV = div(V, 2)
    idx0 = idx - 1  

    parity = idx0 >= halfV ? 1 : 0

    site4 = 2 * (idx0 % halfV) + parity

    ix1 = site4 % NX
    tmp1 = div(site4, NX)
    iy1 = tmp1 % NY
    tmp2 = div(tmp1, NY)
    iz1 = tmp2 % NZ
    it1 = div(tmp2, NZ)

    ix, iy, iz, it = ix1 + 1, iy1 + 1, iz1 + 1, it1 + 1

    i = evenodd_index(ix,iy,iz,it,NX,NY,NZ,NT)
    @assert i == idx "$i $idx ix,iy,iz,it $ix $iy $iz $it"


    return ix, iy, iz, it
end

function applyfunction!(M::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Gaugefields_4D_nowing{NC}, f!::Function) where {NC,TU,TUv,TS}
    N = M.NX * M.NY * M.NZ * M.NT

    Mcpu = Array(M.U)
    for i = 1:N
        ix, iy, iz, it = index_to_coords(i, M.NX, M.NY, M.NZ, M.NT)
        f!(i, ix, iy, iz, it, Mcpu, A)
    end
    Mgpu = JACC.array(Mcpu)
    M.U .= Mgpu
    return
end

function applyfunction!(M::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    B::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS}, f!::Function) where {NC,TU,TUv,TS}
    N = M.NX * M.NY * M.NZ * M.NT

    JACC.parallel_for(N, f!, M.U, A.U, B.U, NC)
    #CUDA.@sync begin
    #    CUDA.@cuda threads = M.blockinfo.blocksize blocks = M.blockinfo.rsize f!(M.blockinfo, M.U, A.U, B.U,NC)
    #end
    return
end

function applyfunction!(M::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    A::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    B::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS}, α, β, f!::Function) where {NC,TU,TUv,TS}
    N = M.NX * M.NY * M.NZ * M.NT

    JACC.parallel_for(N, f!, M.U, A.U, B.U, α, β, NC)
    #CUDA.@sync begin
    #    CUDA.@cuda threads = M.blockinfo.blocksize blocks = M.blockinfo.rsize f!(M.blockinfo, M.U, A.U, B.U,NC)
    #end
    return
end

function substitute_U!(A::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS}, B::Gaugefields_4D_nowing{NC}) where {NC,TU,TUv,TS}
    applyfunction!(A, B, substitute_each!)
end

function substitute_each!(i::TN, ix::TN, iy::TN, iz::TN, it::TN, M::Array{T,3},
    A::Gaugefields_4D_nowing{NC}) where {NC,T,TN<:Integer}
    for jc in 1:NC
        for ic in 1:NC
            M[ic, jc, i] = A.U[ic, jc, ix, iy, iz, it]
        end
    end
end



function substitute_each!(i::TN, ix::TN,
    iy::TN, iz::TN, it::TN, M::Gaugefields_4D_nowing{NC},
    A::Array{T,3}) where {NC,T,TN<:Integer}
    for jc in 1:NC
        for ic in 1:NC
            M.U[ic, jc, ix, iy, iz, it] = A[ic, jc, i]
        end
    end
end

function identityfields!(i::Integer, M, NC)
    for jc in 1:NC
        for ic in 1:NC
            M[ic, jc, i] = ifelse(ic == jc, 1.0, 0.0)
        end
    end
end


function set_identity!(U::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS}) where {NC,TU,TUv,TS}
    applyfunction!(U, identityfields!)
end

function applyfunction!(M::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS},
    f!::Function) where {NC,TU,TUv,TS}
    N = M.NX * M.NY * M.NZ * M.NT
    JACC.parallel_for(N, f!, M.U, NC)
    return
end

function shiftedU_jacc!(i::Integer, NX, NY, NZ, NT, Ushifted, U, shift, NC)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    for jc in 1:NC
        for ic in 1:NC
            Ushifted[ic, jc, i] = U[ic, jc, i_shifted]
        end
    end
end

function jacckernel_partial_tr!(i::Integer, NX, NY, NZ, NT, U, NC, μ)
    NN = index_to_coords(i, NX, NY, NZ, NT)
    res = zero(eltype(U))
    if NN[μ] == 1
        @inbounds for k = 1:NC
            res += U[k, k, i]
        end
    end
    return res
end



function shifted_U!(M::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TshifedU}, shift) where {NC,TU,TUv,TshifedU}
    N = M.NX * M.NY * M.NZ * M.NT
    JACC.parallel_for(N, shiftedU_jacc!, M.NX, M.NY, M.NZ, M.NT, M.Ushifted, M.U, shift, NC)

    #for r = 1:U.blockinfo.rsize
    #    for b = 1:U.blockinfo.blocksize
    #        kernel_NC_shiftedU!(b, r, U.Ushifted, U.U,
    #            shift, U.blockinfo, NC)
    #    end
    #end
end

function multiply!(i::Integer, C::T1,
    A::T1, B::T1, NC) where {T1}
    for jc in 1:NC
        for ic in 1:NC
            C[ic, jc, i] = 0.0
            for kc in 1:NC
                C[ic, jc, i] += A[ic, kc, i] * B[kc, jc, i]
            end
        end
    end
end

function multiply!(i::Integer, C::T1,
    A::T1, B::T1, α, β, NC) where {T1}
    for jc in 1:NC
        for ic in 1:NC
            C[ic, jc, i] = α * C[ic, jc, i]
            for kc in 1:NC
                C[ic, jc, i] += β * A[ic, kc, i] * B[kc, jc, i]
            end
        end
    end
end


function partial_tr(a::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc}, μ) where {NC,TU,TUv}
    N = a.NX * a.NY * a.NZ * a.NT
    s = JACC.parallel_reduce(N, +, jacckernel_partial_tr!, a.NX, a.NY, a.NZ, a.NT, a.U, NC, μ; init=zero(eltype(a.U)))
    return s
end

function clear_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU,TUv}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_clear_U!, c.U, NC)
end

function add_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc}, a::T1) where {NC,T1<:Gaugefields_4D_accelerator,TU,TUv}
    N = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(N, jacckernel_add_U!, c.U, a.U, NC)
end

function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    α::N,
    a::T1,
) where {NC,T1<:Gaugefields_4D_accelerator{NC},N<:Number,TU,TUv}
    NN = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(NN, jacckernel_add_U_αa!, c.U, a.U, α, NC)

end



function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    α::N,
    a::T1,
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,N<:Number,TU,TUv}
    NN = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(NN, jacckernel_add_U_αshifta!,
        c.U, a.parent.U, α,
        a.shift, a.parent.blockinfo, NC)
end



function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc},
    α::N,
    a::Adjoint_Gaugefields{T1},
) where {NC,T1<:Gaugefields_4D_accelerator{NC},N<:Number,TU,TUv}
    NN = c.NX * c.NY * c.NZ * c.NT
    JACC.parallel_for(NN, jacckernel_add_U_αadag!, c.U, a.parent.U, α, NC)
end


function randomize_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS}) where {NC,TU,TUv,TS}
    N = c.NX * c.NY * c.NZ * c.NT
    ccpu = randomGaugefields_4D_nowing(
        NC,
        c.NX,
        c.NY,
        c.NZ,
        c.NT)
    #ccpu = Gaugefields_4D_nowing(NC, c.NX, c.NY, c.NZ, c.NT)
    #substitute_U!(ccpu, c)
    #randomize_U!(ccpu)
    substitute_U!(c, ccpu)
    #jacckernel_randomGaugefields!(i, U, NC)
    #JACC.parallel_for(N, jacckernel_randomGaugefields!, c.U, NC)
end



function normalize_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv,:jacc,TS}) where {NC,TU,TUv,TS}
    N = c.NX * c.NY * c.NZ * c.NT
    A = JACC.array(zeros(eltype(c.U), NC, NC, N))
    #jacckernel_randomGaugefields!(i, U, NC)
    JACC.parallel_for(N, jacckernel_normalize_U_NC!, c.U, A, NC)
end


function normalize_U!(c::Gaugefields_4D_accelerator{3,TU,TUv,:jacc,TS}) where {TU,TUv,TS}
    N = c.NX * c.NY * c.NZ * c.NT
    #jacckernel_randomGaugefields!(i, U, NC)
    JACC.parallel_for(N, jacckernel_normalize_U_NC3!, c.U)
end

function normalize_U!(c::Gaugefields_4D_accelerator{2,TU,TUv,:jacc,TS}) where {TU,TUv,TS}
    N = c.NX * c.NY * c.NZ * c.NT
    #jacckernel_randomGaugefields!(i, U, NC)
    JACC.parallel_for(N, jacckernel_normalize_U_NC2!, c.U)
end





#import CUDA
