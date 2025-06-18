function index_to_coords(i, NX, NY, NZ, NT)
    ix = mod1(i, NX)
    iy = mod1(div(i - 1, NX) + 1, NY)
    iz = mod1(div(i - 1, NX * NY) + 1, NZ)
    it = div(i - 1, NX * NY * NZ) + 1
    return ix, iy, iz, it
end

function coords_to_index(ix, iy, iz, it, NX, NY, NZ, NT)
    i = (it - 1) * NX * NY * NZ + (iz - 1) * NX * NY + (iy - 1) * NX + ix
    return i
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
