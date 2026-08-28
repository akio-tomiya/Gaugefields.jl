# CUDA is provided by the surrounding `@require CUDA` callback.  This legacy
# backend must not re-initialize JACC, because CUDA may be loaded solely as the
# active JACC backend for the LatticeMatrices implementation.


function cudakernel_identityGaugefields!(U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_identityGaugefields!(b, r, U, NC)
    #@inbounds for ic = 1:NC
    #    U[ic, ic, b, r] = 1
    #end
end


function set_identity!(U::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda}) where {NC,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_identityGaugefields!(U.U, NC)
    end
end

function substitute_U!(A::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TS}, B::Gaugefields_4D_nowing{NC}) where {NC,TU<:CUDA.CuArray,TUv,TS}
    acpu = Array(A.U)

    blockinfo = A.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
            #println((ix,iy,iz,it))
            for ic = 1:NC
                for jc = 1:NC
                    acpu[jc, ic, b, r] = B[jc, ic, ix, iy, iz, it]
                end
            end
        end
    end
    agpu = CUDA.CuArray(acpu)
    A.U .= agpu

end


function cudakernel_randomGaugefields!(U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for ic = 1:NC
        for jc = 1:NC
            U[jc, ic, b, r] = CUDA.rand() - 0.5 + im * (CUDA.rand() - 0.5)
        end
    end
    #kernel_randomGaugefields!(b, r, U.U, NC)
end


function randomize_U!(U::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda}) where {NC,TU<:CUDA.CuArray,TUv}
    Ucpu = Array(U.U)
    for r = 1:U.blockinfo.rsize
        for b = 1:U.blockinfo.blocksize
            kernel_randomGaugefields!(b, r, Ucpu, NC)
        end
    end
    U.U .= CUDA.CuArray(Ucpu)
    #CUDA.@sync begin
    #    CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_randomGaugefields!(U.U, NC)
    #end
end


function cudakernel_normalize_U_NC2!(u)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_normalize_U_NC2!(b, r, u)
    return
end

function cudakernel_normalize_U_NC3!(u)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_normalize_U_NC3!(b, r, u)
    return
end

function normalize_U!(U::Gaugefields_4D_accelerator{2,TU,TUv}) where {TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_normalize_U_NC2!(U.U)
    end
end

function normalize_U!(U::Gaugefields_4D_accelerator{3,TU,TUv}) where {TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_normalize_U_NC3!(U.U)
    end
end

function cudakernel_mul_NC!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC!(b, r, C, A, B, NC)
end


function cudakernel_mul_NC3!(C, A, B, α, β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3!(b, r, C, A, B, α, β)
    return
end

function cudakernel_mul_NC3!(C, A, B)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3!(b, r, C, A, B)
    return
end

function cudakernel_mul_NC_abdag!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abdag!(b, r, C, A, B, NC)
end

function cudakernel_mul_NC3_abdag!(C, A, B)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3_abdag!(b, r, C, A, B)
end

function cudakernel_mul_NC3_abdag!(C, A, B, α, β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3_abdag!(b, r, C, A, B, α, β)
end

function cudakernel_mul_NC_abdag!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abdag!(b, r, C, A, B, α, β, NC)
end




function cudakernel_mul_NC3!(C, A, B, α, β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3!(b, r, C, A, B, α, β)
    return
end



function cudakernel_mul_NC_abdag!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abdag!(b, r, C, A, B, α, β, NC)
end


function cudakernel_mul_NC_adagbdag!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbdag!(b, r, C, A, B, α, β, NC)
end


function cudakernel_mul_NC_adagbdag!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbdag!(b, r, C, A, B, NC)
end



function cudakernel_mul_NC3_adagbdag!(C, A, B, α, β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3_adagbdag!(b, r, C, A, B, α, β)
end


function cudakernel_mul_NC3_adagbdag!(C, A, B)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3_adagbdag!(b, r, C, A, B)
end




function cudakernel_mul_NC3_adagb!(C, A, B, α, β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3_adagb!(b, r, C, A, B, α, β)
end

function cudakernel_mul_NC3_adagb!(C, A, B)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3_adagb!(b, r, C, A, B)
end

function cudakernel_mul_NC_adagb!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagb!(b, r, C, A, B, α, β, NC)
end






function cudakernel_mul_NC_abshift!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abshift!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end




function cudakernel_mul_NC_ashiftb!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftb!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end




function cudakernel_mul_NC_ashiftbshift!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftbshift!(b, r, C, A, B, α, β, ashift, bshift, blockinfo, NC)
end




function cudakernel_mul_NC_ashiftbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftbshiftdag!(b, r, C, A, B, α, β, ashift, bshift, blockinfo, NC)
end





function cudakernel_mul_NC_adagbshift!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbshift!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end




function cudakernel_mul_NC_adagbshiftdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbshiftdag!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end


function cudakernel_mul_NC_ashiftbdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftbdag!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end





function cudakernel_mul_NC_ashiftdagbdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagbdag!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end




function cudakernel_mul_NC_abshiftdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abshiftdag!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end




function cudakernel_mul_NC_ashiftdagb!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagb!(b, r, C, A, B, α, β, shift, blockinfo, NC)
end




function cudakernel_mul_NC_ashiftdagbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagbshiftdag!(b, r, C, A, B, α, β, ashift, bshift, blockinfo, NC)
end



function cudakernel_mul_NC_ashiftdagbshift!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagbshift!(b, r, C, A, B, α, β, ashift, bshift, blockinfo, NC)
end


function cudakernel_tr!(temp_volume, U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_tr!(b, r, temp_volume, U, NC)
    return
end



function LinearAlgebra.tr(a::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda}) where {NC,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, NC)
    end

    s = CUDA.reduce(+, a.temp_volume)
    return s

end

function cudakernel_tr!(temp_volume, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_tr!(b, r, temp_volume, A, B, NC)
    return
end


function LinearAlgebra.tr(
    a::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda},
    b::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda},
) where {NC,TU<:CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, b.U, NC)
    end

    s = CUDA.reduce(+, a.temp_volume)

    return s
end



function substitute_U!(A::Gaugefields_4D_nowing{NC}, B::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda}) where {NC,TU<:CUDA.CuArray,TUv}
    bcpu = Array(B.U)

    blockinfo = B.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)

            for ic = 1:NC
                for jc = 1:NC
                    A[jc, ic, ix, iy, iz, it] = bcpu[jc, ic, b, r]
                end
            end
        end
    end
end

function cudakernel_add_U!(c, a, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_U!(b, r, c, a, NC)
    return
end



function add_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda}, a::T1) where {NC,T1<:Gaugefields_4D_accelerator,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U!(c.U, a.U, NC)
    end
end

function cudakernel_add_U_αa!(c, a, α, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_U_αa!(b, r, c, a, α, NC)
    return
end



function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda},
    α::N,
    a::T1,
) where {NC,T1<:Gaugefields_4D_accelerator{NC},N<:Number,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αa!(c.U, a.U, α, NC)
    end
end

function cudakernel_add_U_αshifta!(c, a, α, shift, blockinfo, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_U_αshifta!(b, r, c, a, α, shift, blockinfo, NC)
    return
end




function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda},
    α::N,
    a::T1,
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,N<:Number,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αshifta!(
            c.U, a.parent.U, α,
            a.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_add_U_αadag!(c, a, α, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_U_αadag!(b, r, c, a, α, NC)
    return
end




function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda},
    α::N,
    a::Adjoint_Gaugefields{T1},
) where {NC,T1<:Gaugefields_4D_accelerator{NC},N<:Number,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αadag!(c.U, a.parent.U, α, NC)
    end
end

function cudakernel_clear_U!(c, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_clear_U!(b, r, c, NC)
    return
end



function clear_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda}) where {NC,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_clear_U!(c.U, NC)
    end
end


function cudakernel_exptU_wvww!(w, v, ww, t, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_wvww!(b, r, w, v, ww, t, NC)
    return

end

function cudakernel_exptU_wvww_NC3!(w, v, ww, t)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_wvww_NC3!(b, r, w, v, ww, t)
    return

end

function cudakernel_exptU_wvww_NC2!(w, uout, v, t)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_wvww_NC2!(b, r, uout, v, t)
    return

end

function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_accelerator{3,TU,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,TU<:CUDA.CuArray,TUv} #uout = exp(t*u)

    ww = temps[1]
    w = temps[2]

    CUDA.@sync begin
        CUDA.@cuda threads = v.blockinfo.blocksize blocks = v.blockinfo.rsize cudakernel_exptU_wvww_NC3!(w.U, v.U, ww.U, t)
    end

    mul!(uout, w', ww)
end

function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_accelerator{2,TU,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,TU<:CUDA.CuArray,TUv} #uout = exp(t*u)

    CUDA.@sync begin
        CUDA.@cuda threads = v.blockinfo.blocksize blocks = v.blockinfo.rsize cudakernel_exptU_wvww_NC2!(uout.U, v.U, t)
    end

end

function cudakernel_Traceless_antihermitian_NC3!(vout, vin)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_Traceless_antihermitian_NC3!(b, r, vout, vin)
    return
end

"""
-----------------------------------------------------c
     !!!!!   vin and vout should be different vectors

     Projectin of the etraceless antiermite part 
     vout = x/2 - Tr(x)/6
     wher   x = vin - Conjg(vin)      
-----------------------------------------------------c
    """

#Q = -(1/2)*(Ω' - Ω) + (1/(2NC))*tr(Ω' - Ω)*I0_2
#Omega' - Omega = -2i imag(Omega)
function Traceless_antihermitian!(
    vout::Gaugefields_4D_accelerator{3,TU,TUv},
    vin::Gaugefields_4D_accelerator{3},
) where {TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = vout.blockinfo.blocksize blocks = vout.blockinfo.rsize cudakernel_Traceless_antihermitian_NC3!(vout.U, vin.U)
    end

end



function cudakernel_partial_tr!(temp_volume, U, NC, blockinfo, μ)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_partial_tr!(b, r, temp_volume, U, NC, blockinfo, μ)
    return
end


function partial_tr(a::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda}, μ) where {NC,TU<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_partial_tr!(a.temp_volume, a.U, NC, a.blockinfo, μ)
    end
    s = CUDA.reduce(+, a.temp_volume)

    return s
end

function cudakernel_NC_shiftedU!(Ushifted, U, shift, blockinfo, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_NC_shiftedU!(b, r, Ushifted, U, shift, blockinfo, NC)
    return
end



function shifted_U!(U::Gaugefields_4D_accelerator{NC,TU,TUv,accdevise,TshifedU}, shift) where {NC,TU<:CUDA.CuArray,TUv,accdevise,TshifedU<:CUDA.CuArray}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_NC_shiftedU!(U.Ushifted, U.U,
            shift, U.blockinfo, NC)
    end
end



function cudakernel_mul_NC!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC!(b, r, C, A, B, α, β, NC)
    return
end

function cudakernel_substitute_U!(A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    for ic = 1:NC
        for jc = 1:NC
            A[jc, ic, b, r] = conj(B[ic, jc, b, r])
        end
    end
end

function substitute_U!(a::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TshifedU}, B::Adjoint_Gaugefields{T1}) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,TU,TUv,TshifedU}
    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_substitute_U!(a.U, B.parent.parent.Ushifted, NC)
    end
end

function cudakernel_unit_U!(U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    for ic = 1:NC
        for jc = 1:NC
            U[jc, ic, b, r] = ifelse(ic == jc, 1, 0)
        end
    end
end


function unit_U!(U::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TshifedU}) where {NC,TU,TUv,TshifedU}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_unit_U!(U.U, NC)
    end
end


@inline function calc_coefficients_Q(Q)
    @assert size(Q) == (3, 3)
    c0 =
        Q[1, 1] * Q[2, 2] * Q[3, 3] +
        Q[1, 2] * Q[2, 3] * Q[3, 1] +
        Q[1, 3] * Q[2, 1] * Q[3, 2] - Q[1, 3] * Q[2, 2] * Q[3, 1] -
        Q[1, 2] * Q[2, 1] * Q[3, 3] - Q[1, 1] * Q[2, 3] * Q[3, 2]

    c1 = 0.0
    for i = 1:3
        for j = 1:3
            c1 += Q[i, j] * Q[j, i]
        end
    end
    c1 /= 2
    c0max = 2 * (c1 / 3)^(3 / 2)
    θ = acos(c0 / c0max)
    u = sqrt(c1 / 3) * cos(θ / 3)
    w = sqrt(c1) * sin(θ / 3)
    ξ0 = sin(w) / w
    ξ1 = cos(w) / w^2 - sin(w) / w^3

    emiu = exp(-im * u)
    e2iu = exp(2 * im * u)

    h0 = (u^2 - w^2) * e2iu + emiu * (8u^2 * cos(w) + 2 * im * u * (3u^2 + w^2) * ξ0)
    h1 = 2u * e2iu - emiu * (2u * cos(w) - im * (3u^2 - w^2) * ξ0)
    h2 = e2iu - emiu * (cos(w) + 3 * im * u * ξ0)

    denom = 9u^2 - w^2

    f0 = h0 / denom
    f1 = h1 / denom
    f2 = h2 / denom

    r10 =
        2 * (u + im * (u^2 - w^2)) * e2iu +
        2 *
        emiu *
        (4u * (2 - im * u) * cos(w) + im * (9u^2 + w^2 - im * u * (3u^2 + w^2)) * ξ0)
    r11 =
        2 * (1 + 2 * im * u) * e2iu +
        emiu * (-2 * (1 - im * u) * cos(w) + im * (6u + im * (w^2 - 3u^2)) * ξ0)
    r12 = 2 * im * e2iu + im * emiu * (cos(w) - 3 * (1 - im * u) * ξ0)
    r20 = -2 * e2iu + 2 * im * u * emiu * (cos(w) + (1 + 4 * im * u) * ξ0 + 3u^2 * ξ1)
    r21 = -im * emiu * (cos(w) + (1 + 2 * im * u) * ξ0 - 3 * u^2 * ξ1)
    r22 = emiu * (ξ0 - 3 * im * u * ξ1)
    b10 = (2 * u * r10 + (3u^2 - w^2) * r20 - 2 * (15u^2 + w^2) * f0) / (2 * denom^2)

    b11 = (2 * u * r11 + (3u^2 - w^2) * r21 - 2 * (15u^2 + w^2) * f1) / (2 * denom^2)
    b12 = (2 * u * r12 + (3u^2 - w^2) * r22 - 2 * (15u^2 + w^2) * f2) / (2 * denom^2)
    b20 = (r10 - 3 * u * r20 - 24 * u * f0) / (2 * denom^2)
    b21 = (r11 - 3 * u * r21 - 24 * u * f1) / (2 * denom^2)
    b22 = (r12 - 3 * u * r22 - 24 * u * f2) / (2 * denom^2)

    return f0, f1, f2, b10, b11, b12, b20, b21, b22
end

function cudakernel_construct_Λmatrix_forSTOUT!(Λ, δ, Q, U, NC, global_buffer)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    
    @inbounds begin
      
        temp1 = view(global_buffer, :, :, 1, b,r)
        temp2 = view(global_buffer, :, :, 2, b,r)
        temp3 = view(global_buffer, :, :, 3, b,r)
        Qn = view(global_buffer, :, :, 4, b,r)
        Mn = view(global_buffer, :, :, 5, b,r)
        Unδn = view(global_buffer, :, :, 6, b,r)
    

    
        for ic in 1:3
            for jc in 1:3
                Qn[ic,jc] = Q[ic,jc,b,r]
            end
        end

        #calc_Mmatrix! --> elementwise operation
        trQ2 = 0.0
        for i = 1:3
            for j = 1:3
                trQ2 += Qn[i, j] * Qn[j, i]
            end
        end
    
    
        
        if abs(trQ2) > 1e-18
            Qn ./= im
            #println("Qn b ",Qn)
            f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qn)            
            
            for ic in 1:3
                for jc in 1:3
                    Unδn[ic,jc] = 0.0f0 + 0im
                    for k = 1:3
                        Unδn[ic,jc] += U[ic,k, b, r] * δ[k,jc, b, r]
                    end
                end
            end
            
            B1 = temp1
            B1 .= 0
            B2 = temp3
            B2 .= 0
            for i = 1:3
                B1[i, i] = b10
                B2[i, i] = b20
            end
            for j = 1:3
                for i = 1:3
                    B1[i, j] += b11 * Qn[i, j]
                    B2[i, j] += b21 * Qn[i, j]
                    for k = 1:3
                        B1[i, j] += b12 * Qn[i, k] * Qn[k, j]
                        B2[i, j] += b22 * Qn[i, k] * Qn[k, j]
                    end
                end
            end
        

            trB1 = 0.0
            trB2 = 0.0
            for i = 1:3
                for j = 1:3
                    trB1 += Unδn[i, j] * B1[j, i]
                    trB2 += Unδn[i, j] * B2[j, i]
                end
            end

            for j = 1:3
                for i = 1:3
                    Mn[i, j] = trB1 * Qn[i, j] + f1 * Unδn[i, j]
                    for k = 1:3
                        Mn[i, j] +=
                            trB2 * Qn[i, k] * Qn[k, j] +
                            f2 * (Qn[i, k] * Unδn[k, j] + Unδn[i, k] * Qn[k, j])
                    end
                end
            end
            
            for i = 1:3
                for j = 1:3
                    Mn[i, j] /= im
                end
            end
        else
           
            #Mn .= 0
            #mul!(Mn, Un, δn) # --> f1 = 1, to have a well-defined point when Q == 0 for θ =0. 
            for ic in 1:3
                for jc in 1:3
                    Mn[ic,jc] = 0.0f0 + 0im
                    for k = 1:3
                        Mn[ic,jc] += U[ic,k, b, r] * δ[k,jc, b, r]
                    end
                end
            end
            
        end


        #calc_Λmatrix!(Λn, Mn, NC) --> elementwise operation
        temp2 .= 0
        for i = 1:3
            for j = 1:3
                temp2[i, j] = (1 / 2) * (Mn[i,j] - conj(Mn[j,i]))
            end
        end
            
        #trMn = (1 / (6)) * tr(Mn - Mn')
        trMn = 0.0
        for i = 1:3
            trMn += ( Mn[i, i] - conj(Mn[i, i]) ) / 6
        end

        for i = 1:3
            temp2[i, i] += -trMn
        end
        
        for jc = 1:NC
            for ic = 1:NC
                Λ[ic,jc,b,r] = temp2[ic, jc]
            end
        end
    
    end
    
    return
end


"""
M = (U*δ_prev) star (dexp(Q)/dQ)
Λ = TA(M)
"""
function construct_Λmatrix_forSTOUT!(
    Λ::Gaugefields_4D_accelerator{NC,TU,TUv},
    δ_current::Gaugefields_4D_accelerator{NC,TU,TUv},
    Q::Gaugefields_4D_accelerator{NC,TU,TUv},
    u::Gaugefields_4D_accelerator{NC,TU,TUv},
) where { NC,TU<:CUDA.CuArray,TUv}

    nthreads = Λ.blockinfo.blocksize

    # Allocate a global buffer: shape (3, 3, total_matrices)
    dtype = eltype(Λ.U)
    global_buffer = CUDA.zeros(dtype, 3, 3, 6, nthreads, Λ.blockinfo.rsize)
    
    CUDA.@sync begin
        CUDA.@cuda threads = nthreads blocks = Λ.blockinfo.rsize cudakernel_construct_Λmatrix_forSTOUT!(
            Λ.U, δ_current.U, Q.U, u.U, NC, global_buffer)
    end
end
