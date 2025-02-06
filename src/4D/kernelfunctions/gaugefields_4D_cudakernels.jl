#import CUDA

function cudakernel_identityGaugefields!(U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_identityGaugefields!(b, r, U, NC)
    #@inbounds for ic = 1:NC
    #    U[ic, ic, b, r] = 1
    #end
end


function set_identity!(U::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU<:CUDA.CuArray,TUv}
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


function randomize_U!(U::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU<:CUDA.CuArray,TUv}
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
    kernel_mul_NC_abdag!(b, r, C, A, B, α, β,NC)
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



function LinearAlgebra.tr(a::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU<:CUDA.CuArray,TUv}
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
    a::Gaugefields_4D_accelerator{NC,TU,TUv},
    b::Gaugefields_4D_accelerator{NC,TU,TUv},
) where {NC,TU<:CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, b.U, NC)
    end

    s = CUDA.reduce(+, a.temp_volume)

    return s
end



function substitute_U!(A::Gaugefields_4D_nowing{NC}, B::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU<:CUDA.CuArray,TUv}
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



function add_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv}, a::T1) where {NC,T1<:Gaugefields_4D_accelerator,TU<:CUDA.CuArray,TUv}
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
    c::Gaugefields_4D_accelerator{NC,TU,TUv},
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
    c::Gaugefields_4D_accelerator{NC,TU,TUv},
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
    c::Gaugefields_4D_accelerator{NC,TU,TUv},
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



function clear_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU<:CUDA.CuArray,TUv}
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

function cudakernel_exptU_wvww_NC2!(w, uout,v, t)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_wvww_NC2!(b, r, uout,v, t)
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
        CUDA.@cuda threads = v.blockinfo.blocksize blocks = v.blockinfo.rsize cudakernel_exptU_wvww_NC2!(uout.U,v.U, t)
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


function partial_tr(a::Gaugefields_4D_accelerator{NC,TU,TUv}, μ) where {NC,TU<:CUDA.CuArray,TUv}
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

function cudakernel_substitute_U!(A,B,NC)
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
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_substitute_U!(a.U,B.parent.parent.Ushifted,NC)
    end
end

function cudakernel_unit_U!(U,NC)
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
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_unit_U!(U.U,NC)
    end
end
