function substitute_U!(
    Uμ::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
    pwork,
) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}

    pwork_gpu = pwork |> CUDA.CuArray
    substitute_U!(
        Uμ,
        pwork_gpu)
end


function cudakernel_substitute_TAU!(Uμ, pwork, blockinfo, NumofBasis, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_substitute_TAU!(b, r, Uμ, pwork, blockinfo, NumofBasis, NX, NY, NZ, NT)
end



function substitute_U!(
    Uμ::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,:cuda},
    pwork::CUDA.CuArray,
) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}

    NT = Uμ.NT
    NZ = Uμ.NZ
    NY = Uμ.NY
    NX = Uμ.NX

    CUDA.@sync begin
        CUDA.@cuda threads = Uμ.blockinfo.blocksize blocks = Uμ.blockinfo.rsize cudakernel_substitute_TAU!(
            Uμ.a, pwork, Uμ.blockinfo, NumofBasis, NX, NY, NZ, NT)
    end

end



function cudakernel_mult_xTAyTA!(temp, x, y, NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mult_xTAyTA!(b, r, temp, x, y, NumofBasis)
    return
end



function Base.:*(
    x::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
    y::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mult_xTAyTA!(x.temp_volume,
            x.a, y.a, NumofBasis)
    end
    s = CUDA.reduce(+, x.temp_volume)
    return s
end

function cudakernel_exptU_TAwuww_NC!(uout, u,t,NC,NG,generators,temp1,temp2,temp3)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    kernel_exptU_TAwuww_NC!(b, r, uout,u, t,NC,NG,generators,temp1,temp2,temp3)
    return
end

function cudakernel_exptU_TAwuww_NC3!(w, u, ww, t)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_TAwuww_NC3!(b, r, w, u, ww, t)
    return
end

function cudakernel_exptU_TAwuww_NC2!(uout, u,  t)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_TAwuww_NC2!(b, r, uout, u,  t)
    return
end

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_accelerator{2,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta<:CUDA.CuArray,TUv} #uout = exp(t*u)     
 

    CUDA.@sync begin
        CUDA.@cuda threads = uout.blockinfo.blocksize blocks = uout.blockinfo.rsize cudakernel_exptU_TAwuww_NC2!(
            uout.U, u.a, t) #w,u,ww,t
    end


end

using InteractiveUtils

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {NC,N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta<:CUDA.CuArray,TUv} #uout = exp(t*u)     
 
    generators = Tuple(CUDA.CuArray.(u.generators.generator))
    NG = length(generators)

    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]
    
    #=
    uoutcpu = Array(uout.U)
    ucpu = Array(u.a)
    temp1cpu = Array(temp1.U)
    temp2cpu = Array(temp2.U)
    temp3cpu = Array(temp3.U)
    generatorscpu = Array.(generators)
    b =1
    r = 1

    kernel_exptU_TAwuww_NC!(b,r,
            uoutcpu, ucpu, t,NC,NG,generatorscpu,temp1cpu,temp2cpu,temp3cpu)
    display(uoutcpu[:,:,b,r])  
    error("dd")
    =#

    CUDA.@sync begin
        CUDA.@cuda threads = uout.blockinfo.blocksize blocks = uout.blockinfo.rsize cudakernel_exptU_TAwuww_NC!(
            uout.U, u.a, t,NC,NG,generators,temp1.U,temp2.U,temp3.U) #w,u,ww,t
    end


end

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_accelerator{3,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta<:CUDA.CuArray,TUv} #uout = exp(t*u)     
    ww = temps[1]
    w = temps[2]
    #clear_U!(uout)

    CUDA.@sync begin
        CUDA.@cuda threads = uout.blockinfo.blocksize blocks = uout.blockinfo.rsize cudakernel_exptU_TAwuww_NC3!(
            w.U, u.a, ww.U, t) #w,u,ww,t
    end
    mul!(uout, w', ww)

end

function cudakernel_Traceless_antihermitian_add_TAU_NC2!(
    c, vin, factor)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_Traceless_antihermitian_add_TAU_NC2!(b, r,
        c, vin, factor)
    return

end

function cudakernel_Traceless_antihermitian_add_TAU_NC3!(
    c, vin, factor)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_Traceless_antihermitian_add_TAU_NC3!(b, r,
        c, vin, factor)
    return

end

function cudakernel_Traceless_antihermitian_add_TAU_NC!(
    c, vin, factor,NC,NG,generators,temp,tempa)


    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_Traceless_antihermitian_add_TAU_NC!(b, r,
        c, vin, factor,NC,NG,generators,view(temp,:,:,b,r),view(tempa,:,b,r))
    return

end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_accelerator{2,NumofBasis,Ta,TUv},
    factor,
    vin::Gaugefields_4D_accelerator{2},
) where {NumofBasis,Ta<:CUDA.CuArray,TUv}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_Traceless_antihermitian_add_TAU_NC2!(
            c.a, vin.U, factor)
    end
end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_accelerator{3,NumofBasis,Ta,TUv},
    factor,
    vin::Gaugefields_4D_accelerator{3},
) where {NumofBasis,Ta<:CUDA.CuArray,TUv}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_Traceless_antihermitian_add_TAU_NC3!(
            c.a, vin.U, factor)
    end
end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
    factor,
    vin::Gaugefields_4D_accelerator{NC},
) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")

    generators = Tuple(CUDA.CuArray.(c.generators.generator))
    NG = length(generators)

    #temp1 = vin.Ushifted #reuse
    temp1 = deepcopy(vin.U)
    tempa = CUDA.CuArray{ComplexF64}(undef,NumofBasis,c.blockinfo.blocksize ,c.blockinfo.rsize)

    #=
    temp1cpu = Array(temp1)
    tempacpu = Array(tempa)
    vincpu = Array(vin.U)
    ccpu = Array(c.a)
    generatorscpu = Array.( generators)

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_Traceless_antihermitian_add_TAU_NC!(b, r,
                ccpu, vincpu, factor,NC,NG,generatorscpu,temp1cpu[:,:,b,r],tempacpu[:,b,r])
        end
    end
    display(ccpu[:,1,1])
    =#
    
    

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_Traceless_antihermitian_add_TAU_NC!(
            c.a, vin.U, factor,NC,NG,generators,temp1,tempa)
    end
    #display(Array(c.a)[:,1,1])
    #error("ta")
end

function cudakernel_clear_TAU!(c, NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_clear_TAU!(b, r, c, NumofBasis)
    return
end



function clear_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_clear_TAU!(c.a, NumofBasis)
    end
end

function cudakernel_add_TAU!(c, a, NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_TAU!(b, r, c, a, NumofBasis)
    return
end



function add_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}, a::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_TAU!(c.a, a.a, NumofBasis)
    end
end

function cudakernel_add_TAU!(c, t::Number, a, NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_TAU!(b, r, c, t, a, NumofBasis)
    return
end




function add_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis}, t::Number, a::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_TAU!(c.a, t, a.a, NumofBasis)
    end
end

function gauss_distribution!(
    p::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv};
    σ=1.0,
) where {NC,NumofBasis,Ta<:CUDA.CuArray,TUv}
    d = Normal(0.0, σ)
    pwork = rand(d, NumofBasis, p.blockinfo.blocksize, p.blockinfo.rsize)
    p.a .= CUDA.CuArray(pwork)
end

function substitute_U!(A::TA_Gaugefields_4D_serial{NC}, B::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta <:CUDA.CuArray,TUv}
    bcpu = Array(B.a)

    blockinfo = B.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)

            for ic = 1:NumofBasis
                A[ic, ix, iy, iz, it] = bcpu[ic, b, r]
            end
        end
    end
end



function substitute_U!(B::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},A::TA_Gaugefields_4D_serial{NC}) where {NC,NumofBasis,Ta <:CUDA.CuArray,TUv}
    bcpu = Array(B.a)

    blockinfo = B.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)

            for ic = 1:NumofBasis
                bcpu[ic, b, r] = A[ic, ix, iy, iz, it]
                
            end
        end
    end
    B.a .= CUDA.CuArray(bcpu)
end
