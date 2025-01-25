import CUDA

struct TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv} <: TA_Gaugefields_4D{NC}
    a::Ta
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NC::Int64
    NumofBasis::Int64
    generators::Union{Nothing,Generator}
    blockinfo::Blockindices
    temp_volume::TUv

    function TA_Gaugefields_4D_cuda(NC, NX, NY, NZ, NT,
        blocks)
        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        L = [NX,NY,NZ,NT]
        blockinfo = Blockindices(L,blocks)
        blocksize = blockinfo.blocksize
        rsize = blockinfo.rsize

        if CUDA.has_cuda()
            a = zeros(Float64, NumofBasis,blocksize,rsize ) |> CUDA.CuArray
            temp_volume = zeros(Float64, blocksize,rsize) |> CUDA.CuArray
        else
            @warn "no cuda devise is found. CPU will be used"
            a = zeros(Float64, NumofBasis,blocksize,rsize ) 
            temp_volume = zeros(Float64, blocksize,rsize) 
        end


        #a = zeros(Float64, NumofBasis,blocksize,rsize ) |> CUDA.CuArray
        Ta = typeof(a)

        #temp_volume = zeros(Float64, blocksize,rsize) |> CUDA.CuArray
        TUv = typeof(temp_volume)

        return new{NC,NumofBasis,Ta,TUv}(
            a,
            NX,
            NY,
            NZ,
            NT,
            NC,
            NumofBasis,
            generators,
            blockinfo,
            temp_volume 
        )
    end
end

function initialize_TA_Gaugefields(u::Gaugefields_4D_cuda) 
    return TA_Gaugefields_4D_cuda(u.NC, u.NX, u.NY, u.NZ, u.NT,
                u.blockinfo.blocks)
end

function cudakernel_substitute_TAU!(Uμ,pwork,blockinfo,NumofBasis,NX,NY,NZ,NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_substitute_TAU!(b,r,Uμ,pwork,blockinfo,NumofBasis,NX,NY,NZ,NT)
end



function substitute_U!(
    Uμ::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv},
    pwork::CUDA.CuArray,
) where {NC,NumofBasis,Ta <:CUDA.CuArray,TUv}

    NT = Uμ.NT
    NZ = Uμ.NZ 
    NY = Uμ.NY
    NX = Uμ.NX

    CUDA.@sync begin
        CUDA.@cuda threads=Uμ.blockinfo.blocksize blocks=Uμ.blockinfo.rsize cudakernel_substitute_TAU!(
                Uμ.a,pwork,Uμ.blockinfo,NumofBasis,NX,NY,NZ,NT)
    end

end

function substitute_U!(
    Uμ::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv},
    pwork,
) where {NC,NumofBasis,Ta,TUv}

    NT = Uμ.NT
    NZ = Uμ.NZ 
    NY = Uμ.NY
    NX = Uμ.NX

    for r=1:Uμ.blockinfo.rsize
        for b=1:Uμ.blockinfo.blocksize
            kernel_substitute_TAU!(b,r,
                Uμ.a,pwork,Uμ.blockinfo,NumofBasis,NX,NY,NZ,NT)
        end
    end


end

function substitute_U!(
    Uμ::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv},
    pwork,
) where {NC,NumofBasis,Ta <:CUDA.CuArray,TUv}

    pwork_gpu = pwork |> CUDA.CuArray
    substitute_U!(
        Uμ,
        pwork_gpu)
end

function Base.similar(u::TA_Gaugefields_4D_cuda{NC,NumofBasis}) where {NC,NumofBasis}
    return TA_Gaugefields_4D_cuda(NC, u.NX, u.NY, u.NZ, u.NT,u.blockinfo.blocks)
    #error("similar! is not implemented in type $(typeof(U)) ")
end


function cudakernel_mult_xTAyTA!(temp,x,y,NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mult_xTAyTA!(b,r,temp,x,y,NumofBasis)
    return
end



function Base.:*(
    x::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv},
    y::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv},
) where {NC,NumofBasis,Ta <: CUDA.CuArray ,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads=x.blockinfo.blocksize blocks=x.blockinfo.rsize cudakernel_mult_xTAyTA!(x.temp_volume,
                x.a,y.a,NumofBasis)
    end
    s = CUDA.reduce(+, x.temp_volume)
    return s
end

function Base.:*(
    x::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv},
    y::TA_Gaugefields_4D_cuda{NC,NumofBasis,Ta,TUv},
) where {NC,NumofBasis,Ta ,TUv}


    for r=1:x.blockinfo.rsize
        for b=1:x.blockinfo.blocksize
            kernel_mult_xTAyTA!(b,r,x.temp_volume,
                x.a,y.a,NumofBasis)
        end
    end

    s = reduce(+, x.temp_volume)
    return s
end



function exptU!(
    uout::T,
    t::N,
    v::TA_Gaugefields_4D_cuda{2,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NumofBasis} #uout = exp(t*u)
    error("exptU with 2 for type $(typeof(v)) is not implemented")
end

function exptU!(
    uout::T,
    t::N,
    v::TA_Gaugefields_4D_cuda{NC,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NumofBasis,NC} #uout = exp(t*u)
    error("exptU with general NC for type $(typeof(v)) is n ot implemented")
end

function cudakernel_exptU_TAwuww_NC3!(w,u,ww,t)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_TAwuww_NC3!(b,r,w,u,ww,t)
    return
end




function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_cuda{3,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NumofBasis,Ta <: CUDA.CuArray,TUv} #uout = exp(t*u)     
    ww = temps[1]
    w = temps[2]
    #clear_U!(uout)

    CUDA.@sync begin
        CUDA.@cuda threads=uout.blockinfo.blocksize blocks=uout.blockinfo.rsize cudakernel_exptU_TAwuww_NC3!(
            w.U,u.a,ww.U,t) #w,u,ww,t
    end
    mul!(uout, w', ww)

end

function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_cuda{3,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NumofBasis,Ta,TUv} #uout = exp(t*u)     
    ww = temps[1]
    w = temps[2]
    #clear_U!(uout)

    for r=1:uout.blockinfo.rsize
        for b=1:uout.blockinfo.blocksize
            kernel_exptU_TAwuww_NC3!(b,r,
                    w.U,u.a,ww.U,t) #w,u,ww,t
        end
    end

    mul!(uout, w', ww)

end

function cudakernel_Traceless_antihermitian_add_TAU_NC3!(
    c,vin,factor)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_Traceless_antihermitian_add_TAU_NC3!(b,r,
            c,vin,factor)
    return

end

function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_cuda{3,NumofBasis},
    factor,
    vin::Gaugefields_4D_cuda{3},
) where {NumofBasis}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")



    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize cudakernel_Traceless_antihermitian_add_TAU_NC3!(
            c.a,vin.U,factor)
    end
end

function cudakernel_clear_TAU!(c,NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for k1 = 1:NumofBasis
        c[k1,b,r] = 0
    end
    return
end

function clear_U!(c::TA_Gaugefields_4D_cuda{NC,NumofBasis}) where {NC,NumofBasis}
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_clear_TAU!(c.a,NumofBasis)
    end
end

function cudakernel_add_TAU!(c,a,NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for k1 = 1:NumofBasis
        c[k1,b,r] += a[k1,b,r]
    end
    return
end


function add_U!(c::TA_Gaugefields_4D_cuda{NC,NumofBasis}, a::TA_Gaugefields_4D_cuda{NC,NumofBasis}) where {NC,NumofBasis}
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_add_TAU!(c.a,a.a,NumofBasis)
    end
end

function cudakernel_add_TAU!(c,t::Number,a,NumofBasis)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for k1 = 1:NumofBasis
        c[k1,b,r] += t*a[k1,b,r]
    end
    return
end


function add_U!(c::TA_Gaugefields_4D_cuda{NC,NumofBasis}, t::Number,    a::TA_Gaugefields_4D_cuda{NC,NumofBasis}) where {NC,NumofBasis}
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_add_TAU!(c.a,t,a.a,NumofBasis)
    end
end