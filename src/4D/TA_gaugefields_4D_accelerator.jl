import CUDA

struct TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv,accdevise} <: TA_Gaugefields_4D{NC}
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
    accelerator::String

    function TA_Gaugefields_4D_accelerator(NC, NX, NY, NZ, NT,
        blocks; accelerator="none")
        NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        L = [NX, NY, NZ, NT]
        blockinfo = Blockindices(L, blocks)
        blocksize = blockinfo.blocksize
        rsize = blockinfo.rsize

        a0 = zeros(Float64, NumofBasis, blocksize, rsize)
        temp_volume0 = zeros(Float64, blocksize, rsize)

        if accelerator == "cuda"
            if CUDA.has_cuda()
                a = CUDA.CuArray(a0)
                temp_volume = CUDA.CuArray(temp_volume0)
                accdevise = :cuda
            else
                @warn "accelerator=\"cuda\" is set but there is no CUDA devise. CPU will be used"
                a = a0
                temp_volume = temp_volume0
                accdevise = :none
            end
        elseif accelerator == "threads"
            a = a0
            temp_volume = temp_volume0
            accdevise = :threads
        else
            a = a0
            temp_volume = temp_volume0
            accdevise = :none
        end


        #a = zeros(Float64, NumofBasis,blocksize,rsize ) |> CUDA.CuArray
        Ta = typeof(a)

        #temp_volume = zeros(Float64, blocksize,rsize) |> CUDA.CuArray
        TUv = typeof(temp_volume)

        return new{NC,NumofBasis,Ta,TUv,accdevise}(
            a,
            NX,
            NY,
            NZ,
            NT,
            NC,
            NumofBasis,
            generators,
            blockinfo,
            temp_volume,
            accelerator
        )
    end
end


function initialize_TA_Gaugefields(u::Gaugefields_4D_accelerator)
    return TA_Gaugefields_4D_accelerator(u.NC, u.NX, u.NY, u.NZ, u.NT,
        u.blockinfo.blocks, accelerator=u.accelerator)
end



function substitute_U!(
    Uμ::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
    pwork,
) where {NC,NumofBasis,Ta,TUv}

    NT = Uμ.NT
    NZ = Uμ.NZ
    NY = Uμ.NY
    NX = Uμ.NX

    for r = 1:Uμ.blockinfo.rsize
        for b = 1:Uμ.blockinfo.blocksize
            kernel_substitute_TAU!(b, r,
                Uμ.a, pwork, Uμ.blockinfo, NumofBasis, NX, NY, NZ, NT)
        end
    end


end


function Base.similar(u::TA_Gaugefields_4D_accelerator{NC,NumofBasis}) where {NC,NumofBasis}
    return TA_Gaugefields_4D_accelerator(NC, u.NX, u.NY, u.NZ, u.NT, u.blockinfo.blocks; accelerator=u.accelerator)
    #error("similar! is not implemented in type $(typeof(U)) ")
end



function Base.:*(
    x::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
    y::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
) where {NC,NumofBasis,Ta,TUv}


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mult_xTAyTA!(b, r, x.temp_volume,
                x.a, y.a, NumofBasis)
        end
    end

    s = reduce(+, x.temp_volume)
    return s
end



function exptU!(
    uout::T,
    t::N,
    v::TA_Gaugefields_4D_accelerator{2,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis} #uout = exp(t*u)
    error("exptU with 2 for type $(typeof(v)) is not implemented")
end

function exptU!(
    uout::T,
    t::N,
    v::TA_Gaugefields_4D_accelerator{NC,NumofBasis},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,NC} #uout = exp(t*u)
    error("exptU with general NC for type $(typeof(v)) is n ot implemented")
end



function exptU!(
    uout::T,
    t::N,
    u::TA_Gaugefields_4D_accelerator{3,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta,TUv} #uout = exp(t*u)     
    ww = temps[1]
    w = temps[2]
    #clear_U!(uout)

    for r = 1:uout.blockinfo.rsize
        for b = 1:uout.blockinfo.blocksize
            kernel_exptU_TAwuww_NC3!(b, r,
                w.U, u.a, ww.U, t) #w,u,ww,t
        end
    end

    mul!(uout, w', ww)

end



function Traceless_antihermitian_add!(
    c::TA_Gaugefields_4D_accelerator{3,NumofBasis,Ta,TUv},
    factor,
    vin::Gaugefields_4D_accelerator{3},
) where {NumofBasis,Ta,TUv}
    #error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_Traceless_antihermitian_add_TAU_NC3!(b, r,
                c.a, vin.U, factor)
        end
    end
end


function clear_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_clear_TAU!(b, r, c.a, NumofBasis)
        end
    end
end


function add_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}, a::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_TAU!(b, r, c.a, a.a, NumofBasis)
        end
    end
end



function add_U!(c::TA_Gaugefields_4D_accelerator{NC,NumofBasis}, t::Number, a::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_TAU!(b, r, c.a, t, a.a, NumofBasis)
        end
    end

end