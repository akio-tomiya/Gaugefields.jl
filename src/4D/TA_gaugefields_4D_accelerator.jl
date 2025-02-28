#import CUDA

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
    singleprecision::Bool

    function TA_Gaugefields_4D_accelerator(NC, NX, NY, NZ, NT,
        blocks; accelerator="none", singleprecision=false)
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

        dtype = ifelse(singleprecision, Float32, Float64)
        a0 = zeros(dtype, NumofBasis, blocksize, rsize)
        temp_volume0 = zeros(dtype, blocksize, rsize)

        if accelerator == "cuda"
            iscudadefined = @isdefined CUDA
            if iscudadefined
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
            else
                @warn "CUDA is not used. using CUDA if you want to use gpu. CPU will be used"
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
            accelerator,
            singleprecision
        )
    end
end


function initialize_TA_Gaugefields(u::Gaugefields_4D_accelerator)
    return TA_Gaugefields_4D_accelerator(u.NC, u.NX, u.NY, u.NZ, u.NT,
        u.blockinfo.blocks, accelerator=u.accelerator, singleprecision=u.singleprecision)
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


function substitute_U!(A::TA_Gaugefields_4D_serial{NC}, B::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv}) where {NC,NumofBasis,Ta,TUv}
    bcpu = B.a

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



function Base.similar(u::TA_Gaugefields_4D_accelerator{NC,NumofBasis}) where {NC,NumofBasis}
    return TA_Gaugefields_4D_accelerator(NC, u.NX, u.NY, u.NZ, u.NT, u.blockinfo.blocks; accelerator=u.accelerator, singleprecision=u.singleprecision)
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
    v::TA_Gaugefields_4D_accelerator{2,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,Ta,TUv} #uout = exp(t*u)
    for r = 1:uout.blockinfo.rsize
        for b = 1:uout.blockinfo.blocksize
            kernel_exptU_TAwuww_NC2!(b, r,
                uout.U, v.a, t) #w,u,ww,t
        end
    end

end

function exptU!(
    uout::T,
    t::N,
    v::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NumofBasis,NC,Ta,TUv} #uout = exp(t*u)
    generators = Tuple(v.generators.generator)
    NG = length(generators)
    temp1 = zeros(ComplexF64, NC, NC)
    temp2 = zeros(ComplexF64, NC, NC)

    for r = 1:uout.blockinfo.rsize
        for b = 1:uout.blockinfo.blocksize
            kernel_exptU_TAwuww_NC!(b, r,
                uout.U, v.a, t, NC, NG, generators, temp1, temp2) #w,u,ww,t
        end
    end
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

function gauss_distribution!(
    p::TA_Gaugefields_4D_accelerator{NC,NumofBasis,Ta,TUv};
    σ=1.0,
) where {NC,NumofBasis,Ta,TUv}
    d = Normal(0.0, σ)
    pwork = rand(d, NumofBasis, p.blockinfo.blocksize, p.blockinfo.rsize)
    p.a .= pwork
end

