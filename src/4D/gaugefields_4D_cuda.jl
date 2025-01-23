
import CUDA
"""
`Gaugefields_4D_nowing{NC} <: Gaugefields_4D{NC}``

SU(N) Gauge fields in four dimensional lattice. 
"""
struct Gaugefields_4D_cuda{NC} <: Gaugefields_4D{NC}
    U::CUDA.CuArray{ComplexF64,4}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64
    mpi::Bool
    verbose_print::Verbose_print
    Ushifted::CUDA.CuArray{ComplexF64,4}
    blocks::NTuple{4,Int64}
    blocknumbers::NTuple{4,Int64}
    blocksize::Int64 #num. of Threads 
    rsize::Int64 #num. of blocks


    function Gaugefields_4D_cuda(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
        blocks;
        verbose_level=2,
    ) where {T<:Integer}
        @assert blocks != nothing "blocks should be set!"

        NV = NX * NY * NZ * NT
        L = [NX,NY,NZ,NT]
        NDW = 0
        #blocksizes = prod(blocks)
        blocknumbers = div.(L,blocks)
        blocksize = prod(blocks)
        rsize = prod(blocknumbers)
        U = zeros(ComplexF64, NC, NC,blocksize,rsize) |> CUDA.CuArray
        #println(typeof(U))
        #U = zeros(ComplexF64, NC, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW)
        Ushifted =zeros(ComplexF64, NC, NC,blocksize,rsize) |> CUDA.CuArray
        mpi = false
        verbose_print = Verbose_print(verbose_level)
        #U = Array{Array{ComplexF64,6}}(undef,4)
        #for μ=1:4
        #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
        #end
        
        return new{NC}(U, NX, NY, NZ, NT, NDW, NV, NC, mpi, verbose_print, Ushifted,
        Tuple(blocks),Tuple(blocknumbers),blocksize,rsize)
    end
end

function cudakernel_identityGaugefields!(U,NC) 
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for ic=1:NC
        U[ic,ic,b,r] = 1
    end 
end

function identityGaugefields_4D_cuda(NC, NX, NY, NZ, NT, blocks; verbose_level=2)
    U = Gaugefields_4D_cuda(NC, NX, NY, NZ, NT, blocks; verbose_level)

    CUDA.@sync begin
        CUDA.@cuda threads=U.blocksize blocks=U.rsize cudakernel_identityGaugefields!(U.U,NC)
    end

    return U
end

function cudakernel_randomGaugefields!(U,NC) 
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for ic=1:NC
        for jc=1:NC
            U[jc, ic, b,r] =rand() - 0.5 + im * (rand() - 0.5)
        end
    end
end



function randomGaugefields_4D_cuda(
    NC,
    NX,
    NY,
    NZ,
    NT,
    blocks;
    verbose_level=2,
    randomnumber="Random",
)
    U = Gaugefields_4D_cuda(NC, NX, NY, NZ, NT, blocks; verbose_level)

    if randomnumber == "Random"
    else
        error(
            "randomnumber should be \"Random\" in CUDA version. Now randomnumber = $randomnumber",
        )
    end

    
    CUDA.@sync begin
        CUDA.@cuda threads=U.blocksize blocks=U.rsize cudakernel_randomGaugefields!(U.U,NC)
    end

    normalize_U!(U)
    return U
end

function cudakernel_normalize_U_NC2!(u)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    α = u[1, 1, b,r]
    β = u[2, 1, b,r]
    detU = sqrt(abs(α)^2 + abs(β)^2)
    u[1, 1, b,r] = α / detU
    u[2, 1, b,r] = β / detU
    u[1, 2, b,r] = -conj(β) / detU
    u[2, 2, b,r]= conj(α) / detU
    return 
end

function cudakernel_normalize_U_NC3!(u)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    w1 = 0
    w2 = 0
    @inbounds for ic = 1:3
        w1 += u[2, ic, b,r] * conj(u[1, ic, b,r])
        w2 += u[1, ic, b,r] * conj(u[1, ic, b,r])
    end
    zerock2 = w2


    w1 = -w1 / w2

    x4 = (u[2, 1, b,r]) + w1 * u[1, 1, b,r]
    x5 = (u[2, 2, b,r]) + w1 * u[1, 2, b,r]
    x6 = (u[2, 3, b,r]) + w1 * u[1, 3, b,r]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3

    u[2, 1, b,r] = x4
    u[2, 2, b,r] = x5
    u[2, 3, b,r] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1, b,r] = u[1, 1, b,r] * w2
    u[1, 2, b,r] = u[1, 2, b,r] * w2
    u[1, 3, b,r] = u[1, 3, b,r] * w2
    u[2, 1, b,r] = u[2, 1, b,r] * w3
    u[2, 2, b,r] = u[2, 2, b,r] * w3
    u[2, 3, b,r] = u[2, 3, b,r] * w3

    aa1 = real(u[1, 1, b,r])
    aa2 = imag(u[1, 1, b,r])
    aa3 = real(u[1, 2, b,r])
    aa4 = imag(u[1, 2, b,r])
    aa5 = real(u[1, 3, b,r])
    aa6 = imag(u[1, 3, b,r])
    aa7 = real(u[2, 1, b,r])
    aa8 = imag(u[2, 1, b,r])
    aa9 = real(u[2, 2, b,r])
    aa10 = imag(u[2, 2, b,r])
    aa11 = real(u[2, 3, b,r])
    aa12 = imag(u[2, 3, b,r])

    aa13 =
        aa3 * aa11 - aa4 * aa12 - aa5 * aa9 + aa6 * aa10
    aa14 =
        aa5 * aa10 + aa6 * aa9 - aa3 * aa12 - aa4 * aa11
    aa15 = aa5 * aa7 - aa6 * aa8 - aa1 * aa11 + aa2 * aa12
    aa16 = aa1 * aa12 + aa2 * aa11 - aa5 * aa8 - aa6 * aa7
    aa17 = aa1 * aa9 - aa2 * aa10 - aa3 * aa7 + aa4 * aa8
    aa18 = aa3 * aa8 + aa4 * aa7 - aa1 * aa10 - aa2 * aa9

    u[3, 1, b,r] = aa13 + im * aa14
    u[3, 2, b,r] = aa15 + im * aa16
    u[3, 3, b,r] = aa17 + im * aa18

    return

end

function normalize_U!(u::Gaugefields_4D_cuda{2})
    NX = u.NX
    NY = u.NY
    NZ = u.NZ
    NT = u.NT

    CUDA.@sync begin
        CUDA.@cuda threads=u.blocksize blocks=u.rsize cudakernel_normalize_U_NC2!(u.U)
    end

end

function normalize_U!(u::Gaugefields_4D_cuda{3})
    NX = u.NX
    NY = u.NY
    NZ = u.NZ
    NT = u.NT

    CUDA.@sync begin
        CUDA.@cuda threads=u.blocksize blocks=u.rsize cudakernel_normalize_U_NC3!(u.U)
    end

end



function set_wing_U!(u::Array{Gaugefields_4D_cuda{NC},1}) where {NC} #do nothing
    return
end

function set_wing_U!(u::Gaugefields_4D_cuda{NC}) where {NC} #do nothing
    return
end
