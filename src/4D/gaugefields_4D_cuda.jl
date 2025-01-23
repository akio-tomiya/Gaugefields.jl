
import CUDA
import ..AbstractGaugefields_module:Adjoint_Gaugefields

struct Blockindices
    blocks::NTuple{4,Int64}
    blocks_s::NTuple{4,Int64}
    blocknumbers::NTuple{4,Int64}
    blocknumbers_s::NTuple{4,Int64}
    blocksize::Int64 #num. of Threads 
    rsize::Int64 #num. of blocks
end

"""
`Gaugefields_4D_nowing{NC} <: Gaugefields_4D{NC}``

SU(N) Gauge fields in four dimensional lattice. 
"""
struct Gaugefields_4D_cuda{NC,TU,TUv} <: Gaugefields_4D{NC}
    U::TU #CUDA.CuArray{ComplexF64,4}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64
    mpi::Bool
    verbose_print::Verbose_print
    #Ushifted::CUDA.CuArray{ComplexF64,4}
    blockinfo::Blockindices
    temp_volume::TUv
    #blocks::NTuple{4,Int64}
    #blocks_s::NTuple{4,Int64}
    #blocknumbers::NTuple{4,Int64}
    #blocknumbers_s::NTuple{4,Int64}
    #blocksize::Int64 #num. of Threads 
    #rsize::Int64 #num. of blocks


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

        dim = 4
        blocks_s = ones(dim)
        blocknumbers_s = ones(dim)
        for i in 2:dim
            for j in 1:i-1
                blocknumbers_s[i] = blocknumbers_s[i]*blocknumbers[j]
                blocks_s[i] = blocks_s[i]*blocks[j]
            end
        end


        blocksize = prod(blocks)
        rsize = prod(blocknumbers)
        U = zeros(ComplexF64, NC, NC,blocksize,rsize) |> CUDA.CuArray
        TU = typeof(U)

        temp_volume = zeros(ComplexF64, blocksize,rsize) |> CUDA.CuArray
        TUv = typeof(temp_volume)
        #println(typeof(U))
        #U = zeros(ComplexF64, NC, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW)
        #Ushifted =zeros(ComplexF64, NC, NC,blocksize,rsize) |> CUDA.CuArray
        mpi = false
        verbose_print = Verbose_print(verbose_level)
        #U = Array{Array{ComplexF64,6}}(undef,4)
        #for μ=1:4
        #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
        #end

        blockinfo = Blockindices(Tuple(blocks),Tuple(blocks_s),Tuple(blocknumbers),Tuple(blocknumbers_s),blocksize,rsize)
        
        return new{NC,TU,TUv}(U, NX, NY, NZ, NT, NDW, NV, NC, mpi, verbose_print, 
        #Ushifted,
        blockinfo,temp_volume)
    end
end

function Base.similar(U::T) where {T<:Gaugefields_4D_cuda}
    Uout = Gaugefields_4D_cuda(
        U.NC,
        U.NX,
        U.NY,
        U.NZ,
        U.NT,
        U.blockinfo.blocks,
        verbose_level=U.verbose_print.level,
    )
    #identityGaugefields_4D_nowing(U.NC,U.NX,U.NY,U.NZ,U.NT,U.NDW)
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Gaugefields_4D_cuda}
    Uout = Array{T,1}(undef, 4)
    for μ = 1:4
        Uout[μ] = similar(U[μ])
    end
    return Uout
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
        CUDA.@cuda threads=U.blockinfo.blocksize blocks=U.blockinfo.rsize cudakernel_identityGaugefields!(U.U,NC)
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
        CUDA.@cuda threads=U.blockinfo.blocksize blocks=U.blockinfo.rsize cudakernel_randomGaugefields!(U.U,NC)
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

function normalize_U!(U::Gaugefields_4D_cuda{2})


    CUDA.@sync begin
        CUDA.@cuda threads=U.blockinfo.blocksize blocks=U.blockinfo.rsize cudakernel_normalize_U_NC2!(U.U)
    end

end

function normalize_U!(U::Gaugefields_4D_cuda{3})
    CUDA.@sync begin
        CUDA.@cuda threads=U.blockinfo.blocksize blocks=U.blockinfo.rsize cudakernel_normalize_U_NC3!(U.U)
    end

end



function set_wing_U!(u::Array{Gaugefields_4D_cuda{NC},1}) where {NC} #do nothing
    return
end

function set_wing_U!(u::Gaugefields_4D_cuda{NC}) where {NC} #do nothing
    return
end


function substitute_U!(a::Gaugefields_4D_cuda{NC}, b::Gaugefields_4D_cuda{NC}) where {NC}
    a.U .= b.U
    set_wing_U!(a)
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

struct Shifted_Gaugefields_4D_cuda{NC} <: Shifted_Gaugefields{NC,4}
    parent::Gaugefields_4D_cuda{NC}
    #parent::T
    shift::NTuple{4,Int8}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64


    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_Gaugefields_4D_cuda(U::Gaugefields_4D_cuda{NC}, shift) where {NC}
        return new{NC}(U, shift, U.NX, U.NY, U.NZ, U.NT)
    end
end


function shift_U(U::Gaugefields_4D_cuda{NC}, ν::T) where {T<:Integer,NC}
    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    return Shifted_Gaugefields_4D_cuda(U, shift)
end

function shift_U(
    U::TU,
    shift::NTuple{Dim,T},
) where {Dim,T<:Integer,TU<:Gaugefields_4D_cuda}
    return Shifted_Gaugefields_4D_cuda(U, shift)
end


function cudakernel_mul_NC!(C,A,B,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,R] = 0

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    A[k1, k3, b,r] * B[k3, k2, b,r]
            end
        end 
    end
end

function cudakernel_mul_NC!(C,A,B,α,β,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,R] = β*C[k1, k2, b,R] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*A[k1, k3, b,r] * B[k3, k2, b,r]
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}
 
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC!(c.U,a.U,b.U,NC)
    end

end

function cudakernel_mul_NC3!(C,A,B)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    a11 = A[1, 1,b,r]
    a21 = A[2, 1,b,r]
    a31 = A[3, 1,b,r]
    a12 = A[1, 2,b,r]
    a22 = A[2, 2,b,r]
    a32 = A[3, 2,b,r]
    a13 = A[1, 3,b,r]
    a23 = A[2, 3,b,r]
    a33 = A[3, 3,b,r]
    b11 = B[1, 1,b,r]
    b21 = B[2, 1,b,r]
    b31 = B[3, 1,b,r]
    b12 = B[1, 2,b,r]
    b22 = B[2, 2,b,r]
    b32 = B[3, 2,b,r]
    b13 = B[1, 3,b,r]
    b23 = B[2, 3,b,r]
    b33 = B[3, 3,b,r]
    C[1, 1,b,r] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1,b,r] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1,b,r] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2,b,r] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2,b,r] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2,b,r] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3,b,r] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3,b,r] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3,b,r] = a31 * b13 + a32 * b23 + a33 * b33

    return

end

function cudakernel_mul_NC3!(C,A,B,α,β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    a11 = α*A[1, 1,b,r]
    a21 = α*A[2, 1,b,r]
    a31 = α*A[3, 1,b,r]
    a12 = α*A[1, 2,b,r]
    a22 = α*A[2, 2,b,r]
    a32 = α*A[3, 2,b,r]
    a13 = α*A[1, 3,b,r]
    a23 = α*A[2, 3,b,r]
    a33 = α*A[3, 3,b,r]
    b11 = B[1, 1,b,r]
    b21 = B[2, 1,b,r]
    b31 = B[3, 1,b,r]
    b12 = B[1, 2,b,r]
    b22 = B[2, 2,b,r]
    b32 = B[3, 2,b,r]
    b13 = B[1, 3,b,r]
    b23 = B[2, 3,b,r]
    b33 = B[3, 3,b,r]
    C[1, 1,b,r] = β*C[1, 1,b,r]+a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1,b,r] = β*C[2, 1,b,r] + a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1,b,r] = β*C[3, 1,b,r] + a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2,b,r] = β*C[1, 2,b,r] + a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2,b,r] = β*C[2, 2,b,r] + a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2,b,r] = β*C[3, 2,b,r] + a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3,b,r] = β*C[1, 3,b,r] + a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3,b,r] = β*C[2, 3,b,r] + a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3,b,r] = β*C[3, 3,b,r] + a31 * b13 + a32 * b23 + a33 * b33

    return

end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3},
    a::T1,
    b::T2,
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC3!(c.U,a.U,b.U)
    end
end

function cudakernel_mul_NC_abdag!(C,A,B,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = 0

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    A[k1, k3, b,r] * conj(B[k2, k3, b,r])
            end
        end 
    end
end

function cudakernel_mul_NC_abdag!(C,A,B,α,β,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r]

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    β*A[k1, k3, b,r] * conj(B[k2, k3, b,r])
            end
        end 
    end
end

function cudakernel_mul_NC_adagbdag!(C,A,B,α,β,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r]

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    β*conj(A[k3, k1, b,r]) * conj(B[k2, k3, b,r])
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::Adjoint_Gaugefields{T2}) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}
 
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_abdag!(c.U,a.U,b.parent.U,NC)
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
)  where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC!(c.U,a.U,b.U, α,β,NC)
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
)  where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC3!(c.U,a.U,b.U, α,β,NC)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_abdag!(c.U,a.U,b.parent.U,α,β,NC)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}
 
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_adagbdag!(c.U,a.parent.U,b.parent.U,α,β,NC)
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2}
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}
 
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_adagbdag!(c.U,a.parent.U,b.parent.U,0,1,NC)
    end

end

function cudakernel_mul_NC_adagb!(C,A,B,α,β,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r]

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    β*conj(A[k3, k1, b,r]) * B[k3, k2, b,r]
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::T2,
    α::Ta,
    β::Tb
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}
 
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_adagb!(c.U,a.parent.U,b.U,α,β,NC)
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::T2
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}
 
    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_adagb!(c.U,a.parent.U,b.U,0,1,NC)
    end
end

@inline function minusshift(x,xb,xb_s) 
    x -(xb-1)*xb_s
end
@inline function plusshift(x,xb,xb_s) 
    x +(xb-1)*xb_s
end

@inline function shiftedindex_each_plus(b,r,blocks,blocks_s,blocknumbers,blocknumbers_s) 
    bshifted = b
    rshifted = r
    ic1 = mod(div(b-1,blocks_s),blocks)
    if (ic1 == blocks-1)
        bshifted = minusshift(bshifted,blocks,blocks_s)
        #bshifted += - (block-1)*block_s

        ic2 = mod(div(r-1,blocknumbers_s),blocknumbers)
        if (ic2 == blocknumbers-1)
            rshifted = minusshift(rshifted,blocknumbers,blocknumbers_s)
            #rshifted = rshiftfunc(rshifted,blocknumbers ,blocknumbers_s) 
            #rshifted += - (blocknumbers -1)*blocknumbers_s
        else
            rshifted += + blocknumbers_s
        end
    else
        bshifted +=  blocks_s #shift in idim direction 
        #rshifted = r
        
    end

    return bshifted,rshifted
    
end

@inline function shiftedindex_each_minus(b,r,blocks,blocks_s,blocknumbers,blocknumbers_s)
    bshifted = b
    rshifted = r
    ic1 = mod(div(b-1,blocks_s),blocks)
    if ic1 == 0
        #bshifted += + (blocks-1)*blocks_s
        bshifted = plusshift(bshifted,blocks,blocks_s)

        ic2 = mod(div(r-1,blocknumbers_s),blocknumbers)
        if ic2 == 0
            rshifted = plusshift(rshifted,blocknumbers,blocknumbers_s)
            #rshifted += + (blocknumbers -1)*blocknumbers_s
        else
            rshifted += - blocknumbers_s
        end
        
    else
        bshifted += - blocks_s #shift in idim direction 
        #rshifted = r
    end

    return bshifted,rshifted
    
end

@inline function shiftedindex(b,r,shift,blockinfo)
    bshifted = b
    rshifted = r
    for idim=1:4
        blocks = blockinfo.blocks[idim]
        blocks_s = blockinfo.blocks_s[idim]
        blocknumbers = blockinfo.blocknumbers[idim]
        blocknumbers_s = blockinfo.blocknumbers_s[idim]
               
        isplus = ifelse(shift[idim] > 0,true,false)
        numshift = abs(shift[idim])
        if isplus
            for ishift = 1:numshift
                bshifted,rshifted = shiftedindex_each_plus(bshifted,rshifted,blocks,blocks_s,blocknumbers,blocknumbers_s)
            end
        else
            for ishift = 1:numshift
                bshifted,rshifted = shiftedindex_each_minus(bshifted,rshifted,blocks,blocks_s,blocknumbers,blocknumbers_s)
            end
        end
    end

    return bshifted,rshifted
end

function cudakernel_mul_NC_abshift!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*A[k1, k3, b,r] * B[k3, k2, bshifted,rshifted]
            end
        end 
    end
end

function cudakernel_mul_NC_ashiftb!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*A[k1, k3, bshifted,rshifted] * B[k3, k2, b,r]
            end
        end 
    end
end

function cudakernel_mul_NC_ashiftbshift!(C,A,B,α,β,ashift,bshift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted_a,rshifted_a = shiftedindex(b,r,ashift,blockinfo)
    bshifted_b,rshifted_b = shiftedindex(b,r,bshift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*A[k1, k3, bshifted_a,rshifted_a] * B[k3, k2, bshifted_b,rshifted_b]
            end
        end 
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_abshift!(c.U,a.U,b.parent.U,α,β,
                    b.shift,b.parent.blockinfo,NC)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb
    ) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_ashiftb!(c.U,a.parent.U,b.U,α,β,
                    a.shift,a.parent.blockinfo,NC)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb
    ) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_ashiftbshift!(c.U,a.parent.U,b.parent.U,α,β,
                    a.shift,b.shift,a.parent.blockinfo,NC)
    end
end

function cudakernel_mul_NC_adagbshift!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*conj(A[k3, k1, b,r]) * B[k3, k2, bshifted,rshifted]
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::T2,
    α::Ta,
    β::Tb
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_adagbshift!(c.U,a.parent.U,b.parent.U,α,β,
                    b.shift,b.parent.blockinfo,NC)
    end
end

function cudakernel_mul_NC_adagbshiftdag!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*conj(A[k3, k1, b,r]) * conj(B[k2, k3, bshifted,rshifted])
            end
        end 
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_adagbshiftdag!(
                    c.U,a.parent.U,b.parent.parent.U,α,β,
                    b.parent.shift,b.parent.parent.blockinfo,NC)
    end
end

function cudakernel_mul_NC_ashiftbdag!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*A[k1, k3, bshifted,rshifted] * conj(B[k2, k3, b,r])
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
    ) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_ashiftbdag!(c.U,a.parent.U,b.parent.U,α,β,
                    a.shift,a.parent.blockinfo,NC)
    end
end

function cudakernel_mul_NC_ashiftdagbdag!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*conj(A[k3, k1, bshifted,rshifted]) * conj(B[k2, k3, b,r])
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
    ) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_ashiftdagbdag!(
                    c.U,a.parent.parent.U,b.parent.U,α,β,
                    a.parent.shift,a.parent.parent.blockinfo,NC)
    end
end


function cudakernel_mul_NC_abshiftdag!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*A[k1, k3, b,r] * conj(B[k2, k3, bshifted,rshifted])
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
    ) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_abshiftdag!(c.U,a.U,b.parent.parent.U,α,β,
                    b.parent.shift,b.parent.parent.blockinfo,NC)
    end
end



function cudakernel_mul_NC_ashiftdagb!(C,A,B,α,β,shift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted,rshifted = shiftedindex(b,r,shift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*conj(A[k3, k1, bshifted,rshifted]) * B[k3, k2, b,r]
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::T2,
    α::Ta,
    β::Tb
    ) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_ashiftdagb!(c.U,a.parent.parent.U,b.U,α,β,
                    a.parent.shift,a.parent.parent.blockinfo,NC)
    end
end


function cudakernel_mul_NC_ashiftdagbshiftdag!(C,A,B,α,β,ashift,bshift,blockinfo::Blockindices,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted_a,rshifted_a = shiftedindex(b,r,ashift,blockinfo)
    bshifted_b,rshifted_b = shiftedindex(b,r,bshift,blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b,r] = β*C[k1, k2, b,r] 

            for k3 = 1:NC
                C[k1, k2, b,r] +=
                    α*A[k1, k3, bshifted_a,rshifted_a] * B[k3, k2, bshifted_b,rshifted_b]
            end
        end 
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
    ) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads=c.blockinfo.blocksize blocks=c.blockinfo.rsize  cudakernel_mul_NC_ashiftdagbshiftdag!(c.U,a.parent.parent.U,b.parent.parent.U,α,β,
                    a.parent.shift,b.parent.shift,a.parent.parent.blockinfo,NC)
    end
end

function cudakernel_tr!(temp_volume,U,NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    temp_volume[b,r] = 0
    @inbounds for k = 1:NC
        temp_volume[b,r] += U[k,k,b,r]
    end
    return
end

function LinearAlgebra.tr(a::Gaugefields_4D_cuda{NC}) where {NC}
    CUDA.@sync begin
        CUDA.@cuda threads=a.blockinfo.blocksize blocks=a.blockinfo.rsize cudakernel_tr!(a.temp_volume,a.U,NC)
    end

    s = CUDA.mapreduce(real, +, a.temp_volume)

    return s

    #println(3*NT*NZ*NY*NX*NC)
    return s
end

@inline convert_x(x, xd,xd_s) = mod(div(x-1,xd_s),xd)
@inline convert_br(b,r,blocks,blocks_s,blocknumbers,blocknumbers_s) = 1+ convert_x(b, blocks,blocks_s) +
                             convert_x(r, blocknumbers,blocknumbers_s)*blocks 



function fourdim_cordinate(b,r,blockinfo)
    blocks = blockinfo.blocks[1]
    blocks_s = blockinfo.blocks_s[1]
    blocknumbers = blockinfo.blocknumbers[1]
    blocknumbers_s = blockinfo.blocknumbers_s[1]
    ix = convert_br(b,r,blocks,blocks_s,blocknumbers,blocknumbers_s)

    blocks = blockinfo.blocks[2]
    blocks_s = blockinfo.blocks_s[2]
    blocknumbers = blockinfo.blocknumbers[2]
    blocknumbers_s = blockinfo.blocknumbers_s[2]
    iy = convert_br(b,r,blocks,blocks_s,blocknumbers,blocknumbers_s)

    blocks = blockinfo.blocks[3]
    blocks_s = blockinfo.blocks_s[3]
    blocknumbers = blockinfo.blocknumbers[3]
    blocknumbers_s = blockinfo.blocknumbers_s[3]
    iz = convert_br(b,r,blocks,blocks_s,blocknumbers,blocknumbers_s)

    blocks = blockinfo.blocks[4]
    blocks_s = blockinfo.blocks_s[4]
    blocknumbers = blockinfo.blocknumbers[4]
    blocknumbers_s = blockinfo.blocknumbers_s[4]
    it = convert_br(b,r,blocks,blocks_s,blocknumbers,blocknumbers_s)

    return ix,iy,iz,it
end


function substitute_U!(A::Gaugefields_4D_cuda{NC}, B::Gaugefields_4D_nowing{NC}) where {NC}
    acpu = Array(A.U)

    blockinfo = A.blockinfo
    for r = 1:blockinfo.rsize
        for b=1:blockinfo.blocksize
            ix,iy,iz,it = fourdim_cordinate(b,r,blockinfo)
            #println((ix,iy,iz,it))
            for ic=1:NC
                for jc=1:NC
                    acpu[jc,ic,b,r] = B[jc,ic,ix,iy,iz,it] 
                end
            end
        end
    end
    agpu = CUDA.CuArray(acpu)
    A.U .= agpu

end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_nowing,T2<:Gaugefields_4D_cuda}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_nowing}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function substitute_U!(A::Gaugefields_4D_nowing{NC},B::Gaugefields_4D_cuda{NC}) where {NC}
    bcpu = Array(B.U)

    blockinfo = B.blockinfo
    for r = 1:blockinfo.rsize
        for b=1:blockinfo.blocksize
            ix,iy,iz,it = fourdim_cordinate(b,r,blockinfo)
            for ic=1:NC
                for jc=1:NC
                    A[jc,ic,ix,iy,iz,it] = bcpu[jc,ic,b,r] 
                end
            end
        end
    end
end
