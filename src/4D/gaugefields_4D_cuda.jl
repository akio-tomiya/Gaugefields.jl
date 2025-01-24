
import CUDA
import ..AbstractGaugefields_module: Adjoint_Gaugefields

struct Blockindices
    blocks::NTuple{4,Int64}
    blocks_s::NTuple{4,Int64}
    blocknumbers::NTuple{4,Int64}
    blocknumbers_s::NTuple{4,Int64}
    blocksize::Int64 #num. of Threads 
    rsize::Int64 #num. of blocks

    function Blockindices(L, blocks)
        blocknumbers = div.(L, blocks)

        dim = length(L)
        blocks_s = ones(dim)
        blocknumbers_s = ones(dim)
        for i in 2:dim
            for j in 1:i-1
                blocknumbers_s[i] = blocknumbers_s[i] * blocknumbers[j]
                blocks_s[i] = blocks_s[i] * blocks[j]
            end
        end

        blocksize = prod(blocks)
        rsize = prod(blocknumbers)

        return new(Tuple(blocks), Tuple(blocks_s), Tuple(blocknumbers), Tuple(blocknumbers_s), blocksize, rsize)

    end
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
        L = [NX, NY, NZ, NT]
        NDW = 0
        #blocksizes = prod(blocks)
        #=
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
        =#


        blockinfo = Blockindices(L, blocks)#Blockindices(Tuple(blocks),Tuple(blocks_s),Tuple(blocknumbers),Tuple(blocknumbers_s),blocksize,rsize)
        blocksize = blockinfo.blocksize
        rsize = blockinfo.rsize

        if CUDA.has_cuda()
            U = zeros(ComplexF64, NC, NC, blocksize, rsize) |> CUDA.CuArray
            temp_volume = zeros(ComplexF64, blocksize, rsize) |> CUDA.CuArray
        else
            @warn "no cuda devise is found. CPU will be used"
            U = zeros(ComplexF64, NC, NC, blocksize, rsize)
            temp_volume = zeros(ComplexF64, blocksize, rsize)
        end
        TU = typeof(U)
        TUv = typeof(temp_volume)

        mpi = false
        verbose_print = Verbose_print(verbose_level)
        #U = Array{Array{ComplexF64,6}}(undef,4)
        #for μ=1:4
        #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
        #end


        return new{NC,TU,TUv}(U, NX, NY, NZ, NT, NDW, NV, NC, mpi, verbose_print,
            #Ushifted,
            blockinfo, temp_volume)
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

function cudakernel_identityGaugefields!(U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_identityGaugefields!(U, NC,b,r)
    #@inbounds for ic = 1:NC
    #    U[ic, ic, b, r] = 1
    #end
end


function identityGaugefields_4D_cuda(NC, NX, NY, NZ, NT, blocks; verbose_level=2)
    U = Gaugefields_4D_cuda(NC, NX, NY, NZ, NT, blocks; verbose_level)

    set_identity!(U)
    return U
    #=
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_identityGaugefields!(U.U, NC)
    end
    =#
end

function set_identity!(U::Gaugefields_4D_cuda{NC,TU,TUv} ) where {NC,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads=U.blockinfo.blocksize blocks=U.blockinfo.rsize cudakernel_identityGaugefields!(U.U,NC)
    end
end

function set_identity!(U::Gaugefields_4D_cuda{NC,TU,TUv} ) where {NC,TU,TUv}
    for r=1:U.blockinfo.rsize
        for b=1:U.blockinfo.blocksize
            kernel_identityGaugefields!(U, NC,b,r)
            #@inbounds for ic=1:NC
            #    U.U[ic,ic,b,r] = 1
            #end 
        end
    end
        #CUDA.@cuda threads=U.blockinfo.blocksize blocks=U.blockinfo.rsize cudakernel_identityGaugefields!(U.U,NC)
    #end
end

function cudakernel_randomGaugefields!(U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_randomGaugefields!(U, NC,b,r)
    #=
    @inbounds for ic = 1:NC
        for jc = 1:NC
            U[jc, ic, b, r] = rand() - 0.5 + im * (rand() - 0.5)
        end
    end
    =#
end



function randomize_U!(U::Gaugefields_4D_cuda{NC,TU,TUv} ) where {NC,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_randomGaugefields!(U.U, NC)
    end
end

function randomize_U!(U::Gaugefields_4D_cuda{NC,TU,TUv} ) where {NC,TU,TUv}
    for r=1:U.blockinfo.rsize
        for b=1:U.blockinfo.blocksize
            kernel_randomGaugefields!(U.U, NC,b,r)
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

    randomize_U!(U)

    normalize_U!(U)
    return U
end

function cudakernel_normalize_U_NC2!(u)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_normalize_U_NC2!(u,b,r)
    return
end



function cudakernel_normalize_U_NC3!(u)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_normalize_U_NC3!(u,b,r)
    return
end



function normalize_U!(U::Gaugefields_4D_cuda{2,TU,TUv} ) where {TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_normalize_U_NC2!(U.U)
    end
end

function normalize_U!(U::Gaugefields_4D_cuda{2,TU,TUv} ) where {TU ,TUv}
    for r=1:U.blockinfo.rsize
        for b=1:U.blockinfo.blocksize
            kernel_normalize_U_NC2!(U.U,b,r)
        end
    end
end

function normalize_U!(U::Gaugefields_4D_cuda{3,TU,TUv} ) where {TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_normalize_U_NC3!(U.U)
    end
end


function normalize_U!(U::Gaugefields_4D_cuda{3,TU,TUv} ) where {TU,TUv}
    for r=1:U.blockinfo.rsize
        for b=1:U.blockinfo.blocksize
            kernel_normalize_U_NC3!(U.U,b,r)
        end
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


function cudakernel_mul_NC!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC!(C, A, B, NC,b,r)
end

function cudakernel_mul_NC!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC!(C, A, B, α, β, NC,b,r)
    return
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::T2) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC!(c.U, a.U, b.U, NC)
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::T2) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU,TUv}
    for r=1:c.blockinfo.rsize
        for b=1:c.blockinfo.blocksize
            kernel_mul_NC!(c.U, a.U, b.U, NC,b,r)
        end
    end

end

function cudakernel_mul_NC3!(C, A, B)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    a11 = A[1, 1, b, r]
    a21 = A[2, 1, b, r]
    a31 = A[3, 1, b, r]
    a12 = A[1, 2, b, r]
    a22 = A[2, 2, b, r]
    a32 = A[3, 2, b, r]
    a13 = A[1, 3, b, r]
    a23 = A[2, 3, b, r]
    a33 = A[3, 3, b, r]
    b11 = B[1, 1, b, r]
    b21 = B[2, 1, b, r]
    b31 = B[3, 1, b, r]
    b12 = B[1, 2, b, r]
    b22 = B[2, 2, b, r]
    b32 = B[3, 2, b, r]
    b13 = B[1, 3, b, r]
    b23 = B[2, 3, b, r]
    b33 = B[3, 3, b, r]
    C[1, 1, b, r] = a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, b, r] = a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, b, r] = a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, b, r] = a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, b, r] = a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, b, r] = a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, b, r] = a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, b, r] = a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, b, r] = a31 * b13 + a32 * b23 + a33 * b33

    return

end

function cudakernel_mul_NC3!(C, A, B, α, β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    a11 = α * A[1, 1, b, r]
    a21 = α * A[2, 1, b, r]
    a31 = α * A[3, 1, b, r]
    a12 = α * A[1, 2, b, r]
    a22 = α * A[2, 2, b, r]
    a32 = α * A[3, 2, b, r]
    a13 = α * A[1, 3, b, r]
    a23 = α * A[2, 3, b, r]
    a33 = α * A[3, 3, b, r]
    b11 = B[1, 1, b, r]
    b21 = B[2, 1, b, r]
    b31 = B[3, 1, b, r]
    b12 = B[1, 2, b, r]
    b22 = B[2, 2, b, r]
    b32 = B[3, 2, b, r]
    b13 = B[1, 3, b, r]
    b23 = B[2, 3, b, r]
    b33 = B[3, 3, b, r]
    C[1, 1, b, r] = β * C[1, 1, b, r] + a11 * b11 + a12 * b21 + a13 * b31
    C[2, 1, b, r] = β * C[2, 1, b, r] + a21 * b11 + a22 * b21 + a23 * b31
    C[3, 1, b, r] = β * C[3, 1, b, r] + a31 * b11 + a32 * b21 + a33 * b31
    C[1, 2, b, r] = β * C[1, 2, b, r] + a11 * b12 + a12 * b22 + a13 * b32
    C[2, 2, b, r] = β * C[2, 2, b, r] + a21 * b12 + a22 * b22 + a23 * b32
    C[3, 2, b, r] = β * C[3, 2, b, r] + a31 * b12 + a32 * b22 + a33 * b32
    C[1, 3, b, r] = β * C[1, 3, b, r] + a11 * b13 + a12 * b23 + a13 * b33
    C[2, 3, b, r] = β * C[2, 3, b, r] + a21 * b13 + a22 * b23 + a23 * b33
    C[3, 3, b, r] = β * C[3, 3, b, r] + a31 * b13 + a32 * b23 + a33 * b33

    return

end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3},
    a::T1,
    b::T2,
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, a.U, b.U)
    end
end

function cudakernel_mul_NC_abdag!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = 0

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    A[k1, k3, b, r] * conj(B[k2, k3, b, r])
            end
        end
    end
end

function cudakernel_mul_NC_abdag!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, b, r] * conj(B[k2, k3, b, r])
            end
        end
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::Adjoint_Gaugefields{T2}) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abdag!(c.U, a.U, b.parent.U, NC)
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC!(c.U, a.U, b.U, α, β, NC)
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, a.U, b.U, α, β, NC)
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abdag!(c.U, a.U, b.parent.U, α, β, NC)
    end
end

function cudakernel_mul_NC_adagbdag!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * conj(B[k2, k3, b, r])
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
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbdag!(c.U, a.parent.U, b.parent.U, α, β, NC)
    end

end

function cudakernel_mul_NC_adagbdag!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = 0

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    conj(A[k3, k1, b, r]) * conj(B[k2, k3, b, r])
            end
        end
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2}
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbdag!(c.U, a.parent.U, b.parent.U, NC)
    end

end

function cudakernel_mul_NC_adagb!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * B[k3, k2, b, r]
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagb!(c.U, a.parent.U, b.U, α, β, NC)
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::Adjoint_Gaugefields{T1},
    b::T2
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagb!(c.U, a.parent.U, b.U, 1, 0, NC)
    end
end

@inline function minusshift(x, xb, xb_s)
    x - (xb - 1) * xb_s
end
@inline function plusshift(x, xb, xb_s)
    x + (xb - 1) * xb_s
end

@inline function shiftedindex_each_plus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bshifted = b
    rshifted = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if (ic1 == blocks - 1)
        bshifted = minusshift(bshifted, blocks, blocks_s)
        #bshifted += - (block-1)*block_s

        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if (ic2 == blocknumbers - 1)
            rshifted = minusshift(rshifted, blocknumbers, blocknumbers_s)
            #rshifted = rshiftfunc(rshifted,blocknumbers ,blocknumbers_s) 
            #rshifted += - (blocknumbers -1)*blocknumbers_s
        else
            rshifted += +blocknumbers_s
        end
    else
        bshifted += blocks_s #shift in idim direction 
        #rshifted = r

    end

    return bshifted, rshifted

end

@inline function shiftedindex_each_minus(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)
    bshifted = b
    rshifted = r
    ic1 = mod(div(b - 1, blocks_s), blocks)
    if ic1 == 0
        #bshifted += + (blocks-1)*blocks_s
        bshifted = plusshift(bshifted, blocks, blocks_s)

        ic2 = mod(div(r - 1, blocknumbers_s), blocknumbers)
        if ic2 == 0
            rshifted = plusshift(rshifted, blocknumbers, blocknumbers_s)
            #rshifted += + (blocknumbers -1)*blocknumbers_s
        else
            rshifted += -blocknumbers_s
        end

    else
        bshifted += -blocks_s #shift in idim direction 
        #rshifted = r
    end

    return bshifted, rshifted

end

@inline function shiftedindex(b, r, shift, blockinfo)
    bshifted = b
    rshifted = r
    for idim = 1:4
        blocks = blockinfo.blocks[idim]
        blocks_s = blockinfo.blocks_s[idim]
        blocknumbers = blockinfo.blocknumbers[idim]
        blocknumbers_s = blockinfo.blocknumbers_s[idim]

        isplus = ifelse(shift[idim] > 0, true, false)
        numshift = abs(shift[idim])
        if isplus
            for ishift = 1:numshift
                bshifted, rshifted = shiftedindex_each_plus(bshifted, rshifted, blocks, blocks_s, blocknumbers, blocknumbers_s)
            end
        else
            for ishift = 1:numshift
                bshifted, rshifted = shiftedindex_each_minus(bshifted, rshifted, blocks, blocks_s, blocknumbers, blocknumbers_s)
            end
        end
    end

    return bshifted, rshifted
end

function cudakernel_mul_NC_abshift!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, b, r] * B[k3, k2, bshifted, rshifted]
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abshift!(c.U, a.U, b.parent.U, α, β,
            b.shift, b.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftb!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted, rshifted] * B[k3, k2, b, r]
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
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftb!(c.U, a.parent.U, b.U, α, β,
            a.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftbshift!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted_a, rshifted_a] * B[k3, k2, bshifted_b, rshifted_b]
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
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftbshift!(c.U, a.parent.U, b.parent.U, α, β,
            a.shift, b.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted_a, rshifted_a] * conj(B[k2, k3, bshifted_b, rshifted_b])
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
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftbshiftdag!(
            c.U, a.parent.U, b.parent.parent.U, α, β,
            a.shift, b.parent.shift, a.parent.blockinfo, NC)
    end
end


function cudakernel_mul_NC_adagbshift!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * B[k3, k2, bshifted, rshifted]
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbshift!(c.U, a.parent.U, b.parent.U, α, β,
            b.shift, b.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_adagbshiftdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, b, r]) * conj(B[k2, k3, bshifted, rshifted])
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbshiftdag!(
            c.U, a.parent.U, b.parent.parent.U, α, β,
            b.parent.shift, b.parent.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftbdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, bshifted, rshifted] * conj(B[k2, k3, b, r])
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftbdag!(c.U, a.parent.U, b.parent.U, α, β,
            a.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftdagbdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted, rshifted]) * conj(B[k2, k3, b, r])
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagbdag!(
            c.U, a.parent.parent.U, b.parent.U, α, β,
            a.parent.shift, a.parent.parent.blockinfo, NC)
    end
end


function cudakernel_mul_NC_abshiftdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * A[k1, k3, b, r] * conj(B[k2, k3, bshifted, rshifted])
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abshiftdag!(c.U, a.U, b.parent.parent.U, α, β,
            b.parent.shift, b.parent.parent.blockinfo, NC)
    end
end



function cudakernel_mul_NC_ashiftdagb!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted, rshifted]) * B[k3, k2, b, r]
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagb!(c.U, a.parent.parent.U, b.U, α, β,
            a.parent.shift, a.parent.parent.blockinfo, NC)
    end
end


function cudakernel_mul_NC_ashiftdagbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted_a, rshifted_a]) * conj(B[k2, k3, bshifted_b, rshifted_b])
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
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagbshiftdag!(c.U, a.parent.parent.U, b.parent.parent.U, α, β,
            a.parent.shift, b.parent.shift, a.parent.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftdagbshift!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    bshifted_a, rshifted_a = shiftedindex(b, r, ashift, blockinfo)
    bshifted_b, rshifted_b = shiftedindex(b, r, bshift, blockinfo)

    @inbounds for k2 = 1:NC
        for k1 = 1:NC
            C[k1, k2, b, r] = β * C[k1, k2, b, r]

            for k3 = 1:NC
                C[k1, k2, b, r] +=
                    α * conj(A[k3, k1, bshifted_a, rshifted_a]) * B[k3, k2, bshifted_b, rshifted_b]
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
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagbshift!(c.U, a.parent.parent.U, b.parent.U, α, β,
            a.parent.shift, b.shift, a.parent.parent.blockinfo, NC)
    end
end


function cudakernel_tr!(temp_volume, U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    temp_volume[b, r] = 0
    @inbounds for k = 1:NC
        temp_volume[b, r] += U[k, k, b, r]
    end
    return
end

function LinearAlgebra.tr(a::Gaugefields_4D_cuda{NC}) where {NC}
    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, NC)
    end

    s = CUDA.reduce(+, a.temp_volume)

    return s

    #println(3*NT*NZ*NY*NX*NC)
    return s
end

function cudakernel_tr!(temp_volume, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    temp_volume[b, r] = 0
    @inbounds for k = 1:NC
        for k2 = 1:NC
            temp_volume[b, r] += A[k, k2, b, r] * B[k2, k, b, r]
        end
    end
    return
end

function LinearAlgebra.tr(
    a::Gaugefields_4D_cuda{NC},
    b::Gaugefields_4D_cuda{NC},
) where {NC}

    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, b.U, NC)
    end

    s = CUDA.reduce(+, a.temp_volume)

    return s
end

@inline convert_x(x, xd, xd_s) = mod(div(x - 1, xd_s), xd)
@inline convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s) = 1 + convert_x(b, blocks, blocks_s) +
                                                                           convert_x(r, blocknumbers, blocknumbers_s) * blocks



function fourdim_cordinate(b, r, blockinfo)
    blocks = blockinfo.blocks[1]
    blocks_s = blockinfo.blocks_s[1]
    blocknumbers = blockinfo.blocknumbers[1]
    blocknumbers_s = blockinfo.blocknumbers_s[1]
    ix = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[2]
    blocks_s = blockinfo.blocks_s[2]
    blocknumbers = blockinfo.blocknumbers[2]
    blocknumbers_s = blockinfo.blocknumbers_s[2]
    iy = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[3]
    blocks_s = blockinfo.blocks_s[3]
    blocknumbers = blockinfo.blocknumbers[3]
    blocknumbers_s = blockinfo.blocknumbers_s[3]
    iz = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    blocks = blockinfo.blocks[4]
    blocks_s = blockinfo.blocks_s[4]
    blocknumbers = blockinfo.blocknumbers[4]
    blocknumbers_s = blockinfo.blocknumbers_s[4]
    it = convert_br(b, r, blocks, blocks_s, blocknumbers, blocknumbers_s)

    return ix, iy, iz, it
end


function substitute_U!(A::Gaugefields_4D_cuda{NC}, B::Gaugefields_4D_nowing{NC}) where {NC}
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

function substitute_U!(A::Gaugefields_4D_nowing{NC}, B::Gaugefields_4D_cuda{NC}) where {NC}
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
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += a[k2, k1, b, r]
        end
    end
    return
end


function add_U!(c::Gaugefields_4D_cuda{NC}, a::T1) where {NC,T1<:Gaugefields_4D_cuda}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U!(c.U, a.U, NC)
    end
end

function cudakernel_add_U_αa!(c, a, α, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += α * a[k2, k1, b, r]
        end
    end
    return
end


function add_U!(
    c::Gaugefields_4D_cuda{NC},
    α::N,
    a::T1,
) where {NC,T1<:Gaugefields_4D_cuda{NC},N<:Number}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αa!(c.U, a.U, α, NC)
    end
end

function cudakernel_add_U_αshifta!(c, a, α, shift, blockinfo, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += α * a[k2, k1, bshifted, rshifted]
        end
    end
    return
end


function add_U!(
    c::Gaugefields_4D_cuda{NC},
    α::N,
    a::T1,
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,N<:Number}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αshifta!(
            c.U, a.parent.U, α,
            a.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_add_U_αadag!(c, a, α, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += α * conj(a[k1, k2, b, r])
        end
    end
    return
end


function add_U!(
    c::Gaugefields_4D_cuda{NC},
    α::N,
    a::Adjoint_Gaugefields{T1},
) where {NC,T1<:Gaugefields_4D_cuda{NC},N<:Number}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αadag!(c.U, a.U, α, NC)
    end
end

function cudakernel_clear_U!(c, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] = 0
        end
    end
    return
end

function clear_U!(c::Gaugefields_4D_cuda{NC}) where {NC}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_clear_U!(c.U, NC)
    end
end



function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2
) where {NC,T1,T2}#

    LinearAlgebra.mul!(c, a, b, 1, 0)
end

function cudakernel_exptU_wvww!(w, v, ww, t, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)


    v11 = v[1, 1, b, r]
    v22 = v[2, 2, b, r]
    v33 = v[3, 3, b, r]

    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im

    v12 = v[1, 2, b, r]
    v13 = v[1, 3, b, r]
    v21 = v[2, 1, b, r]
    v23 = v[2, 3, b, r]
    v31 = v[3, 1, b, r]
    v32 = v[3, 2, b, r]

    x12 = v12 - conj(v21)
    x13 = v13 - conj(v31)
    x23 = v23 - conj(v32)

    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = 0.5 * x21
    y23 = 0.5 * x23
    y31 = 0.5 * x31
    y32 = 0.5 * x32

    c1_0 = (imag(y12) + imag(y21))
    c2_0 = (real(y12) - real(y21))
    c3_0 = (imag(y11) - imag(y22))
    c4_0 = (imag(y13) + imag(y31))
    c5_0 = (real(y13) - real(y31))

    c6_0 = (imag(y23) + imag(y32))
    c7_0 = (real(y23) - real(y32))
    c8_0 = sr3i * (imag(y11) + imag(y22) - 2 * imag(y33))

    c1 = t * c1_0 * 0.5
    c2 = t * c2_0 * 0.5
    c3 = t * c3_0 * 0.5
    c4 = t * c4_0 * 0.5
    c5 = t * c5_0 * 0.5
    c6 = t * c6_0 * 0.5
    c7 = t * c7_0 * 0.5
    c8 = t * c8_0 * 0.5
    csum = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    if csum == 0
        w[1, 1, b, r] = 1
        w[1, 2, b, r] = 0
        w[1, 3, b, r] = 0
        w[2, 1, b, r] = 0
        w[2, 2, b, r] = 1
        w[2, 3, b, r] = 0
        w[3, 1, b, r] = 0
        w[3, 2, b, r] = 0
        w[3, 3, b, r] = 1

        ww[1, 1, b, r] = 1
        ww[1, 2, b, r] = 0
        ww[1, 3, b, r] = 0
        ww[2, 1, b, r] = 0
        ww[2, 2, b, r] = 1
        ww[2, 3, b, r] = 0
        ww[3, 1, b, r] = 0
        ww[3, 2, b, r] = 0
        ww[3, 3, b, r] = 1
        return
    end


    #x[1,1,icum] =  c3+sr3i*c8 +im*(  0.0 )
    v1 = c3 + sr3i * c8
    v2 = 0.0
    #x[1,2,icum] =  c1         +im*( -c2   )
    v3 = c1
    v4 = -c2
    #x[1,3,icum] =  c4         +im*(-c5   )
    v5 = c4
    v6 = -c5

    #x[2,1,icum] =  c1         +im*(  c2   )
    v7 = c1
    v8 = c2

    #x[2,2,icum] =  -c3+sr3i*c8+im*(  0.0 )
    v9 = -c3 + sr3i * c8
    v10 = 0.0

    #x[2,3,icum] =  c6         +im*( -c7   )
    v11 = c6
    v12 = -c7

    #x[3,1,icum] =  c4         +im*(  c5   )
    v13 = c4
    v14 = c5

    #x[3,2,icum] =  c6         +im*(  c7   )
    v15 = c6
    v16 = c7
    #x[3,3,icum] =  -sr3i2*c8  +im*(  0.0 )
    v17 = -sr3i2 * c8
    v18 = 0.0


    #c find eigenvalues of v
    trv3 = (v1 + v9 + v17) / 3.0
    cofac =
        v1 * v9 - v3^2 - v4^2 + v1 * v17 - v5^2 - v6^2 + v9 * v17 - v11^2 -
        v12^2
    det =
        v1 * v9 * v17 - v1 * (v11^2 + v12^2) - v9 * (v5^2 + v6^2) -
        v17 * (v3^2 + v4^2) +
        (v5 * (v3 * v11 - v4 * v12) + v6 * (v3 * v12 + v4 * v11)) * 2.0
    p3 = cofac / 3.0 - trv3^2
    q = trv3 * cofac - det - 2.0 * trv3^3
    x = sqrt(-4.0 * p3) + tinyvalue
    arg = q / (x * p3)

    arg = min(1, max(-1, arg))
    theta = acos(arg) / 3.0
    e1 = x * cos(theta) + trv3
    theta = theta + pi23
    e2 = x * cos(theta) + trv3
    #       theta = theta + pi23
    #       e3 = x * cos(theta) + trv3
    e3 = 3.0 * trv3 - e1 - e2

    # solve for eigenvectors

    w1 = v5 * (v9 - e1) - v3 * v11 + v4 * v12
    w2 = -v6 * (v9 - e1) + v4 * v11 + v3 * v12
    w3 = (v1 - e1) * v11 - v3 * v5 - v4 * v6
    w4 = -(v1 - e1) * v12 - v4 * v5 + v3 * v6
    w5 = -(v1 - e1) * (v9 - e1) + v3^2 + v4^2
    w6 = 0.0

    coeff = 1.0 / sqrt(w1^2 + w2^2 + w3^2 + w4^2 + w5^2)


    w1 = w1 * coeff
    w2 = w2 * coeff
    w3 = w3 * coeff
    w4 = w4 * coeff
    w5 = w5 * coeff

    w7 = v5 * (v9 - e2) - v3 * v11 + v4 * v12
    w8 = -v6 * (v9 - e2) + v4 * v11 + v3 * v12
    w9 = (v1 - e2) * v11 - v3 * v5 - v4 * v6
    w10 = -(v1 - e2) * v12 - v4 * v5 + v3 * v6
    w11 = -(v1 - e2) * (v9 - e2) + v3^2 + v4^2
    w12 = 0.0

    coeff = 1.0 / sqrt(w7^2 + w8^2 + w9^2 + w10^2 + w11^2)

    w7 = w7 * coeff
    w8 = w8 * coeff
    w9 = w9 * coeff
    w10 = w10 * coeff
    w11 = w11 * coeff

    w13 = v5 * (v9 - e3) - v3 * v11 + v4 * v12
    w14 = -v6 * (v9 - e3) + v4 * v11 + v3 * v12
    w15 = (v1 - e3) * v11 - v3 * v5 - v4 * v6
    w16 = -(v1 - e3) * v12 - v4 * v5 + v3 * v6
    w17 = -(v1 - e3) * (v9 - e3) + v3^2 + v4^2
    w18 = 0.0

    coeff = 1.0 / sqrt(w13^2 + w14^2 + w15^2 + w16^2 + w17^2)
    w13 = w13 * coeff
    w14 = w14 * coeff
    w15 = w15 * coeff
    w16 = w16 * coeff
    w17 = w17 * coeff

    # construct the projection v
    c1 = cos(e1)
    s1 = sin(e1)
    ww1 = w1 * c1 - w2 * s1
    ww2 = w2 * c1 + w1 * s1
    ww3 = w3 * c1 - w4 * s1
    ww4 = w4 * c1 + w3 * s1
    ww5 = w5 * c1 - w6 * s1
    ww6 = w6 * c1 + w5 * s1

    c2 = cos(e2)
    s2 = sin(e2)
    ww7 = w7 * c2 - w8 * s2
    ww8 = w8 * c2 + w7 * s2
    ww9 = w9 * c2 - w10 * s2
    ww10 = w10 * c2 + w9 * s2
    ww11 = w11 * c2 - w12 * s2
    ww12 = w12 * c2 + w11 * s2

    c3 = cos(e3)
    s3 = sin(e3)
    ww13 = w13 * c3 - w14 * s3
    ww14 = w14 * c3 + w13 * s3
    ww15 = w15 * c3 - w16 * s3
    ww16 = w16 * c3 + w15 * s3
    ww17 = w17 * c3 - w18 * s3
    ww18 = w18 * c3 + w17 * s3

    w[1, 1, b, r] = w1 + im * w2
    w[1, 2, b, r] = w3 + im * w4
    w[1, 3, b, r] = w5 + im * w6
    w[2, 1, b, r] = w7 + im * w8
    w[2, 2, b, r] = w9 + im * w10
    w[2, 3, b, r] = w11 + im * w12
    w[3, 1, b, r] = w13 + im * w14
    w[3, 2, b, r] = w15 + im * w16
    w[3, 3, b, r] = w17 + im * w18

    ww[1, 1, b, r] = ww1 + im * ww2
    ww[1, 2, b, r] = ww3 + im * ww4
    ww[1, 3, b, r] = ww5 + im * ww6
    ww[2, 1, b, r] = ww7 + im * ww8
    ww[2, 2, b, r] = ww9 + im * ww10
    ww[2, 3, b, r] = ww11 + im * ww12
    ww[3, 1, b, r] = ww13 + im * ww14
    ww[3, 2, b, r] = ww15 + im * ww16
    ww[3, 3, b, r] = ww17 + im * ww18

    #a = ww[:,:,ix,iy,iz,it]
    #b = w[:,:,ix,iy,iz,it]
    #println(b'*a)
    #println(exp(im*t*v[:,:,ix,iy,iz,it]))
    #error("d")



end

function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_cuda{2},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda} #uout = exp(t*u)
    error("exptU with NC=2 is not implemented")
end

function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_cuda{NC},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda,NC} #uout = exp(t*u)
    error("exptU with general NC is not implemented")
end

function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_cuda{3},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_cuda} #uout = exp(t*u)

    ww = temps[1]
    w = temps[2]

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_exptU_wvww!(w.U, v.U, ww.U, t, NC)
    end

    mul!(uout, w', ww)
end



function mul_skiplastindex!(
    c::Gaugefields_4D_cuda{NC},
    a::T1,
    b::T2,
) where {NC,T1<:Abstractfields,T2<:Abstractfields}
    #@assert NC != 2 && NC != 3 "This function is for NC != 2,3"

    mul!(c, a, b)

end

function cudakernel_Traceless_antihermitian_NC3!(vout, vin)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)

    fac13 = 1 / 3

    v11 = vin[1, 1, b, r]
    v21 = vin[2, 1, b, r]
    v31 = vin[3, 1, b, r]

    v12 = vin[1, 2, b, r]
    v22 = vin[2, 2, b, r]
    v32 = vin[3, 2, b, r]

    v13 = vin[1, 3, b, r]
    v23 = vin[2, 3, b, r]
    v33 = vin[3, 3, b, r]


    tri = fac13 * (imag(v11) + imag(v22) + imag(v33))

    #=
    vout[1,1,ix,iy,iz,it] = (imag(v11)-tri)*im
    vout[2,2,ix,iy,iz,it] = (imag(v22)-tri)*im
    vout[3,3,ix,iy,iz,it] = (imag(v33)-tri)*im
    =#
    y11 = (imag(v11) - tri) * im
    y22 = (imag(v22) - tri) * im
    y33 = (imag(v33) - tri) * im



    x12 = v12 - conj(v21)
    x13 = v13 - conj(v31)
    x23 = v23 - conj(v32)

    x21 = -conj(x12)
    x31 = -conj(x13)
    x32 = -conj(x23)

    #=
    vout[1,2,ix,iy,iz,it] = 0.5  * x12
    vout[1,3,ix,iy,iz,it] = 0.5  * x13
    vout[2,1,ix,iy,iz,it] = 0.5  * x21
    vout[2,3,ix,iy,iz,it] = 0.5  * x23
    vout[3,1,ix,iy,iz,it] = 0.5  * x31
    vout[3,2,ix,iy,iz,it] = 0.5  * x32
    =#
    y12 = 0.5 * x12
    y13 = 0.5 * x13
    y21 = 0.5 * x21
    y23 = 0.5 * x23
    y31 = 0.5 * x31
    y32 = 0.5 * x32


    vout[1, 1, b, r] = y11
    vout[2, 1, b, r] = y21
    vout[3, 1, b, r] = y31

    vout[1, 2, b, r] = y12
    vout[2, 2, b, r] = y22
    vout[3, 2, b, r] = y32

    vout[1, 3, b, r] = y13
    vout[2, 3, b, r] = y23
    vout[3, 3, b, r] = y33

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
    vout::Gaugefields_4D_cuda{3},
    vin::Gaugefields_4D_cuda{3},
)
    CUDA.@sync begin
        CUDA.@cuda threads = vout.blockinfo.blocksize blocks = vout.blockinfo.rsize cudakernel_Traceless_antihermitian_NC3!(vout.U, vin.U)
    end

end