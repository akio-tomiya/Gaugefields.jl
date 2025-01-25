
import CUDA
import ..AbstractGaugefields_module: Adjoint_Gaugefields


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
    A::T1,
    B::T2) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU,TUv}
    for r=1:c.blockinfo.rsize
        for b=1:c.blockinfo.blocksize
            kernel_mul_NC!(c.U, A.U, B.U, NC,b,r)
        end
    end
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
    b::T2,
    α::Ta,
    β::Tb,
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC!(c.U, a.U, b.U, α, β, NC)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb,
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU,TUv}

    for r=1:c.blockinfo.rsize
        for b=1:c.blockinfo.blocksize
            cudakernel_mul_NC!(c.U, A.U, B.U, α, β, NC,b,r)
        end
    end
end


function cudakernel_mul_NC3!(C, A, B)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3!(C, A, B,b,r)
    return
end





function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3,TU,TUv},
    a::T1,
    b::T2,
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, a.U, b.U)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3,TU,TUv},
    A::T1,
    B::T2,
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU,TUv}
    for r=1:c.blockinfo.rsize
        for b=1:c.blockinfo.blocksize
            kernel_mul_NC3!(c.U, A.U, B.U,b,r)
        end
    end
end

function cudakernel_mul_NC_abdag!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abdag!(C, A, B, NC,b,r)
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    A::T1,
    B::Adjoint_Gaugefields{T2}) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abdag!(c.U, A.U, B.parent.U, NC)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    A::T1,
    B::Adjoint_Gaugefields{T2}) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU,TUv}

    for r=1:c.blockinfo.rsize
        for b=1:c.blockinfo.blocksize
            kernel_mul_NC_abdag!(c.U, A.U, B.parent.U, NC,b,r)
        end
    end
end


function cudakernel_mul_NC3!(C, A, B, α, β)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC3!(C, A, B, α, β,b,r)
    return
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3,TU,TUv},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC3!(c.U, a.U, b.U, α, β, NC)
    end
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{3,TU,TUv},
    A::T1,
    B::T2,
    α::Ta,
    β::Tb,
) where {T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU,TUv}
    for r=1:c.blockinfo.rsize
        for b=1:c.blockinfo.blocksize
            kernel_mul_NC3!(c.U, A.U, B.U, α, β, NC,b,r)
        end
    end
end

function cudakernel_mul_NC_abdag!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abdag!(C, A, B, α, β, NC,b,r)
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abdag!(c.U, a.U, b.parent.U, α, β, NC)
    end
end

function cudakernel_mul_NC_adagbdag!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbdag!(C, A, B, α, β, NC,b,r)
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbdag!(c.U, a.parent.U, b.parent.U, α, β, NC)
    end
end

#c = A'* B'
function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    A::Adjoint_Gaugefields{T1},
    B::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU,TUv}

    for r=1:c.blockinfo.rsize
        for b=1:c.blockinfo.blocksize
            kernel_mul_NC_adagbdag!(c.U, A.parent.U, B.parent.U, α, β, NC,b,r)
        end
    end
end

function cudakernel_mul_NC_adagbdag!(C, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbdag!(C, A, B, NC,b,r)
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2}
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbdag!(c.U, a.parent.U, b.parent.U, NC)
    end

end

function cudakernel_mul_NC_adagb!(C, A, B, α, β, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagb!(C, A, B, α, β, NC,b,r)
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagb!(c.U, a.parent.U, b.U, α, β, NC)
    end

end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::T2
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagb!(c.U, a.parent.U, b.U, 1, 0, NC)
    end
end


function cudakernel_mul_NC_abshift!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abshift!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abshift!(c.U, a.U, b.parent.U, α, β,
            b.shift, b.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftb!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftb!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end




function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftb!(c.U, a.parent.U, b.U, α, β,
            a.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftbshift!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftbshift!(C, A, B, α, β, ashift, bshift, blockinfo, NC,b,r)
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftbshift!(c.U, a.parent.U, b.parent.U, α, β,
            a.shift, b.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo, NC,b,r)
end




function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftbshiftdag!(
            c.U, a.parent.U, b.parent.parent.U, α, β,
            a.shift, b.parent.shift, a.parent.blockinfo, NC)
    end
end


function cudakernel_mul_NC_adagbshift!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbshift!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end



function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbshift!(c.U, a.parent.U, b.parent.U, α, β,
            b.shift, b.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_adagbshiftdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_adagbshiftdag!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end




function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_adagbshiftdag!(
            c.U, a.parent.U, b.parent.parent.U, α, β,
            b.parent.shift, b.parent.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftbdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftbdag!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end



function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftbdag!(c.U, a.parent.U, b.parent.U, α, β,
            a.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftdagbdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagbdag!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagbdag!(
            c.U, a.parent.parent.U, b.parent.U, α, β,
            a.parent.shift, a.parent.parent.blockinfo, NC)
    end
end


function cudakernel_mul_NC_abshiftdag!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_abshiftdag!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end



function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_abshiftdag!(c.U, a.U, b.parent.parent.U, α, β,
            b.parent.shift, b.parent.parent.blockinfo, NC)
    end
end



function cudakernel_mul_NC_ashiftdagb!(C, A, B, α, β, shift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagb!(C, A, B, α, β, shift, blockinfo, NC,b,r)
end




function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagb!(c.U, a.parent.parent.U, b.U, α, β,
            a.parent.shift, a.parent.parent.blockinfo, NC)
    end
end


function cudakernel_mul_NC_ashiftdagbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagbshiftdag!(C, A, B, α, β, ashift, bshift, blockinfo, NC,b,r)
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::Adjoint_Gaugefields{T2},
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagbshiftdag!(c.U, a.parent.parent.U, b.parent.parent.U, α, β,
            a.parent.shift, b.parent.shift, a.parent.parent.blockinfo, NC)
    end
end

function cudakernel_mul_NC_ashiftdagbshift!(C, A, B, α, β, ashift, bshift, blockinfo::Blockindices, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_NC_ashiftdagbshift!(C, A, B, α, β, ashift, bshift, blockinfo, NC,b,r)
end

function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::Adjoint_Gaugefields{T1},
    b::T2,
    α::Ta,
    β::Tb
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,T2<:Shifted_Gaugefields_4D_cuda,Ta<:Number,Tb<:Number,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_mul_NC_ashiftdagbshift!(c.U, a.parent.parent.U, b.parent.U, α, β,
            a.parent.shift, b.shift, a.parent.parent.blockinfo, NC)
    end
end


function cudakernel_tr!(temp_volume, U, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_tr!(temp_volume, U, NC,b,r)
    return
end



function LinearAlgebra.tr(a::Gaugefields_4D_cuda{NC,TU,TUv}) where {NC,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, NC)
    end

    s = CUDA.reduce(+, a.temp_volume)
    return s

end

function cudakernel_tr!(temp_volume, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_tr!(temp_volume, A, B, NC,b,r)
    return
end

function kernel_tr!(temp_volume, A, B, NC,b,r)
    temp_volume[b, r] = 0
    @inbounds for k = 1:NC
        for k2 = 1:NC
            temp_volume[b, r] += A[k, k2, b, r] * B[k2, k, b, r]
        end
    end
    return
end

function LinearAlgebra.tr(
    a::Gaugefields_4D_cuda{NC,TU,TUv},
    b::Gaugefields_4D_cuda{NC,TU,TUv},
) where {NC,TU <: CUDA.CuArray,TUv}

    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_tr!(a.temp_volume, a.U, b.U, NC)
    end

    s = CUDA.reduce(+, a.temp_volume)

    return s
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
    kernel_add_U!(c, a, NC,b,r)
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
    kernel_add_U_αa!(c, a, α, NC,b,r)
    return
end



function add_U!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    α::N,
    a::T1,
) where {NC,T1<:Gaugefields_4D_cuda{NC},N<:Number,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αa!(c.U, a.U, α, NC)
    end
end

function cudakernel_add_U_αshifta!(c, a, α, shift, blockinfo, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_U_αshifta!(c, a, α, shift, blockinfo, NC,b,r)
    return
end

function kernel_add_U_αshifta!(c, a, α, shift, blockinfo, NC,b,r)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    @inbounds for k1 = 1:NC
        for k2 = 1:NC
            c[k2, k1, b, r] += α * a[k2, k1, bshifted, rshifted]
        end
    end
    return
end


function add_U!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    α::N,
    a::T1,
) where {NC,T1<:Shifted_Gaugefields_4D_cuda,N<:Number,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αshifta!(
            c.U, a.parent.U, α,
            a.shift, a.parent.blockinfo, NC)
    end
end

function cudakernel_add_U_αadag!(c, a, α, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_U_αadag!(c, a, α, NC,b,r)
    return
end




function add_U!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    α::N,
    a::Adjoint_Gaugefields{T1},
) where {NC,T1<:Gaugefields_4D_cuda{NC},N<:Number,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_U_αadag!(c.U, a.U, α, NC)
    end
end

function cudakernel_clear_U!(c, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_clear_U!(c, NC,b,r)
    return
end



function clear_U!(c::Gaugefields_4D_cuda{NC,TU,TUv}) where {NC,TU <: CUDA.CuArray,TUv}
    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_clear_U!(c.U, NC)
    end
end


function LinearAlgebra.mul!(
    c::Gaugefields_4D_cuda{NC,TU,TUv},
    a::T1,
    b::T2
) where {NC,T1,T2,TU <: CUDA.CuArray,TUv}#

    LinearAlgebra.mul!(c, a, b, 1, 0)
end

function cudakernel_exptU_wvww!(w, v, ww, t, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_exptU_wvww!(w, v, ww, t, NC,b,r)
    return

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
    kernel_Traceless_antihermitian_NC3!(vout, vin,b,r)
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