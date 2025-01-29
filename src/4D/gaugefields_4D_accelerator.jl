
import CUDA
import ..AbstractGaugefields_module: Adjoint_Gaugefields


"""
`Gaugefields_4D_accelerator{NC} <: Gaugefields_4D{NC}``

SU(N) Gauge fields in four dimensional lattice. 
"""
struct Gaugefields_4D_accelerator{NC,TU,TUv,accdevise,TshifedU} <: Gaugefields_4D{NC}
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
    blockinfo::Blockindices
    temp_volume::TUv
    accelerator::String
    Ushifted::TshifedU #CUDA.CuArray{ComplexF64,4}
    #blocks::NTuple{4,Int64}
    #blocks_s::NTuple{4,Int64}
    #blocknumbers::NTuple{4,Int64}
    #blocknumbers_s::NTuple{4,Int64}
    #blocksize::Int64 #num. of Threads 
    #rsize::Int64 #num. of blocks


    function Gaugefields_4D_accelerator(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
        blocks;
        verbose_level=2,
        accelerator="none"
    ) where {T<:Integer}
        @assert blocks != nothing "blocks should be set!"

        useshiftedU = true
        #useshiftedU = false

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


        if accelerator == "cuda"
            if CUDA.has_cuda()
                U = zeros(ComplexF64, NC, NC, blocksize, rsize) |> CUDA.CuArray
                temp_volume = zeros(ComplexF64, blocksize, rsize) |> CUDA.CuArray
                accdevise = :cuda
            else
                @warn "accelerator=\"cuda\" is set but there is no CUDA devise. CPU will be used"
                U = zeros(ComplexF64, NC, NC, blocksize, rsize)
                temp_volume = zeros(ComplexF64, blocksize, rsize)
                #accdevise = :threads
                accdevise = :none
            end
        elseif accelerator == "threads"
            U = zeros(ComplexF64, NC, NC, blocksize, rsize)
            temp_volume = zeros(ComplexF64, blocksize, rsize)
            accdevise = :threads
        else
            U = zeros(ComplexF64, NC, NC, blocksize, rsize)
            temp_volume = zeros(ComplexF64, blocksize, rsize)
            accdevise = :none
        end

        if useshiftedU
            Ushifted = similar(U)
        else
            Ushifted = nothing
        end

        TU = typeof(U)
        TUv = typeof(temp_volume)

        mpi = false
        verbose_print = Verbose_print(verbose_level)
        #U = Array{Array{ComplexF64,6}}(undef,4)
        #for μ=1:4
        #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
        #end

        TshifedU = typeof(Ushifted)


        return new{NC,TU,TUv,accdevise,TshifedU}(U, NX, NY, NZ, NT, NDW, NV, NC, mpi, verbose_print,
            #Ushifted,
            blockinfo, temp_volume, accelerator,
            Ushifted)
    end
end

function Base.similar(U::T) where {T<:Gaugefields_4D_accelerator}
    Uout = Gaugefields_4D_accelerator(
        U.NC,
        U.NX,
        U.NY,
        U.NZ,
        U.NT,
        U.blockinfo.blocks,
        verbose_level=U.verbose_print.level,
        accelerator=U.accelerator
    )
    #identityGaugefields_4D_nowing(U.NC,U.NX,U.NY,U.NZ,U.NT,U.NDW)
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Gaugefields_4D_accelerator}
    Uout = Array{T,1}(undef, 4)
    for μ = 1:4
        Uout[μ] = similar(U[μ])
    end
    return Uout
end



function identityGaugefields_4D_accelerator(NC, NX, NY, NZ, NT, blocks; verbose_level=2, accelerator="none")
    U = Gaugefields_4D_accelerator(NC, NX, NY, NZ, NT, blocks; verbose_level, accelerator)

    set_identity!(U)
    return U
    #=
    CUDA.@sync begin
        CUDA.@cuda threads = U.blockinfo.blocksize blocks = U.blockinfo.rsize cudakernel_identityGaugefields!(U.U, NC)
    end
    =#
end


function set_identity!(U::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU,TUv}
    for r = 1:U.blockinfo.rsize
        for b = 1:U.blockinfo.blocksize
            kernel_identityGaugefields!(b, r, U.U, NC)
            #@inbounds for ic=1:NC
            #    U.U[ic,ic,b,r] = 1
            #end 
        end
    end
    #CUDA.@cuda threads=U.blockinfo.blocksize blocks=U.blockinfo.rsize cudakernel_identityGaugefields!(U.U,NC)
    #end
end



function randomize_U!(U::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU,TUv}
    for r = 1:U.blockinfo.rsize
        for b = 1:U.blockinfo.blocksize
            kernel_randomGaugefields!(b, r, U.U, NC)
        end
    end
end


function randomGaugefields_4D_accelerator(
    NC,
    NX,
    NY,
    NZ,
    NT,
    blocks;
    verbose_level=2,
    randomnumber="Random",
    accelerator="none"
)
    U = Gaugefields_4D_accelerator(NC, NX, NY, NZ, NT, blocks; verbose_level, accelerator)

    if randomnumber == "Random"
    else
        error(
            "randomnumber should be \"Random\" in accelerator version. Now randomnumber = $randomnumber",
        )
    end

    randomize_U!(U)

    normalize_U!(U)
    return U
end








function normalize_U!(U::Gaugefields_4D_accelerator{2,TU,TUv}) where {TU,TUv}
    for r = 1:U.blockinfo.rsize
        for b = 1:U.blockinfo.blocksize
            kernel_normalize_U_NC2!(b, r, U.U)
        end
    end
end



function normalize_U!(U::Gaugefields_4D_accelerator{3,TU,TUv}) where {TU,TUv}
    for r = 1:U.blockinfo.rsize
        for b = 1:U.blockinfo.blocksize
            kernel_normalize_U_NC3!(b, r, U.U)
        end
    end
end


function set_wing_U!(u::Array{Gaugefields_4D_accelerator{NC},1}) where {NC} #do nothing
    return
end

function set_wing_U!(u::Gaugefields_4D_accelerator{NC}) where {NC} #do nothing
    return
end


function substitute_U!(a::Gaugefields_4D_accelerator{NC}, b::Gaugefields_4D_accelerator{NC}) where {NC}
    a.U .= b.U
    set_wing_U!(a)
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_accelerator}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

struct Shifted_Gaugefields_4D_accelerator{NC} <: Shifted_Gaugefields{NC,4}
    parent::Gaugefields_4D_accelerator{NC}
    #parent::T
    shift::NTuple{4,Int8}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64


    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_Gaugefields_4D_accelerator(U::Gaugefields_4D_accelerator{NC}, shift) where {NC}
        shifted_U!(U, shift)
        return new{NC}(U, shift, U.NX, U.NY, U.NZ, U.NT)
    end
end

function shifted_U!(U::Gaugefields_4D_accelerator{NC,TU,TUv,accdevise,TshifedU}, shift) where {NC,TU,TUv,accdevise,TshifedU<:Nothing}
    return
end



function shifted_U!(U::Gaugefields_4D_accelerator{NC,TU,TUv,accdevise,TshifedU}, shift) where {NC,TU,TUv,accdevise,TshifedU}
    for r = 1:U.blockinfo.rsize
        for b = 1:U.blockinfo.blocksize
            kernel_NC_shiftedU!(b, r, U.Ushifted, U.U,
                shift, U.blockinfo, NC)
        end
    end
end




function shift_U(U::Gaugefields_4D_accelerator{NC}, ν::T) where {T<:Integer,NC}
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

    return Shifted_Gaugefields_4D_accelerator(U, shift)
end

function shift_U(
    U::TU,
    shift::NTuple{Dim,T},
) where {Dim,T<:Integer,TU<:Gaugefields_4D_accelerator}
    return Shifted_Gaugefields_4D_accelerator(U, shift)
end

include("kernelfunctions/linearalgebra_mul_NC.jl")
include("kernelfunctions/linearalgebra_mul_NC3.jl")



function LinearAlgebra.tr(a::Gaugefields_4D_accelerator{NC,TU,TUv,:none}) where {NC,TU,TUv}
    for r = 1:a.blockinfo.rsize
        for b = 1:a.blockinfo.blocksize
            kernel_tr!(b, r, a.temp_volume, a.U, NC)
        end
    end
    s = reduce(+, a.temp_volume)
    return s

end

function LinearAlgebra.tr(a::Gaugefields_4D_accelerator{NC,TU,TUv,:threads}) where {NC,TU,TUv}
    Threads.@threads for r = 1:a.blockinfo.rsize
        Threads.@threads for b = 1:a.blockinfo.blocksize
            kernel_tr!(b, r, a.temp_volume, a.U, NC)
        end
    end
    s = reduce(+, a.temp_volume)
    return s

end



function LinearAlgebra.tr(
    A::Gaugefields_4D_accelerator{NC,TU,TUv,:none},
    B::Gaugefields_4D_accelerator{NC,TU,TUv,:none},
) where {NC,TU,TUv}

    for r = 1:A.blockinfo.rsize
        for b = 1:B.blockinfo.blocksize
            kernel_tr!(b, r, A.temp_volume, A.U, B.U, NC)
        end
    end

    s = reduce(+, A.temp_volume)

    return s
end

function LinearAlgebra.tr(
    A::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
    B::Gaugefields_4D_accelerator{NC,TU,TUv,:threads},
) where {NC,TU,TUv}

    Threads.@threads for r = 1:A.blockinfo.rsize
        Threads.@threads for b = 1:B.blockinfo.blocksize
            kernel_tr!(b, r, A.temp_volume, A.U, B.U, NC)
        end
    end

    s = reduce(+, A.temp_volume)

    return s
end



function substitute_U!(A::Gaugefields_4D_accelerator{NC,TU,TUv,:none,TS}, B::Gaugefields_4D_nowing{NC}) where {NC,TU,TUv,TS}
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
    #agpu = CUDA.CuArray(acpu)
    A.U .= acpu

end



function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_nowing,T2<:Gaugefields_4D_accelerator}

    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1}
) where {T1<:Gaugefields_4D_accelerator,T2<:Gaugefields_4D_nowing}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function substitute_U!(A::Gaugefields_4D_nowing{NC}, B::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TS}) where {NC,TU,TUv,TS}
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

function substitute_U!(A::Gaugefields_4D_nowing{NC}, B::Gaugefields_4D_accelerator{NC,TU,TUv,:none,TS}) where {NC,TU,TUv,TS}
    bcpu = B.U

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





function add_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv}, a::T1) where {NC,T1<:Gaugefields_4D_accelerator,TU,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_U!(b, r, c.U, a.U, NC)
        end
    end
end



function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv},
    α::N,
    a::T1,
) where {NC,T1<:Gaugefields_4D_accelerator{NC},N<:Number,TU,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_U_αa!(b, r, c.U, a.U, α, NC)
        end
    end
end



function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv},
    α::N,
    a::T1,
) where {NC,T1<:Shifted_Gaugefields_4D_accelerator,N<:Number,TU,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_U_αshifta!(b, r,
                c.U, a.parent.U, α,
                a.shift, a.parent.blockinfo, NC)
        end
    end
end



function add_U!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv},
    α::N,
    a::Adjoint_Gaugefields{T1},
) where {NC,T1<:Gaugefields_4D_accelerator{NC},N<:Number,TU,TUv}

    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_U_αadag!(b, r, c.U, a.U, α, NC)
        end
    end
end


function clear_U!(c::Gaugefields_4D_accelerator{NC,TU,TUv}) where {NC,TU,TUv}
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_clear_U!(b, r, c.U, NC)
        end
    end

end



function LinearAlgebra.mul!(
    c::Gaugefields_4D_accelerator{NC,TU,TUv},
    a::T1,
    b::T2
) where {NC,T1,T2,TU,TUv}#
    LinearAlgebra.mul!(c, a, b, 1, 0)
end



function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_accelerator{2},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator} #uout = exp(t*u)
    error("exptU with NC=2 is not implemented")
end

function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_accelerator{NC},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,NC} #uout = exp(t*u)
    error("exptU with general NC is not implemented")
end




function exptU!(
    uout::T,
    t::N,
    v::Gaugefields_4D_accelerator{3,TU,TUv},
    temps::Array{T,1},
) where {N<:Number,T<:Gaugefields_4D_accelerator,TU,TUv} #uout = exp(t*u)

    ww = temps[1]
    w = temps[2]

    for r = 1:v.blockinfo.rsize
        for b = 1:v.blockinfo.blocksize
            kernel_exptU_wvww!(b, r, w.U, v.U, ww.U, t, NC)
            #kernel_clear_U!(b,r,c.U, NC)
        end
    end

    mul!(uout, w', ww)
end




function mul_skiplastindex!(
    c::Gaugefields_4D_accelerator{NC},
    a::T1,
    b::T2,
) where {NC,T1<:Abstractfields,T2<:Abstractfields}
    #@assert NC != 2 && NC != 3 "This function is for NC != 2,3"

    mul!(c, a, b)

end



#Q = -(1/2)*(Ω' - Ω) + (1/(2NC))*tr(Ω' - Ω)*I0_2
#Omega' - Omega = -2i imag(Omega)
function Traceless_antihermitian!(
    vout::Gaugefields_4D_accelerator{3,TU,TUv},
    vin::Gaugefields_4D_accelerator{3},
) where {TU,TUv}

    for r = 1:vout.blockinfo.rsize
        for b = 1:vout.blockinfo.blocksize
            kernel_Traceless_antihermitian_NC3!(b, r, vout.U, vin.U)
        end
    end

end


function partial_tr(a::Gaugefields_4D_accelerator{NC,TU,TUv}, μ) where {NC,TU,TUv}
    for r = 1:a.blockinfo.rsize
        for b = 1:a.blockinfo.blocksize
            kernel_partial_tr!(b, r, a.temp_volume, a.U, NC, a.blockinfo, μ)
            #kernel_clear_U!(b,r,c.U, NC)
        end
    end
    s = reduce(+, a.temp_volume)

    return s
end

