using CUDA
"""
`Gaugefields_4D_gpu{NC} <: Gaugefields_4D{NC}``

SU(N) Gauge fields in four dimensional lattice. 
"""
struct Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT} <: Gaugefields_4D{NC}
    U::A
    Ushifted::A
    verbose_level::Int64
    #verbose_print::Verbose_print
end

function Gaugefields_4D_gpu(
    NC::T,
    NX::T,
    NY::T,
    NZ::T,
    NT::T;
    verbose_level = 2,
) where {T<:Integer}
    NV = NX * NY * NZ * NT
    NDW = 0
    U =  CuArray(zeros(ComplexF64, NC, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW) )
    Ushifted = zero(U) 
    mpi = false
    verbose_print = Verbose_print(verbose_level)
    A = typeof(U)
    #U = Array{Array{ComplexF64,6}}(undef,4)
    #for μ=1:4
    #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
    #end
    #return Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}(U,verbose_print, Ushifted)
    return Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}(U,Ushifted,verbose_level)
end

function Gaugefields_4D_gpu(U,Ushifted,verbose_level)
    NC,_,NX,NY,NZ,NT = size(U)
    A = typeof(U)
    Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}(U,Ushifted,verbose_level)
end

import Adapt

Adapt.@adapt_structure Gaugefields_4D_gpu
#=
function Adapt.adapt_structure(to, U::Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}) where {A,NC,NX,NY,NZ,NT}
    U_c = Adapt.adapt_structure(to, U.U)
    Ushifted_c = Adapt.adapt_structure(to, U.Ushifted)
    #verbose_print_c = Adapt.adapt_structure(to, U.verbose_print)
    #Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}(U_c,verbose_print_c, Ushifted_c)
    Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}(U_c, Ushifted_c)
end
=#







@inline function Base.getindex(x::Gaugefields_4D_gpu, i)
    @inbounds return x.U[i]
end


function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1},
) where {T1<:Gaugefields_4D_gpu,T2<:Gaugefields_4D_gpu}
    for μ = 1:4
        substitute_U!(a[μ], b[μ])
    end
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1},
    iseven,
) where {T1<:Gaugefields_4D_gpu,T2<:Gaugefields_4D_gpu}
    for μ = 1:4
        substitute_U!(a[μ], b[μ], iseven)
    end
end

function Base.similar(U::Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}) where {A,NC,NX,NY,NZ,NT}
    Uout = Gaugefields_4D_gpu(
        NC,
        NX,
        NY,
        NZ,
        NT,
        verbose_level = U.verbose_level
    )
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Gaugefields_4D_gpu}
    Uout = Array{T,1}(undef, 4)
    for μ = 1:4
        Uout[μ] = similar(U[μ])
    end
    return Uout
end

function substitute_U!(a::T, b::T) where {T<:Gaugefields_4D_gpu}
    for i = 1:length(a.U)
        a.U .= b.U
    end
    return
end

function IdentityGauges_4D(NC, NX, NY, NZ, NT; verbose_level = 2)
    return identityGaugefields_4D_gpu(NC, NX, NY, NZ, NT, verbose_level = verbose_level)
end

#=
function identityGaugefields_4D_gpu_core!(U::Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT},N) where {A,NC,NX,NY,NZ,NT}
    error("NC = $NC is not supported in identityGaugefields_4D_gpu_core!")
end
=#

function identityGaugefields_4D_gpu_core!(U::Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT},N) where {A,NC,NX,NY,NZ,NT}
    #@assert NC == 3 "NC should be 3 now! NC = $NC"
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    #@cuprintln("thread $index, block $stride")
    for i = index:stride:N
        ii = (i-1)*NC^2 + 1
        U.U[ii] = 1
        ii = (i-1)*NC^2 + 4
        U.U[ii] = 1
        ii = (i-1)*NC^2 + 9
        U.U[ii] = 1
        #@inbounds y[i] += x[i]
    end
    return nothing
end

function identityGaugefields_4D_gpu(NC, NX, NY, NZ, NT; verbose_level = 2)
    U = Gaugefields_4D_gpu(NC, NX, NY, NZ, NT, verbose_level = verbose_level)
    N = NX*NY*NZ*NT

    CUDA.@sync begin
        @cuda threads=N identityGaugefields_4D_gpu_core!(U,N)
    end
    #=

    kernel = @cuda launch=false identityGaugefields_4D_gpu_core!(U,N)
    config = launch_configuration(kernel.fun)
    @cuprintln("threads $(config.threads)")
    threads = min(N, config.threads)
    @cuprintln("threads $threads")
    blocks = cld(N, threads)
    @cuprintln("threads $threads, blocks $blocks")

    CUDA.@sync begin
        kernel(U; threads, blocks)
    end
    =#

    return U
end

function randomGaugefields_4D_gpu_core!(U,NNC,rng)#,randomnumber = "Random")

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    #@cuprintln("thread $index, block $stride")
    for i = index:stride:N
        ii = (i-1)*NC^2 + 1
        U.U[ii] = 1
        ii = (i-1)*NC^2 + 4
        U.U[ii] = 1
        ii = (i-1)*NC^2 + 9
        U.U[ii] = 1
        #@inbounds y[i] += x[i]
    end

end

function randomGaugefields_4D_gpu(
    NC,
    NX,
    NY,
    NZ,
    NT;
    verbose_level = 2
)
    U = Gaugefields_4D_gpu(NC, NX, NY, NZ, NT, verbose_level = verbose_level)
    U.U .=  Random.rand(CURAND.default_rng(), Float64, NC,NC,NX,NY,NZ,NT) .-0.5 .+ im.*( Random.rand(CURAND.default_rng(), Float64, NC,NC,NX,NY,NZ,NT) .- 0.5)

   # normalize_U!(U)
    return U
end