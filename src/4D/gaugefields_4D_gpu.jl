using CUDA
using StaticArrays
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

function identityGaugefields_4D_gpu_core!(U::Gaugefields_4D_gpu{A,3,NX,NY,NZ,NT},N) where {A,NX,NY,NZ,NT}
    #@assert NC == 3 "NC should be 3 now! NC = $NC"
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    #@cuprintln("thread $index, block $stride")
    for i = index:stride:N
        ii = (i-1)*3^2 + 1
        U.U[ii] = 1
        ii = (i-1)*3^2 + 4
        U.U[ii] = 1
        ii = (i-1)*3^2 + 9
        U.U[ii] = 1
        #@inbounds y[i] += x[i]
    end
    return nothing
end

function identityGaugefields_4D_gpu(NC, NX, NY, NZ, NT; verbose_level = 2)
    U = Gaugefields_4D_gpu(NC, NX, NY, NZ, NT, verbose_level = verbose_level)
    N = NX*NY*NZ*NT
    numblocks = ceil(Int, N/256)

    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks identityGaugefields_4D_gpu_core!(U,N)
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

    normalize_U!(U)
    return U
end

function normalize_U_each!(u)
    w1 = 0
    w2 = 0
    for ic = 1:3
        w1 += u[2, ic] * conj(u[1, ic])
        w2 += u[1, ic] * conj(u[1, ic])
    end
    zerock2 = w2


    w1 = -w1 / w2

    x4 = (u[2, 1]) + w1 * u[1, 1]
    x5 = (u[2, 2]) + w1 * u[1, 2]
    x6 = (u[2, 3]) + w1 * u[1, 3]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3
    u[2, 1] = x4
    u[2, 2] = x5
    u[2, 3] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1] = u[1, 1] * w2
    u[1, 2] = u[1, 2] * w2
    u[1, 3] = u[1, 3] * w2
    u[2, 1] = u[2, 1] * w3
    u[2, 2] = u[2, 2] * w3
    u[2, 3] = u[2, 3] * w3
    return
end


function normalize_U_core!(U::Gaugefields_4D_gpu{A,3,NX,NY,NZ,NT},N) where {A,NX,NY,NZ,NT}
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    #u = zeros(ComplexF64,3,3)
    NC = 3
    #u = CuArray{ComplexF64}(undef, NC,NC)
    u = zeros(MMatrix{3,3,ComplexF64})
    aa = zeros(MVector{18,Float64})

    u .= 0
    aa .=0
    #u .= 0
    #@cuprintln("thread $index, block $stride")
    for i = index:stride:N
        u[1,1] = U.U[(i-1)*NC^2+1]
        u[2,1] = U.U[(i-1)*NC^2+2]
        u[3,1] = U.U[(i-1)*NC^2+3]

        u[1,2] = U.U[(i-1)*NC^2+4]
        u[2,2] = U.U[(i-1)*NC^2+5]
        u[3,2] = U.U[(i-1)*NC^2+6]

        u[1,3] = U.U[(i-1)*NC^2+7]
        u[2,3] = U.U[(i-1)*NC^2+8]
        u[3,3] = U.U[(i-1)*NC^2+9]

        normalize_U_each!(u)
        m3complv_each!(u,aa)

        U.U[(i-1)*NC^2+1] = u[1,1]
        U.U[(i-1)*NC^2+2] = u[2,1]
        U.U[(i-1)*NC^2+3] = u[3,1]

        U.U[(i-1)*NC^2+4] = u[1,2]
        U.U[(i-1)*NC^2+5] = u[2,2]
        U.U[(i-1)*NC^2+6] = u[3,2]

        U.U[(i-1)*NC^2+7] = u[1,3]
        U.U[(i-1)*NC^2+8] = u[2,3]
        U.U[(i-1)*NC^2+9] = u[3,3]

        #@inbounds y[i] += x[i]
    end
    return nothing    
end


function normalize_U!(U::Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}) where {A,NC,NX,NY,NZ,NT}
    N = NX*NY*NZ*NT
    numblocks = ceil(Int, N/256)

    #u = CUDA.zeros(ComplexF64,NC,NC)

    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks normalize_U_core!(U,N)
    end

    #=

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX

                    w1 = 0
                    w2 = 0
                    @simd for ic = 1:3
                        w1 += u[2, ic] * conj(u[1, ic])
                        w2 += u[1, ic] * conj(u[1, ic])
                    end
                    zerock2 = w2
                    if zerock2 == 0
                        println("w2 is zero  !!  (in normlz)")
                        println(
                            "u[1,1),u[1,2),u[1,3) : ",
                            u[1, 1],
                            "\t",
                            u[1, 2],
                            "\t",
                            u[1, 3],
                        )
                    end

                    w1 = -w1 / w2

                    x4 = (u[2, 1]) + w1 * u[1, 1]
                    x5 = (u[2, 2]) + w1 * u[1, 2]
                    x6 = (u[2, 3]) + w1 * u[1, 3]

                    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

                    zerock3 = w3
                    if zerock3 == 0
                        println("w3 is zero  !!  (in normlz)")
                        println("x4, x5, x6 : $x4, $x5, $x6")
                        exit()
                    end

                    u[2, 1] = x4
                    u[2, 2] = x5
                    u[2, 3] = x6

                    w3 = 1 / sqrt(w3)
                    w2 = 1 / sqrt(w2)

                    u[1, 1] = u[1, 1] * w2
                    u[1, 2] = u[1, 2] * w2
                    u[1, 3] = u[1, 3] * w2
                    u[2, 1] = u[2, 1] * w3
                    u[2, 2] = u[2, 2] * w3
                    u[2, 3] = u[2, 3] * w3

                    if zerock2 * zerock3 == 0
                        println("!! devided by zero !! (in normalize)")
                        println("w2 or w3 in normlz is zero !!")
                        println("w2, w3 : $w2, $w3   ")
                        exit()
                    end
                    #println(u[:,:,ix,iy,iz,it]'*u[:,:,ix,iy,iz,it])
                end
            end
        end
    end
    m3complv!(u)
    =#
end

function m3complv_each!(a,aa)
    #aa = zeros(Float64, 18)

    aa[1] = real(a[1, 1])
    aa[2] = imag(a[1, 1])
    aa[3] = real(a[1, 2])
    aa[4] = imag(a[1, 2])
    aa[5] = real(a[1, 3])
    aa[6] = imag(a[1, 3])
    aa[7] = real(a[2, 1])
    aa[8] = imag(a[2, 1])
    aa[9] = real(a[2, 2])
    aa[10] = imag(a[2, 2])
    aa[11] = real(a[2, 3])
    aa[12] = imag(a[2, 3])

    aa[13] =
        aa[3] * aa[11] - aa[4] * aa[12] - aa[5] * aa[9] + aa[6] * aa[10]
    aa[14] =
        aa[5] * aa[10] + aa[6] * aa[9] - aa[3] * aa[12] - aa[4] * aa[11]
    aa[15] = aa[5] * aa[7] - aa[6] * aa[8] - aa[1] * aa[11] + aa[2] * aa[12]
    aa[16] = aa[1] * aa[12] + aa[2] * aa[11] - aa[5] * aa[8] - aa[6] * aa[7]
    aa[17] = aa[1] * aa[9] - aa[2] * aa[10] - aa[3] * aa[7] + aa[4] * aa[8]
    aa[18] = aa[3] * aa[8] + aa[4] * aa[7] - aa[1] * aa[10] - aa[2] * aa[9]

    a[3, 1] = aa[13] + im * aa[14]
    a[3, 2] = aa[15] + im * aa[16]
    a[3, 3] = aa[17] + im * aa[18]

     #println(a[:,:,ix,iy,iz,it]'*a[:,:,ix,iy,iz,it] )

end