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

function identityGaugefields_4D_gpu(NC, NX, NY, NZ, NT; verbose_level = 2)
    U = Gaugefields_4D_gpu(NC, NX, NY, NZ, NT, verbose_level = verbose_level)

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    @simd for ic = 1:NC
                        U[ic, ic, ix, iy, iz, it] = 1
                    end
                end
            end
        end
    end
    set_wing_U!(U)
    return U
end