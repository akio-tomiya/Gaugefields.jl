using CUDA
"""
`Gaugefields_4D_gpu{NC} <: Gaugefields_4D{NC}``

SU(N) Gauge fields in four dimensional lattice. 
"""
struct Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT} <: Gaugefields_4D{NC}
    U::A
    #verbose_print::Verbose_print
    Ushifted::A

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
    return Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}(U,Ushifted)
end

function Gaugefields_4D_gpu(U,Ushifted)
    NC,_,NX,NY,NZ,NT = size(U)
    A = typeof(U)
    Gaugefields_4D_gpu{A,NC,NX,NY,NZ,NT}(U,Ushifted)
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


