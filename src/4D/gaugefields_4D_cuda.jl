using CUDA

"""
`Gaugefields_4D_nowing{NC} <: Gaugefields_4D{NC}``

SU(N) Gauge fields in four dimensional lattice. 
"""
struct Gaugefields_4D_cuda{NC} <: Gaugefields_4D{NC}
    U::CuArray{ComplexF64,6}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64
    mpi::Bool
    verbose_print::Verbose_print
    Ushifted::CuArray{ComplexF64,6}

    function Gaugefields_4D_cuda(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T;
        blocks,
        verbose_level=2,
    ) where {T<:Integer}
        NV = NX * NY * NZ * NT
        NDW = 0
        blocksize = prod(blocks)
        #U =
        U = zeros(ComplexF64, NC, NC, NX + 2NDW, NY + 2NDW, NZ + 2NDW, NT + 2NDW)
        Ushifted = zero(U)
        mpi = false
        verbose_print = Verbose_print(verbose_level)
        #U = Array{Array{ComplexF64,6}}(undef,4)
        #for μ=1:4
        #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
        #end
        return new{NC}(U, NX, NY, NZ, NT, NDW, NV, NC, mpi, verbose_print, Ushifted)
    end
end

function identityGaugefields_4D_cuda(NC, NX, NY, NZ, NT, blocks; verbose_level=2)
    U = Gaugefields_4D_cuda(NC, NX, NY, NZ, NT, blocks, verbose_level=verbose_level)

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
    return U
end