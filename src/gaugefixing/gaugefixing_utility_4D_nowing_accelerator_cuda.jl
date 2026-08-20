# Direct CUDA adapter retained from the CUDA-only implementation in PR #155.
# It deliberately uses the accelerator storage's block/rank layout and
# `CUDA.@cuda`; the separate `:jacc` adapter remains the portable path.
include("kernelfunctions/gaugefixing_utility_cuda.jl")

# The original CUDA kernels were specialized to SU(3) in double precision.
# State that limitation explicitly instead of accepting configurations that
# would only fail during GPU compilation.
gaugefixing_backend_supported(
    g::Gaugefields_4D_accelerator{3,TU,TUv,:cuda,TS},
) where {TU,TUv,TS} = eltype(g.U) === ComplexF64

function make_g_transform!(
    U::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TS},
    temp::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TS},
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
) where {NC,TU<:CUDA.CuArray,TUv,TS,T<:Gaugefields_4D_accelerator}
    RT = typeof(real(zero(eltype(g.U))))
    CUDA.@sync begin
        CUDA.@cuda threads=g.blockinfo.blocksize blocks=g.blockinfo.rsize cudakernel_gaugefix_transform!(
                U[1].U,
                U[2].U,
                U[3].U,
                U[4].U,
                g.U,
                parity,
                convert(RT, overrelax),
                convert(RT, ovr_coeff2),
                convert(RT, ovr_coeff3),
                D_fix,
                Val(NC),
                g.blockinfo,
            )
    end
    normalize_U!(g)
    return nothing
end

function make_g_steepest_descent!(
    U::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TS},
    Δ::Gaugefields_4D_accelerator{NC,TU,TUv,:cuda,TS},
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix::Int=4,
) where {NC,TU<:CUDA.CuArray,TUv,TS,T<:Gaugefields_4D_accelerator}
    get_Δ!(Δ, U, temps[1:4], D_fix)
    RT = typeof(real(zero(eltype(g.U))))
    CUDA.@sync begin
        CUDA.@cuda threads=g.blockinfo.blocksize blocks=g.blockinfo.rsize cudakernel_gaugefix_steepest_descent!(
                U[1].U,
                U[2].U,
                U[3].U,
                U[4].U,
                Δ.U,
                g.U,
                parity,
                convert(RT, overrelax),
                D_fix,
                Val(NC),
                g.blockinfo,
            )
    end
    normalize_U!(g)
    return nothing
end
