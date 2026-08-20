# for Gaugefields_4D_accelerator
include("./kernelfunctions/gaugefixing_utility_cuda.jl")

function make_g_transform!(
    U::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv},
    temp::Gaugefields_4D_accelerator{NC,TU,TUv},
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int,
    ) where {T<:Gaugefields_4D_accelerator, NC,TU<:CUDA.CuArray,TUv}

    blockinfo = U[1].blockinfo
    u1 = U[1].U
    u2 = U[2].U
    u3 = U[3].U
    u4 = U[4].U

    if D_fix == 3
        CUDA.@sync begin
                CUDA.@cuda threads = U[1].blockinfo.blocksize blocks = U[1].blockinfo.rsize shmem=36864 cudakernel_SU2_subgroup_hit_D3(u1, u2, u3, u4, g.U, parity,
                overrelax, ovr_coeff2, ovr_coeff3, NC, blockinfo) 
        end
    elseif D_fix == 4
        CUDA.@sync begin
                CUDA.@cuda threads = U[1].blockinfo.blocksize blocks = U[1].blockinfo.rsize shmem=36864 cudakernel_SU2_subgroup_hit(u1, u2, u3, u4, g.U, parity,
                overrelax, ovr_coeff2, ovr_coeff3, NC, blockinfo) 
        end
    else
        println_verbose_level1(U[1], "D fix = $D_fix -> not Landau nor Coulomb gauge ")
    end

end


function make_g_steepest_descent!(
    U::Array{T,1},
    g::Gaugefields_4D_accelerator{NC,TU,TUv},
    Δ::Gaugefields_4D_accelerator{NC,TU,TUv},
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix
    ) where {T<:Gaugefields_4D_accelerator, NC,TU<:CUDA.CuArray,TUv}
    
    get_Δ!(Δ, U, temps[1:4], D_fix)

    ##=
    
    blockinfo = U[1].blockinfo
    u1 = U[1].U
    u2 = U[2].U
    u3 = U[3].U
    u4 = U[4].U

    if D_fix == 3
        CUDA.@sync begin
            CUDA.@cuda threads = U[1].blockinfo.blocksize blocks = U[1].blockinfo.rsize cudakernel_mino_method_D3(u1, u2, u3, u4, Δ.U, g.U, parity, overrelax, NC, blockinfo) 
        end
    elseif D_fix == 4
        CUDA.@sync begin
            CUDA.@cuda threads = U[1].blockinfo.blocksize blocks = U[1].blockinfo.rsize cudakernel_mino_method(u1, u2, u3, u4, Δ.U, g.U, parity, overrelax, NC, blockinfo) 
        end
    else
        println_verbose_level1(U[1], "D fix = $D_fix -> not Landau nor Coulomb gauge ")
    end
    normalize_U!(g)

end
