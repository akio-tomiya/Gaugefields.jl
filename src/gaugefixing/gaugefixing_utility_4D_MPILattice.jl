# for Gaugefields_4D_MPILattice
include("./kernelfunctions/gaugefixing_utility_jacc.jl")


# For Gaugefields_4D_MPILattice
function trace_AAdagger(dA::Gaugefields_4D_MPILattice{NC}, temp::Gaugefields_4D_MPILattice{NC}; D_fix::Int = 4) where {NC}

    trace = zero(eltype(dA.U.A))
    
    mul!(temp, dA, dA')
    trace += tr(temp) 
    return real(trace / (NC * dA.NV*D_fix))
end

function make_g_transform!(
    U::Array{T,1},
    g::Gaugefields_4D_MPILattice,
    temp::Gaugefields_4D_MPILattice,
    parity::Int,
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    D_fix::Int=4,
    ) where {T<:Gaugefields_4D_MPILattice}
    
    @timeit to "make W" begin
    W = temp
    clear_U!(W)

    #Compute W = U_μ(x) + U_μ^†( x - μ)
    for μ in 1:D_fix
        U_shift = shift_U(U[μ], -μ)
        add_U!(W, U[μ])
        add_U!(W, U_shift')
    end
    end

    SU2_subgroup_hit_matrix!(g.U, W.U, parity, overrelax, ovr_coeff2, ovr_coeff3, g.NC)
    
end


function SU2_subgroup_hit_matrix!(
    g::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, W,
    parity::Int, overrelax::Float64, ovr_coeff2::Float64, ovr_coeff3::Float64,
    NC::Int) where {D,T,AT,NC1,NC2,nw,DI}

    @timeit to "LA_kernel" JACC.parallel_for(prod(g.PN), jacckernel_SU2_subgroup_hit!, g.A, W.A, g.indexer, parity, overrelax, ovr_coeff2, ovr_coeff3, Val(NC), Val(nw))
    @timeit to "set_halo" set_halo!(g)
end


function make_g_steepest_descent!(
    U::Array{T,1},
    g::Gaugefields_4D_MPILattice,
    Δ::Gaugefields_4D_MPILattice,
    parity::Int,
    overrelax::Float64,
    temps::Array{T,1},
    D_fix::Int=4,
    ) where {T<:Gaugefields_4D_MPILattice}
    
    @timeit to "get_delta" get_Δ!(Δ, U, temps[1:4], D_fix)

    @timeit to "make Up & Um" begin

    Um = temps[1]
    clear_U!(Um)
    
    Up = temps[2]
    clear_U!(Up)
    
    for μ in 1:D_fix
        U_shift = shift_U(U[μ], -μ)
        #Compute Um = U_μ(x) - U_μ( x - μ)
        add_U!(Um, U[μ])
        add_U!(Um, -1, U_shift)

        #Compute Up = U_μ(x) + U_μ( x - μ)
        add_U!(Up, U[μ])
        add_U!(Up, U_shift)
    end
    end

    jacc_steepest_decent!(g.U, Um.U, Up.U, Δ.U, parity, overrelax)
end


function jacc_steepest_decent!(g::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, Um, Up, Δ, parity::Int, overrelax::Float64) where {D,T,AT,NC1,NC2,nw,DI}
    @timeit to "SD_kernel" JACC.parallel_for(prod(g.PN), jacckernel_mino_method!, g.indexer, g.A, Um.A, Up.A, Δ.A, parity, overrelax, Val(nw))
    @timeit to "set_halo" set_halo!(g)
end
