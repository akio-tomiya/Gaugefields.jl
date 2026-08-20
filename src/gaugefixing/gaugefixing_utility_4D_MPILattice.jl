# for Gaugefields_4D_MPILattice
include("./kernelfunctions/gaugefixing_utility_jacc.jl")

gaugefixing_backend_supported(::Gaugefields_4D_MPILattice) = true


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
    
    W = temp
    clear_U!(W)

    #Compute W = U_μ(x) + U_μ^†( x - μ)
    for μ in 1:D_fix
        U_shift = shift_U(U[μ], -μ)
        add_U!(W, U[μ])
        add_U!(W, U_shift')
    end

    SU2_subgroup_hit_matrix!(g.U, W.U, parity, overrelax, ovr_coeff2, ovr_coeff3, g.NC)
    return nothing
end


function SU2_subgroup_hit_matrix!(
    g::LatticeMatrix{D,T,AT,NC1,NC2,nw,DI}, W,
    parity::Int, overrelax::Float64, ovr_coeff2::Float64, ovr_coeff3::Float64,
    NC::Int) where {D,T,AT,NC1,NC2,nw,DI}

    RT = typeof(real(zero(T)))
    JACC.parallel_for(
        prod(g.PN),
        jacckernel_SU2_subgroup_hit!,
        g.A,
        W.A,
        g.indexer,
        g.coords,
        g.PN,
        parity,
        convert(RT, overrelax),
        convert(RT, ovr_coeff2),
        convert(RT, ovr_coeff3),
        Val(NC),
        Val(nw),
    )
    normalize_matrix!(g)
    set_halo!(g)
    return nothing
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
    
    get_Δ!(Δ, U, temps[1:4], D_fix)

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

    jacc_steepest_decent!(g.U, Um.U, Up.U, Δ.U, parity, overrelax)
    return nothing
end


function jacc_steepest_decent!(
    g::LatticeMatrix{D,T,AT,NC,NC,nw,DI},
    Um,
    Up,
    Δ,
    parity::Int,
    overrelax::Float64,
) where {D,T,AT,NC,nw,DI}
    RT = typeof(real(zero(T)))
    JACC.parallel_for(
        prod(g.PN),
        jacckernel_mino_method!,
        g.indexer,
        g.A,
        Um.A,
        Up.A,
        Δ.A,
        g.coords,
        g.PN,
        parity,
        convert(RT, overrelax),
        Val(NC),
        Val(nw),
    )
    normalize_matrix!(g)
    set_halo!(g)
    return nothing
end
