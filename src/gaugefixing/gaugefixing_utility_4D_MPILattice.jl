# for Gaugefields_4D_MPILattice
include("./kernelfunctions/gaugefixing_utility_jacc.jl")

gaugefixing_backend_supported(::Gaugefields_4D_MPILattice) = true


function make_g_los_alamos!(
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
    NC > 3 && normalize_matrix!(g)
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


# ---------------------------------------------------------------------------
# Fused gauge-fixing steps. The field-level versions above build the sums of
# links, Δ and the transformed links with one lattice pass per term (about 80
# passes and 13 halo exchanges per steepest-descent iteration). Here g is built in
# one kernel per checkerboard and each link is transformed in one kernel. The
# arithmetic on each site is that of the field-level operations, so the gauge-fixed
# links are the same. The halos of the links are refreshed lazily: LatticeMatrices
# marks a written field dirty, and ensure_halo! exchanges it before a kernel reads
# neighbours; gaugefixing! returns the links with their halos set.

_link_arrays(U) = ntuple(μ -> U[μ].U.A, length(U))

function _ensure_link_halos!(U, D_fix)
    for μ in 1:D_fix
        ensure_halo!(U[μ].U)
    end
    return nothing
end

function _gauge_transform_links!(U, g::Gaugefields_4D_MPILattice)
    G = g.U
    ensure_halo!(G)
    for μ in eachindex(U)
        L = U[μ].U
        parallel_for_mutating!(L, prod(L.PN), jacckernel_gauge_transform_link!,
            L.A, G.A, L.indexer, μ, Val(G.NC1), Val(L.nw))
    end
    return nothing
end

function gaugefixing_step_sd!(
    U::Array{T,1},
    g_transform::Gaugefields_4D_MPILattice{NC},
    Δ::Gaugefields_4D_MPILattice{NC},
    overrelax::Float64,
    temps::Array{T,1};
    D_fix=4,
) where {NC,T<:Gaugefields_4D_MPILattice}
    G = g_transform.U
    RT = typeof(real(zero(eltype(G.A))))
    for parity in (0, 1)
        _ensure_link_halos!(U, D_fix)
        parallel_for_mutating!(G, prod(G.PN), jacckernel_steepest_descent_fused!,
            G.A, _link_arrays(U), G.indexer, G.coords, G.PN, parity,
            convert(RT, overrelax), D_fix, Val(NC), Val(G.nw))
        normalize_matrix!(G)
        set_halo!(G)
        _gauge_transform_links!(U, g_transform)
    end
    return nothing
end

function gaugefixing_step_la!(
    U::Array{T,1},
    g_transform::Gaugefields_4D_MPILattice{NC},
    overrelax::Float64,
    ovr_coeff2::Float64,
    ovr_coeff3::Float64,
    temp::Gaugefields_4D_MPILattice{NC};
    D_fix=4,
) where {NC,T<:Gaugefields_4D_MPILattice}
    # W is built in one kernel; the SU(2) hits stay in their own kernel, because
    # they multiply StaticArrays matrices, whose fused multiply-adds depend on the
    # surrounding code, and the result would otherwise differ in the last bit.
    W = temp.U
    for parity in (0, 1)
        _ensure_link_halos!(U, D_fix)
        parallel_for_mutating!(W, prod(W.PN), jacckernel_los_alamos_sum!,
            W.A, _link_arrays(U), W.indexer, D_fix, Val(NC), Val(W.nw))
        SU2_subgroup_hit_matrix!(g_transform.U, W, parity, overrelax, ovr_coeff2,
            ovr_coeff3, NC)
        _gauge_transform_links!(U, g_transform)
    end
    return nothing
end

# Δ for the convergence test, in one kernel; as get_Δ!, Δ leaves with its halo set.
function _gaugefixing_delta!(
    Δ::Gaugefields_4D_MPILattice{NC},
    U::Array{T,1},
    temps,
    D_fix,
) where {NC,T<:Gaugefields_4D_MPILattice}
    _ensure_link_halos!(U, D_fix)
    D = Δ.U
    parallel_for_mutating!(D, prod(D.PN), jacckernel_delta_fused!,
        D.A, _link_arrays(U), D.indexer, D_fix, Val(NC), Val(D.nw))
    set_halo!(D)
    return nothing
end

function _gaugefixing_finish!(U::Array{T,1}) where {T<:Gaugefields_4D_MPILattice}
    _ensure_link_halos!(U, length(U))
    return nothing
end
