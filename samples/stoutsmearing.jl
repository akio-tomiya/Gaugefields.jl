# Stout smearing (one step, rho = 0.1) of a hot-start SU(3) configuration on a
# 4^4 lattice, and the analytic pullback of the Wilson-action derivative from
# the smeared links to the thin links (Gaugefields v1 API).
# Run: julia -t 4 --project=. stoutsmearing.jl
import JACC
JACC.@init_backend

using Gaugefields
using LinearAlgebra

function stoutsmearing(lattice, colors)
    U = gauge_configuration(lattice; colors, start=:hot, seed=1234)
    println("thin plaquette = ", measure_plaquette(U))

    stout = StoutSmearing(rho=0.1, iterations=1)

    # smear returns a new configuration and leaves U unchanged.
    @time V = smear(U, stout)
    println("smeared plaquette = ", measure_plaquette(V))

    # The derivative of S[V(U)] with respect to the thin links U.
    beta = 5.7
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, beta / 2, plaquettes)

    # link_smear also returns the cache needed by the reverse pass.
    V, cache = link_smear(U, stout)

    # calc_dSdUμ! uses delta S = Re tr(delta V * dSdV), while the pullback
    # takes the cotangent C with delta S = Re tr(C' * delta V); thus C = dSdV'.
    dSdV = similar(V[1])
    cotangent = similar(V)
    for μ in 1:4
        calc_dSdUμ!(dSdV, action, μ, V)
        clear_U!(cotangent[μ])
        add_U!(cotangent[μ], dSdV')
    end
    set_wing_U!(cotangent)

    @time dU = link_smear_pullback(cotangent, U, cache)

    # Project to the traceless anti-Hermitian MD force, as the HMC driver does.
    force = gauge_momenta(U)
    work = similar(U[1])
    for μ in 1:4
        mul!(work, U[μ], dU[μ]')
        Traceless_antihermitian_add!(force[μ], -1 / colors, work)
    end
    println("squared norm of the force = ", real(force * force))
    return dU
end

lattice = (4, 4, 4, 4)
colors = 3
stoutsmearing(lattice, colors)
