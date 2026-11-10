# Quenched HMC for SU(3) on a 4^4 lattice in which the Wilson plaquette action
# (beta = 5.7) is evaluated on stout-smeared links (one step, rho = 0.1).
# The MD force is pulled back analytically to the thin links (Gaugefields v1).
# Run: julia -t 4 --project=. stouthmc.jl
# Each trajectory of length 1 uses 40 QPQ steps. The former version of this
# sample used 10 steps, with which |deltaH| is about 17 from this hot start
# and no trajectory is accepted.
import JACC
JACC.@init_backend

using Gaugefields
using Random

function main()
    lattice = (4, 4, 4, 4)
    U = gauge_configuration(lattice; colors=3, start=:hot, seed=1234)

    beta = 5.7
    thin_link_action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(thin_link_action, beta / 2, plaquettes)
    show(thin_link_action)

    # S[U] = S_W[V(U)] with V the stout-smeared links.
    stout = StoutSmearing(rho=0.1, iterations=1)
    action = SmearedGaugeAction(thin_link_action, stout)

    md = md_driver(U, action; steps=40, trajectory_length=1.0, integrator=QPQ())

    old_U = copy_configuration(U)
    p = gauge_momenta(U)
    metropolis_rng = Xoshiro(123)
    momentum_seed = 0x5678

    numtrj = 100
    numaccepted = 0
    for trajectory in 1:numtrj
        gaussian_momenta!(p; seed=momentum_seed, sweep=trajectory - 1)
        copy_configuration!(old_U, U)

        result = md_trajectory!(U, p, md)
        dH = result.delta_hamiltonian
        accepted = log(rand(metropolis_rng)) < min(0, -dH)
        if accepted
            numaccepted += 1
        else
            copy_configuration!(U, old_U)
        end

        plaq_thin = measure_plaquette(U)
        plaq_smeared = measure_plaquette(smear(U, stout))
        println("$trajectory plaquette = $plaq_thin smeared plaquette = $plaq_smeared ",
                "deltaH = $dH accepted = $accepted")
    end
    println("acceptance ratio = ", numaccepted / numtrj)
end

main()
