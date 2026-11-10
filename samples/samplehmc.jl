# Quenched HMC for SU(3) with the Wilson plaquette action at beta = 6.0,
# written with the Gaugefields v1 MD driver (cold start, 20 QPQ steps of
# length 1/20 per trajectory, 100 trajectories).
# Run: julia -t 4 --project=. samplehmc.jl
# The lattice is 8^4 so that a laptop finishes in a few minutes; the former
# version of this sample used 16^4.
import JACC
JACC.@init_backend

using Gaugefields
using Random

function main()
    lattice = (8, 8, 8, 8)
    U = gauge_configuration(lattice; colors=3, start=:cold)

    beta = 6.0
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, beta / 2, plaquettes)
    show(action)

    md = md_driver(U, action; steps=20, trajectory_length=1.0, integrator=QPQ())

    old_U = copy_configuration(U)
    p = gauge_momenta(U)
    metropolis_rng = Xoshiro(123)
    momentum_seed = 0x5678

    println("initial plaquette = ", measure_plaquette(U))
    numtrj = 100
    numaccepted = 0
    for trajectory in 1:numtrj
        # A new momentum stream for every trajectory.
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

        println("$trajectory plaquette = $(measure_plaquette(U)) ",
                "deltaH = $dH accepted = $accepted")
    end
    println("acceptance ratio = ", numaccepted / numtrj)
end

main()
