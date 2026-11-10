# Measure the plaquette and the Polyakov loop of the stored 8^4 SU(3)
# configuration conf_08080808.ildg (Gaugefields v1 API).
# Run from any directory: julia -t 4 --project=. plaq.jl
import JACC
JACC.@init_backend

using Gaugefields

function main()
    U = gauge_configuration((8, 8, 8, 8); colors=3)
    filename = joinpath(@__DIR__, "conf_08080808.ildg")
    load_configuration!(U, filename; format=:ildg)

    # Normalized observables: a cold configuration gives one for both.
    plaq = measure_plaquette(U)
    poly = measure_polyakov_loop(U)
    println("plaquette = $plaq")
    println("Polyakov loop = $(real(poly)) $(imag(poly))")

    # The historical unnormalized conventions of calculate_Plaquette and
    # calculate_Polyakov_loop.
    println("plaquette (normalize=false) = ", measure_plaquette(U; normalize=false))
    println("Polyakov loop (normalize=false) = ", measure_polyakov_loop(U; normalize=false))
end

main()
