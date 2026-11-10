# Save an SU(3) configuration in the ILDG format, load it back, and check
# that the plaquette and Polyakov loop are unchanged (Gaugefields v1 API).
# Run: julia -t 4 --project=. ildgtest.jl
import JACC
JACC.@init_backend

using Gaugefields

function main()
    lattice = (4, 4, 4, 4)
    colors = 3

    # A hot start makes the round trip a nontrivial check.
    U = gauge_configuration(lattice; colors, start=:hot, seed=1234)
    plaq = measure_plaquette(U)
    poly = measure_polyakov_loop(U)
    println("before save: plaquette = $plaq")
    println("before save: Polyakov loop = $(real(poly)) $(imag(poly))")

    # Write into a temporary directory so that no file is left behind.
    filename = joinpath(mktempdir(), "configuration.ildg")
    save_configuration(filename, U; format=:ildg)

    # ILDG input needs a preallocated configuration of the same size.
    Uloaded = gauge_configuration(lattice; colors)
    load_configuration!(Uloaded, filename; format=:ildg)

    plaq_loaded = measure_plaquette(Uloaded)
    poly_loaded = measure_polyakov_loop(Uloaded)
    println("after load: plaquette = $plaq_loaded")
    println("after load: Polyakov loop = $(real(poly_loaded)) $(imag(poly_loaded))")
    println("difference of plaquettes = $(plaq_loaded - plaq)")
end

main()
