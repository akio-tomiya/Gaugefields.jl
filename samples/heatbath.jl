# Heatbath updates of SU(3) gauge fields on a 4^4 lattice with a plaquette
# plus rectangle action, written with the Gaugefields v1 API.
# Run: julia -t 4 --project=. heatbath.jl
import JACC
JACC.@init_backend

using Gaugefields

function heatbath_4d(lattice, beta, colors; sweeps=1000, seed=5678)
    U = gauge_configuration(lattice; colors, start=:cold)

    # Action: (beta/2) * (plaquettes + h.c.) + (beta/2) * (rectangles + h.c.)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, beta / 2, plaquettes)

    rectangles = make_loops_fromname("rectangular", Dim=4)
    append!(rectangles, rectangles')
    push!(action, beta / 2, rectangles)
    show(action)

    # A general-action heatbath updater; construct it once and reuse it.
    updater = heatbath_updater(U, action; seed)

    # measure_polyakov_loop is normalized by NC (a cold start gives 1).
    plaq = measure_plaquette(U)
    poly = measure_polyakov_loop(U)
    println("initial plaquette = $plaq")
    println("initial Polyakov loop = $(real(poly)) $(imag(poly))")

    for sweep in 1:sweeps
        heatbath!(U, updater)
        if sweep % 40 == 0
            plaq = measure_plaquette(U)
            poly = measure_polyakov_loop(U)
            println("$sweep plaquette = $plaq")
            println("$sweep Polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return measure_plaquette(U)
end

lattice = (4, 4, 4, 4)
colors = 3
beta = 5.7
heatbath_4d(lattice, beta, colors)
