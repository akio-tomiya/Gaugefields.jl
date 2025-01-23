
using Gaugefields
using Random

import Gaugefields.Temporalfields_module: Temporalfields, get_temp, unused!

function heatbathtest_bare()
    Nwing = 0
    nc = 3
    L = [8, 8, 8, 8]
    U = Initialize_Gaugefields(nc, Nwing, L..., condition="cold")
    nsteps = 10
    β = 6.0
    numOR = 4
    temp = Temporalfields(U[1]; num=5)
    t_tot = 0.0

    volume = prod(L)
    dim = length(L)
    factor = 1 / (binomial(dim, 2) * nc * volume)
    measure_every = 10

    for istep = 1:nsteps

        println("# istep = $istep")

        t = @timed begin

            heatbath!(U, temp, β)
            unused!(temp)

            for _ = 1:numOR

                overrelaxation!(U, temp, β)
                unused!(temp)

            end

        end
        t_tot += t.time

        println("Update: Elapsed time $(t.time) [s]")

        plaq = calculate_Plaquette(U, temp[1], temp[2]) * factor
        unused!(temp)
        println("$istep $plaq # plaq")

        if istep % measure_every == 0

            poly = calculate_Polyakov_loop(U, temp[1], temp[2])
            unused!(temp)
            println("$istep $(real(poly)) $(imag(poly)) # poly")

        end

    end

    println("Total elapsed time $t_tot [s]")

end

@test heatbathtest_bare()