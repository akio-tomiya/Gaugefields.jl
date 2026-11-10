# Wilson gradient flow of a hot-start SU(3) configuration on a 16^4 lattice
# (Gaugefields v1 API). The plaquette approaches one as the flow time grows.
# Run: julia -t 4 --project=. gradientflow.jl
# The same script runs on MPI ranks or GPUs when the backend is selected as in
# docs/src/tutorial4d.md; the former mpigradientflow.jl is in legacy/.
import JACC
JACC.@init_backend

using Gaugefields

function main()
    lattice = (16, 16, 16, 16)
    U = gauge_configuration(lattice; colors=3, start=:hot, seed=1234)
    println("flow time 0.0 plaquette = ", measure_plaquette(U))

    # One flow! call advances 10 RK3 steps of size 0.01.
    step_size = 0.01
    flow = gradient_flow(U; steps=10, step_size)
    for i in 1:10
        @time flow!(U, flow)
        t = 10 * i * step_size
        println("flow time $t plaquette = ", measure_plaquette(U))
    end
end

main()
