# Applications with the v1 API

The examples on this page share one configuration and use only the public v1
entry points. Select a JACC backend before starting Julia, then initialize it
before loading Gaugefields:

~~~julia
import JACC
JACC.@init_backend

using Gaugefields

U = gauge_configuration(
    (8, 8, 8, 16);
    colors=3,
    halo=1,
    start=:hot,
    seed=0x1234,
)
~~~

## Wilson gradient flow

Construct the reusable flow object and apply it in place:

~~~julia
flow = gradient_flow(U; steps=10, step_size=0.01)

for block in 1:20
    flow!(U, flow)
    println(block, " ", measure_plaquette(U))
end
~~~

One call to `flow!` performs the number of RK3 steps stored in the flow object.

For a general Wilson-loop flow, provide groups of loops and one coefficient
per group:

~~~julia
plaquettes = make_loops_fromname("plaquette", Dim=4)
rectangles = make_loops_fromname("rectangular", Dim=4)

improved_flow = gradient_flow(
    U,
    [plaquettes, rectangles],
    [1.0, 0.1];
    steps=10,
    step_size=0.01,
)
flow!(U, improved_flow)
~~~

## Heatbath and overrelaxation

For the plaquette action, construct an updater once and reuse it:

~~~julia
updater = heatbath_updater(
    U;
    beta=6.0,
    seed=0x5678,
    sweep=0,
)

for sweep in 1:100
    heatbath!(U, updater)
    overrelaxation!(U, updater)

    if sweep % 10 == 0
        println(sweep, " ", measure_plaquette(U))
    end
end
~~~

The updater owns separate heatbath and overrelaxation counters. Reproducible
restart information is described in
[Randomness and reproducibility](randomness.md).

A general `GaugeAction` uses the same application interface:

~~~julia
action = GaugeAction(U)
loops = make_loops_fromname("plaquette", Dim=4)
append!(loops, loops')
push!(action, 6.0 / 2, loops)

general_updater = heatbath_updater(U, action; seed=0x9abc)
heatbath!(U, general_updater)
~~~

Named plaquettes are only the shortest example. See
[Wilson loops and gauge actions](wilsonloops_actions.md) to construct an
arbitrary `Wilsonline`, evaluate the action, and obtain its analytic
derivative.

## Stout smearing

~~~julia
stout = stout_smearing(U; loops=:plaquette, rho=0.1)
Ustout = smear(U, stout)

println("before = ", measure_plaquette(U))
println("after  = ", measure_plaquette(Ustout))
~~~

The input is not overwritten by `smear`. See [Stout smearing](smearing.md) for
history recording and the current force boundary.

## Molecular dynamics

Build the action, refresh momenta explicitly, and reuse one driver:

~~~julia
action = GaugeAction(U)
loops = make_loops_fromname("plaquette", Dim=4)
append!(loops, loops')
push!(action, 6.0 / 2, loops)

driver = md_driver(
    U,
    action;
    steps=20,
    trajectory_length=1.0,
    integrator=QPQ(),
)

P = gaussian_momenta(U; seed=0xdef0, sweep=0)
result = md_trajectory!(U, P, driver)
println("delta H = ", result.delta_hamiltonian)
~~~

`md_trajectory!` is deterministic and does not perform momentum refresh,
Metropolis acceptance, or rollback. The complete application-owned HMC loop is
shown in [HMC and custom integrators](hmc.md).

## Save a checkpoint

~~~julia
save_configuration("checkpoint.jld2", U)
U = load_configuration("checkpoint.jld2")
~~~

A reproducible Markov-chain checkpoint must also store the relevant sweep
counters and Metropolis RNG state; saving `U` alone is not sufficient.

All workflows above use the same source on CPU threads, one GPU, MPI, and
multiple GPUs. Only JACC backend selection and `process_grid` change; see
[MPI, GPU, and multi-GPU execution](mpi.md).
