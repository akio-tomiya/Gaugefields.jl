# Building HMC on the MD driver

Gaugefields owns deterministic molecular-dynamics evolution. An application
adds momentum refresh, Metropolis acceptance, rollback, and the trajectory
counter. Keeping that boundary explicit makes the same MD driver usable in
different HMC algorithms.

## A complete Wilson-action loop

This example runs on a serial LM backend, a GPU, or an MPI process grid. Every
MPI rank must execute the same code and consume the same Metropolis random
stream.

```julia
import JACC
JACC.@init_backend

using Gaugefields
using StableRNGs
using Random

lattice = (8, 8, 8, 8)
U = gauge_configuration(
    lattice;
    colors=3,
    halo=1,
    start=:hot,
    seed=0x1234,
    process_grid=(1, 1, 1, 1),
)

beta = 6.0
action = GaugeAction(U)
plaquettes = make_loops_fromname("plaquette", Dim=4)
append!(plaquettes, plaquettes')
push!(action, beta / 2, plaquettes)

md = md_driver(
    U,
    action;
    steps=20,
    trajectory_length=1.0,
    integrator=QPQ(),
)

old_U = [similar(link) for link in U]
metropolis_rng = StableRNG(0x9abc)
accepted = 0

for trajectory in 1:100
    p = gaussian_momenta(
        U;
        seed=0x5678,
        sweep=trajectory - 1,
    )
    substitute_U!.(old_U, U)

    result = md_trajectory!(U, p, md)
    accept = log(rand(metropolis_rng)) < min(0, -result.delta_hamiltonian)
    if accept
        accepted += 1
    else
        substitute_U!.(U, old_U)
    end

    println(
        trajectory,
        " deltaH=", result.delta_hamiltonian,
        " accepted=", accept,
        " plaquette=", measure_plaquette(U),
    )
end

println("acceptance = ", accepted / 100)
```

The action can contain any closed paths, not only named plaquettes. Path
construction, coefficient conventions, direct action evaluation, and the raw
analytic derivative are described in
[Wilson loops and gauge actions](wilsonloops_actions.md).

`md_driver` can be reused after either acceptance or rollback because its
fields are workspaces, not a saved copy of the configuration. The momenta do
not need to be restored after rejection when the next trajectory refreshes
them.

For an exactly restartable chain, save the configuration, the next trajectory
number (used as `sweep` above), and the state of `metropolis_rng`. A hot-start
seed does not seed either of the later streams. Under MPI, initialize the same
Metropolis RNG on every rank and consume it identically, or draw on rank zero
and broadcast the decision.

## Writing a custom integrator

The elementary update functions put the mutated field first:

```julia
update_momenta!(P, U, delta_tau, driver)
update_gaugefields!(U, P, delta_tau, driver)
```

The first function changes only `P`; the second changes only `U`. Both return
their first argument. Thus the `!`, function name, and argument order all show
which field is updated.

For a one-off scheme, write a function that performs one complete MD step:

```julia
function five_update_pqp!(U, P, delta_tau, driver)
    update_momenta!(P, U, 0.25 * delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    update_momenta!(P, U, 0.5 * delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    update_momenta!(P, U, 0.25 * delta_tau, driver)
    return nothing
end

md = md_driver(U, action; steps=20, integrator=five_update_pqp!)
```

For a named and parameterized scheme, define a normal Julia type in your
application and extend `Gaugefields.md_step!`:

```julia
struct OmelyanPQP{T}
    lambda::T
end

function Gaugefields.md_step!(
    integrator::OmelyanPQP,
    U,
    P,
    delta_tau,
    driver,
)
    lambda = integrator.lambda
    update_momenta!(P, U, lambda * delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    update_momenta!(P, U, (1 - 2lambda) * delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    update_momenta!(P, U, lambda * delta_tau, driver)
    return nothing
end

omelyan = OmelyanPQP(0.1931833275037836)
md = md_driver(U, action; steps=20, integrator=omelyan)
```

No Gaugefields subtype is required. `md_trajectory!` calls `md_step!` exactly
`steps` times, passing `delta_tau = trajectory_length / steps`. The custom
method is responsible for scaling each coefficient by `delta_tau`. Negative
coefficients are supported.

For standard HMC, the composition should normally be palindromic so that a
negative-length trajectory reverses it. A useful test for every new
integrator is to save `U` and `P`, integrate with `trajectory_length=tau`, then
integrate with `-tau` and compare against the saved fields. The integrator is
independent of the action provider, so no changes are needed to use it with an
analytic or Enzyme force.

## Enzyme forces

Loading Enzyme activates `enzyme_md_action`:

```julia
using Enzyme

function potential(U1, U2, U3, U4, coupling, temps)
    # Calculate and return the complete real V(U), reusing `temps`.
end

ad_action = enzyme_md_action(potential, coupling; num_temps=3)
md = md_driver(
    U,
    ad_action;
    steps=20,
    integrator=OmelyanPQP(0.1931833275037836),
)
```

The potential receives the four links separately, followed by constant
arguments, followed by the reusable `temps` collection when `num_temps > 0`. It
must include its own sign and normalization and return a real scalar. The same
function is used for Hamiltonian evaluation and reverse-mode differentiation.
Gaugefields converts the resulting matrix gradient to the common traceless
anti-Hermitian `dp/dtau` representation before the integrator sees it.

Enzyme MD currently requires a four-dimensional LatticeMatrices
configuration with `halo >= 1`. It supports the same CUDA and MPI LM fields as
the underlying Enzyme derivative. CUDA multi-GPU use requires a CUDA-aware MPI
installation and matching launcher/library selection. Use a regular
`GaugeAction` for the analytic Wilson-loop force or for dimensions not yet
covered by the Enzyme provider.

See [High-level API parameters](highlevelapi.md#Analytic-and-Enzyme-forces) for
the custom action-provider protocol.
