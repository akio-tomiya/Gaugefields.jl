# Hybrid Monte Carlo

Gaugefields supports two HMC construction styles. The traditional style below
combines the momentum force, link exponential, integrator, Hamiltonian, and
Metropolis test directly. The second style delegates deterministic evolution
to the reusable MD driver. Both remain supported.

## Traditional HMC assembled from elementary operations

This is the historical construction used by Gaugefields applications. It is
shown explicitly because it remains useful when an application needs complete
control over every update stage.

```julia
using Gaugefields
using LinearAlgebra
using Random

import Gaugefields.Temporalfields_module:
    Temporalfields, get_temp, unused!

function traditional_hamiltonian(action, U, P)
    potential = -real(evaluate_GaugeAction(action, U)) / U[1].NC
    kinetic = real(P * P) / 2
    return potential + kinetic
end

function traditional_link_update!(U, P, coefficient, delta_tau, temps)
    temp1, i1 = get_temp(temps)
    temp2, i2 = get_temp(temps)
    exponential, iexp = get_temp(temps)
    work, iwork = get_temp(temps)
    for direction in eachindex(U)
        exptU!(
            exponential,
            coefficient * delta_tau,
            P[direction],
            [temp1, temp2],
        )
        mul!(work, exponential, U[direction])
        substitute_U!(U[direction], work)
    end
    unused!(temps, i1)
    unused!(temps, i2)
    unused!(temps, iexp)
    unused!(temps, iwork)
    return U
end

function traditional_momentum_update!(P, U, action, coefficient, delta_tau, temps)
    derivative, iderivative = get_temp(temps)
    work, iwork = get_temp(temps)
    factor = -coefficient * delta_tau / U[1].NC
    for direction in eachindex(U)
        calc_dSdUμ!(derivative, action, direction, U)
        mul!(work, U[direction], derivative)
        Traceless_antihermitian_add!(P[direction], factor, work)
    end
    unused!(temps, iderivative)
    unused!(temps, iwork)
    return P
end

function traditional_hmc_trajectory!(U, P, old_U, action, temps, steps)
    delta_tau = 1 / steps
    gauss_distribution!(P)
    initial_hamiltonian = traditional_hamiltonian(action, U, P)
    substitute_U!(old_U, U)

    for _ in 1:steps
        traditional_link_update!(U, P, 1 / 2, delta_tau, temps)
        traditional_momentum_update!(P, U, action, 1, delta_tau, temps)
        traditional_link_update!(U, P, 1 / 2, delta_tau, temps)
    end

    final_hamiltonian = traditional_hamiltonian(action, U, P)
    delta_hamiltonian = final_hamiltonian - initial_hamiltonian
    accepted = log(rand()) < min(0, -delta_hamiltonian)
    accepted || substitute_U!(U, old_U)
    return (; accepted, delta_hamiltonian)
end

Random.seed!(0x1234)
U = Initialize_Gaugefields(
    3, 0, 8, 8, 8, 8;
    condition="hot",
    randomnumber="Reproducible",
)
action = GaugeAction(U)
plaquettes = make_loops_fromname("plaquette", Dim=4)
append!(plaquettes, plaquettes')
push!(action, 6.0 / 2, plaquettes)

P = initialize_TA_Gaugefields(U)
old_U = similar(U)
substitute_U!(old_U, U)
temps = Temporalfields(U[1]; num=8)

for trajectory in 1:100
    result = traditional_hmc_trajectory!(
        U, P, old_U, action, temps, 20,
    )
    println(
        trajectory,
        " deltaH=", result.delta_hamiltonian,
        " accepted=", result.accepted,
    )
end
```

This form intentionally exposes all temporary-field and force-normalization
details. The application owns every allocation and must keep the update
composition symmetric. The historical API is also collected on the
[Legacy API](legacyapi.md#hybrid-monte-carlo) page.

## HMC using the MD driver

With the driver, Gaugefields owns deterministic molecular-dynamics evolution.
The application still owns momentum refresh, Metropolis acceptance, rollback,
and the trajectory counter.

### A complete Wilson-action loop

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

old_U = copy_configuration(U)
p = gauge_momenta(U)
metropolis_rng = StableRNG(0x9abc)
accepted = 0

for trajectory in 1:100
    gaussian_momenta!(
        p;
        seed=0x5678,
        sweep=trajectory - 1,
    )
    copy_configuration!(old_U, U)

    result = md_trajectory!(U, p, md)
    accept = log(rand(metropolis_rng)) < min(0, -result.delta_hamiltonian)
    if accept
        accepted += 1
    else
        copy_configuration!(U, old_U)
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

The gauge configuration checkpoint is the portable JLD2 file:

```julia
save_configuration("checkpoint.jld2", U)
load_configuration!(U, "checkpoint.jld2")
```

For MPI or multi-GPU fields every rank calls these functions. Rank 0 gathers
one global configuration and is the only rank that writes the JLD2 file.

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

## Sexton--Weingarten time-scale separation

Expensive and inexpensive forces can be integrated at different time scales
without introducing another MD driver. Give each action provider a name and
select the slow and fast groups in `SextonWeingarten`:

```julia
actions = MDActionSet(;
    gauge=gauge_action,
    fermion=fermion_action,
)

integrator = SextonWeingarten(;
    slow=:fermion,
    fast=:gauge,
    n_fast=4,
)

md = md_driver(
    U,
    actions;
    steps=10,
    trajectory_length=1.0,
    integrator,
)
```

The default `QPQ()` ordering performs one slow-force evaluation and
`2n_fast` fast-force evaluations per outer step. `ordering=PQP()` performs two
slow-force evaluations and `n_fast` fast-force evaluations per outer step.
Both compositions are symmetric and support a negative trajectory length for
reversibility tests.

The historical LatticeQCD QPQ implementation counted the fast substeps across
both halves and required an even `N_SextonWeingarten`. Its equivalent setting
is `n_fast = N_SextonWeingarten ÷ 2`; the new parameter counts substeps in
each half and may be any positive integer.

Groups may contain several terms:

```julia
actions = MDActionSet(;
    gauge=gauge_action,
    light=light_pseudofermion_action,
    strange=strange_pseudofermion_action,
)

integrator = SextonWeingarten(;
    slow=(:light, :strange),
    fast=:gauge,
    n_fast=4,
)
```

All terms must occur in exactly one group. Pseudofermion refresh remains an
HMC-layer operation and must happen before `md_trajectory!`; the deterministic
trajectory holds those pseudofermion fields fixed.

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
arguments, followed by the reusable `temps` collection when `num_temps > 0`.
The links and work fields seen inside the potential are the underlying
LatticeMatrices objects; the simulation variable `U` remains the normal vector
of Gaugefields links. Use `mul_shifted!`, `mul_shifted_adjoint!`, and
`mul_adjoint!` for shifted products in differentiated code. The potential must
include its own sign and normalization and return a real scalar. The same
function is used for Hamiltonian evaluation and reverse-mode differentiation.
Gaugefields converts the resulting matrix gradient to the common traceless
anti-Hermitian `dp/dtau` representation before the integrator sees it.

Enzyme MD currently requires a four-dimensional LatticeMatrices
configuration with `halo >= 1`. It supports the same CUDA and MPI LM fields as
the underlying Enzyme derivative. CUDA multi-GPU use requires a CUDA-aware MPI
installation and matching launcher/library selection. Use a regular
`GaugeAction` for the analytic Wilson-loop force or for dimensions not yet
covered by the Enzyme provider.

See [Automatic differentiation with Enzyme](autodiff.md) for a complete
plaquette potential and the supported product operations.

See [High-level API parameters](highlevelapi.md#Analytic-and-Enzyme-forces) for
the custom action-provider protocol.
