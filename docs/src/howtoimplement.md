# Extending the v1 API

Gaugefields v1 has two intentional application extension points: MD action
providers define the potential and force, and integrators define the ordering
of the elementary gauge and momentum updates. A new storage backend is not
required for ordinary applications; LatticeMatrices/JACC supplies the portable
CPU, GPU, MPI, and multi-GPU field implementation.

## Custom MD action provider

Define a type that owns the immutable action parameters:

~~~julia
struct MyAction{T}
    coupling::T
end
~~~

Then implement the three provider methods:

~~~julia
Gaugefields.md_action_workspace(action::MyAction, U) = MyWorkspace(U)

function Gaugefields.md_potential(action::MyAction, U, workspace)
    # Return the complete real potential V(U).
    return potential
end

function Gaugefields.md_force!(force, action::MyAction, U, workspace)
    # Overwrite every force[mu] with dP[mu]/dtau.
    # Each result must be traceless and anti-Hermitian.
    return nothing
end
~~~

`md_action_workspace` is called once by `md_driver`. Allocate reusable gauge
fields and other large buffers there. `md_potential` and `md_force!` must use
the same sign and normalization convention so that Hamiltonian diagnostics and
the MD evolution agree.

Use the provider like a built-in `GaugeAction`:

~~~julia
action = MyAction(coupling)
driver = md_driver(U, action; steps=20)
P = gaussian_momenta(U; seed=1234, sweep=0)
result = md_trajectory!(U, P, driver)
~~~

## Enzyme-defined force

For a differentiable four-dimensional LM potential, loading Enzyme activates
`enzyme_md_action`:

~~~julia
using Enzyme

function potential(U1, U2, U3, U4, coupling, temps)
    # Reuse temps and return the complete real V(U).
end

action = enzyme_md_action(potential, coupling; num_temps=3)
driver = md_driver(U, action; steps=20)
~~~

The current Enzyme provider requires 4D LatticeMatrices fields with
`halo >= 1`. CUDA multi-GPU use additionally requires CUDA-aware MPI.

## Custom integrator function

An integrator function implements one MD step. The mutated object is always
the first argument of each elementary update:

~~~julia
function my_qpq!(U, P, delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    update_momenta!(P, U, delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    return nothing
end

driver = md_driver(U, action; steps=20, integrator=my_qpq!)
~~~

## Parameterized integrator

Define an ordinary Julia type and extend `Gaugefields.md_step!`:

~~~julia
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
~~~

Subtyping a Gaugefields type is optional. The method must apply exactly one
step and multiply all coefficients by `delta_tau`. HMC integrators should
normally be symmetric and tested for reversibility.

## Storage implementations

The old guide for defining `AbstractGaugefields` subtypes described the
pre-v1 wing, direct-shift, accelerator, and MPI storage families. Those types
remain readable for compatibility, but new portable work should extend
LatticeMatrices rather than add another accelerator-specific Gaugefields
storage tree. The historical implementation contract is retained in
[Legacy API](legacyapi.md).
