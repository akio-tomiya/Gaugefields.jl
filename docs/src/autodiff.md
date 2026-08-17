# Automatic differentiation with Enzyme

Gaugefields supports two related Enzyme workflows on four-dimensional
LatticeMatrices configurations:

- `Enzyme_derivative!` differentiates a real scalar directly with respect to
  four link fields;
- `enzyme_md_action` adapts such a scalar potential to the common MD action
  provider interface.

Use the first form when the matrix gradient itself is the desired result. Use
the second form for HMC or another application built on `md_driver`.

## Differentiate a scalar link function

Enzyme is an optional dependency of Gaugefields and must be added to the
application environment:

~~~julia
pkg> add Enzyme
~~~

Initialize JACC before loading Gaugefields:

~~~julia
import JACC
JACC.@init_backend

using Enzyme
using Gaugefields

U = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=1,
    start=:cold,
)

function link_trace_action(U1, U2, U3, U4)
    return realtrace(U1) + realtrace(U2) +
           realtrace(U3) + realtrace(U4)
end

dU = [similar(U[1]) for _ in 1:4]
clear_U!.(dU)

Enzyme_derivative!(
    link_trace_action,
    U[1], U[2], U[3], U[4],
    dU[1], dU[2], dU[3], dU[4],
)
~~~

The differentiated function must return one real scalar and must receive the
four links as separate arguments. The vector form `f(U)` is deliberately not
accepted by this interface.

## Constant arguments and work fields

Wrap constant arguments with `nodiff`:

~~~julia
Enzyme_derivative!(
    potential,
    U[1], U[2], U[3], U[4],
    dU[1], dU[2], dU[3], dU[4],
    nodiff(coupling),
)
~~~

For a function that uses reusable link-field workspaces, pass matching primal
and adjoint collections. Gaugefields clears both collections before the
reverse pass and supplies the primal collection as the final function
argument:

~~~julia
temps = [similar(U[1]) for _ in 1:3]
dtemps = [similar(U[1]) for _ in 1:3]

Enzyme_derivative!(
    potential_with_temps,
    U[1], U[2], U[3], U[4],
    dU[1], dU[2], dU[3], dU[4],
    nodiff(coupling);
    temp=temps,
    dtemp=dtemps,
)
~~~

The function in this example has the signature
`potential_with_temps(U1, U2, U3, U4, coupling, temps)`.

## Use the differentiated potential in MD

For molecular dynamics, let Gaugefields own gradient allocation and force
projection:

~~~julia
function potential(U1, U2, U3, U4, coupling, temps)
    # Reuse temps and return the complete, real potential V(U).
end

action = enzyme_md_action(potential, coupling; num_temps=3)
driver = md_driver(U, action; steps=20, integrator=QPQ())
~~~

The potential must include its complete sign and normalization. Gaugefields
uses the same function for Hamiltonian evaluation and differentiation, then
converts its matrix gradient to the traceless anti-Hermitian `dp/dτ`
representation expected by every MD integrator.

## Current boundary

The direct and MD Enzyme paths require four-dimensional LM fields; use the
default `halo=1`. They support the JACC CPU and CUDA/MPI paths supported by the
underlying LatticeMatrices Enzyme extension. CUDA multi-GPU differentiation
also requires CUDA-aware MPI and consistent launcher/library selection.

For a Wilson-loop action, the analytic [`GaugeAction`](wilsonloops_actions.md)
force is simpler and also covers 2D and 3D. Use Enzyme when the potential is
more naturally expressed as differentiable lattice algebra.
