# Stout smearing

Gaugefields v1 exposes stout smearing without requiring applications to
construct the internal neural-network representation.

## One plaquette layer

~~~julia
import JACC
JACC.@init_backend

using Gaugefields

U = gauge_configuration(
    (8, 8, 8, 16);
    colors=3,
    start=:hot,
    seed=0x1234,
)

stout = stout_smearing(U; loops=:plaquette, rho=0.1)
Ustout = smear(U, stout)
~~~

`smear` returns a new configuration and leaves `U` unchanged.

## Multiple loop terms

Provide one coefficient per loop name:

~~~julia
stout = stout_smearing(
    U;
    loops=(:plaquette, :rectangular),
    rho=(0.1, 0.01),
)
Ustout = smear(U, stout)
~~~

A scalar `rho` is broadcast to every requested loop.

## Recording the forward pass

~~~julia
result = smear(U, stout; record=true)

Ustout = result.configuration
history = result.history
derivative = result.derivative
~~~

With the default `calcdSdU=false`, `derivative` is `nothing`. The history
contains the reusable state returned by the underlying smearing pipeline.

## Force boundary

The v1 `smear` function covers the forward transformation. The historical
application-level stout-HMC and explicit back-propagation examples remain in
[Legacy API](legacyapi.md). A new MD action should expose its force through the
`md_action_workspace`, `md_potential`, and `md_force!` provider interface, or
use `enzyme_md_action` where its current 4D LM restrictions are acceptable.

See [Extending the v1 API](howtoimplement.md) and
[HMC and custom integrators](hmc.md).

## Backends

The same call is supported for 2D, 3D, and 4D LM configurations, including
MPI and the JACC GPU backends supported by LatticeMatrices. Backend selection
does not appear in the smearing definition.
