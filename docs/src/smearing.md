# Link smearing

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

## Learned L-CNN smearing

Only an LCNN ending in `LExp` is a link smearing. `Plaq`/`LCB` feature models
do not change `U` and are deliberately not accepted by `smear`.

~~~julia
using Random
const LCNN = Gaugefields.LCNN

feature_model = LCNN.LCNNFeatureModel(
    U, 4, 2; kernel_size=2, shifts=:both,
)
link_model = LCNN.LCNNLinkModel(feature_model)
parameters = LCNN.initial_parameters(
    MersenneTwister(1234), link_model, Float32,
)

learned = lcnn_smearing(link_model, parameters)
Ulearned = smear(U, learned)
~~~

`LExp` forms one traceless anti-Hermitian generator per direction and applies
`Unew[mu] = exp(Q[mu]) * U[mu]`. Therefore `Ulearned`, like `Ustout`, is an
SU(N) gauge-link configuration with the correct local gauge transformation
law. See [L-CNN](lcnn.md#update-gauge-links-with-lexp) for the formula, parameter
layout, and tests.

## Recording the forward pass

~~~julia
result = smear(U, stout; record=true)

Ustout = result.configuration
history = result.history
derivative = result.derivative
~~~

With the default `calcdSdU=false`, `derivative` is `nothing`. The history
contains the reusable state returned by the underlying smearing pipeline.
`LCNNLinkSmearing` has the same named-tuple layout; its history is an
`LCNN.LCNNLinkSmearingCache` reusable through the `temps` keyword.

For an L-CNN ending in `LExp`, loading Enzyme enables the recorded VJP:

~~~julia
using Enzyme

result = smear(U, learned; record=true, calcdSdU=true)
dU = result.derivative(dUlearned)

# Includes both link and parameter cotangents.
all_cotangents = LCNN.link_model_pullback(
    dUlearned, U, result.history,
)
~~~

`dUlearned` is the cotangent of the smeared links. A link-valued map has no
single parameter gradient without such an output cotangent, so the API returns
the vector--Jacobian product instead of allocating its full Jacobian.

## Force boundary

Native link smearings and `LCNNLinkSmearing` expose cotangent pullbacks rather
than materialized link Jacobians. `SmearedGaugeAction` currently adapts the
native link-smearing family directly to HMC. An L-CNN MD action should expose
its force through `md_action_workspace`, `md_potential`, and `md_force!`, using
the recorded L-CNN VJP at the smearing boundary.

See [Extending the v1 API](howtoimplement.md) and
[HMC and custom integrators](hmc.md).

## Backends

The same call is supported for 2D, 3D, and 4D LM configurations, including
MPI and the JACC GPU backends supported by LatticeMatrices. Backend selection
does not appear in the smearing definition.
