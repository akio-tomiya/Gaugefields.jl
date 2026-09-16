# L-CNN

Lattice Gauge Equivariant Convolutional Neural Networks (L-CNNs) were
introduced by Favoni, Ipp, Müller, and Schuh in
[*Physical Review Letters* **128**, 032003 (2022)](https://doi.org/10.1103/PhysRevLett.128.032003)
([arXiv:2012.12901](https://arxiv.org/abs/2012.12901)). Their
[reference implementation](https://gitlab.com/openpixi/lge-cnn) is written in
PyTorch. `Gaugefields.LCNN` implements the same gauge-equivariant construction
for `LatticeMatrices`-backed gauge fields and connects it to Gaugefields link
smearing, Enzyme differentiation, and Julia training tools.

The paper formulates the layers for SU(`N_c`) in arbitrary lattice dimension.
Its numerical experiments use SU(2) configurations in 1+1 and 3+1 dimensions.
The implementation described here supports two-, three-, and four-dimensional
SU(`N_c`) fields; the test suite covers SU(2) and SU(3).

## Definition of the implemented model

Let `d = length(U)` be the number of lattice directions. The link fields and
the `N_\ell` feature channels at layer `\ell` transform as

```math
U_{x,\mu}\mapsto
\Omega_xU_{x,\mu}\Omega^\dagger_{x+\hat\mu},
\qquad
W^{(\ell)}_{x,j}\mapsto
\Omega_xW^{(\ell)}_{x,j}\Omega_x^\dagger.
```

Below, values passed to constructors such as `kernel_size` and `component`
are called **structural arguments**. They select an equation or tensor shape
but are not optimized. **Trainable parameters** means only the real arrays in
the `NamedTuple` returned by `LCNN.initial_parameters`.

### Plaquette input

`LCNNFeatureModel` begins with `Plaq()`. It constructs one positively oriented
untraced plaquette for every `\mu<\nu`,

```math
W^{(0)}_{x,(\mu,\nu)}=
U_{x,\mu}U_{x+\hat\mu,\nu}
U^\dagger_{x+\hat\nu,\mu}U^\dagger_{x,\nu}.
```

Thus the initial width is

```math
N_0=\binom{d}{2}.
```

`Plaq` has no trainable parameters. The first `LCB` must accept `N_0` input
channels.

### One LCB layer

Favoni et al. define separate L-Conv and L-Bilin operations in Eqs. (5) and
(6). Their released code combines them into L-CB (Supplementary Material,
Eqs. (11)--(12)); L-CB means the combined lattice gauge-equivariant
convolution-bilinear layer. `LCNN.LCB` implements this parametrization.

For an input channel `j` and a path descriptor `p=(\mu,k)`, define

```math
\widetilde W^{(\ell-1)}_{x,j,p}
=P_{x,\mu,k}W^{(\ell-1)}_{x+k\hat\mu,j}
 P^\dagger_{x,\mu,k},
```

where `P_{x,\mu,k}` is the straight Wilson line from `x+k\hat\mu` to `x`.
The set of descriptors `\mathcal P_\ell` is determined exactly by the LCB
constructor arguments.

For `convention=:favoni_prl`, let `K = kernel_size`,
`\delta = dilation`, and let `\Sigma=\{+1\}` for `shifts=:positive` or
`\Sigma=\{-1,+1\}` for `shifts=:both`. Then

```math
\mathcal P_\ell=
\{(0,0)\}\cup
\{(\mu,\sigma r\delta):
  \mu=1,\ldots,d;\ r=1,\ldots,K-1;\ \sigma\in\Sigma\}.
```

The `(0,0)` descriptor is the centered transported term. For
`convention=:favoni_arxiv`, only positive shifts are allowed, the centered
term is omitted, and

```math
\mathcal P_\ell=
\{(\mu,r(\delta+1)):
  \mu=1,\ldots,d;\ r=1,\ldots,K-1\}.
```

Write `a=1` when `adjoints=true` and `a=0` otherwise, and similarly write
`u=1` when `identity=true`. The local basis and transported basis are ordered
as

```math
\begin{aligned}
\mathcal B_x={}&
(W_{x,1},\ldots,W_{x,N_{\ell-1}},
  W^\dagger_{x,1},\ldots,W^\dagger_{x,N_{\ell-1}},
  \mathbf 1),\\
\widetilde{\mathcal B}_x={}&
(\widetilde W_{x,j,p},
  \widetilde W^\dagger_{x,j,p},
  \mathbf 1),
\end{aligned}
```

with the adjoint and identity parts omitted when their options are false. In
the transported basis, paths are ordered as returned by
`LCNN.displacements(layer, d)`, and the channel index `j` varies inside each
path.

The layer evaluates

```math
W^{(\ell)}_{x,i}
=\sum_{A=1}^{N_{\mathrm{local}}}
 \sum_{B=1}^{N_{\mathrm{transport}}}
 \theta^{(\ell)}_{iAB}\,
 \mathcal B_{x,A}\widetilde{\mathcal B}_{x,B},
\qquad i=1,\ldots,N_\ell,
```

where

```math
N_{\mathrm{local}}=(1+a)N_{\ell-1}+u,
\qquad
N_{\mathrm{transport}}=(1+a)N_{\ell-1}
|\mathcal P_\ell|+u.
```

The coefficients `\theta^{(\ell)}_{iAB}` are the trainable LCB parameters:

```julia
parameters.layers[ell].weight[i, A, B]
```

Consequently,

```julia
size(parameters.layers[ell].weight) ==
    (N_out, N_local, N_transport)
```

and the layer contains
`N_out * N_local * N_transport` independently stored real parameters. There
is no separate trainable L-Conv tensor in `LCNN.LCB`; the path label is already
part of the transported-basis index `B`.

The constructor is

```julia
LCNN.LCB(
    N_in => N_out;
    convention=:favoni_prl,
    kernel_size=2,
    dilation=nothing,
    shifts=nothing,
    identity=true,
    adjoints=true,
    init_weight_factor=1.0,
)
```

With `dilation=nothing`, `\delta` is 1 for `:favoni_prl` and 0 for
`:favoni_arxiv`. With `shifts=nothing`, the defaults are `:both` and
`:positive`, respectively. The constructor arguments map to the definition as
follows:

| Julia argument | symbol or role in the equations | trainable? |
|---|---|---|
| `LCB(N_in => N_out)` | `N_{\ell-1}` and `N_\ell` | no |
| `kernel_size=K` | path radii `r=1,\ldots,K-1`; default `K=2` | no |
| `dilation=delta` | path spacing `\delta`; convention-dependent default above | no |
| `shifts=:positive` or `:both` | sign set `\Sigma`; convention-dependent default above | no |
| `identity=true` | includes `\mathbf 1`, setting `u=1` | no |
| `adjoints=true` | includes daggered channels, setting `a=1` | no |
| `convention=:favoni_prl` | definition of the path set | no |
| `parameters.layers[ell].weight` | `\theta^{(\ell)}_{iAB}` | yes |
| `init_weight_factor=f` | initialization scale only | no |

`initial_parameters` samples each LCB weight with standard deviation

```math
\frac{f}{\sqrt{N_{\mathrm{local}}N_{\mathrm{transport}}}}.
```

Changing `init_weight_factor` changes the random initialization, not the
forward equation after parameters have been created.

### Stack of LCB layers

In

```julia
feature_model = LCNN.LCNNFeatureModel(U, N1, N2, ...; lcb_options...)
```

the integers `N1, N2, ...` are `N_1,N_2,\ldots`, the output widths of the
successive LCB layers. All layers receive the same `lcb_options`. Passing an
explicit tuple of `LCB` objects allows a different path set and basis in every
layer. The complete feature parameter tree is

```julia
(layers=(
    (weight=theta1,),
    (weight=theta2,),
    # ...
),)
```

The constructor arguments have the following roles:

| Julia argument | role | trainable? |
|---|---|---|
| `U` | supplies the lattice dimension `d` used to set `N_0` and validate the model | no |
| `N1, N2, ...` | layer widths `N_1,N_2,\ldots` | no |
| `input=Plaq()` | defines `W^{(0)}` | no |
| `lcb_options...` | common path set, bases, and initializer for every LCB | no |
| `parameters.layers[ell].weight` | coefficients `\theta^{(\ell)}_{iAB}` | yes |

### Scalar head: `LCNNAction`

Let the last feature layer have `N_L` channels. `component` defines a real
trace vector `z_x`:

```math
\begin{array}{ll}
\texttt{:real}: & z_x=(\operatorname{ReTr}W_{x,1},\ldots,
                         \operatorname{ReTr}W_{x,N_L}),\\
\texttt{:imag}: & z_x=(\operatorname{ImTr}W_{x,1},\ldots,
                         \operatorname{ImTr}W_{x,N_L}),\\
\texttt{:both}: & z_x=(\operatorname{ReTr}W_{x,1},
                         \operatorname{ImTr}W_{x,1},\ldots,
                         \operatorname{ReTr}W_{x,N_L},
                         \operatorname{ImTr}W_{x,N_L}).
\end{array}
```

The site-local prediction is

```math
\widehat y_x=b+\sum_c q_c z_{x,c},
```

where the trace is not divided by `N_c`, matching the reference
implementation. The exact parameter correspondence is

```julia
parameters.readout.weight[c] == q[c]
parameters.readout.bias[1]   == b
```

For `component=:real` or `:imag`, `q` has length `N_L`; for `:both` it has
length `2N_L`. `site_predictions(action, U, parameters)` returns
`\widehat y_x`. Calling `action(U, parameters)` first applies `reduction=:sum`
or `:mean` to each component of `z_x`, then applies the same `q` and adds `b`.
Neither `component` nor `reduction` is trainable.

| Julia argument or parameter | role | trainable? |
|---|---|---|
| `feature_model` | supplies the final channels `W^{(L)}` | no |
| `component` | defines the ordering and length of `z_x` | no |
| `reduction` | chooses the site sum or mean in the scalar call | no |
| `parameters.readout.weight` | coefficients `q_c` | yes |
| `parameters.readout.bias` | scalar bias `b` | yes |

If `C` is the length of `z_x`, `initial_parameters` samples `q_c` with
standard deviation `1/\sqrt{C}` and initializes `b=0`. The scalar head has
`C+1` parameters, so the complete action contains

```math
\sum_{\ell=1}^{L}
N_\ell N_{\mathrm{local},\ell}N_{\mathrm{transport},\ell}
+C+1
```

trainable real numbers.

### Link head: `LCNNLinkModel` and `LExp`

For a link model, the trainable LExp weights are

```julia
parameters.lexp.weight[mu, i] == beta[mu, i]
```

with shape `(d, N_L)`. The link update is

```math
A_{x,\mu}=\sum_{i=1}^{N_L}\beta_{\mu i}W^{(L)}_{x,i},
\qquad
U'_{x,\mu}=\exp\!\left([A_{x,\mu}]_{\mathrm{TA}}\right)U_{x,\mu},
```

where `[\cdot]_{\mathrm{TA}}` is Gaugefields' traceless anti-Hermitian
projection. This is the L-Exp operation in Eqs. (8)--(9) of Favoni et al.
`LExp` has `dN_L` real trainable parameters and no bias.
`LExp(channels; init_weight_factor=f)` initializes `\beta` with standard
deviation `f/\sqrt{N_L}`; `f` is not itself trainable.

| Julia argument or parameter | role | trainable? |
|---|---|---|
| `feature_model` | supplies `W^{(L)}` and `N_L` | no |
| `LExp(N_L; init_weight_factor=f)` | selects the input width and initializer scale | no |
| `parameters.lexp.weight[mu, i]` | coefficient `\beta_{\mu i}` | yes |

The complete link model therefore contains

```math
\sum_{\ell=1}^{L}
N_\ell N_{\mathrm{local},\ell}N_{\mathrm{transport},\ell}
+dN_L
```

trainable real numbers.

The resulting parameter trees are therefore

```text
LCNNAction:    (layers=(...), readout=(weight=..., bias=...))
LCNNLinkModel: (layers=(...), lexp=(weight=...))
```

## Build and run a first scalar model

The following example constructs a two-layer network on an `8 x 8` SU(2)
configuration:

~~~julia
import JACC
JACC.@init_backend

using Gaugefields, Random
const LCNN = Gaugefields.LCNN

U = gauge_configuration(
    (8, 8);
    colors=2,
    start=:hot,
    seed=0x1234,
)

# Plaq(1 channel) -> LCB(1 => 4) -> LCB(4 => 2)
feature_model = LCNN.LCNNFeatureModel(
    U, 4, 2;
    kernel_size=2,
    shifts=:both,
    identity=true,
    adjoints=true,
)

model = LCNN.LCNNAction(
    feature_model; component=:both, reduction=:mean,
)
parameters = LCNN.initial_parameters(
    MersenneTwister(1234), model, Float32,
)

# Gauge-invariant prediction at every lattice site.
local_prediction = LCNN.site_predictions(model, U, parameters)

# Mean of the site predictions.
scalar_prediction = model(U, parameters)
~~~

The dimensions of `U` determine the plaquette input width. The same
constructor applied to a four-dimensional configuration begins with six
plaquette channels. The integers `4, 2` are the output widths of the two LCB
layers.

Parameters are ordinary nested `NamedTuple`s containing real arrays:

~~~julia
LCNN.parameter_shapes(model)
# (layers=((weight=(4, 3, 11),), (weight=(2, 9, 41),)),
#  readout=(weight=(4,), bias=(1,)))

LCNN.parameter_count(model)
# 875

LCNN.validate_parameters(model, parameters)

parameters.layers[1].weight
parameters.layers[2].weight
parameters.readout.weight
parameters.readout.bias
~~~

The model object describes the architecture; all fitted values live in the
parameter tree.

## Define the feature network

An LCNN is specified by three parts:

1. a plaquette input and a sequential tuple of `LCB` feature layers;
2. either a scalar `Trace` readout or an `LExp` link-update head;
3. an ordinary nested `NamedTuple` containing every trainable parameter.

### Choose the depth and the settings of every layer

The compact constructor accepts any number of output widths and applies the
same LCB options to each layer:

~~~julia
feature_model = LCNN.LCNNFeatureModel(
    U,
    8, 8, 4;                 # three LCBs with these output widths
    kernel_size=2,
    dilation=1,
    shifts=:both,
    identity=true,
    adjoints=true,
    init_weight_factor=1.0,
)
~~~

To give the layers different path ranges or channel bases, construct the tuple
explicitly. The plaquette input has `binomial(length(U), 2)` channels:

~~~julia
nplaq = binomial(length(U), 2)
layers = (
    LCNN.LCB(
        nplaq => 8;
        kernel_size=2,
        dilation=1,
        shifts=:both,
        identity=true,
        adjoints=true,
    ),
    LCNN.LCB(
        8 => 4;
        kernel_size=3,
        dilation=2,
        shifts=:positive,
        identity=true,
        adjoints=false,
    ),
    LCNN.LCB(
        4 => 2;
        kernel_size=2,
        dilation=1,
        shifts=:both,
        identity=false,
        adjoints=true,
    ),
)
feature_model = LCNN.LCNNFeatureModel(U, layers)
~~~

Each layer's input width must equal the preceding stage's output width. In the
example above the second LCB therefore accepts the eight channels produced by
the first.

The exact mapping from every LCB option to its path set, bases, and weight
tensor is given under [One LCB layer](#one-lcb-layer).

`LCNN.displacements(layer, length(U))` returns the ordered `(direction, step)`
descriptors used by the transported basis, while
`LCNN.parameter_shape(layer, length(U))` returns the corresponding
`(N_out, N_local, N_transport)` weight shape.

`LCNNFeatureModel` represents a sequential `Plaq -> LCB -> ...` backbone. Its
depth, widths, and per-layer LCB options are freely selectable within that
structure.

## Attach an output head

For a gauge-invariant scalar or site-local regression model, attach
`LCNNAction`. `component` chooses the real and/or imaginary trace components;
`reduction` controls the scalar call. `site_predictions` always retains the
site axis for a site-local loss:

~~~julia
action = LCNN.LCNNAction(
    feature_model;
    component=:both,         # :real, :imag, or :both
    reduction=:mean,         # :sum or :mean
)
parameters = LCNN.initial_parameters(
    MersenneTwister(7), action, Float32,
)

local_values = LCNN.site_predictions(action, U, parameters)
scalar_value = action(U, parameters)
~~~

For a learned link transformation, attach `LExp` instead:

~~~julia
link_model = LCNN.LCNNLinkModel(
    feature_model,
    LCNN.LExp(
        LCNN.output_channels(feature_model);
        init_weight_factor=0.01,
    ),
)
link_parameters = LCNN.initial_parameters(
    MersenneTwister(8), link_model, Float32,
)

Unew = LCNN.forward_links(link_model, U, link_parameters)
Unew_via_smearing = smear(U, lcnn_smearing(link_model, link_parameters))
~~~

The two complete parameter trees have different heads: `readout` for an
`LCNNAction` and `lexp` for an `LCNNLinkModel`.

## Inspect or specify every parameter

`parameter_shapes` gives the schema of the complete parameter tree, and
`validate_parameters` checks its names, shapes, and real element types:

~~~julia
shapes = LCNN.parameter_shapes(action)
# (layers=((weight=...), ...), readout=(weight=..., bias=(1,)))

rng = MersenneTwister(19)
manual_parameters = (
    layers=map(shapes.layers) do layer_shape
        (
            weight=0.01f0 .* randn(
                rng, Float32, layer_shape.weight...,
            ),
        )
    end,
    readout=(
        weight=randn(rng, Float32, shapes.readout.weight...),
        bias=zeros(Float32, shapes.readout.bias...),
    ),
)
LCNN.validate_parameters(action, manual_parameters)
~~~

For a link model the schema is
`(layers=(...), lexp=(weight=(dimension, final_channels),))`. Array leaves can
be edited directly, for example
`parameters.layers[2].weight .= new_weight` or
`link_parameters.lexp.weight .= beta`.

## Use the same model through LuxCore

Gaugefields depends on the lightweight `LuxCore` interface. The architecture
and parameter tree above can be wrapped as a Lux layer without changing their
meaning:

~~~julia
using LuxCore

lux_model = LCNN.lux_layer(action; parameter_type=Float32)
lux_parameters, state = LuxCore.setup(MersenneTwister(20), lux_model)
workspace = LCNN.ModelWorkspace(action, U)
value, state = lux_model((U, workspace), lux_parameters, state)
~~~

`ModelWorkspace(model, U)` preallocates the lattice-sized intermediates for
repeated evaluation and differentiation. The direct API also accepts this
workspace through `forward_features!`, `forward_links!`, and model calls.

## Features and link smearing are different outputs

### Inspect site-covariant features

The `Plaq -> LCB -> ...` backbone produces site fields transforming as

```math
W_{x,i} \longmapsto \Omega_x W_{x,i}\Omega_x^\dagger.
```

Evaluate this intermediate representation with `forward_features`:

~~~julia
feature_parameters = (layers=parameters.layers,)
features = LCNN.forward_features(
    feature_model, U, feature_parameters,
)
~~~

This feature map has not changed the gauge links and is therefore not a
smearing. It is intentionally rejected by the Gaugefields `smear` API, whose
result must always be another gauge-link configuration.

### Update gauge links with LExp

`LExp` converts the feature channels into one traceless anti-Hermitian
generator per link direction and performs

```math
A_{x,\mu}=\sum_i\beta_{\mu i}W_{x,i},\qquad
U'_{x,\mu}=\exp([A_{x,\mu}]_{\mathrm{TA}})U_{x,\mu}.
```

Because the exponential transforms at site `x`, `Unew` has the ordinary link
transformation law. `LCNNLinkModel(feature_model)` adds an `LExp` head with
the correct number of channels automatically:

~~~julia
link_model = LCNN.LCNNLinkModel(feature_model)
link_parameters = LCNN.initial_parameters(
    MersenneTwister(9), link_model, Float32,
)

lcnn = lcnn_smearing(link_model, link_parameters)
Unew = smear(U, lcnn)

# A conventional smearing has the same link-to-link call shape.
stout = stout_link_smearing(U; rho=0.1)
Ustout = smear(U, stout)
~~~

Both results are vectors of SU(N) gauge links. A feature model or scalar
`LCNNAction` cannot be passed to `lcnn_smearing`; an explicit
`LCNNLinkModel` ending in `LExp` is required.

The recording form provides reusable storage just like other smearings:

~~~julia
recorded = smear(U, lcnn; record=true)

Unew = recorded.configuration
cache = recorded.history
@assert recorded.derivative === nothing

# Reuse feature, Lie-algebra, exponential, and output link fields.
Unew = smear(U, lcnn; temps=cache)
~~~

Reusing a cache overwrites its previously returned output configuration.

## Differentiation with Enzyme

Loading Enzyme activates parameter and link differentiation.

### Differentiate a scalar model

A scalar model can be differentiated with respect to both its gauge links and
all trainable parameters:

~~~julia
using Enzyme

workspace = LCNN.ModelWorkspace(model, U)
dS_dU = LCNN.dSdu(model, U, parameters; workspace)
dS_dparameters = LCNN.parameter_gradient(
    model, U, parameters; workspace,
)

# Allocation-reusing form
dworkspace = Enzyme.make_zero(workspace)
LCNN.dSdu!(dS_dU, model, U, parameters, workspace, dworkspace)
~~~

The result is the ambient complex-matrix derivative satisfying
`delta S = real(dot(dS_dU, delta U))`. HMC-specific traceless
anti-Hermitian projection remains a separate adapter step.

`parameter_gradient` returns the same nested `NamedTuple` shape as
`parameters`. Custom Enzyme rules stop at the JACC/LatticeMatrices plaquette,
parallel-transport, bilinear, and trace kernels. The user-selected tuple of
LCB layers remains normal differentiable Julia composition, so changing layer
count does not require a new adjoint.

### Differentiate a link model

The derivative of a link-valued output is represented as a
vector--Jacobian product rather than a materialized Jacobian:

~~~julia
recorded = smear(U, lcnn; record=true, calcdSdU=true)

# dUnew has the same link-field layout as recorded.configuration.
dU = recorded.derivative(dUnew)

# Request both the input-link and every parameter cotangent.
pullback = LCNN.link_model_pullback(dUnew, U, recorded.history)
dU = pullback.links
dparameters = pullback.parameters
~~~

The LExp exponential Fréchet pullback is a custom Enzyme boundary backed by
the LatticeMatrices JACC kernel. The public VJP walks the user-selected LCB
tuple in reverse and therefore does not fix the model depth.

## Reference model from Favoni et al.

The cited paper studies supervised regression of local gauge-invariant observables
in pure SU(2) Yang--Mills configurations. Its Wilson-loop datasets contain
plaquettes as input features and site-local real traces of 1 by 1, 1 by 2,
2 by 2, and 4 by 4 Wilson loops as targets. The experiments cover 1+1D and
3+1D lattices.

### The small 1 by 2 Wilson-loop experiment

The released `D2_W1x2_lcnn_s` experiment has the following architecture:

- 1+1D SU(2) data on the training lattice `8 x 8`;
- target `trW_1x2`, evaluated separately at every site;
- positive-orientation 1 by 1 plaquette input, one channel in 2D;
- one `LCB` with kernel size 2 and channel width `1 => 2`;
- complex trace split as `re(W1), im(W1), re(W2), im(W2)`;
- one real `Linear(4 => 1)` site-local readout;
- no global average in the training loss.

For exactly this specification, the whole architecture can be constructed
with the reproduction convenience function:

~~~julia
model = LCNN.favoni2022_wilson_1x2_small(
    U; convention=:favoni_prl,
)
parameters = LCNN.initial_parameters(
    MersenneTwister(1234), model, Float32,
)

local_prediction = LCNN.site_predictions(model, U, parameters)
scalar_prediction = model(U, parameters)
~~~

This is only a paper-reproduction preset; ordinary LCNNs should be assembled
with `LCNNFeatureModel` and the desired output head as shown above. The preset
uses `reduction=:mean` for its scalar call. That final mean is not inserted
into the site-local training path: `site_predictions` and `site_mse_loss`
retain one prediction per site, matching the reference model.

### Two released layer conventions

The arXiv table and the later released result files use different transported
bases:

~~~julia
arxiv_model = LCNN.favoni2022_wilson_1x2_small(
    U; convention=:favoni_arxiv,
) # 35 parameters

prl_model = LCNN.favoni2022_wilson_1x2_small(
    U; convention=:favoni_prl,
) # 47 parameters; the default
~~~

| convention | upstream source | transported terms | parameters |
|---|---|---:|---:|
| `favoni_arxiv` | tag `arxiv_v1`, Supplementary Table V counting | positive nonzero shifts, no centered term | 35 |
| `favoni_prl` | tag `prl_2022`, released result pickle | centered term plus positive shifts | 47 |

These counts follow directly from the LCB shape above. In two dimensions,
`N_in=1`, `N_out=2`, `kernel_size=2`, `identity=true`, and `adjoints=true`, so
`N_local=3`. The arXiv basis has two path descriptors and
`N_transport=2*1*2+1=5`, giving `2*3*5=30` LCB weights. The four trace weights
and one bias bring the total to 35. The released-PRL basis adds the centered
descriptor, so `N_transport=2*1*3+1=7`: its total is `2*3*7+4+1=47`.

The convention also fixes dilation semantics and basis order; it is not merely
a parameter-count switch. New code should use these explicit names. Older NPZ
metadata names remain accepted only when reading existing files.

## Training on the paper HDF5 layout

Training uses the optional `HDF5`, `Enzyme`, and `Optimisers` dependencies; no
separate LCNN training package is involved.

~~~julia
pkg> add Enzyme HDF5 Optimisers
~~~

~~~julia
using Enzyme, Gaugefields, HDF5, Optimisers, Random
const LCNN = Gaugefields.LCNN

train = LCNN.read_favoni2022_dataset(
    "datasets/D2_8/train.hdf5"; target="trW_1x2",
)
validation = LCNN.read_favoni2022_dataset(
    "datasets/D2_8/val.hdf5"; target="trW_1x2",
)
test = LCNN.read_favoni2022_dataset(
    "datasets/D2_8/test.hdf5"; target="trW_1x2",
)

model = LCNN.favoni2022_wilson_1x2_small(
    ; convention=:favoni_prl,
)
parameters = LCNN.initial_parameters(
    MersenneTwister(1234), model, Float32,
)
config = LCNN.TrainingConfig(
    max_epochs=20,
    batch_size=50,
    learning_rate=3f-3,
    weight_decay=0f0,
    amsgrad=true,
    patience=5,
    min_delta=0f0,
    seed=1234,
)
result = LCNN.fit!(model, parameters, train, validation; config)

site_test_mse = LCNN.evaluate_dataset(model, result.parameters, test)
reported_test_mse = LCNN.evaluate_dataset(
    model, result.parameters, test; global_average=true,
)
~~~

The loader understands the released Python layout
`u[sample, flattened_site, direction, row, column]`. It expands the C-order
site index using `dims`, keeps the real component of the requested target,
and ignores the stored `w`: plaquettes are recomputed from `u` by `Plaq`.
A subset such as `samples=1:100` is useful for a short smoke run.

The released 2D datasets use ten beta values from 0.1 through 6.0, with 1000
training and 100 validation/test configurations per beta on an 8 by 8
lattice. Their observable-generation step adds `trW_1x2` to the HDF5 files.

Training minimizes MSE over all sites. Since `LatticeMatrix` currently has no
batch axis, gradients are accumulated per configuration and averaged before
one `Optimisers.update!`. Validation uses `min_delta`, stops after `patience`
non-improving epochs, and restores the best parameter copy.

`Optimisers.AdamW` is used when `amsgrad=false`. For `amsgrad=true`,
`LCNN.adamw_amsgrad` implements the PyTorch update including bias correction,
the maximum second moment, and decoupled weight decay.

The complete runnable entry point is
`examples/lcnn/favoni2022_hdf5_training.jl`; set `LCNN_MAX_SAMPLES=100` for a
short run.

## GPU and MPI smoke tests

`test/lcnn_backend_smoke.jl` is an opt-in backend test covering the released
2D SU(2) model and a compact 4D SU(3) model. It checks forward evaluation,
scalar parameter gradients, scalar `dS/dU`, and LExp link/parameter VJPs. An
MPI run additionally requires all ranks to agree. A CUDA run verifies that
lattice storage is a `CuArray` while scalar parameters remain ordinary Julia
arrays.

After selecting the JACC backend in a test environment, a single-GPU run is:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  julia --project=/path/to/cuda-test-env test/lcnn_backend_smoke.jl
~~~

Two CPU MPI ranks use:

~~~sh
LCNN_USE_MPI=true LCNN_TEST_CASES=2d \
  mpiexec -n 2 julia --project=/path/to/mpi-test-env \
  test/lcnn_backend_smoke.jl
~~~

For two NVIDIA GPUs, the MPI library selected by MPI.jl must match the launcher
and support CUDA buffers:

~~~sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
LCNN_USE_MPI=true LCNN_TEST_CASES=2d \
  mpiexec -n 2 julia --project=/path/to/cuda-mpi-test-env \
  test/lcnn_backend_smoke.jl
~~~

Set `LCNN_TEST_PARAMETER_GRADIENT=false` and
`LCNN_TEST_LINK_GRADIENT=false` and
`LCNN_TEST_LINK_MODEL_PULLBACK=false` for a forward-only communication check.
Set `LCNN_TEST_LINK_MODEL_PARAMETERS=false` to isolate the link-only LExp VJP.
`LCNN_TEST_CASES=2d`, `4d`, or `2d,4d` controls the dimensional cases. These
tests are not part of ordinary CI because accelerator availability, MPI
launchers, and first Enzyme compilation are environment-dependent.

## Remaining layer work

The current implementation includes `Plaq`, configurable `LCB` stacks,
`TraceReadout`, and an `LExp` link-update head with reusable storage. The
complete layer family still requires:

- `Poly` for non-contractible Polyakov-loop input channels;
- standalone `LConv` and `LBilin`;
- `LAct` for gauge-invariant scalar activation.

The current `LExp` path supplies true learned link smearing and a custom
Enzyme/JACC pullback for links and parameters. Remaining composition work is
automatic `Plaq`/`Poly` refresh when several learned link updates are chained.

## Optional: PyTorch checkpoint interoperability

Everything below is optional. Normal model construction, evaluation,
smearing, differentiation, and Julia training do not require Python.

### Load a portable checkpoint in Julia

The convenience loader constructs the architecture recorded in an NPZ file
and returns framework-independent parameters:

~~~julia
model, parameters, metadata = LCNN.load_pytorch_model(
    U,
    "D2_W1x2_lcnn_s_model1.npz";
    parameter_type=Float32,
    strict=true,
)

prediction = model(U, parameters)
~~~

For an already constructed model, load only its parameters:

~~~julia
checkpoint = LCNN.read_pytorch_checkpoint("model.npz")
parameters = LCNN.load_pytorch_parameters(
    model, checkpoint; parameter_type=Float32, strict=true,
)
~~~

Strict loading checks the format version, dimension, color count, convention,
layer count, widths, kernel sizes, dilations, shifts, unit/adjoint channels,
complex-component order, parameter shapes, and finite values. Gaugefields
does not guess a convention or pad a 35-parameter model to 47 parameters.

### Export from an existing PyTorch model

Run the exporter in the Python environment that owns the trusted model:

~~~python
import sys
sys.path.insert(0, "examples/lcnn")

from export_lge_checkpoint import export_lge_checkpoint

export_lge_checkpoint(
    model.state_dict(),
    "D2_W1x2_small.npz",
    convention="favoni_prl",
    dimension=2,
    colors=2,
    conv_channels=[2],
    kernel_size=[2],
    dilation=[1],
    symmetric=False,
    use_unit_elements=True,
    global_average=False,
    linear_sizes=[],
    source_commit="prl_2022",
)
~~~

The command-line form is:

~~~sh
python examples/lcnn/export_lge_checkpoint.py model.ckpt model.npz \
    --convention favoni_prl \
    --dimension 2 \
    --colors 2 \
    --conv-channels 2 \
    --kernel-size 2 \
    --dilation 1 \
    --source-commit prl_2022
~~~

`torch.load` and Python pickle can execute code. Only load a checkpoint from a
trusted source. Gaugefields consumes the resulting numeric NPZ without Python.

The published `D2_W1x2_lcnn_s_results.pickle` uses legacy CUDA storage. Convert
one stored ensemble member with:

~~~sh
python examples/lcnn/extract_favoni_checkpoint.py \
    D2_W1x2_lcnn_s_results.pickle \
    D2_W1x2_lcnn_s_model1.npz \
    --model 0
~~~

The converter reconstructs numeric tensors without requiring PyTorch or CUDA,
but its pickle input must still be trusted.

### Optional live comparison with the official implementation

Set the Python executable and `layers.py` from the matching upstream tag:

~~~sh
export LCNN_TORCH_PYTHON=/path/to/python
export LCNN_UPSTREAM_LAYERS=/path/to/lge-cnn/lge_cnn/nn/layers.py

julia --project examples/lcnn/favoni2022_checkpoint_parity.jl \
    favoni_prl D2_W1x2_lcnn_s_results.pickle
~~~

The script evaluates two independent SU(2) configurations and compares every
complex matrix at every site before tracing, followed by the final scalar.
Set `LCNN_TORCH_DTYPE=float32` for publication precision; Float64 is the
default stringent formula check.

The short training comparison starts PyTorch and Julia from identical weights
and performs three full-batch SGD steps on one 4 by 4 configuration:

~~~sh
julia --project examples/lcnn/favoni2022_training_parity.jl favoni_prl
~~~

An optional step count and learning rate may follow, for example
`favoni_prl 2 1e-3`.

Live PyTorch is never run in Gaugefields CI. The ordinary test suite instead
compares Julia with frozen official-PyTorch outputs in
`test/data/lcnn/favoni2022_pytorch_outputs.npz`. The fixture contains the full
complex pre-trace field and final scalar for both conventions on two
independent gauge configurations. Provenance and a manual regeneration command
are recorded in `test/data/lcnn/README.md`.
