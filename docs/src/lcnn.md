# L-CNN

`Gaugefields.LCNN` provides lattice-gauge-equivariant feature maps, scalar
models, and learned link smearings for 2D, 3D, and 4D
`LatticeMatrices`-backed gauge fields. The ordinary Julia API does not require
Python or PyTorch. `LuxCore` is a lightweight direct dependency; full Lux is
not required to construct or evaluate a model.

## What an LCNN computes

Gauge links and site features transform differently:

```math
U_{x,\mu}\longmapsto
\Omega_xU_{x,\mu}\Omega^\dagger_{x+\hat\mu},\qquad
W_{x,i}\longmapsto\Omega_xW_{x,i}\Omega_x^\dagger.
```

An LCNN transports neighboring features to a common site before combining
them. In this package a model is assembled as

```text
U -> Plaq -> LCB -> ... -> LCB -> site-covariant features
                                      |-> Trace -> real scalar
                                      `-> LExp  -> updated gauge links
```

`Plaq` creates the initial matrix-valued site features. Each `LCB` performs a
gauge-equivariant convolution and bilinear channel mixing. The final head
determines whether the network is a gauge-invariant scalar model or a genuine
link smearing.

## Build and run a first model

The following example constructs a small network directly from its layer
widths. It does not use a paper-specific preset:

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

# Plaq -> LCB(1 => 4) -> LCB(4 => 2) in two dimensions.
feature_model = LCNN.LCNNFeatureModel(
    U, 4, 2;
    kernel_size=2,
    shifts=:both,
    identity=true,
    adjoints=true,
)

# Trace the two final channels and learn a real linear readout.
model = LCNN.LCNNAction(
    feature_model; component=:both, reduction=:mean,
)
parameters = LCNN.initial_parameters(
    MersenneTwister(1234), model, Float32,
)

# One real prediction at every lattice site.
local_prediction = LCNN.site_predictions(model, U, parameters)

# The same predictions reduced to one scalar by the selected mean reduction.
scalar_prediction = model(U, parameters)
~~~

`U` determines the dimension, color count, lattice shape, and backend, so
users do not pass `Val(2)` or `Val(4)`. The same construction works for SU(N).
For example, a 4D field automatically gives the first `LCB` six plaquette
input channels instead of one.

Parameters are ordinary nested `NamedTuple`s containing real arrays:

~~~julia
LCNN.parameter_shapes(model)
LCNN.parameter_count(model)
LCNN.validate_parameters(model, parameters)

parameters.layers[1].weight
parameters.layers[2].weight
parameters.readout.weight
parameters.readout.bias
~~~

Every trainable scalar is visible and may be initialized, loaded, or replaced
without an ML framework.

## Define your own network

An LCNN is specified in three independent parts:

1. a plaquette input and a sequential tuple of `LCB` feature layers;
2. either a scalar `Trace` readout or an `LExp` link-update head;
3. an ordinary nested `NamedTuple` containing every trainable parameter.

### Choose the depth and the settings of every layer

The short constructor accepts any number of channel widths. The same keyword
settings are then used by every `LCB`:

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

To configure the layers independently, construct the tuple explicitly. The
number of plaquette input channels is `binomial(length(U), 2)`: one in 2D and
six in 4D. Thus this same code works for both dimensions and for any SU(N):

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

The constructor checks every channel boundary immediately. For example, the
second layer above must accept the eight channels produced by the first.

The `LCB` keywords have the following meanings. `convention` exists primarily
for exact upstream compatibility; most new models can leave its default
unchanged and consult [Two released layer conventions](#two-released-layer-conventions)
only when reproducing those results.

| keyword | choices | effect |
|---|---|---|
| `convention` | `:favoni_prl`, `:favoni_arxiv` | Selects the released centered-term/dilation convention or the arXiv-table convention. |
| `kernel_size` | integer at least 2 | Uses nonzero path distances `1:(kernel_size - 1)`. |
| `dilation` | positive integer for `:favoni_prl` | Sets the spacing between path distances. The arXiv convention uses its original zero-based dilation. |
| `shifts` | `:positive`, `:both` | Uses positive axes only or positive and negative axes. `:favoni_arxiv` requires `:positive`. |
| `identity` | `true`, `false` | Includes or removes the identity from the bilinear basis. |
| `adjoints` | `true`, `false` | Includes or removes Hermitian-conjugate feature channels. |
| `init_weight_factor` | finite real | Scales only the random initializer; it does not alter evaluation. |

This API currently describes a sequential `Plaq -> LCB -> ...` backbone.
Arbitrary depth, widths, and per-layer options are supported, but arbitrary
Lux layers, skip connections, and branching DAGs are not silently accepted as
LCNN layers. Standalone `LConv`, `LBilin`, `LAct`, and `Poly` composition is
listed under [Remaining layer work](#remaining-layer-work).

### Choose the output head

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

For a genuine learned smearing, attach `LExp` instead. This produces links,
not a scalar:

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

The backbone may be shared conceptually, but the two complete parameter trees
have different heads: `readout` for `LCNNAction` and `lexp` for
`LCNNLinkModel`.

### Inspect, replace, or specify every parameter

No parameter is hidden in the model object. `parameter_shapes` is the schema
for the complete parameter tree, and `validate_parameters` checks names,
shapes, and real element types:

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

For a link model, the analogous schema is
`(layers=(...), lexp=(weight=(Dim, final_channels),))`. Array leaves can also
be edited directly, for example
`parameters.layers[2].weight .= new_weight` or
`link_parameters.lexp.weight .= beta`.

### Use the same model through LuxCore

The architecture is defined before it is wrapped, so the LuxCore adapter does
not restrict the selected layer tuple:

~~~julia
using LuxCore

lux_model = LCNN.lux_layer(action; parameter_type=Float32)
lux_parameters, state = LuxCore.setup(MersenneTwister(20), lux_model)
workspace = LCNN.ModelWorkspace(action, U)
value, state = lux_model((U, workspace), lux_parameters, state)
~~~

`ModelWorkspace(model, U)` preallocates every lattice-sized intermediate.
Use `forward_features!`, `forward_links!`, or pass the workspace to the model
when repeatedly evaluating the same architecture.

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

## Reproduce the Favoni et al. model

The L-CNN construction implemented here is based on M. Favoni, A. Ipp,
D. I. Müller, and D. Schuh, “Lattice Gauge Equivariant Convolutional Neural
Networks,” *Physical Review Letters* **128**, 032003 (2022),
[doi:10.1103/PhysRevLett.128.032003](https://doi.org/10.1103/PhysRevLett.128.032003),
[arXiv:2012.12901](https://arxiv.org/abs/2012.12901).

The paper studies supervised regression of local gauge-invariant observables
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
| `favoni_arxiv` | tag `arxiv_v1`, supplemental Table V counting | positive nonzero shifts, no centered term | 35 |
| `favoni_prl` | tag `prl_2022`, released result pickle | centered term plus positive shifts | 47 |

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
