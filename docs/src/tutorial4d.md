# Four-dimensional quick start

This tutorial uses the recommended Gaugefields v1 API. A four-dimensional
configuration is a vector of four link fields backed by LatticeMatrices:

```math
U = [U_x, U_y, U_z, U_t].
```

The same Julia simulation code can run on CPU threads, one GPU, MPI ranks, or
multiple GPUs. The execution backend is selected by JACC outside the
simulation code.

## Select a backend

Add both packages directly to the application environment before importing
JACC:

```julia
pkg> add Gaugefields JACC
```

JACC uses the `"threads"` backend by default. To record a backend explicitly
in the active Julia environment, run one of the following commands once:

```julia
import JACC

JACC.set_backend("threads") # CPU
# JACC.set_backend("cuda")   # NVIDIA GPU
# JACC.set_backend("amdgpu") # AMD GPU
# JACC.set_backend("oneapi") # Intel GPU
```

Restart Julia after changing the backend. A simulation script should always
initialize the selected backend at top level, before loading Gaugefields:

```julia
import JACC
JACC.@init_backend

using Gaugefields
```

Keeping these lines in CPU scripts too makes the same script portable to a GPU
backend later.

## Create a configuration

The first positional argument specifies the global lattice size in
`(NX, NY, NZ, NT)` order:

```julia
U = gauge_configuration(
    (16, 16, 16, 32);
    colors=3,
    start=:hot,
    seed=1234,
)
```

The result keeps the existing vector representation:

```julia
@assert length(U) == 4
@assert gauge_lattice_size(U) == (16, 16, 16, 32)
@assert gauge_num_colors(U) == 3
@assert gauge_halo_width(U) == 1
@assert size(U[1]) == (3, 3, 16, 16, 16, 32)
```

`U[1]`, `U[2]`, `U[3]`, and `U[4]` are the links in the x, y, z, and t
directions, respectively. `size(U[mu])` is the logical global size. Programs
should use the public gauge-field operations instead of accessing the backend
array inside a link directly; the latter contains local MPI storage and halo
cells and is an implementation detail.

The default backend is `LatticeMatricesBackend()`. The historical
`Initialize_Gaugefields` entry point retains its legacy default independently
of this API.

## Configuration parameters

`gauge_configuration(lattice; ...)` accepts the following keywords:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `backend` | `LatticeMatricesBackend()` | Storage and execution implementation. Use `LegacyBackend()` only for compatibility. |
| `colors` | `3` | Number of colors, ``N_c``. |
| `halo` | `1` | Halo width in lattice sites. |
| `start` | `:cold` | Initial field, either `:cold` or `:hot`. |
| `seed` | `nothing` | Global-site seed for a reproducible LM hot start. |
| `process_grid` | `nothing` | MPI process grid. In 4D, automatic decomposition is `(1, 1, 1, nranks)`. |
| `boundary` | `:periodic` | `:periodic` or four boundary phases. |
| `eltype` | `ComplexF64` | LM element type: `Float32`, `Float64`, `ComplexF32`, or `ComplexF64`. |
| `rng` | `Philox4x32()` | Site-local RNG algorithm used for reproducible fields. |
| `verbose` | `0` | Diagnostic output level. |

The lattice can also be an integer vector, but a tuple makes its dimensionality
especially clear. With a fixed `seed`, an LM hot start is reproducible across
repeated runs and MPI process-grid decompositions.

The configuration seed controls only the hot start. It does not implicitly
seed later heatbath updates or HMC random choices. See
[Randomness and reproducibility](randomness.md) for the separate random
streams and the state needed for restartable simulations.

## Measure the field

The high-level measurement functions return normalized observables. A cold
configuration gives one for both examples below:

```julia
plaq = measure_plaquette(U)
poly = measure_polyakov_loop(U)

println("plaquette = ", plaq)
println("temporal Polyakov loop = ", poly)
```

In four dimensions, `measure_polyakov_loop` uses the final, t direction.
Pass `normalize=false` to obtain the historical unnormalized convention.

## Molecular-dynamics trajectories

Gaugefields provides deterministic molecular-dynamics updates as a reusable,
preallocated driver. First construct the gauge action and momenta:

```julia
beta = 6.0
action = GaugeAction(U)
plaquettes = make_loops_fromname("plaquette", Dim=4)
append!(plaquettes, plaquettes')
push!(action, beta / 2, plaquettes)

p = gaussian_momenta(
    U;
    seed=3000,
    sweep=0,
    rng=Philox4x32(),
)
```

Then construct and run the driver:

```julia
md = md_driver(
    U,
    action;
    steps=20,
    trajectory_length=1.0,
    integrator=QPQ(),
)

result = md_trajectory!(U, p, md)
println("delta H = ", result.delta_hamiltonian)
```

Both `U` and `p` are updated in place. `QPQ()` is the default and applies
`Q(1/2) P(1) Q(1/2)` per step; `PQP()` selects the other standard leapfrog
ordering. The same driver works without source changes on CPU, GPU, MPI, and
multi-GPU LM configurations.

### Defining an integrator

For a short experiment, define one MD step as an ordinary function. The first
argument of each elementary update is the object changed by that call:

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

To give a method a name and parameters, define an ordinary Julia type and its
one-step update. It does not have to subtype a Gaugefields type:

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

md = md_driver(
    U,
    action;
    steps=20,
    trajectory_length=1.0,
    integrator=OmelyanPQP(0.1931833275037836),
)
```

`md_step!` describes one step and Gaugefields calls it `steps` times. Negative
coefficients are accepted. For HMC, use a palindromic sequence when time
reversibility is required. The integrator is independent of the force
implementation, so the same definition also works with an Enzyme action. See
[High-level API parameters](highlevelapi.md#Custom-integrators) for the exact
function contracts and the lower-level action-provider interface.

The MD driver itself consumes no random numbers. It does not refresh momenta,
perform a Metropolis accept/reject decision, or restore a rejected field.
Those HMC policies belong to the application or a higher-level package. See
[High-level API parameters](highlevelapi.md) for custom integrators and
[Randomness and reproducibility](randomness.md) for HMC stream ownership.

## Heatbath updates

Construct an updater once and reuse its buffers:

```julia
updater = heatbath_updater(U; beta=6.0, seed=5678)

for sweep in 1:10
    heatbath!(U, updater)
    println(sweep, " ", measure_plaquette(U))
end
```

The same `U` and updater interface is used on threads, a GPU, and distributed
LM fields. The updater starts with `sweep=0` by default and increments that
counter after every successful call. Reconstructing an updater in the middle
of a run therefore requires both its `seed` and current `sweep` value.

## Gradient flow and stout smearing

Gradient flow updates its input configuration:

```julia
flow = gradient_flow(U; steps=10, step_size=0.01)
flow!(U, flow)
```

Stout smearing returns a new configuration by default:

```julia
stout = stout_smearing(U; loops=:plaquette, rho=0.1)
Ustout = smear(U, stout)
```

## Save and load

JLD2 is the default portable checkpoint format:

```julia
save_configuration("configuration.jld2", U)
Uloaded = load_configuration("configuration.jld2")
```

For MPI, GPU, and multi-GPU fields, every rank calls `save_configuration`.
Rank 0 gathers the physical links into one global host configuration and is
the only rank that writes the file. `load_configuration` allocates on the
current JACC backend, while `load_configuration!` redistributes into an
existing destination. The file does not contain a communicator, process grid,
device array, or halo storage.

Bridge and ILDG data can be loaded into a configuration whose lattice and
color sizes have already been specified; see the I/O API for those formats.

## MPI execution on CPUs

The following complete script makes the process grid explicit. Save it as
`four_d_mpi.jl`:

```julia
using MPI

import JACC
JACC.@init_backend

using Gaugefields

MPI.Init()
comm = MPI.COMM_WORLD
nranks = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# The global t extent remains divisible by the number of ranks.
lattice = (16, 16, 16, 8 * nranks)
grid = (1, 1, 1, nranks)

U = gauge_configuration(
    lattice;
    colors=3,
    halo=1,
    start=:hot,
    seed=1234,
    process_grid=grid,
    comm=comm,
)

plaq = measure_plaquette(U) # all ranks participate in the reduction
rank == 0 && println("plaquette = ", plaq)

MPI.Barrier(comm)
MPI.Finalize()
```

Select the threads backend once, restart Julia, and launch the script with the
desired numbers of ranks and Julia threads per rank:

```julia
import JACC
JACC.set_backend("threads")
```

```bash
mpiexec -n 4 julia --threads=4 --project=. four_d_mpi.jl
```

For every process grid, `prod(process_grid)` must equal the number of MPI
ranks, and each global lattice extent must be divisible by the corresponding
process-grid entry. If `process_grid` is omitted or set to `:auto`, Gaugefields
chooses a valid low-surface decomposition of `comm`.

## One GPU

Select the GPU backend once and restart Julia. For example, on NVIDIA:

```julia
import JACC
JACC.set_backend("cuda")
```

The ordinary single-process code at the beginning of this tutorial then
allocates its LM fields on the GPU. No `cuda=true` or accelerator-specific
Gaugefields constructor keyword is used. Those keywords belong to the legacy
storage implementations.

The same procedure uses `"amdgpu"` for AMD GPUs and `"oneapi"` for Intel
GPUs.

## Multiple GPUs with MPI

Use the same `four_d_mpi.jl` script with a GPU backend and launch one MPI rank
per GPU:

```bash
mpiexec -n 4 julia --project=. four_d_mpi.jl
```

LatticeMatrices automatically maps each node-local MPI rank to a visible GPU.
If all GPUs are visible to all ranks, local ranks 0, 1, and so on select device
ordinals 0, 1, and so on. If a scheduler exposes one different GPU to each
rank, each rank retains its sole visible device.

The number of ranks sharing a node must not exceed the number of GPUs visible
to those ranks. The machine or cluster must also provide a compatible MPI
installation and the driver/runtime for the selected CUDA, AMDGPU, or oneAPI
backend. The domain decomposition is controlled only by `process_grid`; device
selection is automatic.

When MPI communicates device arrays directly, the MPI library must be built
with support for that GPU runtime. The `mpiexec` used to launch the job and the
`libmpi` selected by MPI.jl must come from the same installation. An artifact
MPI without CUDA-aware transport is not a substitute for a CUDA-aware MPI in a
multi-GPU run.

For production multi-node runs, the MPI launcher and GPU-resource flags are
scheduler-specific, but the Julia source code remains unchanged.
