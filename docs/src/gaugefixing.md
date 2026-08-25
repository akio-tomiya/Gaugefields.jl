# Gauge fixing

Gaugefields v1.1.1 provides in-place Landau and Coulomb gauge fixing through
the same `gaugefixing!` interface on the recommended LatticeMatrices backend.
The simulation source is unchanged between CPU, GPU, MPI, and multi-GPU
execution.

The implementation applies checkerboard Los Alamos (LA) updates followed by
checkerboard steepest-descent (SD) updates. Both stages are optional, have
independent overrelaxation parameters and iteration limits, and use the same
gauge-condition residual for convergence.

## Complete example

The following example fixes a reproducible hot SU(3) configuration to Coulomb
gauge:

```julia
import JACC
JACC.@init_backend

using Gaugefields

U = gauge_configuration(
    (8, 8, 8, 8);
    colors=3,
    start=:hot,
    seed=1234,
    verbose=1,
)

# Reuse these allocations when fixing more than one configuration.
g_transform = similar(U[1])
work = [similar(U[1]) for _ in 1:6]

plaquette_before = measure_plaquette(U)

gaugefixing!(
    U,
    g_transform,
    1.0,       # LA overrelaxation
    200,       # maximum LA iterations
    1.99,      # SD overrelaxation
    10_000,    # maximum SD iterations
    1e-14,     # gauge-condition tolerance
    1234,      # configuration label used in diagnostic output
    work;
    D_fix=3,   # Coulomb gauge
    min_iterations=500,
)

plaquette_after = measure_plaquette(U)
@assert isapprox(plaquette_after, plaquette_before; rtol=5e-12, atol=5e-12)
```

`gaugefixing!` modifies `U` and returns the same configuration. Gauge
transformations preserve gauge-invariant observables such as the plaquette;
the check above is therefore a useful application-level validation.

For Landau gauge, use the same code with:

```julia
D_fix = 4
```

## Landau and Coulomb conditions

At each site, Gaugefields forms the traceless anti-Hermitian part of

```math
\Delta(x) = \sum_{\mu=1}^{D_{\mathrm{fix}}}
\left[U_\mu(x)-U_\mu(x-\hat\mu)\right].
```

The two standard four-dimensional choices are:

| Gauge | `D_fix` | Links entering the gauge functional and condition |
| --- | ---: | --- |
| Landau | `4` | Spatial and temporal links |
| Coulomb | `3` | Spatial links only |

Every local transformation is nevertheless applied consistently to all four
link directions as ``U_\mu(x) \to g(x)U_\mu(x)g^\dagger(x+\hat\mu)``.

The logged value `tr[U]` is the normalized gauge functional over the first
`D_fix` directions. The logged `tr[dA dA']` value is the nonnegative
gauge-condition residual used for the `tol` comparison. It should decrease
toward zero as the gauge condition is satisfied.

## Arguments and convergence

The complete call is:

```julia
gaugefixing!(
    U, g_transform,
    LA_overrelax, LA_iteration,
    SD_overrelax, SD_iteration,
    tol, config_n, work;
    D_fix=4,
    min_iterations=500,
)
```

| Argument | Meaning |
| --- | --- |
| `U` | Four-link configuration modified in place. |
| `g_transform` | Field used for the current checkerboard transformation. Allocate it with `similar(U[1])`. |
| `LA_overrelax` | Los Alamos overrelaxation parameter. `1.0` applies the unaccelerated update. |
| `LA_iteration` | Nonnegative maximum number of LA iterations. Use `0` to skip this stage. |
| `SD_overrelax` | Steepest-descent overrelaxation parameter. |
| `SD_iteration` | Nonnegative maximum number of SD iterations. Use `0` to skip this stage. |
| `tol` | Nonnegative residual threshold. Use `0.0` to disable early convergence for ordinary floating-point data. |
| `config_n` | Integer label printed in diagnostics; it does not seed the algorithm. |
| `work` | At least six fields compatible with `U[1]`. Reuse them between calls. |
| `D_fix` | Number of directions in the gauge condition: normally `4` for Landau or `3` for Coulomb gauge. |
| `min_iterations` | Minimum iterations attempted in each enabled stage before its tolerance test may stop it. |

For each stage, the convergence test begins after
`min(min_iterations, requested_iterations)` iterations. Consequently, the
default `min_iterations=500` runs a stage with fewer than 500 requested
iterations to its requested limit. Set `min_iterations=0` to permit a stop
after the first update whose residual is below `tol`.

## Reusing storage over an ensemble

Allocate the transformation and six work fields once for configurations with
the same type and layout:

```julia
g_transform = similar(U[1])
work = [similar(U[1]) for _ in 1:6]

for (configuration_number, U) in enumerate(configurations)
    gaugefixing!(
        U, g_transform,
        1.0, 200,
        1.99, 10_000,
        1e-14, configuration_number, work;
        D_fix=3,
    )
end
```

All configurations in this loop must have storage compatible with the fields
used to allocate `g_transform` and `work`. Allocate a new workspace after
changing the lattice, color count, precision, backend, process grid, or
communicator.

## CPU, GPU, and MPI execution

For the recommended `gauge_configuration` API, gauge-fixing code does not
contain device-specific branches. Select the JACC backend before importing
Gaugefields, as described in the
[four-dimensional quick start](tutorial4d.md#select-a-backend):

```julia
import JACC
JACC.set_backend("cuda")   # NVIDIA
# JACC.set_backend("amdgpu") # AMD
# JACC.set_backend("oneapi") # Intel
```

Restart Julia, initialize the selected backend with `JACC.@init_backend`, and
run the ordinary example. The threads backend is selected by default.

For MPI or multi-GPU execution, construct `U` with the desired communicator
and process grid. Every rank must call `gaugefixing!` in the same order:

```julia
using MPI

import JACC
JACC.@init_backend

using Gaugefields

nranks = MPI.Comm_size(MPI.COMM_WORLD)
U = gauge_configuration(
    (8, 8, 8, 8 * nranks);
    colors=3,
    start=:hot,
    seed=1234,
    process_grid=(1, 1, 1, nranks),
    comm=MPI.COMM_WORLD,
)

g_transform = similar(U[1])
work = [similar(U[1]) for _ in 1:6]
gaugefixing!(
    U, g_transform,
    1.0, 200,
    1.99, 10_000,
    1e-14, 1234, work;
    D_fix=3,
)
```

Launch one MPI rank per GPU for multi-GPU execution. CUDA-aware MPI and the
relationship between MPI.jl's library and the system launcher are covered in
[MPI, GPU, and multi-GPU execution](mpi.md#multiple-gpus).

## Supported storage paths

| Storage path | Gauge-fixing support |
| --- | --- |
| LatticeMatrices from `gauge_configuration` | Recommended; CPU, JACC GPU, MPI, and multi-GPU. Generic SU(N), with SU(3) and SU(4) regression coverage. |
| Serial legacy nowing storage | Compatibility path for SU(2) and SU(3). |
| Portable legacy JACC accelerator storage | Compatibility path for SU(2) and SU(3). |
| Deprecated nowing/wing MPI storage | Compatibility path for SU(2) and SU(3); MPI.jl is loaded only when requested. |
| Direct legacy CUDA storage | Optional CUDA.jl path specialized to double-precision SU(3). |

New programs should use the first row. The other implementations remain to
validate compatibility with older Gaugefields applications.
