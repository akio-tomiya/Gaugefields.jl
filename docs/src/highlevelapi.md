# High-level API parameters

This page lists the public convenience API introduced for Gaugefields v1.
Detailed examples start in the [four-dimensional tutorial](tutorial4d.md).

## Gauge configurations

```julia
gauge_configuration(lattice; kwargs...)
```

| Keyword | Default | Choices or constraints |
| --- | --- | --- |
| `backend` | `LatticeMatricesBackend()` | `LatticeMatricesBackend()` or `LegacyBackend()` |
| `colors` | `3` | Positive integer |
| `halo` | `1` | Nonnegative integer |
| `start` | `:cold` | `:cold`, `:hot` |
| `seed` | `nothing` | Integer seed or `nothing`; explicit seeds require LM |
| `process_grid` | `nothing` | `nothing`/`:auto`, or a positive integer tuple/vector of length `Dim` |
| `comm` | `nothing` | `nothing` for `MPI.COMM_WORLD`, or an explicit MPI communicator |
| `boundary` | `:periodic` | `:periodic` or one phase per dimension |
| `eltype` | `ComplexF64` | LM: `Float32`, `Float64`, `ComplexF32`, or `ComplexF64` |
| `rng` | `Philox4x32()` | `Philox4x32()`, `PCG32()`, `Xoshiro256PlusPlus()` |
| `verbose` | `0` | Integer output level |

The supported lattice dimensionalities are 2, 3, and 4. The return value is a
vector of length `Dim`. A 3D `LegacyBackend` configuration requires `halo=0`;
the recommended LM backend uses the common default `halo=1`.

For the LM backend, `process_grid=nothing` and `process_grid=:auto` choose a
valid decomposition on `comm` by minimizing a surface-to-volume score. The
application may pass a subcommunicator or `MPI.COMM_SELF`; Gaugefields never
finalizes the communicator.

## Metadata

These functions accept either one link or a complete configuration where
applicable:

| Function | Result |
| --- | --- |
| `gauge_backend(U)` | `LatticeMatricesBackend()` or `LegacyBackend()` |
| `gauge_lattice_size(U)` | Global lattice-size tuple |
| `gauge_num_colors(U)` | Number of colors |
| `gauge_halo_width(U)` | Halo width |
| `gauge_process_grid(U)` | MPI process-grid tuple |
| `gauge_communicator(U)` | MPI communicator, or `nothing` for serial legacy storage |

`copy_configuration(U)` allocates an independent backend-compatible copy.
`copy_configuration!(destination, source)` reuses a previous allocation and
performs the backend-specific device and halo updates needed for rollback.

## Conjugate momenta

`gauge_momenta(U)` allocates zero-valued momentum fields compatible with `U`.

```julia
gaussian_momenta(U; kwargs...)
gaussian_momenta!(P; kwargs...)
```

| Keyword | Default | Choices or constraints |
| --- | --- | --- |
| `sigma` | `1.0` | Gaussian standard deviation |
| `seed` | `nothing` | Integer or `nothing`; explicit seed requires LM |
| `sweep` | `0` | Nonnegative trajectory/stream counter |
| `rng` | `Philox4x32()` | `Philox4x32()`, `PCG32()`, `Xoshiro256PlusPlus()` |

The allocating form is convenient for one-off use. HMC applications should
allocate `P = gauge_momenta(U)` once and refresh it with `gaussian_momenta!`.

## Measurements

```julia
measure_plaquette(U; normalize=true)
measure_polyakov_loop(U; normalize=true)
```

With `normalize=true`, a cold field returns one. `normalize=false` selects the
historical summed/unnormalized convention.

Topological charge, topological-charge density, energy density, and other
physics measurements are provided by QCDMeasurements.jl; see
[Measurements](measurements.md#Advanced-observables).

## Wilson loops and gauge actions

Create a coefficient-weighted action from one or more groups of closed Wilson
paths:

```julia
action = GaugeAction(U)
loops = make_loops_fromname("plaquette"; Dim=length(U))
append!(loops, loops')
push!(action, coefficient, loops)
```

| Function | Result |
| --- | --- |
| `evaluate_gaugelinks!(output, loops, U, temps)` | Ordered path product, summed for a loop collection |
| `evaluate_GaugeAction(action, U)` | Coefficient-weighted traced lattice sum |
| `evaluate_GaugeAction_untraced(action, U)` | Allocated matrix-valued action sum |
| `evaluate_GaugeAction_untraced!(output, action, U)` | In-place matrix-valued action sum |
| `calc_dSdUμ(action, μ, U)` | Allocated raw matrix derivative in direction `μ` |
| `calc_dSdUμ!(output, action, μ, U)` | In-place raw matrix derivative |

See [Wilson loops and gauge actions](wilsonloops_actions.md) for explicit
`Wilsonline` paths, coefficient conventions, and reuse in heatbath, gradient
flow, and MD.

## Heatbath and overrelaxation

For a plaquette action:

```julia
heatbath_updater(U; beta, kwargs...)
heatbath_updater(U, beta; kwargs...)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `beta` | required | Plaquette coupling |
| `ITERATION_MAX` | `10^5` | Positive rejection-loop limit |
| `seed` | `0` | LM site-stream seed |
| `sweep` | `0` | Initial heatbath sweep counter |
| `overrelaxation_sweep` | `sweep` | Initial independent overrelaxation counter |
| `rng_algorithm` | `Philox4x32()` | `Philox4x32()`, `PCG32()`, `Xoshiro256PlusPlus()` |

Apply the updater with `heatbath!(U, updater)` or
`overrelaxation!(U, updater)`. The corresponding counter advances after a
successful call.

For a general `GaugeAction`:

```julia
heatbath_updater(U, action; kwargs...)
```

This accepts the same random-stream parameters and also:

| Keyword | Default | Choices or constraints |
| --- | --- | --- |
| `coloring` | `:auto` | `:auto`, `:sequential`, or a valid tuple of `HeatbathColoring` values |
| `max_colors` | `256` | Positive search limit used by `:auto` |

`coloring=:auto` derives a safe periodic coloring from the action staples.
`:sequential` is an exact but deliberately slow fallback.

## Gradient flow

```julia
gradient_flow(U; steps=1, step_size=0.01)
gradient_flow(U, loops, coefficients; steps=1, step_size=0.01)
```

`steps` and `step_size` must be positive. The second form constructs a
general-action flow from explicit loops and coefficients. Apply the resulting
integrator with `flow!(U, flow)`.

## Stout smearing

```julia
stout_smearing(U; loops=:plaquette, rho=0.1)
```

`loops` may be one symbol/string or a collection. `rho` may be one scalar or
one coefficient per loop.

```julia
smear(U, smearing; record=false, calcdSdU=false, temps=nothing)
```

The default return value is the smeared configuration. `record=true` returns a
named tuple containing `configuration`, `history`, and `derivative`.

## Configuration I/O

```julia
save_configuration(filename, U; format=:jld2, kwargs...)
load_configuration(filename; format=:jld2, process_grid=nothing, comm=nothing,
                   halo=nothing, boundary=nothing, eltype=nothing)
load_configuration!(U, filename; format=:jld2)
```

`save_configuration` and `load_configuration!` accept `:jld2`, `:bridge`, and
`:ildg`. Allocating `load_configuration` currently accepts only `:jld2`;
Bridge and ILDG input require a preallocated destination.

JLD2 is the portable Gaugefields checkpoint format. During output, all ranks
participate and rank 0 gathers one global configuration as ordinary host
arrays. The file contains physical links and lattice metadata, not backend,
device, communicator, process-grid, or halo arrays. It can therefore be loaded
on CPU, GPU, MPI, or multi-GPU execution with a different decomposition.

The allocating loader uses the current JACC backend. Its `process_grid`,
`comm`, and `eltype` keywords may override the writer's execution layout and
stored precision. The in-place loader uses the destination configuration's
layout. Legacy object-serialized JLD2 files remain readable on one rank.

ILDG output creates unique temporary paths in the output directory and removes
them after packing; all ranks in `gauge_communicator(U)` must participate.

## Molecular dynamics

Construct a pure-gauge MD driver once and reuse its preallocated work fields:

```julia
driver = md_driver(
    U,
    action;
    steps=20,
    trajectory_length=1.0,
    integrator=QPQ(),
)

result = md_trajectory!(U, p, driver)
```

| Parameter | Default | Choices or constraints |
| --- | --- | --- |
| `U` | required | Gauge configuration used to allocate the driver |
| `action` | required | A `GaugeAction`, `MDActionSet`, or an object implementing the MD action-provider interface |
| `steps` | required | Positive integer number of MD steps |
| `trajectory_length` | `1.0` | Finite, nonzero number; a negative value is useful for reversibility tests |
| `integrator` | `QPQ()` | `QPQ()`, `PQP()`, a step function, or an object implementing `md_step!` |

The trajectory length and all built-in integration coefficients use the real
component type of `U`; for example a `ComplexF32` configuration produces a
`Float32` step size.

`QPQ()` applies `Q(1/2) P(1) Q(1/2)` in each step and matches the ordering in
the historical Gaugefields HMC examples. `PQP()` applies
`P(1/2) Q(1) P(1/2)`. Here `Q` updates the gauge links and `P` updates the
momenta. `md_step_size(driver)` returns `trajectory_length / steps`.

`md_trajectory!` mutates both `U` and `p`. With its default
`diagnostics=true`, it returns
`initial_hamiltonian`, `final_hamiltonian`, and `delta_hamiltonian`. Pass
`diagnostics=false` to omit those two global Hamiltonian evaluations and
return `nothing`. `md_hamiltonian(U, p, driver)` is also available directly.

### Analytic and Enzyme forces

A `GaugeAction` uses Gaugefields' analytic force. To differentiate a custom
scalar potential with Enzyme, load Enzyme and wrap the function:

```julia
using Enzyme

# The potential must return the complete, real potential energy V(U).
# If num_temps is nonzero, Gaugefields appends `temps` to this argument list.
function my_potential(U1, U2, U3, U4, coupling, temps)
    # Fill/reuse temps and calculate V(U1, U2, U3, U4).
    return value
end

action = enzyme_md_action(my_potential, coupling; num_temps=3)
driver = md_driver(U, action; steps=20)
```

The four underlying LatticeMatrices links are passed as separate arguments so
that Enzyme can differentiate them. The outer simulation variable is still
the normal vector of Gaugefields links. Extra positional arguments given to
`enzyme_md_action` are treated as constants. When `num_temps > 0`, the final
argument received by the potential is a collection of that many reusable LM
work fields; Gaugefields clears it before each evaluation and allocates the
matching shadow workspace. Use `mul_shifted!`, `mul_shifted_adjoint!`, and
`mul_adjoint!` for shifted products in differentiated code. The function must
return a real scalar with the desired sign and normalization already included.
The same function supplies both the Hamiltonian value and the automatically
differentiated force, preventing an action/force mismatch.

The initial Enzyme MD implementation supports four-dimensional
LatticeMatrices configurations with `halo >= 1`, including CUDA and MPI. A
regular `GaugeAction` remains the portable choice for Legacy fields and for
two- or three-dimensional configurations. CUDA multi-GPU execution requires a
CUDA-aware MPI installation, with matching `mpiexec` and `libmpi` selections.

For direct access to the matrix gradient, constant-argument annotations, and
primal/adjoint work fields, see
[Automatic differentiation with Enzyme](autodiff.md).

Other packages can implement an action provider by defining all three methods:

```julia
Gaugefields.md_action_workspace(action::MyAction, U) = MyWorkspace(U)
Gaugefields.md_potential(action::MyAction, U, workspace) = potential_value
function Gaugefields.md_force!(force, action::MyAction, U, workspace)
    # Write dp/dtau, as traceless anti-Hermitian fields, into `force`.
    return nothing
end
```

`md_action_workspace` is called once by `md_driver`; the other two methods
must reuse it rather than allocate large work fields on every stage.

### Multiple actions and time scales

Use `MDActionSet` to combine independently implemented action providers. A
type-stable `NamedTuple` stores the terms and their reusable workspaces:

```julia
actions = MDActionSet(;
    gauge=gauge_action,
    fermion=fermion_action,
)
```

The ordinary `md_potential`, `md_force!`, `md_hamiltonian`, and
`update_momenta!` operations sum all members. A custom integrator can kick a
selected group without changing the action-provider interface:

```julia
gauge_group = MDForceGroup(:gauge)
update_momenta!(P, U, delta_tau, driver, gauge_group)
```

The built-in two-scale Sexton--Weingarten integrator schedules named groups:

```julia
integrator = SextonWeingarten(;
    slow=:fermion,
    fast=:gauge,
    n_fast=4,
    ordering=QPQ(),
)

driver = md_driver(
    U,
    actions;
    steps=10,
    trajectory_length=1.0,
    integrator,
)
```

`slow` and `fast` also accept tuples such as `(:light, :strange)` or explicit
`MDForceGroup` objects. The two groups must be disjoint and together cover
every member of the action set. `n_fast` is a positive runtime integer, so
changing it does not create a new driver or integrator type.

With the default `QPQ()` ordering, each outer step evolves the fast
Hamiltonian for half a step, applies one full slow-force kick, and evolves the
fast Hamiltonian for the other half. Each half contains `n_fast` QPQ
substeps. `ordering=PQP()` instead surrounds one full fast evolution with
half slow-force kicks; in that ordering the full fast evolution contains
`n_fast` substeps.

### Custom integrators

The two elementary MD operations make the mutated field explicit as their
first argument:

- `update_momenta!(P, U, delta_tau, driver)` changes `P` and leaves `U`
  unchanged.
- `update_gaugefields!(U, P, delta_tau, driver)` changes `U` and leaves `P`
  unchanged.

Both functions return the field named by their first argument. The `!` and the
argument order therefore show exactly which object is updated.

For a one-off integrator, write an ordinary Julia function for one MD step.
`md_trajectory!` calls it `steps` times with
`delta_tau = trajectory_length / steps`:

```julia
function five_update_pqp!(U, P, delta_tau, driver)
    update_momenta!(P, U, 0.25 * delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    update_momenta!(P, U, 0.5 * delta_tau, driver)
    update_gaugefields!(U, P, 0.5 * delta_tau, driver)
    update_momenta!(P, U, 0.25 * delta_tau, driver)
    return nothing
end

driver = md_driver(U, action; steps=20, integrator=five_update_pqp!)
```

For a reusable named integrator, define a type in the application or package
that owns it, then extend `Gaugefields.md_step!`. Subtyping
`AbstractMDIntegrator` is optional. For example, an Omelyan-style PQP update
is:

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

driver = md_driver(
    U,
    action;
    steps=20,
    integrator=OmelyanPQP(0.1931833275037836),
)
```

The custom method must implement exactly one step. It is responsible for
multiplying every coefficient by `delta_tau`; negative coefficients are
allowed. For standard HMC, the sequence should normally be symmetric so that
changing the sign of `trajectory_length` reverses the evolution. The original
integrator function or object is retained as `driver.integrator`.

This extension point changes only the ordering and coefficients. It is
independent of the force source, so the same custom integrator works with a
`GaugeAction`, `enzyme_md_action`, or another action provider. For multiple
independently scheduled actions, use `MDActionSet` and the group-selecting
`update_momenta!` method shown above.

## HMC responsibility boundary

Gaugefields deliberately provides deterministic MD evolution rather than an
HMC updater. It does not refresh momenta, draw a Metropolis random number,
accept or reject a proposal, restore a rejected configuration, or own a
trajectory counter. Those policy and random-state operations belong in an
application or a higher-level HMC package.

An HMC layer built on `md_driver` therefore owns the following parameters:

| HMC parameter | Owner or choice |
| --- | --- |
| Gauge action | Passed to `md_driver` |
| MD integrator | `QPQ()`, `PQP()`, a step function, or a custom `md_step!` method |
| Trajectory length and step count | Passed to `md_driver` |
| Momentum width | `sigma=1.0` by default in `gaussian_momenta` |
| Momentum random stream | `seed`, trajectory `sweep`, and `rng` passed to `gaussian_momenta` |
| Metropolis random stream | Supplied and synchronized by the HMC layer |
| Acceptance, rollback, and trajectory counter | Maintained by the HMC layer |

See [Randomness and reproducibility](randomness.md) for the complete state that
a reproducible HMC chain must save.
