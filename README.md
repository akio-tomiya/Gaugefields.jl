# Gaugefields

[![CI](https://github.com/akio-tomiya/Gaugefields.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/Gaugefields.jl/actions/workflows/CI.yml)
[![v1 documentation](https://img.shields.io/badge/docs-v1-blue.svg)](https://akio-tomiya.github.io/Gaugefields.jl/v1/)

🎉 **Gaugefields.jl v1 is available!**

Gaugefields.jl reached its first stable major release with v1.0.0.

The development version adds native APE, stout/EXP, HYP, HEX, and QEX-compatible nHYP smearing, analytic pullbacks, and smeared GaugeAction providers for HMC on LatticeMatrices-backed 4D fields; see [changes.md](changes.md).

Gaugefields.jl v1.1.4 adds an opt-in Grid/Bridge++ momentum normalization to
the MD driver while preserving the historical default; see [changes.md](changes.md).

Gaugefields.jl v1.1.3 adds shared STOUT pullbacks for legacy CUDA and MPI fields, faster serial ILDG loading, and public MPI-safe `normalize_U!`; see [changes.md](changes.md).

Gaugefields.jl v1.1.2 fixes rectangle-stout temporary storage; see [changes.md](changes.md).

Gaugefields.jl v1.1.1 adds Landau and Coulomb gauge fixing, improves
LatticeMatrices updates, and restores fast local-volume ILDG loading; see
[changes.md](changes.md).

Gaugefields.jl v1.1.0 made MPI.jl optional while preserving serial, GPU, MPI,
and multi-GPU execution.

## What's fixed in v1.0.3

- Empty `Vector{Union{}}` values now retain Base's normal `similar` behavior
  instead of ambiguously matching every gauge-field-vector specialization.
  This keeps Pkg/TOML operations usable after Gaugefields has been loaded.

## What's fixed in v1.0.1

- Enzyme MD differentiates the underlying LatticeMatrices links directly,
  avoiding Julia 1.12 failures involving composite gauge-field and lazy
  shifted/adjoint wrappers.
- `mul_shifted!`, `mul_shifted_adjoint!`, and `mul_adjoint!` provide clear,
  Enzyme-safe lattice-product operations for custom potentials.
- The Enzyme guide now includes the Julia 1.12-compatible potential contract
  and complete plaquette/HMC examples.

## What's new in v1.0.0

Compared with the previous release, v0.7.3, v1.0.0 adds and stabilizes:

- **A new high-level API** centered on `gauge_configuration`. It returns a
  vector of `Dim` gauge-link fields, uses the LatticeMatrices backend by
  default, and has a default halo width of one.
- **Portable 2D, 3D, and 4D execution** with
  [LatticeMatrices.jl](https://github.com/cometscome/LatticeMatrices.jl) v1.1
  and [JACC.jl](https://github.com/JuliaORNL/JACC.jl), covering threaded CPUs,
  NVIDIA, AMD, and Intel GPUs, MPI domain decomposition, and multi-GPU jobs.
- **Portable update and analysis workflows**, including plaquette and general
  Wilson-loop actions, heatbath and overrelaxation, gradient flow, 4D stout
  smearing, special initial configurations, and configuration I/O.
- **Decomposition-independent random-number streams** for seeded LM hot
  starts, Gaussian momenta, and heatbath updates, with explicit seeds and
  sweep counters for reproducible simulations.
- **A deterministic molecular-dynamics driver** with `QPQ()`, `PQP()`, and
  user-defined integrators, plus optional Enzyme-based action derivatives.
- **A reorganized v1 manual** with four-dimensional examples first, dedicated
  MPI/GPU, randomness, HMC, Enzyme, and Wilson-action guides, and the complete
  historical interface collected on one Legacy API page.

**Upgrading from v0.7:** Existing programs using `Initialize_Gaugefields` and
the historical API remain supported and retain their legacy backend and
defaults. New programs should use `gauge_configuration`, whose default backend
is LatticeMatrices. Gaugefields v1.1 requires LatticeMatrices v1.2 or later;
Enzyme users must add `Enzyme` as a direct dependency. See the
[high-level API](docs/src/highlevelapi.md) and
[Legacy API migration map](docs/src/legacyapi.md#migration-map).

# Abstract

This is a package for lattice QCD codes.
Treating gauge fields (links), gauge actions with MPI and autograd.

<img src="LQCDjl_block.png" width=300> 

This package is used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl)
and a code in a project [JuliaQCD](https://github.com/JuliaQCD/).

[NOTE: This is an extended version in order to implement higher-form gauge fields
 (i.e., 't Hooft twisted boundary condition/flux).
See [o-morikawa/Gaugefields.jl](https://github.com/o-morikawa/Gaugefields.jl)]

If you have questions and comments. Please use the issues section of this repository or use [Discussions in JuliaQCD](https://github.com/orgs/JuliaQCD/discussions/4).

[In Japanese] 質問やコメントを日本語でしたい方は[JuliaQCDのディスカッションボード](https://github.com/orgs/JuliaQCD/discussions/3)に書き込みをしてください。
 
# What this package can do:
This package has following functionarities

- SU(Nc) (Nc > 1) gauge fields in 2, 3, or 4 dimensions with arbitrary actions.
- **Z(Nc) 2-form gauge fields in 4 dimensions, which are given as 't Hooft flux.**
- U(1) gauge fields in 2 dimensions with arbitrary actions. 
- Configuration generation
    - Heatbath
    - quenched Hybrid Monte Carlo
    - quenched Hybrid Monte Carlo being subject to 't Hooft twisted b.c.
        - with external (non-dynamical) Z(Nc) 2-form gauge fields
    - quenched Hybrid Monte Carlo for SU(Nc)/Z(Nc) gauge theory
        - with dynamical Z(Nc) 2-form gauge fields
- Gradient flow via RK3
    - Yang-Mills gradient flow
    - Yang-Mills gradient flow being subject to 't Hooft twisted b.c.
    - Gradient flow for SU(Nc)/Z(Nc) gauge theory
- Landau and Coulomb gauge fixing with Los Alamos and steepest-descent updates
- I/O: portable JLD2 checkpoints are supported across CPU, GPU, MPI, and
  multi-GPU execution. ILDG and Bridge++ interoperability is also supported
  ([c-lime](https://usqcd-software.github.io/c-lime/) is installed implicitly
  with [CLIME_jll](https://github.com/JuliaBinaryWrappers/CLIME_jll.jl)).
- MPI parallel computation (experimental. See documents.)
    - quenched HMC with MPI being subject to 't Hooft twisted b.c.

- Portable GPU and multi-GPU computation through
  [LatticeMatrices.jl](https://github.com/cometscome/LatticeMatrices.jl) v1.2.0
  and [JACC.jl](https://github.com/JuliaORNL/JACC.jl). See the
  [GPU and multi-GPU tutorial](docs/src/tutorial4d.md#multiple-gpus-with-mpi).
    - NVIDIA GPUs through CUDA.jl
    - AMD GPUs through AMDGPU.jl/ROCm
    - Intel GPUs through oneAPI.jl
    - MPI domain decomposition with automatic node-local rank-to-device mapping

**The implementation of higher-form gauge fields is based on
[arXiv:2303.10977 [hep-lat]](https://arxiv.org/abs/2303.10977).**

Dynamical fermions will be supported with [LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl).

In addition, this supports followings
- **Autograd for functions with SU(Nc) variables**
- Stout smearing (exp projecting smearing)
- Stout force via [backpropagation](https://arxiv.org/abs/2103.11965)
- LLVM-level Automatic differentiation with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) (above 0.7. Experimental)

Autograd can be worked for general Wilson lines except for ones have overlaps.

# Install

In Julia REPL in the package mode,
```
add Gaugefields JACC
```

Add `MPI` as a direct dependency for MPI applications. JACC installs or
selects the package required by the requested GPU backend.

Serial CPU and single-GPU applications do not need MPI.jl. Loading MPI selects
the MPI path, and Gaugefields initializes MPI lazily when the first MPI-backed
field is constructed:

```julia
using MPI
using Gaugefields
```

Call `MPI.Init(...)` explicitly before constructing a field only when custom
initialization options such as the thread level are needed. Gaugefields never
calls `MPI.Finalize()`.

# How to use

## Recommended high-level API

The high-level API creates a vector with one gauge link field for each lattice
direction. It uses the LatticeMatrices/JACC backend by default:

```julia
import JACC
JACC.@init_backend

using Gaugefields

U = gauge_configuration(
    (16, 16, 16, 32);
    colors=3,
    start=:hot,
    seed=1234,
    process_grid=(1, 1, 1, 1),
    eltype=ComplexF64,
)

@assert length(U) == 4
@assert gauge_halo_width(U) == 1
println("plaquette = ", measure_plaquette(U))
println("Polyakov loop = ", measure_polyakov_loop(U))
```

Use `backend=LegacyBackend()` to request the serial compatibility backend.
The existing `Initialize_Gaugefields` API and its legacy default remain
unchanged.

## Two ways to construct HMC

Gaugefields supports both the traditional construction from elementary
operations and a deterministic, preallocated MD driver. The examples below
use the same Wilson gauge action and QPQ integrator. First define the shared
system:

```julia
function wilson_hmc_system()
    U = gauge_configuration(
        (4, 4, 4, 4);
        colors=3,
        halo=1,
        start=:hot,
        seed=0x1234,
        process_grid=(1, 1, 1, 1),
    )
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, 6.0 / 2, plaquettes)
    return U, action
end
```

### Traditional HMC from elementary operations

This is the historical style. The application explicitly assembles the link
update, analytic gauge force, QPQ integrator, Hamiltonian, Metropolis test, and
rollback:

```julia
using LinearAlgebra

function traditional_workspace(U)
    return (
        derivative=similar(U[1]),
        force_product=similar(U[1]),
        exponential=similar(U[1]),
        link_product=similar(U[1]),
        exponential_temps=[similar(U[1]), similar(U[1])],
    )
end

function traditional_hamiltonian(action, U, momenta)
    potential = -real(evaluate_GaugeAction(action, U)) / U[1].NC
    kinetic = real(momenta * momenta) / 2
    return potential + kinetic
end

function traditional_link_update!(U, momenta, step_size, workspace)
    for direction in eachindex(U)
        exptU!(
            workspace.exponential,
            step_size,
            momenta[direction],
            workspace.exponential_temps,
        )
        mul!(
            workspace.link_product,
            workspace.exponential,
            U[direction],
        )
        substitute_U!(U[direction], workspace.link_product)
    end
    return nothing
end

function traditional_momentum_update!(
    momenta,
    U,
    action,
    step_size,
    workspace,
)
    factor = -step_size / U[1].NC
    for direction in eachindex(U)
        calc_dSdUμ!(workspace.derivative, action, direction, U)
        mul!(
            workspace.force_product,
            U[direction],
            workspace.derivative,
        )
        Traceless_antihermitian_add!(
            momenta[direction],
            factor,
            workspace.force_product,
        )
    end
    return nothing
end

function traditional_hmc!(
    U,
    action;
    steps=4,
    trajectory_length=0.02,
    accept_uniform=0.5,
)
    momenta = gaussian_momenta(
        U;
        seed=0x5678,
        sweep=0,
    )
    old_U = copy_configuration(U)
    workspace = traditional_workspace(U)
    initial_hamiltonian = traditional_hamiltonian(action, U, momenta)

    step_size = trajectory_length / steps
    for _ in 1:steps
        traditional_link_update!(U, momenta, step_size / 2, workspace)
        traditional_momentum_update!(
            momenta,
            U,
            action,
            step_size,
            workspace,
        )
        traditional_link_update!(U, momenta, step_size / 2, workspace)
    end

    final_hamiltonian = traditional_hamiltonian(action, U, momenta)
    delta_hamiltonian = final_hamiltonian - initial_hamiltonian
    probability = exp(-max(0, delta_hamiltonian))
    accepted = accept_uniform < probability
    accepted || copy_configuration!(U, old_U)
    return (; accepted, delta_hamiltonian)
end

U, action = wilson_hmc_system()
traditional_result = traditional_hmc!(U, action)
println(traditional_result)
```

### HMC using the MD driver

With the driver, Gaugefields owns the deterministic QPQ evolution and its
workspaces. The application still owns momentum refresh, the Metropolis
decision, configuration backup, and rollback:

```julia
U, action = wilson_hmc_system()
momenta = gaussian_momenta(
    U;
    seed=0x5678,
    sweep=0,
)
old_U = copy_configuration(U)

md = md_driver(
    U,
    action;
    steps=4,
    trajectory_length=0.02,
    integrator=QPQ(),
)
diagnostics = md_trajectory!(U, momenta, md)

accept_uniform = 0.5
probability = exp(-max(0, diagnostics.delta_hamiltonian))
accepted = accept_uniform < probability
accepted || copy_configuration!(U, old_U)

driver_result = (
    accepted=accepted,
    delta_hamiltonian=diagnostics.delta_hamiltonian,
)
println(driver_result)
```

### Native APE, stout/EXP, HYP, HEX, and nHYP smearing

On a four-dimensional LatticeMatrices-backed configuration, construct a
smearing specification and use the ordinary high-level `smear` API:

```julia
# APE/HYP default to Bridge++-compatible iterative MaxReTr projection.
ape = APESmearing(alpha=0.6)
hyp = HYPSmearing(
    alpha_outer=0.75, alpha_middle=0.6, alpha_inner=0.3)
stout = StoutSmearing(rho=0.1, iterations=2) # EXPSmearing is an alias
hex = HEXSmearing(
    alpha_outer=0.125, alpha_middle=0.15, alpha_inner=0.15)
nhyp = NHYPSmearing(
    alpha_outer=0.5, alpha_middle=0.5, alpha_inner=0.4)

smeared = smear(U, ape)
recorded = smear(U, stout; record=true)
```

The explicit allocating interface returns the cache needed by an analytic
reverse pass:

```julia
polar_hyp = HYPSmearing(
    alpha_outer=0.75, alpha_middle=0.6, alpha_inner=0.3,
    projection=:polar)
V, cache = link_smear(U, polar_hyp)

# Fill dV with the cotangent with respect to V.
dU = link_smear_pullback(dV, U, cache)
```

MaxReTr is the natural choice for comparison with conventional lattice-QCD
software. It is forward-only here. For APE/HYP HMC or another differentiable
calculation, request `projection=:polar`; stout/EXP, HEX, nHYP, and polar
APE/HYP have analytic pullbacks. `ape_smearing(U; ...)`, `hyp_smearing`,
`stout_link_smearing`, `exp_smearing`, `hex_smearing`, and `nhyp_smearing`
are configuration-validating convenience builders. The existing
`stout_smearing` name remains the arbitrary-loop `CovNeuralnet` compatibility
API, so use `stout_link_smearing` for this native plaquette-only path.

For HMC, wrap a thin-link action with `SmearedGaugeAction`; APE and HYP must
use the polar choice:

```julia
action = SmearedGaugeAction(
    thin_link_action,
    APESmearing(alpha=0.6, projection=:polar),
)
md = md_driver(U, action; steps=4, trajectory_length=0.02, integrator=QPQ())
```

### nHYP-smeared HMC using the MD driver

Wrap the same thin-link Wilson action in `NHYPSmearedGaugeAction`. The driver
then evaluates the potential on nHYP-smeared links and analytically pulls its
force back to the thin links:

```julia
U, thin_link_action = wilson_hmc_system()  # hot start from the example above
action = NHYPSmearedGaugeAction(
    thin_link_action;
    alpha_outer=0.5,
    alpha_middle=0.5,
    alpha_inner=0.4,
)

momenta = gaussian_momenta(U; seed=0x4e485950, sweep=0)
old_U = copy_configuration(U)
md = md_driver(
    U,
    action;
    steps=4,
    trajectory_length=0.02,
    integrator=QPQ(),
)
diagnostics = md_trajectory!(U, momenta, md)

accept_uniform = 0.5
probability = exp(-max(0, diagnostics.delta_hamiltonian))
accepted = accept_uniform < probability
accepted || copy_configuration!(U, old_U)

println((; accepted, diagnostics.delta_hamiltonian))
```

`NHYPSmearedGaugeAction` and its driver workspace reuse the smeared fields,
the three-level nHYP cache, and all force fields. As in the preceding example,
the application owns momentum refresh, the Metropolis draw, and rollback.

The fixed `accept_uniform` makes the short examples reproducible. A
production HMC loop should refresh the momenta with a new `sweep` and draw a
uniform random number for every trajectory. `PQP()` and custom integrators are
also supported. See the complete [HMC guide](docs/src/hmc.md) for production
loops, MPI acceptance policy, restartable random streams, and
Sexton--Weingarten time-scale separation.

## Documentation

The manual contains the complete v1 API description and task-oriented examples:

- [Four-dimensional quick start](docs/src/tutorial4d.md)
- [Wilson loops and gauge actions](docs/src/wilsonloops_actions.md)
- [Measurements and QCDMeasurements.jl](docs/src/measurements.md)
- [HMC assembled from traditional operations and with the MD driver](docs/src/hmc.md)
- [Automatic differentiation with Enzyme](docs/src/autodiff.md)
- [Two- and three-dimensional fields](docs/src/dimensions.md)
- [Randomness and reproducibility](docs/src/randomness.md)
- [High-level API parameters](docs/src/highlevelapi.md)
- [MPI, GPU, and multi-GPU execution](docs/src/tutorial4d.md#multiple-gpus-with-mpi)

The APIs and examples that appeared in earlier versions of this README remain
available for compatibility. Their complete documentation has moved to the
single [Legacy API](docs/src/legacyapi.md) page. New code should use the
high-level v1 API above.

# Acknowledgment
If you write a paper using this package, please refer this code.

BibTeX citation is following
```
@article{Nagai:2024yaf,
    author = "Nagai, Yuki and Tomiya, Akio",
    title = "{JuliaQCD: Portable lattice QCD package in Julia language}",
    eprint = "2409.03030",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "9",
    year = "2024"
}
```
and the paper is [arXiv:2409.03030](https://arxiv.org/abs/2409.03030).
