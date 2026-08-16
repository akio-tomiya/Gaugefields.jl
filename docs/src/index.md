# Gaugefields.jl v1

Gaugefields.jl provides gauge-link fields and core algorithms for lattice QCD.
The v1 API uses LatticeMatrices and JACC by default and represents a
`Dim`-dimensional gauge configuration as a vector of `Dim` link fields.

## Start here

For the main workflow, begin with the
[four-dimensional quick start](tutorial4d.md). It covers configuration
creation, measurements, heatbath, molecular dynamics, stout smearing, file
I/O, MPI, and GPU execution.

~~~julia
import JACC
JACC.@init_backend

using Gaugefields

U = gauge_configuration(
    (16, 16, 16, 32);
    colors=3,
    halo=1,
    start=:hot,
    seed=1234,
)

println("plaquette = ", measure_plaquette(U))
println("Polyakov loop = ", measure_polyakov_loop(U))
~~~

The same API is used for two and three dimensions; see
[Two and three dimensions](dimensions.md).

## Manual structure

- [Applications](applications.md) combines the v1 building blocks into common
  simulation workflows.
- [Measurements](measurements.md) documents the normalized observable API.
- [Utilities and I/O](utilities.md) covers metadata, link-field algebra, and
  configuration formats.
- [MPI, GPU, and multi-GPU execution](mpi.md) explains portable execution with
  one code path.
- [HMC and custom integrators](hmc.md) builds HMC around the deterministic MD
  driver.
- [High-level API parameters](highlevelapi.md) is the parameter reference.
- [Public API index](usefulfunctions.md) collects the v1 docstrings.

## Backends

`LatticeMatricesBackend()` is the default. CPU threads, GPUs, MPI, and
multi-GPU execution are selected through JACC and `process_grid` without
accelerator-specific Gaugefields constructors.

`LegacyBackend()` is available only when an application deliberately needs
the serial compatibility implementation.

!!! warning "Pre-v1 compatibility API"
    Programs written against earlier Gaugefields releases remain supported,
    but their constructors, backend flags, low-level HMC drivers, and historical
    examples are documented only on the
    [Legacy API (compatibility)](legacyapi.md) page. They are not the recommended
    interface for new programs.

## Installation

In Julia package mode:

~~~julia
pkg> add Gaugefields
~~~

Gaugefields v1 requires LatticeMatrices v1.1 or later within the compatibility
bounds declared by the package.
