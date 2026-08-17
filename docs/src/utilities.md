# Utilities and configuration I/O

This page uses the Gaugefields v1 API. The examples work with the default
LatticeMatrices backend and do not depend on a particular JACC execution
backend.

## Create a configuration

~~~julia
import JACC
JACC.@init_backend

using Gaugefields

U = gauge_configuration(
    (8, 8, 8, 16);
    colors=3,
    halo=1,
    start=:cold,
)
~~~

A configuration is a vector with one link field per direction:

~~~julia
@assert length(U) == 4
Ux, Uy, Uz, Ut = U
~~~

The backend array inside an individual link is an implementation detail. Use
the public metadata functions instead of reading backend-specific fields:

~~~julia
gauge_backend(U)
gauge_lattice_size(U)   # (8, 8, 8, 16)
gauge_num_colors(U)     # 3
gauge_halo_width(U)     # 1
gauge_process_grid(U)   # one entry per direction
gauge_communicator(U)   # MPI communicator for LM fields
~~~

## Temporary link fields and link algebra

`similar(U[1])` allocates one compatible link field on the same CPU, GPU, or
distributed backend. Standard Gaugefields link operations therefore do not
need backend branches:

~~~julia
using LinearAlgebra

shifted_x = shift_U(U[1], (0, 1, 0, 0))
product = similar(U[1])
mul!(product, U[1], shifted_x')

summed_trace = tr(product)
~~~

`shift_U` and adjoint (`'`) are lazy field expressions. `mul!` evaluates the
site-wise matrix product into the explicitly supplied destination.
`tr(field)` includes the lattice-site and color trace and performs the
necessary distributed reduction.

For common observables, prefer `measure_plaquette` and
`measure_polyakov_loop` over manually allocating measurement work fields.

## Conjugate momenta

Allocate zero-valued traceless anti-Hermitian momenta with:

~~~julia
P = gauge_momenta(U)
~~~

For a reproducible Gaussian field on the LM backend:

~~~julia
P = gaussian_momenta(
    U;
    sigma=1.0,
    seed=0x1234,
    sweep=0,
    rng=Philox4x32(),
)
~~~

The seed and sweep belong to the momentum stream, not to the hot-start or
heatbath streams. See [Randomness and reproducibility](randomness.md).

## JLD2

JLD2 is the only format that currently supports allocating the destination
from the file:

~~~julia
save_configuration("configuration.jld2", U)
Uloaded = load_configuration("configuration.jld2")
~~~

To load into an already allocated configuration:

~~~julia
load_configuration!(U, "configuration.jld2")
~~~

The allocating form preserves the stored Gaugefields backend type. Use an
in-place load when the application must control lattice size, process grid, or
device allocation before reading.

## Bridge text format

Bridge input requires a preallocated target:

~~~julia
load_configuration!(U, "configuration.txt"; format=:bridge)
save_configuration("configuration.txt", U; format=:bridge)
~~~

## ILDG format

ILDG input also requires a preallocated target:

~~~julia
load_configuration!(U, "configuration.ildg"; format=:ildg)
save_configuration("configuration.ildg", U; format=:ildg)
~~~

Optional ILDG output keywords are forwarded by `save_configuration`. All
ranks that own a distributed configuration must participate in collective
operations required by the underlying format.

The pre-v1 I/O entry points and examples are retained on the
[Legacy API](legacyapi.md) page.
