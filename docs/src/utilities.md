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

For an independent configuration snapshot on the same backend:

~~~julia
Uold = copy_configuration(U)
copy_configuration!(Uold, U) # refresh an existing snapshot
~~~

These operations are preferred to `deepcopy` for MPI and device fields.

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

To avoid allocating on every trajectory:

~~~julia
P = gauge_momenta(U)
gaussian_momenta!(P; seed=0x1234, sweep=0)
~~~

The seed and sweep belong to the momentum stream, not to the hot-start or
heatbath streams. See [Randomness and reproducibility](randomness.md).

## JLD2

JLD2 is the default checkpoint format. Both JLD2 and NERSC support allocating
the destination from the file:

~~~julia
save_configuration("configuration.jld2", U)
Uloaded = load_configuration("configuration.jld2")
~~~

To load into an already allocated configuration:

~~~julia
load_configuration!(U, "configuration.jld2")
~~~

The high-level JLD2 format stores only global physical link arrays plus lattice
metadata. It does not serialize Gaugefields, MPI, or device objects. For a
distributed configuration every rank participates; rank 0 gathers the links
and writes the file.

An allocating load uses the current JACC backend and may select a new
communicator, process grid, or precision:

~~~julia
Uloaded = load_configuration(
    "configuration.jld2";
    comm=MPI.COMM_WORLD,
    process_grid=:auto,
    eltype=ComplexF32,
)
~~~

An in-place load uses the backend, device placement, communicator, and process
grid already owned by `U`. Thus a file written on multiple GPUs can be read on
one CPU core, and the reverse direction uses the same file.

## NERSC format

Use `format=:nersc` for four-dimensional SU(3) gauge configurations. Specify
the format explicitly; the filename extension does not select the reader.
The allocating loader reads the lattice dimensions from the header:

~~~julia
using Gaugefields

U = load_configuration("ensemble.nersc"; format=:nersc, eltype=ComplexF32)
save_configuration("copy.nersc", U; format=:nersc)
~~~

Omit `eltype` to use the file precision. To reuse a configuration with matching
lattice dimensions and three colors, load in place:

~~~julia
load_configuration!(U, "ensemble.nersc"; format=:nersc)
~~~

An in-place load preserves the destination precision, JACC backend, and MPI
layout, and refreshes its halos. File values are converted to the destination
precision on the host before transfer to the device.

### Output precision and compression

~~~julia
save_configuration("compact.nersc", U; format=:nersc,
                   precision=32, endian=:little, compressed=true)
~~~

| Output keyword | Default | Meaning |
| --- | --- | --- |
| `precision` | `64` | Store IEEE 32-bit or 64-bit real and imaginary components; choose `32` or `64`. |
| `endian` | `:big` | Byte order: `:big` or `:little`. |
| `compressed` | `false` | Store the full matrix, or store two rows and reconstruct the third using the SU(3) relation. |

Output precision is independent of the field precision: saving a ComplexF32
field with default options writes IEEE64 data without recovering precision
already lost in the field. Two-row output checks that reconstruction agrees
with the input within the source/output precision tolerance.

The reader accepts `4D_SU3_GAUGE` (two rows) and `4D_SU3_GAUGE_3x3`
(full matrix), with IEEE32/IEEE64 and either byte order. Output labels the byte
order explicitly with `BIG` or `LITTLE`. For unsuffixed `IEEE32` or `IEEE64`,
the reader selects a byte order only if all validation checks pass.

### MPI and JACC devices

After selecting the application's JACC backend as described in
[MPI, GPU, and multi-GPU execution](mpi.md), use the same I/O calls on CPU,
GPU, MPI, or MPI+GPU. For a distributed load, every rank executes:

~~~julia
using MPI, Gaugefields
MPI.Initialized() || MPI.Init()

U = load_configuration("ensemble.nersc"; format=:nersc,
                       comm=MPI.COMM_WORLD, process_grid=:auto,
                       eltype=ComplexF32)
save_configuration("distributed-copy.nersc", U; format=:nersc,
                   precision=32, compressed=true)
~~~

All ranks in the configuration communicator must call load/save in the same
order; do not restrict these calls to rank zero. The selected process grid
must divide the file's lattice dimensions. Rank zero reads or writes one
global host configuration, and existing LatticeMatrices communication
distributes or gathers the links. This is collective host-staged I/O, not
parallel-file I/O; rank-zero memory usage scales with the global volume.
The file stores physical links, not the process grid, device, or halo buffers.

### Input validation

Before replacing an existing configuration, the loader checks the payload
size, unsigned 32-bit checksum, plaquette, and average real link trace.
Observable checks use absolute and relative tolerances of `5e-6` for IEEE32
and `5e-8` for IEEE64, allowing rounded header values. Invalid headers,
truncated data, nonfinite values, or mismatches raise an error; a failed file
validation leaves the existing target unchanged. Under MPI, file-validation
errors are propagated to all participating ranks.

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
operations required by the underlying format. Temporary payload and file-list
paths are unique and automatically removed unless both `tempfile1` and
`tempfile2` are supplied explicitly.

ILDG remains available for interoperability with external lattice-QCD
software; it is not required for Gaugefields checkpoints.

The pre-v1 I/O entry points and examples are retained on the
[Legacy API](legacyapi.md) page.
