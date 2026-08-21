# Changes

## v1.0.5

### Molecular dynamics

- Add type-stable `MDActionSet` composition and named `MDForceGroup` updates
  for independently implemented action providers.
- Add the two-time-scale `SextonWeingarten` integrator with QPQ and PQP
  orderings, runtime-configurable fast substeps, and constructor validation.
- Cover summed potentials and forces, force scheduling, reversibility, and MPI
  domain decomposition in the MD tests.
- Preserve the real component precision of the gauge field in MD trajectory
  lengths, step sizes, and built-in integrator coefficients.

### High-level API

- Add allocation-free `gaussian_momenta!` refresh and semantic
  `copy_configuration`/`copy_configuration!` snapshot operations.
- Accept explicit MPI communicators in `gauge_configuration` and add automatic
  process-grid selection based on lattice divisibility and surface-to-volume
  cost.
- Preserve `Float32` step sizes in the standard and general gradient-flow
  drivers.

### Configuration I/O

- Store high-level JLD2 checkpoints as backend-independent global link arrays.
  Rank 0 gathers and writes distributed CPU/GPU fields, and loading may use a
  different process grid, device backend, or floating-point precision.
- Generate and clean unique ILDG save temporaries automatically, and propagate
  root packing failures collectively instead of leaving non-root ranks stuck
  at a barrier.

### Documentation

- Document complete HMC loops for both the traditional composition of
  elementary updates and the reusable MD driver.

## v1.0.4

### ILDG I/O

- Read and write both ILDG 32-bit and 64-bit floating-point payloads correctly,
  using the precision declared in the `ildg-format` metadata.
- Add `precision=:field`, `precision=32`, and `precision=64` to
  `save_binarydata`; `:field` preserves the gauge field's component precision.
- Emit ILDG v1.2 XML metadata with the required namespace and record ordering,
  and store the binary payload in big-endian byte order.
- Update the CLIME_jll command invocation and ensure that opened files and
  temporary resources are closed reliably.

### MPI and GPU correctness

- Restrict LIME extraction and packing to the MPI root rank, use the gauge
  field's communicator instead of assuming `MPI.COMM_WORLD`, and share unique
  temporary payload paths safely across ranks.
- Synchronize direct JACC/GPU writes, mark lattice data as modified, and refresh
  halo regions after loading a configuration.

### Tests

- Add 32-bit and 64-bit ILDG round-trip and metadata tests on CPU, MPI domain
  decompositions, and GPU/JACC backends.
- Add regression coverage for portable element types, halo epochs, and
  long-distance shifted-field operations.
