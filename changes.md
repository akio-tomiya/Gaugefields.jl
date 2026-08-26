# Changes

## v1.1.2

### Stout smearing

- Release materialized wider-than-halo shifts immediately in the stout
  pullback, preventing the temporary-field pool from growing on every force
  evaluation when rectangular loops are used.

## v1.1.1

### Gauge fixing

- Add `gaugefixing!` for four-dimensional Landau (`D_fix=4`) and Coulomb
  (`D_fix=3`) gauge fixing, with Los Alamos and steepest-descent checkerboard
  updates, overrelaxation, residual-based stopping, and configurable minimum
  iteration counts.
- Provide shared gauge-fixing diagnostics for the normalized link trace and
  gauge-condition residual, while preserving the plaquette under gauge
  transformations.
- Use generic SU(N) normalization on the LatticeMatrices backend, including
  tested SU(4) support, and preserve component precision on the portable
  `ComplexF32` path.
- Support LatticeMatrices, serial nowing, portable JACC accelerator, and
  deprecated MPI storage paths while retaining the optional direct-CUDA
  implementation for legacy CUDA storage.
- Correct the SU(3) projection used by the deprecated nowing-MPI
  normalization path.

### Portable update performance

- Require LatticeMatrices v1.2.1 and use its concrete, Adapt-compatible
  mutating-kernel launcher in heatbath and overrelaxation updates. This removes
  the per-site argument boxing seen on the JACC Threads backend without adding
  CUDA-, AMDGPU-, oneAPI-, or Threads-specific branches.
- Specialize four-dimensional LatticeMatrices even/odd Wilson-line evaluation
  so accumulated products stay in reusable ping-pong fields. Intermediate
  products no longer trigger repeated whole-field copies and eager halo
  exchanges; halo synchronization remains lazy and is performed when a later
  shifted read actually needs it.
- In one-thread CPU benchmarks, the LatticeMatrices backend is 4--38% faster
  than `Gaugefields.LegacyBackend()` across SU(2)/SU(3) heatbath, heatbath with
  overrelaxation, and gauge-only HMC cases. See the LatticeQCD benchmark record
  for the full conditions and allocation counts.

### Configuration I/O

- Restore fast ILDG loading for LatticeMatrices and deprecated nowing-MPI
  fields by coalescing per-site seeks and reads into maximal contiguous local
  blocks, using bounded chunks without reverting to full-volume reads on every
  MPI rank.

### Validation

- Add deterministic Landau and Coulomb trajectories, including a converged
  8^4 Coulomb regression with gauge functional `0.6755459192810215` for the
  documented seed and parameters.
- Check gauge-fixing agreement between LatticeMatrices and the serial, JACC
  accelerator, and deprecated MPI implementations, including one- and
  two-rank MPI domain decompositions.
- Test SU(N) unitarity and unit determinant, plaquette preservation,
  convergence behavior, input validation, Float32 execution, and the optional
  direct-CUDA path.
- Compare complete SU(2) and SU(3) heatbath sweeps element by element with the
  legacy storage implementation while sharing identical site-based random
  streams. Add both comparisons to the two-rank MPI CI job.
- Verify the optimized SU(3) heatbath path on an NVIDIA H100. After 13 seeded
  sweeps, CPU and CUDA both give the same normalized plaquette,
  `0.5589393945177883`.
- Cover ILDG local-volume reads for Float32 and Float64 payloads with x-, y-,
  z-, and t-direction process decompositions.

## v1.1.0

### Optional MPI support

- Move MPI.jl from a required dependency to a weak dependency and load the
  communicator implementation through a package extension.
- Use `LatticeMatrices.SerialCommunicator` when MPI is not loaded, so serial
  CPU and single-GPU applications can install and run Gaugefields without
  MPI.jl.
- Initialize MPI lazily when the first MPI-backed field is constructed, matching
  the historical portable API while keeping `using Gaugefields` side-effect
  free. Applications may still call `MPI.Init(...)` first to select custom
  initialization options; Gaugefields never finalizes MPI.
- Allow one-process applications to choose between the MPI path and the serial
  path by passing `MPI.COMM_WORLD` or `SerialCommunicator()`.
- Keep portable field wrappers concrete in 2D, 3D, and 4D so communicator
  selection happens at construction and does not add dynamic dispatch to hot
  lattice kernels.
- Keep deprecated MPI type names available for downstream compatibility
  without importing MPI.jl, and activate their MPI-dependent constructors only
  after MPI.jl is loaded. Keep these removal-bound implementations isolated
  under `ext/deprecated/mpi/`, separate from the supported MPI extension.

### Compatibility and tests

- Require LatticeMatrices v1.2 or later.
- Add clean-environment serial coverage with no MPI installation, lazy MPI
  lifecycle coverage, and one- and two-rank MPI tests.

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

### Tests

- Skip CLIME-backed ILDG I/O tests on Windows until the upstream binary-mode
  handling is fixed, while retaining ILDG coverage on Linux and macOS.

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
