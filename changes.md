# Changes

## v1.1.5

### Native UV smearing and analytic HMC

- Add the common `NativeLinkSmearing` interface with `StoutSmearing`
  (`EXPSmearing`), `HEXSmearing`, `APESmearing`, and `HYPSmearing` alongside
  `NHYPSmearing`. Each specification supports complete-step iteration and
  the allocating/preallocated `link_smear`, `link_smear!`,
  `link_smear_pullback`, and `link_smear_pullback!` APIs.
- Make APE/HYP `projection=:max_retr` the default for Bridge++ compatibility,
  with configurable iteration limit and convergence tolerance. Add
  `projection=:polar` as the explicit differentiable choice; stout/EXP, HEX,
  nHYP, and polar APE/HYP provide analytic pullbacks, while MaxReTr pullback
  requests fail explicitly.
- Extend the high-level `smear` API to every native specification and add the
  `stout_link_smearing`, `exp_smearing`, `hex_smearing`, `ape_smearing`, and
  `hyp_smearing` builders. The existing arbitrary-loop `stout_smearing`
  compatibility API is unchanged.
- Add `SmearedGaugeAction` for evaluating any analytically smeared gauge
  action and pulling its force back to the thin links. APE/HYP molecular
  dynamics requires `projection=:polar` and is rejected at construction when
  the default MaxReTr choice is used.
- Require LatticeMatrices v1.2.5. Its external comparisons agree with
  Bridge++ 2.1.3 for APE, HYP, and HEX to maximum absolute differences
  `1.78e-15`, `1.78e-15`, and `4.85e-12`; stout forward and pullback agree
  with QEX to `5.90e-16` and `1.89e-15`.

### Normalized HYP smearing

- Add `NHYPSmearing` and the `nhyp_smearing(U; alpha_outer,
  alpha_middle, alpha_inner)` builder for QEX-compatible normalized HYP
  smearing. QEX's `(alpha1, alpha2, alpha3)` coefficient order corresponds to
  `(alpha_inner, alpha_middle, alpha_outer)`.
- Add allocating and preallocated interfaces through `nhyp_smear` and
  `nhyp_smear!`. The existing high-level `smear` API accepts an
  `NHYPSmearing`; `record=true` returns the smeared configuration together
  with the cache required by the reverse pass.
- Add `NHYPSmearingCache` as the Gaugefields wrapper around
  LatticeMatrices' reusable nHYP workspace. It retains the three-level
  forward intermediates, detects thin links changed after the forward pass,
  and controls allocations across repeated HMC trajectories.
- Add allocating and preallocated analytic reverse passes through
  `nhyp_pullback` and `nhyp_pullback!`. They return the unconstrained
  thin-link cotangent; projection onto the gauge Lie algebra remains the HMC
  integrator's responsibility.
- Add `NHYPSmearedGaugeAction`, an `md_driver` action provider that evaluates
  a `GaugeAction` on nHYP-smeared links, converts its raw matrix derivative to
  the LatticeMatrices cotangent convention, pulls it back to the thin links,
  and performs the standard traceless anti-Hermitian force projection. Its
  workspace reuses the smeared configuration, nHYP cache, cotangents, and
  force temporaries across trajectories.
- Restrict this interface to four-link, four-dimensional
  `Gaugefields_4D_MPILattice` configurations with `NDW >= 1`. Legacy storage
  and non-4D fields fail with an explicit `ArgumentError`. CPU, MPI, and GPU
  execution are delegated to LatticeMatrices and JACC without copying links
  through host storage.
- Use the LatticeMatrices v1.2.5 native smearing kernels; nHYP itself remains
  compatible with the v1.2.4 definition and coefficient convention.

### Validation

- Compare the Gaugefields allocating, preallocated, high-level, and pullback
  APIs with direct LatticeMatrices results on fixed-seed hot SU(3) fields.
  The serial wrapper suite passes 34/34 tests with one and four CPU threads.
- Compare two-rank MPI forward and pullback results with an undecomposed
  calculation of the same global hot field; all eight distributed checks
  pass on both ranks.
- Run the complete wrapper suite on an NVIDIA H100 NVL (compute capability
  9.0). The input, smeared output, and pullback output remain
  `CuArray{ComplexF64}` throughout, and all 34 tests pass.
- Check the nHYP MD provider against the ordinary `GaugeAction` potential and
  force at zero smearing coefficients, then evolve a fixed-seed hot SU(3)
  field and verify finite Hamiltonian diagnostics and forward/backward
  reversibility. On CPU, halving the QPQ step from 1/100 to 1/200 and 1/400
  reduces `|delta_hamiltonian|` from `1.60e-4` to `3.95e-5` and `9.85e-6`,
  respectively, as expected for a second-order integrator.
- Run the hot-field nHYP MD trajectory on an NVIDIA H100 NVL. A four-step
  trajectory gives `delta_hamiltonian = -3.95e-5`; the forward/backward link
  and momentum errors are `1.11e-15` and `8.88e-16`, and the thin links,
  smeared links, and pullback fields all remain in `CuArray{ComplexF64}`
  storage.
- Pass 39 native-smearing wrapper checks and four finite-difference gauge
  action force checks for stout/EXP, HEX, polar APE, and polar HYP on CPU.

## v1.1.4

### Molecular dynamics

- Add the backend-neutral `momentum_denominator` keyword to `md_driver`. The
  kinetic term is `p*p/(2*momentum_denominator)`, and all momentum kicks,
  including grouped Sexton--Weingarten kicks, use the same denominator.
- Keep the default denominator at one for exact compatibility with the
  historical Gaugefields/LTK convention. A denominator of two together with
  Gaussian coefficient width `sqrt(2)` implements the Grid/Bridge++
  convention without CPU, MPI, or GPU-specific branches.
- Verify that the two conventions give identical discrete MD trajectories,
  momenta, Hamiltonians, and `delta_hamiltonian` after the corresponding
  `1/sqrt(2)` MD-time conversion.

## v1.1.3

### Stout smearing

- Add the public `construct_Λmatrix_forSTOUT!` compatibility API for legacy
  direct-CUDA accelerator and `Gaugefields_4D_MPILattice` fields. The wrappers
  reuse the active `calc_dSdQ!`/`calc_dSdΩ!` pullback path, including
  LatticeMatrices' matrix-exponential pullback, instead of maintaining a
  separate SU(3) derivative implementation.
- Correct the legacy accelerator matrix-exponential pullback at `Q = 0`, where
  the Frechet derivative is the identity map and must copy the incoming
  cotangent.

### Configuration I/O

- Add a bulk ILDG loader for `Gaugefields_4D_nowing` that uses the existing
  contiguous local-volume reader and avoids an extra permuted allocation,
  while preserving the generic scalar fallback for fields with wings or
  halos.

### Normalization

- Export `normalize_U!` from the public API and add a
  `Gaugefields_4D_MPILattice` wrapper that normalizes its LatticeMatrices
  storage and refreshes the wing/halo data.

### Validation

- Compare the legacy CUDA STOUT pullback with its CPU implementation for zero
  and nonzero `Q` in `ComplexF64` and `ComplexF32`, including direct execution
  on an NVIDIA H100.

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
