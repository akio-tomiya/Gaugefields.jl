# Fused MPILattice gauge-fixing regression

`gaugefixing_fused.jl` runs the same tests with the selected JACC backend and
LatticeMatrices communicator. It is included in the standard CPU tests. The
two-rank MPI CI job runs `MPIJACCtest/gaugefixing_fused.jl`.

From an environment containing this checkout, JACC, LatticeMatrices, and Test:

```sh
julia --project=ENV test/gaugefixing_fused.jl
mpiexecjl --project=ENV -n 2 julia --project=ENV test/MPIJACCtest/gaugefixing_fused.jl
```

For GPU runs, configure JACC's GPU backend before loading the package and use
the same test file. For GPU+MPI, initialize MPI and select the device for each
rank using LatticeMatrices' existing `select_device_by_mpi_rank!` before
including the test. The tests themselves contain no device-specific APIs.

The fixtures use a `6×4×4×6` lattice. Two-rank spatial
and temporal decompositions have odd local extents, exercising the global
checkerboard offset. Coverage includes Float32/Float64, SU(2)/SU(3)/SU(4),
Coulomb/Landau gauge, halo widths 0/1/2, and convergence exits. Halo-free
storage uses the existing field shifts instead of the direct-halo kernels.
Both backend-generated hot links and a common host-generated SU(N) field are
tested. The latter is built as the exponential of a small traceless
antihermitian matrix and copied unchanged to each backend, isolating update
and communication differences from hot-start orthogonalization rounding.

Checks compare the divergence with an independent dense periodic reference,
compare each fused sweep with the existing field-level implementation, verify
plaquette invariance and link unitarity, and check that returned halos equal
a fresh exchange. `FusedGaugeFixingTests.snapshots` contains the common input,
complete global final links, transformations, divergences, and scalar
diagnostics for cross-backend comparison. Initial arrays must match exactly;
final arrays use the tolerances below. All ranks must participate in collecting
these snapshots, but only rank zero should write them. The separate
`hot_snapshots` dictionary retains the rough-field results for diagnosis.

The numerical tolerances are `rtol=atol=5e-5` for Float32 and `5e-12` for
Float64. Bitwise agreement is not required between different GPU/CPU
compilers. Independently initialized hot trajectories are not expected to stay
within these tolerances: small initial rounding differences can grow during
the nonlinear updates. Hot-field regressions instead compare fused and
unfused operations on the same backend and verify gauge invariants. The
original CPU SU(3) bitwise comparison remains in
`gaugefixing_lm.jl`.
