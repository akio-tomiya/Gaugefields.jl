# Pre-v1 samples

The scripts in this directory use the API of Gaugefields before v1
(`Initialize_Gaugefields`, `calculate_Plaquette`, `Gradientflow`,
`CovNeuralnet`, the `mpi=true` and `cuda=true` constructor flags, and
hand-written `MDstep!`, `U_update!`, and `P_update!` functions). They are kept
for reference and for comparison with existing programs; they are not
maintained.

Gaugefields v1 keeps these functions as a compatibility API. With the
registered Gaugefields v1.1.13 in one CPU process, the following scripts run
unchanged:

- `mpigradientflow.jl` with the arguments `1 1 1 1 false` or `1 1 1 1 true`
- `mpigradient_general.jl 1 1 1 1 false`
- `mpihmctest.jl 1 1 1 1 false`

The other scripts do not run as they are. `scalarnn.jl` uses `ScalarNN`,
which was renamed `GaugeAction` before v1. `stouttest.jl` calls
`CovNeuralnet()` without a configuration, which was removed before v1; use
`CovNeuralnet(U)`. `gaugefieldreadme.jl` calls `Gradientflow_general(U, B, ...)`,
which is now `Gradientflow_general_Bfields`. `samplehmc_ILDG.jl` reads
`conf_00000008.ildg`, which is not in this repository. The CUDA scripts and
the scripts that need several MPI processes (`mpiscalarnn.jl`,
`mpiscalarnn_HMC.jl`) were not run.

For new code, start from the v1 samples in the parent directory and the
[four-dimensional quick start](../../docs/src/tutorial4d.md). The migration map
on the [Legacy API](../../docs/src/legacyapi.md#Migration-map) page lists the
v1 replacement of each pre-v1 function.
