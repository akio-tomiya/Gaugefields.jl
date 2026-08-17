# MPI, GPU, and multi-GPU execution

Gaugefields v1 uses the LatticeMatrices/JACC backend for distributed fields.
There is no separate MPI constructor and no accelerator flag in the v1 API.
The simulation source is selected by `process_grid`; JACC determines where its
arrays and kernels run.

## Complete MPI example

The application environment must list MPI and JACC directly:

~~~julia
pkg> add Gaugefields JACC MPI
~~~

Save the following as `mpi_example.jl`:

~~~julia
using MPI

import JACC
JACC.@init_backend

using Gaugefields

MPI.Init()

world = MPI.COMM_WORLD
nranks = MPI.Comm_size(world)

lattice = (8, 8, 8, 8 * nranks)
process_grid = (1, 1, 1, nranks)

U = gauge_configuration(
    lattice;
    colors=3,
    halo=1,
    start=:hot,
    seed=0x1234,
    process_grid=process_grid,
)

comm = gauge_communicator(U)
rank = MPI.Comm_rank(comm)

plaq = measure_plaquette(U)
rank == 0 && println("plaquette = ", plaq)

flow = gradient_flow(U; steps=2, step_size=0.01)
flow!(U, flow)

rank == 0 && println("flowed plaquette = ", measure_plaquette(U))

MPI.Barrier(comm)
MPI.Finalize()
~~~

The process grid has one entry per lattice direction. It must satisfy:

- `prod(process_grid) == nranks`;
- every global extent is divisible by the corresponding grid entry;
- local extents must be large enough for the selected halo and operation.

When `process_grid` is omitted in 4D, the default decomposition places all
ranks in the final direction.

## CPU execution

Select the threads backend once, then restart Julia:

~~~julia
import JACC
JACC.set_backend("threads")
~~~

Launch MPI with the desired threads per rank:

~~~bash
mpiexec -n 4 julia --threads=4 --project=. mpi_example.jl
~~~

## One GPU

Select the backend once and restart Julia:

~~~julia
import JACC
JACC.set_backend("cuda")   # NVIDIA
# JACC.set_backend("amdgpu") # AMD
# JACC.set_backend("oneapi") # Intel
~~~

Run the same Julia source with one process. No Gaugefields GPU keyword is
needed.

## Multiple GPUs

Launch one MPI rank per GPU with the same `mpi_example.jl`:

~~~bash
mpiexec -n 4 julia --project=. mpi_example.jl
~~~

LatticeMatrices maps node-local ranks to visible devices. The number of ranks
on a node must not exceed the GPUs visible to those ranks. Scheduler resource
flags are system-specific, but `process_grid` remains the only decomposition
setting in Gaugefields code.

Direct device-buffer communication requires an MPI installation built for the
selected GPU runtime. In particular, CUDA multi-GPU runs require CUDA-aware
MPI, and MPI.jl's `libmpi` must match the `mpiexec` used to launch the job.

## Portable algorithms

The following v1 interfaces use the same LM configuration on CPU, GPU, MPI,
and multi-GPU:

- `measure_plaquette` and `measure_polyakov_loop`;
- `gradient_flow` and `flow!`;
- `heatbath_updater`, `heatbath!`, and `overrelaxation!`;
- `stout_smearing` and `smear`;
- `gaussian_momenta`, `md_driver`, and `md_trajectory!`;
- `save_configuration` and in-place `load_configuration!` where supported by
  the selected file format.

All ranks must execute collective algorithms in the same order.

## Reproducibility across decompositions

An explicit `seed` uses global-site streams for LM hot starts, Gaussian
momenta, heatbath, and overrelaxation. Keeping the seed, algorithm, and logical
sweep counter fixed makes these streams independent of the MPI process-grid
decomposition.

A complete HMC chain must additionally synchronize or broadcast its
Metropolis decision. See [Randomness and reproducibility](randomness.md) and
[HMC and custom integrators](hmc.md).

The pre-v1 `mpi`, `PEs`, `isMPILattice`, `cuda`, and `accelerator` constructor
flags are documented only in [Legacy API](legacyapi.md).
