# Deprecated extension implementations

This directory contains compatibility-only implementations that are scheduled
for removal in a future breaking release.

New MPI code belongs in `ext/GaugefieldsMPIExt.jl` and must use the portable
LatticeMatrices communication path. Do not add new features to the legacy MPI
implementations under `mpi/`.
