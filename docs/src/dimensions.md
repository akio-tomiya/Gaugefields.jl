# Two and three dimensions

The recommended API uses the number of lattice extents to determine the
dimension. Apart from the vector length and logical link-field size, the API is
the same as in the [four-dimensional tutorial](tutorial4d.md).

Initialize JACC before Gaugefields so that these examples remain portable to
GPU backends:

```julia
import JACC
JACC.@init_backend

using Gaugefields
```

## Two dimensions

```julia
U2 = gauge_configuration(
    (32, 48);
    colors=3,
    start=:hot,
    seed=1234,
)

@assert length(U2) == 2
@assert gauge_lattice_size(U2) == (32, 48)
@assert gauge_halo_width(U2) == 1
@assert size(U2[1]) == (3, 3, 32, 48)
```

`U2[1]` and `U2[2]` contain the links in the first and second directions.
The Polyakov loop is measured in the second, final direction. With MPI and no
explicit process grid, the default decomposition is `(1, nranks)`.

## Three dimensions

```julia
U3 = gauge_configuration(
    (24, 24, 48);
    colors=3,
    start=:hot,
    seed=1234,
)

@assert length(U3) == 3
@assert gauge_lattice_size(U3) == (24, 24, 48)
@assert gauge_halo_width(U3) == 1
@assert size(U3[1]) == (3, 3, 24, 24, 48)
```

`U3[1]`, `U3[2]`, and `U3[3]` contain the links in the three directions. The
Polyakov loop is measured in the third, final direction. With MPI and no
explicit process grid, the default decomposition is `(1, 1, nranks)`.

The LM backend supports the same reproducible hot starts, measurements,
heatbath updates, gradient flow, stout smearing, I/O, GPU execution, and MPI
domain decomposition in 2D and 3D. The compatibility `LegacyBackend` has
additional historical restrictions; in particular, its 3D configuration must
be requested with `halo=0`.

