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

### U(1) fields

Set `colors=1` for a two-dimensional U(1) field. General Wilson-loop actions
use the same interface:

```julia
U1 = gauge_configuration(
    (32, 48);
    colors=1,
    start=:hot,
    seed=1234,
)

loops = make_loops_fromname("plaquette"; Dim=2)
append!(loops, loops')

action = GaugeAction(U1)
push!(action, 1.0, loops)
action_value = evaluate_GaugeAction(action, U1)
```

See [Wilson loops and gauge actions](wilsonloops_actions.md) for custom U(1)
paths and general-action gradient flow.

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

## Special initial configurations

The high-level `gauge_configuration` constructor deliberately limits `start`
to `:cold` and `:hot`. Gaugefields also retains the specialized
`Oneinstanton` and `Oneinstanton_SUN_embedded` constructors. Their LM forms
use `isMPILattice=true`, a positive halo, and `PEs` for the process grid:

```julia
instanton2 = Oneinstanton(
    2,
    1,
    32,
    48;
    isMPILattice=true,
    PEs=(1, 1),
    verbose_level=0,
)

embedded4 = Oneinstanton_SUN_embedded(
    3,
    8,
    8,
    8,
    16;
    NDW=1,
    isMPILattice=true,
    PEs=(1, 1, 1, 1),
    verbose_level=0,
)
```

For MPI, the product of `PEs` must equal the number of ranks. These names are
specialized compatibility constructors rather than new `start` choices; their
full parameter history is kept in [Legacy API](legacyapi.md).
