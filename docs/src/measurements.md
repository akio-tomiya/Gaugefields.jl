# Measurements

Gaugefields v1 provides normalized convenience functions for the two common
configuration-level observables.

## Plaquette

~~~julia
plaq = measure_plaquette(U)
~~~

The default normalization is

~~~math
\frac{1}{\binom{D}{2}\,V\,N_c}
\sum_{n,\mu<\nu}\operatorname{ReTr} U_{\mu\nu}(n),
~~~

so a cold configuration returns one in two, three, and four dimensions.

Use `normalize=false` only when the historical summed convention is required:

~~~julia
plaquette_sum = measure_plaquette(U; normalize=false)
~~~

## Polyakov loop

~~~julia
polyakov = measure_polyakov_loop(U)
~~~

The loop is taken in the final lattice direction: y in 2D, t in the package's
3D convention, and t in 4D. The default divides the color trace by `N_c`, so a
cold configuration returns one.

~~~julia
polyakov_trace = measure_polyakov_loop(U; normalize=false)
~~~

## Inspecting geometry and normalization

Use public metadata rather than backend fields:

~~~julia
D = length(U)
lattice = gauge_lattice_size(U)
volume = prod(lattice)
colors = gauge_num_colors(U)
~~~

Both measurement functions accept a configuration created by
`gauge_configuration` with either `LatticeMatricesBackend()` or the explicit
`LegacyBackend()` compatibility selector.

## MPI behavior

On an LM process grid, measurements perform the required global reduction.
Every rank in `gauge_communicator(U)` must call the measurement in the same
order. Usually only rank zero prints the returned value:

~~~julia
using MPI

plaq = measure_plaquette(U)
comm = gauge_communicator(U)
MPI.Comm_rank(comm) == 0 && println("plaquette = ", plaq)
~~~

## Advanced observables

The v1 convenience layer currently exposes plaquette and Polyakov loop
measurements. Energy-density, clover, and topological-charge routines remain
available through the compatibility-level field operations but do not yet
have backend-neutral v1 wrappers. Their historical definitions and examples
are kept in [Legacy API](legacyapi.md), rather than mixing them into the v1
measurement interface.
