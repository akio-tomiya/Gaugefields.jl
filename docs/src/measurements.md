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

Gaugefields is the core field and update package. Higher-level observables,
including energy density, Wilson-loop measurements, topological charge, and
topological-charge-density correlations, belong in
[QCDMeasurements.jl](https://github.com/akio-tomiya/QCDMeasurements.jl).
Add it separately to an application environment:

~~~julia
pkg> add QCDMeasurements
~~~

Use a QCDMeasurements release whose compatibility bounds include Gaugefields
v1. A topological-charge measurement is constructed once and reused:

~~~julia
using QCDMeasurements

topology = Topological_charge_measurement(
    U;
    TC_methods=["plaquette", "clover"],
    verbose_level=0,
)
charges = get_value(measure(topology, U))

println("plaquette = ", charges["plaquette"])
println("clover = ", charges["clover"])
println("improved = ", charges["clover improved"])
~~~

For observables built from the site-local topological charge density, use the
QCDMeasurements density-correlation measurement rather than building a new
workflow around Gaugefields' compatibility-level `topological_charge_density`
helper:

~~~julia
density_correlation =
    QCDMeasurements.Topological_charge_density_correlation_measurement(
        U;
        TC_methods=["plaquette", "clover"],
        verbose_level=0,
    )

origin = [1, 1, 1, 1]
separation = [1, 0, 0, 0]
correlation = get_value(measure(
    density_correlation,
    U,
    origin,
    separation,
))
~~~

The measurement package owns the operator definitions, improved-charge
choices, output format, and future backend-specific measurement kernels. The
historical Gaugefields implementations remain documented in
[Legacy API](legacyapi.md#Topological-charge) only for compatibility and
comparison with existing programs.
