# Higher-form B fields

Gaugefields represents a four-dimensional ``Z(N_c)`` two-form field as one
center-valued matrix field for every oriented plane,

```math
B_{\mu\nu}(x)=z_{\mu\nu}(x)I, \qquad z_{\mu\nu}(x)\in Z(N_c).
```

The B field multiplies the Wilson paths used by a `GaugeAction`. Gauge-link
values U and B-field values may evolve during a simulation, while the
geometry of a fixed `Wilsonline` remains unchanged.

## Initialize a B field and action

The compatibility constructor creates a four-dimensional gauge field and a B
field carrying the requested 't Hooft flux:

~~~julia
using Gaugefields

NC = 2
NDW = 0
NX, NY, NZ, NT = 4, 4, 4, 4
flux = [1, 0, 1, 1, 0, 1]

U = Initialize_Gaugefields(
    NC, NDW, NX, NY, NZ, NT;
    condition="hot",
    randomnumber="Reproducible",
    verbose_level=0,
)

B = Initialize_Bfields(
    NC, flux, NDW, NX, NY, NZ, NT;
    condition="tflux",
    verbose_level=0,
)

plaquettes = make_loops_fromname("plaquette"; Dim=4)
append!(plaquettes, plaquettes')

action = GaugeAction(U, B)
push!(action, 1.0, plaquettes)

value = evaluate_GaugeAction(action, U, B)
dSdU1 = calc_dSdUμ(action, 1, U, B)
~~~

`flux` lists the six planes in the order `(1,2)`, `(1,3)`, `(1,4)`, `(2,3)`,
`(2,4)`, `(3,4)`. Flux values are understood modulo `NC`.

## Plaquette measurement and molecular dynamics

Measure the unnormalized real plaquette sum without changing U or B:

~~~julia
temps = [similar(U[1]), similar(U[1])]
plaquette_sum = calculate_Plaquette(U, B, temps)
# For the action above (unit coefficient, both orientations):
@assert isapprox(plaquette_sum, real(evaluate_GaugeAction(action, U, B)) / 2)
~~~

For a normalized plaquette, divide by `6 * NC * NX * NY * NZ * NT`.
The old B-dependent measurement overwrote gauge links and used the wrong
reverse-orientation sign. These bugs are fixed in every evaluation mode,
including `:legacy`; reproducing the erroneous measurement is not supported.
The serial wing constructors (`NDW > 0`) also now resolve the `tflux` and
`tloop` initialization functions correctly.

`GaugeAction(U, B)` does **not** store B. To include B in both the MD
potential and U force, pass it explicitly:

~~~julia
driver = md_driver(U, action, B; steps=10, trajectory_length=0.1)
p = initialize_TA_Gaugefields(U)
gauss_distribution!(p)
result = md_trajectory!(U, p, driver)

# Equivalent explicit action provider, also usable inside MDActionSet:
bound_action = BfieldGaugeAction(action, B)
driver = md_driver(U, bound_action; steps=10, trajectory_length=0.1)
~~~

The two-argument `md_driver(U, action; ...)` with a plain `GaugeAction`
still represents a B-free action. The wrapper holds B by reference, so
`substitute_U!(B, Bnew)` is seen by an existing driver; rebinding a local
variable with `B = Bnew` does not change that reference. The driver evolves
U, not B. A dynamical-B proposal and its acceptance/rejection rule remain
the caller's responsibility. The trajectory call itself is MD, not a full
Metropolis HMC update. Fixed-B time reversibility is tested; this is not a
validation of an arbitrary combined U/B sampling algorithm.

The plaquette-only flow also accepts B:

~~~julia
g = Gradientflow(U; eps=0.01, Nflow=1)
flow!(U, B, g)
~~~

This now uses B in its force and agrees with
`Gradientflow_general_Bfields(U, B, ["plaquette"], [1.0]; eps=0.01, Nflow=1)`.
For other loop actions, use `Gradientflow_general_Bfields`. Flow updates U
but leaves B unchanged.

## Select the evaluation method

Gaugefields v1.1.8 provides three evaluation choices. They initialize the
same B values and differ only in how Wilson-path B factors are evaluated.

| Initialization keyword | Path geometry | B multiplication |
|---|---|---|
| default | cached | center phase on supported fields |
| `use_center_fastpath=false` | cached | full color matrix |
| `bfield_evaluation=:legacy` | recomputed for every link | full color matrix |

The current default is:

~~~julia
B = Initialize_Bfields(
    NC, flux, NDW, NX, NY, NZ, NT;
    condition="tflux",
)
~~~

To isolate the path-cache optimization while retaining the previous matrix
multiplication, use:

~~~julia
B_cached_fullmatrix = Initialize_Bfields(
    NC, flux, NDW, NX, NY, NZ, NT;
    condition="tflux",
    use_center_fastpath=false,
)
~~~

To reproduce the complete pre-v1.1.8 evaluation, including repeated path
construction, use:

~~~julia
B_legacy = Initialize_Bfields(
    NC, flux, NDW, NX, NY, NZ, NT;
    condition="tflux",
    bfield_evaluation=:legacy,
)
~~~

The legacy mode retains correctness fixes made around the evaluator; it
selects the old numerical calculation and performance behavior rather than
reintroducing earlier runtime errors.

The center-phase specialization currently applies to serial
`Gaugefields_4D_nowing` fields. Other backends automatically use full-matrix
multiplication. They still benefit from path caching unless `:legacy` is
selected.

## Dynamical B fields

Only path geometry is cached. No U values, B values, fluxes, or center phases
are stored in a path plan. Replacing B therefore does not invalidate the
cache:

~~~julia
new_flux = [0, 1, 0, 1, 0, 1]
Bnew = Initialize_Bfields(
    NC, new_flux, NDW, NX, NY, NZ, NT;
    condition="tflux",
)

substitute_U!(B, Bnew)
value_after_update = evaluate_GaugeAction(action, U, B)
~~~

The destination `B` keeps its selected evaluation mode. Call
`clear_Bpath_cache!(B)` only after mutating the links, order, directions, or
positions of a previously evaluated `Wilsonline` object. A newly constructed
`Wilsonline` receives a new cache entry automatically.

## Numerical equivalence and performance

The optimized and legacy calculations were compared for SU(2), SU(3), and
SU(4), including all 36 plaquette/rectangle paths, two gauge actions, all four
force directions, and evaluation after changing B. The optimized results and
the explicit `:legacy` results were bitwise identical to the pre-v1.1.8 code;
all measured maximum absolute differences were `0.0`.

That equivalence concerns Wilson-path/action/force evaluation, not the
previously incorrect `calculate_Plaquette(U, B, ...)` measurement. The fixed
measurement is checked against an independent site-by-site plaquette sum
for SU(2) and SU(3), with and without serial wings and for both `tflux` and
`tloop`. B-dependent MD potential, force, in-place B updates, and PQP/QPQ
time reversibility are also covered by regression tests.

For a representative single-thread Julia 1.11.8 run on a ``4^4`` SU(2)
lattice, the v1.1.8 default gave the following median speedups over v1.1.7:

| Calculation | Speedup |
|---|---:|
| B plaquette path | 20.50x |
| B rectangle path | 18.03x |
| Plaquette action | 2.31x |
| Plaquette + rectangle action | 2.60x |
| Mixed-action force, direction 1 | 2.92x |

Absolute timings depend on the machine, lattice, action, and backend. See the
[release changelog](https://github.com/akio-tomiya/Gaugefields.jl/blob/master/changes.md)
for the measured times and allocation reductions.
