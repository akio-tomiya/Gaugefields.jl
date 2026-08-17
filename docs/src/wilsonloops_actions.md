# Wilson loops and gauge actions

Gaugefields uses [Wilsonloop.jl](https://github.com/akio-tomiya/Wilsonloop.jl)
to describe paths. Named loop sets are sufficient for common actions, while a
`Wilsonline` can represent any explicit path.

## Named loop sets

`make_loops_fromname` is re-exported by Gaugefields. Always pass the dimension
when code may be used outside four dimensions:

~~~julia
import JACC
JACC.@init_backend

using Gaugefields

U = gauge_configuration(
    (8, 8, 8, 16);
    colors=3,
    halo=1,
    start=:cold,
)

Dim = length(U)
plaquettes = make_loops_fromname("plaquette"; Dim)
rectangles = make_loops_fromname("rectangular"; Dim)
~~~

The result is a vector of closed Wilson paths. Taking its adjoint reverses and
conjugates every path:

~~~julia
append!(plaquettes, plaquettes')
~~~

Include both orientations, as above, when a loop set is not already closed
under Hermitian conjugation and the resulting action must be real.

## Define an arbitrary path

Applications that construct `Wilsonline` objects directly should add
Wilsonloop as a direct dependency:

~~~julia
pkg> add Wilsonloop
~~~

A signed pair `(direction, distance)` moves forward for a positive distance
and backward for a negative distance. For example, the following is a
plaquette in the first and second directions:

~~~julia
using Wilsonloop

path = [(1, 1), (2, 1), (1, -1), (2, -1)]
w = Wilsonline(path; Dim=length(U))
~~~

Directions are numbered from `1` through `length(U)`. A distance may have
magnitude greater than one, so rectangles and longer closed paths use the same
representation.

## Evaluate a Wilson path

`evaluate_gaugelinks!` writes the ordered matrix product at every lattice site
to an explicitly supplied link field:

~~~julia
using LinearAlgebra

W = similar(U[1])
temps = [similar(U[1]) for _ in 1:4]
evaluate_gaugelinks!(W, w, U, temps)

wilson_sum = tr(W)
~~~

`tr(W)` includes the color trace and lattice sum. On an MPI LM configuration
it also performs the required global reduction. All ranks must call it in the
same order.

A vector of paths can be evaluated in one call. Their matrix-valued results
are added into the destination:

~~~julia
closed_paths = [w, w']
evaluate_gaugelinks!(W, closed_paths, U, temps)
~~~

## Build and evaluate a general action

Construct an empty action, then add one coefficient and one closed-path group
at a time:

~~~julia
beta = 6.0
action = GaugeAction(U)
closed_paths = [w, w']
push!(action, beta / 2, closed_paths)

loop_sum = evaluate_GaugeAction(action, U)
potential = -real(loop_sum) / gauge_num_colors(U)
~~~

`evaluate_GaugeAction` returns the coefficient-weighted, color-traced, and
lattice-summed loop expression. The final line shows the sign and
normalization used when a `GaugeAction` is passed to `md_driver`.

The untraced matrix field is available in allocating and in-place forms:

~~~julia
untraced = evaluate_GaugeAction_untraced(action, U)

untraced_work = similar(U[1])
evaluate_GaugeAction_untraced!(untraced_work, action, U)
~~~

## Analytic action derivative

The allocating and in-place derivative forms operate on one direction at a
time:

~~~julia
direction = 1
dSdU = calc_dSdUμ(action, direction, U)

dSdU_work = similar(U[1])
calc_dSdUμ!(dSdU_work, action, direction, U)
~~~

This is the raw matrix derivative of the stored Wilson-loop sum. It is not yet
the traceless anti-Hermitian MD force. The built-in `GaugeAction` provider for
`md_driver` performs the link multiplication, normalization, and projection
needed by `update_momenta!`.

## Reuse the loop definition

The same `GaugeAction` can drive heatbath and molecular dynamics:

~~~julia
updater = heatbath_updater(U, action; seed=0x1234)
heatbath!(U, updater)

driver = md_driver(U, action; steps=20, integrator=QPQ())
P = gaussian_momenta(U; seed=0x5678)
md_trajectory!(U, P, driver)
~~~

General gradient flow takes loop groups and coefficients directly rather than
a `GaugeAction` object:

~~~julia
flow = gradient_flow(
    U,
    [closed_paths],
    [beta / 2];
    steps=10,
    step_size=0.01,
)
flow!(U, flow)
~~~

These interfaces use the same loop descriptions for 2D, 3D, and 4D LM
configurations. CPU, GPU, MPI, and multi-GPU execution are selected through
JACC and `process_grid`; the action definition itself does not change.

Higher-form actions that also include a `B` field remain in the
[Legacy API](legacyapi.md#Non-dynamical-higher-form-gauge-fields) compatibility
section.
