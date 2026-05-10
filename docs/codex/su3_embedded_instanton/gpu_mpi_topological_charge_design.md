# GPU and MPI topological charge design note

This note records the current boundary for `topological_charge_density` and
`topological_charge` after public `method=:plaquette`, `method=:clover`, and
`method=:improved` support was added for serial 4D gauge fields.

It does not propose an implementation PR yet. The goal is to avoid accidentally
inferring GPU or MPI correctness from initialization support.

## Current public contract

For serial 4D `Gaugefields_4D_nowing` and `Gaugefields_4D_wing` fields, the
public API supports:

```julia
q = topological_charge_density(U; method=:plaquette)
Q = topological_charge(U; method=:plaquette)

q = topological_charge_density(U; method=:clover)
Q = topological_charge(U; method=:clover)

q = topological_charge_density(U; method=:improved)
Q = topological_charge(U; method=:improved)
```

For every supported public method, `Q == sum(q)` within the tested tolerance.
The density `q` is a host `Array` indexed by the site where the loop set is
evaluated. `method=:rect` remains private.

The public helpers deliberately reject non-serial storage with:

```julia
ArgumentError("topological charge only supports serial 4D gauge fields")
```

That guard is part of the current stable behavior.

## Why initialization support is not enough

Instanton initialization and topological-charge measurement exercise different
parts of Gaugefields.jl.

Initialization writes gauge links into a field layout. In contrast, density
measurement:

- builds temporary gauge fields for loop products,
- evaluates plaquette, clover, or rectangle loop operators,
- reads site-wise matrix elements into a host `Array`,
- assumes a single global lattice indexing convention,
- defines the scalar charge as a site-wise sum over that density.

An accelerator or MPI-backed field can exist without making all of those
measurement assumptions valid. Therefore GPU/MPI support should not be enabled
by relaxing the existing serial-type guard.

## GPU design choices

A future GPU PR should choose one explicit strategy:

1. Host fallback:
   - copy or materialize a serial host field,
   - reuse the current serial density implementation,
   - return a host `Array`,
   - document the data movement and cost.

2. Native accelerator implementation:
   - implement loop evaluation and density accumulation on the accelerator,
   - decide whether `topological_charge_density` returns a host `Array` or an
     accelerator array,
   - preserve `topological_charge(U; method) == sum(topological_charge_density(U; method))`
     or document any necessary reduction semantics,
   - compare against serial results on small lattices.

The host fallback is probably safer for a first experimental GPU PR, but it
should still be explicit and tested. Silent fallback hidden inside the current
serial method would make performance and memory behavior hard for users to
understand.

## MPI design choices

MPI support has a separate density question and scalar question.

For scalar `Q`, a future implementation could compute local contributions and
reduce them across ranks. For density `q(x)`, the API must decide whether it
returns:

- only each rank's local density,
- a gathered global host `Array` on one rank,
- a distributed object matching the MPI field layout,
- or a pair of local density plus metadata.

The current serial API returns a global host `Array`, so changing that behavior
for MPI would need explicit documentation. Halo and boundary treatment also
must be tested because clover and rectangle loops cross local subdomain
boundaries.

## Test plan before enabling GPU/MPI

Any implementation PR should include small, deterministic tests before touching
large examples:

- serial-vs-GPU agreement for cold fields and small instantons,
- `sum(q) == Q` for every supported method,
- block and sign invariance for `Oneinstanton_SUN_embedded`,
- explicit tests that unsupported storage still throws until enabled,
- MPI scalar reduction agrees with the serial scalar on one-rank and multi-rank
  layouts,
- MPI density shape and site ordering are documented and tested,
- boundary-crossing loops agree with serial results on a small decomposed
  lattice.

CI availability may limit GPU/MPI automation, but local or documented manual
test commands should still be part of the PR that enables support.

## Non-goals

This note does not:

- expose `method=:rect`,
- change the public return type of `topological_charge_density`,
- add GPU or MPI implementations,
- change instanton initialization behavior,
- add VisualizingLQCD integration.

## Suggested follow-up sequence

1. Add explicit guard tests for MPI-backed fields if a lightweight fixture is
   available.
2. Prototype a host fallback for accelerator fields in a private helper, then
   compare against serial fields.
3. Design MPI scalar `Q` reduction independently from density gathering.
4. Only after the return type and ordering are pinned, expose public GPU/MPI
   measurement support.
5. Connect VisualizingLQCD to the serial public density API first, then decide
   whether it needs gathered MPI density.
