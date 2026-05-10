# Topological Charge Method Roadmap

This note is a docs-only checkpoint after the SU(Nc)-embedded instanton,
plaquette density, clover density, rectangle-density helper, and improved
topological-charge PRs. It keeps the next measurement work separate from the
already-merged public behavior.

## Current supported surface

For serial 4D gauge fields, the public API is:

```julia
q = topological_charge_density(U; method=:plaquette)
Q = topological_charge(U; method=:plaquette)

q = topological_charge_density(U; method=:clover)
Q = topological_charge(U; method=:clover)

q = topological_charge_density(U; method=:improved)
Q = topological_charge(U; method=:improved)
```

`method=:plaquette` remains the default. For each supported method, the scalar
charge is defined as the site-wise sum:

```julia
topological_charge(U; method=method) ==
    sum(topological_charge_density(U; method=method))
```

`method=:improved` is defined as the tested linear combination of the clover
and rectangle densities. `method=:rect` remains private and should keep
throwing `ArgumentError` until there is a clear public use case.

## Improved Method And Rectangle Candidate

The existing sample measurement code contains the improved scalar convention
now matched by the public `method=:improved` helpers:

```julia
Qrect = 2 * calc_Q(UmunuTA, numofloops, U)
Qimproved = (5 / 3) * Qclover - (1 / 12) * Qrect
```

The corresponding public density preserves the same normalization:

```julia
q_improved(x) = (5 / 3) * q_clover(x) - (1 / 12) * q_rect(x)
sum(q_improved) == Qimproved
```

The rectangle density is still treated as the risky piece and remains private.
It should only be exposed as `method=:rect` if a later PR identifies a clear
user-facing reason and keeps the scalar/density normalization pinned by tests.

## Suggested small PR sequence

1. Done: add a private rectangle-density helper and focused tests against the
   existing sample scalar formula. Do not expose a public method yet.
2. Done: add a private improved-density helper as a linear combination of the clover
   and rectangle densities. Test `sum(q_improved) == Qimproved`.
3. Done: route `method=:improved` publicly only after the private helper is pinned.
   Keep `method=:rect` private unless there is a clear user-facing reason to
   expose it.
4. Done: add short manual docs for `method=:improved` after the API is public.
5. Done: treat GPU/MPI support as a separate design topic before
   implementation. See `gpu_mpi_topological_charge_design.md`.

## Tests for rectangle and improved density

Use focused serial 4D tests before touching the full suite:

- cold fields have zero rectangle and improved density,
- `sum(q_rect)` matches the sample scalar `Qrect`,
- `sum(q_improved)` matches the sample scalar `Qimproved`,
- SU(2) instantons and SU(Nc)-embedded instantons agree for the same method,
- changing `block` does not change scalar charge,
- `sign=-1` flips scalar charge,
- existing plaquette and clover tests remain unchanged,
- `method=:rect` still throws until deliberately enabled.

Before merging an implementation PR, run:

```sh
julia --project=. test/sun_embedded_instanton.jl
julia --project=. test/runtests.jl
```

## GPU and MPI notes

The instanton initialization path and the topological-charge measurement path
should be judged separately. Even if an initialized field can exist on an
accelerator or MPI-backed storage type, the current density implementation
builds temporary gauge fields, evaluates loop operators, and reads site-wise
matrix elements into a host `Array`.

That means GPU and MPI support should not be inferred from initialization
support. A later design should decide:

- whether density is computed locally and then gathered,
- how boundary/halo data is represented in site-wise `q(x)`,
- whether the public return type should remain a host `Array`,
- how to test MPI reductions of scalar `Q` independently from local density,
- whether accelerator fields need a host fallback or native kernels.

Until that design exists, serial 4D support is the stable public contract for
`topological_charge_density` and `topological_charge`.

The more detailed checkpoint is in `gpu_mpi_topological_charge_design.md`.

## Remaining risks

- Rectangle-loop normalization is easy to get wrong because the sample scalar
  includes both `numofloops` and an extra factor of `2`.
- The density location convention should remain "indexed by the site where the
  loop set is evaluated"; visualization should not reinterpret this inside
  Gaugefields.jl.
- Coarse instanton tests should check method consistency and sign/block
  invariants, not claim continuum integer charge too strongly.
- Additional public methods should only be enabled after both scalar and density
  tests are in place.
