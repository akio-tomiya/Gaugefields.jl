# SU(Nc)-Embedded Instanton Usage and Clover Density Design

This note is a docs-only follow-up to the SU(Nc)-embedded SU(2) instanton work.
It records the next small step before implementing `method=:clover` support for
`topological_charge_density` and `topological_charge`.

This PR should not change package behavior. In particular, it should not enable
`Oneinstanton(3, ...)`, should not alter `Oneinstanton_SUN_embedded`, and should
not change the existing `method=:plaquette` result.

## Current state

The merged public instanton entry point is explicit:

```julia
U = Oneinstanton_SUN_embedded(
    NC, NX, NY, NZ, NT;
    NDW=0,
    center=(NX / 2 + 0.5, NY / 2 + 0.5, NZ / 2 + 0.5, NT / 2 + 0.5),
    radius=div(NX, 2),
    sign=+1,
    block=(1, 2),
)
```

This keeps the old `Oneinstanton` constructor unchanged and makes SU(Nc)
embedding opt-in. The implementation supports serial 4D `Gaugefields_4D_nowing`
and `Gaugefields_4D_wing` fields through the same explicit API.

The merged topological-charge API is:

```julia
q = topological_charge_density(U; method=:plaquette)
Q = topological_charge(U; method=:plaquette)
```

with `Q == sum(q)` by construction. The API currently accepts only
`method=:plaquette`; `method=:clover` throws a clear `ArgumentError`.

## Usage example for the embedded instanton

The safe user-facing pattern is to construct the embedded field explicitly and
then measure it with the current plaquette method:

```julia
using Gaugefields

L = (4, 4, 4, 4)
U2 = Oneinstanton_SUN_embedded(2, L...; block=(1, 2))
U3 = Oneinstanton_SUN_embedded(3, L...; block=(1, 2))
U3_alt = Oneinstanton_SUN_embedded(3, L...; block=(2, 3))

Q2 = topological_charge(U2; method=:plaquette)
Q3 = topological_charge(U3; method=:plaquette)
q3 = topological_charge_density(U3; method=:plaquette)

@assert isapprox(Q3, Q2; rtol=1e-12, atol=1e-12)
@assert isapprox(topological_charge(U3_alt; method=:plaquette), Q3;
                 rtol=1e-12, atol=1e-12)
@assert isapprox(sum(q3), Q3; rtol=1e-12, atol=1e-12)
```

The `block` keyword selects the embedded SU(2) subgroup. For SU(3), the useful
choices are `(1, 2)`, `(1, 3)`, and `(2, 3)`. For SU(Nc), `block=(i, j)` should
satisfy `1 <= i < j <= NC`.

The `sign` keyword flips instanton versus anti-instanton orientation:

```julia
U_plus = Oneinstanton_SUN_embedded(3, L...; sign=+1)
U_minus = Oneinstanton_SUN_embedded(3, L...; sign=-1)

@assert isapprox(topological_charge(U_minus; method=:plaquette),
                 -topological_charge(U_plus; method=:plaquette);
                 rtol=1e-12, atol=1e-12)
```

## Clover density API target

The intended public surface should extend the existing keyword, not add a new
function:

```julia
q = topological_charge_density(U; method=:clover)
Q = topological_charge(U; method=:clover)
```

The required invariant is the same as for the plaquette method:

```julia
sum(topological_charge_density(U; method=:clover)) ==
    topological_charge(U; method=:clover)
```

The first implementation should keep the existing serial 4D guard. GPU, MPI,
and accelerator fields are outside the first `method=:clover` density scope
because the density routine builds temporary gauge fields, evaluates loops, and
then reads site-wise matrix elements on the host. The fact that an instanton is
an initialization path does not by itself make the measurement path safe for
GPU or MPI storage.

## Clover construction

The existing sample scalar measurement in
`samples/measurements/topologicalcharge.jl` computes clover topological charge
by:

- building four clover loops for each ordered pair `(mu, nu)`,
- evaluating their sum,
- projecting with `Traceless_antihermitian!`,
- contracting with the 4D epsilon tensor,
- dividing by `numofloops^2`, where `numofloops == 4`.

The density implementation should follow that scalar convention so that a
site-wise sum reproduces the existing sample scalar result.

A private implementation sketch:

```julia
function _clover_field_strengths(U)
    temps = [similar(U[1]), similar(U[1]), similar(U[1]), similar(U[1])]
    F = Matrix{eltype(U)}(undef, 4, 4)

    for mu = 1:4, nu = 1:4
        F[mu, nu] = similar(U[1])
        if mu != nu
            loops = make_cloverloops(mu, nu, Dim=4)
            evaluate_gaugelinks!(temps[1], loops, U, temps)
            Traceless_antihermitian!(F[mu, nu], temps[1])
        end
    end

    return F
end
```

Then, if `F[mu, nu]` stores the unnormalized sum of the four clover loops:

```julia
q[x] = -real(sum(epsilon(mu, nu, rho, sigma) *
                 tr_site(F[mu, nu], F[rho, sigma], x))) /
       (32 * pi^2 * 4^2)
```

Equivalently, the implementation can scale each `F[mu, nu]` by `1 / 4` and then
reuse the plaquette normalization. The test should pin the chosen convention to
the existing sample scalar calculation.

Do not reuse `make_Cloverloopterms!` blindly for this measurement. That helper
is for Wilson clover terms and currently uses `Antihermitian!`; the topological
charge sample uses `Traceless_antihermitian!`.

## Density location convention

`topological_charge_density` should continue returning a plain
`Array{Float64,4}` with shape `(NX, NY, NZ, NT)`.

For the plaquette method, the local operator is anchored at a site. For the
clover method, the operator is more symmetric around the site, but the returned
array should still be indexed by the lattice site where the loop set is
evaluated. Gaugefields.jl should not bake in visualization choices such as
slicing, interpolation, color scaling, or smoothing.

## Tests for the implementation PR

The docs-only PR does not need runtime tests. The implementation PR that enables
`method=:clover` should add focused tests:

- cold serial 4D fields have zero clover density and zero scalar charge,
- `sum(topological_charge_density(U; method=:clover))` matches
  `topological_charge(U; method=:clover)`,
- the public clover scalar matches the existing sample scalar formula on a
  small field,
- SU(2) and SU(Nc)-embedded instantons have matching clover scalar charge,
- changing `block` does not change the scalar charge,
- `sign=-1` flips the scalar charge,
- `method=:plaquette` tests remain unchanged,
- unsupported dimensions and non-serial storage keep throwing clear errors.

Suggested focused command:

```sh
julia --project=. test/sun_embedded_instanton.jl
```

Before merging the implementation PR, also run:

```sh
julia --project=. test/runtests.jl
```

## Suggested next PRs

1. This docs-only PR: document embedded-instanton usage and the clover-density
   design.
2. Add private clover field-strength and density helpers, leaving the public
   `method=:clover` error in place if useful for review size.
3. Route `topological_charge_density(U; method=:clover)` and
   `topological_charge(U; method=:clover)` to the clover helpers, with the
   focused tests above.
4. Revisit examples or manual docs after the implementation is merged.
