# Document SU(Nc)-embedded SU(2) instanton plan

This is a survey/design note for a small first PR. It documents the current
`Oneinstanton` behavior and a safe path toward embedding an SU(2) instanton in
an SU(Nc) gauge field, with SU(3) as the first target. This PR should not change
package code or enable `Oneinstanton(3, ...)`.

## Existing `Oneinstanton` survey

`Oneinstanton(NC, NDW, NN...; mpi=false, PEs=nothing, mpiinit=nothing)` is
defined in `src/AbstractGaugefields.jl`. It dispatches by dimension and wing
width:

- 4D, `NDW == 0`: `Oneinstanton_4D_nowing`.
- 4D, `NDW != 0`: `Oneinstanton_4D_wing`.
- 2D, `NDW == 0`: `Oneinstanton_2D_nowing`.
- 2D, `NDW != 0`: `Oneinstanton_2D_wing`.
- MPI 4D currently dispatches to `Oneinstanton_4D_wing_mpi`.

The 4D constructors in `src/4D/nowing/gaugefields_4D_nowing.jl` and
`src/4D/wing/gaugefields_4D_wing.jl` assert `NC == 2`. They build the link
matrices directly from 2 by 2 Pauli matrices, so the current implementation is
SU(2)-specific. `test/init.jl` also tests the instanton path with `NC = 2`.

Because of this, `Oneinstanton(3, ...)` should not be enabled by changing the
existing constructor in the first implementation PR. A new explicit API is safer
for existing users.

There is also a positional-call hazard to avoid in the new API. The definition
is `Oneinstanton(NC, NDW, NN...)`, while current tests call it as
`Oneinstanton(NC, NX, NY, NZ, NT, Nwing)`. A new public constructor should use
keywords for options such as `NDW`, `center`, `radius`, `sign`, and `block`
instead of copying that positional shape.

## SU(2) link embedding into SU(Nc)

The intended construction is to create an SU(2) instanton link `u_mu(x)` and
place it into a 2 by 2 color block of an SU(Nc) identity matrix:

```julia
U_mu(x) = I_Nc
U_mu(x)[block, block] = u_mu(x)
```

For SU(3), the supported block choices should be:

- `(1, 2)`: upper-left SU(2) subgroup.
- `(1, 3)`: colors 1 and 3.
- `(2, 3)`: lower-right SU(2) subgroup.

For general SU(Nc), `block=(i, j)` should satisfy `1 <= i < j <= NC`. All
remaining spectator colors should stay exactly identity and should have zero
off-block coupling. Since the SU(2) link has determinant one, the embedded
SU(Nc) link also has determinant one.

The serial 4D gauge field storage, identity initialization, `clear_U!`,
`unit_U!`, `set_wing_U!`, `tr`, `Traceless_antihermitian!`, and plaquette path
already have `NC`-generic code paths. This makes an SU(Nc)-generic embedding
helper feasible. MPI and accelerator paths should still be out of scope for the
first implementation.

## First implementation API proposal

Add a new explicit constructor rather than changing `Oneinstanton(3, ...)`.
Possible names:

- `Oneinstanton_SUN_embedded`
- `SUNEmbeddedInstanton`
- `SUNInstantonFromSU2`

A conservative first signature could be:

```julia
Oneinstanton_SUN_embedded(
    NC, NX, NY, NZ, NT;
    NDW=0,
    center=(NX / 2 + 0.5, NY / 2 + 0.5, NZ / 2 + 0.5, NT / 2 + 0.5),
    radius=div(NX, 2),
    sign=+1,
    block=(1, 2),
)
```

Parameter notes:

- `center`: location of the instanton center, matching the current default.
- `radius`: instanton radius, matching the current `div(NX, 2)` default.
- `sign`: `+1` for instanton and `-1` for anti-instanton. The implementation
  should use this to select the self-dual or anti-self-dual convention.
- `block`: SU(2) color subgroup inside SU(Nc), defaulting to `(1, 2)`.

The first code PR can initially support 4D serial non-MPI fields. Wing and
no-wing support should be included together only if the implementation can share
the core link construction cleanly.

## Validation plan

The first tests should focus on local algebraic correctness and compatibility:

- Compare the selected SU(Nc) 2 by 2 block against the existing SU(2)
  `Oneinstanton` link for the same lattice, center, and radius.
- Check that spectator color diagonal entries are one.
- Check that off-block spectator couplings are zero.
- Check unitarity of each SU(Nc) link.
- Check `det(U_mu(x)) == 1` within tolerance.
- Verify all supported SU(3) `block` values.
- Verify at least one `NC > 3` block, such as `NC=4, block=(2, 4)`.
- Verify invalid `block` values throw a clear error.
- Keep existing `Oneinstanton` tests unchanged.

Topological charge tests can be added after the basic embedding helper and API
are stable, because they also depend on measurement conventions and smoothing.

## Topological charge density and visualization

The scalar topological charge `Q` and the local density `q(x)` should be treated
as separate outputs.

The current sample code in `samples/measurements/topologicalcharge.jl` computes
a summed scalar `Q`. For visualization, a future density routine should return
site-wise `q(x)` before the final reduction. Then the scalar `Q` can be checked
as the sum of the density with the same normalization.

This separation is useful for connecting to VisualizingLQCD: Gaugefields.jl can
provide the lattice density, while visualization code can decide how to slice,
project, smooth, or color-map it.

## Not in the first PR

The first PR should not:

- Enable `Oneinstanton(3, ...)` directly.
- Change the behavior of the existing SU(2) `Oneinstanton`.
- Change topological charge measurement APIs.
- Add visualization code.
- Add MPI support.
- Refactor unrelated gauge field initialization code.

## Implementation procedure

PR-1: survey/design docs.

Tests: none beyond reviewing the diff, because this PR is docs-only.

PR-2: add a lattice-independent matrix helper that embeds an SU(2) link into an
SU(Nc) block.

Candidate internals:

```julia
_validate_su2_embedding_block(NC, block)
_embed_su2_matrix_in_sun!(U, u2, block)
```

Tests:

- `NC=3` with `(1, 2)`, `(1, 3)`, and `(2, 3)`.
- `NC=4` or `NC=5` with a nontrivial block such as `(2, 4)` or `(2, 5)`.
- The selected 2 by 2 block equals the input SU(2) matrix.
- Spectator diagonal entries are one.
- Spectator off-block entries are zero.
- `U' * U` is identity within tolerance.
- `det(U)` is one within tolerance.
- Invalid blocks such as `(1, 1)`, `(0, 2)`, and `(1, NC + 1)` throw clear
  errors.

PR-3: factor out SU(2) instanton link generation from the existing 4D
constructors.

Candidate internal:

```julia
_su2_instanton_link(mu, ix, iy, iz, it, L; center, radius, sign)
```

Tests:

- With default `center`, `radius`, and `sign=+1`, the helper reproduces the
  existing SU(2) `Oneinstanton` links in the physical region.
- Each generated SU(2) link is unitary.
- Each generated SU(2) link has determinant one within tolerance.
- Invalid `radius` and invalid `sign` throw clear errors.
- `sign=-1` keeps unitarity and determinant one. Topological-charge sign tests
  can wait until PR-5.

PR-4: add the explicit public API, for example `Oneinstanton_SUN_embedded`.

Tests:

- `NC=3` with all three SU(3) block choices.
- `NC=4` or `NC=5` with at least one representative block.
- `NDW=0` and `NDW=1` both work for serial 4D fields.
- The selected block equals the SU(2) instanton link.
- Spectator diagonal entries are one and spectator off-block entries are zero.
- Every link is unitary.
- Every link has determinant one within tolerance.
- The plaquette matches the embedding expectation
  `P_NC = (2 * P_SU2 + (NC - 2)) / NC`.
- Existing `Oneinstanton` tests remain unchanged and still pass.

PR-5: add topological-charge checks for the embedded instanton.

Tests:

- Cold fields have `Q` close to zero.
- SU(2) instanton and SU(Nc)-embedded instanton have matching scalar `Q`.
- `NC=3`, `NC=4`, and `NC=5` embedded instantons have matching `Q`.
- Changing `block` does not change `Q`.
- `sign=-1` flips the sign of `Q`.
- Start with the plaquette definition. Clover or improved definitions can be
  added after the measurement conventions are clear.

PR-6: investigate and add site-wise topological charge density `q(x)`.

Tests:

- `sum(q) == Q` within tolerance using the same normalization.
- Cold fields have site-wise density close to zero.
- SU(2) and SU(Nc)-embedded instantons have matching `q(x)`.
- `sign=-1` flips the sign of `q(x)`.
- The density shape matches the lattice shape.

PR-7: connect `q(x)` visualization on the VisualizingLQCD side.

Tests:

- Gaugefields.jl returns numerical density data without visualization
  assumptions.
- VisualizingLQCD handles slicing, projection, smoothing, and color maps.
- A small fixture confirms that the visualized density sums back to the scalar
  `Q` used by Gaugefields.jl.

## Test placement and commands

Add focused tests in a new file such as `test/sun_embedded_instanton.jl`, then
include it from `test/runtests.jl` near the initialization tests.

During development, run the focused file first. Before each PR, also run:

```sh
julia --project=. test/runtests.jl
```

GPU, MPI, and MPILattice tests should stay out of the early PRs unless the
public API explicitly supports those paths.

## Branch and PR workflow

Do not push directly to `main`.

Use one local branch and one remote branch per PR. After a PR is merged, update
local `main` and create the next PR branch from the updated `main`.

Suggested branch sequence:

- PR-1: `codex/sun-embedded-instanton-design`
- PR-2: `codex/sun-embedded-instanton-helper`
- PR-3: `codex/sun-embedded-instanton-link-helper`
- PR-4: `codex/sun-embedded-instanton-api`
- PR-5: `codex/sun-embedded-instanton-q-tests`
- PR-6: `codex/sun-embedded-instanton-density`
- PR-7: `codex/sun-embedded-instanton-visualization`

Normal flow:

```sh
git switch main
git pull --ff-only
git switch -c codex/sun-embedded-instanton-design
# edit, test, commit
git push -u origin codex/sun-embedded-instanton-design
```

After PR-1 is merged:

```sh
git switch main
git pull --ff-only
git switch -c codex/sun-embedded-instanton-helper
```

If the next PR must start before the previous PR is merged, use a stacked
branch:

```sh
git switch codex/sun-embedded-instanton-design
git switch -c codex/sun-embedded-instanton-helper
```

In stacked mode, keep the PR description clear about the dependency on the
previous PR. Rebase the later branch onto updated `main` after the earlier PR is
merged.
