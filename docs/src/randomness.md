# Randomness and reproducibility

Gaugefields uses separate random streams for configuration initialization,
heatbath updates, HMC momenta, and HMC acceptance decisions. A `seed` passed to
one operation intentionally does not change hidden global state or seed the
other operations.

## What one seed controls

| Operation | Seed parameter | Counter/state | LM behavior |
| --- | --- | --- | --- |
| Hot start | `gauge_configuration(...; seed=...)` | none | Reproducible global field, independent of MPI decomposition. |
| Gaussian momenta | `gaussian_momenta(...; seed=..., sweep=...)` | caller supplies `sweep` | Reproducible global momenta, independent of MPI decomposition. |
| Heatbath | `heatbath_updater(...; seed=..., sweep=...)` | updater increments `sweep` | Reproducible site streams and update sequence. |
| Overrelaxation | updater `seed` and `overrelaxation_sweep` | updater increments its separate counter | Reproducible LM update sequence. |
| MD trajectory | none | action, integrator, trajectory length, steps, input `U`, and input `p` | Deterministic; `md_driver` draws no random numbers. |
| HMC acceptance | none in Gaugefields | controlled by the higher-level HMC code | Must be seeded and synchronized separately. |

`seed=nothing` on an LM hot start or Gaussian momentum fill obtains a fresh
seed from `RandomDevice` on rank zero and broadcasts it. It is therefore not
reproducible between runs. Heatbath updaters instead use `seed=0` by default,
so an updater is deterministic when its initial field and all other parameters
are the same.

The following RNG algorithms are available for LM site-local streams:

- `Philox4x32()` (default)
- `PCG32()`
- `Xoshiro256PlusPlus()`

The algorithm is part of the reproducibility specification. Changing it while
keeping the same integer seed produces a different stream.

## Reproducible hot starts

```julia
U1 = gauge_configuration(
    (16, 16, 16, 32);
    start=:hot,
    seed=1234,
    rng=Philox4x32(),
)

U2 = gauge_configuration(
    (16, 16, 16, 32);
    start=:hot,
    seed=1234,
    rng=Philox4x32(),
)
```

`U1` and `U2` represent the same global field. The global-site stream does not
include an MPI rank or local-site number, so changing a valid process grid
does not change the generated global field.

For `start=:cold`, no random numbers are used and `seed` has no effect.

## Reproducible heatbath chains

```julia
U = gauge_configuration(
    (16, 16, 16, 32);
    start=:hot,
    seed=100,
)

updater = heatbath_updater(
    U;
    beta=6.0,
    seed=200,
    sweep=0,
    rng_algorithm=Philox4x32(),
)

for _ in 1:100
    heatbath!(U, updater)
end

@assert updater.sweep == 100
```

The hot-start seed and heatbath seed are deliberately different streams.
Every successful `heatbath!` increments `updater.sweep`; a failed sweep does
not. To resume exactly from a checkpoint, save the configuration together
with at least the updater's `seed`, `sweep`, `overrelaxation_sweep`, RNG
algorithm, action parameters, and coloring choice.

For LM fields, the random stream assigned to a site is independent of the MPI
decomposition. Floating-point arithmetic can nevertheless differ in its last
bits across CPU and GPU backends, precisions, MPI reduction trees, package
versions, or device math implementations. Therefore:

- the same software/backend/precision and state should reproduce the same run;
- different MPI decompositions use the same random streams and are expected to
  agree numerically;
- bitwise identity across different hardware backends is not promised.

Legacy fields retain their historical use of Julia's global RNG in several
updates. `Random.seed!` can reproduce a fixed serial execution order, but it
does not provide the LM global-site guarantee across MPI decompositions or GPU
execution.

### Heatbath stream boundary in v1

The v1 portable heatbath kernel starts directly with the first
Kennedy--Pendleton proposal. For every proposal it draws two logarithm inputs
and two ordinary uniforms; after acceptance it draws two orientation values.
There are no discarded prefix draws in LM site streams. SU(3) applies this
contract in fixed `(1,2)`, `(2,3)`, `(1,3)` subgroup order, using one tagged
site stream per subgroup.

The serial Julia-RNG adapter separately consumes the four historical
overwritten draws before each SU(2) subgroup. This keeps released serial runs
and their following global RNG state reproducible while leaving the portable
LM/Web stream clean. Compatibility with random-number consumption from older
experimental LM heatbath implementations is intentionally not part of the v1
contract. The source-level host interface is documented next to the kernels in
`src/heatbath/portable/README.md`.

## Gaussian momenta for HMC

LM momenta have an explicit trajectory counter:

```julia
trajectory = 0
p = gaussian_momenta(
    U;
    sigma=1.0,
    seed=300,
    sweep=trajectory,
    rng=Philox4x32(),
)
```

Use a different `sweep` value for every trajectory. Reusing the same
`(seed, sweep, rng)` tuple deliberately regenerates the same momentum field.

## Deterministic MD and HMC ownership

`md_trajectory!` is deterministic for a fixed input configuration, momentum,
action, integrator, trajectory length, step count, backend, and precision. It
does not consume the configuration seed or momentum seed, and it owns no RNG
state. The optional Hamiltonian diagnostics also draw no random numbers.

Gaugefields deliberately stops at the MD layer. Momentum refresh, Metropolis
acceptance, proposal rollback, and the trajectory counter are responsibilities
of an application or a higher-level HMC package. Consequently, a seed on
`gauge_configuration` alone cannot reproduce a complete HMC chain.

A reproducible LM HMC implementation must control all of the following:

1. the initial configuration seed;
2. the momentum seed, RNG algorithm, and trajectory counter;
3. the Metropolis acceptance RNG state;
4. the MD integrator, trajectory length, step count, action, and precision;
5. on MPI, one acceptance draw on rank zero followed by a broadcast;
6. all counters and RNG state needed when restarting from a checkpoint.

The HMC layer must save these states explicitly rather than derive them from
the configuration seed. On MPI it should make a single acceptance draw on rank
zero and broadcast the decision, so every rank accepts or restores the same
proposal.
