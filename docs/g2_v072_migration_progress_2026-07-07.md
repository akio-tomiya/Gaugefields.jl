# G2 implementation migration to Gaugefields.jl v0.7.2

Date: 2026-07-07

## Recovery points

- Preserved old working branch:
  `archive/g2-core-v012-before-v072-migration`
- Preserved old working tag:
  `g2-core-v012-before-v072-migration-20260707`
- Old G2 branch head:
  `18fe0eb0d96826489b494ac61263de057520445f`
- New migration branch:
  `codex/g2-core-v072`
- New branch base:
  `origin/master` / `v0.7.2` at `9e5719970770f4497405a856315c90bef7f74449`

The old branch and tag were pushed to GitHub before migration work started.

## Migration commits

- `0de8f38` Add G2 algebra on Gaugefields v0.7.2
- `b839e34` Place G2 algebra in gaugefield module
- `506e6e3` Add G2 link and momentum fields on v0.7.2
- `4256eec` Add G2 gauge action oracle test
- `d2c627e` Validate G2 plaquette force on v0.7.2
- `c39f21a` Add G2 HMC smoke tests on v0.7.2

## Implemented scope

- G2 algebra basis and projections.
- G2 7 by 7 link container with wing cells.
- G2 14-component real momentum container.
- G2 projection and exponential link update.
- G2 plaquette and `GaugeAction` oracle tests.
- G2 plaquette force finite-difference test.
- G2 quenched HMC smoke test.

The v0.7.2 `Wilsonloop.isdag(link)` route is preserved. The old
`typeof(link) <: Adjoint_GLink` route was not restored.

## Test results

Commands run on the migration branch:

```sh
julia --project=. test/g2_algebra_test.jl
julia --project=. test/g2_gaugefields_test.jl
julia --project=. test/g2_momentum_test.jl
julia --project=. test/g2_projection_update_test.jl
julia --project=. test/g2_interface_test.jl
julia --project=. test/g2_gauge_action_test.jl
julia --project=. test/g2_force_test.jl
julia --project=. test/g2_hmc_test.jl
julia --project=. test/g2_runtests.jl
julia --project=. test/sun_embedded_instanton.jl
```

Observed results:

- `test/g2_runtests.jl`: 190 / 190 pass.
- `test/sun_embedded_instanton.jl`: existing SU(N) smoke tests pass.

`Pkg.test()` was not run in full during this migration checkpoint because the
full suite includes heavier HMC, heatbath, gradient-flow, and B-field tests.

## v0.7.2-specific adjustment

The old force test used an array of temporary fields and the factor
`-1/(2NC)`. In v0.7.2, `add_force!` takes `Temporalfields`, and the
finite-difference oracle for

```julia
real(-evaluate_GaugeAction(gauge_action, U) / U[1].NC)
```

matches `add_force!(; plaqonly=true, factor=1/NC)`.

This convention is covered by `test/g2_force_test.jl`.
