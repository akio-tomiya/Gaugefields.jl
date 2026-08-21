# Public v1 API index

This page collects the docstrings for the recommended Gaugefields v1 entry
points. Detailed parameter tables are in
[High-level API parameters](highlevelapi.md).

## Configurations and metadata

```@docs
gauge_configuration
LatticeMatricesBackend
LegacyBackend
gauge_backend
gauge_lattice_size
gauge_num_colors
gauge_halo_width
gauge_process_grid
gauge_communicator
copy_configuration
copy_configuration!
```

U(1) fields and the specialized `Oneinstanton` and
`Oneinstanton_SUN_embedded` constructors are covered in
[Two and three dimensions](dimensions.md).

## Momenta and measurements

```@docs
gauge_momenta
gaussian_momenta
gaussian_momenta!
measure_plaquette
measure_polyakov_loop
```

Topological charge and topological-charge-density measurements are provided
by QCDMeasurements.jl; see [Measurements](measurements.md#Advanced-observables).

## Wilson-loop actions

```@docs
GaugeAction
evaluate_GaugeAction
evaluate_GaugeAction_untraced
evaluate_GaugeAction_untraced!
calc_dSdUμ
calc_dSdUμ!
```

`make_loops_fromname` is re-exported from Wilsonloop.jl. Direct construction
of `Wilsonline` paths and low-level `evaluate_gaugelinks!` usage are documented
in [Wilson loops and gauge actions](wilsonloops_actions.md).

## Updates, flow, smearing, and I/O

```@docs
gradient_flow
heatbath_updater
stout_smearing
smear
save_configuration
load_configuration
load_configuration!
```

## Molecular dynamics

```@docs
AbstractMDIntegrator
MDDriver
md_driver
MDActionSet
MDForceGroup
QPQ
PQP
SextonWeingarten
md_step_size
md_hamiltonian
md_trajectory!
update_momenta!
update_gaugefields!
md_step!
```

## MD action providers

```@docs
md_action_workspace
md_potential
md_force!
enzyme_md_action
```

## Automatic differentiation

`Enzyme_derivative!` writes the matrix gradient of a real scalar function.
`nodiff`, `diff`, `realtrace`, `Wiltinger_derivative!`, and
`Wiltinger_numerical_derivative` are also exported for differentiation
workflows. Their argument conventions and the distinction between a raw
gradient and an MD force are explained in
[Automatic differentiation with Enzyme](autodiff.md).

The pre-v1 exported surface remains available for compatibility but is listed
separately in [Legacy API](legacyapi.md).
