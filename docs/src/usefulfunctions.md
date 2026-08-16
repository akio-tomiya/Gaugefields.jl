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
```

## Momenta and measurements

```@docs
gauge_momenta
gaussian_momenta
measure_plaquette
measure_polyakov_loop
```

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
QPQ
PQP
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

The pre-v1 exported surface remains available for compatibility but is listed
separately in [Legacy API](legacyapi.md).
