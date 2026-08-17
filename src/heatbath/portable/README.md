# Portable heatbath kernel boundary

Gaugefields v1 keeps the SU(2) Kennedy--Pendleton and fixed-order SU(3)
Cabibbo--Marinari physics kernels in this directory. These two source files
are the supported copy/include boundary for hosts that cannot depend on
LatticeMatrices:

1. `rng_protocol.jl`
2. `kernels.jl`

Include them, in that order, inside a host module which has imported
`LinearAlgebra.mul!`. Neither file imports LatticeMatrices, JACC, MPI, Random,
or a GPU-vendor package. Gaugefields-specific RNG adapters and legacy failure
handling live in `../rng_adapters.jl` and are deliberately outside this
boundary.

## Required host hooks

A host RNG type must implement:

```julia
heatbath_uniform(rng, ::Type{T}) -> updated_rng, value
heatbath_log_uniform(rng, ::Type{T}) -> updated_rng, value
```

`heatbath_uniform` supplies values in `[0, 1)`. `heatbath_log_uniform` supplies
the logarithm inputs and should normally exclude zero. An immutable RNG returns
its new state; a mutable RNG may return the same mutated object.

SU(3) additionally calls:

```julia
heatbath_normalize3!(u) -> success::Bool
```

The hook must reunitarize the site-local 3x3 matrix and return `false` instead
of throwing when normalization cannot be completed. This leaves storage and
backend-specific normalization outside the portable kernel.

The protocol supplies clean defaults for two optional adapter hooks:

```julia
heatbath_prepare_su2(rng, ::Type{T}) = rng
heatbath_beta(rng, beta, ::Type{T}) = beta
```

`heatbath_prepare_su2` exists only so a compatibility adapter can advance an
older stream before the clean kernel. New portable hosts should retain the
default. `heatbath_beta` may be specialized when a device backend must convert
`beta` to its matrix real type.

## Kernel calls and return values

```julia
SU2update_KP!(rng, Unew, V, beta, NC, temps2, iteration_max)
# -> updated_rng, accepted, tries

SU3update_matrix!(rng, u, V, beta, 3, temps2, temps3, iteration_max)
# -> updated_rng, accepted, failed_subgroup

SU3update_matrix!(rngs::NTuple{3}, u, V, beta, 3,
                  temps2, temps3, iteration_max)
# -> updated_rngs, accepted, failed_subgroup
```

The single-stream SU(3) form consumes one sequential stream across the fixed
`(1,2)`, `(2,3)`, `(1,3)` subgroup order. The tuple form assigns one stream to
each subgroup in that same order. `failed_subgroup` is zero on success, 1--3
for a rejected subgroup, 4 for normalization failure, and -1 for an invalid
`NC` argument.

`temps2` contains two 2x2 matrices for SU(2), or four 2x2 matrices for SU(3).
`temps3` contains three 3x3 matrices. The caller owns all storage. The
successful path allocates no heap storage and performs no global RNG access,
I/O, or exception handling. A rejected SU(2) update leaves `Unew` unchanged;
an SU(3) caller should discard its site-local `u` if any subgroup fails.

## Random-stream contract

The clean SU(2) kernel has no discarded prefix draws. Each proposal consumes,
in order, two `heatbath_log_uniform` values and two `heatbath_uniform` values.
After acceptance it consumes two more `heatbath_uniform` values for the
orientation. Rejected proposals repeat only the four proposal draws.

Gaugefields' serial Julia-RNG adapter consumes the four historical overwritten
draws in `heatbath_prepare_su2`. This preserves the released serial stream and
matrix bits without contaminating the portable kernel. LatticeMatrices site
streams use the clean contract. Gaugefields v1 intentionally does not promise
random-stream compatibility with older experimental LatticeMatrices heatbath
implementations.

The fixed-draw and released-stream references are maintained in:

- `test/su2update_allocationfree.jl`
- `test/su3update_allocationfree.jl`

