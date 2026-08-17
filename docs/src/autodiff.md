# Automatic differentiation with Enzyme

Gaugefields can obtain the force of a custom four-dimensional scalar
potential from Enzyme and use it with the common molecular-dynamics driver.
Enzyme is optional, so add and load it explicitly:

~~~julia
pkg> add Enzyme
~~~

~~~julia
import JACC
JACC.@init_backend

using Enzyme
using Gaugefields
~~~

## What the potential receives

The simulation configuration `U` remains the usual vector of four
Gaugefields link objects. A function passed to `enzyme_md_action`, however,
receives the **underlying LatticeMatrices link matrices** as four separate
arguments:

~~~julia
potential(U1, U2, U3, U4, arguments..., temps)
~~~

If `num_temps > 0`, `temps` is the final argument and contains that many
reusable LatticeMatrices work fields. Gaugefields clears these fields before
each call. Extra positional arguments passed to `enzyme_md_action` are
constants during differentiation.

Keeping the outer Gaugefields metadata out of the differentiated function is
important on Julia 1.12, whose calling convention separates GC roots from the
inline fields of immutable composite arguments. It also lets the driver use
the Enzyme rules supplied by LatticeMatrices v1.1 directly.

## Enzyme-safe lattice products

Use these Gaugefields functions inside an Enzyme potential:

- `mul_shifted!(C, A, B, shift)` sets `C` to `A * B(x + shift)`;
- `mul_shifted_adjoint!(C, A, B, shift)` sets `C` to
  `A * B(x + shift)'`;
- `mul_adjoint!(C, A, B)` sets `C` to `A * B'`.

They are public aliases of LatticeMatrices operations with custom reverse-mode
rules. In particular, they avoid constructing a lazy shifted or adjoint
wrapper as an argument of `mul!`, which is not a reliable Enzyme path on
Julia 1.12.

Here is a complete two-orientation plaquette contribution:

~~~julia
function plaquette_pair!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    mul_shifted!(C, Uμ, Uν, shift_μ)
    mul_shifted_adjoint!(D, C, Uμ, shift_ν)
    mul_adjoint!(E, D, Uν)
    value = realtrace(E)

    mul_shifted!(C, Uν, Uμ, shift_ν)
    mul_shifted_adjoint!(D, C, Uν, shift_μ)
    mul_adjoint!(E, D, Uμ)
    return value + realtrace(E)
end

function plaquette_potential(U1, U2, U3, U4, coefficient, colors, temps)
    C, D, E = temps
    s1 = (1, 0, 0, 0)
    s2 = (0, 1, 0, 0)
    s3 = (0, 0, 1, 0)
    s4 = (0, 0, 0, 1)

    value = plaquette_pair!(C, D, E, U1, U2, s1, s2)
    value += plaquette_pair!(C, D, E, U1, U3, s1, s3)
    value += plaquette_pair!(C, D, E, U1, U4, s1, s4)
    value += plaquette_pair!(C, D, E, U2, U3, s2, s3)
    value += plaquette_pair!(C, D, E, U2, U4, s2, s4)
    value += plaquette_pair!(C, D, E, U3, U4, s3, s4)
    return -coefficient * value / colors
end
~~~

The explicit four link arguments and six plaquette pairs keep the function
type-stable and make the differentiated call graph predictable.

## Use the potential in molecular dynamics

~~~julia
U = gauge_configuration(
    (4, 4, 4, 4);
    colors=3,
    halo=1,
    start=:hot,
    seed=1234,
)

action = enzyme_md_action(
    plaquette_potential,
    5.7, # coefficient
    3;   # colors
    num_temps=3,
)

driver = md_driver(
    U,
    action;
    steps=20,
    trajectory_length=1.0,
    integrator=QPQ(),
)
P = gaussian_momenta(U; seed=5678, sweep=0)
result = md_trajectory!(U, P, driver)
~~~

The potential must include its complete sign and normalization and return one
real scalar. Gaugefields uses exactly the same function for Hamiltonian
evaluation and reverse-mode differentiation, then projects the matrix
gradient to the traceless anti-Hermitian `dp/dτ` representation expected by
the integrator.

`num_temps` defaults to `0`; set it to the exact number of reusable work
fields expected by the potential. `steps`, `trajectory_length`, and
`integrator` have the same meanings and defaults as for an analytic
`GaugeAction`.

## Direct matrix gradient

When the matrix gradient itself is needed, differentiate the underlying LM
links directly:

~~~julia
using LatticeMatrices

links = getproperty.(U, :U)
dlinks = similar.(links)
clear_matrix!.(dlinks)

function link_trace_action(U1, U2, U3, U4)
    return realtrace(U1) + realtrace(U2) +
           realtrace(U3) + realtrace(U4)
end

Enzyme_derivative!(
    link_trace_action,
    links[1], links[2], links[3], links[4],
    dlinks[1], dlinks[2], dlinks[3], dlinks[4],
)
~~~

Wrap additional constant arguments with `nodiff`. For a function using work
fields, pass matching LM collections through the `temp` and `dtemp` keywords.
The historical direct method that takes outer Gaugefields wrappers remains
available for compatibility, but the raw-LM form above is the portable form
for Julia 1.12.

## Current boundary

Enzyme MD currently requires four-dimensional LatticeMatrices fields and
`halo >= 1`. It uses the JACC CPU and CUDA/MPI implementations supported by
the underlying LatticeMatrices Enzyme extension. CUDA multi-GPU
differentiation also requires CUDA-aware MPI and consistent launcher/library
selection.

For a Wilson-loop action, the analytic [`GaugeAction`](wilsonloops_actions.md)
force is simpler and also covers 2D and 3D. Use Enzyme when the potential is
more naturally expressed as differentiable lattice algebra.
