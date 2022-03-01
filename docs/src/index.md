# Gaugefields



Documentation for [Gaugefields](https://github.com/akio-tomiya/Gaugefields.jl).

```@index
```


[![CI](https://github.com/akio-tomiya/Gaugefields.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/Gaugefields.jl/actions/workflows/CI.yml)

# Abstract

This is a package for lattice QCD codes.
Treating gauge fields (links), gauge actions with MPI and autograd.


```@raw html
<img src="./LQCDjl_block.png" width=300>
```


This package will be used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl). 

# What this package can do:
This package has following functionarities

- SU(Nc) (Nc > 1) gauge fields in 2 or 4 dimensions with arbitrary actions.
- U(1) gauge fields in 2 dimensions with arbitrary actions. 
- Configuration generation
    - Heatbath
    - quenched Hybrid Monte Carlo
- Gradient flow via RK3
- I/O: ILDG and Bridge++ formats are supported ([c-lime](https://usqcd-software.github.io/c-lime/) will be installed implicitly with [CLIME_jll](https://github.com/JuliaBinaryWrappers/CLIME_jll.jl) )
- MPI parallel computation (experimental. not shown)

Dynamical fermions will be supported with [LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl).

In addition, this supports followings
- **Autograd for functions with SU(Nc) variables**
- Stout smearing (exp projecting smearing)
- Stout force via [backpropagation](https://arxiv.org/abs/2103.11965)

Autograd can be worked for general Wilson lines except for ones have overlaps.

# Install

```
add Gaugefields
```













