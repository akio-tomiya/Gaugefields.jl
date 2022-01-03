# Gaugefields

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cometscome.github.io/Gaugefields.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cometscome.github.io/Gaugefields.jl/dev)
[![Build Status](https://github.com/cometscome/Gaugefields.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/Gaugefields.jl/actions/workflows/CI.yml?query=branch%3Amain)

# This is the package for Lattice QCD codes. 

This is used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl)

# Install

```
add https://github.com/akio-tomiya/Gaugefields.jl
```

# How to use

## File loading
## ILDG format
[ILDG](https://www-zeuthen.desy.de/~pleiter/ildg/ildg-file-format-1.1.pdf) format is one of standard formats for LatticeQCD configurations.

We can read ILDG format like: 

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4
u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
U = Array{typeof(u1),1}(undef,Dim)
ildg = ILDG(filename)
i = 1
for μ=1:Dim
    U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
end
L = [NX,NY,NZ,NT]
load_gaugefield!(U,i,ildg,L,NC)
```
Then, we can calculate the plaquette: 

```julia
temp1 = similar(U[1])
temp2 = similar(U[1])

comb = 6
factor = 1/(comb*U[1].NV*U[1].NC)
@time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
println("plaq_t = $plaq_t")
poly = calculate_Polyakov_loop(U,temp1,temp2) 
println("polyakov loop = $(real(poly)) $(imag(poly))")
```

We can write a configuration as the ILDG format like 

```julia
filename = "hoge.ildg"
save_binarydata(U,filename)
```

## Text format for Bridge++
Gaugefield.jl also supports a text format for [Bridge++](https://bridge.kek.jp/Lattice-code/index_e.html). 

### File loading

```julia
filename = "testconf.txt"
load_BridgeText!(filename,U,L,NC)
```

### File saving

```julia
filename = "testconf.txt"
save_textdata(U,filename)
```


## Heatbath updates (even-odd method)

```julia
using Gaugefields


function heatbath_SU3!(U,NC,temps,β)
    Dim = 4
    temp1 = temps[1]
    temp2 = temps[2]
    V = temps[3]
    ITERATION_MAX = 10^5

    mapfunc!(A,B) = SU3update_matrix!(A,B,β,NC,ITERATION_MAX)

    for μ=1:Dim

        loops = loops_staple[(Dim,μ)]
        iseven = true

        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 

        iseven = false
        evaluate_gaugelinks_evenodd!(V,loops,U,[temp1,temp2],iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 
    end
    
end

function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 1

    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    end

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 40
    for itrj = 1:numhb
        if NC == 2
            heatbath_SU2!(U,NC,[temp1,temp2,temp3],β)
        elseif NC == 3
            heatbath_SU3!(U,NC,[temp1,temp2,temp3],β)
        else
            heatbath_SUN!(U,NC,[temp1,temp2,temp3],β)
        end

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    

    return plaq_t

end

NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1

β = 5.7
NC = 3
@time plaq_t = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
```

# Useful functions

## Data structure
We can access the gauge field defined on the bond between two neigbohr points. 
In 4D system, the gauge field is like ```u[ic,jc,ix,iy,iz,it]```. 
There are four directions in 4D system. Gaugefields.jl uses the array like: 

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4
u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT) #Unit matrix everywhere. 
U = Array{typeof(u1),1}(undef,Dim)
for μ=1:Dim
    U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
end
```

## Adjoint operator
If you want to get the hermitian conjugate of the gaugefields, you can do like 

```julia
u'
```
This is a lazy evaluation. So there is no memory copy. 

## Shift operator
If you want to shift the gaugefields, you can do like 

```julia
ushift = shift_U(u,shift)
```
This is also a lazy evaluation. 

## matrix-matrix operation
If you want to calculate the matrix-matrix multiplicaetion on each lattice site, you can do like

```julia
mul!(A,B,C)
```
which means ```A = B*C``` on each lattice site. 
## Trace operation 
If you want to calculate the trace of the gauge field, you can do like 

```julia
tr(A)
```
It is useful to evaluation actions. 

# Wilsonloops
We develop [Wilsonloop.jl](https://github.com/akio-tomiya/Wilsonloop.jl.git), which is useful to calculate Wilson loops. 
If you want to use this, please install like

```
add https://github.com/akio-tomiya/Wilsonloop.jl
```

For example, if you want to calculate the following quantity: 

```math
U_{1}(n)U_{2}(n+\hat{1}) U^{\dagger}_{1}(n+\hat{2}) U^{\dagger}_2(n)
```

You can use Wilsonloop.jl as follows

```julia
using Wilsonloop
loop = [(1,1),(2,1),(1,-1),(2,-1)]
w = Wilsonline(loop)
```
The output is ```L"$U_{1}(n)U_{2}(n+e_{1})U^{\dagger}_{1}(n+e_{2})U^{\dagger}_{2}(n)$"```. 
Then, you can evaluate this loop with the use of the Gaugefields.jl like: 

```julia
using LinearAlgebra
NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3
Nwing = 1
Dim = 4
u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT) #Unit matrix everywhere. 
U = Array{typeof(u1),1}(undef,Dim)
for μ=1:Dim
    U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
end

temp1 = similar(U[1])
V = similar(U[1])

evaluate_gaugelinks!(V,w,U,[temp1])
println(tr(V))
```







