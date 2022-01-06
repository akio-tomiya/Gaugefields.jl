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

    temps2 = Array{Matrix{ComplexF64},1}(undef,5) 
    temps3 = Array{Matrix{ComplexF64},1}(undef,5) 
    for i=1:5
        temps2[i] = zeros(ComplexF64,2,2)
        temps3[i] = zeros(ComplexF64,NC,NC)
    end


    mapfunc!(A,B) = SU3update_matrix!(A,B,β,NC,temps2,temps3,ITERATION_MAX)

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
        heatbath_SU3!(U,NC,[temp1,temp2,temp3],β)

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

## Gradientflow
We can use Lüscher's gradient flow.

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
NC = 3
u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
U = Array{typeof(u1),1}(undef,Dim)
U[1] = u1
for μ=2:Dim
    U[μ] = RandomGauges(NC,Nwing,NX,NY,NZ,NT)
end

temp1 = similar(U[1])
temp2 = similar(U[1])
temp3 = similar(U[1])

comb = 6
factor = 1/(comb*U[1].NV*U[1].NC)

g = Gradientflow(U)
for itrj=1:100
    flow!(U,g)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("$itrj plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2) 
    println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
end

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

For example, if you want to calculate the clover operators, you can define like: 

```julia
function make_cloverloop(μ,ν,Dim)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim) # Pmunu
    push!(loops,loop_righttop)
    loop_rightbottom = Wilsonline([(ν,-1),(μ,1),(ν,1),(μ,-1)],Dim = Dim) # Qmunu
    push!(loops,loop_rightbottom)
    loop_leftbottom= Wilsonline([(μ,-1),(ν,-1),(μ,1),(ν,1)],Dim = Dim) # Rmunu
    push!(loops,loop_leftbottom)
    loop_lefttop = Wilsonline([(ν,1),(μ,-1),(ν,-1),(μ,1)],Dim = Dim) # Smunu
    push!(loops,loop_lefttop)
    return loops
end
```

The energy density defined in the paper (Ramos and Sint, [Eur. Phys. J. C (2016) 76:15](https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-015-3831-9)) can be calculated as follows.  Note: the coefficient in the equation (3.40) in the preprint version is wrong. 

```julia
function make_clover(G,U,temps,Dim)
    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]
    
    for μ=1:Dim
        for ν=1:Dim
            if μ == ν
                continue
            end
            loops = make_cloverloop(μ,ν,Dim)
            evaluate_gaugelinks!(temp3,loops,U,[temp1,temp2])

            Traceless_antihermitian!(G[μ,ν],temp3)
        end
    end
end

function calc_energydensity(G,U,temps,Dim)
    temp1 = temps[1]
    s = 0
    for μ=1:Dim
        for ν=1:Dim
            if μ == ν
                continue
            end
            mul!(temp1,G[μ,ν],G[μ,ν])
            s += -real(tr(temp1))/2
        end
    end
    return  s/(4^2*U[1].NV)
end
```

Then, we can calculate the energy density: 

```julia
function test(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 1

    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    end

    filename = "./conf_00000010.txt" 
    L = [NX,NY,NZ,NT]
    load_BridgeText!(filename,U,L,NC) # We load a configuration from a file. 

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    println("Make clover operator")
    G = Array{typeof(u1),2}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            G[μ,ν] = similar(U[1])
        end
    end

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")

    g = Gradientflow(U,eps = 0.01)
    for itrj=1:100
        flow!(U,g)

        make_clover(G,U,[temp1,temp2,temp3],Dim)
        E = calc_energydensity(G,U,[temp1,temp2,temp3],Dim)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj $(itrj*0.01) plaq_t = $plaq_t , E = $E")
    end

end
NX = 8
NY = 8
NZ = 8
NT = 8
β = 5.7
NC = 3
test(NX,NY,NZ,NT,β,NC)
```


# How to calculate actions
We can calculate actions from this packages with fixed gaugefields U. 
We introduce the concenpt "Scalar neural networks", which is S(U) -> V, where U and V are gaugefields. 


```julia
using Gaugefields
using LinearAlgebra
function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    end

    snet = ScalarNN(U) #empty network
    plaqloop = make_loops_fromname("plaquette") #This is a plaquette loops. 
    append!(plaqloop,plaqloop') #We need hermitian conjugate loops for making the action real. 
    β = 1 #This is a coefficient.
    push!(snet,β,plaqloop)
    
    show(snet)

    Uout = apply_snet(snet,U)
    println(tr(Uout))
end

test1()
```

The output is 

```
----------------------------------------------
Structure of scalar neural networks
num. of terms: 1
-------------------------------
      1-st term: 
          coefficient: 1.0
      -------------------------
1-st loop
L"$U_{1}(n)U_{2}(n+e_{1})U^{\dagger}_{1}(n+e_{2})U^{\dagger}_{2}(n)$"	
2-nd loop
L"$U_{1}(n)U_{3}(n+e_{1})U^{\dagger}_{1}(n+e_{3})U^{\dagger}_{3}(n)$"	
3-rd loop
L"$U_{1}(n)U_{4}(n+e_{1})U^{\dagger}_{1}(n+e_{4})U^{\dagger}_{4}(n)$"	
4-th loop
L"$U_{2}(n)U_{3}(n+e_{2})U^{\dagger}_{2}(n+e_{3})U^{\dagger}_{3}(n)$"	
5-th loop
L"$U_{2}(n)U_{4}(n+e_{2})U^{\dagger}_{2}(n+e_{4})U^{\dagger}_{4}(n)$"	
6-th loop
L"$U_{3}(n)U_{4}(n+e_{3})U^{\dagger}_{3}(n+e_{4})U^{\dagger}_{4}(n)$"	
7-th loop
L"$U_{2}(n)U_{1}(n+e_{2})U^{\dagger}_{2}(n+e_{1})U^{\dagger}_{1}(n)$"	
8-th loop
L"$U_{3}(n)U_{1}(n+e_{3})U^{\dagger}_{3}(n+e_{1})U^{\dagger}_{1}(n)$"	
9-th loop
L"$U_{4}(n)U_{1}(n+e_{4})U^{\dagger}_{4}(n+e_{1})U^{\dagger}_{1}(n)$"	
10-th loop
L"$U_{3}(n)U_{2}(n+e_{3})U^{\dagger}_{3}(n+e_{2})U^{\dagger}_{2}(n)$"	
11-th loop
L"$U_{4}(n)U_{2}(n+e_{4})U^{\dagger}_{4}(n+e_{2})U^{\dagger}_{2}(n)$"	
12-th loop
L"$U_{4}(n)U_{3}(n+e_{4})U^{\dagger}_{4}(n+e_{3})U^{\dagger}_{3}(n)$"	
      -------------------------
----------------------------------------------
9216.0 + 0.0im

```

# How to calculate derivatives
We can easily calculate the matrix derivative of the scalar neural networks. The matrix derivative is defined as 

```math
[\frac{\partial S}{\partial U_{\mu}(n)}]_{ij} = \frac{\partial S}{\partial U_{\mu,ji}(n)}
```

We can calculate this like 

```julia
dSdUμ = calc_dSdUμ(snet,μ,U)
```

or

```julia
calc_dSdUμ!(dSdUμ,snet,μ,U)
```
# Hybrid Monte Carlo method
With the use of the matrix derivative, we can do the Hybrid Monte Carlo method. 
The simple code is as follows. 

```julia
using Gaugefields
using LinearAlgebra

function MDtest!(snet,U,Dim)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(snet,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end
```

We define the functions as 

```julia

function calc_action(snet,U,p)
    NC = U[1].NC
    Sg = -calc_scalar(snet,U)/NC #calc_scalar(snet,U) = tr(apply_snet(snet,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(snet,U,p,MDsteps,Dim,Uold)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(snet,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,snet)

        P_update!(U,p,1.0,Δτ,Dim,snet)

        U_update!(U,p,0.5,Δτ,Dim,snet)
    end
    Snew = calc_action(snet,U,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(Snew-Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,snet)
    temps = get_temporal_gauges(snet)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,snet) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporal_gauges(snet)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,snet,μ,U)
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end
```

Then, we can do the HMC: 

```julia
function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    U = Array{typeof(u1),1}(undef,Dim)
    U[1] = u1
    for μ=2:Dim
        U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
    end


    snet = ScalarNN(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(snet,β,plaqloop)
    
    show(snet)

    MDtest!(snet,U,Dim)

end


test1()
```







