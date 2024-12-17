# Gaugefields

[![CI](https://github.com/akio-tomiya/Gaugefields.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/akio-tomiya/Gaugefields.jl/actions/workflows/CI.yml)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://akio-tomiya.github.io//Gaugefields.jl/dev)

# Abstract

This is a package for lattice QCD codes.
Treating gauge fields (links), gauge actions with MPI and autograd.

<img src="LQCDjl_block.png" width=300> 

This package is used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl)
and a code in a project [JuliaQCD](https://github.com/JuliaQCD/).

[NOTE: This is an extended version in order to implement higher-form gauge fields
 (i.e., 't Hooft twisted boundary condition/flux).
See [o-morikawa/Gaugefields.jl](https://github.com/o-morikawa/Gaugefields.jl)]
 
# What this package can do:
This package has following functionarities

- SU(Nc) (Nc > 1) gauge fields in 2 or 4 dimensions with arbitrary actions.
- **Z(Nc) 2-form gauge fields in 4 dimensions, which are given as 't Hooft flux.**
- U(1) gauge fields in 2 dimensions with arbitrary actions. 
- Configuration generation
    - Heatbath
    - quenched Hybrid Monte Carlo
    - quenched Hybrid Monte Carlo being subject to 't Hooft twisted b.c.
        - with external (non-dynamical) Z(Nc) 2-form gauge fields
    - quenched Hybrid Monte Carlo for SU(Nc)/Z(Nc) gauge theory
        - with dynamical Z(Nc) 2-form gauge fields
- Gradient flow via RK3
    - Yang-Mills gradient flow
    - Yang-Mills gradient flow being subject to 't Hooft twisted b.c.
    - Gradient flow for SU(Nc)/Z(Nc) gauge theory
- I/O: ILDG and Bridge++ formats are supported ([c-lime](https://usqcd-software.github.io/c-lime/) will be installed implicitly with [CLIME_jll](https://github.com/JuliaBinaryWrappers/CLIME_jll.jl) )
- MPI parallel computation (experimental. See documents.)
    - quenched HMC with MPI being subject to 't Hooft twisted b.c.

**The implementation of higher-form gauge fields is based on
[arXiv:2303.10977 [hep-lat]](https://arxiv.org/abs/2303.10977).**

Dynamical fermions will be supported with [LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl).

In addition, this supports followings
- **Autograd for functions with SU(Nc) variables**
- Stout smearing (exp projecting smearing)
- Stout force via [backpropagation](https://arxiv.org/abs/2103.11965)

Autograd can be worked for general Wilson lines except for ones have overlaps.

# Install

In Julia REPL in the package mode,
```
add Gaugefields.jl
```

# How to use

## File loading
## ILDG format
[ILDG](https://www-zeuthen.desy.de/~pleiter/ildg/ildg-file-format-1.1.pdf) format is one of standard formats for LatticeQCD configurations.

We can read ILDG format like: 

```julia
using Gaugefields

NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3
Nwing = 1
Dim = 4

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

filename = "hoge.ildg"
ildg = ILDG(filename)
i = 1
L = [NX,NY,NZ,NT]
load_gaugefield!(U,i,ildg,L,NC)
```
Then, we can calculate the plaquette: 

```julia
temps = Temporalfields(U[1], num=2)
comb, factor = set_comb(U,Dim)

@time plaq_t = calculate_Plaquette(U,temps)*factor
println("plaq_t = $plaq_t")
poly = calculate_Polyakov_loop(U,temps) 
println("polyakov loop = $(real(poly)) $(imag(poly))")
```

We can write a configuration as the ILDG format like 

```julia
filename = "hoge.ildg"
save_binarydata(U,filename)
```

## Text format for Bridge++
Gaugefields.jl also supports a text format for [Bridge++](https://bridge.kek.jp/Lattice-code/index_e.html). 

### File loading

```julia
using Gaugefields

filename = "testconf.txt"
load_BridgeText!(filename,U,L,NC)
```

### File saving

```julia
filename = "testconf.txt"
save_textdata(U,filename)
```

## JLD2 format
Gaugefields.jl also supports [JLD2 format](https://github.com/JuliaIO/JLD2.jl).

### File saving and loading

```julia
function main()
using Gaugefields

function savingexample()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    Nwing = 0
    Dim = 4

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U,Dim)

    @time plaq_t = calculate_Plaquette(U, temps) * factor
    println("plaq_t = $plaq_t")

    filename = "test.jld2"
    saveU(filename, U)
end

function loadingexample()
    filename = "test.jld2"
    U = loadU(filename)

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U,Dim)

    @time plaq_t = calculate_Plaquette(U, temps) * factor
    println("plaq_t = $plaq_t")
end

savingexample()
loadingexample()
```



## Z(Nc) 2-form gauge fields

SU(N) gauge fields possess Z(N) center symmetry,
which is called 1-form global symmetry, a type of generalized symmetry.
To gauge the 1-form center symmetry,
we can define the Z(N) 2-form gauge fields in four dimensions, B, as
```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 0
NC = 3

flux=[1,0,0,0,0,1] # FLUX=[Z12,Z13,Z14,Z23,Z24,Z34]

println("Flux is ", flux)

B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")

println("Initial conf of B at [1,2][2,2,:,:,NZ,NT]")
display(B[1,2][2,2,:,:,NZ,NT])
```

## Heatbath updates
### Even-odd method

```julia
using Gaugefields

function heatbath_SU3!(U,NC,temps_g,β)
    Dim = 4
    V, it_V = get_temp(temps_g)
    ITERATION_MAX = 10^5

    temps, it_temps = get_temp(temps_g, 5)

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

        evaluate_gaugelinks_evenodd!(V,loops,U,temps,iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 

        iseven = false
        evaluate_gaugelinks_evenodd!(V,loops,U,temps,iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 
    end

    unused!(temps_g, it_V)
    unused!(temps_g, it_temps)
end

function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 1

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    temps = Temporalfields(U[1], num=5)
    comb, factor = set_comb(U,Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 40
    for itrj = 1:numhb
        heatbath_SU3!(U,NC,temps,β)

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
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

### Heatbath updates with general actions
We can do heatbath updates with a general action.

```julia
using Gaugefields

function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 1

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    println(typeof(U))

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette",Dim=Dim)
    append!(plaqloop,plaqloop')
    βinp = β/2
    push!(gauge_action,βinp,plaqloop)

    rectloop = make_loops_fromname("rectangular",Dim=Dim)
    append!(rectloop,rectloop')
    βinp = β/2
    push!(gauge_action,βinp,rectloop)

    hnew = Heatbath_update(U,gauge_action)

    show(gauge_action)

    temps = Temporalfields(U[1], num=9)
    comb, factor = set_comb(U,Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps)
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 1000
    for itrj = 1:numhb

        heatbath!(U,hnew)

        plaq_t = calculate_Plaquette(U,temps)*factor
        poly = calculate_Polyakov_loop(U,temps) 

        if itrj % 40 == 0
            println("$itrj plaq_t = $plaq_t")
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    
    return plaq_t

end

NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3
β = 5.7
heatbathtest_4D(NX,NY,NZ,NT,β,NC)
```

In this code, we consider the plaquette and rectangular actions. 

### Heatbath coupled with B fields
```julia
using Gaugefields

function heatbath_SU3!(U,B,NC,temps_g,β)
    Dim = 4
    V, it_V = get_temp(temps_g)
    ITERATION_MAX = 10^5

    temps, it_temps = get_temp(temps_g, 5)

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

        evaluate_gaugelinks_evenodd!(V,loops,U,B,temps,iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 

        iseven = false
        evaluate_gaugelinks_evenodd!(V,loops,U,B,temps,iseven)
        map_U!(U[μ],mapfunc!,V,iseven) 
    end

    unused!(temps_g, it_V)
    unused!(temps_g, it_temps)
end

function heatbathtest_4D_b(NX,NY,NZ,NT,β,NC,Flux)
    Dim = 4
    Nwing = 1

    flux = Flux

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    temps = Temporalfields(U[1], num=5)
    comb, factor = set_comb(U,Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 40
    for itrj = 1:numhb
        heatbath_SU3!(U,NC,temps,β)

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
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
flux = [1,0,0,0,0,1] # FLUX=[Z12,Z13,Z14,Z23,Z24,Z34]
@time plaq_t = heatbathtest_4D_b(NX,NY,NZ,NT,β,NC,flux)
```

## Gradient flow
We can use Lüscher's gradient flow.

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
NC = 3

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")

temps = Temporalfields(U[1], num=3)
comb, factor = set_comb(U,Dim)

g = Gradientflow(U)
for itrj=1:100
    flow!(U,g)
    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("$itrj plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
end

```

For a theory coupled with B fields,
```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 0
NC = 3

flux=[1,0,0,0,0,1] # FLUX=[Z12,Z13,Z14,Z23,Z24,Z34]

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")


temps = Temporalfields(U[1], num=3)
comb, factor = set_comb(U,Dim)

g = Gradientflow(U, B)
for itrj=1:100
    flow!(U,B,g)
    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("$itrj plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
end

```

## Hybrid Monte Carlo
### HMC for SU(Nc) gauge theory
We can do the HMC simulations. The example code is as follows.
```julia

using Random
using Gaugefields
using LinearAlgebra

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, temps)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function HMC_test_4D(NX,NY,NZ,NT,NC,β)
    Dim = 4
    Nwing = 0

    Random.seed!(123)

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")
    #"Reproducible"
    println(typeof(U))

    temps = Temporalfields(U[1], num=6)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    MDsteps = 100
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temps)
        end
        if get_myrank(U) == 0
            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,temps)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end
    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 5.7
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    NC = 3
    HMC_test_4D(NX,NY,NZ,NT,NC,β)
end
main()
```

### Non-dynamical higher-form gauge fields
We can do the HMC simulations with B fields. The example code is as follows.
```julia

using Random
using Gaugefields
using LinearAlgebra

function MDstep!(gauge_action, U, B, p, MDsteps, Dim, Uold, temps)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, B, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, B, p, 1.0, Δτ, Dim, gauge_action, temps)

        U_update!(U,    p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, B, p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))
    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
    Dim = 4
    Nwing = 0

    flux = Flux
    println("Flux : ", flux)

    Random.seed!(123)


    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    temps = Temporalfields(U[1], num=9)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    MDsteps = 50
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temps)
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 5.7
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    Flux = [0,0,1,1,0,0]
    #HMC_test_4D(NX,NY,NZ,NT,NC,β)
    HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
end
main()
```

### Dynamical higher-form gauge fields
HMC simulations with dynamical B fields are as follows:
```julia

using Random
using Gaugefields
using Wilsonloop
using LinearAlgebra

function MDstep!(
    gauge_action,
    U,
    B,
    flux,
    p,
    MDsteps, # MDsteps should be an even integer
    Dim,
    Uold,
    Bold,
    flux_old,
    temps
) # Halfway-updating HMC
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temps)

        U_update!(U,  p,0.5,Δτ,Dim,gauge_action,temps)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,flux)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end

function MDstep!(
    gauge_action,
    U,
    B,
    flux,
    p,
    MDsteps,
    num_HMC,
    Dim,
    Uold1,
    Uold2,
    Bold,
    flux_old,
    temps
) # Double-tesing HMC
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update!(B,flux)

    for ihmc=1:num_HMC
        MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold2,temps)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    #println("Sold = $Sold, Snew = $Snew")
    #println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold1)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end

function HMC_test_4D_dynamicalB(NX,NY,NZ,NT,NC,β)
    Dim = 4
    Nwing = 0

    Random.seed!(123)

    flux = [1,1,1,1,2,0]

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    L = [NX,NY,NZ,NT]
    filename = "test/confs/U_beta6.0_L8_F111120_4000.txt"
    load_BridgeText!(filename,U,L,NC)

    temps = Temporalfields(U[1], num=9)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U)
    Uold  = similar(U)
    Bold = similar(B)
    flux_old = zeros(Int, 6)

    MDsteps = 50 # even integer!!!
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(
                gauge_action,
                U,
                B,
                flux,
                p,
                MDsteps,
                Dim,
                Uold,
                Bold,
                flux_old,
                temps
            )
        end
        if get_myrank(U) == 0
             println("Flux : ", flux)
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temps)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 6.0
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    NC = 3
    HMC_test_4D_dynamicalB(NX,NY,NZ,NT,NC,β)
end
main()
```

## Gradient flow with general terms
We can do the gradient flow with general terms with the use of Wilsonloop.jl, which is shown below.
The coefficient of the action can be complex. The complex conjugate of the action defined here is added automatically to make the total action hermitian.   
The code is 

```julia

using Random
using Test
using Gaugefields
using Wilsonloop

function gradientflow_test_4D(NX,NY,NZ,NT,NC)
    Dim = 4
    Nwing = 1

    Random.seed!(123)

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #Plaquette term
    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)
        end
    end

    #Rectangular term
    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
            loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
            
            push!(loops,loop1)
        end
    end

    listloops = [loops_p,loops]
    listvalues = [1+im,0.1]
    g = Gradientflow_general(U,listloops,listvalues,eps = 0.01)

    for itrj=1:100
        flow!(U,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return plaq_t

end


function gradientflow_test_2D(NX,NT,NC)
    Dim = 2
    Nwing = 1
    U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot",randomnumber="Reproducible")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #g = Gradientflow(U,eps = 0.01)
    #listnames = ["plaquette"]
    #listvalues = [1]
    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end

            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)

        end
    end


    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
            loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
            
            push!(loops,loop1)
        end
    end

    listloops = [loops_p,loops]
    listvalues = [1+im,0.1]
    g = Gradientflow_general(U,listloops,listvalues,eps = 0.01)

    for itrj=1:100
        flow!(U,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end

    return plaq_t

end



const eps = 0.1


println("2D system")
@testset "2D" begin
    NX = 4
    #NY = 4
    #NZ = 4
    NT = 4
    Nwing = 1

    @testset "NC=1" begin
        β = 2.3
        NC = 1
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end
    #error("d")
    
    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end
end

println("4D system")
@testset "4D" begin
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1


    
    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")

        val = 0.7301232810349298
        @time plaq_t =gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end


end
```

```julia

using Random
using Test
using Gaugefields
using Wilsonloop

function gradientflow_test_4D(NX,NY,NZ,NT,NC)
    Dim = 4
    Nwing = 0

    flux = [0,0,1,1,0,0]

    Random.seed!(123)

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #Plaquette term
    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)
        end
    end

    #Rectangular term
    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
            loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
            
            push!(loops,loop1)
        end
    end

    listloops = [loops_p,loops]
    listvalues = [1+im,0.1]
    g = Gradientflow_general(U,B,listloops,listvalues,eps = 0.1)

    for itrj=1:10
        flow!(U,B,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return plaq_t

end

const eps = 0.1

println("4D system")
@testset "4D" begin
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1

    
    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")

        val = 0.7301232810349298
        @time plaq_t =gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end


end
```


## HMC with MPI
Here, we show the HMC with MPI.
the REPL and Jupyternotebook can not be used when one wants to use MPI.
At first, in Julia REPL in the package mode,
```
add MPI
```
Then,
```julia
using MPI
MPI.install_mpiexecjl()
```
and
```
export PATH="/<your home path>/.julia/bin/:$PATH"
```

The command is like:
```
mpiexecjl -np 2 julia mpi_sample.jl 1 1 1 2 true
```
```1 1 1 2``` means ```PEX PEY PEZ PET```. In this case, the time-direction is diveded by 2. 

The sample code is written as 
```julia

using Random
using Gaugefields
using LinearAlgebra
using MPI

if length(ARGS) < 5
    error("USAGE: ","""
    mpiexecjl -np 2 exe.jl 1 1 1 2 true
    """)
end
const pes = Tuple(parse.(Int64,ARGS[1:4]))
const mpi = parse(Bool,ARGS[5])

function MDstep!(gauge_action, U, p, MDsteps, Dim, Uold, temps)
    Δτ = 1.0 / MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action, U, p)
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)

        P_update!(U, p, 1.0, Δτ, Dim, gauge_action, temps)

        U_update!(U, p, 0.5, Δτ, Dim, gauge_action, temps)
    end
    Snew = calc_action(gauge_action, U, p)
    if get_myrank(U)==0 && displayon
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1, exp(-Snew + Sold))
    r = rand()
    if mpi
        r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    end
    if r > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function HMC_test_4D(NX,NY,NZ,NT,NC,β)
    Dim = 4
    Nwing = 0

    Random.seed!(123)

    if mpi
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",mpi=true,PEs = PEs,mpiinit = false) 
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    end

    if get_myrank(U) == 0
        println(typeof(U))
    end

    temps = Temporalfields(U[1], num=10)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    if get_myrank(U) == 0
        println("0 plaq_t = $plaq_t")
    end
    poly = calculate_Polyakov_loop(U,temps) 
    if get_myrank(U) == 0
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")
    end

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    MDsteps = 100
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temps)
        end
        if get_myrank(U) == 0
            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,temps)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            plaq_t = calculate_Plaquette(U,temps)*factor
            if get_myrank(U) == 0
                println("$itrj plaq_t = $plaq_t")
            end
            poly = calculate_Polyakov_loop(U,temps) 
            if get_myrank(U) == 0
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
                println("acceptance ratio ",numaccepted/itrj)
            end
        end
    end


    return plaq_t,numaccepted/numtrj

end



function main()
    β = 5.7
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    NC = 3
    HMC_test_4D(NX,NY,NZ,NT,NC,β)
end
main()
```

Also we can implement higher-form gauge fields.

# Utilities

## Data structure
We can access the gauge field defined on the bond between two neigbohr points. 
In 4D system, the gauge field is like ```u[ic,jc,ix,iy,iz,it]```. 
There are four directions in 4D system. Gaugefields.jl uses the array like: 

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3
Nwing = 1
Dim = 4

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

```

In the later exaples, we use, ``mu=1`` and ``u=U[mu]`` as an example.

## Hermitian conjugate (Adjoint operator)
If you want to get the hermitian conjugate of the gauge fields, you can do like 

```julia
u'
```

This is evaluated with the lazy evaluation. 
So there is no memory copy. 
This returms $U_\mu^\dagger$ for all sites.

## Shift operator
If you want to shift the gauge fields, you can do like 

```julia
shifted_u = shift_U(u, shift)
```
This is also evaluated with the lazy evaluation. 
Here ``shift`` is ``shift=(1,0,0,0)`` for example.

## Evaluate Wilson links
Here the example to evaluate the Wilson links.

```julia
using Gaugefields
using Wilsonloop
function main()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    NC = 3

    U1 = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")

    temps = typeof(U1[1])[]
    for i=1:10
        push!(temps,similar(U1[1]))
    end

    loop = [(1,+1),(2,+1),(1,-1),(2,-1)]
    println(loop)
    w = Wilsonline(loop)
    println("P: ")
    show(w)

    Uloop = similar(U1[1])

    Gaugefields.evaluate_gaugelinks!(Uloop, w, U1, temps)
    display(Uloop[:,:,1,1,1,1])
end
main()
```

## matrix-field matrix-field product
If you want to calculate the matrix-matrix multiplicaetion on each lattice site, you can do like

As a mathematical expression, for matrix-valued fields ``A(n), B(n)``,
we define "matrix-field matrix-field product" as,

```math
[A(n)B(n)]_{ij} = \sum_k [A(n)]_{ik} [B(n)]_{kj}
```

for all site index n.
<!--<img src="https://latex.codecogs.com/svg.image?[A(n)B(n)]_{ij}&space;=&space;\sum_k&space;[A(n)]_{ik}&space;[B(n)]_{kj}" title="[A(n)B(n)]_{ij} = \sum_k [A(n)]_{ik} [B(n)]_{kj}" />-->

In our package, this is expressed as,

```julia
mul!(C,A,B)
```
which means ```C = A*B``` on each lattice site. 
Here ``A, B, C`` are same type of ``u``.

## Trace operation 
If you want to calculate the trace of the gauge field, you can do like 

```julia
tr(A)
```
It is useful to evaluation actions. 
This trace operation summing up all indecis, spacetime and color.

# Applications

This package and Wilsonloop.jl enable you to perform several calcurations.
Here we demonstrate them.

Some of them will be simplified in LatticeQCD.jl.

## Wilson loops
We develop [Wilsonloop.jl](https://github.com/akio-tomiya/Wilsonloop.jl.git), which is useful to calculate Wilson loops. 
If you want to use this, please install like

```
add Wilsonloop.jl
```

For example, if you want to calculate the following quantity: 

```math
U_{1}(n)U_{2}(n+\hat{1}) U^{\dagger}_{1}(n+\hat{2}) U^{\dagger}_2(n)
```
or
```math
U_{1}(n)U_{2}(n+\hat{1}) U^{\dagger}_{1}(n+\hat{2}) U^{\dagger}_2(n) e^{-2\pi B_{12}(n) / N} ,
```
which is Z(Nc) 1-form gauge invariant [[arXiv:2303.10977 [hep-lat]](https://arxiv.org/abs/2303.10977)].

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
U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

temp1 = similar(U[1])
temp2 = similar(U[1])
temp3 = similar(U[1])
V = similar(U[1])

evaluate_gaugelinks!(V,w,U,[temp1,temp2,temp3])
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

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    filename = "./conf_00000010.txt" 
    L = [NX,NY,NZ,NT]
    load_BridgeText!(filename,U,L,NC) # We load a configuration from a file. 

    temps_g = Temporalfields(U[1], num=5)

    println("Make clover operator")
    G = Array{typeof(u1),2}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            G[μ,ν] = similar(U[1])
        end
    end

    comb, factor = set_comb(U, Dim)
    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("plaq_t = $plaq_t")

    temps, it_temps = get_temp(temps_g, 3)

    g = Gradientflow(U,eps = 0.01)
    for itrj=1:100
        flow!(U,g)

        make_clover(G,U,temps,Dim)
        E = calc_energydensity(G,U,temps,Dim)

        plaq_t = calculate_Plaquette(U,temps_g)*factor
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


## Calculating actions
We can calculate actions from this packages with fixed gauge fields U. 
We introduce the concenpt "Scalar-valued neural network", which is S(U) -> V, where U and V are gauge fields. 


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

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U) #empty network
    plaqloop = make_loops_fromname("plaquette") #This is a plaquette loops. 
    append!(plaqloop,plaqloop') #We need hermitian conjugate loops for making the action real. 
    β = 1 #This is a coefficient.
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    Uout = evaluate_Gaugeaction_untraced(gauge_action,U)
    println(tr(Uout))
end

test1()
```

The output is 

```
----------------------------------------------
Structure of the actions for Gaugefields
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


## Fractional topological charge
```julia
function calculate_topological_charge_plaq(U::Array{T,1}, B::Array{T,2}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "plaq", U, B, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end
function calculate_topological_charge_clover(U::Array{T,1}, B::Array{T,2}, temp_UμνTA, temps) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "clover", U, B, temps)
    Q = calc_Q(UμνTA, numofloops, U)
    return Q
end
function calculate_topological_charge_improved(
    U::Array{T,1},
    B::Array{T,2},
    temp_UμνTA,
    Qclover,
    temps,
) where {T}
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA, "rect", U, B, temps)
    Qrect = 2 * calc_Q(UμνTA, numofloops, U)
    c1 = -1 / 12
    c0 = 5 / 3
    Q = c0 * Qclover + c1 * Qrect
    return Q
end
function calc_UμνTA!(
    temp_UμνTA,
    name::String,
    U::Array{T,1},
    B::Array{T,2},
    temps,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    loops_μν, numofloops = calc_loopset_μν_name(name, Dim)
    calc_UμνTA!(temp_UμνTA, loops_μν, U, B, temps)
    return numofloops
end
function calc_UμνTA!(
    temp_UμνTA,
    loops_μν,
    U::Array{T,1},
    B::Array{T,2},
    temps,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    UμνTA = temp_UμνTA
    for μ = 1:Dim
        for ν = 1:Dim
            if ν == μ
                continue
            end
            evaluate_gaugelinks!(temps[1], loops_μν[μ, ν], U, B, temps[2:6])
            Traceless_antihermitian!(UμνTA[μ, ν], temps[1])
        end
    end
    return
end

#=
implementation of topological charge is based on
https://arxiv.org/abs/1509.04259
=#
function calc_Q(UμνTA, numofloops, U::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    Q = 0.0
    if Dim == 4
        ε(μ, ν, ρ, σ) = epsilon_tensor(μ, ν, ρ, σ)
    else
        error("Dimension $Dim is not supported")
    end
    for μ = 1:Dim
        for ν = 1:Dim
            if ν == μ
                continue
            end
            Uμν = UμνTA[μ, ν]
            for ρ = 1:Dim
                for σ = 1:Dim
                    if ρ == σ
                        continue
                    end
                    Uρσ = UμνTA[ρ, σ]
                    s = tr(Uμν, Uρσ)
                    Q += ε(μ, ν, ρ, σ) * s / numofloops^2
                end
            end
        end
    end
    return -Q / (32 * (π^2))
end
#topological charge
function epsilon_tensor(mu::Int, nu::Int, rho::Int, sigma::Int)
    sign = 1 # (3) 1710.09474 extended epsilon tensor
    if mu < 0
        sign *= -1
        mu = -mu
    end
    if nu < 0
        sign *= -1
        nu = -nu
    end
    if rho < 0
        sign *= -1
        rh = -rho
    end
    if sigma < 0
        sign *= -1
        sigma = -sigma
    end
    epsilon = zeros(Int, 4, 4, 4, 4)
    epsilon[1, 2, 3, 4] = 1
    epsilon[1, 2, 4, 3] = -1
    epsilon[1, 3, 2, 4] = -1
    epsilon[1, 3, 4, 2] = 1
    epsilon[1, 4, 2, 3] = 1
    epsilon[1, 4, 3, 2] = -1
    epsilon[2, 1, 3, 4] = -1
    epsilon[2, 1, 4, 3] = 1
    epsilon[2, 3, 1, 4] = 1
    epsilon[2, 3, 4, 1] = -1
    epsilon[2, 4, 1, 3] = -1
    epsilon[2, 4, 3, 1] = 1
    epsilon[3, 1, 2, 4] = 1
    epsilon[3, 1, 4, 2] = -1
    epsilon[3, 2, 1, 4] = -1
    epsilon[3, 2, 4, 1] = 1
    epsilon[3, 4, 1, 2] = 1
    epsilon[3, 4, 2, 1] = -1
    epsilon[4, 1, 2, 3] = -1
    epsilon[4, 1, 3, 2] = 1
    epsilon[4, 2, 1, 3] = 1
    epsilon[4, 2, 3, 1] = -1
    epsilon[4, 3, 1, 2] = -1
    epsilon[4, 3, 2, 1] = 1
    return epsilon[mu, nu, rho, sigma] * sign
end
function calc_loopset_μν_name(name, Dim)
    loops_μν = Array{Vector{Wilsonline{Dim}},2}(undef, Dim, Dim)
    if name == "plaq"
        numofloops = 1
        for μ = 1:Dim
            for ν = 1:Dim
                loops_μν[μ, ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                plaq = make_plaq(μ, ν, Dim = Dim)
                push!(loops_μν[μ, ν], plaq)
            end
        end
    elseif name == "clover"
        numofloops = 4
        for μ = 1:Dim
            for ν = 1:Dim
                loops_μν[μ, ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                loops_μν[μ, ν] = make_cloverloops_topo(μ, ν, Dim = Dim)
            end
        end
    elseif name == "rect"
        numofloops = 8
        for μ = 1:4
            for ν = 1:4
                if ν == μ
                    continue
                end
                loops = Wilsonline{Dim}[]
                loop_righttop = Wilsonline([(μ, 2), (ν, 1), (μ, -2), (ν, -1)])
                loop_lefttop = Wilsonline([(ν, 1), (μ, -2), (ν, -1), (μ, 2)])
                loop_rightbottom = Wilsonline([(ν, -1), (μ, 2), (ν, 1), (μ, -2)])
                loop_leftbottom = Wilsonline([(μ, -2), (ν, -1), (μ, 2), (ν, 1)])
                push!(loops, loop_righttop)
                push!(loops, loop_lefttop)
                push!(loops, loop_rightbottom)
                push!(loops, loop_leftbottom)
                loop_righttop = Wilsonline([(μ, 1), (ν, 2), (μ, -1), (ν, -2)])
                loop_lefttop = Wilsonline([(ν, 2), (μ, -1), (ν, -2), (μ, 1)])
                loop_rightbottom = Wilsonline([(ν, -2), (μ, 1), (ν, 2), (μ, -1)])
                loop_leftbottom = Wilsonline([(μ, -1), (ν, -2), (μ, 1), (ν, 2)])
                push!(loops, loop_righttop)
                push!(loops, loop_lefttop)
                push!(loops, loop_rightbottom)
                push!(loops, loop_leftbottom)
                loops_μν[μ, ν] = loops
            end
        end
    else
        error("$name is not supported")
    end
    return loops_μν, numofloops
end
function make_cloverloops_topo(μ, ν; Dim = 4)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)])
    loop_lefttop = Wilsonline([(ν, 1), (μ, -1), (ν, -1), (μ, 1)])
    loop_rightbottom = Wilsonline([(ν, -1), (μ, 1), (ν, 1), (μ, -1)])
    loop_leftbottom = Wilsonline([(μ, -1), (ν, -1), (μ, 1), (ν, 1)])
    push!(loops, loop_righttop)
    push!(loops, loop_lefttop)
    push!(loops, loop_rightbottom)
    push!(loops, loop_leftbottom)
    return loops
end
```
We can calculate the topological charge as
```Qplaq = calculate_topological_charge_plaq(U,B,temp_UμνTA,temps[1:6])```,
```Qclover = calculate_topological_charge_clover(U,B,temp_UμνTA,temps[1:6])```,
```Qimproved= calculate_topological_charge_improved(U,B,temp_UμνTA,Qclover,temps[1:6])```.


# How to calculate derivatives
We can easily calculate the matrix derivative of the actions. The matrix derivative is defined as 

```math
[\frac{\partial S}{\partial U_{\mu}(n)}]_{ij} = \frac{\partial S}{\partial U_{\mu,ji}(n)}
```

<!--<img src="https://latex.codecogs.com/svg.image?[\frac{\partial&space;S}{\partial&space;U_{\mu}(n)}]_{ij}&space;=&space;\frac{\partial&space;S}{\partial&space;U_{\mu,ji}(n)}" title="[\frac{\partial S}{\partial U_{\mu}(n)}]_{ij} = \frac{\partial S}{\partial U_{\mu,ji}(n)}" />-->


We can calculate this like 

```julia
dSdUμ = calc_dSdUμ(gauge_action,μ,U)
```

or

```julia
calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
```

## Hybrid Monte Carlo

With the use of the matrix derivative, we can do the Hybrid Monte Carlo method. 
The simple code is as follows. 

```julia
using Gaugefields
using LinearAlgebra

function MDtest!(gauge_action,U,Dim)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    MDsteps = 100
    temps = Temporalfields(U[1], num=10)
    comb, factor = set_comb(U, Dim)
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temps)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temps)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end
```

We define the functions as 

```julia

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_Gaugeaction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temps)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action,temps)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action,temps)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action,temps)
    end
    Snew = calc_action(gauge_action,U,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action,temps)
    temp1, it_temp1 = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    expU, it_expU = get_temp(temps)
    W, it_W = get_temp(temps)

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    unused!(temps, it_expU)
    unused!(temps, it_W)
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action,temps) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp1, it_temp1 = get_temp(temps)
    dSdUμ, it_dSdUμ = get_temp(temps)
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temp1,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp1)
    end
    unused!(temps, it_dSdUμ)
    unused!(temps, it_temp1)
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

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop') # add hermitian conjugate
    β = 5.7/2 # real part; re[p] = (p+p')/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    MDtest!(gauge_action,U,Dim)

end


test1()
```

## Stout smearing
We can use stout smearing. 

```math
U_{\rm fat} = {\cal F}(U)
```

<!--<img src="https://latex.codecogs.com/svg.image?U_{\rm&space;fat}&space;=&space;{\cal&space;F}(U)" title="U_{\rm fat} = {\cal F}(U)" />-->

The smearing is regarded as gauge covariant neural networks [Tomiya and Nagai, arXiv:2103.11965](https://arxiv.org/abs/2103.11965). 
The network is constructed as follows. 

```julia
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    L = [NX,NY,NZ,NT]

    nn = CovNeuralnet()
    ρ = [0.1]
    layername = ["plaquette"]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)

    show(nn)
```

The output is 

```
num. of layers: 1
- 1-st layer: STOUT
num. of terms: 1
-------------------------------
      1-st term: 
          coefficient: 0.1
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
      -------------------------
```

Since we ragard the smearing as the neural networks, we can calculate the derivative with the use of the back propergation techques. 

```math
\frac{\partial S}{\partial U} = G \left( \frac{dS}{dU_{\rm fat}},U \right)
```

<!--<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;S}{\partial&space;U}&space;=&space;G&space;\left(&space;\frac{dS}{dU_{\rm&space;fat}},U&space;\right)" title="\frac{\partial S}{\partial U} = G \left( \frac{dS}{dU_{\rm fat}},U \right)" />-->

For example, 

```julia
using Gaugefields
using Wilsonloop

function stoutsmearing(NX,NY,NZ,NT,NC)
    Nwing = 1
    Dim = 4
    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    L = [NX,NY,NZ,NT]

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println(" plaq_t = $plaq_t")


    nn = CovNeuralnet()
    ρ = [0.1]
    layername = ["plaquette"]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)

    show(nn)

    
    @time Uout,Uout_multi,_ = calc_smearedU(U,nn)
    plaq_t = calculate_Plaquette(Uout,temp1,temp2)*factor
    println("plaq_t = $plaq_t")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')# add hermitian conjugate
    β = 5.7/2 # real part; re[p] = (p+p')/2
    push!(gauge_action,β,plaqloop)

    μ = 1
    dSdUμ = similar(U)
    for μ=1:Dim
        dSdUμ[μ] = calc_dSdUμ(gauge_action,μ,U)
    end

    @time dSdUbareμ = back_prop(dSdUμ,nn,Uout_multi,U) 

end

NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3
stoutsmearing(NX,NY,NZ,NT,NC)
```

# HMC with stout smearing
With the use of the derivatives, we can do the HMC with the stout smearing. 
The code is shown as follows

```julia
using Gaugefields
using LinearAlgebra

function MDtest!(gauge_action,U,Dim,nn)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    dSdU = similar(U)
    
    substitute_U!(Uold,U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,nn,dSdU)
    

    Δτ = 1/MDsteps
    gauss_distribution!(p)

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    Sold = calc_action(gauge_action,Uout,p)

    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action,dSdU,nn)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end

    Uout,Uout_multi,_ = calc_smearedU(U,nn)
    Snew = calc_action(gauge_action,Uout,p)

    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")

    accept = exp(Sold - Snew) >= rand()

    if accept != true #rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end

end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
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

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action,dSdU,nn) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor =  -ϵ*Δτ/(NC)
    temps = get_temporary_gaugefields(gauge_action)
    Uout,Uout_multi,_ = calc_smearedU(U,nn)

    for μ=1:Dim
        calc_dSdUμ!(dSdU[μ],gauge_action,μ,Uout)
    end

    dSdUbare = back_prop(dSdU,nn,Uout_multi,U) 
    
    for μ=1:Dim
        mul!(temps[1],U[μ],dSdUbare[μ]) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)

    show(gauge_action)

    L = [NX,NY,NZ,NT]
    nn = CovNeuralnet()
    ρ = [0.1]
    layername = ["plaquette"]
    st = STOUT_Layer(layername,ρ,L)
    push!(nn,st)

    

    MDtest!(gauge_action,U,Dim,nn)

end


test1()
```

# Acknowledgment
If you write a paper using this package, please refer this code.

BibTeX citation is following
```
@article{Nagai:2024yaf,
    author = "Nagai, Yuki and Tomiya, Akio",
    title = "{JuliaQCD: Portable lattice QCD package in Julia language}",
    eprint = "2409.03030",
    archivePrefix = "arXiv",
    primaryClass = "hep-lat",
    month = "9",
    year = "2024"
}
```
and the paper is [arXiv:2409.03030](https://arxiv.org/abs/2409.03030).
