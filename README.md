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
