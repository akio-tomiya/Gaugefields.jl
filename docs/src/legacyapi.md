# Legacy API

!!! note "Compatibility API"
    The APIs documented on this page remain available in Gaugefields v1 for
    compatibility with existing programs. They are classified as legacy APIs:
    new applications should use the [recommended high-level API](highlevelapi.md)
    and the [four-dimensional quick start](tutorial4d.md).

    Compatibility means that existing calls continue to be readable and usable.
    New features and examples are documented against the v1 API, and legacy APIs
    may be deprecated in a future major release only after a separate migration
    policy is announced.

This page preserves the former README reference and examples in one place.
In particular, constructor flags such as `cuda`, `mpi`, `accelerator`, and
`isMPILattice`, and low-level routines such as `Initialize_Gaugefields`,
`MDstep!`, `U_update!`, and `P_update!` belong to the compatibility API.

```@docs
Initialize_Gaugefields
```

## Migration map

The compatibility API remains callable, but new code should use the v1 entry
points in the right column.

| Pre-v1 API | Recommended v1 API |
| --- | --- |
| `Initialize_Gaugefields(...)` | `gauge_configuration(lattice; ...)` |
| `Nwing` or `NDW` | `halo` |
| `condition="cold"/"hot"` | `start=:cold` or `start=:hot` |
| `randomnumber="Reproducible"` | explicit `seed` and `rng` |
| `mpi`, `PEs`, `isMPILattice` | LM default plus `process_grid` |
| `cuda` and `accelerator` | select the JACC backend outside Gaugefields |
| `calculate_Plaquette` | `measure_plaquette` |
| `calculate_Polyakov_loop` | `measure_polyakov_loop` |
| `Wilsonline`, `evaluate_gaugelinks!` | same path API; see the current Wilson-loop guide |
| `GaugeAction`, `evaluate_GaugeAction`, `calc_dSdUμ!` | same action API; see the current Wilson-loop guide |
| `initialize_TA_Gaugefields` | `gauge_momenta` |
| `gauss_distribution!` | `gaussian_momenta` |
| `Gradientflow` and `Gradientflow_general` | `gradient_flow` |
| `Heatbath` and `Heatbath_update` | `heatbath_updater` |
| `CovNeuralnet`, `STOUT_Layer`, `calc_smearedU` | `stout_smearing` and `smear` |
| `saveU`, `loadU`, format-specific I/O | `save_configuration` and `load_configuration!` |
| application-defined `MDstep!`, `U_update!`, `P_update!` | `md_driver`, `md_trajectory!`, and explicit v1 update functions |
| `topological_charge_density` and site-density analysis | QCDMeasurements.jl measurement interfaces |

Some advanced compatibility routines, including energy-density and
topological-charge helpers, remain callable but should not be presented as the
primary v1 interface. New topological-charge-density analysis should use
[QCDMeasurements.jl](https://github.com/akio-tomiya/QCDMeasurements.jl); see
the current [Measurements](measurements.md#Advanced-observables) guide.

`Wilsonline`, `GaugeAction`, action evaluation, and analytic action
derivatives remain part of the current API even though historical examples
also appear below. Their maintained documentation is
[Wilson loops and gauge actions](wilsonloops_actions.md).

# Additional archived compatibility material

The following sections preserve material that previously lived on separate
manual pages. They intentionally use the compatibility API.

## Former measurements manual

We show examples of measurements.

### Plaquette and Polyakov loops
This is the example to measure Plaquette and Polyakov loop observable.

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

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 1000
    for itrj = 1:numhb

        heatbath!(U,hnew)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        poly = calculate_Polyakov_loop(U,temp1,temp2)

        if itrj % 40 == 0
            println("$itrj plaq_t = $plaq_t")
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end

    #close(fp)
    filename = "hoge.ildg"
    save_binarydata(U,filename)
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

### Energy density
We show the code to measure the energy density.

This is the code example:

```julia
using Gaugefields
using Wilsonloop
using LinearAlgebra

function make_cloverloops(μ,ν;Dim=4)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)])
    loop_lefttop = Wilsonline([(ν,1),(μ,-1),(ν,-1),(μ,1)])
    loop_rightbottom = Wilsonline([(ν,-1),(μ,1),(ν,1),(μ,-1)])
    loop_leftbottom= Wilsonline([(μ,-1),(ν,-1),(μ,1),(ν,1)])
    push!(loops,loop_righttop)
    push!(loops,loop_lefttop)
    push!(loops,loop_rightbottom)
    push!(loops,loop_leftbottom)
    return loops
end

function cloverloops(Dim)
    loops_μν= Matrix{Vector{Wilsonline{Dim}}}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            loops_μν[μ,ν] = Wilsonline{Dim}[]
            if ν == μ
                continue
            end
            loops_μν[μ,ν] = make_cloverloops(μ,ν,Dim=Dim)
        end
    end
    return  loops_μν
end

function make_energy_density!(Wmat,U::Vector{<: AbstractGaugefields{NC,Dim}},temps) where {NC,Dim}
    W_operator = cloverloops(Dim)
    calc_wilson_loop!(Wmat,W_operator,U,temps)
    return
end


function calc_wilson_loop!(W,W_operator,U::Vector{<: AbstractGaugefields{NC,Dim}},temps) where {NC,Dim}
    for μ=1:Dim
        for ν=1:Dim
            if μ == ν
                continue
            end
            evaluate_gaugelinks!(W[μ,ν],W_operator[μ,ν],U,temps)
            W[μ,ν] = Traceless_antihermitian(W[μ,ν])
        end
    end
    return
end

function  make_energy_density_core(Wmat::Matrix{<: AbstractGaugefields{NC,Dim}}) where {NC,Dim}
    @assert Dim == 4
    W = 0.0 + 0.0im
    for μ=1:Dim # all directions
        for ν=1:Dim
            if μ == ν
                continue
            end
            W += -tr(Wmat[μ,ν],Wmat[μ,ν])/2
        end
    end
    return W
end


function calculate_energy_density(U::Array{T,1}, Wmat,temps) where T <: AbstractGaugefields
    # Making a ( Ls × Lt) Wilson loop operator for potential calculations
    WL = 0.0+0.0im
    NV = U[1].NV
    NC = U[1].NC
    make_energy_density!(Wmat,U,temps) # make wilon loop operator and evaluate as a field, not traced.
    WL =  make_energy_density_core(Wmat) # tracing over color and average over spacetime and x,y,z.
    return real(WL)/(NV*4^2)
end

function test()
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 0
    NC = 3

    Dim = 4

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    filename="./conf_08080808.ildg"

    ildg = ILDG(filename)
    i = 1
    L = [NX,NY,NZ,NT]
    load_gaugefield!(U,i,ildg,L,NC)

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    β = 5.7
    W_temp = Matrix{typeof(U[1])}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            W_temp[μ,ν] = similar(U[1])
        end
    end

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println(" plaq_t = $plaq_t")

    dt = 0.01

    g = Gradientflow(U,eps = dt)

    for itrj=1:10
        flow!(U,g)
        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        e = calculate_energy_density(U, W_temp,[temp1,temp2])
        println("$itrj $dt $plaq_t $e # itrj dt plaq energy")
    end


end
test()
```



### Topological charge

!!! note "Recommended measurement package"
    This section is retained to explain historical Gaugefields programs. New
    topological-charge and topological-charge-density analyses should use
    [QCDMeasurements.jl](https://github.com/akio-tomiya/QCDMeasurements.jl),
    as described in [Measurements](measurements.md#Advanced-observables).

We show the code to calculate the topological charge.
We show three definitions.

The public API returns either the scalar topological charge `Q` or the
site-wise topological charge density `q(x)`. The scalar result is defined as
the sum of the density with the same method and normalization.

```julia
using Gaugefields

NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3

U = Oneinstanton_SUN_embedded(NC, NX, NY, NZ, NT; block=(1, 2))

q_plaq = topological_charge_density(U; method=:plaquette)
Q_plaq = topological_charge(U; method=:plaquette)

q_clover = topological_charge_density(U; method=:clover)
Q_clover = topological_charge(U; method=:clover)

q_improved = topological_charge_density(U; method=:improved)
Q_improved = topological_charge(U; method=:improved)

@assert size(q_plaq) == (NX, NY, NZ, NT)
@assert size(q_clover) == (NX, NY, NZ, NT)
@assert size(q_improved) == (NX, NY, NZ, NT)
@assert isapprox(sum(q_plaq), Q_plaq; rtol=1e-12, atol=1e-12)
@assert isapprox(sum(q_clover), Q_clover; rtol=1e-12, atol=1e-12)
@assert isapprox(sum(q_improved), Q_improved; rtol=1e-12, atol=1e-12)
```

`method=:plaquette` is the default. `method=:clover` uses the four-loop clover
definition for each ordered direction pair. `method=:improved` uses the
normalization
`q_improved(x) = (5 / 3) * q_clover(x) - (1 / 12) * q_rect(x)`, matching the
existing improved scalar convention in the sample measurement code. The
rectangle density is an internal implementation detail for now; `method=:rect`
is not a public method. These public helpers currently support serial 4D gauge
fields. MPI and accelerator fields are separate topics.

```julia
using Gaugefields
using Wilsonloop
using LinearAlgebra
using Combinatorics

function calculate_topological_charge_plaq(U::Array{T,1},temp_UμνTA,temps) where T
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA,"plaq",U,temps)
    Q = calc_Q(UμνTA,numofloops,U)
    return Q
end

function calculate_topological_charge_clover(U::Array{T,1},temp_UμνTA,temps) where T
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA,"clover",U,temps)
    Q = calc_Q(UμνTA,numofloops,U)
    return Q
end

function calculate_topological_charge_improved(U::Array{T,1},temp_UμνTA,Qclover,temps) where T
    UμνTA = temp_UμνTA
    numofloops = calc_UμνTA!(UμνTA,"rect",U,temps)
    Qrect = 2*calc_Q(UμνTA,numofloops,U)
    c1 = -1/12
    c0 = 5/3
    Q = c0*Qclover + c1*Qrect
    return Q
end

function calc_Q(UμνTA,numofloops,U::Array{<: AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    Q = 0.0
    if Dim == 4
        ε(μ,ν,ρ,σ) = epsilon_tensor(μ,ν,ρ,σ)
    else
        error("Dimension $Dim is not supported")
    end
    for μ=1:Dim
        for ν=1:Dim
            if ν == μ
                continue
            end
            Uμν = UμνTA[μ,ν]
            for ρ =1:Dim
                for σ=1:Dim
                    if ρ == σ
                        continue
                    end
                    Uρσ = UμνTA[ρ,σ]
                    s = tr(Uμν,Uρσ)
                    Q += ε(μ,ν,ρ,σ)*s/numofloops^2
                end
            end
        end
    end

    return -real(Q)/(32*(π^2))
end


function calc_UμνTA!(temp_UμνTA,name::String,U::Array{<: AbstractGaugefields{NC,Dim},1},temps) where {NC,Dim}
    loops_μν,numofloops = calc_loopset_μν_name(name,Dim)
    calc_UμνTA!(temp_UμνTA,loops_μν,U,temps)
    return numofloops
end


function calc_UμνTA!(UμνTA,loops_μν,U::Array{<: AbstractGaugefields{NC,Dim},1},temps) where {NC,Dim}
    for μ=1:Dim
        for ν=1:Dim
            if ν == μ
                continue
            end
            evaluate_gaugelinks!(temps[1],loops_μν[μ,ν],U,temps[2:3])
            Traceless_antihermitian!(UμνTA[μ,ν],temps[1])
        end
    end
    return
end

function calc_loopset_μν_name(name,Dim)
    loops_μν= Array{Vector{Wilsonline{Dim}},2}(undef,Dim,Dim)
    if name == "plaq"
        numofloops = 1
        for μ=1:Dim
            for ν=1:Dim
                loops_μν[μ,ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                plaq = make_plaq(μ,ν,Dim=Dim)
                push!(loops_μν[μ,ν],plaq)
            end
        end
    elseif name == "clover"
        numofloops = 4
        for μ=1:Dim
            for ν=1:Dim
                loops_μν[μ,ν] = Wilsonline{Dim}[]
                if ν == μ
                    continue
                end
                loops_μν[μ,ν] = make_cloverloops(μ,ν,Dim=Dim)
            end
        end
    elseif name == "rect"
        numofloops = 8
        for μ=1:4
            for ν=1:4
                if ν == μ
                    continue
                end
                loops = Wilsonline{Dim}[]
                loop_righttop = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)])
                loop_lefttop = Wilsonline([(ν,1),(μ,-2),(ν,-1),(μ,2)])
                loop_rightbottom = Wilsonline([(ν,-1),(μ,2),(ν,1),(μ,-2)])
                loop_leftbottom= Wilsonline([(μ,-2),(ν,-1),(μ,2),(ν,1)])
                push!(loops,loop_righttop)
                push!(loops,loop_lefttop)
                push!(loops,loop_rightbottom)
                push!(loops,loop_leftbottom)

                loop_righttop = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)])
                loop_lefttop = Wilsonline([(ν,2),(μ,-1),(ν,-2),(μ,1)])
                loop_rightbottom = Wilsonline([(ν,-2),(μ,1),(ν,2),(μ,-1)])
                loop_leftbottom= Wilsonline([(μ,-1),(ν,-2),(μ,1),(ν,2)])
                push!(loops,loop_righttop)
                push!(loops,loop_lefttop)
                push!(loops,loop_rightbottom)
                push!(loops,loop_leftbottom)

                loops_μν[μ,ν] = loops
            end
        end
    else
        error("$name is not supported")
    end
    return loops_μν,numofloops
end

function make_cloverloops(μ,ν;Dim=4)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)])
    loop_lefttop = Wilsonline([(ν,1),(μ,-1),(ν,-1),(μ,1)])
    loop_rightbottom = Wilsonline([(ν,-1),(μ,1),(ν,1),(μ,-1)])
    loop_leftbottom= Wilsonline([(μ,-1),(ν,-1),(μ,1),(ν,1)])
    push!(loops,loop_righttop)
    push!(loops,loop_lefttop)
    push!(loops,loop_rightbottom)
    push!(loops,loop_leftbottom)
    return loops
end


#topological charge
function epsilon_tensor(inputindex...)
    sign = 1
    for mu in inputindex
        if mu < 0
            sign *= -1
        end
    end
    epsilon = levicivita(abs.(collect(inputindex)))
    return epsilon*sign
end


function test()
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 0
    NC = 3

    Dim = 4

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    filename="./conf_08080808.ildg"

    ildg = ILDG(filename)
    i = 1
    L = [NX,NY,NZ,NT]
    load_gaugefield!(U,i,ildg,L,NC)

    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    β = 5.7
    temp_UμνTA= Matrix{typeof(U[1])}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            temp_UμνTA[μ,ν] = similar(U[1])
        end
    end

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println(" plaq_t = $plaq_t")

    dt = 0.01

    g = Gradientflow(U,eps = dt)

    for itrj=1:10
        flow!(U,g)
        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        Qplaq = calculate_topological_charge_plaq(U,temp_UμνTA,[temp1,temp2,temp3])
        Qclover = calculate_topological_charge_clover(U,temp_UμνTA,[temp1,temp2,temp3])
        Qimproved= calculate_topological_charge_improved(U,temp_UμνTA,Qclover,[temp1,temp2,temp3])
        println("$itrj $dt $plaq_t $Qplaq $Qclover $Qimproved # itrj dt plaq Qplaq Qclover Qimproved ")
    end




end
test()
```

## Former custom gauge-field implementation guide

## How to implement new gauge fields

It is easy to implement new gauge fields with different internal structures or different parallel computations.

### AbstractGaugefields and interfaces
All types of gauge fields belong `AbstractGaugefields{NC,Dim}`.

The concrete types for gauge fields should have following functions.

* `LinearAlgebra.mul!(c::T,a::T1,b::T2) where {T<: AbstractGaugefields,T1 <: Abstractfields,T2 <: Abstractfields}`
* `LinearAlgebra.mul!(c::T,a::N,b::T2) where {T<: AbstractGaugefields,N <: Number ,T2 <: Abstractfields}`
* `LinearAlgebra.mul!(c::T,a::T1,b::T2,α::Ta,β::Tb) where {T<: AbstractGaugefields,T1 <: Abstractfields,T2 <: Abstractfields,Ta <: Number, Tb <: Number}`
* `substitute_U!(a::Array{T1,1},b::Array{T2,1}) where {T1 <: AbstractGaugefields,T2 <: AbstractGaugefields}`
* `Base.similar(U::T) where T <: AbstractGaugefields`
* `clear_U!(U::T) where T <: AbstractGaugefields`
* `set_wing_U!(U::T) where T <: AbstractGaugefields`
* `Base.size(U::T) where T <: AbstractGaugefields`
* `add_U!(c::T,a::T1) where {T<: AbstractGaugefields,T1 <: Abstractfields}`
* `add_U!(c::T,α::N,a::T1) where {T<: AbstractGaugefields,T1 <: Abstractfields, N<:Number}`
* `LinearAlgebra.tr(a::T) where T<: Abstractfields`
* `LinearAlgebra.tr(a::T,b::T) where T<: Abstractfields`
* `Base.setindex!(x::Gaugefields_4D_wing,v,i1,i2,i3,i4,i5,i6)`
* `Base.getindex(x::Gaugefields_4D_wing,i1,i2,i3,i4,i5,i6)`

### matrix-field matrix-field product

`LinearAlgebra.mul!(c::T,a::T1,b::T2) where {T<: AbstractGaugefields,T1 <: Abstractfields,T2 <: Abstractfields}`

This calculates the matrix-matrix multiplicaetion on each lattice site.

As a mathematical expression, for matrix-valued fields ``A(n), B(n)``,
we define "matrix-field matrix-field product" as,

```math
[A(n)B(n)]_{ij} = \sum_k [A(n)]_{ik} [B(n)]_{kj}
```

for all site index n.

In our package, this is expressed as,

```julia
mul!(C,A,B)
```
which means ```C = A*B``` on each lattice site.
Here ``A, B, C`` are same type of ``u``.

### Several ways to treat periodic boundary.
There are several ways to treat periodic boundary.
Now, there are two kinds of methods, halo updates and direct-shift method.

- halo updates: wing-buffer (so-called halo) type implementations for gauge fields. In this type, there are additional halo sites. If ```NDW > 0``` is set, this update is used.
- direct-shift method: In ```shift_U!```  function, the data is copied. If ```NDW = 0``` is set, this update is used.


Therefore, the important functions are

- ```set_wing_U!``` : This is used in halo updates, but not used in direct-shift method (returns nothing).
- ```shift_U!``` : In this function, the data is copied in direct-shift method. This is the lazy evaluation in halo updates.



### halo updates
There is a wing-buffer (so-called halo) type implementations for gauge fields. In this type, there are additional halo sites.
`set_wing_U!` function is used for updating halo sites.
If you do not want to use halo type implementations for gauge fields, you can write `set_wing_U!(U::Yourtype) = nothing`.

#### examples

```@meta
CurrentModule = Gaugefields.AbstractGaugefields_module
```

We have two kinds of $SU(N)$ gauge fields in four dimension.

Serial version:

```@docs
Gaugefields_4D_wing
```


```julia
    struct Gaugefields_4D_wing{NC} <: Gaugefields_4D{NC}
        U::Array{ComplexF64,6}
        NX::Int64
        NY::Int64
        NZ::Int64
        NT::Int64
        NDW::Int64
        NV::Int64
        NC::Int64
        mpi::Bool
        verbose_print::Verbose_print

        function Gaugefields_4D_wing(NC::T,NDW::T,NX::T,NY::T,NZ::T,NT::T;verbose_level = 2) where T<: Integer
            NV = NX*NY*NZ*NT
            U = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
            mpi = false
            verbose_print = Verbose_print(verbose_level )
            #U = Array{Array{ComplexF64,6}}(undef,4)
            #for μ=1:4
            #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
            #end
            return new{NC}(U,NX,NY,NZ,NT,NDW,NV,NC,mpi,verbose_print)
        end
    end
```

Usually, we do not consider local SU(N) matrix on each lattice bond. The gauge fields are manupulated like

```julia
mul!(C,A,B)
s = tr(C)
```
Details of `mul!` and `tr` depends on the gauge field type that we use. In other words, the details which depends on kinds of gauge fields are only in the definitions of `mul!` or `tr`.
So, we implement kind-dependent `mul!` like

```julia
    function LinearAlgebra.mul!(c::Gaugefields_4D_wing{NC},a::T1,b::T2) where {NC,T1 <: Abstractfields,T2 <: Abstractfields}
        @assert NC != 2 && NC != 3 "This function is for NC != 2,3"
        NT = c.NT
        NZ = c.NZ
        NY = c.NY
        NX = c.NX
        @inbounds for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    for ix=1:NX
                        for k2=1:NC
                            for k1=1:NC
                                c[k1,k2,ix,iy,iz,it] = 0

                                @simd for k3=1:NC
                                    c[k1,k2,ix,iy,iz,it] += a[k1,k3,ix,iy,iz,it]*b[k3,k2,ix,iy,iz,it]
                                end
                            end
                        end
                    end
                end
            end
        end
        set_wing_U!(c)
    end
```

If we want to use MPI parallel computations, we use diffrent type of gauge fields.
The definition is


```julia
    struct Gaugefields_4D_wing_mpi{NC} <: Gaugefields_4D{NC}
        U::Array{ComplexF64,6}
        NX::Int64
        NY::Int64
        NZ::Int64
        NT::Int64
        NDW::Int64
        NV::Int64
        NC::Int64
        PEs::NTuple{4,Int64}
        PN::NTuple{4,Int64}
        mpiinit::Bool
        myrank::Int64
        nprocs::Int64
        myrank_xyzt::NTuple{4,Int64}
        mpi::Bool
        verbose_print::Verbose_print

        function Gaugefields_4D_wing_mpi(NC::T,NDW::T,NX::T,NY::T,NZ::T,NT::T,PEs;mpiinit=true,
                                                            verbose_level = 2) where T<: Integer
            NV = NX*NY*NZ*NT
            @assert NX % PEs[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs"
            @assert NY % PEs[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs"
            @assert NZ % PEs[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs"
            @assert NT % PEs[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs"

            PN = (NX ÷ PEs[1],
                    NY ÷ PEs[2],
                    NZ ÷ PEs[3],
                    NT ÷ PEs[4],
            )

            if mpiinit == false
                MPI.Init()
                mpiinit = true
            end

            comm = MPI.COMM_WORLD

            nprocs = MPI.Comm_size(comm)
            @assert prod(PEs) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
            myrank = MPI.Comm_rank(comm)

            verbose_print = Verbose_print(verbose_level,myid = myrank)

            myrank_xyzt = get_myrank_xyzt(myrank,PEs)

            #println("Hello world, I am $(MPI.Comm_rank(comm)) of $(MPI.Comm_size(comm))")

            U = zeros(ComplexF64,NC,NC,PN[1]+2NDW,PN[2]+2NDW,PN[3]+2NDW,PN[4]+2NDW)
            #U = Array{Array{ComplexF64,6}}(undef,4)
            #for μ=1:4
            #    U[μ] = zeros(ComplexF64,NC,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
            #end
            mpi = true
            return new{NC}(U,NX,NY,NZ,NT,NDW,NV,NC,Tuple(PEs),PN,mpiinit,myrank,nprocs,myrank_xyzt,mpi,verbose_print)
        end
    end
```

Its `mul!` is implemented as

```julia
    function LinearAlgebra.mul!(c::Gaugefields_4D_wing_mpi{NC},a::T1,b::T2) where {NC,T1 <: Abstractfields,T2 <: Abstractfields}
        @assert NC != 2 && NC != 3 "This function is for NC != 2,3"
        NT = c.NT
        NZ = c.NZ
        NY = c.NY
        NX = c.NX
        PN = c.PN
        for it=1:PN[4]
            for iz=1:PN[3]
                for iy=1:PN[2]
                    for ix=1:PN[1]
                        for k2=1:NC
                            for k1=1:NC
                                v = 0
                                setvalue!(c,v,k1,k2,ix,iy,iz,it)
                                #c[k1,k2,ix,iy,iz,it] = 0

                                @simd for k3=1:NC
                                    vc = getvalue(c,k1,k2,ix,iy,iz,it) + getvalue(a,k1,k3,ix,iy,iz,it)*getvalue(b,k3,k2,ix,iy,iz,it)
                                    setvalue!(c,vc,k1,k2,ix,iy,iz,it)
                                    #c[k1,k2,ix,iy,iz,it] += a[k1,k3,ix,iy,iz,it]*b[k3,k2,ix,iy,iz,it]
                                end
                            end
                        end
                    end
                end
            end
        end
        #set_wing_U!(c)
    end

```

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
Nwing = 0
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

We need two temporal files to write ILDG format.
If you want to change the temporal files, you can add the names.

```julia
save_binarydata(U, filename; tempfile1="temp1.dat", tempfile2="temp2.dat”)
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
    temp1 = similar(U[1])
    temp2 = similar(U[1])

    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("plaq_t = $plaq_t")


    filename = "test.jld2"
    saveU(filename, U)
end

function loadingexample()
    filename = "test.jld2"
    U = loadU(filename)

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
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

## Heatbath updates (even-odd method)

```julia
using Gaugefields


function heatbath_SU3!(U,NC,temps,β)
    Dim = 4
    V = temps[5]
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

        evaluate_gaugelinks_evenodd!(V,loops,U,temps[1:4],iseven)
        map_U!(U[μ],mapfunc!,V,iseven)

        iseven = false
        evaluate_gaugelinks_evenodd!(V,loops,U,temps[1:4],iseven)
        map_U!(U[μ],mapfunc!,V,iseven)
    end

end

function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 0

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    # for heatbath update
    temp3 = similar(U[1])
    temp4 = similar(U[1])
    temp5 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 40
    for itrj = 1:numhb
        heatbath_SU3!(U,NC,[temp1,temp2,temp3,temp4,temp5],β)

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
Nwing = 0

β = 5.7
NC = 3
@time plaq_t = heatbathtest_4D(NX,NY,NZ,NT,β,NC)
```

## Heatbath updates with general actions
We can do heatbath updates with a general action.

```julia
using Gaugefields

function heatbathtest_4D(NX,NY,NZ,NT,β,NC)
    Dim = 4
    Nwing = 0

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

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    println("polyakov loop = $(real(poly)) $(imag(poly))")

    numhb = 1000
    for itrj = 1:numhb

        heatbath!(U,hnew)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        poly = calculate_Polyakov_loop(U,temp1,temp2)

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

## Gradient flow
We can use Lüscher's gradient flow.

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 0
NC = 3

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")

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

temp1 = similar(U[1])
temp2 = similar(U[1])

comb = 6
factor = 1/(comb*U[1].NV*U[1].NC)

g = Gradientflow(U, B)
for itrj=1:100
    flow!(U,B,g)
    @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
    println("$itrj plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
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

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC # evaluate_GaugeAction(...) = tr(evaluate_GaugeAction_untraced(...))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temp1,temp2)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
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

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
    end
end


function HMC_test_4D(NX,NY,NZ,NT,NC,β)
    Dim = 4
    Nwing = 0

    Random.seed!(123)


    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")
    #"Reproducible"
    println(typeof(U))

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)


    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)

    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients.
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temp1,temp2)
        end
        if get_myrank(U) == 0
            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2)
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

## Non-dynamical higher-form gauge fields
We can do the HMC simulations with B fields. The example code is as follows.
```julia

using Random
using Gaugefields
using LinearAlgebra

function calc_action(gauge_action,U,B,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U,B)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temp1,temp2)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,B,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Snew = calc_action(gauge_action,U,B,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold)) # bug is fixed
    if rand() > ratio
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

function P_update!(U,B,p,ϵ,Δτ,Dim,gauge_action,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp  = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U,B)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
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

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)

    @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)

    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients.
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 50
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temp1,temp2)
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2)
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

## Dynamical higher-form gauge fields
HMC simulations with dynamical B fields are as follows:
```julia

using Random
using Gaugefields
using Wilsonloop
using LinearAlgebra

function calc_action(gauge_action,U,B,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U,B)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temp1,temp2)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,B,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Snew = calc_action(gauge_action,U,B,p)
#    println("Sold = $Sold, Snew = $Snew")
#    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

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
    temp1,
    temp2
) # Halfway-updating HMC
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        if itrj == Int(MDsteps/2)
            Flux_update!(B,flux)
        end
    end

    Snew = calc_action(gauge_action,U,B,p)
#    println("Sold = $Sold, Snew = $Snew")
#    println("Snew - Sold = $(Snew-Sold)")
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
    temp1,
    temp2
) # Double-tesing HMC
    p0 = initialize_TA_Gaugefields(U)
    Sold = calc_action(gauge_action,U,B,p0)

    substitute_U!(Uold1,U)
    substitute_U!(Bold, B)
    flux_old[:] = flux[:]

    Flux_update!(B,flux)

    for ihmc=1:num_HMC
        MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold2,temp1,temp2)
    end

    Snew = calc_action(gauge_action,U,B,p0)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
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

function Flux_update!(B,flux)
    NC  = B[1,2].NC
    NDW = B[1,2].NDW
    NX  = B[1,2].NX
    NY  = B[1,2].NY
    NZ  = B[1,2].NZ
    NT  = B[1,2].NT

    i = rand(1:6)
    flux[i] += rand(-1:1)
    flux[i] %= NC
    flux[i] += (flux[i] < 0) ? NC : 0
#    flux[:] = rand(0:NC-1,6)
    B = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux")

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

function P_update!(U,B,p,ϵ,Δτ,Dim,gauge_action,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp  = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U,B)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
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

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)

    @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)

    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients.

    Uold  = similar(U)
    substitute_U!(Uold, U)
    Bold = similar(B)
    substitute_U!(Bold,B)
    flux_old = zeros(Int, 6)

    MDsteps = 50 # even integer!!!
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
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
                temp1,
                temp2
            )
        end
        if get_myrank(U) == 0
             println("Flux : ", flux)
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2)
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

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end
    factor = 1/(comb*U[1].NV*U[1].NC)


    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
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
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return plaq_t

end


function gradientflow_test_2D(NX,NT,NC)
    Dim = 2
    Nwing = 1
    U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot",randomnumber="Reproducible")

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
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
            @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2)
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

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end
    factor = 1/(comb*U[1].NV*U[1].NC)


    @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temp1,temp2)
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
    g = Gradientflow_general_Bfields(U,B,listloops,listvalues,eps = 0.1)

    for itrj=1:10
        flow!(U,B,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temp1,temp2)
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

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC # evaluate_GaugeAction(...) = tr(evaluate_GaugeAction_untraced(...))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temp1,temp2)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Snew = calc_action(gauge_action,U,p)
    if get_myrank(U) == 0
        println("Sold = $Sold, Snew = $Snew")
        println("Snew - Sold = $(Snew-Sold)")
    end
    ratio = min(1,exp(-Snew+Sold))
    r = rand()
    if mpi
        r = MPI.bcast(r, 0, MPI.COMM_WORLD)
    end
    #ratio = min(1,exp(Snew-Sold))
    if r > ratio
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

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
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

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)

    @time plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
    if get_myrank(U) == 0
        println("0 plaq_t = $plaq_t")
    end
    poly = calculate_Polyakov_loop(U,temp1,temp2)
    if get_myrank(U) == 0
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")
    end

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)

    #show(gauge_action)

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
        t = @timed begin
            accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temp1,temp2)
        end
        if get_myrank(U) == 0
            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
            if get_myrank(U) == 0
                println("$itrj plaq_t = $plaq_t")
            end
            poly = calculate_Polyakov_loop(U,temp1,temp2)
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

The LatticeMatrices/JACC GPU and multi-GPU interface is part of the v1 API,
not the compatibility API. See [GPU and multi-GPU execution](tutorial4d.md#multiple-gpus-with-mpi)
for the current setup.

# Legacy single-GPU CUDA backend (experimental)

From v0.5.0, the original Gaugefields accelerator backend can use an NVIDIA GPU
for accelerating simulations. This backend remains available independently of
the LatticeMatrices/JACC multi-GPU path above.
To use GPU, you just add ```using CUDA``` and put two keywords on Initialize_Gaugefields like:
```julia
using CUDA
Nwing = 0
U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",
    cuda = true,
    blocks =[4,4,4,4])
```

With the `LatticeMatrices` backend, `randomnumber="Reproducible"` uses
deterministic global-site random-number streams and is independent of the MPI
decomposition. The deprecated legacy accelerator backend does not support this
option. Only `Nwing=0` is supported on that legacy backend.

The gradient flow example is as follows.
```julia

using Random
using Test
using CUDA
using Gaugefields
using Wilsonloop

function gradientflow_test_4D(NX, NY, NZ, NT, NC)
    Dim = 4
    Nwing = 1

    Random.seed!(123)

    #U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
    #U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot",cuda = true,blocks = [4,4,4,4])


    temps = Temporalfields(U[1]; num=10)
    temp1 = temps[1]#similar(U[1])
    temp2 = temps[2] #similar(U[1])
    #temp1 = similar(U[1])
    #temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end
    factor = 1 / (comb * U[1].NV * U[1].NC)


    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #Plaquette term
    loops_p = Wilsonline{Dim}[]
    for μ = 1:Dim
        for ν = μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ, 1), (ν, 1), (μ, -1), (ν, -1)], Dim=Dim)
            push!(loops_p, loop1)
            #loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            #push!(loops_p,loop1)
        end
    end


    #Rectangular term
    loops = Wilsonline{Dim}[]
    for μ = 1:Dim
        for ν = μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ, 1), (ν, 2), (μ, -1), (ν, -2)], Dim=Dim)
            push!(loops, loop1)
            loop1 = Wilsonline([(μ, 2), (ν, 1), (μ, -2), (ν, -1)], Dim=Dim)

            push!(loops, loop1)
        end
    end

    listloops = [loops_p, loops]
    listvalues = [1 + im, 0.1]
    g = Gradientflow_general(U, listloops, listvalues, eps=0.01)

    for itrj = 1:100
        flow!(U, g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temp1, temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return plaq_t

end



println("4D system")
@testset "4D" begin
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    Nwing = 0



    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        #val =0.6414596466929057
        #val = 0.5920897445000382
        val = 0.9440125563836135
        @time plaq_t = gradientflow_test_4D(NX, NY, NZ, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        #val = 0.5779454661484242
        #val  =0.9440125563836135
        #val = 0.5385142466966718
        val = 0.8786515255315753
        @time plaq_t = gradientflow_test_4D(NX, NY, NZ, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        #val  =0.19127260002797497
        #val = 0.1904815857904191
        val = 0.7301232810349298
        @time plaq_t = gradientflow_test_4D(NX, NY, NZ, NT, NC)
        #@test abs(plaq_t-val)/abs(val) < eps
    end


end
```


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

```julia
pkg> add Wilsonloop
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

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

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

    Uout = evaluate_GaugeAction_untraced(gauge_action,U)
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
    substitute_U!(Uold,U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end
```

We define the functions as

```julia

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC # evaluate_GaugeAction(...) = tr(evaluate_GaugeAction_untraced(...))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
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

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
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

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


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
    st = STOUT_Layer(layername, ρ, U)
    #st = STOUT_Layer(layername,ρ,L)
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
    st = STOUT_Layer(layername, ρ, U)
    #st = STOUT_Layer(layername,ρ,L)
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
    #st = STOUT_Layer(layername,ρ,L)
    st = STOUT_Layer(layername, ρ, U)
    push!(nn,st)



    MDtest!(gauge_action,U,Dim,nn)

end


test1()
```

## HMC with Enzyme-based Force Computation
With Enzyme.jl, Gaugefields.jl can compute this force automatically once the action is written as a Julia function.

The following script demonstrates a full HMC trajectory where:
	•	the gauge fields are evolved using leapfrog integration
	•	the force is obtained via Enzyme_derivative!

*Note*: Enzyme-based automatic differentiation in Gaugefields.jl is enabled via the lattice matrix infrastructure provided by [LatticeMatrices.jl](https://github.com/cometscome/LatticeMatrices.jl).
At present, AD support is implemented only for the MPI lattice backend, so isMPILattice=true must be set even when running on a single rank.

```julia
using Random
using LinearAlgebra
using PreallocatedArrays
using Enzyme
import JACC
JACC.@init_backend
using Gaugefields


function _calc_action_step!(C, D, E, Uμ, Uν, shift_μ, shift_ν)
    Uμ_pν = shift_U(Uμ, shift_ν)
    Uν_pμ = shift_U(Uν, shift_μ)

    mul!(C, Uμ, Uν_pμ)
    mul!(D, C, Uμ_pν')
    mul!(E, D, Uν')
    S = realtrace(E)

    mul!(C, Uν, Uμ_pν)
    mul!(D, C, Uν_pμ')
    mul!(E, D, Uμ')
    S += realtrace(E)

    return S
end


function calc_action(U1, U2, U3, U4, β, NC, temp)
    dim = 4
    U = (U1, U2, U3, U4)
    C = temp[1]
    D = temp[2]
    E = temp[3]
    S = 0.0

    for μ = 1:dim
        shift_μ = ntuple(i -> ifelse(i == μ, 1, 0), dim)
        for ν = μ:dim
            if ν == μ
                continue
            end
            shift_ν = ntuple(i -> ifelse(i == ν, 1, 0), dim)
            S += _calc_action_step!(C, D, E, U[μ], U[ν], shift_μ, shift_ν)
        end
    end

    return -S * β / NC
end

function MDstep!(U, p, MDsteps, Dim, Uold, tempvec, β)
    NC = U[1].NC
    Δτ = 1.0 / MDsteps
    temp1, it_temp1 = get_block(tempvec)
    temp, its_temp = get_block(tempvec, 3)
    dtemp, its_dtemp = get_block(tempvec, 3)


    gauss_distribution!(p)
    #Sold = calc_action(gauge_action, U, p)
    Sold = calc_action(U..., β, NC, temp) + p * p / 2
    substitute_U!(Uold, U)

    for itrj = 1:MDsteps
        U_update!(U, p, 0.5, Δτ, Dim, tempvec)

        P_update!(U, p, 1.0, Δτ, Dim, temp1, temp, dtemp, tempvec, β)

        U_update!(U, p, 0.5, Δτ, Dim, tempvec)
    end
    Snew = calc_action(U..., β, NC, temp) + p * p / 2
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1, exp(-Snew + Sold))

    unused!(tempvec, it_temp1)
    unused!(tempvec, its_temp)
    unused!(tempvec, its_dtemp)


    if rand() > ratio
        substitute_U!(U, Uold)
        return false
    else
        return true
    end
end

function U_update!(U, p, ϵ, Δτ, Dim, tempvec)
    temp1, it_temp1 = get_block(tempvec)
    temp2, it_temp2 = get_block(tempvec)
    expU, it_expU = get_block(tempvec)
    W, it_W = get_block(tempvec)


    for μ = 1:Dim
        exptU!(expU, ϵ * Δτ, p[μ], [temp1, temp2])
        mul!(W, expU, U[μ])
        substitute_U!(U[μ], W)
    end
    unused!(tempvec, it_temp1)
    unused!(tempvec, it_temp2)
    unused!(tempvec, it_expU)
    unused!(tempvec, it_W)

end

function P_update!(U, p, ϵ, Δτ, Dim, temp1, temp, dtemp, temps, β) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    factor = -ϵ * Δτ / (NC)
    factor_ad = ϵ * Δτ / 2

    dSdU, it_dSdU = get_block(temps, 4)#temps[end]
    Gaugefields.clear_U!(dSdU)

    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]
    Enzyme_derivative!(
        calc_action,
        U1, U2, U3, U4,
        dSdU[1], dSdU[2], dSdU[3], dSdU[4], nodiff(β), nodiff(NC);
        temp,
        dtemp
    )

    for μ = 1:Dim
        mul!(temp1, U[μ], dSdU[μ]')
        Traceless_antihermitian_add!(p[μ], factor_ad, temp1)
    end

    unused!(temps, it_dSdU)
end


function HMC_test_4D(NX, NY, NZ, NT, NC, β)
    Dim = 4
    Nwing = 1

    Random.seed!(123)


    #U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="hot", randomnumber="Reproducible")
    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT,
        condition="hot";
        isMPILattice=true)
    #"Reproducible"
    println(typeof(U))

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1 / (comb * U[1].NV * U[1].NC)
    tempvec = PreallocatedArray(U[1]; num=30, haslabel=false)


    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")


    β = β / 2

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients.
    Uold = similar(U)
    substitute_U!(Uold, U)
    MDsteps = 100
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    numaccepted = 0

    numtrj = 10
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(U, p, MDsteps, Dim, Uold, tempvec, β)
        end
        if get_myrank(U) == 0
            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted, 1, 0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")

        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U, temp1, temp2)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ", numaccepted / itrj)
        end
    end
    return plaq_t, numaccepted / numtrj

end


function main()
    β = 6
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    HMC_test_4D(NX, NY, NZ, NT, NC, β)
end
main()
```
