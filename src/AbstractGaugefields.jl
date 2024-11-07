module AbstractGaugefields_module
using LinearAlgebra
import ..Wilsonloops_module:
    Wilson_loop_set,
    calc_coordinate,
    make_plaq_staple_prime,
    calc_shift,
    make_plaq,
    make_plaq_staple,
    Tensor_wilson_lines_set,
    Tensor_wilson_lines,
    Tensor_derivative_set,
    get_leftstartposition,
    get_rightstartposition,
    Wilson_loop,
    calc_loopset_μν_name
import Wilsonloop: loops_staple_prime, Wilsonline, get_position, get_direction, GLink, isdag,make_cloverloops
using Requires
using Distributions
using StableRNGs
import ..Verboseprint_mpi:
    Verbose_print, println_verbose_level1, println_verbose_level2, println_verbose_level3

#using MPI
using InteractiveUtils


abstract type Abstractfields end


abstract type AbstractGaugefields{NC,Dim} <: Abstractfields end


abstract type Adjoint_fields{T} <: Abstractfields end

struct Adjoint_Gaugefields{T} <: Adjoint_fields{T}
    parent::T
end

function Base.adjoint(U::T) where {T<:Abstractfields}
    Adjoint_Gaugefields{T}(U)
end

abstract type Unit_Gaugefield{NC,Dim} <: Abstractfields end

function Base.adjoint(U::T) where {T<:Unit_Gaugefield}
    return U
end

abstract type Shifted_Gaugefields{NC,Dim} <: Abstractfields end

struct Staggered_Gaugefields{T,μ} <: Abstractfields
    parent::T

    function Staggered_Gaugefields(u::T, μ) where {T<:Abstractfields}
        return new{T,μ}(u)
    end
end

function staggered_U(u::T, μ) where {T<:Abstractfields}
    return Staggered_Gaugefields(u, μ)
end

function Base.setindex!(U::T, v...) where {T<:Staggered_Gaugefields}
    error("type $(typeof(U)) has no setindex method. This type is read only.")
end

function Base.setindex!(U::T, v...) where {T<:Unit_Gaugefield}
    error("type $(typeof(U)) has no setindex method. This type is read only.")
end

function Base.getindex(U::T, v...) where {T<:Staggered_Gaugefields}
    error("type $(typeof(U)) has no getindex method")
end

struct Gaugefield_latticeindices{Dim,NC,T}
    NN::NTuple{Dim,Int64}
    NC::Int8

    function Gaugefield_latticeindices(u::AbstractGaugefields{Dim,NC}) where {Dim,NC}
        _, _, NN... = size(u)
        return new{Dim,NC,typeof{u}}(NN, NC)
    end
end

function Gaugefield_latticeindices(U::Array{<:AbstractGaugefields{Dim,NC},1}) where {Dim,NC}
    return Gaugefield_latticeindices(U[1])
end

function Base.iterate(g::Gaugefield_latticeindices{Dim,NC,T}) where {Dim,NC,T}
    N < 1 && return nothing
    return (1, 2)
end


mutable struct Data_sent{NC} #data format for MPI
    count::Int64
    data::Array{ComplexF64,3}
    positions::Vector{Int64}

    function Data_sent(N, NC)
        data = zeros(ComplexF64, NC, NC, N)
        count = 0
        positions = zeros(Int64, N)

        return new{NC}(count, data, positions)
    end
end

include("./2D/gaugefields_2D.jl")
include("./4D/gaugefields_4D.jl")

include("TA_Gaugefields.jl")
include("Adjoint_rep_Gaugefields.jl")



function LinearAlgebra.mul!(C, A::T, B) where {T<:Unit_Gaugefield}
    substitute_U!(C, B)
end

function LinearAlgebra.mul!(C, A, B::T) where {T<:Unit_Gaugefield}
    substitute_U!(C, A)
end


function Staggered_Gaugefields(u::AbstractGaugefields{NC,Dim}) where {NC,Dim}
    if Dim == 4
        return Staggered_Gaugefields_4D(u)
    else
        error("Dim = $Dim is not supported")
    end
end

function println_verbose_level1(u::T, val...) where {T<:AbstractGaugefields}
    println_verbose_level1(u.verbose_print, val...)
end

function println_verbose_level2(u::T, val...) where {T<:AbstractGaugefields}
    println_verbose_level2(u.verbose_print, val...)
end

function println_verbose_level3(u::T, val...) where {T<:AbstractGaugefields}
    println_verbose_level3(u.verbose_print, val...)
end

function get_myrank(U::T) where {T<:AbstractGaugefields}
    return 0
end

function get_myrank(U::Array{T,1}) where {T<:AbstractGaugefields}
    return 0
end

function get_nprocs(U::T) where {T<:AbstractGaugefields}
    return 1
end

function get_nprocs(U::Array{T,1}) where {T<:AbstractGaugefields}
    return 1
end

function barrier(x::T) where {T<:AbstractGaugefields}
    return
end


function getvalue(U::T, i1, i2, i3, i4, i5, i6) where {T<:Abstractfields}
    return U[i1, i2, i3, i4, i5, i6]
end

function getvalue(U::T, i1, i2, i3, i6) where {T<:Abstractfields}
    return U[i1, i2, i3, i6]
end

#include("./gaugefields_4D_wing_mpi.jl")

using NPZ

function write_to_numpyarray(U::T, filename) where {T<:AbstractGaugefields}
    error("write_to_numpyarray! is not implemented in type $(typeof(U)) ")
end

function Base.similar(U::T) where {T<:AbstractGaugefields}
    error("similar! is not implemented in type $(typeof(U)) ")
end

function substitute_U!(
    a::Array{<:AbstractGaugefields{NC,Dim},1},
    b::Array{<:AbstractGaugefields{NC,Dim},1},
) where {NC,Dim}
    error("substitute_U! is not implemented in type $(typeof(a)) and $(typeof(b))")
    for i = 1:Dim
        substitute_U!(a[i], b[i])
    end
end

function substitute_U!(a::T1, b::T2) where {T1<:Abstractfields,T2<:Abstractfields}
    error("substitute_U! is not implemented in type $(typeof(a)) and $(typeof(b))")
    return
end

function substitute_U!(
    a::Array{T1,1},
    b::Array{T2,1},
    iseven::Bool,
) where {T1<:AbstractGaugefields,T2<:AbstractGaugefields}
    error("substitute_U! is not implemented in type $(typeof(a)) and $(typeof(b))")
end

function substitute_U!(
    a::T1,
    b::T2,
    iseven::Bool,
) where {T1<:Abstractfields,T2<:Abstractfields}
    error(
        "substitute_U!(a,b,iseven) is not implemented in type $(typeof(a)) and $(typeof(b))",
    )
    return
end

function Initialize_4DGaugefields(
    NC,
    NDW,
    NN...;
    condition = "cold",
    verbose_level = 2,
    randomnumber = "Random",
)
    if condition == "cold"
        if NDW == 0

            u1 = IdentityGauges_4D(NC, NN..., verbose_level = verbose_level)
        else
            u1 = IdentityGauges_4D(NC, NDW, NN..., verbose_level = verbose_level)
        end
    elseif condition == "hot"
        if NDW == 0
            u1 = RandomGauges_4D(
                NC,
                NN...,
                verbose_level = verbose_level,
                randomnumber = randomnumber,
            )
        else
            u1 = RandomGauges_4D(
                NC,
                NDW,
                NN...,
                verbose_level = verbose_level,
                randomnumber = randomnumber,
            )
        end
    else
        error("not supported")
    end
    @assert length(NN) == 4 "Dimension should be 4. "
    Dim = 4

    U = Array{typeof(u1),1}(undef, Dim)
    U[1] = u1

    for μ = 2:Dim
        if condition == "cold"
            if NDW == 0
                U[μ] = IdentityGauges_4D(NC, NN..., verbose_level = verbose_level)
            else
                U[μ] = IdentityGauges_4D(NC, NDW, NN..., verbose_level = verbose_level)
            end
        elseif condition == "hot"
            if NDW == 0
                U[μ] = RandomGauges_4D(
                    NC,
                    NN...,
                    verbose_level = verbose_level,
                    randomnumber = randomnumber,
                )
            else
                U[μ] = RandomGauges_4D(
                    NC,
                    NDW,
                    NN...,
                    verbose_level = verbose_level,
                    randomnumber = randomnumber,
                )
            end
        else
            error("not supported")
        end
    end
    return U
end
"""
```
Initialize_Gaugefields(NC,NDW,NN...;
    condition = "cold",mpi = false,PEs=nothing,mpiinit = nothing,verbose_level = 2,randomnumber="Random")
```
Initialize gaugefields. 2D or 4D is supported. 

### Examples
If you want to have randomely distributed gauge fields (so-called "hot start") in four dimension, just do:

```julia
U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
```

If you want to have uniform gauge fields (so-called "cold start") in four dimension, just do:

```julia
U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
```

### Return values:
*   `U`: Dim dimensional vector. Here, Dim is a dimension (Dim = 2 or 4). 

### Keyword arguments:
* `condition`: ```cold```(cold start) or ```hot```(hot start). The default is ```cold```
* `mpi`: Using MPI or not. The default is ```false```. If you want to use MPI, you should call ```using MPI```. 
* `PEs`: If ```mpi = true```, we have to set ```PEs = [2,2,2,2]```, which is numbers of regions for MPI process. ```prod(PEs)``` should be the number of total MPI processes. 

### MPI
Gaugefields with using MPI is not well tested. 
"""
function Initialize_Gaugefields(
    NC,
    NDW,
    NN...;
    condition = "cold",
    mpi = false,
    PEs = nothing,
    mpiinit = nothing,
    verbose_level = 2,
    randomnumber = "Random",
)

    Dim = length(NN)
    if condition == "cold"
        u1 = IdentityGauges(
            NC,
            NDW,
            NN...,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
        )
    elseif condition == "hot"
        u1 = RandomGauges(
            NC,
            NDW,
            NN...,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
            randomnumber = randomnumber,
        )
    else
        error("not supported")
    end

    U = Array{typeof(u1),1}(undef, Dim)
    U[1] = u1

    for μ = 2:Dim
        if condition == "cold"
            U[μ] = IdentityGauges(
                NC,
                NDW,
                NN...,
                mpi = mpi,
                PEs = PEs,
                mpiinit = false,
                verbose_level = verbose_level,
            )
        elseif condition == "hot"
            U[μ] = RandomGauges(
                NC,
                NDW,
                NN...,
                mpi = mpi,
                PEs = PEs,
                mpiinit = false,
                verbose_level = verbose_level,
                randomnumber = randomnumber,
            )
        else
            error("not supported")
        end
    end
    return U
end

function RandomGauges(
    NC,
    NDW,
    NN...;
    mpi = false,
    PEs = nothing,
    mpiinit = nothing,
    verbose_level = 2,
    randomnumber = "Random",
)
    dim = length(NN)
    if mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4
                if NDW == 0
                    U = randomGaugefields_4D_nowing_mpi(
                        NC,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        randomnumber = randomnumber,
                    )
                else
                    U = randomGaugefields_4D_wing_mpi(
                        NC,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        NDW,
                        PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        randomnumber = randomnumber,
                    )
                end
            else
                error("not implemented yet!")
            end

        end
    else
        if dim == 4
            if NDW == 0
                U = randomGaugefields_4D_nowing(
                    NC,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    verbose_level = verbose_level,
                    randomnumber = randomnumber,
                )

            else
                U = randomGaugefields_4D_wing(
                    NC,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    NDW,
                    verbose_level = verbose_level,
                    randomnumber = randomnumber,
                )
            end
        elseif dim == 2
            if NDW == 0
                U = randomGaugefields_2D_nowing(
                    NC,
                    NN[1],
                    NN[2],
                    verbose_level = verbose_level,
                    randomnumber = randomnumber,
                )
            else
                U = randomGaugefields_2D_wing(
                    NC,
                    NN[1],
                    NN[2],
                    NDW,
                    verbose_level = verbose_level,
                    randomnumber = randomnumber,
                )
            end
        else
            error("not implemented yet!")
        end
    end
    return U
end

function IdentityGauges(
    NC,
    NDW,
    NN...;
    mpi = false,
    PEs = nothing,
    mpiinit = nothing,
    verbose_level = 2,
)
    dim = length(NN)

    if mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4
                if NDW == 0
                    U = identityGaugefields_4D_nowing_mpi(
                        NC,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                    )
                else
                    U = identityGaugefields_4D_wing_mpi(
                        NC,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        NDW,
                        PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                    )
                end
            elseif dim == 2
                if NDW == 0
                    U = identityGaugefields_2D_nowing_mpi(
                        NC,
                        NN[1],
                        NN[2],
                        PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                    )
                end
            else
                error("$dim dimension with $NDW  is not implemented yet! set NDW = 0")
            end

        end
    else
        if dim == 4
            if NDW == 0
                U = identityGaugefields_4D_nowing(
                    NC,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    verbose_level = verbose_level,
                )
            else
                U = identityGaugefields_4D_wing(
                    NC,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    NDW,
                    verbose_level = verbose_level,
                )
            end
        elseif dim == 2
            if NDW == 0

                U = identityGaugefields_2D_nowing(
                    NC,
                    NN[1],
                    NN[2],
                    verbose_level = verbose_level,
                )
            else
                U = identityGaugefields_2D_wing(
                    NC,
                    NN[1],
                    NN[2],
                    NDW,
                    verbose_level = verbose_level,
                )
            end
        else
            error("$dim dimension system is not implemented yet!")
        end
    end
    set_wing_U!(U)
    return U
end

function Oneinstanton(NC, NDW, NN...; mpi = false, PEs = nothing, mpiinit = nothing)
    dim = length(NN)
    if mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4

                U = Oneinstanton_4D_wing_mpi(
                    NC,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    NDW,
                    PEs,
                    mpiinit = mpiinit,
                )
            else
                error("not implemented yet!")
            end

        end
    else
        if dim == 4
            if NDW == 0
                U = Oneinstanton_4D_nowing(NC, NN[1], NN[2], NN[3], NN[4])
            else
                U = Oneinstanton_4D_wing(NC, NN[1], NN[2], NN[3], NN[4], NDW)
            end
        elseif dim == 2
            if NDW == 0
                U = Oneinstanton_2D_nowing(NC, NN[1], NN[2])
            else
                U = Oneinstanton_2D_wing(NC, NN[1], NN[2], NDW)

            end
        else
            error("not implemented yet!")
        end
    end
    return U
end

function construct_gauges(NC, NDW, NN...; mpi = false, PEs = nothing, mpiinit = nothing)
    dim = length(NN)
    if mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4
                U = identityGaugefields_4D_wing_mpi(
                    NC,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    NDW,
                    PEs,
                    mpiinit = mpiinit,
                )
            else
                error("not implemented yet!")
            end

        end
    else
        if dim == 4
            if NDW == 0
                U = identityGaugefields_4D_nowing(NC, NN[1], NN[2], NN[3], NN[4])
            else
                U = identityGaugefields_4D_wing(NC, NN[1], NN[2], NN[3], NN[4], NDW)
            end
        elseif dim == 2
            if NDW == 0
                U = identityGaugefields_2D_nowing(NC, NN[1], NN[4])
            else
                U = identityGaugefields_2D_wing(NC, NN[1], NN[4], NDW)
            end
        else
            error("not implemented yet!")
        end
    end
    return U
end


function Initialize_Bfields(
    NC,
    Flux,
    NDW,
    NN...;
    condition = "tflux",
    mpi = false,
    PEs = nothing,
    mpiinit = nothing,
    verbose_level = 2,
    randomnumber = "Random",
    tloop_pos  = [1,1,1,1],
    tloop_dir  = [1,4],
    tloop_dis  = 1,
)

    Dim = length(NN)
    fluxnum = 1
    if condition == "tflux"
        u1 = B_TfluxGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus = false,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
        )
        u2 = B_TfluxGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus = true,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
        )
    elseif condition == "tloop"
        u1 = B_TloopGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus = false,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
            tloop_pos  = tloop_pos,
            tloop_dir  = tloop_dir,
            tloop_dis  = tloop_dis,
        )
        u2 = B_TloopGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus = true,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
            tloop_pos  = tloop_pos,
            tloop_dir  = tloop_dir,
            tloop_dis  = tloop_dis,
        )
    elseif condition == "random"
        u1 = B_RandomGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus = false,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
            randomnumber = randomnumber,
        )
        u2 = B_RandomGauges(
            NC,
            Flux[fluxnum],
            fluxnum,
            NDW,
            NN...,
            overallminus = true,
            mpi = mpi,
            PEs = PEs,
            mpiinit = mpiinit,
            verbose_level = verbose_level,
            randomnumber = randomnumber,
        )
    # elseif condition == "hot"
    #     u1 = RandomGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level,randomnumber = "Random")
    # elseif condition == "identity"
    #     u1 = IdentityGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level)
    else
        error("not supported")
    end

    U = Array{typeof(u1),2}(undef, Dim,Dim)

    U[1,2] = u1
    U[2,1] = u2

    for μ = 1:Dim
        for ν = μ+1:Dim
            if (μ,ν) != (1,2)
                fluxnum += 1
                if condition == "tflux"
                    U[μ,ν] = B_TfluxGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus = false,
                        mpi = mpi,
                        PEs = PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                    )
                    U[ν,μ] = B_TfluxGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus = true,
                        mpi = mpi,
                        PEs = PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                    )
                elseif condition == "tloop"
                    U[μ,ν] = B_TloopGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus = false,
                        mpi = mpi,
                        PEs = PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        tloop_pos  = tloop_pos,
                        tloop_dir  = tloop_dir,
                        tloop_dis  = tloop_dis,
                    )
                    U[ν,μ] = B_TloopGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus = true,
                        mpi = mpi,
                        PEs = PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        tloop_pos  = tloop_pos,
                        tloop_dir  = tloop_dir,
                        tloop_dis  = tloop_dis,
                    )
                elseif condition == "random"
                    U[μ,ν] = B_RandomGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus = false,
                        mpi = mpi,
                        PEs = PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        randomnumber = randomnumber,
                    )
                    U[ν,μ] = B_RandomGauges(
                        NC,
                        Flux[fluxnum],
                        fluxnum,
                        NDW,
                        NN...,
                        overallminus = true,
                        mpi = mpi,
                        PEs = PEs,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        randomnumber = randomnumber,
                    )
                # elseif condition == "hot"
                #     U[μ,ν] = RandomGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level,randomnumber = "Random")
                # elseif condition == "identity"
                #     U[μ,ν] = IdentityGauges(NC,NDW,NN...,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level)
                else
                    error("not supported")
                end
            end
        end
    end
    return U
end

function B_RandomGauges(
    NC,
    Flux,
    FluxNum,
    NDW,
    NN...;
    overallminus = false,
    mpi = false,
    PEs = nothing,
    mpiinit = nothing,
    verbose_level = 2,
    randomnumber = "Random",
)
    dim = length(NN)
    println("Not implemented yet! In what follows, let us use B_TfluxGauges.")
    U = B_TfluxGauges(NC,Flux,FluxNum,NDW,NN...,overallminus = overallminus,mpi = mpi,PEs = PEs,mpiinit = mpiinit,verbose_level = verbose_level)
    return U
end

function B_TfluxGauges(
    NC,
    Flux,
    FluxNum,
    NDW,
    NN...;
    overallminus = false,
    mpi = false,
    PEs = nothing,
    mpiinit = nothing,
    verbose_level = 2,
)
    dim = length(NN)
    if mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4
                if NDW == 0
                    U = thooftFlux_4D_B_at_bndry_nowing_mpi(
                        NC,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus = overallminus,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                    )
                else
                    U = thooftFlux_4D_B_at_bndry_wing_mpi(
                        NC,
                        NDW,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus = overallminus,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                    )
                end
            else
                error("$dim dimension is not implemented yet!")
            end
        end
    else
        if dim == 4
            if NDW == 0
                U = thooftFlux_4D_B_at_bndry(
                    NC,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus = overallminus,
                    verbose_level = 2,
                )
            else
                U = thooftFlux_4D_B_at_bndry_wing(
                    NC,
                    NDW,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus = overallminus,
                    verbose_level = 2,
                )
            end
        else
            error("$dim dimension is not implemented yet!")
        end
    end
    set_wing_U!(U)
    return U
end

function B_TloopGauges(
    NC,
    Flux,
    FluxNum,
    NDW,
    NN...;
    overallminus = false,
    mpi = false,
    PEs = nothing,
    mpiinit = nothing,
    verbose_level = 2,
    tloop_pos  = [1,1,1,1],
    tloop_dir  = [1,4],
    tloop_dis  = 1,
)
    # pos = position of Polyakov loop
    # dir = [1-dir shift of anti-Polyakov loop,temporal 4-dir]
    # dis = distance between two loops in 1-dir with sign
    #
    # Polyakov loop at [ix,iy+1/2,iz+1/2,:]
    # anti-Polyakov loop at [ix+dis,iy+1/2,iz+1/2,end:1]
    #
    #           NT |     |
    #              |     |
    #              |     |
    # Polyakovloop |     | antiPolyakovloop
    #              |     |
    #              |     |
    #           1  |     |
    #              x     x+dis
    # and
    #         ----
    #        /   /
    #   ----/  ----- t
    #      /   /
    #     ----  y-z plaquette
    #
    dim = length(NN)
    if mpi
        if PEs == nothing || mpiinit == nothing
            error("not implemented yet!")
        else
            if dim == 4
                if NDW == 0
                    U = thooftLoop_4D_B_temporal_nowing_mpi(
                        NC,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus = overallminus,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        tloop_pos  = tloop_pos,
                        tloop_dir  = tloop_dir,
                        tloop_dis  = tloop_dis,
                    )
                else
                    U = thooftLoop_4D_B_temporal_wing_mpi(
                        NC,
                        NDW,
                        Flux,
                        FluxNum,
                        NN[1],
                        NN[2],
                        NN[3],
                        NN[4],
                        PEs,
                        overallminus = overallminus,
                        mpiinit = mpiinit,
                        verbose_level = verbose_level,
                        tloop_pos  = tloop_pos,
                        tloop_dir  = tloop_dir,
                        tloop_dis  = tloop_dis,
                    )
                end
            else
                error("$dim dimension is not implemented yet!")
            end
        end
    else
        if dim == 4
            if NDW == 0
                U = thooftLoop_4D_B_temporal(
                    NC,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus = overallminus,
                    verbose_level = 2,
                    tloop_pos  = tloop_pos,
                    tloop_dir  = tloop_dir,
                    tloop_dis  = tloop_dis,
                )
            else
                U = thooftLoop_4D_B_temporal_wing(
                    NC,
                    NDW,
                    Flux,
                    FluxNum,
                    NN[1],
                    NN[2],
                    NN[3],
                    NN[4],
                    overallminus = overallminus,
                    verbose_level = 2,
                    tloop_pos  = tloop_pos,
                    tloop_dir  = tloop_dir,
                    tloop_dis  = tloop_dis,
                )
            end
        else
            error("$dim dimension is not implemented yet!")
        end
    end
    set_wing_U!(U)
    return U
end



function clear_U!(U::T) where {T<:AbstractGaugefields}
    error("clear_U! is not implemented in type $(typeof(U)) ")
end

function clear_U!(U::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    for μ = 1:Dim
        clear_U!(U[μ])
    end
end

function clear_U!(U::T, iseven::Bool) where {T<:AbstractGaugefields}
    error("clear_U! is not implemented in type $(typeof(U)) ")
end

function clear_U!(U::Array{<:AbstractGaugefields{NC,Dim},1}, iseven::Bool) where {NC,Dim}
    for μ = 1:Dim
        clear_U!(U[μ], iseven)
    end
end

function unit_U!(U::T) where {T<:AbstractGaugefields}
    error("unit_U! is not implemented in type $(typeof(U)) ")
end

function unit_U!(U::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    for μ = 1:Dim
        unit_U!(U[μ])
    end
end


function shift_U(U::AbstractGaugefields{NC,Dim}, ν) where {NC,Dim}
    error("shift_U is not implemented in type $(typeof(U)) ")
    return nothing
end

function map_U!(
    U::AbstractGaugefields{NC,Dim},
    f::Function,
    V::AbstractGaugefields{NC,Dim},
    iseven::Bool,
) where {NC,Dim}
    error("map_U! is not implemented in type $(typeof(U)) ")
    return nothing
end

function map_U!(
    U::AbstractGaugefields{NC,Dim},
    f::Function,
    V::AbstractGaugefields{NC,Dim},
) where {NC,Dim}
    error("map_U! is not implemented in type $(typeof(U)) ")
    return nothing
end


function map_U_sequential!(
    U::AbstractGaugefields{NC,Dim},
    f::Function,
    Uin,
) where {NC,Dim}
    error("map_U_sequential! is not implemented in type $(typeof(U)) ")
    return nothing
end


function identitymatrix(U::T) where {T<:AbstractGaugefields}
    error("identitymatrix is not implemented in type $(typeof(U)) ")
end

function set_wing_U!(U::Array{<:AbstractGaugefields{NC,Dim},1}) where {NC,Dim}
    for μ = 1:Dim
        set_wing_U!(U[μ])
    end
end

function set_wing_U!(U::T) where {T<:AbstractGaugefields}
    error("set_wing_U! is not implemented in type $(typeof(U)) ")
end

function evaluate_gaugelinks_evenodd!(
    uout::T,
    w::Wilsonline{Dim},
    U::Array{T,1},
    temps::Array{T,1}, # length >= 3
    iseven,
) where {T<:AbstractGaugefields,Dim}

    #Uold = temps[1]
    Unew = temps[1]
    #Utemp2 = temps[2]
    #clear_U!(uout)
    origin = Tuple(zeros(Int64, Dim))

    Ushift1 = temps[2]
    Ushift2 = temps[3]

    glinks = w
    numlinks = length(glinks)
    if numlinks == 0
        unit_U!(uout)
        return
    end

    j = 1
    U1link = glinks[1]
    direction = get_direction(U1link)
    position = get_position(U1link)
    isU1dag = isdag(U1link)
    #isU1dag = ifelse(typeof(U1link) <: Adjoint_GLink,true,false)


    if numlinks == 1
        substitute_U!(Unew, U[direction])
        Ushift1 = shift_U(Unew, position)
        if isU1dag
            #println("Ushift1 ",Ushift1'[1,1,1,1,1,1])
            substitute_U!(uout, Ushift1', iseven)
        else
            substitute_U!(uout, Ushift1, iseven)
        end
        return
    end

    substitute_U!(Unew, U[direction])
    #println(position)
    Ushift1 = shift_U(Unew, position)

    for j = 2:numlinks
        Ujlink = glinks[j]
        isUkdag = isdag(Ujlink)
        #isUkdag = ifelse(typeof(Ujlink) <: Adjoint_GLink,true,false)
        position = get_position(Ujlink)
        direction = get_direction(Ujlink)

        Ushift2 = shift_U(U[direction], position)
        multiply_12!(uout, Ushift1, Ushift2, j, isUkdag, isU1dag, iseven)


        substitute_U!(Unew, uout)

        Ushift1 = shift_U(Unew, origin)
    end


end

function evaluate_gaugelinks!(
    uout::T,
    w::Wilsonline{Dim},
    U::Array{T,1},
    temps::Array{T,1}, # length >= 3
) where {T<:AbstractGaugefields,Dim}
    #println_verbose_level3(uout,"evaluating Wilson loops")
    #Uold = temps[1]
    #set_wing_U!(U)
    Unew = temps[1]
    #Utemp2 = temps[2]
    #clear_U!(uout)
    origin = Tuple(zeros(Int64, Dim))

    Ushift1 = temps[2]
    Ushift2 = temps[3]

    glinks = w
    numlinks = length(glinks)
    if numlinks == 0
        unit_U!(uout)
        return
    end

    j = 1
    U1link = glinks[1]
    direction = get_direction(U1link)
    position = get_position(U1link)
    isU1dag = isdag(U1link)
    #isU1dag = ifelse(typeof(U1link) <: Adjoint_GLink,true,false)

    #show(glinks)   
    #println("in evaluate_gaugelinks!")
    #show(w)
    #println("numlinks = $numlinks")
    if numlinks == 1
        substitute_U!(Unew, U[direction])
        Ushift1 = shift_U(Unew, position)
        if isU1dag
            #println("Ushift1 ",Ushift1'[1,1,1,1,1,1])
            substitute_U!(uout, Ushift1')
        else
            substitute_U!(uout, Ushift1)
        end

        return
    end

    #j = 1    
    #U1link = glinks[1]
    #direction = get_direction(U1link)
    #position = get_position(U1link)
    #println("i = $i j = $j position = $position")
    substitute_U!(Unew, U[direction])
    Ushift1 = shift_U(Unew, position)

    #ix,iy,iz,it=(2,2,2,2)
    #println("posotion = $position")
    #pos = Tuple([ix,iy,iz,it] .+ collect(position))
    #U1 = Unew[:,:,pos...]
    #println("U1, ",Unew[:,:,pos...])
    #isU1dag = U1link.isdag
    #isU1dag = ifelse(typeof(U1link) <: Adjoint_GLink,true,false)




    for j = 2:numlinks
        Ujlink = glinks[j]
        isUkdag = isdag(Ujlink)
        #isUkdag = ifelse(typeof(Ujlink) <: Adjoint_GLink,true,false)
        position = get_position(Ujlink)
        direction = get_direction(Ujlink)
        #println("j = $j position = $position")
        #println("a,b, $isUkdag , $isU1dag")
        Ushift2 = shift_U(U[direction], position)

        #zerocheck(U,Ushift1.parent.Ushifted,"Ushift1")
        #zerocheck(U,Ushift2.parent.Ushifted,"Ushift2 position $position")

        multiply_12!(uout, Ushift1, Ushift2, j, isUkdag, isU1dag)


        #zerocheck(U,uout.U,"uout")

        #pos = Tuple([ix,iy,iz,it] .+ collect(position))
        #U2 = U[direction][:,:,pos...]
        #println("U1U2dag ", U1*U2')
        substitute_U!(Unew, uout)


        #println("Unew ", Unew[:,:,ix,iy,iz,it])

        Ushift1 = shift_U(Unew, origin)
        #println("uout ", uout[:,:,ix,iy,iz,it])
    end

    #zerocheck(U,uout.U,"uoutfinal")


    #println("uout2 ", uout[:,:,ix,iy,iz,it])


end

function evaluate_gaugelinks!(
    uout::T,
    w::Wilsonline{Dim},
    U::Array{T,1},
    B::Array{T,2},
    temps::Array{T,1}, # length >= 3
) where {T<:AbstractGaugefields,Dim}
    Unew = temps[1]
    origin = Tuple(zeros(Int64, Dim))

    Ushift1 = temps[2]
    Ushift2 = temps[3]

    glinks = w
    numlinks = length(glinks)
    if numlinks == 0
        unit_U!(uout)
        return
    end

    j = 1
    U1link = glinks[1]
    direction = get_direction(U1link)
    position = get_position(U1link)
    isU1dag = isdag(U1link)

    if numlinks == 1
        substitute_U!(Unew, U[direction])
        Ushift1 = shift_U(Unew, position)
        if isU1dag
            substitute_U!(uout, Ushift1')
        else
            substitute_U!(uout, Ushift1)
        end

        return
    end

    substitute_U!(Unew, U[direction])
    Ushift1 = shift_U(Unew, position)

    for j = 2:numlinks
        Ujlink = glinks[j]
        isUkdag = isdag(Ujlink)
        position = get_position(Ujlink)
        direction = get_direction(Ujlink)
        Ushift2 = shift_U(U[direction], position)

        multiply_12!(uout, Ushift1, Ushift2, j, isUkdag, isU1dag)

        substitute_U!(Unew, uout)
        Ushift1 = shift_U(Unew, origin)
    end

    multiply_Bplaquettes!(uout, w, B, temps)

end

function evaluate_Bplaquettes!(
    uout::T,
    w::Wilsonline{Dim},
    B::Array{T,2},
    temps::Array{T,1},
) where {T<:AbstractGaugefields,Dim}
    multiply_Bplaquettes!(uout,w,B,temps,true)
end
function multiply_Bplaquettes!(
    uout::T,
    w::Wilsonline{Dim},
    B::Array{T,2},
    temps::Array{T,1},
    unity = false,
) where {T<:AbstractGaugefields,Dim}
    if unity
        unit_U!(uout)
    end

    glinks = w
    numlinks = length(glinks)
    if numlinks < 3
        return
    end

    if !(isLoopwithB(glinks) || isStaplewithB(glinks))
        return
    end

    for j = 1:numlinks
        sweepaway_4D_Bplaquettes!(uout, glinks, B, temps, j)
    end

end

function sweepaway_4D_Bplaquettes!(
    uout::T,
    w::Wilsonline{Dim},
    B::Array{T,2},
    temps::Array{T,1}, # length(temps) >= 4
    linknum,
) where {T<:AbstractGaugefields,Dim}
    Unew = temps[1]
    glinks = w
    origin = get_position(glinks[1])  #Tuple(zeros(Int64, Dim))
    if isdag(glinks[1])
        origin_shift = [0,0,0,0]
        origin_shift[get_direction(glinks[1])] += 1
        origin = Tuple(origin_shift .+ collect(origin))
    end

    numlinks = length(glinks)
    if numlinks < linknum
        return
    end

    U1link = glinks[linknum]
    direction = get_direction(U1link)
    isU1dag = isdag(U1link)

    coordinate = [0,0,0,0] .+ collect(origin)
    for j = 1:(linknum-1)
        Ujlink = glinks[j]
        j_direction = get_direction(Ujlink)
        isUjdag = isdag(Ujlink)

        if isUjdag
            coordinate[j_direction] += -1
        else
            coordinate[j_direction] += +1
        end
    end
    if isU1dag
        coordinate[direction] += -1
    end

    substitute_U!(Unew,uout)
    Ushift = shift_U(Unew, (0,0,0,0))

    if direction == 1
        if isU1dag
            Bshift12 = shift_U(B[1,2], (0,0,0,0))
            Bshift13 = shift_U(B[1,3], (0,0,0,0))
            Bshift14 = shift_U(B[1,4], (0,0,0,0))
        else
            Bshift12 = shift_U(B[1,2], (0,0,0,0))'
            Bshift13 = shift_U(B[1,3], (0,0,0,0))'
            Bshift14 = shift_U(B[1,4], (0,0,0,0))'
        end

        Bshift12new = temps[2]
        Bshift13new = temps[3]
        Bshift14new = temps[4]

        for ix = 1:abs(coordinate[1])
            if coordinate[1] > 0
                substitute_U!(Bshift12new,Bshift12)
                Bshift12 = shift_U(Bshift12new, (1,0,0,0))
                substitute_U!(Bshift13new,Bshift13)
                Bshift13 = shift_U(Bshift13new, (1,0,0,0))
                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (1,0,0,0))
            else # coordinate[1] < 0
                substitute_U!(Bshift12new,Bshift12)
                Bshift12 = shift_U(Bshift12new, (-1,0,0,0))
                substitute_U!(Bshift13new,Bshift13)
                Bshift13 = shift_U(Bshift13new, (-1,0,0,0))
                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (-1,0,0,0))
            end
        end
        
        for iy = 1:abs(coordinate[2])
            if coordinate[2] > 0
                multiply_12!(uout, Ushift, Bshift12, 0, false, false)

                substitute_U!(Bshift12new,Bshift12)
                Bshift12 = shift_U(Bshift12new, (0,1,0,0))
                substitute_U!(Bshift13new,Bshift13)
                Bshift13 = shift_U(Bshift13new, (0,1,0,0))
                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (0,1,0,0))
            else # coordinate[2] < 0
                substitute_U!(Bshift12new,Bshift12)
                Bshift12 = shift_U(Bshift12new, (0,-1,0,0))
                substitute_U!(Bshift13new,Bshift13)
                Bshift13 = shift_U(Bshift13new, (0,-1,0,0))
                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (0,-1,0,0))

                multiply_12!(uout, Ushift, Bshift12, 0, true, false)
            end
            
            substitute_U!(Unew,uout)
            Ushift = shift_U(Unew, origin)
            
        end
        
        for iz = 1:abs(coordinate[3])
            if coordinate[3] > 0
                multiply_12!(uout, Ushift, Bshift13, 0, false, false)

                substitute_U!(Bshift13new,Bshift13)
                Bshift13 = shift_U(Bshift13new, (0,0,1,0))
                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (0,0,1,0))
            else # coordinate[3] < 0
                substitute_U!(Bshift13new,Bshift13)
                Bshift13 = shift_U(Bshift13new, (0,0,-1,0))
                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (0,0,-1,0))

                multiply_12!(uout, Ushift, Bshift13, 0, true, false)
            end
            
            substitute_U!(Unew,uout)
            Ushift = shift_U(Unew, origin)
            
        end

        for it = 1:abs(coordinate[4])
            if coordinate[4] > 0
                multiply_12!(uout, Ushift, Bshift14, 0, false, false)

                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (0,0,0,1))
            else # coordinate[4] < 0
                substitute_U!(Bshift14new,Bshift14)
                Bshift14 = shift_U(Bshift14new, (0,0,0,-1))

                multiply_12!(uout, Ushift, Bshift14, 0, true, false)
            end
            
            substitute_U!(Unew,uout)
            Ushift = shift_U(Unew, origin)
            
        end
    elseif direction == 2
        if isU1dag
            Bshift23 = shift_U(B[2,3], (0,0,0,0))
            Bshift24 = shift_U(B[2,4], (0,0,0,0))
        else
            Bshift23 = shift_U(B[2,3], (0,0,0,0))'
            Bshift24 = shift_U(B[2,4], (0,0,0,0))'
        end

        Bshift23new = temps[2]
        Bshift24new = temps[3]

        for ix = 1:abs(coordinate[1])
            if coordinate[1] > 0
                substitute_U!(Bshift23new,Bshift23)
                Bshift23 = shift_U(Bshift23new, (1,0,0,0))
                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (1,0,0,0))
            else # coordinate[1] < 0
                substitute_U!(Bshift23new,Bshift23)
                Bshift23 = shift_U(Bshift23new, (-1,0,0,0))
                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (-1,0,0,0))
            end
        end
        
        for iy = 1:abs(coordinate[2])
            if coordinate[2] > 0
                substitute_U!(Bshift23new,Bshift23)
                Bshift23 = shift_U(Bshift23new, (0,1,0,0))
                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (0,1,0,0))
            else # coordinate[2] < 0
                substitute_U!(Bshift23new,Bshift23)
                Bshift23 = shift_U(Bshift23new, (0,-1,0,0))
                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (0,-1,0,0))
            end
        end

        for iz = 1:abs(coordinate[3])
            if coordinate[3] > 0
                multiply_12!(uout, Ushift, Bshift23, 0, false, false)

                substitute_U!(Bshift23new,Bshift23)
                Bshift23 = shift_U(Bshift23new, (0,0,1,0))
                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (0,0,1,0))
            else # coordinate[3] < 0
                substitute_U!(Bshift23new,Bshift23)
                Bshift23 = shift_U(Bshift23new, (0,0,-1,0))
                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (0,0,-1,0))

                multiply_12!(uout, Ushift, Bshift23, 0, true, false)
            end
            
            substitute_U!(Unew,uout)
            Ushift = shift_U(Unew, origin)
            
        end

        for it = 1:abs(coordinate[4])
            if coordinate[4] > 0
                multiply_12!(uout, Ushift, Bshift24, 0, false, false)

                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (0,0,0,1))
            else # coordinate[4] < 0
                substitute_U!(Bshift24new,Bshift24)
                Bshift24 = shift_U(Bshift24new, (0,0,0,-1))

                multiply_12!(uout, Ushift, Bshift24, 0, true, false)
            end
            
            substitute_U!(Unew,uout)
            Ushift = shift_U(Unew, origin)
        end
    elseif direction == 3
        if isU1dag
            Bshift34 = shift_U(B[3,4], (0,0,0,0))
        else
            Bshift34 = shift_U(B[3,4], (0,0,0,0))'
        end

        Bshift34new = temps[2]

        for ix = 1:abs(coordinate[1])
            if coordinate[1] > 0
                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (1,0,0,0))
            else # coordinate[1] < 0
                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (-1,0,0,0))
            end
        end
        
        for iy = 1:abs(coordinate[2])
            if coordinate[2] > 0
                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (0,1,0,0))
            else # coordinate[2] < 0
                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (0,-1,0,0))
            end
        end
        
        for iz = 1:abs(coordinate[3])
            if coordinate[3] > 0
                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (0,0,1,0))
            else # coordinate[3] < 0
                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (0,0,-1,0))
            end
        end
        
        for it = 1:abs(coordinate[4])
            if coordinate[4] > 0
                multiply_12!(uout, Ushift, Bshift34, 0, false, false)

                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (0,0,0,1))
            else # coordinate[4] < 0
                substitute_U!(Bshift34new,Bshift34)
                Bshift34 = shift_U(Bshift34new, (0,0,0,-1))

                multiply_12!(uout, Ushift, Bshift34, 0, true, false)
            end
            
            substitute_U!(Unew,uout)
            Ushift = shift_U(Unew, origin)
            
        end
    else
        # direction==4: no multiplications
    end
end


function isLoopwithB(
    w::Wilsonline{Dim},
) where {Dim}
    glinks = w
    numlinks = length(glinks)
    if numlinks < 4
        return false
    end

    coordinate = [0,0,0,0]
    for j = 1:numlinks
        Ujlink = glinks[j]
        direction = get_direction(Ujlink)
        isU1dag = isdag(Ujlink)
        if isU1dag
            coordinate[direction] += -1
        else
            coordinate[direction] += +1
        end
    end

    if coordinate == [0,0,0,0]
        return true
    else
        return false
    end
    
end

function isStaplewithB(
    w::Wilsonline{Dim},
) where {Dim}
    glinks = w
    numlinks = length(glinks)
    if numlinks < 3
        return false
    end

    coordinate = [0,0,0,0]
    for j = 1:numlinks
        Ujlink = glinks[j]
        direction = get_direction(Ujlink)
        isU1dag = isdag(Ujlink)
        if isU1dag
            coordinate[direction] += -1
        else
            coordinate[direction] += +1
        end
    end

    if norm(coordinate,1) == 1.0
        return true
    else
        return false
    end
    
end

function zerocheck(U, U2bare, Uname)
    for myrank = 0:(get_nprocs(U)-1)
        #println(get_nprocs(U))
        if get_myrank(U) == myrank

            #println("rank = $myrank Ushift1",Ushift1.parent.Ushifted)
            #println("rank = $myrank Ushift2",Ushift2.parent.Ushifted)

            nc, _, nx, ny, nz, nt = size(U2bare)
            for it = 1:nt
                for iz = 1:nz
                    for iy = 1:ny
                        for ix = 1:nx
                            ic = 1
                            jc = 1
                            #for ic=1:nc
                            #    for jc = 1:nc
                            v = U2bare[jc, ic, ix, iy, iz, it]
                            if abs(v) <= 1e-15
                                println("rank = $myrank ", Uname)
                                println(
                                    "$position type: $(typeof(U))\n ",
                                    v,
                                    " at $((jc,ic,ix,iy,iz,it))",
                                )
                            end

                            #    end
                            #end
                        end
                    end
                end
            end

            #println("rank $myrank j = $j $position Ushift1 $(typeof(Ushift1)) ",getvalue(Ushift1,1,1,1,1,1,1))
            #println("rank $myrank j = $j $position Ushift2 $(typeof(Ushift2)) ",getvalue(Ushift2,1,1,1,1,1,1))
        end
        barrier(U[1])
    end
end
function evaluate_gaugelinks_evenodd!(
    xout::T,
    w::Array{<:Wilsonline{Dim},1},
    U::Array{T,1},
    temps::Array{T,1}, # length >= 4
    iseven,
) where {T<:AbstractGaugefields,Dim}
    num = length(w)
    temp4 = temps[4]

    #ix,iy,iz,it=(2,2,2,2)

    clear_U!(xout, iseven)
    for i = 1:num
        glinks = w[i]
        evaluate_gaugelinks_evenodd!(temp4, glinks, U, temps[1:3], iseven) # length >= 3
        #println("uout2 ", temp2[:,:,ix,iy,iz,it])
        add_U!(xout, temp4, iseven)
        #println("xout ", xout[:,:,ix,iy,iz,it])
    end

    #println("xout2 ", xout[:,:,ix,iy,iz,it])
    return


end

function evaluate_gaugelinks!(
    xout::T,
    w::Array{WL,1},
    U::Array{T,1},
    temps::Array{T,1}, # length >= 4
) where {Dim,WL<:Wilsonline{Dim},T<:AbstractGaugefields}
    num = length(w)
    temp4 = temps[4]

    #ix,iy,iz,it=(2,2,2,2)
    #ix,iy,iz,it=(1,1,1,1)

    clear_U!(xout)
    for i = 1:num
        glinks = w[i]
        evaluate_gaugelinks!(temp4, glinks, U, temps[1:3]) # length >= 3
        #println("uout2 ", temp2[:,:,ix,iy,iz,it])
        add_U!(xout, temp4)
        #println("xout ", xout[:,:,ix,iy,iz,it])
    end

    #println("xout2 ", xout[:,:,ix,iy,iz,it])
    return


end

function evaluate_gaugelinks!(
    xout::T,
    w::Array{WL,1},
    U::Array{T,1},
    B::Array{T,2},
    temps::Array{T,1}, # length >= 5
) where {Dim,WL<:Wilsonline{Dim},T<:AbstractGaugefields}
    num = length(w)
    temp1 = temps[5]

    clear_U!(xout)
    for i = 1:num
        glinks = w[i]
        evaluate_gaugelinks!(temp1, glinks, U, B, temps[1:4]) # length >= 4
        add_U!(xout, temp1)
    end

    return
end

#=
function evaluate_gaugelinks_inside!(U,glinks,Ushift1,Uold,Unew,numlinks)
    for k=2:numlinks
        position =get_position(glinks[k])
        direction = get_direction(glinks[k])
        Ushift2 = shift_U(U[direction],position)
        multiply_12!(Unew,Ushift1,Ushift2,k,loopk,loopk1_2)

        Unew,Uold = Uold,Unew
        Ushift1 = shift_U(Uold,(0,0,0,0))
    end
end
=#

function multiply_12!(temp3, temp1, temp2, k, isUkdag::Bool, isU1dag::Bool)
    if k == 2
        if isUkdag
            if isU1dag
                mul!(temp3, temp1', temp2')
            else
                mul!(temp3, temp1, temp2')
            end
        else
            if isU1dag
                mul!(temp3, temp1', temp2)
            else
                mul!(temp3, temp1, temp2)
            end
        end
    else
        if isUkdag
            mul!(temp3, temp1, temp2')
        else
            mul!(temp3, temp1, temp2)
        end
    end
    return
end


function multiply_12!(temp3, temp1, temp2, k, isUkdag::Bool, isU1dag::Bool, iseven)
    if k == 2
        if isUkdag
            if isU1dag
                mul!(temp3, temp1', temp2', iseven)
            else
                mul!(temp3, temp1, temp2', iseven)
            end
        else
            if isU1dag
                mul!(temp3, temp1', temp2, iseven)
            else
                mul!(temp3, temp1, temp2, iseven)
            end
        end
    else
        if isUkdag
            mul!(temp3, temp1, temp2', iseven)
        else
            mul!(temp3, temp1, temp2, iseven)
        end
    end
    return
end



function evaluate_gaugelinks_eachsite!(
    uout_mat::T1,
    w::Array{<:Wilsonline{Dim},1},
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps, # length >= 4
    indices...,
) where {Dim,T1<:Matrix{ComplexF64},NC}

    #xout::T,w::Array{<:Wilsonline{Dim},1},U::Array{T,1},temps::Array{T,1},iseven) where {T<: AbstractGaugefields,Dim}
    num = length(w)
    temp = temps[4]


    #ix,iy,iz,it=(2,2,2,2)

    uout_mat .= 0
    for i = 1:num
        glinks = w[i]
        evaluate_gaugelinks_eachsite!(temp, glinks, U, view(temps, 1:3), indices...) # length >= 3

        #println("uout2 ", temp2[:,:,ix,iy,iz,it])
        uout_mat .+= temp
        #println("xout ", xout[:,:,ix,iy,iz,it])
    end

    #println("xout2 ", xout[:,:,ix,iy,iz,it])
    return


end


function evaluate_gaugelinks_eachsite!(
    uout_mat::T1,
    w::Wilsonline{Dim},
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    temps, # length >= 3
    indices...,
) where {Dim,T1<:Matrix{ComplexF64},NC}
    _, _, NN... = size(U[1])
    Unew = temps[1]
    Ushift1 = temps[2]
    Ushift2 = temps[3]

    #origin = Tuple(zeros(Int64,Dim))
    shifted = zeros(Int64, Dim)

    glinks = w
    numlinks = length(glinks)
    if numlinks == 0
        for i = 1:NC
            for j = 1:NC
                uout_mat[j, i] = ifelse(i == j, 1, 0)
            end
        end
        return
    end

    j = 1
    U1link = glinks[1]
    direction = get_direction(U1link)
    position = get_position(U1link)
    isU1dag = isdag(U1link)#.isdag
    #isU1dag = ifelse(typeof(U1link) <: Adjoint_GLink,true,false)

    #show(glinks)   
    #println("in evaluate_gaugelinks!")
    #show(w)
    #println("numlinks = $numlinks")
    if numlinks == 1
        for i = 1:NC
            for j = 1:NC
                Unew[j, i] = U[direction][j, i, indices...]
            end
        end

        #substitute_U!(Unew,U[direction])
        for μ = 1:Dim
            shifted[μ] = indices[μ] + position[μ]# collect(position)
        end
        #shifted = collect(indices) .+ collect(position)
        for μ = 1:Dim
            shifted[μ] += ifelse(shifted[μ] < 1, NN[μ], 0)
            shifted[μ] += ifelse(shifted[μ] > NN[μ], -NN[μ], 0)
        end
        #shiftedposition = Tuple(shifted)
        #shiftedposition = Tuple(collect(indices) .+ collect(position))
        for i = 1:NC
            for j = 1:NC
                #Ushift1[j,i] = Unew[j,i,shiftedposition...]
                Ushift1[j, i] = Unew[j, i, shifted...]
            end
        end

        #Ushift1 = shift_U(Unew,position)
        if isU1dag
            #println("Ushift1 ",Ushift1'[1,1,1,1,1,1])
            for i = 1:NC
                for j = 1:NC
                    Unew[j, i] = Ushift1'[j, i]#,indices]
                end
            end
            #substitute_U!(uout,Ushift1')
        else
            for i = 1:NC
                for j = 1:NC
                    Unew[j, i] = Ushift1[j, i]#,indices]
                end
            end
            #substitute_U!(uout,Ushift1)
        end
        return
    end

    #j = 1    
    #U1link = glinks[1]
    #direction = get_direction(U1link)
    #position = get_position(U1link)
    #println("i = $i j = $j position = $position")
    #substitute_U!(Unew,U[direction])
    #for b=1:NC
    #    for a=1:NC
    #        Unew[a,b] = U[direction][a,b,indices...]
    #    end
    #end

    for μ = 1:Dim
        shifted[μ] = indices[μ] + position[μ]# collect(position)
    end
    #shifted = collect(indices) .+ collect(position)
    for μ = 1:Dim
        shifted[μ] += ifelse(shifted[μ] < 1, NN[μ], 0)
        shifted[μ] += ifelse(shifted[μ] > NN[μ], -NN[μ], 0)
    end
    #shiftedposition = Tuple(shifted)
    for i = 1:NC
        for j = 1:NC
            #Ushift1[j,i] = U[direction][j,i,shiftedposition...]
            Ushift1[j, i] = U[direction][j, i, shifted...]
        end
    end

    #Ushift1 = shift_U(Unew,position)

    #ix,iy,iz,it=(2,2,2,2)
    #println("posotion = $position")
    #pos = Tuple([ix,iy,iz,it] .+ collect(position))
    #U1 = Unew[:,:,pos...]
    #println("U1, ",Unew[:,:,pos...])
    #isU1dag = ifelse(typeof(U1link) <: Adjoint_GLink,true,false)




    for j = 2:numlinks
        Ujlink = glinks[j]
        isUkdag = isdag(Ujlink)
        #isUkdag = ifelse(typeof(Ujlink) <: Adjoint_GLink,true,false)
        position = get_position(Ujlink)
        direction = get_direction(Ujlink)
        #println("j = $j position = $position")
        #println("a,b, $isUkdag , $isU1dag")
        for μ = 1:Dim
            shifted[μ] = indices[μ] + position[μ]# collect(position)
        end
        #shifted = collect(indices) .+ collect(position)
        for μ = 1:Dim
            shifted[μ] += ifelse(shifted[μ] < 1, NN[μ], 0)
            shifted[μ] += ifelse(shifted[μ] > NN[μ], -NN[μ], 0)
        end
        #shiftedposition2 = Tuple(shifted)
        ##  shiftedposition2 = Tuple(collect(indices) .+ collect(position))
        for b = 1:NC
            for a = 1:NC
                #Ushift2[a,b] = U[direction][a,b,shiftedposition2...]
                Ushift2[a, b] = U[direction][a, b, shifted...]
            end
        end

        #Ushift2 = shift_U(U[direction],position)
        multiply_12!(uout_mat, Ushift1, Ushift2, j, isUkdag, isU1dag)

        #pos = Tuple([ix,iy,iz,it] .+ collect(position))
        #U2 = U[direction][:,:,pos...]
        #println("U1U2dag ", U1*U2')
        #substitute_U!(Unew,uout)
        for b = 1:NC
            for a = 1:NC
                Unew[a, b] = uout_mat[a, b]
            end
        end

        #println("Unew ", Unew[:,:,ix,iy,iz,it])
        for b = 1:NC
            for a = 1:NC
                Ushift1[a, b] = Unew[a, b]
            end
        end

        #Ushift1 = shift_U(Unew,origin)
        #println("uout ", uout[:,:,ix,iy,iz,it])
    end


    #println("uout2 ", uout[:,:,ix,iy,iz,it])


end


const shifted_4_temp = zeros(Int64, 4)
const NN_4_temp = zeros(Int64, 4)

function evaluate_gaugelinks_eachsite!(
    uout_mat::T1,
    w::Wilsonline{4},
    U::Array{<:AbstractGaugefields{NC,4},1},
    temps, # length >= 3
    indices...,
) where {T1<:Matrix{ComplexF64},NC}
    #NN = NN_4_temp
    #NN[1] = U[1].NX
    #NN[2] = U[1].NY
    #NN[3] = U[1].NZ
    #NN[4] = U[1].NT
    _, _, NN... = size(U[1])


    Unew = temps[1]
    Ushift1 = temps[2]
    Ushift2 = temps[3]

    #origin = Tuple(zeros(Int64,Dim))
    shifted = shifted_4_temp #zeros(Int64,Dim)

    glinks = w
    numlinks = length(glinks)
    if numlinks == 0
        for i = 1:NC
            for j = 1:NC
                uout_mat[j, i] = ifelse(i == j, 1, 0)
            end
        end
        return
    end

    j = 1
    U1link = glinks[1]
    direction = get_direction(U1link)
    position = get_position(U1link)
    isU1dag = isdag(U1link)#.isdag
    #isU1dag = ifelse(typeof(U1link) <: Adjoint_GLink,true,false)

    #show(glinks)   
    #println("in evaluate_gaugelinks!")
    #show(w)
    #println("numlinks = $numlinks")
    if numlinks == 1
        for i = 1:NC
            for j = 1:NC
                Unew[j, i] = U[direction][j, i, indices...]
            end
        end

        #substitute_U!(Unew,U[direction])
        for μ = 1:4
            shifted[μ] = indices[μ] + position[μ]# collect(position)
        end
        #shifted = collect(indices) .+ collect(position)
        for μ = 1:4
            shifted[μ] += ifelse(shifted[μ] < 1, NN[μ], 0)
            shifted[μ] += ifelse(shifted[μ] > NN[μ], -NN[μ], 0)
        end
        #shiftedposition = Tuple(shifted)
        #shiftedposition = Tuple(collect(indices) .+ collect(position))
        for i = 1:NC
            for j = 1:NC
                #Ushift1[j,i] = Unew[j,i,shiftedposition...]
                #Ushift1[j,i] = Unew[j,i,shifted...]
                Ushift1[j, i] = Unew[j, i, shifted[1], shifted[2], shifted[3], shifted[4]]
            end
        end

        #Ushift1 = shift_U(Unew,position)
        if isU1dag
            #println("Ushift1 ",Ushift1'[1,1,1,1,1,1])
            for i = 1:NC
                for j = 1:NC
                    Unew[j, i] = Ushift1'[j, i]#,indices]
                end
            end
            #substitute_U!(uout,Ushift1')
        else
            for i = 1:NC
                for j = 1:NC
                    Unew[j, i] = Ushift1[j, i]#,indices]
                end
            end
            #substitute_U!(uout,Ushift1)
        end
        return
    end

    #j = 1    
    #U1link = glinks[1]
    #direction = get_direction(U1link)
    #position = get_position(U1link)
    #println("i = $i j = $j position = $position")
    #substitute_U!(Unew,U[direction])
    #for b=1:NC
    #    for a=1:NC
    #        Unew[a,b] = U[direction][a,b,indices...]
    #    end
    #end


    for μ = 1:4
        shifted[μ] = indices[μ] + position[μ]# collect(position)
    end
    #shifted = collect(indices) .+ collect(position)
    for μ = 1:4
        shifted[μ] += ifelse(shifted[μ] < 1, NN[μ], 0)
        shifted[μ] += ifelse(shifted[μ] > NN[μ], -NN[μ], 0)
    end
    #shiftedposition = Tuple(shifted)
    for i = 1:NC
        for j = 1:NC
            #Ushift1[j,i] = U[direction][j,i,shiftedposition...]
            #Ushift1[j,i] = U[direction][j,i,shifted...]
            Ushift1[j, i] =
                U[direction][j, i, shifted[1], shifted[2], shifted[3], shifted[4]]

        end
    end

    #Ushift1 = shift_U(Unew,position)

    #ix,iy,iz,it=(2,2,2,2)
    #println("posotion = $position")
    #pos = Tuple([ix,iy,iz,it] .+ collect(position))
    #U1 = Unew[:,:,pos...]
    #println("U1, ",Unew[:,:,pos...])
    #isU1dag = ifelse(typeof(U1link) <: Adjoint_GLink,true,false)




    for j = 2:numlinks
        Ujlink = glinks[j]
        isUkdag = isdag(Ujlink)
        #isUkdag = ifelse(typeof(Ujlink) <: Adjoint_GLink,true,false)
        position = get_position(Ujlink)
        direction = get_direction(Ujlink)
        #println("j = $j position = $position")
        #println("a,b, $isUkdag , $isU1dag")
        for μ = 1:4
            shifted[μ] = indices[μ] + position[μ]# collect(position)
        end
        #shifted = collect(indices) .+ collect(position)
        for μ = 1:4
            shifted[μ] += ifelse(shifted[μ] < 1, NN[μ], 0)
            shifted[μ] += ifelse(shifted[μ] > NN[μ], -NN[μ], 0)
        end
        #shiftedposition2 = Tuple(shifted)
        ##  shiftedposition2 = Tuple(collect(indices) .+ collect(position))
        for b = 1:NC
            for a = 1:NC
                #Ushift2[a,b] = U[direction][a,b,shiftedposition2...]
                #Ushift2[a,b] = U[direction][a,b,shifted...]
                Ushift2[a, b] =
                    U[direction][a, b, shifted[1], shifted[2], shifted[3], shifted[4]]
            end
        end

        #Ushift2 = shift_U(U[direction],position)
        multiply_12!(uout_mat, Ushift1, Ushift2, j, isUkdag, isU1dag)

        #pos = Tuple([ix,iy,iz,it] .+ collect(position))
        #U2 = U[direction][:,:,pos...]
        #println("U1U2dag ", U1*U2')
        #substitute_U!(Unew,uout)
        for b = 1:NC
            for a = 1:NC
                Unew[a, b] = uout_mat[a, b]
            end
        end

        #println("Unew ", Unew[:,:,ix,iy,iz,it])
        for b = 1:NC
            for a = 1:NC
                Ushift1[a, b] = Unew[a, b]
            end
        end

        #Ushift1 = shift_U(Unew,origin)
        #println("uout ", uout[:,:,ix,iy,iz,it])
    end


    #println("uout2 ", uout[:,:,ix,iy,iz,it])


end






function evaluate_wilson_loops!(
    xout::T,
    w::Wilson_loop_set,
    U::Array{T,1},
    temps::Array{T,1},
) where {T<:AbstractGaugefields}
    num = length(w)
    clear_U!(xout)
    Uold = temps[1]
    Unew = temps[2]

    for i = 1:num
        wi = w[i]
        numloops = length(wi)

        shifts = calc_shift(wi)
        #println("shift ",shifts)

        loopk = wi[1]
        k = 1
        #println("k = $k shift: ",shifts[k])
        substitute_U!(Uold, U[loopk[1]])
        Ushift1 = shift_U(Uold, shifts[1])

        #gauge_shift_all!(temp1,shifts[1],U[loopk[1]])


        loopk1_2 = loopk[2]
        evaluate_wilson_loops_inside!(
            U,
            shifts,
            wi,
            Ushift1,
            Uold,
            Unew,
            numloops,
            loopk,
            loopk1_2,
        )

        #=
        for k=2:numloops

            loopk = wi[k]
            #println("k = $k shift: ",shifts[k])
            #println("gauge_shift!(temp2,$(shifts[k]),$(loopk[1]) )")
            #clear!(temp2)
            Ushift2 = shift_U(U[loopk[1]],shifts[k])
            #gauge_shift_all!(temp2,shifts[k],U[loopk[1]])

            #multiply_12!(temp3,temp1,temp2,k,loopk,loopk1_2)
            multiply_12!(Unew,Ushift1,Ushift2,k,loopk,loopk1_2)
            Unew,Uold = Uold,Unew
            #Ushift1 = shift_U(Uold,(0,0,0,0))
            Ushift1 = Uold
            #temp1,temp3 = temp3,temp1
        end
        =#
        add_U!(xout, Uold)
        #println("i = $i ",Uold[1,1,1,1,1,1])
        #add_U!(xout,Ushift1)
        #add!(xout,temp1)

    end
end
function evaluate_wilson_loops!(
    xout::T,
    w::Wilson_loop_set,
    U::Array{T,1},
    B::Array{T,2},
    temps::Array{T,1},
) where {T<:AbstractGaugefields}
    num = length(w)
    clear_U!(xout)
    Uold = temps[1]
    Unew = temps[2]

    for i = 1:num
        wi = w[i]
        numloops = length(wi)
        shifts = calc_shift(wi)

        loopk = wi[1]
        k = 1
        substitute_U!(Uold, U[loopk[1]])
        Ushift1 = shift_U(Uold, shifts[1])

        loopk1_2 = loopk[2]
        evaluate_wilson_loops_inside!(
            U,
            B,
            shifts,
            wi,
            Ushift1,
            Uold,
            Unew,
            numloops,
            loopk,
            loopk1_2,
            temps,
        )
        add_U!(xout, Uold)
    end
end

function evaluate_wilson_loops_inside!(
    U,
    shifts,
    wi,
    Ushift1,
    Uold,
    Unew,
    numloops,
    loopk,
    loopk1_2,
)
    for k = 2:numloops

        loopk = wi[k]
        #println("k = $k shift: ",shifts[k])
        #println("gauge_shift!(temp2,$(shifts[k]),$(loopk[1]) )")
        #clear!(temp2)
        Ushift2 = shift_U(U[loopk[1]], shifts[k])
        #gauge_shift_all!(temp2,shifts[k],U[loopk[1]])

        #multiply_12!(temp3,temp1,temp2,k,loopk,loopk1_2)
        multiply_12!(Unew, Ushift1, Ushift2, k, loopk, loopk1_2)

        Unew, Uold = Uold, Unew
        Ushift1 = shift_U(Uold, (0, 0, 0, 0))

        #Ushift1 = Uold
        #temp1,temp3 = temp3,temp1
    end
end
function evaluate_wilson_loops_inside!(
    U,
    B,
    shifts,
    wi,
    Ushift1,
    Uold,
    Unew,
    numloops,
    loopk,
    loopk1_2,
    temps,
)
    for k = 2:numloops
        loopk = wi[k]
        Ushift2 = shift_U(U[loopk[1]], shifts[k])

        multiply_12!(Unew, Ushift1, Ushift2, k, loopk, loopk1_2)

        Unew, Uold = Uold, Unew
        Ushift1 = shift_U(Uold, (0, 0, 0, 0))
    end
    multiply_Bplaquettes!(Unew, wi, B, temps)
end


function multiply_12!(temp3, temp1, temp2, k, loopk, loopk1_2)
    if loopk[2] == 1
        if k == 2
            if loopk1_2 == 1
                mul!(temp3, temp1, temp2)
            else
                mul!(temp3, temp1', temp2)
            end
        else
            mul!(temp3, temp1, temp2)
        end
    elseif loopk[2] == -1
        if k == 2
            if loopk1_2 == 1
                mul!(temp3, temp1, temp2')
            else
                mul!(temp3, temp1', temp2')
            end
        else
            mul!(temp3, temp1, temp2')
        end
    else
        error("Second element should be 1 or -1 but now $(loopk)")
    end
    return
end


function calculate_Plaquette(U::Array{T,1}) where {T<:AbstractGaugefields}
    error("calculate_Plaquette is not implemented in type $(typeof(U)) ")
end
function calculate_Plaquette(
    U::Array{T,1},
    B::Array{T,2},
) where {T<:AbstractGaugefields}
    error("calculate_Plaquette is not implemented in type $(typeof(U)) ")
end

function calculate_Plaquette(
    U::Array{T,1},
    temps::Array{T1,1},
) where {T<:AbstractGaugefields,T1<:AbstractGaugefields}
    return calculate_Plaquette(U, temps[1], temps[2])
end
function calculate_Plaquette(
    U::Array{T,1},
    B::Array{T,2},
    temps::Array{T1,1},
) where {T<:AbstractGaugefields,T1<:AbstractGaugefields}
    return calculate_Plaquette(U, B, temps[1], temps[2])
end

function calculate_Plaquette(
    U::Array{T,1},
    temp::AbstractGaugefields{NC,Dim},
    staple::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    plaq = 0
    V = staple
    for μ = 1:Dim
        construct_staple!(V, U, μ, temp)
        mul!(temp, U[μ], V')
        plaq += tr(temp)

    end
    return real(plaq * 0.5)
end
function calculate_Plaquette(
    U::Array{T,1},
    B::Array{T,2},
    temp::AbstractGaugefields{NC,Dim},
    staple::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    plaq = 0
    V = staple
    for μ = 1:Dim
        construct_staple!(V, U, B, μ, temp)
        mul!(temp, U[μ], V')
        plaq += tr(temp)

    end
    return real(plaq * 0.5)
end

function construct_staple!(staple::T, U, μ) where {T<:AbstractGaugefields}
    error("construct_staple! is not implemented in type $(typeof(U)) ")
end
function construct_staple!(staple::T, U, B, μ) where {T<:AbstractGaugefields}
    error("construct_staple! is not implemented in type $(typeof(U)) ")
end

function add_force!(
    F::Array{T1,1},
    U::Array{T2,1},
    temps::Array{<:AbstractGaugefields{NC,Dim},1};
    plaqonly = false,
    staplefactors::Union{Array{<:Number,1},Nothing} = nothing,
    factor = 1,
) where {NC,Dim,T1<:AbstractGaugefields,T2<:AbstractGaugefields}
    error("add_force! is not implemented in type $(typeof(F)) ")
end
function add_force!(
    F::Array{T1,1},
    U::Array{T2,1},
    B::Array{T2,2},
    temps::Array{<:AbstractGaugefields{NC,Dim},1};
    plaqonly = false,
    staplefactors::Union{Array{<:Number,1},Nothing} = nothing,
    factor = 1,
) where {NC,Dim,T1<:AbstractGaugefields,T2<:AbstractGaugefields}
    error("add_force! is not implemented in type $(typeof(F)) ")
end

function add_force!(
    F::Array{T1,1},
    U::Array{T2,1},
    temps::Array{<:AbstractGaugefields{NC,Dim},1};
    plaqonly = false,
    staplefactors::Union{Array{<:Number,1},Nothing} = nothing,
    factor = 1,
) where {NC,Dim,T1<:TA_Gaugefields,T2<:AbstractGaugefields}
    @assert length(temps) >= 3 "length(temps) should be >= 3. But $(length(temps))"
    #println("add force, plaqonly = $plaqonly")

    V = temps[3]
    temp1 = temps[1]
    temp2 = temps[2]

    for μ = 1:Dim
        if plaqonly

            construct_double_staple!(V, U, μ, temps[1:2])

            mul!(temp1, U[μ], V') #U U*V
        else
            clear_U!(V)
            for i = 1:gparam.numactions
                loops = gparam.staples[i][μ]
                evaluate_wilson_loops!(temp3, loops, U, [temp1, temp2])
                add_U!(V, staplefactors[i], temp3)
                #add_U!(V,gparam.βs[i]/gparam.β,temp3)
            end
            mul!(temp1, U[μ], V) #U U*V
        end

        Traceless_antihermitian_add!(F[μ], factor, temp1)
        #add_U!(F[μ],factor,temp2)
    end

end
function add_force!(
    F::Array{T1,1},
    U::Array{T2,1},
    B::Array{T2,2},
    temps::Array{<:AbstractGaugefields{NC,Dim},1};
    plaqonly = false,
    staplefactors::Union{Array{<:Number,1},Nothing} = nothing,
    factor = 1,
) where {NC,Dim,T1<:TA_Gaugefields,T2<:AbstractGaugefields}
    @assert length(temps) >= 3 "length(temps) should be >= 3. But $(length(temps))"

    V = temps[3]
    temp1 = temps[1]
    temp2 = temps[2]

    for μ = 1:Dim
        if plaqonly
            construct_double_staple!(V, U, μ, temps[1:2])
            mul!(temp1, U[μ], V') #U U*V
        else
            clear_U!(V)
            for i = 1:gparam.numactions
                loops = gparam.staples[i][μ]
                evaluate_wilson_loops!(temp3, loops, U, B, [temp1, temp2])
                add_U!(V, staplefactors[i], temp3)
            end
            mul!(temp1, U[μ], V) #U U*V
        end

        Traceless_antihermitian_add!(F[μ], factor, temp1)
    end
end

#=
function add_force!(F::Array{T,1},U::Array{T,1},temps::Array{<: AbstractGaugefields{NC,Dim},1},factor = 1) where {NC,Dim,T <: AbstractGaugefields,GP}
    @assert length(temps) >= 3 "length(temps) should be >= 3. But $(length(temps))"
    clear_U!(F)
    V = temps[3]  
    temp1 = temps[1]
    temp2 = temps[2]    

    for μ=1:Dim
        construct_double_staple!(V,U,μ,temps[1:2])

        mul!(temp1,U[μ],V') #U U*V

        a = temp1[:,:,1,1,1,1]
        println(a'*a)

        Traceless_antihermitian!(temp2,temp1)
        #println(temp2[1,1,1,1,1,1])
        a = temp2[:,:,1,1,1,1]
        println(a'*a)
        error("a")
        add_U!(F[μ],factor,temp2)
    end
end
=#



function exptU!(
    uout::T,
    t::N,
    f::T1,
    temps::Array{T,1},
) where {N<:Number,T<:AbstractGaugefields,T1<:AbstractGaugefields} #uout = exp(t*u)
    error("expUt! is not implemented in type $(typeof(f)) uout: $(typeof(uout))")
end


function exptU!(
    uout::T,
    f::T1,
    temps::Array{T,1},
) where {T<:AbstractGaugefields,T1<:AbstractGaugefields}
    expU!(uout, 1, f, temps)
end

function exp_aF_U!(
    W::Array{<:AbstractGaugefields{NC,Dim},1},
    a::N,
    F::Array{T1,1},
    U::Array{T,1},
    temps::Array{T,1},
) where {NC,Dim,N<:Number,T<:AbstractGaugefields,T1<:AbstractGaugefields} #exp(a*F)*U
    @assert a != 0 "Δτ should not be zero in expF_U! function!"
    expU = temps[1]
    temp1 = temps[2]
    temp2 = temps[3]
    #clear_U!(temp1)
    #clear_U!(temp2)
    #clear_U!(expU)
    #clear_U!(W)

    for μ = 1:Dim
        exptU!(expU, a, F[μ], [temp1, temp2])
        mul!(W[μ], expU, U[μ])
    end

    set_wing_U!(W)
end


function staple_prime()
    loops_staple_prime = Array{Wilson_loop_set,2}(undef, 4, 4)
    for Dim = 1:4
        for μ = 1:Dim
            loops_staple_prime[Dim, μ] = make_plaq_staple_prime(μ, Dim)
        end
    end
    return loops_staple_prime
end
const loops_staple_prime_old = staple_prime()


function construct_double_staple!(
    staple::AbstractGaugefields{NC,Dim},
    U::Array{T,1},
    μ,
    temps::Array{<:AbstractGaugefields{NC,Dim},1},
) where {NC,Dim,T<:AbstractGaugefields}
    #println("mu = ",μ)
    #loops = loops_staple_prime_old[Dim,μ] #make_plaq_staple_prime(μ,Dim)
    #println("staple")
    #@time evaluate_wilson_loops!(staple,loops,U,temps)
    #println(staple[1,1,1,1,1,1])
    loops = loops_staple_prime[(Dim, μ)]
    evaluate_gaugelinks!(staple, loops, U, temps)
    #println(staple[1,1,1,1,1,1])
    #error("construct!!")
end
function construct_double_staple!(
    staple::AbstractGaugefields{NC,Dim},
    U::Array{T,1},
    B::Array{T,2},
    μ,
    temps::Array{<:AbstractGaugefields{NC,Dim},1},
) where {NC,Dim,T<:AbstractGaugefields}
    loops = loops_staple_prime[(Dim, μ)]
    evaluate_gaugelinks!(staple, loops, U, B, temps)
end


function construct_staple!(
    staple::AbstractGaugefields{NC,Dim},
    U::Array{T,1},
    μ,
    temp::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    U1U2 = temp
    firstterm = true

    for ν = 1:Dim
        if ν == μ
            continue
        end

        #=
                x+nu temp2
                .---------.
                I         I
          temp1 I         I
                I         I
                .         .
                x        x+mu
        =#
        U1 = U[ν]
        U2 = shift_U(U[μ], ν)
        #println(typeof(U1))
        mul!(U1U2, U1, U2)
        #error("test")

        U3 = shift_U(U[ν], μ)
        #mul!(staple,temp,Uμ')
        #  mul!(C, A, B, α, β) -> C, A B α + C β
        if firstterm
            β = 0
            firstterm = false
        else
            β = 1
        end
        mul!(staple, U1U2, U3', 1, β) #C = alpha*A*B + beta*C

        #println("staple ",staple[1,1,1,1,1,1])


        #mul!(staple,U0,Uν,Uμ')
    end
    set_wing_U!(staple)
end
function construct_staple!(
    staple::AbstractGaugefields{NC,Dim},
    U::Array{T,1},
    B::Array{T,2},
    μ,
    temp::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    U1U2 = temp
    firstterm = true

    for ν = 1:Dim
        if ν == μ
            continue
        end

        U1 = U[ν]
        # mul!(U1, U[ν], B[μ,ν]')
        if μ < ν
            mul!(U1, U[ν], B[μ,ν]')
        else
            mul!(U1, U[ν], B[μ,ν])
        end
        U2 = shift_U(U[μ], ν)
        mul!(U1U2, U1, U2)

        U3 = shift_U(U[ν], μ)
        if firstterm
            β = 0
            firstterm = false
        else
            β = 1
        end
        mul!(staple, U1U2, U3', 1, β)
    end
    set_wing_U!(staple)
end

function Base.size(U::T) where {T<:AbstractGaugefields}
    error("Base.size is not implemented in type $(typeof(U)) ")
end

function calculate_Polyakov_loop(
    U::Array{T,1},
    temp1::AbstractGaugefields{NC,Dim},
    temp2::AbstractGaugefields{NC,Dim},
) where {NC,Dim,T<:AbstractGaugefields}
    Uold = temp1
    Unew = temp2
    shift = zeros(Int64, Dim)

    μ = Dim
    _, _, NN... = size(U[1]) #NC,NC,NX,NY,NZ,NT 4D case
    lastaxis = NN[end]
    #println(lastaxis)

    substitute_U!(Uold, U[μ])
    for i = 2:lastaxis
        shift[μ] = i - 1
        U1 = shift_U(U[μ], Tuple(shift))
        mul_skiplastindex!(Unew, Uold, U1)
        Uold, Unew = Unew, Uold
    end

    set_wing_U!(Uold)
    poly = partial_tr(Uold, μ) / prod(NN[1:Dim-1])
    return poly

end





function mul_skiplastindex!(
    c::T,
    a::T1,
    b::T2,
) where {T<:AbstractGaugefields,T1<:Abstractfields,T2<:Abstractfields}
    error("mul_skiplastindex! is not implemented in type $(typeof(c)) ")
end


function Traceless_antihermitian(vin::T) where {T<:AbstractGaugefields}
    #error("Traceless_antihermitian is not implemented in type $(typeof(vin)) ")
    vout = deepcopy(vin)
    Traceless_antihermitian!(vout, vin)
    return vout
end

function Traceless_antihermitian_add!(U::T, factor, temp1) where {T<:AbstractGaugefields}
    error("Traceless_antihermitian_add! is not implemented in type $(typeof(U)) ")
end

function Traceless_antihermitian!(vout::T, vin::T) where {T<:AbstractGaugefields}
    error("Traceless_antihermitian! is not implemented in type $(typeof(vout)) ")
end

function Antihermitian!(vout::T, vin::T;factor=1) where {T<:AbstractGaugefields} #vout = vin - vin^+
    error("Antihermitian! is not implemented in type $(typeof(vout)) ")
end

function add_U!(c::T, a::T1) where {T<:AbstractGaugefields,T1<:Abstractfields}
    error("add_U! is not implemented in type $(typeof(c)) ")
end

function add_U!(
    c::Array{<:AbstractGaugefields{NC,Dim},1},
    α::N,
    a::Array{T1,1},
) where {NC,Dim,T1<:Abstractfields,N<:Number}
    for μ = 1:Dim
        add_U!(c[μ], α, a[μ])
    end
end

function add_U!(c::T, a::T1, iseven) where {T<:AbstractGaugefields,T1<:Abstractfields}
    error("add_U! is not implemented in type $(typeof(c)) ")
end

function add_U!(
    c::Array{<:AbstractGaugefields{NC,Dim},1},
    α::N,
    a::Array{T1,1},
    iseven,
) where {NC,Dim,T1<:Abstractfields,N<:Number}
    for μ = 1:Dim
        add_U!(c[μ], α, a[μ], iseven)
    end
end

function add_U!(
    c::T,
    α::N,
    a::T1,
) where {T<:AbstractGaugefields,T1<:Abstractfields,N<:Number}
    error("add_U! is not implemented in type $(typeof(c)) ")
end

function LinearAlgebra.mul!(
    c::T,
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {T<:AbstractGaugefields,T1<:Abstractfields,T2<:Abstractfields,Ta<:Number,Tb<:Number}
    error("LinearAlgebra.mul! is not implemented in type $(typeof(c)) ")
end

function LinearAlgebra.mul!(
    c::T,
    a::N,
    b::T2,
) where {T<:AbstractGaugefields,N<:Number,T2<:Abstractfields}
    error("LinearAlgebra.mul! is not implemented in type $(typeof(c)) ")
end

function partial_tr(a::T, μ) where {T<:Abstractfields}
    error("partial_tr is not implemented in type $(typeof(a)) ")
end

function LinearAlgebra.tr(a::T) where {T<:Abstractfields}
    error("LinearAlgebra.tr! is not implemented in type $(typeof(a)) ")
end

"""
    Tr(A*B)
"""
function LinearAlgebra.tr(a::T, b::T) where {T<:Abstractfields}
    error("LinearAlgebra.tr! is not implemented in type $(typeof(a)) ")
end

"""
M = (U*δ_prev) star (dexp(Q)/dQ)
Λ = TA(M)
"""
function construct_Λmatrix_forSTOUT!(Λ, δ_prev::T, Q, u::T) where {T<:AbstractGaugefields}
    error("construct_Λmatrix_forSTOUT! is not implemented in type $(typeof(u)) ")
end

const eps_Q = 1e-18

function calc_Λmatrix!(Λ, M, NC)
    #println("M= ", M)
    if NC == 1
        #Λ = -(M - M')
        @. Λ[:, :] = (M - M')
    elseif NC == 2
        #Λ = (1/2)*(M - M') - (1/(2NC))*tr(M - M')*I0_2
        @. Λ[:, :] = (1 / 2) * (M - M')
        trM = (1 / (2NC)) * (M[1, 1] - conj(M[1, 1]) + M[2, 2] - conj(M[2, 2]))#  tr(M - M')
        #trM = (1/(2NC))*tr(M - M')
        for i = 1:NC
            Λ[i, i] += -trM
        end
        #Λ = 2*Λ
    else
        @. Λ[:, :] = (1 / 2) * (M - M')
        trM = (1 / (2NC)) * tr(M - M')
        for i = 1:NC
            Λ[i, i] += -trM
        end
    end
    #display(Λ)
    #println("\t")
    #exit()
    return
    #        return Λ
end


function calc_Mmatrix!(
    Mn,
    δn_prev,
    Qn,
    Un,
    u::AbstractGaugefields{2,Dim},
    tempmatrices,
) where {Dim}
    Unδn = tempmatrices[1]
    B = tempmatrices[2]
    tmp_matrix1 = tempmatrices[3]
    NC = 2

    trQ2 = 0.0
    for i = 1:2
        for j = 1:2
            trQ2 += Qn[i, j] * Qn[j, i]
        end
    end

    if abs(trQ2) > eps_Q
        q = sqrt((-1 / 2) * trQ2)
        calc_Bmatrix!(B, q, Qn, NC)
        for i = 1:2
            for j = 1:2
                tmp_matrix1[j, i] = Un[j, i]
            end
        end

        mul!(Unδn, tmp_matrix1, δn_prev)
        trsum = 0.0im
        for i = 1:2
            for j = 1:2
                trsum += Unδn[i, j] * B[j, i]
            end
        end
        for i = 1:2
            for j = 1:2
                Mn[j, i] = (sin(q) / q) * Unδn[j, i] + trsum * Qn[j, i]
            end
        end
    end
end

function calc_Mmatrix!(
    Mn,
    δn_prev,
    Qn,
    Un,
    u::AbstractGaugefields{3,Dim},
    tempmatrices,
) where {Dim}
    Unδn = tempmatrices[1]
    tmp_matrix1 = tempmatrices[2]
    tmp_matrix2 = tempmatrices[3]
    #println("Qn ", Qn)
    trQ2 = 0.0
    for i = 1:3
        for j = 1:3
            trQ2 += Qn[i, j] * Qn[j, i]
        end
    end
    #println("tr", trQ2)

    if abs(trQ2) > eps_Q
        Qn ./= im
        #println("Qn b ",Qn)
        f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qn)
        for i = 1:3
            for j = 1:3
                tmp_matrix1[j, i] = Un[j, i]
            end
        end
        mul!(Unδn, tmp_matrix1, δn_prev)

        B1 = tmp_matrix1
        B1 .= 0
        B2 = tmp_matrix2
        B2 .= 0
        for i = 1:3
            B1[i, i] = b10
            B2[i, i] = b20
        end
        for j = 1:3
            for i = 1:3
                B1[i, j] += b11 * Qn[i, j]
                B2[i, j] += b21 * Qn[i, j]
                for k = 1:3
                    B1[i, j] += b12 * Qn[i, k] * Qn[k, j]
                    B2[i, j] += b22 * Qn[i, k] * Qn[k, j]
                end
            end
        end
        #println("coeff, ",(f0,f1,f2,b10,b11,b12,b20,b21,b22))
        #println("B1 ",B1)
        #println("B2 ",B2)

        trB1 = 0.0
        trB2 = 0.0
        for i = 1:3
            for j = 1:3
                trB1 += Unδn[i, j] * B1[j, i]
                trB2 += Unδn[i, j] * B2[j, i]
            end
        end

        for j = 1:3
            for i = 1:3
                Mn[i, j] = trB1 * Qn[i, j] + f1 * Unδn[i, j]
                for k = 1:3
                    Mn[i, j] +=
                        trB2 * Qn[i, k] * Qn[k, j] +
                        f2 * (Qn[i, k] * Unδn[k, j] + Unδn[i, k] * Qn[k, j])
                end
            end
        end
        Mn ./= im
    end
end

function calc_Mmatrix!(
    Mn,
    δn_prev,
    Qn,
    Un,
    u::AbstractGaugefields{NC,Dim},
    tempmatrices,
) where {NC,Dim}
    error("not supported yet")

    @assert NC > 3 "NC > 3 not NC = $NC"
    Unδn = tempmatrices[1]
    B = tempmatrices[2]
    tempmatrix = tempmatrices[3]



    trQ2 = 0.0
    for i = 1:NC
        for j = 1:NC
            trQ2 += Qn[i, j] * Qn[j, i]
        end
    end

    if abs(trQ2) > eps_Q
        e, v = eigen(Qn)
        mul!(Unδn, Un, δn_prev)
        #=
                A star dexp(Q)/dQ = \sum_{n=0}^{infty} \frac{1}{n!} 
                                        \sum_{k=0}^{n-1} i^{n-1-k}P^+ D^{n-1-k} P A P^+ D^k P i^k
                                = P^+ (\sum_{n=0}^{infty} (i^{n-1}/n!) sum_{k=0}^{n-1} D^{n-1-k} B D^k)  P
                                B = P A P+
            =#

        mul!(tempmatrix, Unδn, v)
        mul!(B, v', tempmatrix)




    end

end

function calc_Bmatrix!(B, q, Q, NC)
    @assert NC == 2 "NC should be 2! now $NC"
    mul!(B, cos(q) / q - sin(q) / q^2, Q)
    for i = 1:NC
        B[i, i] += -sin(q)
    end
    B .*= -1 / 2q
    #B[:,:] .= (cos(q)/q -sin(q)/q^2 )*Q

    #q = sqrt((-1/2)*tr(Q^2))
    #B = -(-sin(q)*I0_2 +(cos(q)/q -sin(q)/q^2 )*Q)*(1/2q)
end

function calc_coefficients_Q(Q)
    @assert size(Q) == (3, 3)
    c0 =
        Q[1, 1] * Q[2, 2] * Q[3, 3] +
        Q[1, 2] * Q[2, 3] * Q[3, 1] +
        Q[1, 3] * Q[2, 1] * Q[3, 2] - Q[1, 3] * Q[2, 2] * Q[3, 1] -
        Q[1, 2] * Q[2, 1] * Q[3, 3] - Q[1, 1] * Q[2, 3] * Q[3, 2]
    #@time cdet = det(Q)
    ##println(c0,"\t",cdet)
    #exit() 

    c1 = 0.0
    for i = 1:3
        for j = 1:3
            c1 += Q[i, j] * Q[j, i]
        end
    end
    c1 /= 2
    c0max = 2 * (c1 / 3)^(3 / 2)
    θ = acos(c0 / c0max)
    u = sqrt(c1 / 3) * cos(θ / 3)
    w = sqrt(c1) * sin(θ / 3)
    ξ0 = sin(w) / w
    ξ1 = cos(w) / w^2 - sin(w) / w^3

    emiu = exp(-im * u)
    e2iu = exp(2 * im * u)

    h0 = (u^2 - w^2) * e2iu + emiu * (8u^2 * cos(w) + 2 * im * u * (3u^2 + w^2) * ξ0)
    h1 = 2u * e2iu - emiu * (2u * cos(w) - im * (3u^2 - w^2) * ξ0)
    h2 = e2iu - emiu * (cos(w) + 3 * im * u * ξ0)

    denom = 9u^2 - w^2

    f0 = h0 / denom
    f1 = h1 / denom
    f2 = h2 / denom

    r10 =
        2 * (u + im * (u^2 - w^2)) * e2iu +
        2 *
        emiu *
        (4u * (2 - im * u) * cos(w) + im * (9u^2 + w^2 - im * u * (3u^2 + w^2)) * ξ0)
    r11 =
        2 * (1 + 2 * im * u) * e2iu +
        emiu * (-2 * (1 - im * u) * cos(w) + im * (6u + im * (w^2 - 3u^2)) * ξ0)
    r12 = 2 * im * e2iu + im * emiu * (cos(w) - 3 * (1 - im * u) * ξ0)
    r20 = -2 * e2iu + 2 * im * u * emiu * (cos(w) + (1 + 4 * im * u) * ξ0 + 3u^2 * ξ1)
    r21 = -im * emiu * (cos(w) + (1 + 2 * im * u) * ξ0 - 3 * u^2 * ξ1)
    r22 = emiu * (ξ0 - 3 * im * u * ξ1)
    b10 = (2 * u * r10 + (3u^2 - w^2) * r20 - 2 * (15u^2 + w^2) * f0) / (2 * denom^2)

    b11 = (2 * u * r11 + (3u^2 - w^2) * r21 - 2 * (15u^2 + w^2) * f1) / (2 * denom^2)
    b12 = (2 * u * r12 + (3u^2 - w^2) * r22 - 2 * (15u^2 + w^2) * f2) / (2 * denom^2)
    b20 = (r10 - 3 * u * r20 - 24 * u * f0) / (2 * denom^2)
    b21 = (r11 - 3 * u * r21 - 24 * u * f1) / (2 * denom^2)
    b22 = (r12 - 3 * u * r22 - 24 * u * f2) / (2 * denom^2)

    return f0, f1, f2, b10, b11, b12, b20, b21, b22
end


function staggered_phase(μ, iii...)
    error("staggered_phase is not implemented")
end



function normalize!(u::Array{ComplexF64,2})
    NC, _ = size(u)
    if NC == 2
        normalize2!(u)
    elseif NC == 3
        normalize3!(u)
    else
        normalizeN!(u)
    end
    return
end

function normalize2!(u)
    α = u[1, 1]
    β = u[2, 1]
    detU = abs(α)^2 + abs(β)^2
    u[1, 1] = α / detU
    u[2, 1] = β / detU
    u[1, 2] = -conj(β) / detU
    u[2, 2] = conj(α) / detU
end

function normalizeN!(u)
    gramschmidt!(u)
end

function normalize3!(u)
    w1 = 0
    w2 = 0
    for ic = 1:3
        w1 += u[2, ic] * conj(u[1, ic])
        w2 += u[1, ic] * conj(u[1, ic])
    end
    zerock2 = w2
    if zerock2 == 0
        println("w2 is zero  !!  (in normlz)")
        println("u[1,1),u[1,2),u[1,3) : ", u[1, 1], "\t", u[1, 2], "\t", u[1, 3])
    end

    w1 = -w1 / w2

    x4 = (u[2, 1]) + w1 * u[1, 1]
    x5 = (u[2, 2]) + w1 * u[1, 2]
    x6 = (u[2, 3]) + w1 * u[1, 3]

    w3 = x4 * conj(x4) + x5 * conj(x5) + x6 * conj(x6)

    zerock3 = w3
    if zerock3 == 0
        println("w3 is zero  !!  (in normlz)")
        println("x4, x5, x6 : $x4, $x5, $x6")
        exit()
    end

    u[2, 1] = x4
    u[2, 2] = x5
    u[2, 3] = x6

    w3 = 1 / sqrt(w3)
    w2 = 1 / sqrt(w2)

    u[1, 1] = u[1, 1] * w2
    u[1, 2] = u[1, 2] * w2
    u[1, 3] = u[1, 3] * w2
    u[2, 1] = u[2, 1] * w3
    u[2, 2] = u[2, 2] * w3
    u[2, 3] = u[2, 3] * w3

    if zerock2 * zerock3 == 0
        println("!! devided by zero !! (in normalize)")
        println("w2 or w3 in normlz is zero !!")
        println("w2, w3 : $w2, $w3   ")
        exit()
    end

    m3complv3!(u)
end


function m3complv3!(a)
    aa = zeros(Float64, 18)
    aa[1] = real(a[1, 1])
    aa[2] = imag(a[1, 1])
    aa[3] = real(a[1, 2])
    aa[4] = imag(a[1, 2])
    aa[5] = real(a[1, 3])
    aa[6] = imag(a[1, 3])
    aa[7] = real(a[2, 1])
    aa[8] = imag(a[2, 1])
    aa[9] = real(a[2, 2])
    aa[10] = imag(a[2, 2])
    aa[11] = real(a[2, 3])
    aa[12] = imag(a[2, 3])

    aa[13] = aa[3] * aa[11] - aa[4] * aa[12] - aa[5] * aa[9] + aa[6] * aa[10]
    aa[14] = aa[5] * aa[10] + aa[6] * aa[9] - aa[3] * aa[12] - aa[4] * aa[11]
    aa[15] = aa[5] * aa[7] - aa[6] * aa[8] - aa[1] * aa[11] + aa[2] * aa[12]
    aa[16] = aa[1] * aa[12] + aa[2] * aa[11] - aa[5] * aa[8] - aa[6] * aa[7]
    aa[17] = aa[1] * aa[9] - aa[2] * aa[10] - aa[3] * aa[7] + aa[4] * aa[8]
    aa[18] = aa[3] * aa[8] + aa[4] * aa[7] - aa[1] * aa[10] - aa[2] * aa[9]

    a[3, 1] = aa[13] + im * aa[14]
    a[3, 2] = aa[15] + im * aa[16]
    a[3, 3] = aa[17] + im * aa[18]
    return
end

function gramschmidt!(v)
    n = size(v)[1]
    for i = 1:n
        for j = 1:i-1
            v[:, i] = v[:, i] - v[:, j]' * v[:, i] * v[:, j]
        end
        v[:, i] = v[:, i] / norm(v[:, i])
    end
end


function gramschmidt_special!(v)
    n = size(v)[1]
    #vdet = det(v)


    vnorm1 = norm(v[:, 1])
    for i = 1:n
        #vnorm[i] = norm(v[:,i])
        for j = 1:i-1
            v[:, i] = v[:, i] - v[:, j]' * v[:, i] * v[:, j]
        end
        v[:, i] = v[:, i] / norm(v[:, i])
    end
    for i = 1:n
        #v[:,i] = v[:,i]*vnorm[i]
        v[:, i] = v[:, i] * vnorm1
    end
end

function make_cloverloops(;Dim=4)
    cloverloops = Vector{Vector{Wilsonline{Dim}}}(undef,6)
    μν = 0
    for μ=1:3
        for ν=μ+1:4
            μν += 1
            if μν > 6
                error("μν > 6 ?")
            end
            cloverloops[μν] = make_cloverloops(μ,ν,Dim=Dim)
        end
    end
    return cloverloops 
end

const cloverloops_4D = make_cloverloops()

"""
    Clover terms.
    If you multiply 0.125*kappa*Clover_coefficients, this becomes the Wilson Clover terms.
"""
function make_Cloverloopterms(U,temps)
    CloverFμν = Array{eltype(U)}(undef,6)
    for μν=1:6
        CloverFμν[μν] = similar(U[1])
    end
    make_Cloverloopterms!(CloverFμν,U,temps)
    return CloverFμν
end

function make_Cloverloopterms!(CloverFμν,U,temps)
    #println(length(temps))
    @assert length(temps) > 2 "length of temp Gaugefields should be larger than 1"
    xout = temps[end]
    μν = 0
    for μ=1:3
        for ν=μ+1:4
            μν += 1
            if μν > 6
                error("μν > 6 ?")
            end
            wclover = cloverloops_4D[μν]
            evaluate_gaugelinks!(xout,wclover,U,temps)
            Antihermitian!(CloverFμν[μν],xout)
        end
    end
end

"""
    b = (lambda_k/2)*a
    lambda_k : GellMann matrices. k=1, 8 
"""
function lambda_k_mul!(a::T1, b::T2,k,generator) where {T1<:Abstractfields,T2<:Abstractfields}
    error("lambda_k_mul! is not implemented in type $(typeof(a)) and $(typeof(b))")
end

end
