# How to implement new gauge fields

It is easy to implement new gauge fields with different internal structures or different parallel computations. 

## AbstractGaugefields and interfaces
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

## matrix-field matrix-field product

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

## Several ways to treat periodic boundary. 
There are several ways to treat periodic boundary. 
Now, there are two kinds of methods, halo updates and direct-shift method. 

- halo updates: wing-buffer (so-called halo) type implementations for gauge fields. In this type, there are additional halo sites. If ```NDW > 0``` is set, this update is used. 
- direct-shift method: In ```shift_U!```  function, the data is copied. If ```NDW = 0``` is set, this update is used. 


Therefore, the important functions are 

- ```set_wing_U!``` : This is used in halo updates, but not used in direct-shift method (returns nothing). 
- ```shift_U!``` : In this function, the data is copied in direct-shift method. This is the lazy evaluation in halo updates. 



## halo updates
There is a wing-buffer (so-called halo) type implementations for gauge fields. In this type, there are additional halo sites. 
`set_wing_U!` function is used for updating halo sites. 
If you do not want to use halo type implementations for gauge fields, you can write `set_wing_U!(U::Yourtype) = nothing`. 

### examples

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