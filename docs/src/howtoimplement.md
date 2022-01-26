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

