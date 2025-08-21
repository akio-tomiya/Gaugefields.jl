using LinearAlgebra
function LinearAlgebra.mul!(C::Lattice, A::Lattice, B::Lattice)
    error("Matrix multiplication is not implemented for Lattice types $(typeof(A)) $(typeof(B)) $(typeof(C)).")
end

function expt!(C::Lattice, A::Lattice, t::S=one(S)) where {S<:Number}
    error("Matrix exponentiation is not implemented for Lattice types $(typeof(A)).")
end


export expt!

include("matrixexp.jl")
include("linearalgebra_1D.jl")
include("linearalgebra_4D.jl")


