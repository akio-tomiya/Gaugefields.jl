abstract type Adjoint_rep_Gaugefields_4D{NC,NumofBasis} <:
              Adjoint_rep_Gaugefields{NC,4,NumofBasis} end

include("./Adjoint_rep_gaugefields_4D_wing.jl")


function Base.size(U::Adjoint_rep_Gaugefields_4D{NC,NumofBasis}) where {NC,NumofBasis}
    return NumofBasis, NumofBasis, U.NX, U.NY, U.NZ, U.NT
end
