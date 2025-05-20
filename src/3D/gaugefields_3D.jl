abstract type Gaugefields_3D{NC} <: AbstractGaugefields{NC,3} end


include("./gaugefields_3D_nowing.jl")

function Base.size(U::Gaugefields_3D{NC}) where {NC}
    return NC, NC, U.NX, U.NY, U.NT
end


function clear_U!(U::Array{T,1}) where {T<:Gaugefields_3D}
    for μ = 1:3
        clear_U!(U[μ])
    end

end
