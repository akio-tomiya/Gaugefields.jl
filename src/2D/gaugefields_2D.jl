abstract type Gaugefields_2D{NC} <: AbstractGaugefields{NC,2} end


include("./gaugefields_2D_wing.jl")
include("./gaugefields_2D_nowing.jl")
include("./Isingfields_2D.jl")

function Base.size(U::Gaugefields_2D{NC}) where {NC}
    return NC, NC, U.NX, U.NT
end


function clear_U!(U::Array{T,1}) where {T<:Gaugefields_2D}
    for μ = 1:2
        clear_U!(U[μ])
    end

end
