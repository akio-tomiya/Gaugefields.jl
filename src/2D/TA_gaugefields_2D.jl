abstract type TA_Gaugefields_2D{NC} <: TA_Gaugefields{NC,2} end

include("./TA_gaugefields_2D_serial.jl")

function TA_Gaugefields(NC, NX, NT; mpi = false)
    if mpi
        return TA_Gaugefields_2D_mpi(NC, NX, NT)
        #error("mpi = $mpi is not supoorted")
    else
        return TA_Gaugefields_2D_serial(NC, NX, NT)
    end
end

function clear_U!(U::Array{T,1}) where {T<:TA_Gaugefields_2D}
    for μ = 1:2
        clear_U!(U[μ])
    end
end
