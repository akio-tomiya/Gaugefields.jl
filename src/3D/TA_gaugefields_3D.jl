abstract type TA_Gaugefields_3D{NC} <: TA_Gaugefields{NC,3} end

include("./TA_gaugefields_3D_serial.jl")

function TA_Gaugefields(NC, NX, NY, NT; mpi=false)
    if mpi
        error("mpi = $mpi is not supoorted")
        return TA_Gaugefields_3D_mpi(NC, NX, NY, NT)
        #error("mpi = $mpi is not supoorted")
    else
        return TA_Gaugefields_3D_serial(NC, NX, NY, NT)
    end
end

function clear_U!(U::Array{T,1}) where {T<:TA_Gaugefields_3D}
    for μ = 1:3
        clear_U!(U[μ])
    end
end
