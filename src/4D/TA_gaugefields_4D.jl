abstract type TA_Gaugefields_4D{NC} <: TA_Gaugefields{NC,4} end

include("./TA_gaugefields_4D_serial.jl")

include("./TA_gaugefields_4D_accelerator.jl")

include("./mpi/TA_gaugefields_4D_mpi.jl")

include("./mpi_jacc/TA_gaugefields_4D_MPILattice.jl")
#=
function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin   
        include("./TA_gaugefields_4D_mpi.jl")    
    end
end

=#


function TA_Gaugefields(NC, NX, NY, NZ, NT; mpi=false, cuda=false, blocks=nothing, isMPILattice=false)
    if isMPILattice
        error("not implemented")
    else
        if mpi
            return TA_Gaugefields_4D_mpi(NC, NX, NY, NZ, NT)
            #error("mpi = $mpi is not supoorted")
        else
            if cuda
                return TA_Gaugefields_4D_accelerator(NC, NX, NY, NZ, NT, blocks)
            else
                return TA_Gaugefields_4D_serial(NC, NX, NY, NZ, NT)
            end
        end
    end
end

function clear_U!(U::Array{T,1}) where {T<:TA_Gaugefields_4D}
    for μ = 1:4
        clear_U!(U[μ])
    end
end

include("./kernelfunctions/TA_gaugefields_4D_jacckernels.jl")
