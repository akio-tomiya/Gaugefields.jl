#=
module Gaugefields_4D_module
    using LinearAlgebra
    import ..AbstractGaugefields_module:AbstractGaugefields,Shifted_Gaugefields,shift_U,
                        Adjoint_Gaugefields,set_wing_U!,Abstractfields,construct_staple!,clear_U!,
                        calculate_Plaquette
    import Base
    =#

abstract type Gaugefields_4D{NC} <: AbstractGaugefields{NC,4} end


include("./wing/gaugefields_4D_wing.jl")
include("./wing/gaugefields_4D_wing_Bfields.jl")
include("./nowing/gaugefields_4D_nowing.jl")
include("./nowing/gaugefields_4D_nowing_Bfields.jl")

function __init__()
    #    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin

    #    end

    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("./kernelfunctions/gaugefields_4D_cudakernels.jl")
        include("./kernelfunctions/TA_gaugefields_4D_cudakernels.jl")
        include("./kernelfunctions/linearalgebra_mul_NC_cuda.jl")
        include("./kernelfunctions/linearalgebra_mul_NC3_cuda.jl")
    end

    #    @require JACC = "0979c8fe-16a4-4796-9b82-89a9f10403ea" begin

    #    end
end

using MPI
using JACC
include("./mpi/gaugefields_4D_mpi.jl")
include("./mpi/gaugefields_4D_mpi_Bfields.jl")
include("./mpi/gaugefields_4D_mpi_nowing.jl")
include("./mpi/gaugefields_4D_mpi_nowing_Bfields.jl")





include("./kernelfunctions/gaugefields_4D_kernels.jl")
include("./gaugefields_4D_accelerator.jl")

include("./kernelfunctions/gaugefields_4D_jacc.jl")
include("./kernelfunctions/linearalgebra_mul_NC_jacc.jl")
include("./kernelfunctions/linearalgebra_mul_NC3_jacc.jl")

include("./kernelfunctions/gaugefields_4D_jacckernels.jl")



include("mpi_jacc/gaugefields_4D_MPILattice.jl")


function Base.size(U::Gaugefields_4D{NC}) where {NC}
    return NC, NC, U.NX, U.NY, U.NZ, U.NT
end




function clear_U!(U::Array{T,1}) where {T<:Gaugefields_4D}
    for μ = 1:4
        clear_U!(U[μ])
    end

end



#=
function calculate_Plaquet(U::Array{T,1}) where T <: Gaugefields_4D
    error("calculate_Plaquet is not implemented in type $(typeof(U)) ")
end
=#



#end
