module Gaugefields

using Requires
include("./output/verboseprint.jl")
include("./SUN_generator.jl")
include("./autostaples/wilsonloops.jl")
include("./AbstractGaugefields.jl") 
include("./output/io.jl")
include("./output/ildg_format.jl")
include("./autostaples/Loops.jl")
include("./smearing/Abstractsmearing.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin   
        import .AbstractGaugefields_module:identityGaugefields_4D_wing_mpi,
                Gaugefields_4D_wing_mpi,calc_rank_and_indices,barrier,comm,
                setvalue!
    end
end

# Write your package code here.
import .AbstractGaugefields_module:AbstractGaugefields,IdentityGauges,RandomGauges,Oneinstanton,calculate_Plaquette
import .ILDG_format:ILDG,load_gaugefield!

export IdentityGauges,RandomGauges,Oneinstanton,calculate_Plaquette
export ILDG,load_gaugefield!

end
