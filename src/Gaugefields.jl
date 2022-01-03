module Gaugefields

using Requires
include("./output/verboseprint.jl")
include("./SUN_generator.jl")
include("./autostaples/wilsonloops.jl")
include("./AbstractGaugefields.jl") 
include("./output/io.jl")
include("./output/ildg_format.jl")
include("./output/bridge_format.jl")
include("./autostaples/Loops.jl")
include("./smearing/Abstractsmearing.jl")
include("./heatbath/heatbathmodule.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin   
        import .AbstractGaugefields_module:identityGaugefields_4D_wing_mpi,
                Gaugefields_4D_wing_mpi,calc_rank_and_indices,barrier,comm,
                setvalue!
    end
end

# Write your package code here.
import .AbstractGaugefields_module:AbstractGaugefields,IdentityGauges,RandomGauges,Oneinstanton,calculate_Plaquette,
                                    calculate_Polyakov_loop,map_U!,evaluate_gaugelinks_evenodd!,normalize!,normalize3!,normalizeN!,
                                    shift_U
import .ILDG_format:ILDG,load_gaugefield!,save_binarydata
import .heatbath_module:SU2update_KP!,SUNupdate_matrix!,SU3update_matrix!
import .Bridge_format:save_textdata,load_BridgeText!
import Wilsonloop:loops_staple

export IdentityGauges,RandomGauges,Oneinstanton,calculate_Plaquette,calculate_Polyakov_loop
export ILDG,load_gaugefield!,save_binarydata
export SU2update_KP!,SUNupdate_matrix!,SU3update_matrix!
export map_U!
export evaluate_gaugelinks_evenodd!,normalize!,normalize3!,normalizeN!
export loops_staple
export save_textdata,load_BridgeText!
export shift_U


end
