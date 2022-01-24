module Gaugefields

using Requires
include("./output/verboseprint_mpi.jl")
#include("./output/verboseprint.jl")
include("./SUN_generator.jl")
include("./autostaples/wilsonloops.jl")
include("./AbstractGaugefields.jl") 
include("./output/io.jl")
include("./output/ildg_format.jl")
include("./output/bridge_format.jl")
include("./autostaples/Loops.jl")
include("./smearing/Abstractsmearing.jl")
include("./heatbath/heatbathmodule.jl")
include("./action/GaugeActions.jl")

function __init__()
    @require MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195" begin   
        import .AbstractGaugefields_module:identityGaugefields_4D_wing_mpi,
                Gaugefields_4D_wing_mpi,calc_rank_and_indices,barrier,comm,
                setvalue!
    end
end

# Write your package code here.
import .Verboseprint_mpi:Verbose_print,println_verbose_level1,println_verbose_level2,println_verbose_level3
import .AbstractGaugefields_module:AbstractGaugefields,IdentityGauges,RandomGauges,Oneinstanton,calculate_Plaquette,
                                    calculate_Polyakov_loop,map_U!,evaluate_gaugelinks_evenodd!,normalize!,normalize3!,normalizeN!,
                                    shift_U,evaluate_gaugelinks!,
                                    TA_Gaugefields,Abstractfields,Staggered_Gaugefields,staggered_U,
                                    clear_U!,set_wing_U!,Traceless_antihermitian!,initialize_TA_Gaugefields,
                                    substitute_U!,Traceless_antihermitian!,Traceless_antihermitian

#=
                                    import .AbstractGaugefields_module:AbstractGaugefields,identitymatrix,Abstractfields,
                                    shift_U,construct_staple!,set_wing_U!,
                                    calculate_Plaquette,substitute_U!,calculate_Polyakov_loop,construct_gauges,
                                    Gaugefields_4D_wing,
                                    identityGaugefields_4D_wing,
                                    add_force!,exp_aF_U!,clear_U!,add_U!,exptU!,
                                    Traceless_antihermitian!,Traceless_antihermitian,Generator,
                                    Staggered_Gaugefields,staggered_U,
                                    Traceless_antihermitian_add!,
                                    IdentityGauges,RandomGauges,Oneinstanton,
                                    construct_Λmatrix_forSTOUT!,
                                    evaluate_gaugelinks_evenodd!,
                                    map_U!   
                                    =#     
import .Loops_module:Loops,evaluate_loops,calc_large_wilson_loop!,evaluate_loops!                                       
import .Wilsonloops_module:Wilson_loop_set,make_staples,Wilson_loop_set,
                                    make_cloverloops,Tensor_derivative_set, make_loops,
                                    make_plaq_staple,make_links,make_plaq,
                                    make_loopforactions,make_plaqloops,make_rectloops,make_polyakovloops,
                                    make_plaq_staple_prime,
                                    calc_coordinate,make_plaq_staple_prime,calc_shift,
                                    Tensor_wilson_lines_set,Tensor_wilson_lines,Tensor_derivative_set,
                                    get_leftstartposition,get_rightstartposition,Wilson_loop,calc_loopset_μν_name,
                                    make_originalactions_fromloops     
import .Abstractsmearing_module:Abstractsmearing,Nosmearing,Stoutsmearing,calc_smearedU,
                                construct_smearing,Gradientflow,get_tempG,flow!,get_eps,back_prop,CovNeuralnet#gradientflow!                 
import .ILDG_format:ILDG,load_gaugefield!,save_binarydata
import .heatbath_module:SU2update_KP!,SUNupdate_matrix!,SU3update_matrix!,heatbath!,Heatbath
import .Bridge_format:save_textdata,load_BridgeText!
import Wilsonloop:loops_staple
import .Abstractsmearing_module:Gradientflow,flow!,STOUT_Layer,CovNeuralnet,calc_smearedU, construct_smearing,set_parameters
import .SUN_generator:Generator
#import .Verbose_print:Verbose_level,Verbose_3,Verbose_2,Verbose_1,println_verbose3,println_verbose2,println_verbose1,
#    print_verbose1,print_verbose2,print_verbose3

import .ILDG_format:ILDG,load_gaugefield,load_gaugefield!,save_binarydata
import .IOmodule:saveU,loadU,loadU!    


import .AbstractGaugefields_module:AbstractGaugefields,identitymatrix,Abstractfields,
            shift_U,construct_staple!,set_wing_U!,
            calculate_Plaquette,substitute_U!,calculate_Polyakov_loop,construct_gauges,
            Gaugefields_4D_wing,
            identityGaugefields_4D_wing,
            add_force!,exp_aF_U!,clear_U!,add_U!,exptU!,
            Traceless_antihermitian!,Traceless_antihermitian,Generator,
            Staggered_Gaugefields,staggered_U,
            Traceless_antihermitian_add!,
            IdentityGauges,RandomGauges,Oneinstanton,Initialize_4DGaugefields,
            construct_Λmatrix_forSTOUT!,
            evaluate_gaugelinks_evenodd!,
            map_U!,initialize_TA_Gaugefields,gauss_distribution!,Initialize_Gaugefields      
import Wilsonloop:make_loops_fromname
import .GaugeAction_module:GaugeAction,evaluate_GaugeAction_untraced!,evaluate_GaugeAction_untraced,calc_dSdUμ,calc_dSdUμ!,get_temporary_gaugefields,evaluate_GaugeAction

export IdentityGauges,RandomGauges,Oneinstanton,calculate_Plaquette,calculate_Polyakov_loop
export ILDG,load_gaugefield!,save_binarydata
export SU2update_KP!,SUNupdate_matrix!,SU3update_matrix!
export map_U!
export evaluate_gaugelinks_evenodd!,normalize!,normalize3!,normalizeN!
export loops_staple
export save_textdata,load_BridgeText!
export shift_U,evaluate_gaugelinks!,Gradientflow,flow!
export heatbath!,Heatbath
export STOUT_Layer,CovNeuralnet,calc_smearedU,make_loops_fromname
export GaugeAction,evaluate_GaugeAction_untraced!,evaluate_GaugeAction_untraced,calc_dSdUμ,calc_dSdUμ!
export initialize_TA_Gaugefields,gauss_distribution!
export exptU!,get_temporary_gaugefields,Traceless_antihermitian_add!,evaluate_GaugeAction,substitute_U!,set_wing_U!,Traceless_antihermitian!
export Initialize_Gaugefields,back_prop
export Initialize_4DGaugefields
export set_parameters






end
