module Abstractsmearing_module
using LinearAlgebra
import ..Wilsonloops_module:
    Wilson_loop_set,
    make_staples,
    Wilson_loop_set,
    make_cloverloops,
    Tensor_derivative_set,
    make_loops
import ..AbstractGaugefields_module:
    AbstractGaugefields,
    Abstractfields,
    initialize_TA_Gaugefields,
    add_force!,
    exp_aF_U!,
    clear_U!,
    add_U!,
    evaluate_wilson_loops!,
    exptU!,
    Traceless_antihermitian_add!,
    set_wing_U!,
    Traceless_antihermitian,
    evaluate_gaugelinks!,
    construct_Λmatrix_forSTOUT!,
    Traceless_antihermitian!,
    shift_U
import Wilsonloop:
    Wilsonline,
    DwDU,
    make_loopforactions,
    make_Cμ,
    derive_U,
    derive_Udag,
    get_leftlinks,
    get_rightlinks
import ..Verboseprint_mpi:
    Verbose_print, println_verbose_level1, println_verbose_level2, println_verbose_level3

import ..Temporalfields_module: Temporalfields, unused!, get_temp, set_reusemode!
#import ..GaugeAction_module:GaugeAction

using Requires


function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("./kernelfunctions/stout_cudakernels.jl")
    end
end



abstract type Abstractsmearing end

struct Nosmearing <: Abstractsmearing end

abstract type CovLayer{Dim} end

struct CovNeuralnet{Dim,T,TA} <: Abstractsmearing
    #numlayers::Int64
    layers::Vector{CovLayer{Dim}}
    _temp_U::Temporalfields{T}
    _temp_UA::Temporalfields{TA}
end

function get_numlayers(c::CovNeuralnet)
    return length(c.layers)
end



function CovNeuralnet(; Dim=4)
    @error "CovNeuralnet() is not supported now. Use CovNeuralnet(U) to determine the type of U."
    layers = CovLayer{Dim}[]
    return CovNeuralnet{Dim}(layers)
end

function CovNeuralnet(U::Vector{<:AbstractGaugefields{NC}}
    ; Dim=4) where {NC}
    return CovNeuralnet(U[1]; Dim)
end

function CovNeuralnet(U::AbstractGaugefields; Dim=4, numtemps=16)
    layers = CovLayer{Dim}[]
    num = numtemps
    _temp_U = Temporalfields(U; num=num)
    UTA = initialize_TA_Gaugefields(U)
    _temp_UA = Temporalfields(UTA; num=1)

    return CovNeuralnet{Dim,typeof(U),typeof(UTA)}(layers, _temp_U, _temp_UA)
end


function Base.push!(c::CovNeuralnet, layer::CovLayer)
    #c.numlayers += 1
    push!(c.layers, layer)
end

function Base.getindex(c::CovNeuralnet, i)
    return c.layers[i]
end

function Base.show(c::CovNeuralnet)
    numlayers = get_numlayers(c)
    println("num. of layers: ", numlayers)
    for i = 1:numlayers
        if i == 1
            string = "st"
        elseif i == 2
            string = "nd"
        elseif i == 3
            string = "rd"
        else
            string = "th"
        end
        println("- $i-$string layer: $(get_name(c[i]))")
        show(c[i])
    end
end

function get_name(s::CovLayer)
    error("layer $s has no name")
end


include("./stout.jl")
include("./stout_fast.jl")
include("./stout_fast_accelerator.jl")
#include("./stout_smearing.jl")
include("./CASK_smearing.jl")

#include("./stout_b.jl")
#include("./stout_smearing.jl")
#include("./gradientflow.jl")

#
#function set_parameters!(s::CovNeuralnet, i, v)
#    set_parameters!(s[i], v)
#end

#function get_parameters(s::CovNeuralnet, i)
#    get_parameters(s[i])
#end

function set_parameters!(s::CovNeuralnet, params)
    numlayers = get_numlayers(s)
    #println("num layers")
    start_index = 1
    for i = 1:numlayers
        numparam_i = get_numparameters(s[i])
        end_index = start_index + numparam_i - 1
        params_i = view(params, start_index:end_index)
        start_index = end_index + 1
        set_parameters!(s[i], params_i)
        #println(get_parameters(s[i]))
    end
end

function get_parameters(s::CovNeuralnet)
    numlayers = get_numlayers(s)
    params = Float64[]
    #params = Vector{Vector{Float64}}[]
    for i = 1:numlayers
        append!(params, get_parameters(s[i]))
        #push!(params, get_parameters(s[i]))
    end
    return params
end

function get_parameter_derivatives(s::CovNeuralnet)
    numlayers = get_numlayers(s)
    parameter_derivatives = Float64[]
    for i = 1:numlayers
        append!(parameter_derivatives, get_parameter_derivatives(s[i]))
    end
    return parameter_derivatives
end

function get_numparameters(s::CovNeuralnet)
    numlayers = get_numlayers(s)
    num = 0
    for i = 1:numlayers
        num += get_numparameters(s[i])
    end
    return num
end

function zero_grad!(s::CovNeuralnet)
    numlayers = get_numlayers(s)
    for i = 1:numlayers
        zero_grad!(s[i])
    end
end





function construct_smearing(smearingparameters, loops_list, L, coefficients, numlayers)
    if smearingparameters == "nothing"
        smearing = Nosmearing()
    elseif smearingparameters == "stout"
        @assert loops_list != nothing "loops should be put if you want to use smearing schemes"
        loops = make_loops(loops_list, L)

        @assert coefficients != nothing "coefficients should be put if you want to use smearing schemes"
        println("stout smearing will be used")
        if numlayers == 1
            input_coefficients = [coefficients]
        else
            input_coefficients = coefficients
        end

        smearing = Stoutsmearing(loops, input_coefficients)
    elseif smearingparameters == "covnet_stout"
        if numlayers == 1
            input_coefficients = [coefficients]
        else
            input_coefficients = coefficients
        end
        println("covnet verion of the stout smearing will be used")
        smearing = CovNeuralnet_STOUT(loops_list, input_coefficients, L; Dim=length(L))
    else
        error("smearing = $smearing is not supported")
    end
    return smearing
end

function calc_smearedU(
    Uin::Array{T,1},
    smearing;
    calcdSdU=false,
    temps=nothing,
) where {T<:AbstractGaugefields}
    if smearing != nothing && typeof(smearing) != Nosmearing
        #println(smearing)
        #println(typeof(smearing))
        Uout_multi = apply_smearing_U(Uin, smearing)
        U = Uout_multi[end]

        #=
        if typeof(smearing) <: SmearingParam_single
            Uout_multi = nothing
            U = apply_smearing_U(Uin,smearing)
        elseif typeof(smearing) <: SmearingParam_multi
            Uout_multi = apply_smearing_U(Uin,smearing)
            U = Uout_multi[end]
        else
            error("something is wrong in calc_smearingU")
        end
        =#
        set_wing_U!(U)  #we want to remove this.
        if calcdSdU
            dSdU = [temps[end-3], temps[end-2], temps[end-1], temps[end]]
        else
            dSdU = nothing
        end
    else
        dSdU = nothing
        Uout_multi = nothing
        U = Uin
    end
    return U, Uout_multi, dSdU
end

function apply_smearing_U(Uin::Array{T,1}, smearing) where {T<:Abstractfields}
    error(
        "apply_smearing_U is not implemented in type $(typeof(Uin)) and smearing type $(typeof(smearing))",
    )
end

function apply_smearing_U(
    Uin::Array{T,1},
    smearing::CovNeuralnet{Dim},
) where {T<:Abstractfields,Dim}
    unused!(smearing._temp_U)

    temps, its_temp1 = get_temp(smearing._temp_U, 4)

    #temp1 = similar(Uin[1])
    #temp2 = similar(Uin[1])
    #temp3 = similar(Uin[1])
    #temp4 = similar(Uin[1])
    F0s, its_F0 = get_temp(smearing._temp_UA, 1)
    #F0 = initialize_TA_Gaugefields(Uin[1])
    numlayers = get_numlayers(smearing)
    Uout_multi = Array{typeof(Uin),1}(undef, numlayers)
    temps_multi, its_temp_multi = get_temp(smearing._temp_U, numlayers * Dim)
    for i = 1:numlayers
        Uout_multi[i] = temps_multi[(i-1)*Dim+1:(i-1)*Dim+Dim]#similar(Uin)
    end
    #apply_neuralnet!(Uout_multi, smearing, Uin, [temp1, temp2, temp3, temp4], [F0])
    #apply_neuralnet!(Uout_multi, smearing, Uin, temps, [F0])
    apply_neuralnet!(Uout_multi, smearing, Uin, temps, F0s)

    unused!(smearing._temp_U, its_temp1)
    unused!(smearing._temp_UA, its_F0)

    unused!(smearing._temp_U, its_temp_multi)

    return Uout_multi

    error(
        "apply_smearing_U is not implemented in type $(typeof(Uin)) and smearing type $(typeof(smearing))",
    )
end

function back_prop(δL, net::CovNeuralnet{Dim}, Uout_multi, Uin) where {Dim}

    δ_current = deepcopy(δL)
    back_prop!(δ_current, δL, net, Uout_multi, Uin)
    return δ_current

    #=
    temps = similar(Uout_multi[1])
    temps_F1 = initialize_TA_Gaugefields(temps[1])
    tempf = [temps_F1]

    layer = net.layers[get_numlayers(net)]
    δ_prev, its_prev = get_temp(net._temp_U, Dim)
    δ_current = deepcopy(δL)
    #δ_current, its_current = get_temp(net._temp_U, Dim)
    #substitute_U!(δ_current, δL)
    #similar(δL)
    #δ_current = deepcopy(δL)
    set_wing_U!(δ_current)

    for i = get_numlayers(net):-1:2
        layer = net.layers[i]
        layer_pullback!(δ_prev, δ_current, layer, Uout_multi[i-1], temps, tempf)
        δ_current, δ_prev = δ_prev, δ_current
        #set_wing_U!(δ_current)
    end
    layer = net.layers[1]
    layer_pullback!(δ_prev, δ_current, layer, Uin, temps, tempf)
    δ_current, δ_prev = δ_prev, δ_current

    unused!(net._temp_U, its_prev)

    return δ_current
    =#
end

function back_prop!(δ_current, δL, net::CovNeuralnet{Dim}, Uout_multi, Uin) where {Dim}
    #temps = similar(Uout_multi[1])
    #temps, its_temps = get_temp(net._temp_U, Dim)
    #tempf, its_tempf = get_temp(net._temp_UA, 1)

    #temps_F1 = initialize_TA_Gaugefields(temps[1])
    #tempf = [temps_F1]

    layer = net.layers[get_numlayers(net)]
    δ_prev, its_prev = get_temp(net._temp_U, Dim)
    δ_prev2, its_prev2 = get_temp(net._temp_U, Dim)
    #δ_current, its_current = get_temp(net._temp_U, Dim)
    clear_U!(δ_prev)
    substitute_U!(δ_prev2, δL)
    #similar(δL)
    #δ_current = deepcopy(δL)
    set_wing_U!(δ_prev2)

    #unused!(net._temp_U, its_temps)
    #unused!(net._temp_UA, its_tempf)


    for i = get_numlayers(net):-1:2
        layer = net.layers[i]

        temps, its_temps = get_temp(net._temp_U, Dim)
        tempf, its_tempf = get_temp(net._temp_UA, 1)
        #layer_pullback!(δ_prev, δ_current, layer, Uout_multi[i-1], temps, tempf)
        layer_pullback!(δ_prev, δ_prev2, layer, Uout_multi[i-1], temps, tempf)
        #display(δ_prev[1].U[:, :, 1, 1])
        #display(δ_prev2[1].U[:, :, 1, 1])

        unused!(net._temp_U, its_temps)
        unused!(net._temp_UA, its_tempf)

        #δ_current, δ_prev = δ_prev, δ_current
        δ_prev2, δ_prev = δ_prev, δ_prev2
        #set_wing_U!(δ_current)
    end
    layer = net.layers[1]

    temps, its_temps = get_temp(net._temp_U, Dim)
    tempf, its_tempf = get_temp(net._temp_UA, 1)
    #layer_pullback!(δ_prev, δ_current, layer, Uin, temps, tempf)
    layer_pullback!(δ_prev, δ_prev2, layer, Uin, temps, tempf)
    #display(δ_prev[1].U[:, :, 1, 1])
    #display(δ_prev2[1].U[:, :, 1, 1])

    δ_prev2, δ_prev = δ_prev, δ_prev2
    #δ_current, δ_prev = δ_prev, δ_current

    substitute_U!(δ_current, δ_prev2)

    unused!(net._temp_U, its_prev)
    unused!(net._temp_U, its_prev2)

    unused!(net._temp_U, its_temps)
    unused!(net._temp_UA, its_tempf)

end




function get_parameter_derivatives(δL, net::CovNeuralnet{Dim}, Uout_multi, Uin) where {Dim}
    temps = similar(Uout_multi[1])
    δs = get_δ_from_back_prop(δL, net, Uout_multi, Uin)
    numlayer = get_numlayers(net)

    i = 1
    layer = net.layers[i]
    U_current = Uout_multi[i]
    dSdps = parameter_derivatives(δs[i], layer, U_current, temps)

    dSdW = Vector{typeof(dSdps)}(undef, numlayer)
    dSdW[1] = dSdps
    #dSdp = Array{Vector{Float64},1}(undef, numlayer)

    for i = 2:numlayer
        layer = net.layers[i]
        U_current = Uout_multi[i]
        dSdW[i] = parameter_derivatives(δs[i], layer, U_current, temps)
        #dSdp[i] = parameter_derivatives(δs[i], layer, U_current, temps)
    end
    return dSdW
    #return dSdp
end


function get_δ_from_back_prop(δL, net::CovNeuralnet{Dim}, Uout_multi, Uin) where {Dim}
    temps = similar(Uout_multi[1])
    temps_F1 = initialize_TA_Gaugefields(temps[1])
    tempf = [temps_F1]

    layer = net.layers[get_numlayers(net)]
    δ_prev = similar(δL)
    δ_current = deepcopy(δL)
    set_wing_U!(δ_current)

    δs = Array{typeof(δL),1}(undef, get_numlayers(net))

    for i = get_numlayers(net):-1:2
        layer_pullback!(δ_prev, δ_current, layer, Uout_multi[i-1], temps, tempf)
        δs[i] = deepcopy(δ_prev)
        δ_current, δ_prev = δ_prev, δ_current
        #set_wing_U!(δ_current)
    end
    layer = net.layers[1]
    layer_pullback!(δ_prev, δ_current, layer, Uin, temps, tempf)
    δs[1] = deepcopy(δ_prev)
    δ_current, δ_prev = δ_prev, δ_current

    return δs
end


function apply_neuralnet!(
    Uout_multi,
    net::CovNeuralnet{Dim},
    Uin,
    temps,
    temps_F,
) where {Dim}
    layer = net.layers[1]
    println_verbose_level3(Uin[1], "apply_neuralnet! on 1st layer")
    apply_layer!(Uout_multi[1], layer, Uin, temps, temps_F)
    #set_wing_U!(Uout_multi[1])
    for i = 2:get_numlayers(net)
        println_verbose_level3(Uin[1], "apply_neuralnet! on $i -th layer")
        layer = net.layers[i]
        apply_layer!(Uout_multi[i], layer, Uout_multi[i-1], temps, temps_F)
        #set_wing_U!(Uout_multi[i])
    end
end


function apply_layer!(Uout, layer::T, Uin, temps, temps_F) where {T<:CovLayer}
    error(
        "apply_layer!(Uout,layer,Uin,temps,temps_F) is not implemented with type $(typeof(layer)) of layer.",
    )
end

"""
layer_pullback!(δ_prev,δ_next,layer::T,Uprev,temps,tempf) 
This is a function for a back propagation
    δ_next,Uprev -> δ_prev
"""
function layer_pullback!(δ_prev, δ_next, layer::T, Uprev, temps, tempf) where {T<:CovLayer}
    error(
        "layer_pullback!(δ_prev,δ_next,layer::T,Uprev,temps,tempf) is not implemented with type $(typeof(layer)) of layer.",
    )
end

function parameter_derivatives(δ_current, layer::T, U_current, temps) where {T<:CovLayer}
    error(
        "parameter_derivatives(δ_current,layer::T,U_current,temps) is not implemented with type $(typeof(layer)) of layer.",
    )
end


function apply_smearing_U(U::Array{T,1}, smearing::Stoutsmearing) where {T<:Abstractfields}
    numlayer = length(smearing.ρs)
    Uout_multi = Array{typeof(U),1}(undef, numlayer)
    for i = 1:numlayer
        Uout_multi[i] = similar(U)
    end
    #println("smearing.ρs ", smearing.ρs)
    #println("type U ",typeof(Uout_multi))
    #Uout = similar(U)
    calc_stout_multi!(Uout_multi, U, smearing.ρs, smearing.staples_for_stout)
    #Uout_multi = calc_stout_multi!(U,smearing.ρs,smearing.staples_for_stout) 
    return Uout_multi
end

#=
function calc_stout_multi(Uin::Array{<: AbstractGaugefields{NC,Dim},1},ρs::Array{Array{T,1},1},staples)  where {NC,Dim,T <: Number}
    numlayer = length(ρs)
    #println("numlayer = ",numlayer,"\t",ρs)
    Utype = eltype(Uin)
    Uout_multi = Array{Array{Utype,1}}(undef,numlayer)
    for i=1:numlayer
        Uout_multi[i] = similar(Uin)
    end
    calc_stout_multi!(Uout_multi,Uin,ρs,staples)

    return Uout_multi
end
=#

function calc_stout_multi!(
    Uout_multi::Vector{<:Vector{<:AbstractGaugefields{NC,Dim}}},
    Uin::Array{<:AbstractGaugefields{NC,Dim},1},
    ρs::Array{Array{T,1},1},
    staples,
) where {NC,Dim,T<:Number}
    numlayer = length(ρs)
    Utmp = similar(Uin)
    #Uout_multi = Array{Array{GaugeFields{SU{NC}},1}}(undef,numlayer)
    U = deepcopy(Uin)
    for i = 1:numlayer
        if i != numlayer
            apply_stout_smearing!(Utmp, U, ρs[i], staples)
            Uout_multi[i] = deepcopy(Utmp)
            Utmp, U = U, Utmp
        else
            apply_stout_smearing!(Uout_multi[i], U, ρs[i], staples)
        end
    end
end

function calc_stout_multi!(
    Uout::Array{<:AbstractGaugefields{NC,Dim},1},
    Uin::Array{<:AbstractGaugefields{NC,Dim},1},
    ρs::Array{Array{T,1},1},
    staples,
) where {NC,Dim,T<:Number}
    numlayer = length(ρs)
    Utmp = similar(Uin)
    #Uout_multi = Array{Array{GaugeFields{SU{NC}},1}}(undef,numlayer)
    U = deepcopy(Uin)
    for i = 1:numlayer
        if i != numlayer
            apply_stout_smearing!(Utmp, U, ρs[i], staples)
            Utmp, U = U, Utmp
        else
            apply_stout_smearing!(Uout, U, ρs[i], staples)
        end
    end

end

function apply_stout_smearing!(
    Uout::Array{<:AbstractGaugefields{NC,Dim},1},
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    ρs::Array{T,1},
    staples,
) where {NC,Dim,T<:Number}
    @assert Uout != U "input U and output U should not be same!"
    V = similar(U[1])
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    temp3 = similar(U[1])
    F0 = initialize_TA_Gaugefields(U[1])

    num = length(ρs)

    for μ = 1:Dim
        clear_U!(V)
        for i = 1:num
            loops = staples[i][μ]
            evaluate_wilson_loops!(temp3, loops, U, [temp1, temp2])
            add_U!(V, ρs[i], temp3)
        end
        mul!(temp1, V, U[μ]') #U U*V
        clear_U!(F0)
        Traceless_antihermitian_add!(F0, 1, temp1)

        exptU!(temp3, 1, F0, [temp1, temp2])


        mul!(Uout[μ], temp3, U[μ])
    end
    set_wing_U!(Uout)


    #error("ee")

    #error("not implemented")
end







#=


abstract type SmearingParam end

struct SmearingParam_nosmearing <: SmearingParam 
end

abstract type SmearingParam_single <: SmearingParam
end

abstract type SmearingParam_multi <: SmearingParam
end

mutable struct SmearingParam_stout <: SmearingParam_single
    staples_for_stout::Array{Array{Wilson_loop_set,1},1}
    tensor_derivative::Array{Tensor_derivative_set,1}
    staples_for_stout_dag::Array{Array{Wilson_loop_set,1},1}
    tensor_derivative_dag::Array{Tensor_derivative_set,1}
    ρs::Array{Float64,1}
    #ρs::Array{Float64,1}
end

mutable struct SmearingParam_stout_multi <: SmearingParam_multi
    staples_for_stout::Array{Array{Wilson_loop_set,1},1}
    tensor_derivative::Array{Tensor_derivative_set,1}
    staples_for_stout_dag::Array{Array{Wilson_loop_set,1},1}
    tensor_derivative_dag::Array{Tensor_derivative_set,1}
    ρs::Array{Array{Float64,1},1}
    #ρs::Array{Float64,1}
end

const Nosmearing = SmearingParam_nosmearing
const Stout = SmearingParam_stout

=#
end
