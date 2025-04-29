module Twoformfields
using Requires
import ..AbstractGaugefields_module: AbstractGaugefields, identityGaugefields_4D_nowing,
    set_wing_U!, minusidentityGaugefields_4D_nowing,
    substitute_U!,
    initialize_TA_Gaugefields,
    evaluate_gaugelinks!,
    TA_Gaugefields,
    clear_U!,
    shift_U,
    multiply_12!,
    add_U!,
    unit_U!,
    Traceless_antihermitian_add!,
    exp_aF_U!
import Wilsonloop: loops_staple_prime, Wilsonline, get_position, get_direction, GLink, isdag, make_cloverloops
import ..Temporalfields_module: Temporalfields, get_temp, unused!
import ..Wilsonloops_module: Wilson_loop_set
using LinearAlgebra
using PreallocatedArrays
import ..GaugeAction_module: get_temporary_gaugefields

abstract type AbstractTwoformfields end

include("Bfields.jl")
export Bfield, Initialize_Bfields

struct PrealocatedTwoformfields{TT<:AbstractTwoformfields,TG,TL}
    twoformfield::TT
    data::PreallocatedArray{TG,TL,true}
    tempgaugefields::PreallocatedArray{TG,Union{Nothing,String},false}
end

include("preallocatedBfields.jl")

function add_Wilsonline!(p::PrealocatedTwoformfields{TT,TG,TL}, w::Wilsonline{Dim}) where {TT,TG,TL,Dim}
    error("add_Wilsonline! not implemented for this type", TT)
end

function load_Wilsonline(p::PrealocatedTwoformfields{TT,TG,TL}, w::Wilsonline{Dim}) where {TT,TG,TL,Dim}
    error("load_Wilsonline not implemented for this type", TT)
end

function evaluate_gaugelinks!(
    xout::T,
    w::Array{WL,1},
    U::Array{T,1},
    pf::PrealocatedTwoformfields{TT,TG,TL},
    temps::Array{T,1}, # length >= 5
) where {Dim,WL<:Wilsonline{Dim},T<:AbstractGaugefields,TT,TG,TL}
    num = length(w)
    temp1 = temps[5]

    clear_U!(xout)
    for i = 1:num
        glinks = w[i]
        ti, index = load_Wilsonline(pf, glinks)
        evaluate_gaugelinks!(temp1, glinks, U, temps[1:4]) # length >= 4
        mul!(temps[1], ti, temp1)
        add_U!(xout, temps[1])
        #add_U!(xout, temp1)
    end

    return
end


export PrealocatedTwoformfields, add_Wilsonline!, load_Wilsonline



end