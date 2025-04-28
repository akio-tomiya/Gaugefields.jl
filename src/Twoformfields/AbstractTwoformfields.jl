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
    unit_U!
import Wilsonloop: loops_staple_prime, Wilsonline, get_position, get_direction, GLink, isdag, make_cloverloops
import ..Temporalfields_module: Temporalfields, get_temp, unused!
import ..Wilsonloops_module: Wilson_loop_set
using LinearAlgebra
using PreallocatedArrays

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

export PrealocatedTwoformfields, add_Wilsonline!, load_Wilsonline



end