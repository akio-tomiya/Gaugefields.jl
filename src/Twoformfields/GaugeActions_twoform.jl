
import Wilsonloop: Wilsonline, make_staple

mutable struct GaugeAction_withTwoform{Dim,Tgauge,Tdata,Ttwoform<:AbstractTwoformfields,Tlabel} <: AbstractGaugeAction{Dim,Tgauge}
    gaugeaction::GaugeAction{Dim,Tgauge,Tdata}
    twoformfield::Ttwoform
    storedTwoformfields_forloop::Vector{PrealocatedTwoformfields{Ttwoform,Tgauge,Tlabel}}
    storedTwoformfields_forstaple::Vector{Vector{PrealocatedTwoformfields{Ttwoform,Tgauge,Tlabel}}}
    usestoreddata::Bool
    usestored_forstaple::Bool
    numfields::Int64
    numfields_forstaple::Int64

    function GaugeAction_withTwoform(
        U::Vector{Tgauge},
        twoformfield::Ttwoform,
        preallocatedtwoformfields::PrealocatedTwoformfields{Ttwoform,Tgauge,Tlabel};
        hascovnet=false,
        usestoreddata=true,
        usestoreddata_forstaple=true,
        numfields=12,
        numfields_forstaple=9
    ) where {Ttwoform<:AbstractTwoformfields,Tgauge,Tlabel}

        gaugeaction = GaugeAction(
            U;
            hascovnet
        )

        Tdata = eltype(gaugeaction.dataset)

        storedTwoformfields = Vector{typeof(preallocatedtwoformfields)}(undef, 0)

        Dim = length(U)
        storedTwoformfields_forstaple = Vector{Vector{typeof(preallocatedtwoformfields)}}(undef, 0)
        #for μ = 1:Dim
        #    storedTwoformfields_forstaple[μ] = Vector{typeof(preallocatedtwoformfields)}(undef, 0)
        #end

        return new{Dim,Tgauge,Tdata,Ttwoform,Tlabel}(
            gaugeaction,
            twoformfield,
            storedTwoformfields,
            storedTwoformfields_forstaple,
            usestoreddata,
            usestoreddata_forstaple,
            numfields,
            numfields_forstaple
        )
    end
end

function get_temporary_gaugefields(S::GaugeAction_withTwoform)
    return get_temporary_gaugefields(S.gaugeaction)
end


function update_twoformfield!(action::GaugeAction_withTwoform, twoformfield::Ttwoform) where {Ttwoform<:AbstractTwoformfields}
    action.twoformfield = twoformfield
    return
end

export update_twoformfield!, GaugeAction_withTwoform


function Base.push!(
    S::GaugeAction_withTwoform{Dim,Tgauge,Tdata,Ttwoform,Tlabel},
    β::T,
    closedloops::Vector{Wilsonline{Dim}},
) where {Dim,Tgauge,Tdata,Ttwoform<:AbstractTwoformfields,Tlabel,T}
    push!(S.gaugeaction, β, closedloops)

    storedTwoformfields_forstaple_i = Vector{eltype(S.storedTwoformfields_forloop)}(undef, Dim)

    numloops = length(closedloops)
    for μ = 1:Dim
        storedTwoformfields_forstaple_i[μ] = PrealocatedTwoformfields(S.twoformfield; num=S.numfields_forstaple)
        #pfields = PrealocatedTwoformfields(S.twoformfield; num=S.numfields_forstaple)

        staples = Wilsonline{Dim}[]
        for i = 1:numloops
            staples_i = make_staple(closedloops[i], μ)
            for j = 1:length(staples_i)
                push!(staples, staples_i[j])
            end
        end
        #println("μ = $μ ", length(staples))
        for staple in staples
            add_Wilsonline!(storedTwoformfields_forstaple_i[μ], staple)
        end
        #println(typeof(S.storedTwoformfields_forstaple[μ]))
        #println(typeof(pfields))
        #push!(S.storedTwoformfields_forstaple[μ], pfields)
        #error("dd")
        #allstaples[μ] = staples
    end
    push!(S.storedTwoformfields_forstaple, storedTwoformfields_forstaple_i)


    pfields = PrealocatedTwoformfields(S.twoformfield; num=S.numfields)

    #println(length(closedloops))
    for loop in closedloops
        add_Wilsonline!(pfields, loop)
    end

    push!(S.storedTwoformfields_forloop, pfields)
end

function calc_dSdUμ!(
    dSdUμ::T, # dSdUμ -> S._temp_U[end] or other-temp
    S::GaugeAction_withTwoform{Dim,T,Tdata,Ttwoform,Tlabel},
    μ,
    U::Vector{T},
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim},Tdata,Ttwoform,Tlabel}
    temp, it_temp = get_temp(S.gaugeaction._temp_U)
    temps, its_temps = get_temp(S.gaugeaction._temp_U, 5)

    #temp = S._temp_U[end-1]
    numterm = length(S.gaugeaction.dataset)

    clear_U!(dSdUμ)
    for i = 1:numterm
        dataset = S.gaugeaction.dataset[i]
        storedTwoformfields_forstaple = S.storedTwoformfields_forstaple[i]

        β = dataset.β
        staples_μ = dataset.staples[μ]
        storedTwoformfields_forstaple_μ = storedTwoformfields_forstaple[μ]
        #println("staples_μ ", staples_μ)
        #error("staple")

        evaluate_gaugelinks!(temp, staples_μ, U, storedTwoformfields_forstaple_μ, temps)
        #evaluate_gaugelinks!(temp, staples_μ, U, B, temps)

        add_U!(dSdUμ, β, temp)
    end
    set_wing_U!(dSdUμ)
    unused!(S.gaugeaction._temp_U, it_temp)
    unused!(S.gaugeaction._temp_U, its_temps)

end