module GaugeAction_module
import ..Abstractsmearing_module: CovNeuralnet
import ..AbstractGaugefields_module:
    AbstractGaugefields,
    evaluate_gaugelinks!,
    add_U!,
    clear_U!,
    set_wing_U!,
    getvalue,
    evaluate_gaugelinks_eachsite!

import Wilsonloop: Wilsonline, make_staple
using LinearAlgebra
using InteractiveUtils



struct GaugeAction_dataset{Dim}
    #β::Float64
    β::ComplexF64
    closedloops::Vector{Wilsonline{Dim}}
    staples::Vector{Vector{Wilsonline{Dim}}}
end

struct GaugeAction{Dim,T,Tdata} 
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    dataset::Vector{Tdata}
    _temp_U::Vector{T}
end

function GaugeAction_dataset(β, closedloops::Vector{Wilsonline{Dim}}) where {Dim}
    allstaples = Array{Vector{Wilsonline{Dim}},1}(undef, Dim)
    numloops = length(closedloops)
    for μ = 1:Dim
        staples = Wilsonline{Dim}[]
        for i = 1:numloops
            staples_i = make_staple(closedloops[i], μ)
            for j = 1:length(staples_i)
                push!(staples, staples_i[j])
            end
        end
        allstaples[μ] = staples
    end
    return GaugeAction_dataset{Dim}(β, closedloops, allstaples)
end

function get_temporary_gaugefields(S::GaugeAction)
    return S._temp_U
end

function calc_dSdUμ(
    S::GaugeAction,
    μ,
    U::Vector{T},
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    dSdUμ = similar(U[1])
    calc_dSdUμ!(dSdUμ, S, μ, U)
    return dSdUμ
end
function calc_dSdUμ(
    S::GaugeAction,
    μ,
    U::Vector{T},
    B::Array{T,2},
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    dSdUμ = similar(U[1])
    calc_dSdUμ!(dSdUμ, S, μ, U, B)
    return dSdUμ
end

function calc_dSdUμ!(
    dSdUμ::T,
    S::GaugeAction,
    μ,
    U::Vector{T},
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    temp = S._temp_U[end-1]
    numterm = length(S.dataset)

    clear_U!(dSdUμ)
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        staples_μ = dataset.staples[μ]
        evaluate_gaugelinks!(temp, staples_μ, U, S._temp_U[1:end-2])

        #println("temp in dSdUμ! ",getvalue(temp,1,1,1,1,1,1))
        add_U!(dSdUμ, β, temp)
        #println("dSdUμ! ",getvalue(dSdUμ,1,1,1,1,1,1))
    end
    set_wing_U!(dSdUμ)

end
function calc_dSdUμ!(
    dSdUμ::T,
    S::GaugeAction,
    μ,
    U::Vector{T},
    B::Array{T,2},
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    temp = S._temp_U[end-1]
    numterm = length(S.dataset)

    clear_U!(dSdUμ)
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        staples_μ = dataset.staples[μ]
        evaluate_gaugelinks!(temp, staples_μ, U, B, S._temp_U[1:end-2])

        add_U!(dSdUμ, β, temp)
    end
    set_wing_U!(dSdUμ)

end

function evaluate_GaugeAction(
    S::GaugeAction,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
) where {Dim,NC}
    temp1 = S._temp_U[end]
    evaluate_GaugeAction_untraced!(temp1, S, U)
    value = tr(temp1)
    return value
end
function evaluate_GaugeAction(
    S::GaugeAction,
    U::Vector{T},
    B::Array{T,2}
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    temp1 = S._temp_U[end]
    evaluate_GaugeAction_untraced!(temp1, S, U, B)
    value = tr(temp1)
    return value
end

function evaluate_GaugeAction_untraced(
    S::GaugeAction,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
) where {Dim,NC}
    uout = similar(U[1])
    clear_U!(uout)

    evaluate_GaugeAction_untraced!(uout, S, U)

    return uout
end
function evaluate_GaugeAction_untraced(
    S::GaugeAction,
    U::Vector{T},
    B::Array{T,2}
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    uout = similar(U[1])
    clear_U!(uout)

    evaluate_GaugeAction_untraced!(uout, S, U, B)

    return uout
end

function evaluate_staple_eachindex!(
    mat_U,
    μ,
    S::GaugeAction,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
    mat_temps,
    indices...,
) where {Dim,NC}
    temp3 = mat_temps[5]
    numterm = length(S.dataset)
    mat_U .= 0
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        staples_μ = dataset.staples[μ]
        evaluate_gaugelinks_eachsite!(temp3, staples_μ, U, view(mat_temps, 1:4), indices...)
        mat_U .+= β * temp3
    end
end

function evaluate_GaugeAction_untraced!(
    uout,
    S::GaugeAction, # length(temps) > 3
    U::Vector{<:AbstractGaugefields{NC,Dim}},
) where {Dim,NC}
    numterm = length(S.dataset)
    temp1 = S._temp_U[1]
    temp2 = S._temp_U[2]
    temp3 = S._temp_U[3]
    clear_U!(uout)

    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        w = dataset.closedloops
        evaluate_gaugelinks!(temp3, w, U, [temp1, temp2])
        add_U!(uout, β, temp3)
    end
    set_wing_U!(uout)

    return
end
function evaluate_GaugeAction_untraced!(
    uout,
    S::GaugeAction, # length(temps) > 9
    U::Vector{T},
    B::Array{T,2}
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    numterm = length(S.dataset)
    temp = S._temp_U[6]
    clear_U!(uout)

    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        w = dataset.closedloops
        evaluate_gaugelinks!(temp, w, U, B, S._temp_U[1:5])
        add_U!(uout, β, temp)
    end
    set_wing_U!(uout)

    return
end

function GaugeAction(
    U::Vector{<:AbstractGaugefields{NC,Dim}};
    hascovnet = false,
) where {NC,Dim}
    if hascovnet
        covneuralnet = CovNeuralnet(Dim = Dim)
    else
        covneuralnet = nothing
    end
    dataset = GaugeAction_dataset{Dim}[]
    num = 4
    _temp_U = Array{eltype(U)}(undef, num)
    for i = 1:num
        _temp_U[i] = similar(U[1])
    end

    return GaugeAction{Dim,eltype(U),eltype(dataset)}(hascovnet, covneuralnet, dataset, _temp_U)
end
function GaugeAction(
    U::Vector{T},
    B::Array{T,2};
    hascovnet = false,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    if hascovnet
        covneuralnet = CovNeuralnet(Dim = Dim)
    else
        covneuralnet = nothing
    end
    dataset = GaugeAction_dataset{Dim}[]
    num = 7
    _temp_U = Array{eltype(U)}(undef, num)
    for i = 1:num
        _temp_U[i] = similar(U[1])
    end

    return GaugeAction{Dim,eltype(U),eltype(dataset)}(hascovnet, covneuralnet, dataset, _temp_U)
end

function Base.push!(
    S::GaugeAction{Dim,T1,Tdata},
    β::T,
    closedloops::Vector{Wilsonline{Dim}},
) where {Dim,T<:Number,T1,Tdata}
    dataset = GaugeAction_dataset(β, closedloops)
    @assert Tdata == typeof(dataset) "type of dataset should be $Tdata but now $(typeof(dataset))"
    push!(S.dataset, dataset)
end

function Base.show(s::GaugeAction{Dim,T,Tdata}) where {Dim,T,Tdata}
    println("----------------------------------------------")
    println("Structure of the actions for Gaugefields")
    println("num. of terms: ", length(s.dataset))
    for i = 1:length(s.dataset)
        if i == 1
            string = "st"
        elseif i == 2
            string = "nd"
        elseif i == 3
            string = "rd"
        else
            string = "th"
        end
        println("-------------------------------")
        println("      $i-$string term: ")
        println("          coefficient: ", s.dataset[i].β)
        println("      -------------------------")
        show(s.dataset[i].closedloops)
        println("      -------------------------")
    end
    println("----------------------------------------------")
end


end
