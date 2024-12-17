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

import ..Temporalfields_module: Temporalfields, unused!, get_temp, set_reusemode!



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
    _temp_U::Temporalfields{T}
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
    set_reusemode!(S._temp_U, true)
    return S._temp_U
end
function free_temporary_gaugefields(S::GaugeAction)
    set_reusemode!(S._temp_U, false)
    return
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
    temp,  it_temp   = get_temp(S._temp_U)
    temps, its_temps = get_temp(S._temp_U, 4)
    numterm = length(S.dataset)

    clear_U!(dSdUμ)
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        staples_μ = dataset.staples[μ]
        evaluate_gaugelinks!(temp, staples_μ, U, temps)

        #println("temp in dSdUμ! ",getvalue(temp,1,1,1,1,1,1))
        add_U!(dSdUμ, β, temp)
        #println("dSdUμ! ",getvalue(dSdUμ,1,1,1,1,1,1))
    end
    set_wing_U!(dSdUμ)
    unused!(S._temp_U, it_temp)
    unused!(S._temp_U, its_temps)

end
function calc_dSdUμ!(
    dSdUμ::T,
    S::GaugeAction,
    μ,
    U::Vector{T},
    B::Array{T,2},
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    temp,  it_temp   = get_temp(S._temp_U)
    temps, its_temps = get_temp(S._temp_U, 5)
    numterm = length(S.dataset)

    clear_U!(dSdUμ)
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        staples_μ = dataset.staples[μ]
        evaluate_gaugelinks!(temp, staples_μ, U, B, temps)

        add_U!(dSdUμ, β, temp)
    end
    set_wing_U!(dSdUμ)
    unused!(S._temp_U, it_temp)
    unused!(S._temp_U, its_temps)

end

function evaluate_GaugeAction(
    S::GaugeAction,
    U::Vector{<:AbstractGaugefields{NC,Dim}},
) where {Dim,NC}
    temp1, it_temp1 = get_temp(S._temp_U)
    evaluate_GaugeAction_untraced!(temp1, S, U)
    value = tr(temp1)
    unused!(S._temp_U, it_temp1)
    return value
end
function evaluate_GaugeAction(
    S::GaugeAction,
    U::Vector{T},
    B::Array{T,2}
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    temp1, it_temp1 = get_temp(S._temp_U)
    evaluate_GaugeAction_untraced!(temp1, S, U, B)
    value = tr(temp1)
    unused!(S._temp_U, it_temp1)
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
    mat_temps, # length >= 5
    indices...,
) where {Dim,NC}

    temp5 = mat_temps[5]
    temps = mat_temps[1:4]
    numterm = length(S.dataset)
    mat_U .= 0
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        staples_μ = dataset.staples[μ]
        evaluate_gaugelinks_eachsite!(temp5, staples_μ, U, temps, indices...) # length >~ 3,4
        mat_U .+= β * temp5
    end

end

function evaluate_GaugeAction_untraced!(
    uout,
    S::GaugeAction, # length(temps) > 3 + 1
    U::Vector{<:AbstractGaugefields{NC,Dim}},
) where {Dim,NC}
    numterm = length(S.dataset)
    temp5, it_temp5 = get_temp(S._temp_U)
    clear_U!(uout)

    temps, its_temps = get_temp(S._temp_U, 4)
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        w = dataset.closedloops
        evaluate_gaugelinks!(temp5, w, U, temps)
        add_U!(uout, β, temp5)
    end
    set_wing_U!(uout)
    unused!(S._temp_U, its_temps)
    unused!(S._temp_U, it_temp5)
    return
end
function evaluate_GaugeAction_untraced!(
    uout,
    S::GaugeAction, # length(temps) > 6
    U::Vector{T},
    B::Array{T,2}
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    numterm = length(S.dataset)
    temp, it_temp = get_temp(S._temp_U)
    temps, its_temps = get_temp(S._temp_U, 5)
    clear_U!(uout)

    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        w = dataset.closedloops
        evaluate_gaugelinks!(temp, w, U, B, temps)
        add_U!(uout, β, temp)
    end
    set_wing_U!(uout)
    unused!(S._temp_U, it_temp)
    unused!(S._temp_U, its_temps)
    return
end

function GaugeAction(
    U::Vector{<:AbstractGaugefields{NC,Dim}};
    hascovnet=false,
) where {NC,Dim}
    if hascovnet
        covneuralnet = CovNeuralnet(Dim=Dim)
    else
        covneuralnet = nothing
    end
    dataset = GaugeAction_dataset{Dim}[]

    num = 12
    _temp_U = Temporalfields(U[1]; num=num)

    return GaugeAction{Dim,eltype(U),eltype(dataset)}(hascovnet, covneuralnet, dataset, _temp_U)
end
function GaugeAction(
    U::Vector{T},
    B::Array{T,2};
    hascovnet=false,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim}}
    if hascovnet
        covneuralnet = CovNeuralnet(Dim=Dim)
    else
        covneuralnet = nothing
    end
    dataset = GaugeAction_dataset{Dim}[]

    num = 12
    _temp_U = Temporalfields(U[1]; num=num)

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
