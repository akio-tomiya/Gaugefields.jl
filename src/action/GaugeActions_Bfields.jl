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
function calc_dSdUμ(
    S::GaugeAction,
    μ,
    U::Vector{T},
    B::Array{T,2},
    Bps::Pz,
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim},Pz<:Storedlinkfields}
    dSdUμ = similar(U[1])
    calc_dSdUμ!(dSdUμ, S, μ, U, B, Bps)
    return dSdUμ
end

function calc_dSdUμ!(
    dSdUμ::T, # dSdUμ -> S._temp_U[end] or other-temp
    S::GaugeAction,
    μ,
    U::Vector{T},
    B::Array{T,2},
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    temp, it_temp = get_temp(S._temp_U)
    temps, its_temps = get_temp(S._temp_U, 5)

    #temp = S._temp_U[end-1]
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
function calc_dSdUμ!(
    dSdUμ::T,
    S::GaugeAction,
    μ,
    U::Vector{T},
    B::Array{T,2},
    Bps::Pz,
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim},Pz<:Storedlinkfields}
    temp,  it_temp   = get_temp(S._temp_U)
    temps, its_temps = get_temp(S._temp_U, 10)
    numterm = length(S.dataset)

    clear_U!(dSdUμ)
    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        staples_μ = dataset.staples[μ]
        evaluate_gaugelinks!(temp, staples_μ, U, B, Bps, temps)

        add_U!(dSdUμ, β, temp)
    end
    set_wing_U!(dSdUμ)
    unused!(S._temp_U, it_temp)
    unused!(S._temp_U, its_temps)

end

function evaluate_GaugeAction(
    S::GaugeAction,
    U::Vector{T},
    B::Array{T,2}
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    temp1, it_temp1 = get_temp(S._temp_U)
    #temp1 = S._temp_U[end]
    evaluate_GaugeAction_untraced!(temp1, S, U, B)
    value = tr(temp1)
    unused!(S._temp_U, it_temp1)
    return value
end
function evaluate_GaugeAction(
    S::GaugeAction,
    U::Vector{T},
    B::Array{T,2},
    Bps::Pz,
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim},Pz<:Storedlinkfields}
    temp1, it_temp1 = get_temp(S._temp_U)
    evaluate_GaugeAction_untraced!(temp1, S, U, B, Bps)
    value = tr(temp1)
    unused!(S._temp_U, it_temp1)
    return value
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
function evaluate_GaugeAction_untraced(
    S::GaugeAction,
    U::Vector{T},
    B::Array{T,2},
    Bps::Pz,
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim},Pz<:Storedlinkfields}
    uout = similar(U[1])
    clear_U!(uout)

    evaluate_GaugeAction_untraced!(uout, S, U, B, Bps)

    return uout
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
    #temp = S._temp_U[6]
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
function evaluate_GaugeAction_untraced!(
    uout,
    S::GaugeAction, # length(temps) > 9 + 2
    U::Vector{T},
    B::Array{T,2}
) where {Dim,NC,T<:AbstractGaugefields{NC,Dim}}
    numterm = length(S.dataset)
    temp, it_temp = get_temp(S._temp_U)
    temps, its_temps = get_temp(S._temp_U, 10)
    clear_U!(uout)

    for i = 1:numterm
        dataset = S.dataset[i]
        β = dataset.β
        w = dataset.closedloops
        evaluate_gaugelinks!(temp, w, U, B, Bps, temps)
        add_U!(uout, β, temp)
    end
    set_wing_U!(uout)
    unused!(S._temp_U, it_temp)
    unused!(S._temp_U, its_temps)
    return
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

    #_temp_U = Array{eltype(U)}(undef, num)
    #for i = 1:num
    #    _temp_U[i] = similar(U[1])
    #end

    return GaugeAction{Dim,eltype(U),eltype(dataset)}(hascovnet, covneuralnet, dataset, _temp_U)
end
function GaugeAction(
    U::Vector{T},
    B::Array{T,2},
    Bps::Pz;
    hascovnet=false,
) where {NC,Dim,T<:AbstractGaugefields{NC,Dim},Pz<:Storedlinkfields}
    if hascovnet
        covneuralnet = CovNeuralnet(Dim=Dim)
    else
        covneuralnet = nothing
    end
    dataset = GaugeAction_dataset{Dim}[]

    num = 14
    _temp_U = Temporalfields(U[1]; num=num)

    return GaugeAction{Dim,eltype(U),eltype(dataset)}(hascovnet, covneuralnet, dataset, _temp_U)
end

