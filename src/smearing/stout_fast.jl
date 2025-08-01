
import ..AbstractGaugefields_module: clear_U!, add_U!, Gaugefields_4D_nowing, substitute_U!, Gaugefields_4D_wing
import ..AbstractGaugefields_module: calc_coefficients_Q, calc_Bmatrix!
import ..Temporalfields_module: Temporalfields, unused!, get_temp

mutable struct STOUT_Layer{T,Dim,TN} <: CovLayer{Dim}
    ρs::TN
    const dataset::Vector{STOUT_dataset{Dim}}
    const Uin::Vector{T}
    const Uinβ::Vector{T}
    const eQs::Vector{T}
    const Cs::Vector{T}
    const Qs::Vector{T}
    const temps::Temporalfields{T}
    const dSdCs::Vector{T}
    islocalρ::Bool
    isαβsame::Bool
    hasdSdCs::Vector{Bool}#Bool
    dSdρ::Union{Nothing,TN}
    #=
    ρs::Vector{TN}
    dataset::Vector{STOUT_dataset{Dim}}
    Uin::Vector{T}
    const Uinβ::Vector{T}
    eQs::Vector{T}
    Cs::Vector{T}
    Qs::Vector{T}
    #temps::Vector{T}
    temps::Temporalfields{T}
    dSdCs::Vector{T}
    hasdSdCs::Vector{Bool}
    dSdρ::Vector{TN}
    =#
end
export STOUT_Layer

function STOUT_Layer(loops_smearing, L, U::Vector{<:AbstractGaugefields{NC,Dim}}, ρs::Vector{<:Number}) where {NC,Dim}
    loopset = make_loopforactions(loops_smearing, L)
    return STOUT_Layer(loopset, U, ρs)
end

function STOUT_Layer(loops_smearing, ρs::Vector{<:Number}, L, U::Vector{<:AbstractGaugefields{NC,Dim}}) where {NC,Dim}
    loopset = make_loopforactions(loops_smearing, L)
    return STOUT_Layer(loopset, U, ρs)
end

function STOUT_Layer(loops_smearing, ρs::Vector{<:Number}, U::Vector{<:AbstractGaugefields{NC,Dim}}) where {NC,Dim}
    _, _, NN... = size(U[1])
    if Dim == 2
        L = [NN[1], NN[2]]
    elseif Dim == 4
        L = [NN[1], NN[2], NN[3], NN[4]]
    end

    loopset = make_loopforactions(loops_smearing, L)
    return STOUT_Layer(loopset, U, ρs)
end



function STOUT_Layer(loopset::Vector{Vector{Wilsonline{Dim}}}, U::Vector{<:AbstractGaugefields{NC,Dim}}, ρs=zeros(Float64, length(loopset))) where {NC,Dim}
    T = eltype(U)
    numg = 5 + Dim - 1 + 2
    temps = Temporalfields(U[1]; num=numg)
    #=
    temps = Vector{T}(undef, numg)
    for i = 1:numg
        temps[i] = similar(U[1])
    end
    =#

    #num = length(loops_smearing)
    #loopset = make_loopforactions(loops_smearing, L)
    num = length(loopset)
    #display(loopset)
    #error("d")
    dataset = Vector{STOUT_dataset{Dim}}(undef, num)
    for i = 1:num
        closedloops = loopset[i] #one of loopset, like plaq. There are several loops. 
        dataset[i] = STOUT_dataset(closedloops, Dim=Dim)
    end

    Uin = Vector{T}(undef, Dim)
    Uinβ = Vector{T}(undef, Dim)
    eQs = Vector{T}(undef, Dim)
    Cs = Vector{T}(undef, Dim)
    Qs = Vector{T}(undef, Dim)
    dSdCs = Vector{T}(undef, Dim)

    for μ = 1:Dim
        Uin[μ] = similar(U[1])
        Uinβ[μ] = similar(U[1])
        eQs[μ] = similar(U[1])
        Cs[μ] = similar(U[1])
        Qs[μ] = similar(U[1])
        dSdCs[μ] = similar(U[1])
    end
    #dSdρ = nothing
    TN = typeof(ρs)#eltype(ρs)
    if eltype(ρs) <: Number
        islocalρ = false
    else
        islocalρ = true
    end
    dSdρ = zero(ρs)

    #=

    for μ = 1:Dim
        Uin[μ] = similar(U[1])
        eQs[μ] = similar(U[1])
        Cs[μ] = similar(U[1])
        Qs[μ] = similar(U[1])
        dSdCs[μ] = similar(U[1])
    end
    dSdρ = zero(ρs)
    TN = eltype(ρs)
    hasdSdCs = [false]

    st = STOUT_Layer{T,Dim,TN}(ρs, dataset, Uin, eQs, Cs, Qs, temps, dSdCs, hasdSdCs, dSdρ)
    =#
    hasdSdCs = [false]
    return STOUT_Layer{T,Dim,TN}(ρs, dataset, Uin, Uinβ, eQs, Cs, Qs, temps, dSdCs, islocalρ, false, hasdSdCs, dSdρ)

    return st
end

function get_name(s::STOUT_Layer)
    return "STOUT"
end

function Base.show(s::STOUT_Layer{T,Dim,TN}) where {T,Dim,TN}
    println("num. of terms: ", length(s.ρs))
    for i = 1:length(s.ρs)
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
        println("          coefficient: ", s.ρs[i])
        println("      -------------------------")
        show(s.dataset[i].closedloop)
        println("      -------------------------")
    end
end


function set_parameters!(s::STOUT_Layer, ρs)
    s.ρs[:] .= ρs
    #println(ρs)
end

function get_parameters(s::STOUT_Layer)
    return deepcopy(s.ρs)
end

function get_parameter_derivatives(s::STOUT_Layer)
    return deepcopy(s.dSdρ)
end

function get_numparameters(s::STOUT_Layer)
    return length(s.ρs)
end

function zero_grad!(s::STOUT_Layer)
    if s.dSdρ != nothing
        s.dSdρ .= 0
    end
end

function Base.length(layer::STOUT_Layer)
    return length(layer.ρs)
end

function get_Cμ(layer::STOUT_Layer, i)
    return layer.dataset[i].Cμ
end

function get_dCμdUν(layer::STOUT_Layer, i)
    return layer.dataset[i].dCμdUν
end

function get_dCμdagdUν(layer::STOUT_Layer, i)
    return layer.dataset[i].dCμdagdUν
end

function get_ρ(layer::STOUT_Layer, i)
    return layer.ρs[i]
end


function make_longstaple_pair(μ, ν, s)
    loop = [(μ, +1), (ν, +s), (μ, -1), (ν, -s)]
    #loop = [(ν, +s), (μ, +1), (ν, -s)]
    w1 = Wilsonline(loop)
    #w = Vector{typeof(w1)}(undef, 2)
    w = Vector{typeof(w1)}(undef, 1)
    w[1] = w1
    #loop = [(ν, -s), (μ, +1), (ν, +s)]
    #w2 = Wilsonline(loop)
    #w[2] = w2
    return w
end
export make_longstaple_pair

function apply_layer!(
    Uout::Array{<:AbstractGaugefields{NC,Dim},1},
    layer::STOUT_Layer{T,Dim,TN},
    Uin,
    temps,
    tempf,
) where {NC,Dim,T,TN}


    ρs = layer.ρs
    forward!(layer, Uout, ρs, Uin)
    set_wing_U!(Uout)
    return
end

function layer_pullback!(
    δ_prev::Array{<:AbstractGaugefields{NC,Dim},1},
    δ_current,
    layer::STOUT_Layer{T,Dim},
    Uprev,
    temps,
    tempf,
) where {NC,Dim,T}
    clear_U!(δ_prev)

    dSdρ = layer.dSdρ

    #dSdU2 = temps[1:Dim]
    #clear_U!(dSdU2)

    backward_dSdUαUβρ_add!(layer, δ_prev, dSdρ, δ_current)
    #add_U!(δ_prev, 1, dSdU2)


    #backward_dSdUα_add!(layer, δ_prev, δ_current)
    #backward_dSdUβ_add!(layer, δ_prev, δ_current)
    #backward_dSdρ_add!(layer, dSdρ, δ_current)
    set_wing_U!(δ_prev)
    return
end


function forward!(s::STOUT_Layer{T,Dim}, Uout, ρs::Vector{TN}, Uin) where {T,Dim,TN<:Number} #Uout = exp(Q(Uin,ρs))*Uinα
    isαβsame = true
    substitute_U!(s.Uin, Uin)
    for i = 1:length(s.ρs)
        s.ρs[i] = deepcopy(ρs[i])
    end
    #temps = s.temps
    #Ω = temps[end]
    for μ = 1:Dim
        calc_C!(s.Cs[μ], μ, ρs, s.dataset, Uin, s.temps)
        Ω, iΩ = get_temp(s.temps)
        #Ω = temps[1]
        mul!(Ω, s.Cs[μ], Uin[μ]') #Ω = C*Udag
        Traceless_antihermitian!(s.Qs[μ], Ω)
        unused!(s.temps, iΩ)
        temps, it_s = get_temp(s.temps, 2)
        exptU!(s.eQs[μ], 1, s.Qs[μ], temps)
        unused!(s.temps, it_s)
        #println(((length(Uout), length(s.eQs), length(Uin))))
        mul!(Uout[μ], s.eQs[μ], Uin[μ])
    end
    set_wing_U!(Uout)
    s.hasdSdCs[1] = false
    unused!(s.temps)
end
export forward!

function forward!(s::STOUT_Layer{T,Dim}, Uout, ρs::Vector{TN}, Uinα, Uinβ) where {T,Dim,TN<:Number} #Uout = exp(Q(Uin,ρs))*Uinα
    s.isαβsame = (Uinα == Uinβ)
    s.islocalρ = false
    #println("is $isαβsame")
    substitute_U!(s.Uin, Uinα)
    if s.isαβsame == false
        substitute_U!(s.Uinβ, Uinβ)
    end
    for i = 1:length(s.ρs)
        s.ρs[i] = deepcopy(ρs[i])
    end
    #temps = s.temps
    #Ω = temps[end]
    Ω, iΩ = get_temp(s.temps)
    temps, it_s = get_temp(s.temps, 2)

    for μ = 1:Dim
        calc_C!(s.Cs[μ], μ, ρs, s.dataset, Uinβ, s.temps)
        mul!(Ω, s.Cs[μ], Uinβ[μ]') #Ω = C*Udag
        Traceless_antihermitian!(s.Qs[μ], Ω)
        exptU!(s.eQs[μ], 1, s.Qs[μ], temps)
        mul!(Uout[μ], s.eQs[μ], Uinα[μ])
    end
    set_wing_U!(Uout)
    s.hasdSdCs .= false
    unused!(s.temps, iΩ)
    unused!(s.temps, it_s)
end






function backward_dSdU_add!(s::STOUT_Layer, dSdUin, dSdUout)
    backward_dSdUα_add!(s, dSdUin, dSdUout)
    backward_dSdUβ_add!(s, dSdUin, dSdUout)
end
export backward_dSdU_add!

function backward_dSdUαUβρ_add!(s::STOUT_Layer{T,Dim,TN}, dSdU, dSdρ, dSdUout) where {T,Dim,TN}
    @assert Dim == 4 "Dim = $Dim is not supported yet. Use Dim = 4"
    #temps = s.temps

    #temp1 = temps[1]
    #dng = 2
    #dSdQ = temps[2+dng]
    #dSdΩ = temps[3+dng]
    #dSdUdag = temps[4+dng]
    #dSdCs = temps[5+dng:5+Dim-1+dng]
    dSdCs, it_dSdCs = get_temp(s.temps, Dim)

    Uin = s.Uin



    for μ = 1:Dim

        #dS/dUα
        temp1, it_1 = get_temp(s.temps)
        calc_dSdu1!(temp1, dSdUout[μ], s.eQs[μ])
        add_U!(dSdU[μ], temp1)
        unused!(s.temps, it_1)

        #dS/dUβ
        Cμ = s.Cs[μ]
        Qμ = s.Qs[μ]

        #dSdQ = temps[2+dng]
        dSdQ, it_dSdQ = get_temp(s.temps)
        #dSdΩ = temps[3+dng]
        dSdΩ, it_dSdΩ = get_temp(s.temps)
        #println("dSdUout[μ]")
        #display(dSdUout[μ].U[:, :, 1, 1])
        #println("s.Uin[μ]")
        #display(s.Uin[μ].U[:, :, 1, 1])
        #println("Qμ")
        #display(Qμ.U[:, :, 1, 1])
        temp1, it_1 = get_temp(s.temps)
        calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, s.Uin[μ], temp1)
        unused!(s.temps, it_1)

        #println("dSdQ")
        #display(dSdQ.U[:, :, 1, 1])
        #unused!(s.temps, it_1)
        calc_dSdΩ!(dSdΩ, dSdQ)
        #println("dSdΩ")
        #display(dSdΩ.U[:, :, 1, 1])
        unused!(s.temps, it_dSdQ)
        calc_dSdC!(dSdCs[μ], dSdΩ, Uin[μ])
        #println("dSdCs[μ]")
        #display(dSdCs[μ].U[:, :, 1, 1])


        dSdUdag, it_dSdUdag = get_temp(s.temps)
        #dSdUdag = temps[4+dng]
        calc_dSdUdag!(dSdUdag, dSdΩ, Cμ)
        unused!(s.temps, it_dSdΩ)
        add_U!(dSdU[μ], dSdUdag')
        unused!(s.temps, it_dSdUdag)


        #Cμi = temps[4+dng] #dSdUdag
        Cμi, it_Cμi = get_temp(s.temps)
        #dS/dρ
        num = length(s.ρs)
        for i = 1:num
            loops = s.dataset[i].Cμ[μ]
            #Cμi = temps[6]
            temps, its_temps = get_temp(s.temps, 4)
            evaluate_gaugelinks!(Cμi, loops, Uin, temps)
            #temp1 = temps[1]
            #dSdCs = temps[7:7+Dim-1]
            temp1, it_1 = get_temp(s.temps)
            mul!(temp1, dSdCs[μ], Cμi)
            dSdρ[i] += real(tr(temp1)) * 2
            unused!(s.temps, it_1)
            unused!(s.temps, its_temps)
        end
        unused!(s.temps, it_Cμi)
    end



    for ν = 1:Dim
        for μ = 1:Dim
            calc_dSdUν_fromdSCμ_add!(dSdU[ν], s.dataset, dSdCs[μ], s.ρs, Uin, μ, ν, s.temps)
        end
    end
    unused!(s.temps)
end
export backward_dSdUαUβρ_add!

function backward_dSdUαUβρ_add!(s::STOUT_Layer{T,Dim}, dSdUα, dSdUβ, dSdρ, dSdUout) where {T,Dim}
    @assert Dim == 4 "Dim = $Dim is not supported yet. Use Dim = 4"
    temp1, itemp1 = get_temp(s.temps)
    dSdQ, idSdQ = get_temp(s.temps)
    dSdΩ, idSdΩ = get_temp(s.temps)
    dSdUdag, idSdUdag = get_temp(s.temps)
    dSdCs, idSdCs = get_temp(s.temps, Dim)
    temps, itemps = get_temp(s.temps, 4)

    #temps = s.temps
    #temp1 = temps[1]
    #dSdQ = temps[2]
    #dSdΩ = temps[3]
    #dSdUdag = temps[4]
    #dSdCs = temps[5:5+Dim-1]

    if s.isαβsame
        Uin = s.Uin
    else
        Uin = s.Uinβ
    end


    for μ = 1:Dim

        #dS/dUα
        calc_dSdu1!(temp1, dSdUout[μ], s.eQs[μ])
        add_U!(dSdUα[μ], temp1)

        #dS/dUβ
        Cμ = s.Cs[μ]
        Qμ = s.Qs[μ]

        calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, s.Uin[μ], temp1)
        calc_dSdΩ!(dSdΩ, dSdQ)
        calc_dSdC!(dSdCs[μ], dSdΩ, Uin[μ])

        calc_dSdUdag!(dSdUdag, dSdΩ, Cμ)
        add_U!(dSdUβ[μ], dSdUdag')

        if s.islocalρ == false
            Cμi, iCμi = get_temp(s.temps)

            #Cμi = temps[4] #dSdUdag
            #dS/dρ
            num = length(s.ρs)
            for i = 1:num
                loops = s.dataset[i].Cμ[μ]
                evaluate_gaugelinks!(Cμi, loops, Uin, temps)
                mul!(temp1, dSdCs[μ], Cμi)
                dSdρ[i] += real(tr(temp1)) * 2
            end
            unused!(s.temps, iCμi)
        else
            error("not supported yet")
        end

    end

    for ν = 1:Dim
        for μ = 1:Dim
            calc_dSdUν_fromdSCμ_add!(dSdUβ[ν], s.dataset, dSdCs[μ], s.ρs, Uin, μ, ν, s.temps)
        end
    end

    unused!(s.temps, itemp1)
    unused!(s.temps, idSdQ)
    unused!(s.temps, idSdΩ)
    unused!(s.temps, idSdCs)
    unused!(s.temps, idSdUdag)
    unused!(s.temps, itemps)



end


function backward_dSdρ_add!(s::STOUT_Layer{T,Dim,TN}, dSdρ, dSdUout) where {T,Dim,TN}
    @assert Dim == 4 "Dim = $Dim is not supported yet. Use Dim = 4"
    #temps = s.temps
    #temp1, it_temp1 = get_temp(s.temps)
    #temp1 = temps[1]
    #dSdQ = temps[2]
    #dSdQ, it_dSdQ = get_temp(s.temps)
    #dSdΩ = temps[3]
    #dSdΩ, it_dSdΩ = get_temp(s.temps)
    #Cμi = temps[4] #dSdUdag
    #Cμi, it_Cμi = get_temp(s.temps)


    Uin = s.Uin

    for μ = 1:Dim

        if s.hasdSdCs[1] == false
            Qμ = s.Qs[μ]
            temp1, it_temp1 = get_temp(s.temps)
            dSdQ, it_dSdQ = get_temp(s.temps)
            calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, Uin[μ], temp1)
            unused!(s.temps, it_temp1)
            dSdΩ, it_dSdΩ = get_temp(s.temps)
            calc_dSdΩ!(dSdΩ, dSdQ)
            unused!(s.temps, it_dSdQ)
            calc_dSdC!(s.dSdCs[μ], dSdΩ, Uin[μ])
            unused!(s.temps, it_dSdQ)
        end


        #dS/dρ
        num = length(s.ρs)
        for i = 1:num
            loops = s.dataset[i].Cμ[μ]
            #println(loops)
            temps, its_temps = get_temp(s.temps, 4)
            Cμi, it_Cμi = get_temp(s.temps)
            evaluate_gaugelinks!(Cμi, loops, Uin, temps)
            unused!(s.temps, its_temps)
            temp1, it_temp1 = get_temp(s.temps)
            mul!(temp1, s.dSdCs[μ], Cμi)
            dSdρ[i] += real(tr(temp1)) * 2

            unused!(s.temps, it_Cμi)
            unused!(s.temps, it_temp1)
        end
    end
    s.hasdSdCs[1] = true

    #unused!(temps)
end
export backward_dSdρ_add!



function backward_dSdUβ_add!(s::STOUT_Layer{T,Dim,TN}, dSdU, dSdUout) where {T,Dim,TN} # Uout =  exp(Q(Uin,ρs))*Uinα
    #temps = s.temps
    #temps = similar(s.temps)

    #temp1 = temps[1]
    #dSdQ = temps[2]
    #dSdΩ = temps[3]
    #dSdUdag = temps[4]

    Uin = s.Uin


    ρs = s.ρs
    #println(ρs)
    dataset = s.dataset
    for μ = 1:Dim


        Cμ = s.Cs[μ]
        Qμ = s.Qs[μ]
        temp1, it_temps = get_temp(s.temps)
        dSdQ, it_dSdQ = get_temp(s.temps)
        calc_dSdQ!(dSdQ, dSdUout[μ], Qμ, Uin[μ], temp1)
        unused!(s.temps, it_temps)
        dSdΩ, it_dSdΩ = get_temp(s.temps)
        calc_dSdΩ!(dSdΩ, dSdQ)
        unused!(s.temps, it_dSdQ)

        calc_dSdC!(s.dSdCs[μ], dSdΩ, Uin[μ])
        dSdUdag, it_dSdUdag = get_temp(s.temps)
        calc_dSdUdag!(dSdUdag, dSdΩ, Cμ)
        unused!(s.temps, it_dSdΩ)
        add_U!(dSdU[μ], dSdUdag')
        unused!(s.temps, it_dSdUdag)
    end

    for ν = 1:Dim
        for μ = 1:Dim
            calc_dSdUν_fromdSCμ_add!(dSdU[ν], s.dataset, s.dSdCs[μ], s.ρs, Uin, μ, ν, s.temps)
        end
    end
    s.hasdSdCs[1] = true
end
export backward_dSdUβ_add!


function backward_dSdUα_add!(s::STOUT_Layer{T,Dim,TN}, dSdU, dSdUout) where {T,Dim,TN}
    #temps = s.temps
    #temps = similar(s.temps)
    #temp1 = temps[1]
    temp1, it_temp1 = get_temp(s.temps)

    for μ = 1:Dim
        calc_dSdu1!(temp1, dSdUout[μ], s.eQs[μ])
        add_U!(dSdU[μ], temp1)
    end
    unused!(s.temps, it_temp1)
end
export backward_dSdUα_add!

function calc_C!(C, μ, ρs::Vector{TN}, dataset::Vector{STOUT_dataset{Dim}}, Uin, temps_g) where {Dim,TN<:Number}
    #temp1 = temps_g[1]
    #temp2 = temps_g[2]
    #temp3 = temps_g[3]
    temp3, it_temp3 = get_temp(temps_g)
    #temp3 = temps_g[5]
    #vec_temps = temps_g[1:4]
    vec_temps, its_vec_temps = get_temp(temps_g, 4)
    num = length(ρs)
    clear_U!(C)
    for i = 1:num
        #println("ρi  = ",ρs[i] )
        loops = dataset[i].Cμ[μ]
        evaluate_gaugelinks!(temp3, loops, Uin, vec_temps)
        #println("i = $i")
        #println(temp3[1,1,1,1,1,1])
        add_U!(C, ρs[i], temp3)
    end
    unused!(temps_g, it_temp3)
    unused!(temps_g, its_vec_temps)
    #println("U ", Uin[1][1,1,1,1,1,1])
end
export calc_C!



function calc_dSdu1!(dSdu1, dSdUbar, expQ) # Ubar = exp(Q)*U
    mul!(dSdu1, dSdUbar, expQ)
end
export calc_dSdu1!

import ..AbstractGaugefields_module: Initialize_Gaugefields

function calc_dSdQ!(dSdQ, dSdUbar, Qμ, Uμ, temp)
    dSdUU = temp
    mul!(dSdUU, Uμ, dSdUbar)

    #=
    Qμcpu = Initialize_Gaugefields(Qμ.NC, 0, Qμ.NX, Qμ.NY, Qμ.NZ, Qμ.NT, condition="hot")[1]
    Uμcpu = similar(Qμcpu)
    dSdUUcpu = similar(Qμcpu)
    dSdQcpu = similar(Qμcpu)

    substitute_U!(Qμcpu, Qμ)
    substitute_U!(Uμcpu, Uμ)
    substitute_U!(dSdUUcpu, dSdUU)
    substitute_U!(dSdQcpu, dSdQ)

    temptemp = similar(dSdQ)
    clear_U!(temptemp)
    temptempcpu = similar(Qμcpu)

    println("Q,UU,Umu,dSdQ")
    display(Qμ.U[:, :, 1, 1])
    display(dSdUU.U[:, :, 1, 1])
    display(Uμ.U[:, :, 1, 1])
    display(dSdQ.U[:, :, 1, 1])
    =#

    CdexpQdQ!(dSdQ, dSdUU, Qμ)
    #=
    CdexpQdQ!(temptemp, dSdUU, Qμ)
    println("cuda")
    display(dSdQ.U[:, :, 1, 1])
    display(temptemp.U[:, :, 1, 1])
    CdexpQdQ!(dSdQcpu, dSdUUcpu, Qμcpu)
    CdexpQdQ!(temptempcpu, dSdUUcpu, Qμcpu)
    println("cpu")
    display(dSdQcpu[:, :, 1, 1, 1, 1])
    display(temptempcpu[:, :, 1, 1, 1, 1])
    #error("d")
    =#
end
export calc_dSdQ!

function LdQdΩ!(LdQdΩ, L) # L star dQ/dΩ
    Traceless_antihermitian!(LdQdΩ, L)
end

function calc_dSdΩ!(dSdΩ, dSdQ)
    LdQdΩ!(dSdΩ, dSdQ)
end
export calc_dSdΩ!

function calc_dSdC!(dSdC, dSdΩ, Uμ)
    mul!(dSdC, Uμ', dSdΩ)
end
export calc_dSdC!

function calc_dSdUdag!(dSdUdag, dSdΩ, C)
    mul!(dSdUdag, dSdΩ, C)
end
export calc_dSdUdag!



function LdCdU_i_add!(LdCdU, L, A, B, ρ, temps_g) #dCdU = ρ sum_i A_i otimes B_i , dCdagdU = ρ sum_i Abar_i otimes Bbar_i
    #BL = temps_g[1]
    BL, it_BL = get_temp(temps_g)
    #BLA = temps_g[2]
    BLA, it_BLA = get_temp(temps_g)
    mul!(BL, B, L)
    mul!(BLA, BL, A)
    add_U!(LdCdU, ρ, BLA)
    unused!(temps_g, it_BL)
    unused!(temps_g, it_BLA)
end

export calc_dSdUν_fromdSCμ_add!
function calc_dSdUν_fromdSCμ_add!(dSdU, dataset::Vector{STOUT_dataset{Dim}}, dSdCμ, ρs, Us, μ, ν, temps_g) where {Dim}  #use pullback for C(U): dS/dCμ star dCμ/dUν
    #temp1 = temps_g[1]
    #temp2 = temps_g[2]
    dng = 2
    #temp3 = temps_g[3+dng]
    #temp4 = temps_g[4+dng]

    numterms = length(ρs)
    for iterm = 1:numterms
        dCμdUν = dataset[iterm].dCμdUν
        dCμdagdUν = dataset[iterm].dCμdagdUν
        ρi = ρs[iterm]

        #for μ = 1:Dim
        numdCμdUν = length(dCμdUν[μ, ν])
        for j = 1:numdCμdUν
            dCμdUν_j = dCμdUν[μ, ν][j]
            position = dCμdUν_j.position
            m = Tuple(-collect(position))
            dSdCμm = shift_U(dSdCμ, m)

            leftlinks = get_leftlinks(dCμdUν_j)
            rightlinks = get_rightlinks(dCμdUν_j)

            #A = temp3
            A, it_A = get_temp(temps_g)
            #A = temps_g[3+dng]
            temps, its_temps = get_temp(temps_g, 4)
            evaluate_gaugelinks!(A, leftlinks, Us, temps)
            unused!(temps_g, its_temps)

            #B = temp4
            #B = temps_g[4+dng]
            B, it_B = get_temp(temps_g)
            temps, its_temps = get_temp(temps_g, 4)
            evaluate_gaugelinks!(B, rightlinks, Us, temps)
            unused!(temps_g, its_temps)
            LdCdU_i_add!(dSdU, dSdCμm, A, B, ρi, temps_g)
            #unused!(temps_g, its_temps)
            unused!(temps_g, it_A)
            unused!(temps_g, it_B)
        end

        numdCμdagdUν = length(dCμdagdUν[μ, ν])
        for j = 1:numdCμdagdUν
            dCμdagdUν_j = dCμdagdUν[μ, ν][j]

            position = dCμdagdUν_j.position
            m = Tuple(-collect(position))
            dSdCμm = shift_U(dSdCμ, m)
            leftlinks = get_leftlinks(dCμdagdUν_j)
            rightlinks = get_rightlinks(dCμdagdUν_j)

            #barA = temp3
            #barA = temps_g[3+dng]
            barA, it_barA = get_temp(temps_g)
            temps, its_temps = get_temp(temps_g, 4)
            evaluate_gaugelinks!(barA, leftlinks, Us, temps)
            unused!(temps_g, its_temps)
            #barB = temp4
            #barB = temps_g[4+dng]
            barB, it_barB = get_temp(temps_g)
            temps, its_temps = get_temp(temps_g, 4)
            evaluate_gaugelinks!(barB, rightlinks, Us, temps)
            unused!(temps_g, its_temps)
            LdCdU_i_add!(dSdU, dSdCμm', barA, barB, ρi, temps_g)

            unused!(temps_g, it_barA)
            unused!(temps_g, it_barB)
        end
        #end

    end
end

function calc_dSdΩ(Ω, Uμ, ix, iy, iz, it, eta=1e-12; NC=3)
    numg = 5
    temps_g = Vector{typeof(Ω)}(undef, numg)
    for i = 1:numg
        temps_g[i] = similar(Ω)
    end
    Qμ = similar(Ω)
    Traceless_antihermitian!(Qμ, Ω)
    eQ = similar(Qμ)
    barU = similar(Qμ)
    eQ = expU(Qμ)
    mul!(barU, eQ, Uμ)
    S = real(tr(barU * barU))

    dSdΩ_n = zeros(ComplexF64, NC, NC)
    Qd = similar(Qμ)
    for ic = 1:NC
        for jc = 1:NC
            Ωd = deepcopy(Ω)
            Ωd[jc, ic, ix, iy, iz, it] += eta

            Traceless_antihermitian!(Qd, Ωd)
            eQ = expU(Qd)
            mul!(barU, eQ, Uμ)
            Sd = real(tr(barU * barU))


            Ωd = deepcopy(Ω)
            Ωd[jc, ic, ix, iy, iz, it] += im * eta
            Traceless_antihermitian!(Qd, Ωd)
            eQ = expU(Qd)
            mul!(barU, eQ, Uμ)
            Sdm = real(tr(barU * barU))

            fg_n = (Sd - S) / eta
            fg_n_im = (Sdm - S) / eta
            dSdΩ_n[ic, jc] = (fg_n - im * fg_n_im) / 2

        end
    end
    return dSdΩ_n
end




function calc_dSdQ(Q, Uμ, ix, iy, iz, it, eta=1e-12; NC=3)

    numg = 5
    temps_g = Vector{typeof(Uμ)}(undef, numg)
    for i = 1:numg
        temps_g[i] = similar(Q)
    end
    #eta = 1e-14
    eQ = similar(Q)
    barU = similar(Q)
    eQ = expU(Q)
    mul!(barU, eQ, Uμ)

    S = real(tr(barU * barU))


    dSdQ_n = zeros(ComplexF64, NC, NC)
    for ic = 1:NC
        for jc = 1:NC


            #S = real(tr(barU * barU))

            Qd = deepcopy(Q)
            Qd[jc, ic, ix, iy, iz, it] += eta
            eQ = expU(Qd)
            #Gaugefields.exptU!(eQ, 1, Qd, temps_g[1:2]) # exp(Q)
            #display(eQ[:, :, ix, iy, iz, it])
            mul!(barU, eQ, Uμ)
            Sd = real(tr(barU * barU))
            #Sd = real(tr(barU))

            #Sd = real(tr(eQ))
            #Sd = real(tr(eQ * eQ))


            Qd = deepcopy(Q)
            Qd[jc, ic, ix, iy, iz, it] += im * eta
            eQ = expU(Qd)
            #Gaugefields.exptU!(eQ, 1, Qd, temps_g[1:2]) # exp(Q)
            mul!(barU, eQ, Uμ)
            Sdm = real(tr(barU * barU))
            #Sdm = real(tr(eQ * eQ))
            #Sdm = real(tr(barU))
            #Sdm = real(tr(eQ))

            fg_n = (Sd - S) / eta
            fg_n_im = (Sdm - S) / eta
            dSdQ_n[ic, jc] = (fg_n - im * fg_n_im) / 2
        end
    end
    return dSdQ_n

end

function CdexpQdQ!(CdeQdQ::Union{Gaugefields_4D_nowing{3},Gaugefields_4D_wing{3}}, C::Union{Gaugefields_4D_nowing{3},Gaugefields_4D_wing{3}},
    Q::Union{Gaugefields_4D_nowing{3},Gaugefields_4D_wing{3}}; eps_Q=1e-18) # C star dexpQ/dQ
    NT = Q.NT
    NY = Q.NY
    NZ = Q.NZ
    NX = Q.NX
    NC = 3
    Qnim = zeros(ComplexF64, NC, NC) #Qn/im
    B1 = zero(Qnim)
    B2 = zero(Qnim)
    Cn = zero(Qnim)
    CdeQdQn = zero(Qnim)

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX

                    trQ2 = 0.0im
                    for i = 1:3
                        for j = 1:3
                            trQ2 += Q[i, j, ix, iy, iz, it] * Q[j, i, ix, iy, iz, it]
                        end
                    end

                    if abs(trQ2) > eps_Q
                        for jc = 1:NC
                            for ic = 1:NC
                                Qnim[ic, jc] = Q[ic, jc, ix, iy, iz, it] / im
                                Cn[ic, jc] = C[ic, jc, ix, iy, iz, it]
                            end
                        end
                        f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qnim)
                        #if ix == iy == iz == it == 1
                        #    println((f0, f1, f2, b10, b11, b12, b20, b21, b22))
                        #end

                        construct_B1B2!(B1, B2, Qnim, b10, b11, b12, b20, b21, b22)
                        trCB1, trCB2 = construct_trCB1B2(B1, B2, Cn)
                        construct_CdeQdQ_3!(CdeQdQn, trCB1, trCB2, f1, f2, Qnim, Cn)

                        for jc = 1:NC
                            for ic = 1:NC
                                CdeQdQ[ic, jc, ix, iy, iz, it] = CdeQdQn[ic, jc]
                            end
                        end
                    else
                        for jc = 1:NC
                            for ic = 1:NC
                                #CdeQdQ[ic, jc, ix, iy, iz, it] = C[ic, jc, ix, iy, iz, it]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(CdeQdQ)
end
function CdexpQdQ!(CdeQdQ::Union{Gaugefields_4D_nowing{2},Gaugefields_4D_wing{2}}, C::Union{Gaugefields_4D_nowing{2},Gaugefields_4D_wing{2}},
    Q::Union{Gaugefields_4D_nowing{2},Gaugefields_4D_wing{2}}; eps_Q=1e-18) # C star dexpQ/dQ
    NT = Q.NT
    NY = Q.NY
    NZ = Q.NZ
    NX = Q.NX
    NC = 2
    Qn = zeros(ComplexF64, NC, NC) #Qn
    B = zero(Qn)
    B2 = zero(Qn)
    Cn = zero(Qn)
    CdeQdQn = zero(Qn)


    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX

                    trQ2 = 0.0im
                    for i = 1:2
                        for j = 1:2
                            trQ2 += Q[i, j, ix, iy, iz, it] * Q[j, i, ix, iy, iz, it]
                        end
                    end


                    if abs(trQ2) > eps_Q
                        q = sqrt((-1 / 2) * trQ2)
                        for jc = 1:NC
                            for ic = 1:NC
                                Qn[ic, jc] = Q[ic, jc, ix, iy, iz, it]
                                Cn[ic, jc] = C[ic, jc, ix, iy, iz, it]
                            end
                        end
                        calc_Bmatrix!(B, q, Qn, NC)
                        trsum = 0.0im
                        for i = 1:2
                            for j = 1:2
                                trsum += Cn[i, j] * B[j, i]
                            end
                        end
                        for i = 1:2
                            for j = 1:2
                                CdeQdQn[j, i] = (sin(q) / q) * Cn[j, i] + trsum * Qn[j, i]
                            end
                        end

                        for jc = 1:NC
                            for ic = 1:NC
                                CdeQdQ[ic, jc, ix, iy, iz, it] = CdeQdQn[ic, jc]
                            end
                        end
                    else
                        for jc = 1:NC
                            for ic = 1:NC
                                #CdeQdQ[ic, jc, ix, iy, iz, it] = C[ic, jc, ix, iy, iz, it]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(CdeQdQ)
end

export CdexpQdQ!


function CdexpQdQ!(CdeQdQ, C, Q)
    error("CdexpQdQ! is not implemented in types of CdeQdQ, C, Q: $(typeof(CdeQdQ)) $(typeof(C)) $(typeof(Q)) ")
end
function construct_B1B2!(B1, B2, Qn, b10, b11, b12, b20, b21, b22)
    B1 .= 0
    B2 .= 0
    for i = 1:3
        B1[i, i] = b10
        B2[i, i] = b20
    end
    for j = 1:3
        for i = 1:3
            B1[i, j] += b11 * Qn[i, j]
            B2[i, j] += b21 * Qn[i, j]
            for k = 1:3
                B1[i, j] += b12 * Qn[i, k] * Qn[k, j]
                B2[i, j] += b22 * Qn[i, k] * Qn[k, j]
            end
        end
    end
end
function construct_trCB1B2(B1, B2, C)
    trB1 = 0.0im
    trB2 = 0.0im
    for i = 1:3
        for j = 1:3
            trB1 += C[i, j] * B1[j, i]
            trB2 += C[i, j] * B2[j, i]
        end
    end
    return trB1, trB2
end

function construct_CdeQdQ_3!(CdeQdQn, trCB1, trCB2, f1, f2, Qn, Cn)
    for j = 1:3
        for i = 1:3
            CdeQdQn[i, j] = trCB1 * Qn[i, j] + f1 * Cn[i, j]
            for k = 1:3
                CdeQdQn[i, j] +=
                    trCB2 * Qn[i, k] * Qn[k, j] +
                    f2 * (Qn[i, k] * Cn[k, j] + Cn[i, k] * Qn[k, j])
            end
            #CdeQdQn[i, j] /= im
        end
    end

    for j = 1:3
        for i = 1:3
            CdeQdQn[i, j] /= im
        end
    end
    #CdeQdQn ./= im
    return
end
