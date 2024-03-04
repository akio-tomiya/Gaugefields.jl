
include("./forbackprop.jl")


struct STOUT_Layer_b{Dim} <: CovLayer{Dim}
    ρs::Vector{Float64}
    dataset::Vector{STOUT_dataset{Dim}}
end

function get_name(s::STOUT_Layer_b)
    return "STOUT"
end

function Base.show(s::STOUT_Layer_b{Dim}) where {Dim}
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

function set_parameters(s::STOUT_Layer_b, ρs)
    s.ρs[:] .= ρs
end

function Base.length(layer::STOUT_Layer_b)
    return length(layer.ρs)
end

function get_Cμ(layer::STOUT_Layer_b, i)
    return layer.dataset[i].Cμ
end

function get_dCμdUν(layer::STOUT_Layer_b, i)
    return layer.dataset[i].dCμdUν
end

function get_dCμdagdUν(layer::STOUT_Layer_b, i)
    return layer.dataset[i].dCμdagdUν
end

function get_ρ(layer::STOUT_Layer_b, i)
    return layer.ρs[i]
end




function CovNeuralnet_STOUT_b(loops_smearing, ρs, L; Dim=4)

    numlayers = length(ρs)
    layers = Array{CovLayer,1}(undef, numlayers)
    for i = 1:numlayers
        stout_layer = STOUT_Layer_b(loops_smearing, ρs[i], L, Dim=Dim)
        layers[i] = stout_layer
    end

    return CovNeuralnet{Dim}(numlayers, layers)#,_temp_gaugefields,_temp_TA_gaugefields)

end

#=
closedloops -> sum_{nm} P_nm
Cmu = dP_nm/dUmu
dCmu/dUnu = (d/dUnu) dP_nm/dUmu
=#



function STOUT_Layer_b(loops_smearing, ρ, L; Dim=4)
    num = length(loops_smearing)
    loopset = make_loopforactions(loops_smearing, L)
    dataset = Array{STOUT_dataset{Dim},1}(undef, num)
    for i = 1:num
        closedloops = loopset[i] #one of loopset, like plaq. There are several loops. 
        dataset[i] = STOUT_dataset(closedloops, Dim=Dim)
    end

    return STOUT_Layer_b{Dim}(ρ, dataset)
end

"""
δ_prev = δ_current*exp(Q) - C^+ Λ 
        + sum_{i} sum_{μ',m}  [B^i_{μ,m} U_{μ'}^+ Λ_{μ',m} A_{μ,m} - bar{B}_{μ,m} Λ_{μ',m}U_{μ',m} bar{A}_{μ,m} ]

"""
function layer_pullback!(
    δ_prev::Array{<:AbstractGaugefields{NC,Dim},1},
    δ_current,
    layer::STOUT_Layer_b{Dim},
    Uprev,
    temps,
    tempf,
) where {NC,Dim}
    clear_U!(δ_prev)

    dSdQ = similar(Uprev[1])
    dSdΩ = similar(Uprev[1])
    dSdCs = similar(Uprev)
    dSdUdag = similar(Uprev[1])


    dataset = layer.dataset
    Cμ = similar(Uprev[1])
    Qμ = similar(Uprev[1])
    eQ = similar(Uprev[1])
    Ω = similar(Uprev[1])
    ρs = layer.ρs
    for μ = 1:Dim
        calc_C!(Cμ, μ, ρs, dataset, Uprev, temps)
        mul!(Ω, Cμ, Uprev[μ]') #Ω = C*Udag
        Traceless_antihermitian!(Qμ, Ω)
        exptU!(eQ, 1, Qμ, temps[1:2])

        calc_dSdu1!(δ_prev[μ], δ_current[μ], eQ)

        calc_dSdQ!(dSdQ, δ_current[μ], Qμ, Uprev[μ], temps[1])
        calc_dSdΩ!(dSdΩ, dSdQ)
        calc_dSdC!(dSdCs[μ], dSdΩ, Uprev[μ])
        calc_dSdUdag!(dSdUdag, dSdΩ, Cμ)
        add_U!(δ_prev[μ], dSdUdag')
    end


    for ν = 1:Dim
        for μ = 1:Dim
            calc_dSdUν_fromdSCμ_add!(δ_prev[ν], dataset, dSdCs[μ], ρs, Uprev, μ, ν, temps)
        end
    end

    set_wing_U!(δ_prev)
    #error("stout")


end

function parameter_derivatives(
    δ_current,
    layer::STOUT_Layer_b{Dim},
    U_current,
    temps,
) where {Dim}
    #δ_prev[ν](n) = δ_current[ν](n)*exp(Qν[Uprev](n)) + F(δ_current,Uprev)
    #F(δ_current,Uprev) = sum_μ sum_m Fm[μ](δ_current,Uprev)
    #δ_prev[ν](n) = dS/dU[ν](n)

    Cμs = similar(δ_current)
    construct_Cμ!(Cμs, layer, U_current, temps)

    Qμs = similar(U_current)

    #F0 = tempf[1]
    #construct_Qμs!(F0,Cμs,Uprev,temps)
    #substitute_U!(Qμs,F0)


    construct_Qμs!(Qμs, Cμs, U_current, temps)
    Λs = similar(U_current)
    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]
    temp4 = temps[4]

    for μ = 1:Dim
        construct_Λmatrix_forSTOUT!(Λs[μ], δ_current[μ], Qμs[μ], U_current[μ])
    end


    #error("lambda!")

    numterms = length(layer)
    dSdρ = zeros(Float64, numterms)

    for i = 1:numterms
        C = get_Cμ(layer, i)
        s = 0.0
        for μ = 1:Dim
            Λμ = Λs[μ]
            Uμ = U_current[μ]
            Cμ = C[μ]

            dCμdρ = temp3
            evaluate_gaugelinks!(dCμdρ, Cμ, U_current, [temp1, temp2])
            #Udag Λ dCμdρ
            mul!(temp1, Λμ, dCμdρ)
            mul!(temp2, Uμ', temp1)
            s += 2 * real(tr(temp2))
        end
        dSdρ[i] = s
    end
    return dSdρ

end

"""
M = U δ star dexp(Q)/dQ
"""



function apply_layer!(
    Uout::Array{<:AbstractGaugefields{NC,Dim},1},
    layer::STOUT_Layer_b{Dim},
    Uin,
    temps,
    tempf,
) where {NC,Dim}
    Cμs = similar(Uin)
    #construct_Cμ!(Cμs, layer, Uin, temps)

    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]
    Qμ = tempf[1]
    #Cμ = temp3
    Ω = temp3
    ρs = layer.ρs
    dataset = layer.dataset

    for μ = 1:Dim
        calc_C!(Cμs[μ], μ, ρs, dataset, Uin, temps)
        mul!(Ω, Cμs[μ], Uin[μ]') #Ω = C*Udag
        Traceless_antihermitian!(Qμ, Ω)
        exptU!(temp3, 1, Qμ, temps[1:2])


        #construct_Qμ!(Qμ, μ, Cμs, Uin, temps)
        # mul!(temp1,Cμs[μ],Uin[μ]') #Cμ*U^+
        #clear_U!(F0)
        #Traceless_antihermitian_add!(F0,1,temp1)

        #exptU!(temp3, 1, Qμ, [temp1, temp2])
        mul!(Uout[μ], temp3, Uin[μ])
    end
    set_wing_U!(Uout)

    #=


    Cμ  = temps[4]
    temp1  = temps[1]
    temp2  = temps[2]
    temp3  = temps[3]

    F0 = tempf[1]
    ρs = layer.ρs

    num = length(ρs)

    for μ=1:Dim
        clear_U!(Cμ)
        for i=1:num
            loops = layer.dataset[i].Cμ[μ]
            evaluate_gaugelinks!(temp3,loops,Uin,[temp1,temp2])
            add_U!(Cμ,ρs[i],temp3)
        end
        mul!(temp1,Cμ,Uin[μ]') #Cμ*U^+
        clear_U!(F0)
        Traceless_antihermitian_add!(F0,1,temp1)

        exptU!(temp3,1,F0,[temp1,temp2])

        mul!(Uout[μ],temp3,Uin[μ])        
    end
    set_wing_U!(Uout)
    =#

end



