mutable struct Stoutsmearing <: Abstractsmearing
    staples_for_stout::Array{Array{Wilson_loop_set,1},1}
    tensor_derivative::Array{Tensor_derivative_set,1}
    staples_for_stout_dag::Array{Array{Wilson_loop_set,1},1}
    tensor_derivative_dag::Array{Tensor_derivative_set,1}
    ρs::Array{Array{Float64,1},1}
end

#=
struct STOUT_dataset{Dim}
    closedloop::Vector{Wilsonline{Dim}}
    Cμ::Vector{Vector{Wilsonline{Dim}}}
    dCμdUν::Array{Array{Dict{Tuple{Int8,Int8},Vector{DwDU{Dim}}},1},1}#Matrix{Vector{DwDU{Dim}}} #(mu,nu) num loops
    dCμdUνdag::Array{Array{Dict{Tuple{Int8,Int8},Vector{DwDU{Dim}}},1},1}#Matrix{Vector{DwDU{Dim}}} #(mu,nu) num loops
    #dCmudUnu::Dict{Tuple{Int64,Int64,Int64},Array{DwDU{Dim},1}}
end
=#

include("stout_dataset.jl")

#=
struct STOUT_Layer{Dim} <: CovLayer{Dim}
    ρs::Vector{Float64}
    dataset::Vector{STOUT_dataset{Dim}}
end

function get_name(s::STOUT_Layer)
    return "STOUT"
end

function Base.show(s::STOUT_Layer{Dim}) where {Dim}
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

function set_parameters(s::STOUT_Layer, ρs)
    s.ρs[:] .= ρs
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

=#
#=



function CovNeuralnet_STOUT(loops_smearing, ρs, L; Dim=4)

    numlayers = length(ρs)
    layers = Array{CovLayer,1}(undef, numlayers)
    for i = 1:numlayers
        stout_layer = STOUT_Layer(loops_smearing, ρs[i], L, Dim=Dim)
        layers[i] = stout_layer
    end

    return CovNeuralnet{Dim}(numlayers, layers)#,_temp_gaugefields,_temp_TA_gaugefields)

end

=#

#=
function STOUT_Layer(loops_smearing, ρ, L; Dim=4)
    num = length(loops_smearing)
    loopset = make_loopforactions(loops_smearing, L)

    dataset = Array{STOUT_dataset{Dim},1}(undef, num)
    for i = 1:num
        closedloops = loopset[i] #one of loopset, like plaq. There are several loops. 
        dataset[i] = STOUT_dataset(closedloops, Dim=Dim)
    end

    return STOUT_Layer{Dim}(ρ, dataset)
end

function STOUT_Layer(loops, ρ; Dim=4)
    num = length(loops_smearing)
    loopset = make_loopforactions(loops_smearing, L)
    dataset = Array{STOUT_dataset{Dim},1}(undef, num)
    for i = 1:num
        closedloops = loopset[i] #one of loopset, like plaq. There are several loops. 
        dataset[i] = STOUT_dataset(closedloops, Dim=Dim)
    end

    return STOUT_Layer{Dim}(ρ, dataset)
end

=#

#=

"""
δ_prev = δ_current*exp(Q) - C^+ Λ 
        + sum_{i} sum_{μ',m}  [B^i_{μ,m} U_{μ'}^+ Λ_{μ',m} A_{μ,m} - bar{B}_{μ,m} Λ_{μ',m}U_{μ',m} bar{A}_{μ,m} ]

"""
function layer_pullback!(
    δ_prev::Array{<:AbstractGaugefields{NC,Dim},1},
    δ_current,
    layer::STOUT_Layer{Dim},
    Uprev,
    temps,
    tempf,
) where {NC,Dim}
    clear_U!(δ_prev)
    #δ_prev[ν](n) = δ_current[ν](n)*exp(Qν[Uprev](n)) + F(δ_current,Uprev)
    #F(δ_current,Uprev) = sum_μ sum_m Fm[μ](δ_current,Uprev)
    #δ_prev[ν](n) = dS/dU[ν](n)

    Cμs = similar(Uprev)
    construct_Cμ!(Cμs, layer, Uprev, temps)

    Qμs = similar(Uprev)

    #F0 = tempf[1]
    #construct_Qμs!(F0,Cμs,Uprev,temps)
    #substitute_U!(Qμs,F0)


    construct_Qμs!(Qμs, Cμs, Uprev, temps)
    Λs = similar(Uprev)
    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]
    temp4 = temps[4]

    for μ = 1:Dim
        construct_Λmatrix_forSTOUT!(Λs[μ], δ_current[μ], Qμs[μ], Uprev[μ])

        #println(Λs[μ][1,1,1,1,1,1])


        exptU!(temp3, 1, Qμs[μ], [temp1, temp2]) # exp(Q)
        #println("Q ",Qμs[μ][:,:,1,1,1,1])
        #println("QdagQ ",Qμs[μ][:,:,1,1,1,1]'*Qμs[μ][:,:,1,1,1,1])
        #println("expQ ",temp3[1,1,1,1,1,1])
        mul!(δ_prev[μ], δ_current[μ], temp3) #δ_prev  =δ_current*exp(Q)
        #println(δ_prev[μ][1,1,1,1,1,1])

        mul!(temp1, Cμs[μ]', Λs[μ])
        #println("temp1 ",temp1[1,1,1,1,1,1])
        add_U!(δ_prev[μ], -1, temp1) #δ_prev += -C'Λ
        #println(δ_prev[μ][1,1,1,1,1,1])

    end


    #error("lambda!")



    numterms = length(layer)


    for i = 1:numterms
        dCμdUν = get_dCμdUν(layer, i)
        dCμdagdUν = get_dCμdagdUν(layer, i)
        ρi = get_ρ(layer, i)
        #println("ρi  = ",ρi )



        for ν = 1:Dim
            for μ = 1:Dim
                Λμ = Λs[μ]
                Uμ = Uprev[μ]



                numdCμdUν = length(dCμdUν[μ, ν])

                for j = 1:numdCμdUν
                    dCμdUν_j = dCμdUν[μ, ν][j]
                    position = dCμdUν_j.position
                    m = Tuple(-collect(position))
                    Λμm = shift_U(Λμ, m)
                    Uμm = shift_U(Uμ, m)


                    leftlinks = get_leftlinks(dCμdUν_j)
                    rightlinks = get_rightlinks(dCμdUν_j)

                    A = temp3
                    evaluate_gaugelinks!(A, leftlinks, Uprev, [temp1, temp2])

                    B = temp4
                    evaluate_gaugelinks!(B, rightlinks, Uprev, [temp1, temp2])

                    #B Udag Λ A
                    mul!(temp1, Λμm, A)
                    mul!(temp2, Uμm', temp1)
                    mul!(temp1, B, temp2)
                    add_U!(δ_prev[ν], ρi, temp1)


                end




                numdCμdagdUν = length(dCμdagdUν[μ, ν])
                for j = 1:numdCμdagdUν
                    dCμdagdUν_j = dCμdagdUν[μ, ν][j]

                    position = dCμdagdUν_j.position
                    m = Tuple(-collect(position))
                    Uμm = shift_U(Uμ, m)
                    Λμm = shift_U(Λμ, m)


                    leftlinks = get_leftlinks(dCμdagdUν_j)
                    rightlinks = get_rightlinks(dCμdagdUν_j)

                    barA = temp3
                    evaluate_gaugelinks!(barA, leftlinks, Uprev, [temp1, temp2])
                    barB = temp4
                    evaluate_gaugelinks!(barB, rightlinks, Uprev, [temp1, temp2])

                    #-B Λ U A
                    mul!(temp1, Uμm, barA)
                    mul!(temp2, Λμm, temp1)
                    mul!(temp1, barB, temp2)

                    add_U!(δ_prev[ν], -ρi, temp1)
                end
            end
        end
    end

    set_wing_U!(δ_prev)
    #error("stout")


end

=#

#=

function parameter_derivatives(
    δ_current,
    layer::STOUT_Layer{Dim},
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



=#

function construct_Qμ!(
    Qμ,
    μ,
    Cμs,
    Uin::Array{<:AbstractGaugefields{NC,Dim},1},
    temps,
) where {NC,Dim}
    temp1 = temps[1]
    mul!(temp1, Cμs[μ], Uin[μ]')
    clear_U!(Qμ)
    Traceless_antihermitian!(Qμ, temp1)

end


function construct_Qμs!(
    Qμs,
    Cμs,
    Uin::Array{<:AbstractGaugefields{NC,Dim},1},
    temps,
) where {NC,Dim}
    for μ = 1:Dim
        construct_Qμ!(Qμs[μ], μ, Cμs, Uin, temps)
    end
end

#=

function construct_Cμ!(
    Cμs,
    layer::STOUT_Layer{Dim},
    Uin::Array{<:AbstractGaugefields{NC,Dim},1},
    temps,
) where {NC,Dim}
    #println_verbose_level3(Uin[1],"construct_Cμ!")
    ρs = layer.ρs
    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]
    num = length(ρs)

    for μ = 1:Dim
        clear_U!(Cμs[μ])
        for i = 1:num
            #println("ρi  = ",ρs[i] )
            loops = layer.dataset[i].Cμ[μ]
            evaluate_gaugelinks!(temp3, loops, Uin, [temp1, temp2])
            #println("i = $i")
            #println(temp3[1,1,1,1,1,1])
            add_U!(Cμs[μ], ρs[i], temp3)
        end
        #println("U ", Uin[1][1,1,1,1,1,1])
        #println("C[1,1] = ",Cμs[μ][1,1,1,1,1,1])

    end
    #error("dd")

end

function apply_layer!(
    Uout::Array{<:AbstractGaugefields{NC,Dim},1},
    layer::STOUT_Layer{Dim},
    Uin,
    temps,
    tempf,
) where {NC,Dim}
    Cμs = similar(Uin)
    construct_Cμ!(Cμs, layer, Uin, temps)

    temp1 = temps[1]
    temp2 = temps[2]
    temp3 = temps[3]
    Qμ = tempf[1]

    for μ = 1:Dim
        construct_Qμ!(Qμ, μ, Cμs, Uin, temps)
        # mul!(temp1,Cμs[μ],Uin[μ]') #Cμ*U^+
        #clear_U!(F0)
        #Traceless_antihermitian_add!(F0,1,temp1)

        exptU!(temp3, 1, Qμ, [temp1, temp2])
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

=#


function Stoutsmearing(loops_smearing, ρs)
    numloops = length(loops_smearing)
    staplesforsmear_set, tensor_derivative, staplesforsmear_dag_set, tensor_derivative_dag =
        make_loops_Stoutsmearing(loops_smearing, rand(numloops))
    return Stoutsmearing(
        staplesforsmear_set,
        tensor_derivative,
        staplesforsmear_dag_set,
        tensor_derivative_dag,
        ρs,
    )
end

function make_loops_Stoutsmearing(loops_smearing, ρs)
    num = length(loops_smearing)
    @assert num == length(ρs) "number of ρ elements in stout smearing scheme should be $num. Now $(length(ρs)). "
    staplesforsmear_set = Array{Wilson_loop_set,1}[]
    staplesforsmear_dag_set = Array{Wilson_loop_set,1}[]
    println("staple for stout smearing")

    tensor_derivative = Array{Tensor_derivative_set,1}(undef, num)
    tensor_derivative_dag = Array{Tensor_derivative_set,1}(undef, num)



    for i = 1:num
        loop_smearing = loops_smearing[i]

        staplesforsmear = Array{Wilson_loop_set,1}(undef, 4)
        staplesforsmear_dag = Array{Wilson_loop_set,1}(undef, 4)



        staple = make_staples(loop_smearing)

        for μ = 1:4
            staplesforsmear_dag[μ] = staple[μ]##make_plaq_staple(μ)
            staplesforsmear[μ] = staplesforsmear_dag[μ]'
            #println("$μ -direction")
            #display(staplesforsmear[μ])
            #staplesforsmear[μ] = make_plaq_staple(μ)
            #staplesforsmear_dag[μ] = staplesforsmear[μ]'
            #println("dagger: $μ -direction")
            #display(staplesforsmear_dag[μ])
        end
        tensor_derivative[i] = Tensor_derivative_set(staplesforsmear)
        tensor_derivative_dag[i] = Tensor_derivative_set(staplesforsmear_dag)

        push!(staplesforsmear_set, staplesforsmear)
        push!(staplesforsmear_dag_set, staplesforsmear_dag)

    end

    return staplesforsmear_set,
    tensor_derivative,
    staplesforsmear_dag_set,
    tensor_derivative_dag

end
